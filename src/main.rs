use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use tower_http::cors::CorsLayer;
use futures::stream;
use ndarray::{Array3, ArrayView3, ArrayView4};
use ort::ep::CPU;
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

mod decode;
mod idiom;
mod matcher;
mod normalize;
mod phrasal;

use idiom::{IdiomClassifier, IdiomMatch};
use matcher::Lexicon;

/// parser SubwordField fix_len: max subwords kept per word.
const FIX_LEN: usize = 20;
/// microsoft/deberta-v3-base sentencepiece ids (RUST_INTEGRATION.md §5.1).
const CLS_ID: i64 = 1; // ROOT row / [CLS]
const UNK_ID: i64 = 3; // word that produced no pieces

const MODEL_PATH: &str = "model/model.fp16.onnx";
const VOCAB_PATH: &str = "model/vocabs.json";
const CLASSIFIER_PATH: &str = "model/idiom_classifier.json";

const LEXICON_PATH: &str = "dic/lexicon.json";
const PHRASAL_PATH: &str = "dic/phrasal-verbs.json";

// --- types ---

#[derive(Deserialize)]
struct ParseBatchRequest {
    sentences: Vec<String>,
}

#[derive(Serialize)]
struct ParsedToken {
    /// 1-based parser word index (== grid / output row, ROOT is 0).
    id: usize,
    word: String,
    /// Conservative lemma (same `normalize::lemma` used for matching).
    lemma: String,
    /// Head word id; 0 == ROOT.
    head: usize,
    rel: String,
    upos: String,
}

#[derive(Serialize)]
struct PhrasalVerb {
    verb: String,
    particle: String,
    verb_id: usize,
    particle_id: usize,
    /// Extending preposition for 3-component verbs ("come up **with**");
    /// absent for bare verb+particle. Optional so the JSON stays
    /// backward-compatible with consumers that only read verb/particle.
    #[serde(skip_serializing_if = "Option::is_none")]
    prep: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prep_id: Option<usize>,
}

#[derive(Serialize)]
struct SentenceResult {
    tokens: Vec<ParsedToken>,
    phrasal_verbs: Vec<PhrasalVerb>,
    idioms: Vec<IdiomMatch>,
}

#[derive(Serialize)]
struct ProgressEvent {
    done: usize,
    total: usize,
    percent: u8,
}

#[derive(Serialize)]
struct DoneEvent {
    results: Vec<SentenceResult>,
}

#[derive(Serialize)]
struct ResultEvent<'a> {
    index: usize,
    result: &'a SentenceResult,
}

struct SentenceJob {
    sentence: String,
    reply: oneshot::Sender<anyhow::Result<SentenceResult>>,
}

// vocabs.json stores { label: index } dicts; invert to index-keyed Vec<String>.
fn vocab_from_map(map: std::collections::HashMap<String, usize>) -> Vec<String> {
    let size = map.values().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut v = vec![String::new(); size];
    for (label, idx) in map {
        v[idx] = label;
    }
    v
}

#[derive(Deserialize)]
struct VocabRaw {
    rel_vocab: std::collections::HashMap<String, usize>,
    pos_vocab: std::collections::HashMap<String, usize>,
}

struct Vocab {
    rels: Vec<String>,
    upos: Vec<String>,
}

impl From<VocabRaw> for Vocab {
    fn from(raw: VocabRaw) -> Self {
        Vocab {
            rels: vocab_from_map(raw.rel_vocab),
            upos: vocab_from_map(raw.pos_vocab),
        }
    }
}

// --- state ---

const DEFAULT_IDLE_UNLOAD_SECS: u64 = 300;

/// Session loaded on first request, dropped after `idle` with no use.
/// Service profile is a few requests/day, so the ~1–2 s cold-start on the
/// first request after eviction is an acceptable trade for ~170 MB idle RSS.
struct LazySession {
    inner: Mutex<Option<Session>>,
    last_used: Mutex<Instant>,
}

impl LazySession {
    fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
        }
    }

    /// Ensures the session is loaded, runs `f`, and bumps the idle timer.
    /// The inner lock is held for the whole call, so a batch in flight
    /// cannot be evicted mid-run and concurrent requests serialize here.
    fn with_session<F, R>(&self, f: F) -> anyhow::Result<R>
    where
        F: FnOnce(&mut Session) -> anyhow::Result<R>,
    {
        let mut guard = self.inner.lock().unwrap();
        if guard.is_none() {
            info!("loading model from {MODEL_PATH} (lazy, cold start)");
            *guard = Some(build_session()?);
        }
        *self.last_used.lock().unwrap() = Instant::now();
        let out = f(guard.as_mut().unwrap());
        *self.last_used.lock().unwrap() = Instant::now();
        out
    }

    fn maybe_evict(&self, idle: Duration) {
        if self.last_used.lock().unwrap().elapsed() <= idle {
            return;
        }
        let mut guard = self.inner.lock().unwrap();
        if guard.is_some() {
            *guard = None; // Drop frees ORT native memory deterministically.
            info!("model evicted after {idle:?} idle; RSS drops to baseline");
        }
    }
}

struct AppState {
    session: LazySession,
    tokenizer: Tokenizer,
    rels: Vec<String>,
    upos: Vec<String>,
    lexicon: Vec<matcher::Entry>,
    classifier: IdiomClassifier,
    phrasal: phrasal::PhrasalLexicon,
    job_tx: mpsc::UnboundedSender<SentenceJob>,
}

fn idle_unload_secs() -> u64 {
    std::env::var("PARSER_IDLE_UNLOAD_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_IDLE_UNLOAD_SECS)
}

/// Background thread (not a tokio task — it briefly holds a std Mutex) that
/// evicts the idle session. Stops itself once `AppState` is dropped.
fn spawn_evictor(state: Weak<AppState>) {
    let idle = Duration::from_secs(idle_unload_secs());
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(60));
        let Some(state) = state.upgrade() else { break };
        state.session.maybe_evict(idle);
    });
}

/// Dedicated std thread that owns the inference loop.
/// Processes one sentence at a time from the shared job channel, releases the
/// session lock between sentences so the evictor can still run.
fn spawn_inference_worker(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<SentenceJob>,
) {
    std::thread::spawn(move || {
        while let Some(job) = rx.blocking_recv() {
            let result = state.session.with_session(|session| {
                run_inference(session, &state, &job.sentence)
            });
            let _ = job.reply.send(result);
        }
    });
}

// --- handlers ---

async fn health() -> &'static str {
    "ok"
}

async fn parse_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseBatchRequest>,
) -> impl IntoResponse {
    let total = req.sentences.len();

    let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Event>();

    tokio::spawn(async move {
        if total == 0 {
            let done_data =
                serde_json::to_string(&DoneEvent { results: vec![] }).unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("done").data(done_data));
            return;
        }

        // Submit every sentence as an independent job and remember the reply channels
        // in order. Jobs from concurrent requests are interleaved in the shared worker
        // queue, so no single request holds the model lock for its entire batch.
        let mut receivers = Vec::with_capacity(total);
        for sentence in &req.sentences {
            let (reply_tx, reply_rx) = oneshot::channel();
            if state
                .job_tx
                .send(SentenceJob {
                    sentence: sentence.trim().to_string(),
                    reply: reply_tx,
                })
                .is_err()
            {
                let _ = sse_tx
                    .send(Event::default().event("error").data("inference worker stopped"));
                return;
            }
            receivers.push(reply_rx);
        }

        let mut results: Vec<SentenceResult> = Vec::with_capacity(total);
        for (i, reply_rx) in receivers.into_iter().enumerate() {
            let sentence_result = match reply_rx.await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    let _ = sse_tx
                        .send(Event::default().event("error").data(format!("sentence {i}: {e}")));
                    return;
                }
                Err(_) => {
                    let _ = sse_tx
                        .send(Event::default().event("error").data("inference worker dropped"));
                    return;
                }
            };

            // Per-sentence result event (clients that don't need it can ignore it).
            let result_data =
                serde_json::to_string(&ResultEvent { index: i, result: &sentence_result })
                    .unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("result").data(result_data));

            let done = i + 1;
            let percent = (done * 100 / total) as u8;
            let progress =
                serde_json::to_string(&ProgressEvent { done, total, percent }).unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("progress").data(progress));

            results.push(sentence_result);
        }

        let done_data = serde_json::to_string(&DoneEvent { results }).unwrap_or_default();
        let _ = sse_tx.send(Event::default().event("done").data(done_data));
    });

    let sse_stream = Box::pin(stream::unfold(sse_rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<Event, std::convert::Infallible>(event), rx))
    }));

    Sse::new(sse_stream).keep_alive(KeepAlive::default())
}

fn argmax(xs: impl Iterator<Item = f32>) -> usize {
    xs.enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn run_inference(
    session: &mut Session,
    state: &AppState,
    sentence: &str,
) -> anyhow::Result<SentenceResult> {
    // §5.1: word boundaries come from the canonical tokenizer, NOT whitespace.
    let words = normalize::tokenize(sentence);
    let n = words.len();
    if n == 0 {
        return Ok(SentenceResult {
            tokens: vec![],
            phrasal_verbs: vec![],
            idioms: vec![],
        });
    }

    // Build the parser subword grid: row 0 = ROOT ([CLS]), row i+1 = word i's
    // sentencepiece ids; F = min(20, longest row), right-padded with 0.
    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(n + 1);
    rows.push(vec![CLS_ID]);
    for word in &words {
        let enc = state
            .tokenizer
            .encode(word.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize '{}': {}", word, e))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        if ids.is_empty() {
            ids.push(UNK_ID);
        }
        rows.push(ids);
    }
    // F dim is static in the ONNX graph (exported with fix_len=20, no dynamic axis).
    let fix_len = FIX_LEN;
    let w_dim = n + 1;

    let mut subwords = Array3::<i64>::zeros((1, w_dim, fix_len));
    for (r, row) in rows.iter().enumerate() {
        for (c, &id) in row.iter().take(fix_len).enumerate() {
            subwords[[0, r, c]] = id;
        }
    }

    let subwords_val =
        ort::value::Value::from_array(subwords).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let outputs = session
        .run(vec![("subwords", subwords_val)])
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let (arc_shape, arc_data) = outputs["s_arc"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let (rel_shape, rel_data) = outputs["s_rel"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let (pos_shape, pos_data) = outputs["s_pos"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let (repr_shape, repr_data) = outputs["word_repr"]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let dims = |s: &[i64]| s.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let ad = dims(arc_shape);
    let arc = ArrayView3::from_shape((ad[0], ad[1], ad[2]), arc_data)?;
    let rd = dims(rel_shape);
    let rel = ArrayView4::from_shape((rd[0], rd[1], rd[2], rd[3]), rel_data)?;
    let pd = dims(pos_shape);
    let pos = ArrayView3::from_shape((pd[0], pd[1], pd[2]), pos_data)?;
    let rpd = dims(repr_shape);
    let repr = ArrayView3::from_shape((rpd[0], rpd[1], rpd[2]), repr_data)?;

    // MST (Chu-Liu/Edmonds): edge u->v (u is head of v) has weight s_arc[v][u].
    let mut score = vec![vec![f32::NEG_INFINITY; w_dim]; w_dim];
    for v in 0..w_dim {
        for (u, srow) in score.iter_mut().enumerate() {
            if u != v {
                srow[v] = arc[[0, v, u]];
            }
        }
    }
    let heads = decode::max_arborescence(w_dim, 0, &score);

    let n_rels = rd[3];
    let n_upos = pd[2];

    // UPOS for every grid row (ROOT + words). Precomputed because the phrasal
    // second pass needs the *head* token's POS, not just the current token's.
    let upos_ids: Vec<usize> = (0..w_dim)
        .map(|w| argmax((0..n_upos).map(|k| pos[[0, w, k]])))
        .collect();
    let upos_str =
        |w: usize| state.upos.get(upos_ids[w]).cloned().unwrap_or_else(|| "X".into());

    // Prefer the longest reading: given a confirmed verb (id `h`) + particle
    // (id `after`) and the prepositions that extend that pair ("come up" ->
    // {with}), find the extending preposition in the tree to upgrade
    // "come up" → "come up with". The preposition must be an ADP, occur after
    // the particle, and hang off the verb — directly, or (the usual UD shape)
    // as the `case` of one of the verb's arguments: with → noise → put.
    let find_prep = |h: usize, after: usize, preps: &[String]| -> (Option<String>, Option<usize>) {
        if preps.is_empty() {
            return (None, None);
        }
        for j in (after + 1)..=n {
            if upos_str(j).as_str() != "ADP" {
                continue;
            }
            let jl = normalize::lemma(&words[j - 1]);
            if !preps.iter().any(|pp| pp.as_str() == jl) {
                continue;
            }
            let hj = heads[j];
            if hj == h || (hj >= 1 && heads[hj] == h) {
                return (Some(words[j - 1].clone()), Some(j));
            }
        }
        (None, None)
    };

    let mut tokens = Vec::with_capacity(n);
    let mut phrasal_verbs = Vec::new();
    // Particle word-ids already claimed, so the second pass never double-emits.
    let mut phrasal_particle_ids: HashSet<usize> = HashSet::new();
    for i in 1..=n {
        let head = heads[i];
        let rel_id = argmax((0..n_rels).map(|k| rel[[0, i, head, k]]));
        let rel = state.rels.get(rel_id).cloned().unwrap_or_else(|| "dep".into());
        let upos = upos_str(i);

        // compound:prt edge ⇒ phrasal verb (head verb + particle), pure
        // syntax. UNTOUCHED in recall: the confident, label-driven path
        // (§6.3) still emits even when the pair is not in the lexicon
        // (`unwrap_or_default` → no prep, bare core). When it *is* known,
        // the preposition upgrades it to the 3-component reading.
        if rel == "compound:prt" && head >= 1 {
            let preps = state
                .phrasal
                .resolve(&words[head - 1], &words[i - 1])
                .unwrap_or_default();
            let (prep, prep_id) = find_prep(head, i, &preps);
            phrasal_verbs.push(PhrasalVerb {
                verb: words[head - 1].clone(),
                particle: words[i - 1].clone(),
                verb_id: head,
                particle_id: i,
                prep,
                prep_id,
            });
            phrasal_particle_ids.insert(i);
        }

        tokens.push(ParsedToken {
            id: i,
            word: words[i - 1].clone(),
            lemma: normalize::lemma(&words[i - 1]),
            head,
            rel,
            upos,
        });
    }

    // Approach-3 second pass — label-agnostic. The particle→verb ARC is the
    // stable signal (the deprel flips between compound:prt/advmod with the
    // verb's surface form, and UD-EWT itself annotates "come up" as advmod),
    // so confirm any verb-headed particle/adposition edge against the closed
    // phrasal-verb inventory instead of trusting the label. Lemma match makes
    // it tense-invariant (came→come); the closed list + the arc keep
    // adverbial false positives ("look up at the sky") out.
    for i in 1..=n {
        if phrasal_particle_ids.contains(&i) {
            continue; // already emitted by the compound:prt path
        }
        let head = heads[i];
        if head < 1 {
            continue;
        }
        if !matches!(upos_str(i).as_str(), "ADP" | "ADV" | "PART") {
            continue;
        }
        if !matches!(upos_str(head).as_str(), "VERB" | "AUX") {
            continue;
        }
        // Infinitive "to" (PART) precedes its verb head — not a particle.
        // Real phrasal verb particles always follow the verb in word order.
        if i < head {
            continue;
        }
        if let Some(preps) = state.phrasal.resolve(&words[head - 1], &words[i - 1]) {
            let (prep, prep_id) = find_prep(head, i, &preps);
            phrasal_verbs.push(PhrasalVerb {
                verb: words[head - 1].clone(),
                particle: words[i - 1].clone(),
                verb_id: head,
                particle_id: i,
                prep,
                prep_id,
            });
            phrasal_particle_ids.insert(i);
        }
    }

    // Model-predicted UPOS==VERB per word (grid row = word index + 1; row 0 is
    // ROOT). Gates the POS-conditional `_IRREGULAR_VERB` remap in the matcher;
    // stage3 localizes training spans with the same s_pos head → no skew.
    let is_verb: Vec<bool> = (0..n).map(|k| upos_str(k + 1) == "VERB").collect();

    let idioms = idiom::detect(
        &words,
        &is_verb,
        &repr,
        &heads,
        &state.lexicon,
        &state.classifier,
    )?;

    info!(
        words = n,
        phrasal = phrasal_verbs.len(),
        idioms = idioms.len(),
        "parsed"
    );
    Ok(SentenceResult {
        tokens,
        phrasal_verbs,
        idioms,
    })
}

// --- session ---

fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn build_session() -> anyhow::Result<Session> {
    // Input length varies per sentence; with mem-pattern ON, ORT caches
    // allocation patterns sized to the longest sentence seen and keeps them —
    // that is the main RSS-peak driver here, so default OFF.
    let mem_pattern = env_flag("PARSER_MEM_PATTERN", false);
    // Arena trades RSS for latency: keep ON for dev throughput, set
    // PARSER_CPU_ARENA=0 on a weak server to shed ~100–300 MB RSS.
    let cpu_arena = env_flag("PARSER_CPU_ARENA", true);
    let intra_threads = std::env::var("PARSER_INTRA_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0);

    info!(
        mem_pattern,
        cpu_arena,
        intra_threads = intra_threads.unwrap_or(0),
        "ORT session config (0 intra_threads = ORT default / all cores)"
    );

    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_memory_pattern(mem_pattern)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_execution_providers([CPU::default().with_arena_allocator(cpu_arena).build()])
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    if let Some(n) = intra_threads {
        builder = builder
            .with_intra_threads(n)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }

    // `.onnx` references `.onnx.data` by relative name → same dir, ORT loads
    // the external-weights sidecar automatically.
    builder
        .commit_from_file(MODEL_PATH)
        .map_err(|e| anyhow::anyhow!("{:?}", e))
}

// --- main ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    info!("loading vocab from {VOCAB_PATH}");
    let vocab: Vocab = serde_json::from_str::<VocabRaw>(&std::fs::read_to_string(VOCAB_PATH)?)?.into();

    info!("loading lexicon from {LEXICON_PATH}");
    let lexicon: Lexicon = serde_json::from_str(&std::fs::read_to_string(LEXICON_PATH)?)?;

    info!("loading idiom classifier from {CLASSIFIER_PATH}");
    let classifier: IdiomClassifier =
        serde_json::from_str(&std::fs::read_to_string(CLASSIFIER_PATH)?)?;

    info!("loading phrasal-verb lexicon from {PHRASAL_PATH}");
    let phrasal = phrasal::PhrasalLexicon::load(PHRASAL_PATH)?;

    info!("loading tokenizer");
    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

    let (job_tx, job_rx) = mpsc::unbounded_channel::<SentenceJob>();

    let state = Arc::new(AppState {
        session: LazySession::new(),
        tokenizer,
        rels: vocab.rels,
        upos: vocab.upos,
        lexicon: lexicon.lexicon,
        classifier,
        phrasal,
        job_tx,
    });

    spawn_evictor(Arc::downgrade(&state));
    spawn_inference_worker(Arc::clone(&state), job_rx);
    info!(
        idle_unload_secs = idle_unload_secs(),
        lexicon = state.lexicon.len(),
        phrasal = state.phrasal.len(),
        "model is lazy-loaded on first request, evicted after idle"
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/parse", post(parse_batch))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr = std::env::var("PARSER_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

// --- end-to-end regression tests ---
//
// These tests require all model artifacts to be present:
//   model/model.fp16.onnx   model/vocabs.json   model/idiom_classifier.json
//   dic/lexicon.json         dic/phrasal-verbs.json
//
// Run with:
//   cargo test -- --include-ignored
//
#[cfg(test)]
mod e2e {
    use super::*;

    fn build_state() -> anyhow::Result<AppState> {
        let vocab: Vocab =
            serde_json::from_str::<VocabRaw>(&std::fs::read_to_string(VOCAB_PATH)?)?.into();
        let lexicon: Lexicon = serde_json::from_str(&std::fs::read_to_string(LEXICON_PATH)?)?;
        let classifier: IdiomClassifier =
            serde_json::from_str(&std::fs::read_to_string(CLASSIFIER_PATH)?)?;
        let phrasal = phrasal::PhrasalLexicon::load(PHRASAL_PATH)?;
        let tokenizer = Tokenizer::from_file("model/tokenizer.json")
            .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        let (job_tx, _job_rx) = mpsc::unbounded_channel();
        Ok(AppState {
            session: LazySession::new(),
            tokenizer,
            rels: vocab.rels,
            upos: vocab.upos,
            lexicon: lexicon.lexicon,
            classifier,
            phrasal,
            job_tx,
        })
    }

    // Idiom detection golden cases.
    // Each tuple: (sentence, exact lexicon surface as returned by the model, expect_idiomatic).
    // surface is the winning entry after overlap resolution — run with --nocapture to see
    // what the model actually returns if a case starts failing.
    const IDIOM_CASES: &[(&str, &str, bool)] = &[
        ("You have an audition today? Break a leg!", "break a leg", true),
        ("He spilled the beans about the surprise party.", "spill [pron] beans", true),
        ("After years of hard work, she finally kicked the bucket.", "kick [pron] bucket", true),
        // Literal use — the model should NOT flag this as idiomatic.
        ("She broke her leg falling off the horse.", "break a leg", false),
    ];

    #[test]
    #[ignore = "requires model artifacts; run with: cargo test -- --include-ignored"]
    fn idiom_detection_golden_cases() {
        let state = build_state().expect("failed to load model artifacts");
        state
            .session
            .with_session(|session| {
                for &(sent, surface, expect_idiomatic) in IDIOM_CASES {
                    let result = run_inference(session, &state, sent)
                        .unwrap_or_else(|e| panic!("inference failed for {:?}: {}", sent, e));

                    let hit = result.idioms.iter().find(|m| m.surface == surface);

                    match (hit, expect_idiomatic) {
                        (None, true) => panic!(
                            "expected idiom {:?} in {:?} but it was not detected",
                            surface, sent
                        ),
                        (Some(m), true) => assert!(
                            m.idiomatic,
                            "expected idiomatic=true for {:?} in {:?} (prob={:.3})",
                            m.surface, sent, m.prob
                        ),
                        (Some(m), false) => assert!(
                            !m.idiomatic,
                            "expected idiomatic=false for {:?} in {:?} (prob={:.3})",
                            m.surface, sent, m.prob
                        ),
                        (None, false) => {} // not detected and not expected — ok
                    }
                }
                Ok(())
            })
            .expect("session error");
    }

    // Phrasal verb detection golden cases.
    // (sentence, verb lemma, particle, prep)
    const PHRASAL_CASES: &[(&str, &str, &str, Option<&str>)] = &[
        ("She came up with a great idea.", "come", "up", Some("with")),
        ("He gave up smoking last year.", "give", "up", None),
        ("They put up with the noise.", "put", "up", Some("with")),
        ("Please turn off the lights.", "turn", "off", None),
    ];

    // False positive cases: no phrasal verb should be detected.
    // TODO: "hold to" fires because the infinitive marker "to" (PART → hold VERB)
    // matches the second-pass arc+inventory check — fix the detector.
    const PHRASAL_FALSE_POSITIVES: &[(&str, &str, &str)] = &[
        ("Kyiv should focus on defense and carefully determine which positions are truly necessary to hold.", "hold", "to"),
        ("Ukraine's broader goal is to turn Pokrovsk.", "turn", "to"),
        ("In order to push them out of the Hryshyne area.", "push", "in"),
    ];

    #[test]
    #[ignore = "requires model artifacts; run with: cargo test -- --include-ignored"]
    fn phrasal_verb_detection_golden_cases() {
        let state = build_state().expect("failed to load model artifacts");
        state
            .session
            .with_session(|session| {
                for &(sent, verb_kw, particle, prep) in PHRASAL_CASES {
                    let result = run_inference(session, &state, sent)
                        .unwrap_or_else(|e| panic!("inference failed for {:?}: {}", sent, e));

                    let found = result.phrasal_verbs.iter().any(|pv| {
                        normalize::lemma(&pv.verb) == verb_kw
                            && pv.particle == particle
                            && pv.prep.as_deref() == prep
                    });
                    assert!(
                        found,
                        "expected phrasal verb ({}, {}, {:?}) in {:?}\n  got: {:?}",
                        verb_kw,
                        particle,
                        prep,
                        sent,
                        result
                            .phrasal_verbs
                            .iter()
                            .map(|pv| format!(
                                "{}+{}{}",
                                pv.verb,
                                pv.particle,
                                pv.prep.as_deref().map(|p| format!("+{p}")).unwrap_or_default()
                            ))
                            .collect::<Vec<_>>()
                    );
                }

                for &(sent, verb_kw, particle) in PHRASAL_FALSE_POSITIVES {
                    let result = run_inference(session, &state, sent)
                        .unwrap_or_else(|e| panic!("inference failed for {:?}: {}", sent, e));

                    let fp = result.phrasal_verbs.iter().find(|pv| {
                        normalize::lemma(&pv.verb) == verb_kw && pv.particle == particle
                    });
                    assert!(
                        fp.is_none(),
                        "false positive phrasal verb ({}, {}) should not be detected in {:?}",
                        verb_kw, particle, sent
                    );
                }
                Ok(())
            })
            .expect("session error");
    }
}
