use ndarray::{Array3, Array4, ArrayView3, ArrayView4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

pub mod decode;
pub mod mwe;
pub mod matcher;
pub mod normalize;

use mwe::MweMatch;

/// parser SubwordField fix_len: max subwords kept per word.
/// This is an upper bound; actual tensor size shrinks to sentence max.
const MAX_FIX_LEN: usize = 20;
// Constants removed; token IDs are now dynamically loaded from the tokenizer into AppState.

pub const MODEL_PATH: &str = "model/model.onnx";
pub const VOCAB_PATH: &str = "model/vocabs.json";

pub const LEXICON_PATH: &str = "dic/lexicon.jsonl";
pub const CUSTOM_LEXICON_PATH: &str = "dic/custom.jsonl";
pub const CORRECTIONS_PATH: &str = "dic/corrections.jsonl";

// --- types ---

#[derive(Serialize)]
pub struct ParsedToken {
    /// 1-based parser word index (== grid / output row, ROOT is 0).
    pub id: usize,
    pub word: String,
    /// Conservative lemma (same `normalize::lemma` used for matching).
    pub lemma: String,
    /// Head word id; 0 == ROOT.
    pub head: usize,
    pub rel: String,
    pub upos: String,
    /// CoNLL-U FEATS field: `Cat=Val|Cat=Val…` (alphabetical), or `_` if none /
    /// the model has no FEATS head.
    pub feats: String,
}

#[derive(Serialize)]
pub struct SentenceResult {
    pub tokens: Vec<ParsedToken>,
    pub mwes: Vec<MweMatch>,
}

pub struct SentenceJob {
    pub sentence: String,
    pub reply: oneshot::Sender<anyhow::Result<SentenceResult>>,
}

// vocabs.json stores { label: index } dicts; invert to index-keyed Vec<String>.
pub fn vocab_from_map(map: std::collections::HashMap<String, usize>) -> Vec<String> {
    let size = map.values().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut v = vec![String::new(); size];
    for (label, idx) in map {
        v[idx] = label;
    }
    v
}

/// FEATS vocab `{ category: { value: idx } }` → ordered `[(category, [value_by_idx])]`.
/// Categories are sorted alphabetically to match the model's s_feats category axis
/// (training builds it via `for cat in sorted(cats)`); value index 0 is `_` (absent).
pub fn feats_from_map(
    map: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
) -> Vec<(String, Vec<String>)> {
    let mut cats: Vec<(String, Vec<String>)> = map
        .into_iter()
        .map(|(cat, values)| (cat, vocab_from_map(values)))
        .collect();
    cats.sort_by(|a, b| a.0.cmp(&b.0));
    cats
}

#[derive(Deserialize)]
pub struct VocabRaw {
    rel_vocab: std::collections::HashMap<String, usize>,
    pos_vocab: std::collections::HashMap<String, usize>,
    // Present only for upos_feats models; empty map for UPOS-only checkpoints.
    #[serde(default)]
    feats_vocab: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
}

pub struct Vocab {
    pub rels: Vec<String>,
    pub upos: Vec<String>,
    /// Ordered `[(category, [value_by_idx])]`; empty when the model has no FEATS head.
    pub feats: Vec<(String, Vec<String>)>,
}

impl From<VocabRaw> for Vocab {
    fn from(raw: VocabRaw) -> Self {
        Vocab {
            rels: vocab_from_map(raw.rel_vocab),
            upos: vocab_from_map(raw.pos_vocab),
            feats: feats_from_map(raw.feats_vocab),
        }
    }
}

// --- jemalloc purge ---

/// Ask jemalloc to return all freed pages to the OS immediately.
/// `MALLCTL_ARENAS_ALL` = 4096 in jemalloc 5.3 (tikv-jemalloc-sys 0.7).
fn jemalloc_purge() {
    #[cfg(not(target_env = "msvc"))]
    {
        let purge = b"arena.4096.purge\0";
        // SAFETY: purge is a valid NUL-terminated mallctl name; no in/out params.
        let rc = unsafe {
            tikv_jemalloc_sys::mallctl(
                purge.as_ptr().cast(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            tracing::warn!("jemalloc arena purge failed (rc={rc})");
        }
    }
}

/// Current RSS in megabytes (Linux: /proc/self/statm, macOS: mach task_info).
pub fn rss_mb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let rss_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        Some(rss_pages as f64 * 4096.0 / 1_048_576.0)
    }
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        // SAFETY: standard mach API; info buffer is properly sized.
        #[allow(deprecated)] // libc suggests mach2 crate; not worth the dep for diagnostics
        unsafe {
            let mut info = MaybeUninit::<libc::mach_task_basic_info_data_t>::uninit();
            let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
                / std::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
            let kr = libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                info.as_mut_ptr().cast(),
                &mut count,
            );
            if kr != libc::KERN_SUCCESS {
                return None;
            }
            Some(info.assume_init().resident_size as f64 / 1_048_576.0)
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

// --- state ---

const DEFAULT_IDLE_UNLOAD_SECS: u64 = 300;

/// Session loaded on first request, dropped after `idle` with no use.
/// Service profile is a few requests/day, so the ~1–2 s cold-start on the
/// first request after eviction is an acceptable trade for ~170 MB idle RSS.
pub struct LazySession {
    inner: Mutex<Option<Session>>,
    last_used: Mutex<Instant>,
}

impl LazySession {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            last_used: Mutex::new(Instant::now()),
        }
    }

    /// Ensures the session is loaded, runs `f`, and bumps the idle timer.
    /// The inner lock is held for the whole call, so a batch in flight
    /// cannot be evicted mid-run and concurrent requests serialize here.
    pub fn with_session<F, R>(&self, f: F) -> anyhow::Result<R>
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

    pub fn maybe_evict(&self, idle: Duration) {
        if self.last_used.lock().unwrap().elapsed() <= idle {
            return;
        }
        let mut guard = self.inner.lock().unwrap();
        if guard.is_some() {
            let before = rss_mb();
            *guard = None; // Drop frees ORT native memory deterministically.
            let after_drop = rss_mb();
            jemalloc_purge();
            let after_purge = rss_mb();
            info!(
                rss_before = before.map(|v| format!("{v:.1} MB")),
                rss_after_drop = after_drop.map(|v| format!("{v:.1} MB")),
                rss_after_purge = after_purge.map(|v| format!("{v:.1} MB")),
                "model evicted after {idle:?} idle"
            );
        }
    }
}

pub struct AppState {
    pub session: LazySession,
    pub tokenizer: Tokenizer,
    pub rels: Vec<String>,
    pub upos: Vec<String>,
    /// FEATS categories, ordered to match the s_feats axis; empty if no FEATS head.
    pub feats: Vec<(String, Vec<String>)>,
    pub lexicon: mwe::MweLexicon,
    pub job_tx: mpsc::UnboundedSender<SentenceJob>,
    pub cls_id: i64,
    pub unk_id: i64,
}

pub fn idle_unload_secs() -> u64 {
    std::env::var("PARSER_IDLE_UNLOAD_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_IDLE_UNLOAD_SECS)
}

/// Background thread (not a tokio task — it briefly holds a std Mutex) that
/// evicts the idle session. Stops itself once `AppState` is dropped.
pub fn spawn_evictor(state: Weak<AppState>) {
    let idle = Duration::from_secs(idle_unload_secs());
    // Poll at most every 30 s but no slower than half the idle timeout,
    // so short timeouts (e.g. 10 s) are actually responsive.
    let poll = Duration::from_secs((idle.as_secs() / 2).max(1).min(30));
    std::thread::spawn(move || loop {
        std::thread::sleep(poll);
        let Some(state) = state.upgrade() else { break };
        state.session.maybe_evict(idle);
    });
}

/// Dedicated std thread that owns the inference loop.
/// Processes one sentence at a time from the shared job channel, releases the
/// session lock between sentences so the evictor can still run.
pub fn spawn_inference_worker(
    state: Arc<AppState>,
    mut rx: mpsc::UnboundedReceiver<SentenceJob>,
) {
    let batch_size: usize = std::env::var("PARSER_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    std::thread::spawn(move || {
        loop {
            let mut batch = Vec::with_capacity(batch_size);
            if let Some(first) = rx.blocking_recv() {
                batch.push(first);
                while batch.len() < batch_size {
                    if let Ok(next) = rx.try_recv() {
                        batch.push(next);
                    } else {
                        break;
                    }
                }

                let sentences: Vec<String> = batch.iter().map(|job| job.sentence.clone()).collect();
                let batch_results = match state.session.with_session(|session| {
                    Ok(run_inference_batch(session, &state, &sentences))
                }) {
                    Ok(res) => res,
                    Err(e) => (0..batch.len()).map(|_| Err(anyhow::anyhow!("session: {}", e))).collect(),
                };

                for (job, res) in batch.into_iter().zip(batch_results) {
                    let _ = job.reply.send(res);
                }
            } else {
                break;
            }
        }
    });
}

fn argmax(xs: impl Iterator<Item = f32>) -> usize {
    xs.enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn run_inference(
    session: &mut Session,
    state: &AppState,
    sentence: &str,
) -> anyhow::Result<SentenceResult> {
    let start_time = Instant::now();
    // §5.1: word boundaries come from the canonical tokenizer, NOT whitespace.
    let words = normalize::tokenize(sentence);
    let n = words.len();
    if n == 0 {
        return Ok(SentenceResult {
            tokens: vec![],
            mwes: vec![],
        });
    }

    // Build the parser subword grid: row 0 = ROOT ([CLS]), row i+1 = word i's
    // sentencepiece ids; F = min(20, longest row), right-padded with 0.
    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(n + 1);
    rows.push(vec![state.cls_id]);
    for word in &words {
        let enc = state
            .tokenizer
            .encode(word.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize '{}': {}", word, e))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        if ids.is_empty() {
            ids.push(state.unk_id);
        }
        rows.push(ids);
    }
    // Compute dynamic fix_len: the longest subword list in this sentence, capped at MAX_FIX_LEN
    let fix_len = rows.iter().map(|row| row.len()).max().unwrap_or(1).min(MAX_FIX_LEN);
    let w_dim = n + 1;

    let mut subwords = Array3::<i64>::zeros((1, w_dim, fix_len));
    for (r, row) in rows.iter().enumerate() {
        for (c, &id) in row.iter().take(fix_len).enumerate() {
            subwords[[0, r, c]] = id;
        }
    }

    let subwords_val =
        Tensor::from_array(subwords).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let input_name = String::from(session.inputs()[0].name());
    let outputs = session
        .run(ort::inputs![input_name.as_str() => subwords_val])
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
    let dims = |s: &[i64]| s.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let ad = dims(arc_shape);
    let arc = ArrayView3::from_shape((ad[0], ad[1], ad[2]), arc_data)?;
    let rd = dims(rel_shape);
    let rel = ArrayView4::from_shape((rd[0], rd[1], rd[2], rd[3]), rel_data)?;
    let pd = dims(pos_shape);
    let pos = ArrayView3::from_shape((pd[0], pd[1], pd[2]), pos_data)?;

    // s_feats [1, W, C, Vmax] — present only for upos_feats models. Owned so the
    // per-word closure below doesn't borrow `outputs`.
    let feats_arr: Option<Array4<f32>> = if state.feats.is_empty() {
        None
    } else {
        let (fs, fdat) = outputs["s_feats"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let fd = dims(fs);
        Some(ArrayView4::from_shape((fd[0], fd[1], fd[2], fd[3]), fdat)?.to_owned())
    };

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

    // UPOS for every grid row (ROOT + words). Precomputed because the MWE
    // matcher needs the *head* token's POS, not just the current token's.
    let upos_ids: Vec<usize> = (0..w_dim)
        .map(|w| argmax((0..n_upos).map(|k| pos[[0, w, k]])))
        .collect();
    let upos_str =
        |w: usize| state.upos.get(upos_ids[w]).cloned().unwrap_or_else(|| "X".into());

    // CoNLL-U FEATS string for grid row `w`: per category, argmax over its own
    // values; value index 0 is `_` (absent) and is skipped. Categories are already
    // alphabetical (feats_from_map), matching CoNLL-U ordering. `_` when no feature.
    let feats_str = |w: usize| -> String {
        let Some(fa) = feats_arr.as_ref() else { return "_".into() };
        let mut parts: Vec<String> = Vec::new();
        for (c, (cat, values)) in state.feats.iter().enumerate() {
            let best = argmax((0..values.len()).map(|k| fa[[0, w, c, k]]));
            if best != 0 {
                if let Some(val) = values.get(best) {
                    parts.push(format!("{cat}={val}"));
                }
            }
        }
        if parts.is_empty() { "_".into() } else { parts.join("|") }
    };

    let mut tokens = Vec::with_capacity(n);
    for i in 1..=n {
        let head = heads[i];
        let rel_id = argmax((0..n_rels).map(|k| rel[[0, i, head, k]]));
        let rel = state.rels.get(rel_id).cloned().unwrap_or_else(|| "dep".into());
        let upos = upos_str(i);

        tokens.push(ParsedToken {
            id: i,
            word: words[i - 1].clone(),
            lemma: normalize::lemma(&words[i - 1]),
            head,
            rel,
            upos,
            feats: feats_str(i),
        });
    }

    // Model-predicted UPOS==VERB per word (grid row = word index + 1; row 0 is
    // ROOT). Gates the POS-conditional `_IRREGULAR_VERB` remap in the matcher;
    // stage3 localizes training spans with the same s_pos head → no skew.
    let is_verb: Vec<bool> = (0..n).map(|k| upos_str(k + 1) == "VERB").collect();

    let word_rels: Vec<String> = tokens.iter().map(|t| t.rel.clone()).collect();
    let word_upos: Vec<String> = tokens.iter().map(|t| t.upos.clone()).collect();
    let mwes = mwe::detect(&words, &is_verb, &heads, &word_rels, &word_upos, &state.lexicon);

    info!(
        words = n,
        mwes = mwes.len(),
        elapsed_ms = start_time.elapsed().as_millis(),
        "parsed"
    );

    // tracing::debug!(
    //     words = n,
    //     mwes = mwes.len(),
    //     "parsed"
    // );

    Ok(SentenceResult {
        tokens,
        mwes,
    })
}

pub fn run_inference_batch(
    session: &mut Session,
    state: &AppState,
    sentences: &[String],
) -> Vec<anyhow::Result<SentenceResult>> {
    let start_time = std::time::Instant::now();
    let b = sentences.len();
    let mut results = Vec::with_capacity(b);
    if b == 0 { return results; }

    let batch_words: Vec<Vec<String>> = sentences.iter().map(|s| crate::normalize::tokenize(s)).collect();
    let batch_n: Vec<usize> = batch_words.iter().map(|w| w.len()).collect();
    let max_n = batch_n.iter().copied().max().unwrap_or(0);

    if max_n == 0 {
        for _ in 0..b {
            results.push(Ok(SentenceResult { tokens: vec![], mwes: vec![] }));
        }
        return results;
    }

    let max_w_dim = max_n + 1;
    let mut batch_rows: Vec<Vec<Vec<i64>>> = Vec::with_capacity(b);
    let mut max_fix_len = 1;

    for (i, words) in batch_words.iter().enumerate() {
        let n = batch_n[i];
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(n + 1);
        rows.push(vec![state.cls_id]);
        for word in words {
            let enc = state.tokenizer.encode(word.as_str(), false);
            let mut ids: Vec<i64> = match enc {
                Ok(e) => e.get_ids().iter().map(|&id| id as i64).collect(),
                Err(_) => vec![state.unk_id],
            };
            if ids.is_empty() {
                ids.push(state.unk_id);
            }
            rows.push(ids);
        }
        let fix_len = rows.iter().map(|row| row.len()).max().unwrap_or(1).min(MAX_FIX_LEN);
        if fix_len > max_fix_len { max_fix_len = fix_len; }
        batch_rows.push(rows);
    }

    let mut flat_ids = vec![0i64; b * max_w_dim * max_fix_len];
    for batch_idx in 0..b {
        let rows = &batch_rows[batch_idx];
        let n = batch_n[batch_idx];
        for w_id in 0..=n {
            let row = &rows[w_id];
            let f_len = row.len().min(max_fix_len);
            for f in 0..f_len {
                let idx = batch_idx * max_w_dim * max_fix_len + w_id * max_fix_len + f;
                flat_ids[idx] = row[f];
            }
        }
    }

    let subwords_tensor_nd = match ndarray::Array3::from_shape_vec((b, max_w_dim, max_fix_len), flat_ids) {
        Ok(a) => a,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("ndarray error: {}", e))); }
            return results;
        }
    };
    let subwords_ort = match ort::value::Tensor::from_array(subwords_tensor_nd) {
        Ok(t) => t,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("ort tensor error: {:?}", e))); }
            return results;
        }
    };

    let input_name = String::from(session.inputs()[0].name());
    let inputs = ort::inputs![input_name.as_str() => subwords_ort];

    let tensors = match session.run(inputs) {
        Ok(t) => t,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("forward failed: {:?}", e))); }
            return results;
        }
    };

    if tensors.len() < 3 {
        for _ in 0..b { results.push(Err(anyhow::anyhow!("Expected at least 3 output tensors"))); }
        return results;
    }

    let arc = match tensors[0].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            match ndarray::ArrayView3::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("arc dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("arc extract error: {:?}", e))); } return results; }
    };

    let rel = match tensors[1].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize);
            match ndarray::ArrayView4::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("rel dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("rel extract error: {:?}", e))); } return results; }
    };

    let pos = match tensors[2].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            match ndarray::ArrayView3::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("pos dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("pos extract error: {:?}", e))); } return results; }
    };

    let feats_arr: Option<ndarray::ArrayView4<'_, f32>> = if state.feats.is_empty() || tensors.len() < 4 {
        None
    } else {
        match tensors[3].try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize);
                match ndarray::ArrayView4::from_shape(dim, data) {
                    Ok(v) => Some(v),
                    Err(_) => None,
                }
            },
            Err(_) => None,
        }
    };

    let n_rels = rel.shape()[3] as usize;
    let n_upos = pos.shape()[2] as usize;

    for batch_idx in 0..b {
        let n = batch_n[batch_idx];
        if n == 0 {
            results.push(Ok(SentenceResult { tokens: vec![], mwes: vec![] }));
            continue;
        }
        let w_dim = n + 1;
        let words = &batch_words[batch_idx];

        let arc_b = arc.index_axis(ndarray::Axis(0), batch_idx);
        let rel_b = rel.index_axis(ndarray::Axis(0), batch_idx);
        let pos_b = pos.index_axis(ndarray::Axis(0), batch_idx);

        let mut score = vec![vec![f32::NEG_INFINITY; w_dim]; w_dim];
        for v in 0..w_dim {
            for (u, srow) in score.iter_mut().enumerate() {
                if u != v {
                    srow[v] = arc_b[[v, u]];
                }
            }
        }
        let heads = crate::decode::max_arborescence(w_dim, 0, &score);

        let upos_ids: Vec<usize> = (0..w_dim)
            .map(|w| argmax((0..n_upos).map(|k| pos_b[[w, k]])))
            .collect();
        let upos_str = |w: usize| state.upos.get(upos_ids[w]).cloned().unwrap_or_else(|| "X".into());

        let feats_str = |w: usize| -> String {
            let Some(fa) = feats_arr.as_ref() else { return "_".into() };
            let mut parts: Vec<String> = Vec::new();
            for (c, (cat, values)) in state.feats.iter().enumerate() {
                let best = argmax((0..values.len()).map(|k| fa[[batch_idx, w, c, k]]));
                if best != 0 {
                    if let Some(val) = values.get(best) {
                        parts.push(format!("{cat}={val}"));
                    }
                }
            }
            if parts.is_empty() { "_".into() } else { parts.join("|") }
        };

        let mut tokens = Vec::with_capacity(n);
        for i in 1..=n {
            let head = heads[i];
            let rel_id = argmax((0..n_rels).map(|k| rel_b[[i, head, k]]));
            let rel = state.rels.get(rel_id).cloned().unwrap_or_else(|| "dep".into());
            let upos = upos_str(i);

            tokens.push(ParsedToken {
                id: i,
                word: words[i - 1].clone(),
                lemma: crate::normalize::lemma(&words[i - 1]),
                head,
                rel,
                upos,
                feats: feats_str(i),
            });
        }

        let is_verb: Vec<bool> = (0..n).map(|k| upos_str(k + 1) == "VERB").collect();
        let word_rels: Vec<String> = tokens.iter().map(|t| t.rel.clone()).collect();
        let word_upos: Vec<String> = tokens.iter().map(|t| t.upos.clone()).collect();
        let mwes = crate::mwe::detect(words, &is_verb, &heads, &word_rels, &word_upos, &state.lexicon);

        results.push(Ok(SentenceResult { tokens, mwes }));
    }

    info!(
        batch_size = b,
        elapsed_ms = start_time.elapsed().as_millis(),
        "parsed batch"
    );

    results
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

pub fn build_session() -> anyhow::Result<Session> {
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

    let _ = ort::init().with_name("lexparse").commit();
    let mut builder = Session::builder()
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_memory_pattern(mem_pattern)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?
        .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().with_arena_allocator(cpu_arena).build()])
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


// --- end-to-end regression tests ---
//
// These tests require all model artifacts to be present:
//   model/model.onnx   model/vocabs.json   model/tokenizer.json
//   dic/lexicon.jsonl
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
        let mut lexicon = mwe::MweLexicon::load_with_custom(LEXICON_PATH, CUSTOM_LEXICON_PATH)?;
        lexicon.apply_corrections(CORRECTIONS_PATH)?;

        let tokenizer = Tokenizer::from_file("model/tokenizer.json")
            .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        let cls_id = tokenizer.token_to_id("[CLS]").unwrap_or(1) as i64;
        let unk_id = tokenizer.token_to_id("[UNK]").unwrap_or(3) as i64;
        let (job_tx, _job_rx) = mpsc::unbounded_channel();
        Ok(AppState {
            session: LazySession::new(),
            tokenizer,
            rels: vocab.rels,
            upos: vocab.upos,
            feats: vocab.feats,
            lexicon,
            job_tx,
            cls_id,
            unk_id,
        })
    }

    // MWE detection golden cases.
    // Each tuple: (sentence, exact lexicon phrase as returned by the model, expect_mwe).
    // phrase is the winning entry after overlap resolution — run with --nocapture to see
    // what the model actually returns if a case starts failing.
    const MWE_CASES: &[(&str, &str)] = &[
        ("You have an audition today? Break a leg!", "break a leg"),
        ("He spilled the beans about the surprise party.", "spill the beans"),
        ("After years of hard work, she finally kicked the bucket.", "kick the bucket"),
        ("She was over the moon when she heard the news.", "over the moon"),
        // prepositional particles (particle → noun → verb arc)
        ("they just did not hold with such nonsense.", "hold with"),
        ("she spent her time spying on the neighbours.", "spy on"),
    ];

    #[test]
    #[ignore = "requires model artifacts; run with: cargo test -- --include-ignored"]
    fn mwe_detection_golden_cases() {
        let state = build_state().expect("failed to load model artifacts");
        state
            .session
            .with_session(|session| {
                for &(sent, phrase) in MWE_CASES {
                    let result = run_inference(session, &state, sent)
                        .unwrap_or_else(|e| panic!("inference failed for {:?}: {}", sent, e));

                    let hit = result.mwes.iter().find(|m| m.phrase == phrase);

                    if hit.is_none() {
                        panic!(
                            "expected MWE {:?} in {:?} but it was not detected",
                            phrase, sent
                        );
                    }
                }
                Ok(())
            })
            .expect("session error");
    }

    #[test]
    #[ignore = "requires model artifacts; run with: cargo test -- --include-ignored"]
    fn feats_output_smoke() {
        let state = build_state().expect("failed to load model artifacts");
        if state.feats.is_empty() {
            eprintln!("model has no FEATS head — skipping");
            return;
        }
        state
            .session
            .with_session(|session| {
                let sent = "The dogs were running quickly through the muddy fields.";
                let result = run_inference(session, &state, sent).unwrap();
                println!("FEATS for {:?}:", sent);
                for t in &result.tokens {
                    println!("  {:>2} {:<10} {:<6} {}", t.id, t.word, t.upos, t.feats);
                }
                // at least one token must carry morphological features
                assert!(
                    result.tokens.iter().any(|t| t.feats != "_"),
                    "no FEATS predicted for any token"
                );
                // FEATS string must be valid CoNLL-U: Cat=Val pairs joined by '|'
                for t in &result.tokens {
                    if t.feats != "_" {
                        for kv in t.feats.split('|') {
                            assert!(kv.contains('='), "malformed FEATS {:?}", t.feats);
                        }
                    }
                }
                Ok(())
            })
            .expect("session error");
    }

    #[test]
    #[ignore = "requires model artifacts; run with: cargo test -- --include-ignored"]
    fn test_user_sentence_mwes() {
        let state = build_state().expect("failed to load model artifacts");
        state
            .session
            .with_session(|session| {
                let sent = "It could have wrapped its body twice around Uncle Vernon’s car and crushed it into a dustbin – but at the moment it didn’t look in the mood.";
                let result = run_inference(session, &state, sent).unwrap();
                println!("Detected MWEs for user sentence:");
                for m in &result.mwes {
                    println!("  phrase: {:?}, words: {:?}", m.phrase, m.words);
                }
                // A single-word match (one fixed lemma) must never surface —
                // the lexicon drops entries with < 2 fixed words.
                for m in &result.mwes {
                    assert!(m.token_ids.len() >= 2, "single-token MWE: {:?}", m.phrase);
                }
                Ok(())
            })
            .expect("session error");
    }
}
