use anyhow::Result;
use lexparse::mwe::LexiconTreeNode;
use lexparse::{
    run_inference, AppState, LazySession, ParsedToken, Vocab, VocabRaw,
    LEXICON_PATH, VOCAB_PATH,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokenizers::Tokenizer;
use tracing::info;

#[derive(Deserialize, Debug)]
struct BuilderConfig {
    input_file: String,
    target_categories: std::collections::HashMap<String, String>,
}

#[derive(Deserialize, Debug)]
struct WikiSense {
    categories: Option<Vec<String>>,
    tags: Option<Vec<String>>,
    glosses: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
struct WikiForm {
    form: Option<String>,
}

#[derive(Deserialize, Debug)]
struct WikiEntry {
    word: Option<String>,
    pos: Option<String>,
    senses: Option<Vec<WikiSense>>,
    forms: Option<Vec<WikiForm>>,
}

#[derive(Serialize, Debug)]
struct BuilderCandidate {
    id: u32,
    phrase: String,
    categories: Vec<String>,
    pos: Option<String>,
    tags: Vec<String>,
    definition: Option<String>,
    #[serde(skip)]
    variants_to_parse: Vec<String>,
    trees: Vec<LexiconTreeNode>,
}

fn hash_string(s: &str) -> u32 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish() as u32
}

fn replace_slots(text: &str, re_poss: &Regex, re_obj: &Regex) -> String {
    let s = re_poss.replace_all(text, "SLOT's");
    re_obj.replace_all(&s, "SLOT").into_owned()
}

fn extract_tree(node: &ParsedToken, tokens: &[ParsedToken]) -> LexiconTreeNode {
    let is_slot = node.lemma == "SLOT" || node.lemma == "slot";

    let mut tree = LexiconTreeNode {
        lemma: if is_slot { "SLOT".to_string() } else { node.lemma.clone() },
        rel: node.rel.clone(),
        is_slot,
        deps: vec![],
    };

    if !is_slot {
        let deps: Vec<LexiconTreeNode> = tokens
            .iter()
            .filter(|c| c.head == node.id)
            .map(|c| extract_tree(c, tokens))
            .collect();
        tree.deps = deps;
    }

    tree
}

fn build_mwe_tree(tokens: &[ParsedToken]) -> Option<LexiconTreeNode> {
    let root = tokens.iter().find(|t| t.rel == "root" || t.head == 0)?;
    let raw = extract_tree(root, tokens);

    // Prune SLOT deps is already handled inside extract_tree (if is_slot is true, it doesn't add deps)
    Some(raw)
}

fn main() -> Result<()> {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,ort=warn"))
        .init();
    info!("Starting native Rust dictionary builder...");

    let re_poss = Regex::new(r"(?i)\b(someone's|one's|somebody's)\b").unwrap();
    let re_obj = Regex::new(r"(?i)\b(someone|something|somebody|one)\b").unwrap();
    // Prepositions and particles that often take an object in a sentence.
    // If a variant ends with one of these, we will generate an extra variant with " SLOT".
    let re_prep_end = Regex::new(r"(?i)\b(from|with|to|in|on|at|for|about|of|into|onto|upon|out|by|as|after|before|over|under|through|against|up|down|around|away|back|off)$").unwrap();

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "builder_config.toml".to_string());
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config from {}: {}", config_path, e))?;
    let config: BuilderConfig = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse TOML from {}: {}", config_path, e))?;
    info!("Loaded config from {}: {:?}", config_path, config);

    let file = File::open(&config.input_file)?;
    let reader = BufReader::new(file);

    let mut candidates = Vec::new();
    let mut processed_words = HashSet::new();
    let mut line_count = 0;

    info!("Scanning Wiktionary Dump...");
    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        if line_count % 50_000 == 0 {
            info!("Scanned {} lines...", line_count);
        }

        if let Ok(entry) = serde_json::from_str::<WikiEntry>(&line) {
            let Some(word) = entry.word else { continue };
            if word.trim().split_whitespace().count() < 2 {
                continue;
            }

            let mut is_target = false;
            let mut matched_types = HashSet::new();
            let mut tags = HashSet::new();
            let mut glosses = Vec::new();

            if let Some(senses) = entry.senses {
                for sense in senses {
                    if let Some(categories) = sense.categories {
                        for cat in categories {
                            if let Some(mapped_cat) = config.target_categories.get(cat.as_str()) {
                                is_target = true;
                                matched_types.insert(mapped_cat.clone());

                                if let Some(t) = &sense.tags {
                                    for tg in t { tags.insert(tg.clone()); }
                                }
                                if let Some(g) = &sense.glosses {
                                    glosses.extend(g.clone());
                                }
                            }
                        }
                    }
                }
            }

            if !is_target || processed_words.contains(&word) {
                continue;
            }
            processed_words.insert(word.clone());

            let mut variants = HashSet::new();
            variants.insert(word.clone());
            if let Some(forms) = entry.forms {
                for form in forms {
                    if let Some(f) = form.form { variants.insert(f); }
                }
            }

            let mut variants_to_parse = Vec::new();
            for v in variants {
                let replaced = replace_slots(&v, &re_poss, &re_obj);
                if re_prep_end.is_match(&replaced) {
                    variants_to_parse.push(format!("{} SLOT", replaced));
                }
                variants_to_parse.push(replaced);
            }

            candidates.push(BuilderCandidate {
                id: hash_string(&word),
                phrase: word,
                categories: matched_types.into_iter().collect(),
                pos: entry.pos,
                tags: tags.into_iter().collect(),
                definition: glosses.into_iter().next(),
                variants_to_parse,
                trees: vec![],
            });
        }
    }

    info!("Scanned {} lines, found {} MWE candidates.", line_count, candidates.len());

    // Prepare Model State
    info!("loading vocab from {VOCAB_PATH}");
    let vocab: Vocab = serde_json::from_str::<VocabRaw>(&std::fs::read_to_string(VOCAB_PATH)?)?.into();

    info!("loading tokenizer");
    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

    let (job_tx, _job_rx) = mpsc::unbounded_channel();

    let state = Arc::new(AppState {
        session: LazySession::new(),
        tokenizer,
        rels: vocab.rels,
        upos: vocab.upos,
        lexicon: lexparse::mwe::MweLexicon { entries: vec![], index: std::collections::HashMap::new() }, // dummy empty lexicon since we don't need detection
        job_tx,
    });

    info!("Generating parse trees natively...");

    let total = candidates.len();

    // Resumption logic
    let mut already_processed = HashSet::new();
    if std::path::Path::new(LEXICON_PATH).exists() {
        if let Ok(file) = File::open(LEXICON_PATH) {
            let reader = BufReader::new(file);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(entry) = serde_json::from_str::<lexparse::mwe::LexiconEntry>(&line) {
                    already_processed.insert(entry.phrase);
                }
            }
        }
        info!("Resuming: found {} already processed phrases.", already_processed.len());
    }

    let mut out_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LEXICON_PATH)?;

    state.session.with_session(|session| {
        for (i, cand) in candidates.iter_mut().enumerate() {
            if already_processed.contains(&cand.phrase) {
                continue;
            }

            if i % 10 == 0 || i == total - 1 {
                let pct = (i * 100) / total;
                info!("🚀 Generating trees: [{}/{}] ({}%)", i, total, pct);
            }

            for var in &cand.variants_to_parse {
                if let Ok(res) = run_inference(session, &state, var) {
                    if let Some(tree) = build_mwe_tree(&res.tokens) {
                        cand.trees.push(tree);
                    }
                }
            }

            if !cand.trees.is_empty() {
                let j = serde_json::to_string(&cand)?;
                writeln!(out_file, "{}", j)?;
            }
        }
        Ok::<(), anyhow::Error>(())
    })?;

    info!("All done! Lexicon generated at {}", LEXICON_PATH);

    Ok(())
}
