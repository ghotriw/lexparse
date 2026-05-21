//! MWE detection: lexicon subgraph matching over sentence graph.
//!
//! Pipeline per sentence:
//!   1. Index lexicon trees by root lemma.
//!   2. Iterate over sentence tokens, trying to match candidate subgraphs.
//!   3. MWE candidates are filtered by overlap.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct LexiconTreeNode {
    pub lemma: String,
    pub rel: String,
    #[serde(default)]
    pub is_slot: bool,
    #[serde(default)]
    pub deps: Vec<LexiconTreeNode>,
}

#[derive(Debug, Deserialize)]
pub struct LexiconEntry {
    pub id: u32,
    pub phrase: String,
    #[serde(default)]
    pub categories: Vec<String>,
    pub pos: Option<String>,
    pub definition: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub trees: Vec<LexiconTreeNode>,
}

pub struct MweLexicon {
    pub entries: Vec<LexiconEntry>,
    pub index: HashMap<String, Vec<(usize, usize)>>,
}

impl MweLexicon {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        use std::io::BufRead;

        let mut entries = Vec::new();
        let mut index = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(mut entry) = serde_json::from_str::<LexiconEntry>(&line) {
                // Filter out 1-node trees (like "SLOT" or "101") because they act like wildcards
                // and match single words, flooding the output with false positives.
                entry.trees.retain(|t| !t.deps.is_empty());
                
                if entry.trees.is_empty() {
                    continue; // Skip entries that have no valid multi-word trees
                }
                
                let entry_idx = entries.len();
                for (tree_idx, tree) in entry.trees.iter().enumerate() {
                    let key = if tree.is_slot {
                        "*".to_string()
                    } else {
                        tree.lemma.clone()
                    };
                    index.entry(key).or_insert_with(Vec::new).push((entry_idx, tree_idx));
                }
                entries.push(entry);
            }
        }

        Ok(Self { entries, index })
    }
}

#[derive(Debug, Serialize)]
pub struct MweMatch {
    pub surface: String,
    pub categories: Vec<String>,
    pub has_slot: bool,
    pub token_ids: Vec<usize>,
    pub words: Vec<String>,
    pub span_text: String,
    pub prob: f32,
    pub discontinuous: bool,
    pub tree_connected: bool,
}

pub fn detect(
    words: &[String],
    is_verb: &[bool],
    heads: &[usize],
    rels: &[String],
    lexicon: &MweLexicon,
) -> Vec<MweMatch> {
    let mut cands = Vec::new();
    let n = words.len();
    
    // Compute lemmatized words for the sentence
    let lemmas = crate::matcher::lemmatize_pos(words, is_verb);
    
    // Build sentence children map: children[head] = vec of child token indices (0-based)
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for i in 0..n {
        let head = heads[i + 1];
        if head <= n {
            children[head].push(i);
        }
    }

    for i in 0..n {
        let lemma = &lemmas[i];
        
        let mut candidates_to_check = Vec::new();
        if let Some(list) = lexicon.index.get(lemma) {
            candidates_to_check.extend(list);
        }
        if let Some(list) = lexicon.index.get("*") {
            candidates_to_check.extend(list);
        }

        for &(entry_idx, tree_idx) in candidates_to_check.iter() {
            let entry = &lexicon.entries[entry_idx];
            let tree = &entry.trees[tree_idx];
            
            let mut matched_nodes = Vec::new();
            if crate::matcher::match_tree(i, tree, true, &lemmas, &children, rels, &mut matched_nodes) {
                matched_nodes.sort_unstable();
                matched_nodes.dedup();
                
                let has_slot = has_slot_recursive(tree);
                let lo = *matched_nodes.iter().min().unwrap();
                let hi = *matched_nodes.iter().max().unwrap();
                let discontinuous = hi - lo + 1 != matched_nodes.len();
                let span_text = (lo..=hi)
                    .map(|j| words[j].clone())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .replace(" 's", "'s")
                    .replace(" n't", "n't")
                    .replace(" ,", ",")
                    .replace(" .", ".")
                    .replace(" !", "!")
                    .replace(" ?", "?");
                
                cands.push(MweMatch {
                    surface: entry.phrase.clone(),
                    categories: entry.categories.clone(),
                    has_slot,
                    token_ids: matched_nodes.iter().map(|&j| j + 1).collect(),
                    words: matched_nodes.iter().map(|&j| words[j].clone()).collect(),
                    span_text,
                    prob: 1.0,
                    discontinuous,
                    tree_connected: true, // graph isomorphism guarantees tree connectivity
                });
            }
        }
    }

    // Filter overlapping/duplicate spans
    let spans: Vec<(usize, usize, f32)> = cands
        .iter()
        .map(|c| {
            let lo = c.token_ids[0] - 1;
            let hi = c.token_ids.last().unwrap() - 1;
            (lo, hi, c.prob)
        })
        .collect();
    let keep = keep_mask(&spans);
    cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect()
}

fn has_slot_recursive(node: &LexiconTreeNode) -> bool {
    if node.is_slot {
        return true;
    }
    node.deps.iter().any(has_slot_recursive)
}

fn keep_mask(spans: &[(usize, usize, f32)]) -> Vec<bool> {
    let n = spans.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        let (ilo, ihi, iprob) = spans[i];
        for (j, &(jlo, jhi, jprob)) in spans.iter().enumerate() {
            if i == j {
                continue;
            }
            if jlo <= ilo && jhi >= ihi {
                let strictly = jlo < ilo || jhi > ihi;
                if strictly || jprob > iprob || (jprob == iprob && j < i) {
                    keep[i] = false;
                    break;
                }
            }
        }
    }
    keep
}
