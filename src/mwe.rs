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
    pub id: u32,
    pub pos: Option<String>,
    pub surface: String,
    pub categories: Vec<String>,
    pub has_slot: bool,
    pub token_ids: Vec<usize>,
    pub words: Vec<String>,
    pub span_text: String,
    pub discontinuous: bool,
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
                let root_idx = matched_nodes[0];
                let is_phrasal_verb = entry.categories.iter().any(|c| c == "phrasal_verb");
                
                if is_phrasal_verb {
                    // 1. For a phrasal verb, the root must actually be used as a verb
                    if !is_verb[root_idx] {
                        continue;
                    }
                    // 2. The verb must be the first word among the matched non-slot literal tokens
                    // (e.g. rejects "to look" matching "look to" because "to" comes before "look")
                    let min_idx = *matched_nodes.iter().min().unwrap();
                    if root_idx != min_idx {
                        continue;
                    }
                }

                matched_nodes.sort_unstable();
                matched_nodes.dedup();
                
                let has_slot = has_slot_recursive(tree);

                // HARDCODED IGNORES FOR INTRANSITIVE FALSE POSITIVES
                // "eat in" is an intransitive phrasal verb. The dictionary builder
                // aggressively generated a wildcard tree (verb -> SLOT -> in) for it.
                // We reject this wildcard match to prevent matching "ate in the restaurant",
                // while still allowing the legitimate continuous usage ("we decided to eat in").
                if has_slot && entry.phrase == "eat in" {
                    continue;
                }

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
                    id: entry.id,
                    pos: entry.pos.clone(),
                    surface: entry.phrase.clone(),
                    categories: entry.categories.clone(),
                    has_slot,
                    token_ids: matched_nodes.iter().map(|&j| j + 1).collect(),
                    words: matched_nodes.iter().map(|&j| words[j].clone()).collect(),
                    span_text,
                    discontinuous,
                });
            }
        }
    }

    // Filter overlapping/duplicate spans based on exact token collision
    let keep = keep_mask(&cands);
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

fn keep_mask(cands: &[MweMatch]) -> Vec<bool> {
    let n = cands.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Check for exact token collision
            let mut overlap = false;
            for &t1 in &cands[i].token_ids {
                if cands[j].token_ids.contains(&t1) {
                    overlap = true;
                    break;
                }
            }
            if overlap {
                let len_i = cands[i].token_ids.len();
                let len_j = cands[j].token_ids.len();
                
                let gap_i = cands[i].token_ids.last().unwrap() - cands[i].token_ids.first().unwrap() + 1 - len_i;
                let gap_j = cands[j].token_ids.last().unwrap() - cands[j].token_ids.first().unwrap() + 1 - len_j;
                
                let j_wins = if len_j != len_i {
                    len_j > len_i // longer phrase wins
                } else if gap_j != gap_i {
                    gap_j < gap_i // more continuous phrase wins
                } else {
                    j < i // tie-breaker
                };
                
                if j_wins {
                    keep[i] = false;
                    break;
                }
            }
        }
    }
    keep
}
