//! MWE detection: slot/gap lemma matching over the sentence token stream.
//!
//! Pipeline per sentence:
//!   1. Lemmatize sentence tokens (POS-aware).
//!   2. For each lexicon entry whose first fixed lemma occurs in the sentence,
//!      run the slot/gap matcher (`matcher::match_entry`).
//!   3. Phrasal-verb hits are gated by a single dependency-arc check.
//!   4. Overlapping/duplicate hits are resolved by `keep_mask`.

use crate::matcher::{self, Element};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct LexiconEntry {
    pub id: u32,
    pub phrase: String,
    #[serde(default)]
    pub categories: Vec<String>,
    pub pos: Option<String>,
    pub definition: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub elements: Vec<Element>,
}

pub struct MweLexicon {
    pub entries: Vec<LexiconEntry>,
    /// First fixed lemma -> indices of entries starting with it.
    pub index: HashMap<String, Vec<usize>>,
}

impl MweLexicon {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::load_paths(&[path])
    }

    /// Load the main lexicon and, if `custom_path` exists, append entries from it.
    pub fn load_with_custom(main_path: &str, custom_path: &str) -> anyhow::Result<Self> {
        if std::path::Path::new(custom_path).exists() {
            Self::load_paths(&[main_path, custom_path])
        } else {
            Self::load_paths(&[main_path])
        }
    }

    fn load_paths(paths: &[&str]) -> anyhow::Result<Self> {
        use std::io::BufRead;
        let mut entries = Vec::new();
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();

        for &path in paths {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(entry) = serde_json::from_str::<LexiconEntry>(&line) else {
                    continue;
                };
                // Entries with fewer than two fixed lemmas act as single-word
                // wildcards and flood the output — skip them.
                if matcher::fixed_count(&entry.elements) < 2 {
                    continue;
                }
                let Some(first) = matcher::first_fixed(&entry.elements) else {
                    continue;
                };
                let idx = entries.len();
                index.entry(first.to_string()).or_default().push(idx);
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
    pub phrase: String,
    pub definition: Option<String>,
    pub categories: Vec<String>,
    pub has_slot: bool,
    pub token_ids: Vec<usize>,
    pub words: Vec<String>,
    pub surface: String,
    pub discontinuous: bool,
}

fn is_phrasal_verb(entry: &LexiconEntry) -> bool {
    entry.categories.iter().any(|c| c == "phrasal_verb")
}

/// Single dependency-arc gate for phrasal verbs: the verb token must be tagged
/// VERB, and the particle must be connected to the verb — either directly
/// (particle → verb, typical for adverbial particles: "look up the word") or
/// via one noun hop (particle → noun → verb, typical for prepositional particles
/// in UD: "hold with such nonsense" → with→nonsense(case)→hold(obl)).
///
/// The 1-hop path explicitly blocks `mark` deprel: infinitival `to` in
/// "went to find" has deprel=mark and points to the infinitive, not the main
/// verb — it is not a particle and must not trigger the phrasal-verb gate.
fn phrasal_arc_ok(idxs: &[usize], is_verb: &[bool], heads: &[usize], rels: &[String]) -> bool {
    let verb = idxs[0];
    if !is_verb.get(verb).copied().unwrap_or(false) {
        return false;
    }
    let Some(&particle) = idxs.get(1) else {
        return true;
    };
    let particle_head = heads.get(particle + 1).copied().unwrap_or(0);
    // Direct: particle → verb
    if particle_head == verb + 1 {
        return true;
    }
    // Indirect via case marker: particle → noun → verb.
    // Exclude `mark` (infinitival `to`) — it attaches to the complement verb,
    // not to the governing verb, and is not a phrasal-verb particle.
    let particle_rel = rels.get(particle).map(String::as_str).unwrap_or("");
    particle_head > 0
        && heads.get(particle_head).copied() == Some(verb + 1)
        && particle_rel != "mark"
}

pub fn detect(
    words: &[String],
    is_verb: &[bool],
    heads: &[usize],
    rels: &[String],
    upos: &[String],
    lexicon: &MweLexicon,
) -> Vec<MweMatch> {
    let lemmas = matcher::lemmatize_pos(words, is_verb);

    // Candidate entries: those whose first fixed lemma occurs in the sentence.
    let mut candidate_entries: Vec<usize> = Vec::new();
    for lemma in &lemmas {
        if let Some(list) = lexicon.index.get(lemma) {
            candidate_entries.extend(list);
        }
    }
    candidate_entries.sort_unstable();
    candidate_entries.dedup();

    let mut cands = Vec::new();
    for &entry_idx in &candidate_entries {
        let entry = &lexicon.entries[entry_idx];
        let Some(idxs) = matcher::match_entry(&lemmas, upos, words, &entry.elements) else {
            continue;
        };

        if is_phrasal_verb(entry) && !phrasal_arc_ok(&idxs, is_verb, heads, rels) {
            continue;
        }

        let lo = *idxs.first().unwrap();
        let hi = *idxs.last().unwrap();
        let discontinuous = hi - lo + 1 != idxs.len();
        let surface = (lo..=hi)
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
            phrase: entry.phrase.clone(),
            definition: entry.definition.clone(),
            categories: entry.categories.clone(),
            has_slot: entry.elements.iter().any(|e| matches!(e, Element::Slot(_))),
            token_ids: idxs.iter().map(|&j| j + 1).collect(),
            words: idxs.iter().map(|&j| words[j].clone()).collect(),
            surface,
            discontinuous,
        });
    }

    let keep = keep_mask(&cands);
    cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect()
}

fn keep_mask(cands: &[MweMatch]) -> Vec<bool> {
    let n = cands.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let overlap = cands[i]
                .token_ids
                .iter()
                .any(|t| cands[j].token_ids.contains(t));
            if !overlap {
                continue;
            }
            let len_i = cands[i].token_ids.len();
            let len_j = cands[j].token_ids.len();
            let gap_i = cands[i].token_ids.last().unwrap() - cands[i].token_ids.first().unwrap()
                + 1
                - len_i;
            let gap_j = cands[j].token_ids.last().unwrap() - cands[j].token_ids.first().unwrap()
                + 1
                - len_j;

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
    keep
}
