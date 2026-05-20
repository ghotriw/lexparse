//! Bit-exact Rust port of `py_example/matcher.py` (anti-skew §7).
//!
//! Lemma-gappy / slot matcher over a tokenized sentence. Given the canonical
//! token list (`normalize::tokenize`) and a lexicon entry's `elements`, locate
//! the content-word token indices the idiom occupies. Discontinuous spans fall
//! out naturally; the returned indices are the FIXED content words only.

use crate::normalize::lemma_pos;
use serde::Deserialize;
use std::collections::BTreeMap;

const SLOT_MAX: usize = 4; // tokens allowed inside an explicit lexicon slot
const GAP_MAX: usize = 1; // tokens allowed between two fixed lemmas, no slot

/// One lexicon element: a fixed word or a gap slot. JSON form is
/// `["w", "<lemma>"]` or `["slot"]`.
#[derive(Debug, Clone)]
pub enum Element {
    Word(String),
    Slot,
}

impl<'de> Deserialize<'de> for Element {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let parts: Vec<String> = Vec::deserialize(d)?;
        match parts.first().map(String::as_str) {
            Some("w") => Ok(Element::Word(parts.get(1).cloned().unwrap_or_default())),
            Some("slot") => Ok(Element::Slot),
            other => Err(serde::de::Error::custom(format!(
                "unknown lexicon element kind: {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    pub elements: Vec<Element>,
    /// Canonical fixed-lemma identity (slots dropped) — used for lookup/tests.
    #[serde(default)]
    #[allow(dead_code)]
    pub fixed_key: Vec<String>,
    #[serde(default)]
    pub surface: String,
    #[serde(default)]
    pub has_slot: bool,
    /// MWE category from the enriched lexicon: idiom, phrasal_verb,
    /// collocation_phrase, proverb_saying, etc.
    #[serde(default)]
    pub category: String,
}





/// POS-aware lemmatize: `is_verb[i]` (from model-predicted UPOS) enables the
/// `_IRREGULAR_VERB` remap for token `i`. Mirrors `matcher.lemmatize(tokens,
/// upos)`. `is_verb` must be aligned 1:1 with `tokens`.
pub fn lemmatize_pos(tokens: &[String], is_verb: &[bool]) -> Vec<String> {
    tokens
        .iter()
        .zip(is_verb.iter())
        .map(|(t, &v)| lemma_pos(t, v))
        .collect()
}

/// Prefix-tolerant lemma equality (`matcher._eq`). Absorbs the residual
/// inflection the deliberately-conservative `lemma` leaves, symmetrically.
fn eq(a: &str, b: &str) -> bool {
    a == b
}

/// `elements` -> (fixed_lemmas, gaps) where `gaps[i]` is the max tokens allowed
/// between fixed word i-1 and fixed word i (`gaps[0]` unused, == 0).
fn plan(elements: &[Element]) -> (Vec<String>, Vec<usize>) {
    let mut fixed = Vec::new();
    let mut gaps = Vec::new();
    let mut pending_slot = false;
    for e in elements {
        match e {
            Element::Slot => pending_slot = true,
            Element::Word(w) => {
                if fixed.is_empty() {
                    gaps.push(0);
                } else {
                    gaps.push(if pending_slot { SLOT_MAX } else { GAP_MAX });
                }
                fixed.push(w.clone());
                pending_slot = false;
            }
        }
    }
    (fixed, gaps)
}

/// Minimal-span match of one lexicon entry against pre-lemmatized sentence
/// tokens. Returns the fixed-content token indices (ascending, possibly
/// non-contiguous), or `None`. "Minimal span" = smallest (last-first); ties
/// broken leftmost. Mirrors `matcher.match_entry`.
pub fn match_entry(lem: &[String], elements: &[Element]) -> Option<Vec<usize>> {
    let (fixed, gaps) = plan(elements);
    if fixed.is_empty() {
        return None;
    }
    let n = lem.len();

    // state[p] = (start, path): best (largest start => smallest span) way to
    // match fixed[0..=k] ending exactly at sentence position p. BTreeMap keeps
    // iteration deterministic (ascending p), matching the Python insertion
    // order this DP relies on for the strict-`>` tie-break.
    let mut state: BTreeMap<usize, (usize, Vec<usize>)> = BTreeMap::new();
    for (p, lp) in lem.iter().enumerate() {
        if eq(lp, &fixed[0]) {
            state.insert(p, (p, vec![p]));
        }
    }
    if state.is_empty() {
        return None;
    }

    for k in 1..fixed.len() {
        let mut nxt: BTreeMap<usize, (usize, Vec<usize>)> = BTreeMap::new();
        for (&p, (start, path)) in &state {
            let hi = (p + 1 + gaps[k]).min(n.saturating_sub(1));
            // `q` is the sentence position (it becomes a path index), so the
            // explicit range mirrors the Python DP one-for-one.
            #[allow(clippy::needless_range_loop)]
            for q in (p + 1)..=hi {
                if q >= n {
                    break;
                }
                if eq(&lem[q], &fixed[k]) {
                    let better = match nxt.get(&q) {
                        None => true,
                        Some((prev_start, _)) => *start > *prev_start,
                    };
                    if better {
                        let mut np = path.clone();
                        np.push(q);
                        nxt.insert(q, (*start, np));
                    }
                }
            }
        }
        state = nxt;
        if state.is_empty() {
            return None;
        }
    }

    // smallest span (q - start), then leftmost start
    state
        .into_iter()
        .min_by_key(|(q, (start, _))| (*q - *start, *start))
        .map(|(_, (_, path))| path)
}

/// All lexicon hits in the sentence (the inference/recall spec). Returns
/// `(entry_index, idxs)`; longer idioms first (stable) so multiword hits are
/// preferred when post-filtering. Mirrors `matcher.scan`.
pub fn scan(lex: &[Entry], lem: &[String]) -> Vec<(usize, Vec<usize>)> {
    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, e) in lex.iter().enumerate() {
        if let Some(idxs) = match_entry(lem, &e.elements) {
            out.push((i, idxs));
        }
    }
    // stable sort by descending span length (preserves lexicon order on ties)
    out.sort_by_key(|(_, idxs)| std::cmp::Reverse(idxs.len()));
    out
}
