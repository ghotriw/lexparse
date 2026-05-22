//! Slot/gap lemma matcher over a tokenized sentence.
//!
//! Given the lemmatized sentence tokens and a lexicon entry's `elements`
//! (`Word` / `Slot`), locate the fixed content-word token indices the MWE
//! occupies. Discontinuous spans fall out naturally; the returned indices are
//! the FIXED content words only (slots are not anchored to specific tokens).

use crate::normalize::lemma_pos;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const SLOT_MAX: usize = 4; // tokens allowed inside an explicit lexicon slot
const GAP_MAX: usize = 1; // tokens allowed between two fixed lemmas, no slot

/// One lexicon element: a fixed lemma or a gap slot.
/// JSON form is `["w", "<lemma>"]` or `["slot"]`.
#[derive(Debug, Clone, PartialEq)]
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

impl Serialize for Element {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Element::Word(w) => ["w", w.as_str()].serialize(s),
            Element::Slot => ["slot"].serialize(s),
        }
    }
}

/// POS-aware lemmatize: `is_verb[i]` (from model-predicted UPOS) enables the
/// `_IRREGULAR_VERB` remap for token `i`. `is_verb` must be aligned 1:1 with
/// `tokens`.
pub fn lemmatize_pos(tokens: &[String], is_verb: &[bool]) -> Vec<String> {
    tokens
        .iter()
        .zip(is_verb.iter())
        .map(|(t, &v)| lemma_pos(t, v))
        .collect()
}

/// Prefix-tolerant lemma equality. Absorbs the residual inflection the
/// deliberately-conservative `lemma` leaves, symmetrically. The 4-char floor
/// keeps short function words (`the`/`but`/`and`) exact — at 3 chars `the`
/// would spuriously prefix-match `then`, `there`, `they`.
fn eq(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let (short, long) = if a.chars().count() <= b.chars().count() {
        (a, b)
    } else {
        (b, a)
    };
    short.chars().count() >= 4 && long.starts_with(short)
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

/// Number of fixed (non-slot) lemmas in an element list.
pub fn fixed_count(elements: &[Element]) -> usize {
    elements
        .iter()
        .filter(|e| matches!(e, Element::Word(_)))
        .count()
}

/// First fixed lemma of an element list, used to index the lexicon.
pub fn first_fixed(elements: &[Element]) -> Option<&str> {
    elements.iter().find_map(|e| match e {
        Element::Word(w) => Some(w.as_str()),
        Element::Slot => None,
    })
}

/// Minimal-span match of one lexicon entry against pre-lemmatized sentence
/// tokens. Returns the fixed-content token indices (ascending, possibly
/// non-contiguous), or `None`. "Minimal span" = smallest (last-first); ties
/// broken leftmost.
pub fn match_entry(lem: &[String], elements: &[Element]) -> Option<Vec<usize>> {
    let (fixed, gaps) = plan(elements);
    if fixed.is_empty() {
        return None;
    }
    let n = lem.len();

    // state[p] = (start, path): best (largest start => smallest span) way to
    // match fixed[0..=k] ending exactly at sentence position p.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Element {
        Element::Word(s.to_string())
    }

    fn lems(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    #[test]
    fn matches_contiguous() {
        let elems = [w("spill"), w("the"), w("bean")];
        let idx = match_entry(&lems("he spill the bean today"), &elems).unwrap();
        assert_eq!(idx, vec![1, 2, 3]);
    }

    #[test]
    fn matches_across_slot() {
        let elems = [w("make"), Element::Slot, w("up"), w("mind")];
        // "make her mind up" — slot absorbs "her", up before mind here won't fit;
        // use canonical order: make <slot> up mind
        let idx = match_entry(&lems("they make her up mind now"), &elems).unwrap();
        assert_eq!(idx, vec![1, 3, 4]);
    }

    #[test]
    fn no_match_when_gap_too_wide() {
        let elems = [w("kick"), w("bucket")];
        // GAP_MAX = 1, three tokens between -> no match
        assert!(match_entry(&lems("kick a big old bucket"), &elems).is_none());
    }

    #[test]
    fn slot_allows_wider_gap() {
        let elems = [w("kick"), Element::Slot, w("bucket")];
        let idx = match_entry(&lems("kick a big old bucket"), &elems).unwrap();
        assert_eq!(idx, vec![0, 4]);
    }

    #[test]
    fn prefix_tolerant_eq() {
        let elems = [w("over"), w("moon")];
        // "moon" is a >=3-char prefix of "moons" -> tolerated
        assert!(match_entry(&lems("over moons"), &elems).is_some());
        // short function words are not prefix-matched (avoids spurious hits)
        assert!(eq("moon", "moons"));
        assert!(!eq("go", "gone"));
        assert!(!eq("the", "then"));
        assert!(!eq("but", "butter"));
    }
}
