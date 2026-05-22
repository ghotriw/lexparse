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

/// Constraint on which tokens may fill a `Slot`.
/// Checked against the UPOS of the first token in the slot span.
#[derive(Debug, Clone, PartialEq)]
pub enum SlotType {
    /// No constraint — matches any fill (legacy `["slot"]`).
    Any,
    /// First fill token must be PRON (possessive/reflexive pronoun placeholder:
    /// "one's", "himself", "yourself", …). Rejects articles and adjectives.
    Pron,
    /// First fill token must be nominal (NOUN, PROPN, PRON, or DET).
    Noun,
}

/// One lexicon element: a fixed lemma or a typed gap slot.
/// JSON forms: `["w", "<lemma>"]`, `["slot"]`, `["slot:pron"]`, `["slot:noun"]`.
#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    Word(String),
    Slot(SlotType),
}

impl<'de> Deserialize<'de> for Element {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let parts: Vec<String> = Vec::deserialize(d)?;
        match parts.first().map(String::as_str) {
            Some("w") => Ok(Element::Word(parts.get(1).cloned().unwrap_or_default())),
            Some("slot") => Ok(Element::Slot(SlotType::Any)),
            Some("slot:pron") => Ok(Element::Slot(SlotType::Pron)),
            Some("slot:noun") => Ok(Element::Slot(SlotType::Noun)),
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
            Element::Slot(SlotType::Any) => ["slot"].serialize(s),
            Element::Slot(SlotType::Pron) => ["slot:pron"].serialize(s),
            Element::Slot(SlotType::Noun) => ["slot:noun"].serialize(s),
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

/// Inflectional endings the conservative `lemma` may leave behind. The residual
/// of a prefix-tolerant match must be one of these — otherwise `thin` would
/// spuriously prefix-match `thing` (residual `g` is not inflectional).
const INFLECTION_SUFFIXES: &[&str] = &["s", "es", "ed", "d", "ing", "n", "en", "ies", "ied"];

/// Prefix-tolerant lemma equality. Absorbs the residual inflection the
/// deliberately-conservative `lemma` leaves, symmetrically. The 4-char floor
/// keeps short function words (`the`/`but`/`and`) exact — at 3 chars `the`
/// would spuriously prefix-match `then`, `there`, `they`. The residual must be
/// a recognized inflectional suffix, so `thin`/`thing` does not match.
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
    if short.chars().count() < 4 || !long.starts_with(short) {
        return false;
    }
    INFLECTION_SUFFIXES.contains(&&long[short.len()..])
}

/// `elements` -> (fixed_lemmas, gap_limits, slot_types).
/// `gaps[i]` = max tokens between fixed[i-1] and fixed[i] (`gaps[0]` unused).
/// `slot_types[i]` = UPOS constraint for that gap; `Any` when no explicit slot.
fn plan(elements: &[Element]) -> (Vec<String>, Vec<usize>, Vec<SlotType>) {
    let mut fixed = Vec::new();
    let mut gaps = Vec::new();
    let mut slot_types = Vec::new();
    let mut pending: Option<SlotType> = None;
    for e in elements {
        match e {
            Element::Slot(t) => pending = Some(t.clone()),
            Element::Word(w) => {
                let slot = pending.take();
                if fixed.is_empty() {
                    gaps.push(0);
                    slot_types.push(SlotType::Any);
                } else {
                    match slot {
                        Some(t) => { gaps.push(SLOT_MAX); slot_types.push(t); }
                        None    => { gaps.push(GAP_MAX);  slot_types.push(SlotType::Any); }
                    }
                }
                fixed.push(w.clone());
            }
        }
    }
    (fixed, gaps, slot_types)
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
        Element::Slot(_) => None,
    })
}

/// Hard sentence/clause terminators no MWE span may cross — in a plain gap or
/// an explicit slot alike. Hyphens and other intra-word punctuation are
/// deliberately excluded: the tokenizer splits `twenty-one` and lexicon phrases
/// may legitimately span such tokens.
fn is_hard_boundary(s: &str) -> bool {
    matches!(
        s,
        ";" | ":"
            | "."
            | "!"
            | "?"
            | "…"
            | "\""
            | "'"
            | "\u{2019}"
            | "\u{201c}"
            | "\u{201d}"
            | "\u{2018}"
            | "--"
            | "—"
            | "("
            | ")"
    ) || (s.len() > 1 && s.chars().all(|c| c == '.'))
}

/// Returns true if the slot constraint is satisfied by the fill span `fill_start..fill_end`
/// (token indices into `upos`).
fn slot_fill_ok(slot_type: &SlotType, fill_start: usize, fill_end: usize, upos: &[String]) -> bool {
    match slot_type {
        SlotType::Any => true,
        SlotType::Pron => {
            // Non-empty fill whose first token is a pronoun.
            fill_end > fill_start
                && upos.get(fill_start).map(String::as_str) == Some("PRON")
        }
        SlotType::Noun => {
            // Non-empty fill whose first token is nominal.
            fill_end > fill_start
                && matches!(
                    upos.get(fill_start).map(String::as_str),
                    Some("NOUN" | "PROPN" | "PRON" | "DET")
                )
        }
    }
}

/// Minimal-span match of one lexicon entry against pre-lemmatized sentence
/// tokens. `upos` must be parallel to `lem` (UPOS tag per token).
/// Returns the fixed-content token indices (ascending, possibly non-contiguous),
/// or `None`. "Minimal span" = smallest (last-first); ties broken leftmost.
pub fn match_entry(
    lem: &[String],
    upos: &[String],
    surface: &[String],
    elements: &[Element],
) -> Option<Vec<usize>> {
    let (fixed, gaps, slot_types) = plan(elements);
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
                    let is_punct = |t: usize| {
                        upos.get(t).map(String::as_str) == Some("PUNCT")
                    };
                    // No MWE span may cross a hard sentence/clause terminator.
                    if (p + 1..q).any(|t| {
                        is_punct(t) && surface.get(t).is_some_and(|s| is_hard_boundary(s))
                    }) {
                        continue;
                    }
                    // An implicit gap (no explicit slot) must not absorb a
                    // content word. ADJ is allowed: it legitimately
                    // pre-modifies the MWE's own noun ("on the concrete
                    // floor").
                    if gaps[k] <= GAP_MAX
                        && (p + 1..q).any(|t| {
                            matches!(
                                upos.get(t).map(String::as_str),
                                Some("NOUN" | "PROPN" | "NUM" | "VERB")
                            )
                        })
                    {
                        continue;
                    }
                    // A comma between only two fixed words signals they are not
                    // really together ("fair, playing"). Longer idioms may
                    // legitimately contain one ("so far, so good").
                    if gaps[k] <= GAP_MAX
                        && fixed.len() < 3
                        && (p + 1..q).any(|t| {
                            is_punct(t) && surface.get(t).map(String::as_str) == Some(",")
                        })
                    {
                        continue;
                    }
                    // For slot gaps, validate UPOS of the fill's first token.
                    if gaps[k] > GAP_MAX
                        && !slot_fill_ok(&slot_types[k], p + 1, q, upos)
                    {
                        continue;
                    }
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

    fn w(s: &str) -> Element { Element::Word(s.to_string()) }
    fn slot() -> Element { Element::Slot(SlotType::Any) }
    fn slot_pron() -> Element { Element::Slot(SlotType::Pron) }

    fn lems(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    fn upos(s: &str) -> Vec<String> {
        s.split_whitespace().map(|t| t.to_string()).collect()
    }

    fn no_upos(n: usize) -> Vec<String> { vec!["X".into(); n] }

    #[test]
    fn matches_contiguous() {
        let elems = [w("spill"), w("the"), w("bean")];
        let sent = lems("he spill the bean today");
        let idx = match_entry(&sent, &no_upos(sent.len()), &sent, &elems).unwrap();
        assert_eq!(idx, vec![1, 2, 3]);
    }

    #[test]
    fn matches_across_slot() {
        // Slot(Any): accepts any fill regardless of UPOS.
        let elems = [w("make"), slot(), w("up"), w("mind")];
        let sent = lems("they make her up mind now");
        let u = upos("X VERB PRON X NOUN X");
        let idx = match_entry(&sent, &u, &sent, &elems).unwrap();
        assert_eq!(idx, vec![1, 3, 4]);
    }

    #[test]
    fn no_match_when_gap_too_wide() {
        let elems = [w("kick"), w("bucket")];
        let sent = lems("kick a big old bucket");
        assert!(match_entry(&sent, &no_upos(sent.len()), &sent, &elems).is_none());
    }

    #[test]
    fn slot_allows_wider_gap() {
        let elems = [w("kick"), slot(), w("bucket")];
        let sent = lems("kick a big old bucket");
        let u = upos("VERB DET ADJ ADJ NOUN");
        let idx = match_entry(&sent, &u, &sent, &elems).unwrap();
        assert_eq!(idx, vec![0, 4]);
    }

    #[test]
    fn prefix_tolerant_eq() {
        let elems = [w("over"), w("moon")];
        let sent = lems("over moons");
        assert!(match_entry(&sent, &no_upos(sent.len()), &sent, &elems).is_some());
        assert!(eq("moon", "moons"));
        assert!(!eq("go", "gone"));
        assert!(!eq("the", "then"));
        assert!(!eq("but", "butter"));
    }

    #[test]
    fn eq_rejects_noninflectional_residual() {
        // "thin" is a 4-char prefix of "thing" but the residual "g" is not
        // an inflectional ending.
        assert!(!eq("thin", "thing"));
        assert!(eq("alway", "always"));
        assert!(!eq("wall", "wallaby"));
    }

    #[test]
    fn no_match_implicit_gap_across_period() {
        // A plain gap must not cross a sentence-final period.
        let elems = [w("fair"), w("play")];
        let sent = lems("fair . play");
        let u = upos("ADJ PUNCT VERB");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn implicit_gap_allows_comma_for_long_idiom() {
        // "so far, so good" (4 fixed words) — the interior comma is part of it.
        let elems = [w("so"), w("far"), w("so"), w("good")];
        let sent = lems("so far , so good");
        let u = upos("ADV ADV PUNCT ADV ADJ");
        assert!(match_entry(&sent, &u, &sent, &elems).is_some());
    }

    #[test]
    fn implicit_gap_rejects_comma_for_two_word_entry() {
        // "fair play" (2 fixed words) — a comma between them means they are
        // not actually the collocation.
        let elems = [w("fair"), w("play")];
        let sent = lems("fair , play");
        let u = upos("ADJ PUNCT VERB");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn matches_across_hyphen_gap() {
        // A hyphen is PUNCT but not a clause boundary — still crossable.
        let elems = [w("over"), w("board")];
        let sent = lems("over - board");
        let u = upos("ADP PUNCT NOUN");
        assert!(match_entry(&sent, &u, &sent, &elems).is_some());
    }

    #[test]
    fn slot_may_span_comma() {
        // "so far, so good" — the lexicon slot deliberately spans the comma.
        let elems = [slot(), w("far"), slot(), w("good")];
        let sent = lems("so far , so good");
        let u = upos("ADV ADV PUNCT ADV ADJ");
        assert!(match_entry(&sent, &u, &sent, &elems).is_some());
    }

    #[test]
    fn no_match_across_hard_boundary() {
        // A slot must NOT span a sentence-final period.
        let elems = [slot(), w("far"), slot(), w("good")];
        let sent = lems("so far . so good");
        let u = upos("ADV ADV PUNCT ADV ADJ");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn implicit_gap_rejects_content_word() {
        // "about time" must not match across the numeral "four".
        let elems = [w("about"), w("time")];
        let sent = lems("about four time");
        let u = upos("ADP NUM NOUN");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn implicit_gap_allows_pron_and_adv() {
        // Separable phrasal verbs: pronoun object / adverb modifier in the gap.
        let elems = [w("wake"), w("up")];
        let sent = lems("wake you up");
        assert!(match_entry(&sent, &upos("VERB PRON ADP"), &sent, &elems).is_some());
        let sent2 = lems("look quickly around");
        let elems2 = [w("look"), w("around")];
        assert!(match_entry(&sent2, &upos("VERB ADV ADV"), &sent2, &elems2).is_some());
    }

    #[test]
    fn pron_slot_accepts_pron() {
        // "pull herself together" — slot fills with PRON → match
        let elems = [w("pull"), slot_pron(), w("together")];
        let sent = lems("pull herself together");
        let u = upos("VERB PRON ADV");
        assert!(match_entry(&sent, &u, &sent, &elems).is_some());
    }

    #[test]
    fn pron_slot_rejects_det() {
        // "pull a team together" — slot first token is DET → no match
        let elems = [w("pull"), slot_pron(), w("together")];
        let sent = lems("pull a team together");
        let u = upos("VERB DET NOUN ADV");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn pron_slot_rejects_adj() {
        // "pull young volunteers together" — slot first token is ADJ → no match
        let elems = [w("pull"), slot_pron(), w("together")];
        let sent = lems("pull young volunteer together");
        let u = upos("VERB ADJ NOUN ADV");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }

    #[test]
    fn pron_slot_rejects_empty_fill() {
        // "pull together" — no tokens in slot → no match for Pron
        let elems = [w("pull"), slot_pron(), w("together")];
        let sent = lems("pull together");
        let u = upos("VERB ADV");
        assert!(match_entry(&sent, &u, &sent, &elems).is_none());
    }
}
