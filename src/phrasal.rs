//! Approach-3 phrasal-verb detection: a label-agnostic second pass layered on
//! top of the untouched `compound:prt` rule (RUST_INTEGRATION.md §6.3).
//!
//! The trained parser's particle→verb ARC is the stable signal; the deprel
//! label flips between `compound:prt` and `advmod` with the verb's surface
//! form (UD-EWT itself annotates "come up" as `advmod`), so a candidate is
//! confirmed against a closed (verb-lemma, particle) inventory instead of
//! trusting the label. Lemma match (via `normalize`) makes it tense-invariant
//! (came→come); the closed list keeps adverbial false positives
//! ("look up at the sky") out — precision is controlled by the data file plus
//! the arc, not by guessing a particle set.
//!
//! Three-component verbs ("come up **with**", "put up **with**") are kept
//! distinct from their bare cores ("come up", "put up"): the extending
//! preposition is stored per (verb, particle) pair so the caller can prefer
//! the longest reading when that preposition is structurally present.

use crate::normalize::{lemma, lemma_pos};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// Accepted on-disk forms of the phrasal inventory. The compact deploy form is
/// a bare array of surface phrases (`["come up with", ...]`); the legacy form
/// is the original object whose keys are the phrases (values ignored). Both
/// reduce to the same phrase list — `untagged` so either parses transparently.
#[derive(Deserialize)]
#[serde(untagged)]
enum PhrasalSource {
    List(Vec<String>),
    Map(HashMap<String, serde::de::IgnoredAny>),
}

/// (verb-lemma, particle-lemma) -> set of preposition-lemmas that *extend*
/// that pair into a 3-component phrasal-prepositional verb. The key existing
/// at all means a verb+particle phrasal is known (this covers the bare
/// "put up" even when only "put up with" was in the source). An empty set
/// means only bare entries were seen for that pair.
pub struct PhrasalLexicon {
    by_pair: HashMap<(String, String), HashSet<String>>,
}

impl PhrasalLexicon {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let phrases: Vec<String> = match serde_json::from_str::<PhrasalSource>(&raw)? {
            PhrasalSource::List(v) => v,
            PhrasalSource::Map(m) => m.into_keys().collect(),
        };
        let mut by_pair: HashMap<(String, String), HashSet<String>> = HashMap::new();
        for key in &phrases {
            if let Some((v, p, prep)) = parse_key(key) {
                let set = by_pair.entry((v, p)).or_default();
                if let Some(prep) = prep {
                    set.insert(prep);
                }
            }
        }
        Ok(Self { by_pair })
    }

    /// Distinct (verb, particle) pairs — for the startup log only.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.by_pair.len()
    }

    /// If (verb_word, particle_word) — lemmatized — is a known phrasal verb,
    /// returns the set of prepositions that extend it into a 3-component verb
    /// ("come up" -> `["with"]`); the bare-only case returns `Some([])`.
    /// `None` means the pair is not in the inventory at all.
    ///
    /// Verb match is prefix-tolerant (absorbs the residual inflection `lemma`
    /// leaves, e.g. gerund "coming"→"com" vs key "come"); particle/preposition
    /// match is exact (function words are never inflected, and the >=3-char
    /// guard in `eq_tol` would otherwise conflate "of"/"off", "in"/"into").
    pub fn resolve(&self, verb_word: &str, particle_word: &str) -> Option<Vec<String>> {
        let pl = lemma(particle_word);
        if pl.is_empty() {
            return None;
        }
        // slot0 is structurally the governing verb (UPOS-guarded in main) →
        // verb-gated lemma, symmetric with `parse_key`'s verb slot.
        let vl = lemma_pos(verb_word, true);
        if vl.is_empty() {
            return None;
        }
        if let Some(set) = self.by_pair.get(&(vl.clone(), pl.clone())) {
            return Some(set.iter().cloned().collect());
        }
        // Residual-inflection fallback (gerund stems etc.); the candidate
        // edges per sentence are a handful, so the scan is negligible.
        self.by_pair
            .iter()
            .find(|((kv, kp), _)| kp == &pl && kv != &vl && eq_tol(&vl, kv))
            .map(|(_, set)| set.iter().cloned().collect())
    }
}

/// "come up with"/"abide by"/"knock it off!"/"pump (money) into" ->
/// (verb, particle, Option<prep>) lemmas. Strips a trailing "!", drops
/// "(...)" variant/slot groups, then token0 = verb, token1 = particle, and
/// for an exactly-3-token key token2 = the extending preposition. 2-token
/// keys are bare (no prep); >=4-token keys keep only the verb+particle core
/// (their token2 is rarely a clean preposition). Non-particle token1
/// ("knock **it** off") is harmless: the runtime UPOS guard rejects a
/// non-ADP/ADV/PART child, so the bogus pair never fires.
fn parse_key(key: &str) -> Option<(String, String, Option<String>)> {
    let mut s = key.to_string();
    if let Some(i) = s.find('!') {
        s.truncate(i);
    }
    // strip parenthetical groups: "(color)", "(money)", "(around)"
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    let toks: Vec<&str> = out.split_whitespace().collect();
    if toks.len() < 2 {
        return None;
    }
    let v = lemma_pos(toks[0], true); // slot0 = verb (symmetric with resolve)
    let p = lemma(toks[1]);
    if v.is_empty() || p.is_empty() {
        return None;
    }
    let prep = if toks.len() == 3 {
        let pr = lemma(toks[2]);
        if pr.is_empty() {
            None
        } else {
            Some(pr)
        }
    } else {
        None
    };
    Some((v, p, prep))
}

/// Prefix-tolerant lemma equality — same rule as `matcher::eq` (kept local;
/// the codebase intentionally re-states the anti-skew spec per module).
fn eq_tol(a: &str, b: &str) -> bool {
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
    short.chars().count() >= 3 && long.starts_with(short)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_key_cleans_and_keeps_prep() {
        assert_eq!(
            parse_key("come up with"),
            Some(("come".into(), "up".into(), Some("with".into())))
        );
        assert_eq!(
            parse_key("put up with"),
            Some(("put".into(), "up".into(), Some("with".into())))
        );
        assert_eq!(parse_key("come up"), Some(("come".into(), "up".into(), None)));
        assert_eq!(parse_key("look up"), Some(("look".into(), "up".into(), None)));
        assert_eq!(
            parse_key("pump (money) into"),
            Some(("pump".into(), "into".into(), None))
        );
        assert_eq!(
            parse_key("bog off!"),
            Some(("bog".into(), "off".into(), None))
        );
        assert_eq!(parse_key("up"), None);
    }

    fn lex_with(pairs: &[(&str, &str, Option<&str>)]) -> PhrasalLexicon {
        let mut by_pair: HashMap<(String, String), HashSet<String>> = HashMap::new();
        for (v, p, prep) in pairs {
            let set = by_pair.entry((v.to_string(), p.to_string())).or_default();
            if let Some(pr) = prep {
                set.insert(pr.to_string());
            }
        }
        PhrasalLexicon { by_pair }
    }

    #[test]
    fn resolve_is_tense_invariant_and_carries_prep() {
        let lex = lex_with(&[("come", "up", Some("with")), ("come", "up", None)]);
        // base / past (irregular) / 3sg / gerund all resolve to (come, up)
        assert_eq!(lex.resolve("come", "up"), Some(vec!["with".into()]));
        assert_eq!(lex.resolve("came", "up"), Some(vec!["with".into()])); // irregular
        assert_eq!(lex.resolve("comes", "up"), Some(vec!["with".into()])); // s-strip
        assert_eq!(lex.resolve("coming", "up"), Some(vec!["with".into()])); // gerund
        assert_eq!(lex.resolve("come", "down"), None); // unknown particle
        assert_eq!(lex.resolve("go", "up"), None); // unknown verb
    }

    #[test]
    fn resolve_bare_only_returns_empty_prep_set() {
        let lex = lex_with(&[("look", "up", None)]);
        assert_eq!(lex.resolve("look", "up"), Some(vec![]));
    }
}
