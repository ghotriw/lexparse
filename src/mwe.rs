//! MWE detection: lexicon candidate generation
//!
//! Pipeline per sentence:
//!   1. `matcher::scan` over the lexicon -> content-word span candidates
//!      (token indices in the `normalize::tokenize` word space).
//!   2. MWE candidates are filtered by overlap and tree connectivity.

use crate::matcher::{self, Element, Entry};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct MweSource {
    phrase: String,
    runs: Option<Vec<String>>,
    alt: Option<Vec<String>>,
    /// MWE category: idiom, phrasal_verb, collocation_phrase, proverb_saying.
    #[serde(rename = "type", default)]
    kind: String,
}

pub struct MweLexicon {
    entries: Vec<Entry>,
}

impl MweLexicon {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let source: Vec<MweSource> = serde_json::from_str(&raw)?;
        let mut entries = Vec::new();
        
        for mwe in source {
            let runs = mwe.runs.unwrap_or_default();
            let alt = mwe.alt.unwrap_or_default();
            if runs.is_empty() || runs.len() != alt.len() {
                continue;
            }
            
            let expanded = expand_runs(&runs, &alt);
            for elements in expanded {
                let fixed_count = elements.iter().filter(|e| matches!(e, Element::Word(_))).count();
                // If the original MWE phrase has multiple words (indicated by a space),
                // we require at least 2 fixed words in the expansion to avoid matching single common words.
                let is_mwe_multi = mwe.phrase.contains(' ');
                if is_mwe_multi && fixed_count < 2 {
                    continue;
                }
                
                let has_word = fixed_count > 0;
                if !has_word {
                    continue;
                }
                
                entries.push(Entry {
                    elements,
                    fixed_key: vec![],
                    surface: mwe.phrase.clone(),
                    has_slot: alt.iter().any(|a| a.contains("variable")),
                    category: mwe.kind.clone(),
                });
            }
        }
        
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }
}

fn expand_runs(runs: &[String], alt: &[String]) -> Vec<Vec<Element>> {
    let mut results: Vec<Vec<Element>> = vec![vec![]];
    for (run, a) in runs.iter().zip(alt.iter()) {
        let mut next_results = Vec::new();
        let is_opt = a.starts_with("o-");
        let is_var = a.ends_with("variable");
        
        let mut elems = Vec::new();
        if is_var {
            elems.push(Element::Slot);
        } else {
            for word in run.split_whitespace() {
                elems.push(Element::Word(crate::normalize::lemma(word)));
            }
        }
        
        for res in results {
            if is_opt {
                next_results.push(res.clone());
            }
            let mut res_copy = res;
            res_copy.extend(elems.clone());
            next_results.push(res_copy);
        }
        results = next_results;
    }
    results
}

#[derive(Debug, Serialize)]
pub struct MweMatch {
    pub surface: String,
    /// MWE category from the lexicon: idiom, phrasal_verb,
    /// collocation_phrase, proverb_saying, etc.
    pub category: String,
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
    lexicon: &MweLexicon,
) -> Vec<MweMatch> {
    let lem = matcher::lemmatize_pos(words, is_verb);
    let hits = matcher::scan(lexicon.entries(), &lem);

    struct Cand {
        m: MweMatch,
        lo: usize,
        hi: usize,
    }
    let mut cands: Vec<Cand> = Vec::with_capacity(hits.len());
    for (entry_idx, span) in hits {
        let entry = &lexicon.entries()[entry_idx];
        let lo = *span.iter().min().unwrap();
        let hi = *span.iter().max().unwrap();
        let discontinuous = hi - lo + 1 != span.len();
        let tree_connected = if discontinuous {
            span_tree_connected(&span, heads)
        } else {
            true
        };
        let span_text = (lo..=hi)
            .map(|i| words[i].clone())
            .collect::<Vec<_>>()
            .join(" ");

        cands.push(Cand {
            m: MweMatch {
                surface: entry.surface.clone(),
                category: entry.category.clone(),
                has_slot: entry.has_slot,
                token_ids: span.iter().map(|&i| i + 1).collect(),
                words: span.iter().map(|&i| words[i].clone()).collect(),
                span_text,
                prob: 1.0,
                discontinuous,
                tree_connected,
            },
            lo,
            hi,
        });
    }

    cands.retain(|c| !c.m.discontinuous || c.m.tree_connected);

    let spans: Vec<(usize, usize, f32)> = cands
        .iter()
        .map(|c| (c.lo, c.hi, c.m.prob))
        .collect();
    let keep = keep_mask(&spans);
    cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c.m))
        .collect()
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

fn span_tree_connected(span: &[usize], heads: &[usize]) -> bool {
    let id = |w: usize| w + 1;
    let lo = *span.iter().min().unwrap() + 1;
    let hi = *span.iter().max().unwrap() + 1;
    let span_ids: std::collections::HashSet<usize> = span.iter().map(|&w| id(w)).collect();

    let ancestors = |mut v: usize| -> Vec<usize> {
        let mut chain = vec![v];
        let mut steps = 0;
        while v != 0 {
            let p = heads.get(v).copied().unwrap_or(0);
            if p == v || steps > heads.len() {
                break;
            }
            chain.push(p);
            v = p;
            steps += 1;
        }
        chain
    };

    let mut sorted: Vec<usize> = span.iter().map(|&w| id(w)).collect();
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let ca = ancestors(a);
        let cb: std::collections::HashSet<usize> = ancestors(b).into_iter().collect();
        let lca = match ca.iter().find(|x| cb.contains(x)) {
            Some(&x) => x,
            None => return false,
        };
        let on_path = ancestors(a)
            .into_iter()
            .take_while(|&x| x != lca)
            .chain(ancestors(b).into_iter().take_while(|&x| x != lca))
            .chain(std::iter::once(lca));
        for node in on_path {
            if node == 0 {
                return false;
            }
            if span_ids.contains(&node) {
                continue;
            }
            if node < lo || node > hi {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::keep_mask;

    #[test]
    fn wider_mwe_span_subsumes_narrower() {
        let spans = [(3, 8, 0.586_f32), (4, 8, 0.808_f32)];
        assert_eq!(keep_mask(&spans), vec![true, false]);
    }

    #[test]
    fn identical_range_keeps_higher_prob_then_earlier() {
        assert_eq!(
            keep_mask(&[(2, 5, 0.4), (2, 5, 0.9)]),
            vec![false, true]
        );
        assert_eq!(
            keep_mask(&[(2, 5, 0.7), (2, 5, 0.7)]),
            vec![true, false]
        );
    }

    #[test]
    fn partial_overlap_both_survive() {
        assert_eq!(
            keep_mask(&[(1, 4, 0.7), (3, 7, 0.7)]),
            vec![true, true]
        );
    }
}
