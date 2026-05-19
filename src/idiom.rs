//! Idiom detection: lexicon candidate generation + linear classifier on the
//! pooled `word_repr` (RUST_INTEGRATION.md §6.4 / §10.3).
//!
//! Pipeline per sentence:
//!   1. `matcher::scan` over the lexicon -> content-word span candidates
//!      (token indices in the `normalize::tokenize` word space).
//!   2. Mean-pool `word_repr` over the span words (word index `i` -> grid /
//!      output row `i + 1`; row 0 is ROOT).
//!   3. `prob = sigmoid(W . pooled + b)`; idiomatic iff `prob >= threshold`.
//!      Standardization is already baked into `W, b` — no extra normalization.

use crate::matcher::{self, Entry};
use ndarray::ArrayView3;
use serde::Deserialize;
use serde::Serialize;

/// Stage-2 linear idiom head (`model/idiom_classifier.json`).
#[derive(Debug, Deserialize)]
pub struct IdiomClassifier {
    #[serde(rename = "W")]
    pub w: Vec<f32>,
    pub b: f32,
    pub threshold_prob: f32,
}

#[derive(Debug, Serialize)]
pub struct IdiomMatch {
    /// Lexicon surface form of the matched idiom.
    pub surface: String,
    /// Whether the lexicon entry declares a free slot.
    pub has_slot: bool,
    /// 1-based word ids of the fixed content words (supar word indices;
    /// these are the `tokens[].id`s the span occupies).
    pub token_ids: Vec<usize>,
    /// Surface words at the fixed-content positions only.
    pub words: Vec<String>,
    /// Human-readable span: the contiguous sentence text from the first to
    /// the last span word *including* in-between slot/gap fillers — e.g.
    /// "costs an arm and a leg" rather than the lexicon template
    /// "cost [pron] arm and [pron] leg". This is what a consumer should show.
    pub span_text: String,
    /// `sigmoid(W . mean(word_repr[span]) + b)`.
    pub prob: f32,
    /// `prob >= threshold_prob` — figurative/idiomatic vs. literal.
    pub idiomatic: bool,
    /// Span words are not contiguous in the sentence.
    pub discontinuous: bool,
    /// Heuristic tree filter for discontinuous spans (§6.4): the span words
    /// plus only in-between filler form a connected dependency subtree. Always
    /// true for contiguous spans. NOTE: a pragmatic stand-in until
    /// IDIOM-DETECTION-STRATEGY_V2 §7's slot-template filter is available.
    pub tree_connected: bool,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Run lexicon scan + classifier for one sentence.
///
/// * `words`     — `normalize::tokenize` output (sentence word surfaces).
/// * `word_repr` — ONNX `word_repr` view `[1, W, D]`, `W == words.len() + 1`.
/// * `heads`     — decoded dependency parents, indexed by word id (1-based);
///   `heads[0]` is the ROOT sentinel.
pub fn detect(
    words: &[String],
    word_repr: &ArrayView3<f32>,
    heads: &[usize],
    lexicon: &[Entry],
    clf: &IdiomClassifier,
) -> anyhow::Result<Vec<IdiomMatch>> {
    let dim = word_repr.shape()[2];
    if clf.w.len() != dim {
        anyhow::bail!(
            "idiom classifier in_dim {} != word_repr dim {} (artifact mismatch)",
            clf.w.len(),
            dim
        );
    }

    let lem = matcher::lemmatize(words);
    let hits = matcher::scan(lexicon, &lem);

    struct Cand {
        m: IdiomMatch,
        lo: usize,
        hi: usize,
    }
    let mut cands: Vec<Cand> = Vec::with_capacity(hits.len());
    for (entry_idx, span) in hits {
        let entry = &lexicon[entry_idx];

        // Mean-pool word_repr over the fixed-content span words (+1 for ROOT).
        let mut pooled = vec![0.0f32; dim];
        for &wi in &span {
            let row = wi + 1;
            for (d, p) in pooled.iter_mut().enumerate() {
                *p += word_repr[[0, row, d]];
            }
        }
        let inv = 1.0 / span.len() as f32;
        let dot: f32 = clf
            .w
            .iter()
            .zip(&pooled)
            .map(|(w, p)| w * p * inv)
            .sum();
        let prob = sigmoid(clf.b + dot);

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
            m: IdiomMatch {
                surface: entry.surface.clone(),
                has_slot: entry.has_slot,
                token_ids: span.iter().map(|&i| i + 1).collect(),
                words: span.iter().map(|&i| words[i].clone()).collect(),
                span_text,
                prob,
                idiomatic: prob >= clf.threshold_prob,
                discontinuous,
                tree_connected,
            },
            lo,
            hi,
        });
    }

    // Overlap resolution. The lexicon records one idiom in several surface
    // variants ("an arm and a leg" ⊂ "cost [pron] arm and [pron] leg"), so a
    // recall-oriented scan double-hits the same expression. Drop a candidate
    // whose [lo,hi] word range is fully contained in another *idiomatic*
    // candidate's range: the wider form is the more complete idiom (keeps the
    // governing verb), so the consumer gets "cost an arm and a leg" once
    // rather than that plus the bare "an arm and a leg". Strictly wider wins
    // regardless of prob; identical range → higher prob, ties → the
    // earlier/longer-span hit (scan is sorted longest-first). A non-idiomatic
    // (literal) span may NOT suppress an idiomatic one. Genuinely distinct
    // idioms only partially overlap (no containment) → both survive.
    let spans: Vec<(usize, usize, bool, f32)> = cands
        .iter()
        .map(|c| (c.lo, c.hi, c.m.idiomatic, c.m.prob))
        .collect();
    let keep = keep_mask(&spans);
    let out: Vec<IdiomMatch> = cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c.m))
        .collect();
    Ok(out)
}

/// Containment-based overlap resolution (pure, so it is unit-tested without
/// the model). `spans[i] = (lo, hi, idiomatic, prob)` in sentence-word index
/// space. Returns a keep-mask: candidate `i` is dropped iff some *idiomatic*
/// candidate `j` fully contains its `[lo,hi]` range and either is strictly
/// wider (wider form wins regardless of prob — keeps the governing verb), or
/// has the same range with higher prob (ties → the earlier hit; `scan` is
/// longest-first). A non-idiomatic span never suppresses an idiomatic one;
/// partially-overlapping distinct idioms (no containment) all survive.
fn keep_mask(spans: &[(usize, usize, bool, f32)]) -> Vec<bool> {
    let n = spans.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        let (ilo, ihi, _, iprob) = spans[i];
        for (j, &(jlo, jhi, jidiom, jprob)) in spans.iter().enumerate() {
            if i == j || !jidiom {
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

/// Heuristic §6.4 tree filter: true iff every dependency-tree node on the path
/// between consecutive span words is itself either a span word or a "gap
/// filler" lying within the lexical span range `[min, max]`. This rejects
/// matches whose words are only spuriously co-located (connected through
/// far-away material) while keeping genuine discontinuous idioms (slot/gap
/// fillers sit between the content words).
fn span_tree_connected(span: &[usize], heads: &[usize]) -> bool {
    // word id = token index + 1; heads is indexed by word id, 0 == ROOT.
    let id = |w: usize| w + 1;
    let lo = *span.iter().min().unwrap() + 1;
    let hi = *span.iter().max().unwrap() + 1;
    let span_ids: std::collections::HashSet<usize> = span.iter().map(|&w| id(w)).collect();

    let ancestors = |mut v: usize| -> Vec<usize> {
        let mut chain = vec![v];
        // follow parents until ROOT (0); guard against malformed cycles
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
        // lowest common ancestor = first node of a's chain also in b's chain
        let lca = match ca.iter().find(|x| cb.contains(x)) {
            Some(&x) => x,
            None => return false,
        };
        // every intermediate node must be a span word or an in-range filler
        let on_path = ancestors(a)
            .into_iter()
            .take_while(|&x| x != lca)
            .chain(ancestors(b).into_iter().take_while(|&x| x != lca))
            .chain(std::iter::once(lca));
        for node in on_path {
            if node == 0 {
                return false; // path escaped through ROOT
            }
            if span_ids.contains(&node) {
                continue;
            }
            if node < lo || node > hi {
                return false; // connected only via far-away material
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::keep_mask;

    #[test]
    fn wider_idiomatic_span_subsumes_narrower() {
        // "That new iPhone costs an arm and a leg." — words 0-based:
        //   3 costs, 4 an, 5 arm, 6 and, 7 a, 8 leg
        // VP entry "cost [pron] arm and [pron] leg": fixed = costs,arm,and,leg
        //   → [lo,hi] = [3,8], idiomatic, p=0.586
        // NP entry "an arm and a leg":              fixed = an,arm,and,a,leg
        //   → [lo,hi] = [4,8], idiomatic, p=0.808 (contained in [3,8])
        // The wider VP must win even though its prob is lower.
        let spans = [(3, 8, true, 0.586_f32), (4, 8, true, 0.808_f32)];
        assert_eq!(keep_mask(&spans), vec![true, false]);
    }

    #[test]
    fn identical_range_keeps_higher_prob_then_earlier() {
        // same span, two entries → keep the higher-prob one
        assert_eq!(
            keep_mask(&[(2, 5, true, 0.4), (2, 5, true, 0.9)]),
            vec![false, true]
        );
        // tie on prob → keep the earlier (scan is longest-first / stable)
        assert_eq!(
            keep_mask(&[(2, 5, true, 0.7), (2, 5, true, 0.7)]),
            vec![true, false]
        );
    }

    #[test]
    fn partial_overlap_both_survive() {
        // distinct idioms sharing one word but neither contains the other
        assert_eq!(
            keep_mask(&[(1, 4, true, 0.7), (3, 7, true, 0.7)]),
            vec![true, true]
        );
    }

    #[test]
    fn literal_span_cannot_suppress_idiom() {
        // a wider *non-idiomatic* (literal) span must not drop a contained
        // idiomatic one
        assert_eq!(
            keep_mask(&[(0, 9, false, 0.9), (4, 8, true, 0.8)]),
            vec![true, true]
        );
    }
}
