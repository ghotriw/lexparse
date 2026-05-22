//! Dictionary builder: scans a Wiktionary JSONL dump for multi-word entries in
//! the target categories and emits a slot/gap lexicon (`dic/lexicon.jsonl`).
//!
//! No model is needed — each MWE pattern is derived directly from the headword
//! string: placeholder words ("someone", "one's", …) become `Slot`, every other
//! token becomes a fixed `Word(lemma)`.

use anyhow::Result;
use lexparse::matcher::{self, Element, SlotType};
use lexparse::mwe::LexiconEntry;
use lexparse::{normalize, LEXICON_PATH};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use tracing::info;

#[derive(Deserialize, Debug)]
struct BuilderConfig {
    input_file: String,
    #[serde(default)]
    lang_code: Option<String>,
    target_categories: std::collections::HashMap<String, String>,
    #[serde(default)]
    pos_filter: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    phrase_blocklist: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct WikiSense {
    categories: Option<Vec<String>>,
    tags: Option<Vec<String>>,
    glosses: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
struct WikiEntry {
    word: Option<String>,
    pos: Option<String>,
    lang_code: Option<String>,
    categories: Option<Vec<String>>,
    senses: Option<Vec<WikiSense>>,
}

fn hash_string(s: &str) -> u32 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish() as u32
}

/// True for the possessive clitic tokens that `normalize::tokenize` splits off
/// (e.g. "one's" -> "one" + "'s"). They carry no content and must be dropped.
fn is_clitic(tok: &str) -> bool {
    matches!(tok, "'s" | "’s" | "'" | "’")
}

/// Connector words that may sit between two placeholder slots in a headword
/// ("someone or something"); collapsed away so the run becomes a single slot.
fn is_connector(tok: &str) -> bool {
    matches!(tok.to_lowercase().as_str(), "or" | "and" | "nor")
}

/// Articles, prepositions, and conjunctions that carry no content on their own.
/// Verbs (do, have, be) are intentionally excluded — they anchor phrasal verbs.
fn is_closed_class(tok: &str) -> bool {
    matches!(
        tok.to_lowercase().as_str(),
        "a" | "an" | "the"
            | "of" | "in" | "on" | "at" | "to" | "for" | "with" | "by" | "from"
            | "up" | "out" | "off" | "over" | "under" | "into" | "onto" | "upon"
            | "about" | "as" | "per" | "via" | "vs"
            | "and" | "or" | "but" | "nor" | "so" | "yet" | "if" | "than"
            | "not" | "no" | "its" | "it" | "be"
    )
}

/// Returns true if the element list has at least one fixed word that is not
/// a closed-class function word (article, preposition, conjunction).
/// Prevents patterns like "of a" or "in the" from flooding match results.
fn has_content_word(elements: &[Element]) -> bool {
    elements.iter().any(|e| match e {
        Element::Word(w) => !is_closed_class(w),
        Element::Slot(_) => false,
    })
}

/// Classify a slot placeholder token into a `SlotType`.
fn slot_type_for(tok: &str) -> SlotType {
    match tok.trim().to_lowercase().as_str() {
        // Possessive/reflexive/personal pronoun placeholders.
        "one's" | "ones" | "one" | "oneself"
        | "someone's" | "somebody's" | "sb's" | "poss"
        | "your" | "yours" | "yourself" | "pron"
        | "someone" | "somebody" | "sb" => SlotType::Pron,
        // Nominal (thing/place) placeholders.
        "something" | "sth" | "sth's" | "sw" => SlotType::Noun,
        // Generic wildcards.
        _ => SlotType::Any,
    }
}

/// Convert a headword string into a slot/gap element pattern.
fn phrase_to_elements(phrase: &str) -> Vec<Element> {
    let mut raw: Vec<Element> = Vec::new();
    for tok in normalize::tokenize(phrase) {
        if is_clitic(&tok) {
            continue;
        }
        if normalize::is_slot_token(&tok) {
            raw.push(Element::Slot(slot_type_for(&tok)));
        } else {
            raw.push(Element::Word(normalize::lemma(&tok)));
        }
    }

    // Drop connector words trapped between two slots, then collapse runs of
    // adjacent slots into one (downgrade to Any when types differ).
    let mut out: Vec<Element> = Vec::new();
    for (i, el) in raw.iter().enumerate() {
        if let Element::Word(w) = el {
            let prev_slot = matches!(raw.get(i.wrapping_sub(1)), Some(Element::Slot(_)));
            let next_slot = matches!(raw.get(i + 1), Some(Element::Slot(_)));
            if prev_slot && next_slot && is_connector(w) {
                continue;
            }
        }
        if matches!(el, Element::Slot(_)) {
            if let Some(Element::Slot(existing)) = out.last_mut() {
                // Two consecutive slots: downgrade to unconstrained.
                *existing = SlotType::Any;
                continue;
            }
        }
        out.push(el.clone());
    }
    out
}

fn main() -> Result<()> {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();
    info!("Starting dictionary builder...");

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "builder_config.toml".to_string());
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config from {}: {}", config_path, e))?;
    let config: BuilderConfig = toml::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse TOML from {}: {}", config_path, e))?;
    info!("Loaded config from {}", config_path);

    let file = File::open(&config.input_file)?;
    let reader = BufReader::new(file);

    let mut out_file = File::create(LEXICON_PATH)?;
    let mut processed_words = HashSet::new();
    let mut line_count = 0usize;
    let mut written = 0usize;
    let mut skipped_too_short = 0usize;

    info!("Scanning Wiktionary dump...");
    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        if line_count % 50_000 == 0 {
            info!("Scanned {} lines, wrote {} entries...", line_count, written);
        }

        let Ok(entry) = serde_json::from_str::<WikiEntry>(&line) else {
            continue;
        };
        if let Some(ref required) = config.lang_code {
            if entry.lang_code.as_deref() != Some(required.as_str()) {
                continue;
            }
        }
        let Some(word) = entry.word else { continue };
        if word.trim().split_whitespace().count() < 2 {
            continue;
        }

        let mut matched_types = HashSet::new();
        let mut tags = HashSet::new();
        let mut glosses = Vec::new();

        let mut consider = |cats: &[String], sense: Option<&WikiSense>| {
            for cat in cats {
                for (target, mapped) in &config.target_categories {
                    if cat == target || cat.starts_with(&format!("{} ", target)) {
                        matched_types.insert(mapped.clone());
                        if let Some(s) = sense {
                            if let Some(t) = &s.tags {
                                tags.extend(t.iter().cloned());
                            }
                            if let Some(g) = &s.glosses {
                                glosses.extend(g.iter().cloned());
                            }
                        }
                    }
                }
            }
        };

        if let Some(top_cats) = &entry.categories {
            consider(top_cats, None);
        }
        if let Some(senses) = &entry.senses {
            for sense in senses {
                if let Some(cats) = &sense.categories {
                    consider(cats, Some(sense));
                }
            }
        }

        // Drop types whose POS filter doesn't match this entry's POS.
        if !config.pos_filter.is_empty() {
            let entry_pos = entry.pos.as_deref().unwrap_or("");
            matched_types.retain(|t| {
                config.pos_filter.get(t).map_or(true, |allowed| {
                    allowed.iter().any(|p| p == entry_pos)
                })
            });
        }

        if matched_types.is_empty() || processed_words.contains(&word) {
            continue;
        }
        if config.phrase_blocklist.contains(&word) {
            continue;
        }
        processed_words.insert(word.clone());

        let elements = phrase_to_elements(&word);
        if matcher::fixed_count(&elements) < 2 {
            skipped_too_short += 1;
            continue;
        }
        if !has_content_word(&elements) {
            skipped_too_short += 1;
            continue;
        }

        let mut categories: Vec<_> = matched_types.into_iter().collect();
        categories.sort_unstable();
        let mut tags: Vec<_> = tags.into_iter().collect();
        tags.sort_unstable();
        let record = LexiconEntry {
            id: hash_string(&word),
            phrase: word,
            categories,
            pos: entry.pos,
            definition: glosses.into_iter().next(),
            tags,
            elements,
        };
        writeln!(out_file, "{}", serde_json::to_string(&record)?)?;
        written += 1;
    }

    info!(
        "Done. Scanned {} lines, wrote {} entries to {} ({} skipped: < 2 fixed words).",
        line_count, written, LEXICON_PATH, skipped_too_short
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(p: &str) -> Vec<Element> {
        phrase_to_elements(p)
    }

    #[test]
    fn plain_phrase() {
        assert_eq!(
            pat("spill the beans"),
            vec![
                Element::Word("spill".into()),
                Element::Word("the".into()),
                Element::Word("bean".into()),
            ]
        );
    }

    #[test]
    fn possessive_placeholder_becomes_slot() {
        // "lose one's mind" -> lose <slot:pron> mind  ("'s" clitic dropped)
        assert_eq!(
            pat("lose one's mind"),
            vec![
                Element::Word("lose".into()),
                Element::Slot(SlotType::Pron),
                Element::Word("mind".into()),
            ]
        );
    }

    #[test]
    fn connector_between_slots_collapsed() {
        // "someone or something": Pron + Noun → collapses to Slot(Any)
        let p = pat("give someone or something away");
        assert_eq!(
            p,
            vec![
                Element::Word("give".into()),
                Element::Slot(SlotType::Any),
                Element::Word("away".into()),
            ]
        );
    }
}
