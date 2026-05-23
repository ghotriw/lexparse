//! Lexicon corrector: re-classifies entries currently tagged `idiom`
//! into either confirmed `idiom` (true figurative — has plausible literal
//! readings too) or `fixed_collocation` (discourse markers / formulas with
//! no real literal use, e.g. "by the way", "not to mention").
//!
//! Reads:   dic/lexicon.jsonl
//! Reads:   dic/corrections.jsonl  (cache — resumable; already-processed
//!          ids are skipped, so a growing lexicon only costs new queries)
//! Appends: dic/corrections.jsonl  (one minimal patch line per processed id,
//!          even when categories are unchanged — that's what marks it done)
//!
//! Each output line: `{"id": <u32>, "categories": [...]}`
//!
//! Requires GEMINI_API_KEY in env or in `.env` (loaded via dotenvy).
//!
//! Run from the `lexparse/` directory:
//!     cargo run --release --bin correct_idioms

use anyhow::{Context, Result};
use lexparse::mwe::LexiconEntry;
use lexparse::{CORRECTIONS_PATH, LEXICON_PATH};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tracing::{info, warn};

const DEFAULT_MODEL: &str = "gemini-3.1-flash-lite-preview";
const BATCH_SIZE: usize = 50;
const RETRY_DELAYS_MS: &[u64] = &[2_000, 4_000, 8_000, 16_000];

#[derive(Serialize)]
struct CorrectionLine {
    id: u32,
    categories: Vec<String>,
}

#[derive(Deserialize)]
struct CachedId {
    id: u32,
}

#[derive(Deserialize)]
struct GeminiOne {
    id: u32,
    kind: String,
}

#[derive(Deserialize)]
struct GeminiResp {
    results: Vec<GeminiOne>,
}

fn load_cache(path: &str) -> Result<HashSet<u32>> {
    let mut seen = HashSet::new();
    if !std::path::Path::new(path).exists() {
        return Ok(seen);
    }
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(c) = serde_json::from_str::<CachedId>(&line) {
            seen.insert(c.id);
        }
    }
    Ok(seen)
}

fn build_prompt(batch: &[&LexiconEntry]) -> String {
    let lines: Vec<String> = batch
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let def = e.definition.as_deref().unwrap_or("");
            format!("{}. id={}: \"{}\" — {}", i + 1, e.id, e.phrase, def)
        })
        .collect();

    format!(
        r#"You are a lexicographer classifying English multi-word expressions.

For each phrase below, decide whether it is:
- "idiom": a TRUE figurative idiom — its primary meaning is non-literal AND a literal reading is also plausible in some real English sentences. Examples: "kick the bucket", "spill the beans", "full of oneself", "break the ice", "let the cat out of the bag", "hit the books".
- "fixed": a fixed expression / discourse marker / formula — a multi-word unit that is essentially always used in one non-compositional sense, with no plausible literal use in modern English. Examples: "by the way", "as a matter of fact", "not to mention", "of course", "in fact", "after all", "for instance", "on the other hand".

The distinguishing test: could a competent English speaker plausibly use the phrase with its literal, word-by-word meaning in some real sentence? If yes — "idiom". If the phrase only ever functions as the fixed/idiomatic unit — "fixed".

Return ONLY valid JSON, no markdown or commentary:
{{"results": [{{"id": 123, "kind": "idiom"}}, {{"id": 456, "kind": "fixed"}}]}}

The "results" array MUST have exactly {} elements, one per phrase, each with the matching id.

Phrases:
{}
"#,
        batch.len(),
        lines.join("\n")
    )
}

fn call_gemini(prompt: &str, api_key: &str, model: &str) -> Result<Vec<GeminiOne>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let body = serde_json::json!({
        "contents": [{ "parts": [{ "text": prompt }] }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0
        }
    });

    let resp = ureq::post(&url)
        .set("content-type", "application/json")
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("gemini http: {}", e))?;

    let json: serde_json::Value = resp.into_json()?;
    let text = json
        .pointer("/candidates/0/content/parts/0/text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("no text in gemini response: {}", json))?;

    let parsed: GeminiResp = serde_json::from_str(text).with_context(|| {
        format!(
            "gemini returned invalid JSON: {}",
            text.chars().take(300).collect::<String>()
        )
    })?;

    Ok(parsed.results)
}

fn call_gemini_retry(prompt: &str, api_key: &str, model: &str) -> Result<Vec<GeminiOne>> {
    let mut attempt = 0usize;
    loop {
        match call_gemini(prompt, api_key, model) {
            Ok(r) => return Ok(r),
            Err(e) => {
                let msg = e.to_string();
                let retryable = msg.contains("429") || msg.contains("503") || msg.contains("500");
                if !retryable || attempt >= RETRY_DELAYS_MS.len() {
                    return Err(e);
                }
                warn!("gemini retry {} after error: {}", attempt + 1, msg);
                sleep(Duration::from_millis(RETRY_DELAYS_MS[attempt]));
                attempt += 1;
            }
        }
    }
}

/// Build the patched categories list: every "idiom" becomes `target`,
/// preserving order and deduplicating the result.
fn map_categories(original: &[String], target: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(original.len());
    for c in original {
        let mapped = if c == "idiom" {
            target.to_string()
        } else {
            c.clone()
        };
        if !out.contains(&mapped) {
            out.push(mapped);
        }
    }
    out
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    // Default to INFO so progress is visible without RUST_LOG; user can still
    // override via RUST_LOG (e.g. RUST_LOG=debug or RUST_LOG=warn).
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let api_key = std::env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY not set (put it in lexparse/.env)")?;
    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    info!("using gemini model: {}", model);

    // Requests-per-minute cap. 0 (default) = no limit; otherwise enforce a
    // minimum interval between consecutive Gemini calls.
    let rpm: u32 = std::env::var("GEMINI_RPM")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let min_interval = if rpm > 0 {
        Some(Duration::from_millis(60_000 / rpm as u64))
    } else {
        None
    };
    if let Some(d) = min_interval {
        info!("rate limit: {} req/min (min interval {:?})", rpm, d);
    }

    info!("loading {}", LEXICON_PATH);
    let mut idiom_entries: Vec<LexiconEntry> = Vec::new();
    for line in BufReader::new(File::open(LEXICON_PATH)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(e) = serde_json::from_str::<LexiconEntry>(&line) else {
            continue;
        };
        if e.categories.iter().any(|c| c == "idiom") {
            idiom_entries.push(e);
        }
    }
    info!("found {} entries with category=idiom", idiom_entries.len());

    let cache = load_cache(CORRECTIONS_PATH)?;
    info!("cache: {} entries already processed", cache.len());

    let to_process: Vec<&LexiconEntry> = idiom_entries
        .iter()
        .filter(|e| !cache.contains(&e.id))
        .collect();
    info!("to process: {} new entries", to_process.len());

    if to_process.is_empty() {
        info!("nothing to do");
        return Ok(());
    }

    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(CORRECTIONS_PATH)?;

    let total_batches = to_process.len().div_ceil(BATCH_SIZE);
    let mut processed = 0usize;
    let mut reclassified = 0usize;
    let mut last_call: Option<Instant> = None;

    for (bi, chunk) in to_process.chunks(BATCH_SIZE).enumerate() {
        // Throttle to respect GEMINI_RPM if configured.
        if let (Some(interval), Some(prev)) = (min_interval, last_call) {
            let elapsed = prev.elapsed();
            if elapsed < interval {
                sleep(interval - elapsed);
            }
        }
        let prompt = build_prompt(chunk);
        last_call = Some(Instant::now());
        let results = call_gemini_retry(&prompt, &api_key, &model)?;
        let kind_by_id: HashMap<u32, String> =
            results.into_iter().map(|r| (r.id, r.kind)).collect();

        for entry in chunk {
            // Default to "idiom" if Gemini omitted this id — fail-safe to
            // preserve original behavior rather than silently reclassify.
            let kind = kind_by_id
                .get(&entry.id)
                .map(String::as_str)
                .unwrap_or("idiom");
            let target = if kind == "fixed" {
                "fixed_collocation"
            } else {
                "idiom"
            };
            let new_cats = map_categories(&entry.categories, target);
            let line = CorrectionLine {
                id: entry.id,
                categories: new_cats,
            };
            writeln!(out, "{}", serde_json::to_string(&line)?)?;
            processed += 1;
            if target == "fixed_collocation" {
                reclassified += 1;
            }
        }
        out.flush()?;
        info!(
            "batch {}/{} done ({} processed, {} reclassified so far)",
            bi + 1,
            total_batches,
            processed,
            reclassified
        );
    }

    info!(
        "Done. Processed {} entries, reclassified {} idiom → fixed_collocation.",
        processed, reclassified
    );
    Ok(())
}
