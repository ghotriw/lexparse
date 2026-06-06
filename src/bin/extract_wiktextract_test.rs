use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Deserialize)]
struct LexiconEntry {
    phrase: String,
}

#[derive(Deserialize)]
struct WiktExample {
    text: Option<String>,
}

#[derive(Deserialize)]
struct WiktSense {
    #[serde(default)]
    examples: Vec<WiktExample>,
}

#[derive(Deserialize)]
struct WiktEntry {
    #[serde(default)]
    word: String,
    #[serde(default)]
    senses: Vec<WiktSense>,
}

#[derive(Serialize)]
struct OutputEntry<'a> {
    text: &'a str,
    expected_phrase: &'a str,
}

fn main() -> Result<()> {
    let lexicon_path = "dic/lexicon.jsonl";
    let wiktextract_path = "tmp/raw-wiktextract-data.jsonl";
    let output_path = "tmp/wiktextract_test_data.jsonl";

    println!("Loading Lexicon...");
    let mut lexicon_phrases = HashSet::new();
    let lex_file = File::open(lexicon_path).context("Failed to open lexicon")?;
    for line in BufReader::new(lex_file).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<LexiconEntry>(line) {
            let phrase = entry.phrase.to_lowercase();
            if phrase.contains(' ') || phrase.contains('-') {
                lexicon_phrases.insert(phrase);
            }
        }
    }
    println!(
        "Loaded {} multi-word phrases from lexicon.",
        lexicon_phrases.len()
    );

    println!("Extracting test sentences from wiktextract...");
    if !std::path::Path::new(wiktextract_path).exists() {
        println!("Error: {} not found.", wiktextract_path);
        return Ok(());
    }

    let wikt_file = File::open(wiktextract_path).context("Failed to open wiktextract data")?;
    let out_file = File::create(output_path).context("Failed to create output file")?;
    let mut out_writer = BufWriter::new(out_file);
    let mut count = 0;

    for line in BufReader::new(wikt_file).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let Ok(entry) = serde_json::from_str::<WiktEntry>(line) else {
            continue;
        };

        let word = entry.word.to_lowercase();
        if lexicon_phrases.contains(&word) {
            for sense in entry.senses {
                for example in sense.examples {
                    if let Some(text) = example.text {
                        let text = text.trim();
                        let char_count = text.chars().count();
                        if char_count > 15 && char_count < 300 {
                            let out_entry = OutputEntry {
                                text,
                                expected_phrase: &word,
                            };
                            serde_json::to_writer(&mut out_writer, &out_entry)?;
                            out_writer.write_all(b"\n")?;
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    out_writer.flush()?;
    println!(
        "Extracted and saved {} test sentences to {}.",
        count, output_path
    );

    Ok(())
}
