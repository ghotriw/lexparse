use lexparse::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

#[derive(Deserialize)]
struct TestCase {
    text: String,
    expected_phrase: String,
}

fn build_state() -> anyhow::Result<AppState> {
    let vocab: Vocab =
        serde_json::from_str::<VocabRaw>(&std::fs::read_to_string(VOCAB_PATH)?)?.into();
    let mut lexicon = mwe::MweLexicon::load_with_custom(LEXICON_PATH, CUSTOM_LEXICON_PATH)?;
    lexicon.apply_corrections(CORRECTIONS_PATH)?;

    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    let cls_id = tokenizer.token_to_id("[CLS]").unwrap_or(1) as i64;
    let unk_id = tokenizer.token_to_id("[UNK]").unwrap_or(3) as i64;

    let (job_tx, _job_rx) = mpsc::unbounded_channel();
    Ok(AppState {
        session: LazySession::new(),
        tokenizer,
        rels: vocab.rels,
        upos: vocab.upos,
        feats: vocab.feats,
        lexicon,
        job_tx,
        cls_id,
        unk_id,
    })
}

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    // Run from the lexparse root dir
    let test_path = "tmp/wiktextract_test_data.jsonl";

    let file = File::open(test_path).expect(&format!("Could not open {}", test_path));
    let reader = BufReader::new(file);

    let mut test_data = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(case) = serde_json::from_str::<TestCase>(&line) {
                test_data.push(case);
            }
        }
    }

    println!("Loaded {} test sentences.", test_data.len());
    if test_data.is_empty() {
        println!("Run the python extraction script first to generate test/wiktextract_test_data.jsonl");
        return Ok(());
    }

    println!("Loading model...");
    let state = build_state().expect("Failed to build model state");

    let mut true_positives = 0;
    let total_expected = test_data.len();
    let mut total_predicted_mwes = 0;

    let mut report_lines = Vec::new();
    let mut fn_report_lines = Vec::new();

    println!("Running Evaluation against local model directly...");

    let start_time = std::time::Instant::now();

    let batch_size = std::env::var("PARSER_BATCH_SIZE")
        .unwrap_or_else(|_| "1".to_string())
        .parse::<usize>()
        .unwrap_or(1)
        .max(1);

    println!("Using PARSER_BATCH_SIZE={}", batch_size);

    state.session.with_session(|session| {
        let mut processed = 0;

        for chunk in test_data.chunks(batch_size) {
            if processed > 0 && processed % 50 < batch_size {
                use std::io::Write;
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed = processed as f64 / elapsed;
                let remaining = (total_expected - processed) as f64 / speed;
                let mins = (remaining / 60.0) as u64;
                let secs = (remaining % 60.0) as u64;
                print!("\rProcessed {}/{} (ETA: {:02}:{:02})   ", processed, total_expected, mins, secs);
                std::io::stdout().flush().unwrap();
            }

            let texts: Vec<String> = chunk.iter().map(|c| c.text.clone()).collect();
            let batch_results = run_inference_batch(session, &state, &texts);

            for (j, res) in batch_results.into_iter().enumerate() {
                let case = &chunk[j];
                let result = match res {
                    Ok(r) => r,
                    Err(e) => {
                        println!("\nInference failed on '{}': {}", case.text, e);
                        continue;
                    }
                };

                total_predicted_mwes += result.mwes.len();

                let expected_lower = case.expected_phrase.to_lowercase();
                let matched = result
                    .mwes
                    .iter()
                    .find(|m| {
                        m.phrase.to_lowercase() == expected_lower
                            || m.surface.to_lowercase() == expected_lower
                            || m.definition
                                .as_deref()
                                .unwrap_or("")
                                .to_lowercase()
                                .contains(&expected_lower)
                    });

                if let Some(m) = matched {
                    true_positives += 1;
                    let log_str = format!(
                        "Sentence: {}\n  [✓] FOUND: '{}' (matched lexicon: {})\n",
                        case.text, m.surface, case.expected_phrase
                    );
                    report_lines.push(log_str);
                } else {
                    let log_str = format!(
                        "Sentence: {}\n  [✗] MISSED (False Negative): Expected '{}'\n",
                        case.text, case.expected_phrase
                    );
                    report_lines.push(log_str.clone());
                    fn_report_lines.push(log_str);
                }
            }
            processed += chunk.len();
        }

        println!("\rProcessed {}/{}      ", total_expected, total_expected);
        Ok::<(), anyhow::Error>(())
    })?;

    let recall = if total_expected > 0 {
        true_positives as f64 / total_expected as f64
    } else {
        0.0
    };

    let metrics_text = format!(
        "--- Evaluation Results (Custom Wiktextract Dataset) ---\n\
        Total Sentences (Expected Phrases): {}\n\
        Total Phrases Extracted by Parser:  {}\n\
        Target Phrases Successfully Found:  {}\n\
        Recall (Sensitivity):               {:.4}\n\
        -------------------------------------------------------\n\
        * Note: Precision is not calculated because the parser might legitimately find \
        other MWEs in these sentences besides the target phrase.\n\n",
        total_expected, total_predicted_mwes, true_positives, recall
    );

    println!("{}", metrics_text);

    let mut f_report = File::create("test/eval_wiktextract_report.txt")?;
    f_report.write_all(metrics_text.as_bytes())?;
    for l in &report_lines {
        f_report.write_all(l.as_bytes())?;
    }

    let mut f_missed = File::create("test/eval_wiktextract_missed.txt")?;
    f_missed.write_all(metrics_text.as_bytes())?;
    for l in &fn_report_lines {
        f_missed.write_all(l.as_bytes())?;
    }

    println!("Reports written to:");
    println!(" - test/eval_wiktextract_report.txt (All results)");
    println!(" - test/eval_wiktextract_missed.txt (Only the missed ones - useful for debugging!)");

    Ok(())
}
