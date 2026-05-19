//! Read one sentence per line on stdin, print the tokenizer output as a JSON
//! array per line. Used by the Python parity harness to diff Rust ↔ Python.

// lemma/irregular/STRIP_CHARS/etc. are live in main.rs but unused by this bin;
// silence at the bin crate root so the parity harness compiles cleanly.
#![allow(dead_code)]

use std::io::{self, BufRead, Write};

#[path = "../normalize.rs"]
mod normalize;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let toks = normalize::tokenize(&line);
        let json = serde_json::to_string(&toks).expect("ser");
        writeln!(out, "{}", json).expect("write");
    }
}
