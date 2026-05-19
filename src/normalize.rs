//! Bit-exact Rust port of `py_example/normalize.py` (anti-skew §7).
//!
//! This is the SINGLE source of truth for how raw text is split into words and
//! how each word is normalized to a match key. `tokenize` here defines the grid
//! rows / word indices the parser encoder sees, and `lemma` defines which words
//! the idiom matcher pools. Any divergence from the Python reference re-introduces
//! train/serve skew upstream of the frozen encoder — do not "improve" the rules.

use regex::Regex;
use std::sync::OnceLock;

/// Word = a run of ASCII letters/digits/apostrophe/hyphen, OR a single
/// non-space, non-ASCII-alnum char. Mirrors `normalize._TOKEN_RE`.
fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]").unwrap())
}

/// `^[\[(<⟨].*[\])>⟩]$` — bracketed/parenthesized lexicon-surface token.
#[allow(dead_code)] // spec-port: only used offline by build_lexicon
fn slot_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[\[(<⟨].*[\])>⟩]$").unwrap())
}

/// Lexicon slot placeholder words (`normalize._SLOT_WORDS`).
#[allow(dead_code)] // spec-port: only used offline by build_lexicon
const SLOT_WORDS: &[&str] = &[
    "one's", "ones", "one", "oneself", "someone", "someone's", "somebody",
    "somebody's", "something", "sb", "sb's", "sth", "sth's", "so", "sw",
    "your", "yours", "yourself", "pron", "poss", "x", "y", "z",
];

/// Canonical Stage-2 word tokenization. Original case preserved.
pub fn tokenize(text: &str) -> Vec<String> {
    token_re()
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// True iff a *lexicon-surface* token denotes a free gap slot.
#[allow(dead_code)] // spec-port: runtime consumes pre-parsed lexicon `elements`
pub fn is_slot_token(tok: &str) -> bool {
    let stripped = tok.trim();
    if slot_re().is_match(stripped) {
        return true;
    }
    let t = stripped.to_lowercase();
    SLOT_WORDS.contains(&t.as_str())
}

/// Common irregular inflections rule-based stripping can never reach
/// (`normalize._IRREGULAR`). Applied on both sides so it stays symmetric.
fn irregular(t: &str) -> Option<&'static str> {
    Some(match t {
        "brought" => "bring", "bought" => "buy", "caught" => "catch",
        "taught" => "teach", "thought" => "think", "sought" => "seek",
        "fought" => "fight", "went" => "go", "gone" => "go", "took" => "take",
        "taken" => "take", "came" => "come", "gave" => "give", "given" => "give",
        "got" => "get", "gotten" => "get", "made" => "make", "kept" => "keep",
        "left" => "leave", "lost" => "lose", "found" => "find", "held" => "hold",
        "broke" => "break", "broken" => "break", "spoke" => "speak",
        "spoken" => "speak", "drew" => "draw", "drawn" => "draw", "blew" => "blow",
        "blown" => "blow", "threw" => "throw", "thrown" => "throw", "knew" => "know",
        "known" => "know", "grew" => "grow", "grown" => "grow", "flew" => "fly",
        "flown" => "fly", "drove" => "drive", "driven" => "drive", "rose" => "rise",
        "risen" => "rise", "fell" => "fall", "fallen" => "fall", "ran" => "run",
        "sat" => "sit", "stood" => "stand", "understood" => "understand",
        "won" => "win", "sent" => "send", "spent" => "spend", "built" => "build",
        "lit" => "light", "led" => "lead", "fed" => "feed", "bled" => "bleed",
        "bit" => "bite", "hid" => "hide", "hidden" => "hide", "rode" => "ride",
        "ridden" => "ride", "wrote" => "write", "written" => "write",
        "shook" => "shake", "shaken" => "shake", "stole" => "steal",
        "stolen" => "steal", "swore" => "swear", "sworn" => "swear",
        "tore" => "tear", "torn" => "tear", "wore" => "wear", "worn" => "wear",
        "bore" => "bear", "borne" => "bear", "chose" => "choose",
        "chosen" => "choose", "froze" => "freeze", "frozen" => "freeze",
        "began" => "begin", "begun" => "begin", "drank" => "drink", "drunk" => "drink",
        "sang" => "sing", "sung" => "sing", "swam" => "swim", "swum" => "swim",
        "rang" => "ring", "rung" => "ring", "sank" => "sink", "sunk" => "sink",
        "sprang" => "spring", "sprung" => "spring", "dealt" => "deal",
        "meant" => "mean", "felt" => "feel", "slept" => "sleep", "wept" => "weep",
        "swept" => "sweep", "crept" => "creep", "leapt" => "leap", "knelt" => "kneel",
        "dwelt" => "dwell", "bound" => "bind", "wound" => "wind", "ground" => "grind",
        "paid" => "pay", "laid" => "lay", "said" => "say", "fled" => "flee",
        "done" => "do", "does" => "do", "did" => "do", "had" => "have", "has" => "have",
        "men" => "man", "women" => "woman", "children" => "child", "feet" => "foot",
        "teeth" => "tooth", "geese" => "goose", "mice" => "mouse", "people" => "person",
        "lives" => "life", "knives" => "knife", "wives" => "wife", "leaves" => "leaf",
        "wolves" => "wolf", "shelves" => "shelf", "halves" => "half",
        _ => return None,
    })
}

/// Chars stripped from both ends in `normalize.lemma` (keeps internal `'`/`-`).
const STRIP_CHARS: &[char] = &[
    '"', '\'', '’', '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '<', '>',
];

/// Deterministic, dependency-free lemma / match key for one surface token.
/// Crude on purpose — applied identically to both lexicon and sentence sides.
pub fn lemma(tok: &str) -> String {
    let lowered = tok.to_lowercase();
    let trimmed = lowered.trim_matches(|c| STRIP_CHARS.contains(&c));
    if trimmed.is_empty() {
        return String::new();
    }

    // Operate on codepoints so slicing matches Python `str` semantics
    // (e.g. the 3-byte `’` counts as one char).
    let mut chars: Vec<char> = trimmed.chars().collect();

    // possessive
    if ends_with_chars(&chars, &['\'', 's']) || ends_with_chars(&chars, &['’', 's']) {
        chars.truncate(chars.len() - 2);
    } else if ends_with_chars(&chars, &['s', '\'']) || ends_with_chars(&chars, &['s', '’']) {
        chars.truncate(chars.len() - 1);
    }

    let t: String = chars.iter().collect();
    if let Some(irr) = irregular(&t) {
        return irr.to_string();
    }

    let n = chars.len();
    let is_alpha = !chars.is_empty() && chars.iter().all(|c| c.is_alphabetic());
    if n <= 3 || !is_alpha {
        return t;
    }
    if t.ends_with("ies") && n > 4 {
        return format!("{}y", take(&chars, n - 3));
    }
    if ["sses", "shes", "ches", "xes", "zes", "ses"]
        .iter()
        .any(|s| t.ends_with(s))
    {
        return take(&chars, n - 2);
    }
    if t.ends_with("ing") && n > 5 {
        return take(&chars, n - 3);
    }
    if t.ends_with("ied") && n > 4 {
        return format!("{}y", take(&chars, n - 3));
    }
    if t.ends_with("ed") && n > 4 {
        return take(&chars, n - 2);
    }
    if t.ends_with('s') && !t.ends_with("ss") && n > 3 {
        return take(&chars, n - 1);
    }
    t
}

fn ends_with_chars(chars: &[char], suffix: &[char]) -> bool {
    chars.len() >= suffix.len() && chars[chars.len() - suffix.len()..] == *suffix
}

fn take(chars: &[char], k: usize) -> String {
    chars[..k].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_keeps_contractions_splits_punct() {
        assert_eq!(tokenize("He didn't like life?"),
                   vec!["He", "didn't", "like", "life", "?"]);
        assert_eq!(tokenize("twenty-one cats."),
                   vec!["twenty-one", "cats", "."]);
    }

    #[test]
    fn lemma_rules_match_python() {
        // mirrors normalize.lemma behaviour
        assert_eq!(lemma("spilled"), "spill");
        assert_eq!(lemma("beans"), "bean");
        assert_eq!(lemma("brought"), "bring");
        assert_eq!(lemma("knees"), "knee");
        assert_eq!(lemma("beaten"), "beaten"); // ed-rule needs len>4 of stem
        assert_eq!(lemma("boy's"), "boy");
        assert_eq!(lemma("studies"), "study");
        assert_eq!(lemma("the"), "the");
        assert_eq!(lemma("?"), "");
    }

    #[test]
    fn slot_tokens() {
        assert!(is_slot_token("[pron]"));
        assert!(is_slot_token("(somebody)"));
        assert!(is_slot_token("one's"));
        assert!(!is_slot_token("apple"));
    }
}
