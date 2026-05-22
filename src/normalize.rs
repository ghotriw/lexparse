//! Bit-exact Rust port of `stage2_idiom/normalize.py` (anti-skew §7).
//!
//! This is the SINGLE source of truth for how raw text is split into words and
//! how each word is normalized to a match key. `tokenize` here defines the grid
//! rows / word indices the parser encoder sees, and `lemma` defines which words
//! the idiom matcher pools. Any divergence from the Python reference re-introduces
//! train/serve skew upstream of the frozen encoder — do not "improve" the rules.
//!
//! Tokenizer = PTB/UD reproducing UD_English-EWT (~10% residual mismatch, of
//! which ~half is Class 8 deferred / EWT annotation noise). Two stages:
//!   Stage A — one big regex find_iter, ordered longest-first.
//!   Stage B — procedural contraction split on apostrophe-bearing tokens
//!     (n't / 's / 'm / 'd / 're / 've / 'll, plus the apostrophe-less PTB
//!     special `cannot → can + not`). Apostrophe-less pseudo-contractions
//!     (its / dont / im — EWT typo-normalization) are NOT split.
//!
//! Both stages use only the Python `re` ∩ Rust `regex` intersection: no
//! lookaround, no backrefs. Parity vs Python verified by harness over the
//! full EWT corpus.

use regex::Regex;
use std::sync::OnceLock;

/// Hybrid abbreviation list (EWT multi-letter ∪ standard English). Lowercase.
/// MUST stay bit-identical to `normalize._HYBRID_ABBREV_LC`.
const HYBRID_ABBREV_LC: &[&str] = &[
    // titles
    "mr", "mrs", "ms", "dr", "drs", "prof", "sr", "jr", "st", "sts",
    "rev", "capt", "gen", "lt", "col", "sgt", "cpl", "cmdr", "gov",
    "sen", "rep", "pres", "supt", "det", "atty", "ofc", "mt", "pvt",
    // corp / org
    "inc", "corp", "co", "ltd", "llc", "plc", "bros", "assn", "dept",
    "univ", "intl", "conf",
    // months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    // weekdays
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    // latin / generic
    "etc", "vs", "ie", "eg", "cf", "al", "viz", "ca", "ibid", "et", "circa",
    // measure / misc
    "no", "nos", "vol", "ch", "pp", "pg", "fig", "figs", "ed", "eds",
    "trans", "sec", "approx", "wkly", "mo", "yr", "yrs",
    "ave", "blvd", "rd", "ln", "pl", "ft", "ext",
    // US/Canadian state/province
    "ala", "ariz", "ark", "calif", "colo", "conn", "del", "fla", "ga",
    "ill", "ind", "kan", "ky", "la", "mass", "mich", "minn", "miss",
    "mont", "neb", "nev", "okla", "ore", "pa", "tenn", "tex", "va",
    "vt", "wash", "wis", "wyo", "ont", "que", "alb",
    // EWT-only (web text)
    "aplo", "arrv", "attn", "dom", "ect", "esp", "eve", "fax", "info",
    "lb", "lv", "mins", "mob", "oz", "para", "ph", "pop", "ps", "rec",
    "reps", "spec", "wa", "est",
];

/// Productive English prefix list — keep `co-workers`, `non-Microsoft`, etc.
/// joined while letting `15-year`, `wheel-chair`, `F-16` split per EWT.
const PRODUCTIVE_PREFIXES: &[&str] = &[
    "co", "non", "anti", "pre", "post", "mid", "pro", "e", "ex", "mis",
    "counter", "semi", "sub", "super", "re", "un", "multi", "inter", "intra",
    "trans", "over", "under", "self", "near", "all", "cross", "pseudo",
    "quasi", "neo",
];

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let mut abbr: Vec<&&str> = HYBRID_ABBREV_LC.iter().collect();
        abbr.sort_by_key(|s| (-(s.len() as i64), s.to_string()));
        let abbr_alt: String = abbr.iter().map(|s| **s).collect::<Vec<&str>>().join("|");
        let prefix_alt = PRODUCTIVE_PREFIXES.join("|");
        let pattern = format!(
            concat!(
                r"https?://\S+",
                r"|www\.[A-Za-z0-9][A-Za-z0-9.\-]*",
                r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+",
                r"|(?:[A-Za-z]\.){{2,}}",
                r"|(?i:{abbr})\.",
                r"|[A-Z]\.",
                r"|\d+(?:[.,:/]\d+)+",
                r"|\d+(?:-\d+){{2,}}",
                r"|['’]\d+(?:[A-Za-z]+)?",
                r"|\.{{2,}}",
                r"|…",
                r"|-{{2,}}",
                r"|\*{{2,}}",
                r"|(?i:{prefix})-[A-Za-z]+(?:['’][A-Za-z0-9]+)*",
                r"|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*",
                r"|[^\sA-Za-z0-9]",
            ),
            abbr = abbr_alt,
            prefix = prefix_alt,
        );
        Regex::new(&pattern).expect("token_re compile")
    })
}

/// `^[\[(<⟨].*[\])>⟩]$` — bracketed/parenthesized lexicon-surface token.
#[allow(dead_code)] // spec-port: only used offline by build_lexicon
fn slot_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[\[(<⟨].*[\])>⟩]$").unwrap())
}

/// Lexicon slot placeholder words (`normalize._SLOT_WORDS`).
/// `"so"` is intentionally omitted: it is a learner-dictionary abbreviation for
/// "someone" that Wiktionary headwords never use, so it only ever mis-slotted
/// the genuine adverb "so" ("so far so good", "so long as", "so to speak").
#[allow(dead_code)] // spec-port: only used offline by build_lexicon
const SLOT_WORDS: &[&str] = &[
    "one's", "ones", "one", "oneself", "someone", "someone's", "somebody",
    "somebody's", "something", "sb", "sb's", "sth", "sth's", "sw",
    "your", "yours", "yourself", "pron", "poss", "x", "y", "z",
];

/// Canonical Stage-2 word tokenization (PTB/UD reproducing UD_English-EWT).
/// Original case preserved; bit-identical to Python `normalize.tokenize`.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in token_re().find_iter(text) {
        split_contraction(m.as_str(), &mut out);
    }
    out
}

/// Stage B: recursive contraction split applied to each Stage-A token.
/// `'s 'm 'd 're 've 'll` clitics and `n't` split off; `cannot → can + not`.
/// Apostrophe-less surfaces (its / dont / im) are NOT split.
fn split_contraction(tok: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = tok.chars().collect();
    let low: String = chars.iter().flat_map(|c| c.to_lowercase()).collect();

    // Apostrophe-less PTB split — cannot → can + not (case preserved by slice).
    if low == "cannot" {
        out.push(chars[..3].iter().collect());
        out.push(chars[3..].iter().collect());
        return;
    }
    if !low.contains('\'') && !low.contains('’') {
        out.push(tok.to_string());
        return;
    }

    // n't suffix — uniform rule: stem = chars[..n-3], particle = chars[n-3..].
    // Naturally yields can't→ca+n't, won't→wo+n't, didn't→did+n't.
    if chars.len() > 3 {
        let last3: String = chars[chars.len() - 3..].iter().collect();
        let last3_low = last3.to_lowercase();
        if last3_low == "n't" || last3_low == "n’t" {
            let stem: String = chars[..chars.len() - 3].iter().collect();
            split_contraction(&stem, out);
            out.push(last3);
            return;
        }
    }

    // 3-char clitics 'll 're 've (handle both straight and curly apostrophe).
    if chars.len() > 3 {
        let last3: String = chars[chars.len() - 3..].iter().collect();
        let l3 = last3.to_lowercase();
        if matches!(l3.as_str(), "'ll" | "'re" | "'ve" | "’ll" | "’re" | "’ve") {
            let stem: String = chars[..chars.len() - 3].iter().collect();
            split_contraction(&stem, out);
            out.push(last3);
            return;
        }
    }

    // 2-char clitics 's 'm 'd.
    if chars.len() > 2 {
        let last2: String = chars[chars.len() - 2..].iter().collect();
        let l2 = last2.to_lowercase();
        if matches!(l2.as_str(), "'s" | "'m" | "'d" | "’s" | "’m" | "’d") {
            let stem: String = chars[..chars.len() - 2].iter().collect();
            split_contraction(&stem, out);
            out.push(last2);
            return;
        }
    }

    out.push(tok.to_string());
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
        // --- extended irregulars (added 2026-05-19): suppletive past/pp +
        // copula that conservative stripping + prefix-tolerant `eq` cannot
        // reach. POS-ambiguous forms (saw/lay/shot/born/smelt/spelt/spat/
        // abode) are deliberately OMITTED — they need a UPOS=VERB gate, not a
        // blind map, or they regress noun-anchored idioms. Must stay
        // bit-identical to `normalize._IRREGULAR`.
        "am" => "be", "is" => "be", "are" => "be", "was" => "be", "were" => "be",
        "been" => "be", "being" => "be",
        "became" => "become", "bent" => "bend", "bitten" => "bite", "dug" => "dig",
        "ate" => "eat", "forgot" => "forget", "forgotten" => "forget",
        "forgave" => "forgive", "hung" => "hang", "lent" => "lend", "lain" => "lie",
        "met" => "meet", "sold" => "sell", "shone" => "shine", "stuck" => "stick",
        "told" => "tell", "woke" => "wake", "woken" => "wake", "swung" => "swing",
        "spun" => "spin", "struck" => "strike", "wove" => "weave", "woven" => "weave",
        "clung" => "cling", "spilt" => "spill", "stung" => "sting", "stank" => "stink",
        "stunk" => "stink", "strove" => "strive", "wrung" => "wring", "arose" => "arise",
        "awoke" => "awake", "awoken" => "awake", "shrank" => "shrink",
        "shrunk" => "shrink", "swollen" => "swell", "slung" => "sling",
        "slain" => "slay", "slew" => "slay", "strode" => "stride",
        "stridden" => "stride", "trod" => "tread", "trodden" => "tread",
        "shorn" => "shear", "withdrew" => "withdraw", "overcame" => "overcome",
        "forbade" => "forbid", "overtook" => "overtake", "underwent" => "undergo",
        "undertook" => "undertake", "upheld" => "uphold", "withstood" => "withstand",
        "foresaw" => "foresee", "forsook" => "forsake", "undid" => "undo",
        "beheld" => "behold", "overslept" => "oversleep",
        "misunderstood" => "misunderstand", "overdid" => "overdo",
        "withheld" => "withhold", "overthrew" => "overthrow", "partook" => "partake",
        _ => return None,
    })
}

/// POS-gated irregulars (`normalize._IRREGULAR_VERB`): only safe when the
/// token is a VERB, because the surface is also a common NOUN/ADJ with a
/// different lemma. NEVER applied on the lexicon/offline side — only sentence-
/// side with model-predicted UPOS. `lay`/`born` are intentionally absent
/// (VERB-tagged in their confounding sense → a POS gate cannot disambiguate).
fn irregular_verb(t: &str) -> Option<&'static str> {
    Some(match t {
        "saw" => "see", "shot" => "shoot", "smelt" => "smell",
        "spelt" => "spell", "spat" => "spit", "abode" => "abide",
        _ => return None,
    })
}

/// Chars stripped from both ends in `normalize.lemma` (keeps internal `'`/`-`).
const STRIP_CHARS: &[char] = &[
    '"', '\'', '’', '.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '<', '>',
];

/// Deterministic, dependency-free lemma / match key for one surface token.
/// Crude on purpose — applied identically to both lexicon and sentence sides.
/// Lexicon/offline callers use this (POS-free, symmetric).
pub fn lemma(tok: &str) -> String {
    lemma_pos(tok, false)
}

/// POS-aware lemma: `is_verb` (sentence-side only, from model-predicted UPOS)
/// enables the `_IRREGULAR_VERB` remap. Mirrors `normalize.lemma(tok, is_verb)`.
pub fn lemma_pos(tok: &str, is_verb: bool) -> String {
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
    if is_verb {
        if let Some(irr) = irregular_verb(&t) {
            return irr.to_string();
        }
    }
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
    fn tokenize_splits_contractions_and_punct() {
        // PTB/UD: contractions split, punctuation breaks off.
        assert_eq!(tokenize("He didn't like life?"),
                   vec!["He", "did", "n't", "like", "life", "?"]);
        // Hyphens generally split (EWT convention); productive prefix kept.
        assert_eq!(tokenize("twenty-one cats."),
                   vec!["twenty", "-", "one", "cats", "."]);
        assert_eq!(tokenize("co-workers e-mail."),
                   vec!["co-workers", "e-mail", "."]);
        // Dotted abbreviation kept as one token (the original p.m. bug).
        assert_eq!(tokenize("at 10 p.m. local"),
                   vec!["at", "10", "p.m.", "local"]);
        // Acronyms.
        assert_eq!(tokenize("U.S. policy"),
                   vec!["U.S.", "policy"]);
        // Special stems wo/ca + cannot.
        assert_eq!(tokenize("I cannot won't can't"),
                   vec!["I", "can", "not", "wo", "n't", "ca", "n't"]);
        // Clitics: possessive 's, 'm, 're, 've, 'll, 'd.
        assert_eq!(tokenize("Bush's I'm they're we've you'll I'd"),
                   vec!["Bush", "'s", "I", "'m", "they", "'re", "we", "'ve", "you", "'ll", "I", "'d"]);
        // Apostrophe-less pseudo-contraction NOT split.
        assert_eq!(tokenize("its dont"),
                   vec!["its", "dont"]);
        // Multi-dash, ellipsis, repeated star.
        assert_eq!(tokenize("end -- mid ... ***"),
                   vec!["end", "--", "mid", "...", "***"]);
        // Number with internal punct; phone.
        assert_eq!(tokenize("256,000 and 14:57 and 303-832-8160"),
                   vec!["256,000", "and", "14:57", "and", "303-832-8160"]);
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
        // extended irregulars (2026-05-19) — copula + suppletive past/pp
        assert_eq!(lemma("is"), "be");
        assert_eq!(lemma("was"), "be");
        assert_eq!(lemma("were"), "be");
        assert_eq!(lemma("been"), "be");
        assert_eq!(lemma("being"), "be");
        assert_eq!(lemma("told"), "tell");
        assert_eq!(lemma("became"), "become");
        assert_eq!(lemma("woke"), "wake");
        // POS-ambiguous forms stay literal (no blind map without a UPOS gate)
        assert_eq!(lemma("saw"), "saw");
        assert_eq!(lemma("lay"), "lay");
        assert_eq!(lemma("shot"), "shot");
    }

    #[test]
    fn pos_gated_irregular_verb() {
        // NOUN/default reading: literal (lemma == lemma_pos(_, false))
        assert_eq!(lemma_pos("saw", false), "saw");
        assert_eq!(lemma_pos("shot", false), "shot");
        assert_eq!(lemma_pos("abode", false), "abode");
        // VERB reading: remapped to the base via _IRREGULAR_VERB
        assert_eq!(lemma_pos("saw", true), "see");
        assert_eq!(lemma_pos("Saw", true), "see"); // case-insensitive
        assert_eq!(lemma_pos("shot", true), "shoot");
        assert_eq!(lemma_pos("smelt", true), "smell");
        assert_eq!(lemma_pos("spelt", true), "spell");
        assert_eq!(lemma_pos("spat", true), "spit");
        assert_eq!(lemma_pos("abode", true), "abide");
        // lay/born intentionally NOT gated (VERB-tagged in confounding sense)
        assert_eq!(lemma_pos("lay", true), "lay");
        assert_eq!(lemma_pos("born", true), "born");
        // a non-gated VERB still flows through the normal rules
        assert_eq!(lemma_pos("walked", true), "walk");
        assert_eq!(lemma_pos("brought", true), "bring");
    }

    #[test]
    fn slot_tokens() {
        assert!(is_slot_token("[pron]"));
        assert!(is_slot_token("(somebody)"));
        assert!(is_slot_token("one's"));
        assert!(!is_slot_token("apple"));
    }
}
