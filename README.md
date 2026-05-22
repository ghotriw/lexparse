# lexparse

Lightweight English NLP microservice: dependency parsing, POS tagging, and MWE (Multi-Word Expression) detection via slot/gap matching.

Built with Rust + ONNX Runtime.

| Model                         | Size                | HuggingFace                                                                                                                           |
| ----------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| deberta-v3-**xsmall** ewt+gum | fastest, lowest RAM | [ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt-gum](https://huggingface.co/ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt-gum) |
| deberta-v3-**xsmall** ewt     | fastest, lowest RAM | [ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt)         |
| deberta-v3-**small** ewt      | faster, less RAM    | [ghotriw/deberta-v3-small-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-small-biaffine-dep-pos-en-ewt)           |
| deberta-v3-**base** ewt       | more accurate       | [ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt)             |

## Setup

### 1. Download model

```bash
./download_models.sh
```

### 2. Build and run

```bash
cargo build --release
./target/release/parser-service
```

Service listens on `0.0.0.0:3000`.

## API

### `POST /parse`

Accepts a JSON array of sentences, returns an SSE stream.

```bash
curl -X POST http://localhost:3000/parse \
  -H 'content-type: application/json' \
  -d '{"sentences": ["She came up with a great idea.", "He spilled the beans."]}'
```

**SSE events:**

| Event      | Payload                                                                            |
| ---------- | ---------------------------------------------------------------------------------- |
| `result`   | `{ index: number; result: SentenceResult }` — emitted per sentence as it completes |
| `progress` | `{ done, total, percent }` — emitted after each sentence                           |
| `done`     | `{ results: SentenceResult[] }` — full array on completion                         |
| `error`    | error message string                                                               |

`result` events arrive incrementally; clients that only need the full batch can ignore them and use `done`.

**`SentenceResult`:**

```ts
{
  tokens: {
    id: number;      // 1-based index in the sentence
    word: string;    // original surface form
    lemma: string;   // base form used for matching
    upos: string;    // Universal POS tag (VERB, NOUN, ADJ, …)
    head: number;    // syntactic head id; 0 = ROOT
    rel: string      // UD dependency relation (nsubj, obj, …)
  }[];

  mwes: {
    id: number;              // lexicon entry id
    pos: string | null;      // syntactic category of the whole MWE ("verb", "noun", …)
    phrase: string;          // canonical dictionary form, e.g. "give up", "spill the beans"
    definition: string | null; // Wiktionary gloss, if available
    surface: string;         // actual text spanned in the sentence (includes gap words if discontinuous)
    categories: ("idiom" | "phrasal_verb" | "proverb" | "fixed_collocation")[];
    has_slot: boolean;       // pattern has a wildcard slot (e.g. "spill someone's beans")
    token_ids: number[];     // 1-based ids of matched tokens (gap words excluded)
    words: string[];         // surface forms of matched tokens (parallel to token_ids)
    discontinuous: boolean   // matched tokens are not contiguous in the sentence
  }[];
}
```

### `GET /health`

Returns `ok`.

## Rebuilding the lexicon

`dic/lexicon.jsonl` is pre-built and committed. Rebuild only if you update the Wiktionary dump or `builder_config.toml`.

### 1. Download the Wiktionary dump

```bash
./download_dict.sh
```

Saves the decompressed dump to `tmp/raw-wiktextract-data.jsonl`.

### 2. Build the lexicon

```bash
cargo run --release --bin builder
```

## Testing

```bash
# Unit and lexicon-level tests (no model required)
cargo test

# Full regression tests including MWE detection (requires model artifacts)
cargo test -- --include-ignored
```

## Configuration

| Env var                   | Default        | Description                                 |
| ------------------------- | -------------- | ------------------------------------------- |
| `PARSER_ADDR`             | `0.0.0.0:3000` | Listen address                              |
| `PARSER_IDLE_UNLOAD_SECS` | `300`          | Unload model after N seconds idle           |
| `PARSER_CPU_ARENA`        | `1`            | ORT arena allocator (set `0` to reduce RSS) |
| `PARSER_MEM_PATTERN`      | `0`            | ORT memory pattern optimization             |
| `PARSER_INTRA_THREADS`    | ORT default    | Number of intra-op threads                  |
