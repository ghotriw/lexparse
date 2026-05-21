# lexparse

Lightweight English NLP microservice: dependency parsing, POS tagging, and MWE (Multi-Word Expression) detection via Subgraph Isomorphism.

Built with Rust + ONNX Runtime.

| Model                 | Size                | HuggingFace                                                                                                                   |
| --------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| deberta-v3-**xsmall** | fastest, lowest RAM | [ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-xsmall-biaffine-dep-pos-en-ewt) |
| deberta-v3-**small**  | faster, less RAM    | [ghotriw/deberta-v3-small-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-small-biaffine-dep-pos-en-ewt)   |
| deberta-v3-**base**   | more accurate       | [ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt](https://huggingface.co/ghotriw/deberta-v3-base-biaffine-dep-pos-en-ewt)     |

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
    id: number;
    word: string;
    lemma: string;
    upos: string;
    head: number;
    rel: string
  }[];

  mwes: {
    surface: string;
    category: "idiom" | "phrasal_verb" | "collocation_phrase" | "proverb_saying";
    has_slot: boolean;
    token_ids: number[];
    words: string[];
    span_text: string;
    discontinuous: boolean;
    tree_connected: boolean
  }[];
}
```

### `GET /health`

Returns `ok`.

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
