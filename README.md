# lexparse

Lightweight English NLP microservice: dependency parsing, POS tagging, phrasal verb and idiom detection.

Built with Rust + ONNX Runtime. Model: [ghotriw/deberta-v3-small-biaffine-dep-pos-en](https://huggingface.co/ghotriw/deberta-v3-small-biaffine-dep-pos-en).

## Setup

### 1. Download model artifacts

Via `hf` CLI:

```bash
hf download ghotriw/deberta-v3-small-biaffine-dep-pos-en \
  model.fp16.onnx vocabs.json idiom_classifier.json tokenizer.json \
  --local-dir model/

hf download ghotriw/deberta-v3-small-biaffine-dep-pos-en \
  lexicon.json phrasal-verbs.json \
  --local-dir dic/
```

Or via the included script (requires only `curl`):

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

| Event      | Payload |
|------------|---------|
| `progress` | `{ done, total, percent }` |
| `done`     | `{ results: SentenceResult[] }` |
| `error`    | error message string |

**`SentenceResult`:**

```ts
{
  tokens: { id: number; word: string; upos: string; head: number; rel: string }[];
  phrasal_verbs: { verb: string; particle: string; verb_id: number; particle_id: number;
                   prep?: string; prep_id?: number }[];
  idioms: { surface: string; span_text: string; token_ids: number[];
            prob: number; idiomatic: boolean }[];
}
```

### `GET /health`

Returns `ok`.

## Testing

```bash
# Unit and lexicon-level tests (no model required)
cargo test

# Full regression tests including idiom and phrasal verb detection (requires model artifacts)
cargo test -- --include-ignored
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `PARSER_ADDR` | `0.0.0.0:3000` | Listen address |
| `PARSER_IDLE_UNLOAD_SECS` | `300` | Unload model after N seconds idle |
| `PARSER_CPU_ARENA` | `1` | ORT arena allocator (set `0` to reduce RSS) |
| `PARSER_MEM_PATTERN` | `0` | ORT memory pattern optimization |
| `PARSER_INTRA_THREADS` | ORT default | Number of intra-op threads |
