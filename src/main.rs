use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::CorsLayer;
use tracing::info;

use lexparse::*;

#[derive(Deserialize)]
struct ParseBatchRequest {
    sentences: Vec<String>,
}

#[derive(Serialize)]
struct ProgressEvent {
    done: usize,
    total: usize,
    percent: u8,
}

#[derive(Serialize)]
struct DoneEvent {
    results: Vec<SentenceResult>,
}

#[derive(Serialize)]
struct ResultEvent<'a> {
    index: usize,
    result: &'a SentenceResult,
}

async fn health() -> &'static str {
    "ok"
}

async fn parse_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ParseBatchRequest>,
) -> impl IntoResponse {
    let total = req.sentences.len();

    let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Event>();

    tokio::spawn(async move {
        if total == 0 {
            let done_data =
                serde_json::to_string(&DoneEvent { results: vec![] }).unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("done").data(done_data));
            return;
        }

        // Submit every sentence as an independent job and remember the reply channels
        // in order. Jobs from concurrent requests are interleaved in the shared worker
        // queue, so no single request holds the model lock for its entire batch.
        let mut receivers = Vec::with_capacity(total);
        for sentence in &req.sentences {
            let (reply_tx, reply_rx) = oneshot::channel();
            if state
                .job_tx
                .send(SentenceJob {
                    sentence: sentence.trim().to_string(),
                    reply: reply_tx,
                })
                .is_err()
            {
                let _ = sse_tx
                    .send(Event::default().event("error").data("inference worker stopped"));
                return;
            }
            receivers.push(reply_rx);
        }

        let mut results: Vec<SentenceResult> = Vec::with_capacity(total);
        for (i, reply_rx) in receivers.into_iter().enumerate() {
            let sentence_result = match reply_rx.await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    let _ = sse_tx
                        .send(Event::default().event("error").data(format!("sentence {i}: {e}")));
                    return;
                }
                Err(_) => {
                    let _ = sse_tx
                        .send(Event::default().event("error").data("inference worker dropped"));
                    return;
                }
            };

            // Per-sentence result event (clients that don't need it can ignore it).
            let result_data =
                serde_json::to_string(&ResultEvent { index: i, result: &sentence_result })
                    .unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("result").data(result_data));

            let done = i + 1;
            let percent = (done * 100 / total) as u8;
            let progress =
                serde_json::to_string(&ProgressEvent { done, total, percent }).unwrap_or_default();
            let _ = sse_tx.send(Event::default().event("progress").data(progress));

            results.push(sentence_result);
        }

        let done_data = serde_json::to_string(&DoneEvent { results }).unwrap_or_default();
        let _ = sse_tx.send(Event::default().event("done").data(done_data));
    });

    let sse_stream = Box::pin(stream::unfold(sse_rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<Event, std::convert::Infallible>(event), rx))
    }));

    Sse::new(sse_stream).keep_alive(KeepAlive::default())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    info!("loading vocab from {VOCAB_PATH}");
    let vocab: Vocab = serde_json::from_str::<VocabRaw>(&std::fs::read_to_string(VOCAB_PATH)?)?.into();

    info!("loading lexicon from {LEXICON_PATH}");
    let lexicon = lexparse::mwe::MweLexicon::load(LEXICON_PATH)?;

    info!("loading tokenizer");
    let tokenizer = tokenizers::Tokenizer::from_file("model/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

    let (job_tx, job_rx) = mpsc::unbounded_channel::<SentenceJob>();

    let state = Arc::new(AppState {
        session: LazySession::new(),
        tokenizer,
        rels: vocab.rels,
        upos: vocab.upos,
        lexicon,
        job_tx,
    });

    spawn_evictor(Arc::downgrade(&state));
    spawn_inference_worker(Arc::clone(&state), job_rx);
    info!(
        idle_unload_secs = idle_unload_secs(),
        lexicon = state.lexicon.entries.len(),
        "model is lazy-loaded on first request, evicted after idle"
    );

    let app = Router::new()
        .route("/health", get(health))
        .route("/parse", post(parse_batch))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr = std::env::var("PARSER_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".into());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
