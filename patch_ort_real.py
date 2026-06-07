import re

with open('src/lib.rs', 'r') as f:
    content = f.read()

# Replace run_inference_batch with the batched tensor one
old_batch = """pub fn run_inference_batch(
    session: &mut Session,
    state: &AppState,
    sentences: &[String],
) -> Vec<anyhow::Result<SentenceResult>> {
    sentences.iter().map(|s| run_inference(session, state, s)).collect()
}"""

new_batch = """pub fn run_inference_batch(
    session: &mut Session,
    state: &AppState,
    sentences: &[String],
) -> Vec<anyhow::Result<SentenceResult>> {
    let start_time = std::time::Instant::now();
    let b = sentences.len();
    let mut results = Vec::with_capacity(b);
    if b == 0 { return results; }

    let batch_words: Vec<Vec<String>> = sentences.iter().map(|s| crate::normalize::tokenize(s)).collect();
    let batch_n: Vec<usize> = batch_words.iter().map(|w| w.len()).collect();
    let max_n = batch_n.iter().copied().max().unwrap_or(0);

    if max_n == 0 {
        for _ in 0..b {
            results.push(Ok(SentenceResult { tokens: vec![], mwes: vec![] }));
        }
        return results;
    }

    let max_w_dim = max_n + 1;
    let mut batch_rows: Vec<Vec<Vec<i64>>> = Vec::with_capacity(b);
    let mut max_fix_len = 1;

    for (i, words) in batch_words.iter().enumerate() {
        let n = batch_n[i];
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(n + 1);
        rows.push(vec![state.cls_id]);
        for word in words {
            let enc = state.tokenizer.encode(word.as_str(), false);
            let mut ids: Vec<i64> = match enc {
                Ok(e) => e.get_ids().iter().map(|&id| id as i64).collect(),
                Err(_) => vec![state.unk_id],
            };
            if ids.is_empty() {
                ids.push(state.unk_id);
            }
            rows.push(ids);
        }
        let fix_len = rows.iter().map(|row| row.len()).max().unwrap_or(1).min(MAX_FIX_LEN);
        if fix_len > max_fix_len { max_fix_len = fix_len; }
        batch_rows.push(rows);
    }

    let mut flat_ids = vec![0i64; b * max_w_dim * max_fix_len];
    for batch_idx in 0..b {
        let rows = &batch_rows[batch_idx];
        let n = batch_n[batch_idx];
        for w_id in 0..=n {
            let row = &rows[w_id];
            let f_len = row.len().min(max_fix_len);
            for f in 0..f_len {
                let idx = batch_idx * max_w_dim * max_fix_len + w_id * max_fix_len + f;
                flat_ids[idx] = row[f];
            }
        }
    }

    let subwords_tensor_nd = match ndarray::Array3::from_shape_vec((b, max_w_dim, max_fix_len), flat_ids) {
        Ok(a) => a,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("ndarray error: {}", e))); }
            return results;
        }
    };
    let subwords_ort = match ort::value::Tensor::from_array(subwords_tensor_nd) {
        Ok(t) => t,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("ort tensor error: {:?}", e))); }
            return results;
        }
    };

    let input_name = String::from(session.inputs()[0].name());
    let inputs = ort::inputs![input_name.as_str() => subwords_ort];

    let tensors = match session.run(inputs) {
        Ok(t) => t,
        Err(e) => {
            for _ in 0..b { results.push(Err(anyhow::anyhow!("forward failed: {:?}", e))); }
            return results;
        }
    };

    if tensors.len() < 3 {
        for _ in 0..b { results.push(Err(anyhow::anyhow!("Expected at least 3 output tensors"))); }
        return results;
    }

    let arc = match tensors[0].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            match ndarray::ArrayView3::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("arc dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("arc extract error: {:?}", e))); } return results; }
    };

    let rel = match tensors[1].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize);
            match ndarray::ArrayView4::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("rel dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("rel extract error: {:?}", e))); } return results; }
    };

    let pos = match tensors[2].try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
            match ndarray::ArrayView3::from_shape(dim, data) {
                Ok(v) => v,
                Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("pos dim error: {:?}", e))); } return results; }
            }
        },
        Err(e) => { for _ in 0..b { results.push(Err(anyhow::anyhow!("pos extract error: {:?}", e))); } return results; }
    };

    let feats_arr: Option<ndarray::ArrayView4<'_, f32>> = if state.feats.is_empty() || tensors.len() < 4 {
        None
    } else {
        match tensors[3].try_extract_tensor::<f32>() {
            Ok((shape, data)) => {
                let dim = (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize);
                match ndarray::ArrayView4::from_shape(dim, data) {
                    Ok(v) => Some(v),
                    Err(_) => None,
                }
            },
            Err(_) => None,
        }
    };

    let n_rels = rel.shape()[3] as usize;
    let n_upos = pos.shape()[2] as usize;

    for batch_idx in 0..b {
        let n = batch_n[batch_idx];
        if n == 0 {
            results.push(Ok(SentenceResult { tokens: vec![], mwes: vec![] }));
            continue;
        }
        let w_dim = n + 1;
        let words = &batch_words[batch_idx];

        let arc_b = arc.index_axis(ndarray::Axis(0), batch_idx);
        let rel_b = rel.index_axis(ndarray::Axis(0), batch_idx);
        let pos_b = pos.index_axis(ndarray::Axis(0), batch_idx);

        let mut score = vec![vec![f32::NEG_INFINITY; w_dim]; w_dim];
        for v in 0..w_dim {
            for (u, srow) in score.iter_mut().enumerate() {
                if u != v {
                    srow[v] = arc_b[[v, u]];
                }
            }
        }
        let heads = crate::decode::max_arborescence(w_dim, 0, &score);

        let upos_ids: Vec<usize> = (0..w_dim)
            .map(|w| crate::decode::argmax((0..n_upos).map(|k| pos_b[[w, k]])))
            .collect();
        let upos_str = |w: usize| state.upos.get(upos_ids[w]).cloned().unwrap_or_else(|| "X".into());

        let feats_str = |w: usize| -> String {
            let Some(fa) = feats_arr.as_ref() else { return "_".into() };
            let mut parts: Vec<String> = Vec::new();
            for (c, (cat, values)) in state.feats.iter().enumerate() {
                let best = crate::decode::argmax((0..values.len()).map(|k| fa[[batch_idx, w, c, k]]));
                if best != 0 {
                    if let Some(val) = values.get(best) {
                        parts.push(format!("{cat}={val}"));
                    }
                }
            }
            if parts.is_empty() { "_".into() } else { parts.join("|") }
        };

        let mut tokens = Vec::with_capacity(n);
        for i in 1..=n {
            let head = heads[i];
            let rel_id = crate::decode::argmax((0..n_rels).map(|k| rel_b[[i, head, k]]));
            let rel = state.rels.get(rel_id).cloned().unwrap_or_else(|| "dep".into());
            let upos = upos_str(i);

            tokens.push(ParsedToken {
                id: i,
                word: words[i - 1].clone(),
                lemma: crate::normalize::lemma(&words[i - 1]),
                head,
                rel,
                upos,
                feats: feats_str(i),
            });
        }

        let is_verb: Vec<bool> = (0..n).map(|k| upos_str(k + 1) == "VERB").collect();
        let word_rels: Vec<String> = tokens.iter().map(|t| t.rel.clone()).collect();
        let word_upos: Vec<String> = tokens.iter().map(|t| t.upos.clone()).collect();
        let mwes = crate::mwe::detect(words, &is_verb, &heads, &word_rels, &word_upos, &state.lexicon);

        results.push(Ok(SentenceResult { tokens, mwes }));
    }

    results
}"""

content = content.replace(old_batch, new_batch)

# In LazySession, replace run_inference with run_inference_batch
content = content.replace(
    'Ok(run_inference(session, state, &sentence))',
    'let mut res = run_inference_batch(session, state, &[sentence.to_string()]); Ok(res.pop().unwrap()?)'
)
# Re-route the batch run too just in case
content = content.replace(
    'Ok(sentences.iter().map(|s| run_inference(session, state, s)).collect())',
    'Ok(run_inference_batch(session, state, sentences))'
)

with open('src/lib.rs', 'w') as f:
    f.write(content)
