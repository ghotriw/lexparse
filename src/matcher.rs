//! Subgraph Isomorphism matcher for recursive dependency trees.
//! 
//! Matches a directed tree of lemmas and dependency relations from the lexicon
//! against the sentence's dependency graph.

use crate::normalize::lemma_pos;
use crate::mwe::LexiconTreeNode;

/// POS-aware lemmatize: `is_verb[i]` (from model-predicted UPOS) enables the
/// `_IRREGULAR_VERB` remap for token `i`. Mirrors `matcher.lemmatize(tokens,
/// upos)`. `is_verb` must be aligned 1:1 with `tokens`.
pub fn lemmatize_pos(tokens: &[String], is_verb: &[bool]) -> Vec<String> {
    tokens
        .iter()
        .zip(is_verb.iter())
        .map(|(t, &v)| lemma_pos(t, v))
        .collect()
}

/// Prefix-tolerant lemma equality.
fn eq(a: &str, b: &str) -> bool {
    a == b
}

/// Recursively match a lexicon tree against the sentence graph using DFS.
///
/// * `target_node`: 0-based token index in the sentence.
/// * `lex_node`: The current node in the MWE dependency tree.
/// * `is_root`: True if this is the root of the MWE tree. The root ignores its `rel`.
/// * `lemmas`: Lemmatized sentence tokens.
/// * `children`: `children[head]` contains 0-based indices of child tokens. `head` is 1-based token ID.
/// * `rels`: Dependency relations for each token.
/// * `matched_nodes`: Accumulates matched 0-based token indices.
pub fn match_tree(
    target_node: usize,
    lex_node: &LexiconTreeNode,
    is_root: bool,
    lemmas: &[String],
    children: &[Vec<usize>],
    rels: &[String],
    matched_nodes: &mut Vec<usize>,
) -> bool {
    // 1. Check lemma (unless it's a slot placeholder)
    if !lex_node.is_slot {
        if !eq(&lemmas[target_node], &lex_node.lemma) {
            return false;
        }
    }

    // 2. Check relation (the root node's relation to the broader sentence is ignored)
    if !is_root {
        if rels[target_node] != lex_node.rel {
            return false;
        }
    }

    matched_nodes.push(target_node);

    // 3. Match all dependencies recursively
    for lex_child in &lex_node.deps {
        let mut found_match = false;
        
        // Children of `target_node`. target_node is 0-based, but `children` uses 1-based indexing.
        let target_head_id = target_node + 1;
        
        if target_head_id < children.len() {
            let child_candidates = &children[target_head_id];
            
            for &sent_child in child_candidates {
                // To be safe, avoid re-matching the same token
                if matched_nodes.contains(&sent_child) {
                    continue;
                }
                
                let backup_len = matched_nodes.len();
                if match_tree(sent_child, lex_child, false, lemmas, children, rels, matched_nodes) {
                    found_match = true;
                    break;
                } else {
                    // Backtrack
                    matched_nodes.truncate(backup_len);
                }
            }
        }
        
        if !found_match {
            return false;
        }
    }

    true
}
