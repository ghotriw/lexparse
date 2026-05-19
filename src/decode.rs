//! Maximum spanning arborescence (Chu-Liu/Edmonds) over the `s_arc` logits.
//!
//! parser decodes EWT with MST (non-projective). `s_arc[dep][head]` is the score
//! that `head` is the head of `dep`; we build a directed graph where edge
//! `u -> v` (u is head of v) has weight `s_arc[v][u]` and find the maximum
//! arborescence rooted at node 0 (ROOT). Returns `parent[v]` for every word;
//! `parent[root] == root` is the sentinel.

/// `score[u][v]` = weight of edge `u -> v` (u is the head of v). `n` includes
/// the ROOT node at index `root`. Recursive cycle-contraction; `n` is the
/// per-sentence word count (+1), so the O(n^3) constant is irrelevant.
pub fn max_arborescence(n: usize, root: usize, score: &[Vec<f32>]) -> Vec<usize> {
    let neg = f32::NEG_INFINITY;

    // 1. Best incoming edge for every non-root node.
    let mut par = vec![root; n];
    let mut inw = vec![neg; n];
    for v in 0..n {
        if v == root {
            continue;
        }
        for (u, srow) in score.iter().enumerate() {
            if u == v {
                continue;
            }
            if srow[v] > inw[v] {
                inw[v] = srow[v];
                par[v] = u;
            }
        }
    }

    // 2. Find cycles in the functional graph defined by `par`.
    let mut cyc = vec![usize::MAX; n]; // cycle id per node (MAX = none)
    let mut color = vec![0u8; n]; // 0 unvisited, 1 on current path, 2 done
    let mut ncyc = 0usize;
    for s in 0..n {
        if color[s] != 0 {
            continue;
        }
        let mut path = Vec::new();
        let mut v = s;
        loop {
            if v == root || color[v] == 2 {
                break;
            }
            if color[v] == 1 {
                // v reappears on the current path => cycle from v..=v
                let mut in_cycle = false;
                for &x in &path {
                    if x == v {
                        in_cycle = true;
                    }
                    if in_cycle {
                        cyc[x] = ncyc;
                    }
                }
                ncyc += 1;
                break;
            }
            color[v] = 1;
            path.push(v);
            v = par[v];
        }
        for &x in &path {
            color[x] = 2;
        }
    }

    if ncyc == 0 {
        return par; // par is already a valid arborescence
    }

    // 3. Relabel: each cycle -> one super-node; each free node -> its own id.
    let mut newid = vec![usize::MAX; n];
    for (v, &c) in cyc.iter().enumerate() {
        if c != usize::MAX {
            newid[v] = c;
        }
    }
    let mut m = ncyc;
    for id in newid.iter_mut() {
        if *id == usize::MAX {
            *id = m;
            m += 1;
        }
    }
    let new_root = newid[root];

    // 4. Contract: edge weight into a cycle node v is discounted by inw[v]
    //    (the cost of breaking the cycle there). Remember the original
    //    (u, v) edge that realizes each contracted edge for reconstruction.
    let mut nscore = vec![vec![neg; m]; m];
    let mut realize = vec![vec![(usize::MAX, usize::MAX); m]; m];
    for u in 0..n {
        for v in 0..n {
            if u == v || v == root {
                continue;
            }
            let (a, b) = (newid[u], newid[v]);
            if a == b {
                continue;
            }
            let w = if cyc[v] != usize::MAX {
                score[u][v] - inw[v]
            } else {
                score[u][v]
            };
            if w > nscore[a][b] {
                nscore[a][b] = w;
                realize[a][b] = (u, v);
            }
        }
    }

    let npar = max_arborescence(m, new_root, &nscore);

    // 5. Expand: start from `par` (keeps every cycle's internal edges), then
    //    for each contracted node apply the externally chosen entering edge,
    //    which breaks its cycle at exactly one node.
    let mut res = par;
    for (b, &a) in npar.iter().enumerate() {
        if b == new_root {
            continue;
        }
        let (ou, ov) = realize[a][b];
        res[ov] = ou;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_arborescence(n: usize, root: usize, par: &[usize]) -> bool {
        for (v, _) in par.iter().enumerate() {
            if v == root {
                continue;
            }
            // every node reaches root by following parents, no cycles
            let mut cur = v;
            let mut steps = 0;
            while cur != root {
                cur = par[cur];
                steps += 1;
                if steps > n {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn no_cycle_is_greedy() {
        // 3 nodes, root=0. Best heads form a tree already.
        let mut s = vec![vec![f32::NEG_INFINITY; 3]; 3];
        s[0][1] = 5.0; // 0->1
        s[1][2] = 4.0; // 1->2
        s[0][2] = 1.0;
        let par = max_arborescence(3, 0, &s);
        assert_eq!(par[1], 0);
        assert_eq!(par[2], 1);
        assert!(is_arborescence(3, 0, &par));
    }

    #[test]
    fn breaks_a_cycle() {
        // Greedy in-edges would form 1<->2; MST must break it.
        let mut s = vec![vec![f32::NEG_INFINITY; 3]; 3];
        s[0][1] = 1.0; // 0->1
        s[2][1] = 3.0; // 2->1 (greedy pick for 1)
        s[1][2] = 3.0; // 1->2 (greedy pick for 2)  => cycle 1<->2
        s[0][2] = 0.5; // 0->2
        let par = max_arborescence(3, 0, &s);
        assert!(is_arborescence(3, 0, &par));
    }
}
