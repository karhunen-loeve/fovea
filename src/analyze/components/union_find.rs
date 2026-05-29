//! Private union-find used by the connected-components engine.
//!
//! Standard disjoint-set with path compression and union-by-rank. The
//! amortised cost of `find` / `union` is effectively constant (inverse
//! Ackermann). Index `0` is reserved as the background sentinel and is
//! never returned as a label.
//!
//! This module is `pub(super)` \u2014 part of the engine's implementation,
//! not the public API.

/// Disjoint-set forest over `u64` labels.
///
/// Stored as parallel `parent` / `rank` vectors of length `n`, where
/// slot `0` is the background sentinel (self-rooted, rank 0, never
/// referenced by any real label) and slots `1..n` are real labels.
pub(super) struct UnionFind {
    parent: Vec<u64>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// Construct an empty union-find with an initial capacity hint
    /// `cap` (number of *foreground* labels expected; the background
    /// slot is always allocated).
    pub(super) fn with_capacity(cap: usize) -> Self {
        let mut parent = Vec::with_capacity(cap + 1);
        let mut rank = Vec::with_capacity(cap + 1);
        // Background sentinel at index 0.
        parent.push(0);
        rank.push(0);
        Self { parent, rank }
    }

    /// Allocate a new singleton set and return its label.
    ///
    /// The first call returns `1`; the `k`th call returns `k`.
    pub(super) fn make_set(&mut self) -> u64 {
        let id = self.parent.len() as u64;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Find the root of `x`, performing two-pass path compression.
    ///
    /// # Panics
    ///
    /// Panics if `x` is out of bounds (programmer error).
    pub(super) fn find(&mut self, x: u64) -> u64 {
        // Pass 1: locate the root.
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Pass 2: compress every node on the path to point directly at root.
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    /// Union the sets containing `a` and `b`, by rank.
    pub(super) fn union(&mut self, a: u64, b: u64) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (rank_a, rank_b) = (self.rank[ra as usize], self.rank[rb as usize]);
        if rank_a < rank_b {
            self.parent[ra as usize] = rb;
        } else if rank_a > rank_b {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] = rank_a + 1;
        }
    }

    /// Total number of allocated slots, including the background
    /// sentinel at index 0. The largest valid foreground label is
    /// `len() - 1`.
    pub(super) fn len(&self) -> u64 {
        self.parent.len() as u64
    }

    /// Number of foreground labels allocated (excludes the background
    /// sentinel). Equal to `len() - 1`.
    #[cfg(test)]
    pub(super) fn label_count(&self) -> u64 {
        self.len() - 1
    }

    /// Test-only accessor that exposes the raw parent of `x` (no
    /// `find`, no compression). Used to verify path compression.
    #[cfg(test)]
    pub(super) fn raw_parent(&self, x: u64) -> u64 {
        self.parent[x as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_sentinel_is_self_rooted() {
        let uf = UnionFind::with_capacity(0);
        assert_eq!(uf.len(), 1);
        assert_eq!(uf.raw_parent(0), 0);
    }

    #[test]
    fn first_make_set_returns_one() {
        let mut uf = UnionFind::with_capacity(0);
        assert_eq!(uf.make_set(), 1);
        assert_eq!(uf.make_set(), 2);
        assert_eq!(uf.make_set(), 3);
        assert_eq!(uf.label_count(), 3);
    }

    #[test]
    fn singleton_is_its_own_root() {
        let mut uf = UnionFind::with_capacity(0);
        let a = uf.make_set();
        assert_eq!(uf.find(a), a);
    }

    #[test]
    fn union_then_find_collapses() {
        let mut uf = UnionFind::with_capacity(0);
        let a = uf.make_set();
        let b = uf.make_set();
        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));
    }

    #[test]
    fn union_chain() {
        let mut uf = UnionFind::with_capacity(0);
        let ids: Vec<u64> = (0..5).map(|_| uf.make_set()).collect();
        uf.union(ids[0], ids[1]);
        uf.union(ids[1], ids[2]);
        uf.union(ids[2], ids[3]);
        uf.union(ids[3], ids[4]);
        let root = uf.find(ids[0]);
        for &id in &ids {
            assert_eq!(uf.find(id), root);
        }
    }

    #[test]
    fn union_self_is_noop() {
        let mut uf = UnionFind::with_capacity(0);
        let a = uf.make_set();
        let rank_before = uf.rank[a as usize];
        uf.union(a, a);
        assert_eq!(uf.find(a), a);
        assert_eq!(uf.rank[a as usize], rank_before);
    }

    #[test]
    fn path_compression_flattens_chain() {
        let mut uf = UnionFind::with_capacity(0);
        // Build a small forest of singletons.
        let ids: Vec<u64> = (0..6).map(|_| uf.make_set()).collect();
        // Union in a deliberately chain-like sequence.
        for w in ids.windows(2) {
            uf.union(w[0], w[1]);
        }
        let root = uf.find(ids[0]);
        // After a `find` of every node, raw parent should be the root
        // directly (path compression).
        for &id in &ids {
            uf.find(id);
            // If the node is the root itself, parent equals self.
            // Otherwise, parent must equal root directly.
            let p = uf.raw_parent(id);
            assert!(
                p == root || p == id,
                "node {} parent {} is neither itself nor root {}",
                id,
                p,
                root
            );
        }
    }

    #[test]
    fn union_by_rank_keeps_tree_balanced() {
        // Union-by-rank guarantees tree height <= log2(n). For n=64
        // pre-compression, the height is at most 6.
        let mut uf = UnionFind::with_capacity(0);
        let ids: Vec<u64> = (0..64).map(|_| uf.make_set()).collect();
        for w in ids.windows(2) {
            uf.union(w[0], w[1]);
        }
        // The root's rank is a sound upper bound on tree height before
        // any find() flattens it.
        let root = {
            let mut r = ids[0];
            while uf.raw_parent(r) != r {
                r = uf.raw_parent(r);
            }
            r
        };
        assert!(
            uf.rank[root as usize] as usize <= 6,
            "rank {} exceeds log2(64) = 6",
            uf.rank[root as usize]
        );
    }
}
