// Copyright (c) 2025 gudzpoz
// Copyright (c) 2019 Sevag Hanssian
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Copyright (c) 2015 - present Microsoft Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// See crates/rope/LICENSE for more license information.

use crate::metrics::{rel_node_at_metric, BaseMetric, Cursor, CursorPos, Metric};
use crate::piece::{DeleteResult, Insertion, RopePiece, SplitResult, Sum};
use crate::rb_base::{Node, RbSlab, Ref, SafeRef, LEFT, RIGHT};
use std::collections::VecDeque;
use std::ops::Range;

/// The rope implementation
pub struct Rope<T: RopePiece> {
    tree: RbSlab<T>,
    sum: T::S,
    context: T::Context,
}
impl<T: RopePiece> Default for Rope<T> where T::Context: Default {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// Information about a particular position in tree
pub struct PiecePosition<'a, T> {
    /// The piece that the position lies in
    pub piece: &'a T,
    /// The relative offset of the position in base metric
    ///
    /// This offset can be at the end of the piece (`piece.len()`).
    pub offset_in_piece: usize,
}

impl<T: RopePiece> Rope<T> {
    /// Creates a new rope with the specified context
    pub fn new(context: T::Context) -> Self {
        Self {
            tree: RbSlab::new(),
            sum: T::S::identity(),
            context,
        }
    }

    /// Returns the inner context
    pub fn context(&self) -> &T::Context {
        &self.context
    }
    /// Returns the inner context
    pub fn context_mut(&mut self) -> &mut T::Context {
        &mut self.context
    }
    
    /// Check if the rope contains nothing
    pub fn is_empty(&self) -> bool {
        self.tree.root().is_none()
    }
    
    /// Returns the length of the rope
    /// 
    /// Note that we allow zero-length nodes, like markers.
    /// So zero-length does not imply an empty tree.
    /// Use [Self::is_empty] for empty checking.
    pub fn len(&self) -> usize {
        self.sum.len()
    }
    
    /// Return the measurement of the whole tree
    pub fn measure<M: Metric<T>>(&self) -> usize {
        M::measure(&self.sum)
    }
    
    /// Converts offsets from one measurement to another
    pub fn convert_metrics<M: Metric<T>, N: Metric<T>>(&self, measurement: usize) -> Option<usize> {
        self.cursor::<M>(measurement).map(|c| c.abs_offset::<N>())
    }

    /// Batch insert multiple values at consecutive nodes
    pub fn init(&mut self, mut value: VecDeque<T>) {
        assert!(self.sum.len() == 0 && !value.is_empty());
        let mut node = self.rb_insert(None, value.pop_front().unwrap(), LEFT);
        while let Some(next) = value.pop_front() {
            node = self.rb_insert(Some(node), next, RIGHT);
        }
    }

    /// Insert a node with `value` at `offset` in base metric
    pub fn insert(&mut self, offset: usize, value: T) {
        let value = Insertion::from(value);
        if let Some(mut pos) = self.node_at_metric::<BaseMetric>(offset) {
            pos.insert(self, value);
        } else {
            assert_eq!(offset, 0);
            self.sum = value.1;
            self.rb_insert(None, value.0, LEFT);
        }
    }

    /// Similar to [Self::insert], but it tries to merge the insertion with adjacent nodes
    pub fn insert_merging(&mut self, offset: usize, value: T) {
        let value = Insertion::from(value);
        if let Some(mut pos) = self.node_at_metric::<BaseMetric>(offset) {
            pos.insert_merging(self, value);
        } else {
            assert_eq!(offset, 0);
            self.sum = value.1;
            self.rb_insert(None, value.0, LEFT);
        }
    }

    /// Delete a substring (or sub-rope?) from the rope
    pub fn delete(&mut self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        if offset == 0 && len == self.sum.len() {
            self.sum = T::S::identity();
            self.tree.clear();
        }
        if let Some(pos) = self.node_at_metric::<BaseMetric>(offset) {
            pos.delete_len(self, len);
        }
    }

    /// Similar to [Self::delete], but it tries to merge the adjacent nodes after deletion
    pub fn delete_merging(&mut self, offset: usize, len: usize, mergeable: impl Fn(&T, &T) -> bool) {
        if len == 0 || (offset == 0 && len == self.sum.len()) {
            self.delete(offset, len);
            return;
        }
        if let Some(pos) = self.node_at_metric::<BaseMetric>(offset) {
            pos.delete_len_merging(self, len, mergeable);
        }
    }

    /// Locate an offset in the current tree
    pub fn search(&self, offset: usize) -> Option<PiecePosition<'_, T>> {
        if let Some(CursorPos { node, offset_in_node, .. }) = self.node_at(offset) {
            Some(PiecePosition { piece: &self.tree[node].piece, offset_in_piece: offset_in_node })
        } else {
            None
        }
    }

    /// Returns a [NodePosition] corresponding to the supplied offset
    ///
    /// Note that the position is put in a tail position if possible.
    /// That is, with adjacent nodes like "(node1)(node2)", it prefers
    /// returning `(node1, len(node1))` instead of `(node2, 0)`. (This
    /// is guaranteed unless `offset` is 0).
    ///
    /// Also, when there are zero-width nodes, the position is *not*
    /// guaranteed to be consistent and can be at the tail position of
    /// any node as long as their tail position matches. The caller is
    /// responsible for adjusting the position if they want to.
    pub(crate) fn node_at(&self, offset: usize) -> Option<CursorPos<T>> {
        self.node_at_metric::<BaseMetric>(offset)
    }

    pub(crate) fn node_at_metric<M: Metric<T>>(&self, offset: usize) -> Option<CursorPos<T>> {
        Self::tree_node_at_metric::<M>(&self.tree, offset)
    }
    fn tree_node_at_metric<M: Metric<T>>(tree: &RbSlab<T>, offset: usize) -> Option<CursorPos<T>> {
        let x = tree.root();
        if offset == 0 {
            let mut x = x?;
            x = tree.edge(x, LEFT);
            Some(CursorPos::new(x, 0))
        } else {
            rel_node_at_metric::<T, M>(tree, x, offset)
        }
    }

    /// Get a cursor into the specified offset with metric `M`
    ///
    /// Note that the position is put in a tail position if possible.
    pub fn cursor<M: Metric<T>>(&self, offset: usize) -> Option<Cursor<'_, T>> {
        let pos = self.node_at_metric::<M>(offset);
        pos.map(|c| c.cursor(&self.tree))
    }

    /// Iterate over the rope within the range
    pub fn for_range<M: Metric<T>>(
        &self, range: Range<usize>,
        mut f: impl FnMut(&T::Context, &T, Range<usize>) -> bool,
    ) {
        let Some(start) = self.cursor::<M>(range.start) else { return };
        let start = start.inner();
        let mut end = start.clone();
        let end = if end.navigate::<BaseMetric>(&self.tree, range.len() as isize) {
            end
        } else {
            let node = self.tree.edge(self.tree.root().unwrap(), RIGHT);
            CursorPos::new(node, self.tree[node].piece.len())
        };

        let mut i = Some(start.node);
        while let Some(idx) = i {
            let offset = if idx == start.node {
                start.offset_in_node
            } else {
                0
            };
            let piece = &self.tree[idx].piece;
            let end_off = if idx == end.node {
                end.offset_in_node
            } else {
                piece.len()
            };
            if !f(&self.context, piece, offset..end_off) {
                break;
            }
            if idx == end.node {
                break;
            }
            i = self.tree.next(idx, RIGHT);
        }
    }

    fn rb_insert(&mut self, node: Ref, piece: T, dir: usize) -> SafeRef {
        let z = self.tree.insert(Node::new(piece));
        match node {
            None => {
                debug_assert!(self.is_empty());
                self.tree.set_root(Some(z));
                self.tree[z].rb.red = false;
            }
            Some(node) => {
                let n = &mut self.tree[node];
                let n_child = n.rb.children[dir];
                if let Some(n_child) = n_child {
                    let prev = self.tree.edge(n_child, dir ^ 1);
                    self.tree[prev].rb.children[dir ^ 1] = Some(z);
                    self.tree[z].rb.parent = Some(prev);
                } else {
                    n.rb.children[dir] = Some(z);
                    self.tree[z].rb.parent = Some(node);
                }
            }
        }
        self.tree.fix_insert(Some(z));
        z
    }

    fn safe_delete(&mut self, node: SafeRef) -> T {
        self.tree[node].piece.delete(&mut self.context);
        self.tree.delete(node)
    }

    fn delete_nodes(&mut self, nodes: Vec<SafeRef>) {
        for node in nodes {
            self.safe_delete(node);
        }
    }

    fn recompute_metadata(&mut self) {
        self.sum = self.tree.calculate_sum(self.tree.root());
    }
    
    #[cfg(test)]
    fn is_valid(&self) {
        let sum = self.tree.is_valid();
        assert!(sum == self.sum);
    }
}

impl<T: RopePiece> CursorPos<T> {
    /// Get necessary information to get value from the piece
    pub fn get<'a>(&self, rope: &'a Rope<T>) -> (&'a T, usize, &'a T::Context) {
        (&rope.tree[self.node].piece, self.offset_in_node, &rope.context)
    }
    /// Get necessary information to mutate value from the piece
    pub fn get_mut<'a>(&self, rope: &'a mut Rope<T>) -> (&'a mut T, usize, &'a mut T::Context) {
        (&mut rope.tree[self.node].piece, self.offset_in_node, &mut rope.context)
    }

    /// Inserts a new piece as a new node to the left of the current node
    ///
    /// Returns a new cursor pointing to the start of the inserted node
    pub fn insert_left(&self, rope: &mut Rope<T>, piece: T) -> Self {
        let new = rope.rb_insert(Some(self.node), piece, LEFT);
        Self::new(new, 0)
    }
    /// Inserts a new piece as a new node to the right of the current node
    ///
    /// Returns a new cursor pointing to the start of the inserted node
    pub fn insert_right(&self, rope: &mut Rope<T>, piece: T) -> Self {
        let new = rope.rb_insert(Some(self.node), piece, RIGHT);
        Self::new(new, 0)
    }
    /// Updates the metadata of parent nodes and moves the cursor forward
    ///
    /// Note that it does not update the metadata of the tree ([Rope.sum]).
    fn node_update(&mut self, rope: &mut Rope<T>, delta: &T::S) {
        self.offset_in_node = self.offset_in_node.wrapping_add(delta.len());
        rope.tree.update_metadata(self.node, delta);
    }
    /// Update the summary of the current node by `delta`
    ///
    /// The cursor itself is incremented by `delta.len()`.
    pub fn update(&mut self, rope: &mut Rope<T>, delta: &T::S) {
        rope.sum.add_assign(delta);
        self.node_update(rope, delta);
    }
    /// Deletes the current node, returns a new cursor if the tree is not empty
    pub fn delete(self, rope: &mut Rope<T>) -> (Option<Self>, T) {
        let mut next = self.clone();
        let next = if next.next_piece(rope) {
            Some(next)
        } else if next.prev_piece(rope) {
            let node = next.node;
            let len = rope.tree[node].piece.len();
            Some(CursorPos::new(next.node, len))
        } else {
            None
        };
        let value = rope.safe_delete(self.node);
        rope.sum.sub_assign(&value.summarize());
        (next, value)
    }

    /// Merges adjacent nodes after deletion
    pub fn delete_len_merging(
        self, rope: &mut Rope<T>, len: usize,
        mergeable: impl Fn(&T, &T) -> bool,
    ) -> Option<Self> {
        let pos = self.delete_len(rope, len)?;
        let mut copy = pos.clone();
        let (mut left, right) = if pos.offset_in_node == 0 {
            if !copy.prev_piece(rope) {
                return None;
            }
            (CursorPos::new(copy.node, rope.tree[copy.node].piece.len()), pos)
        } else if pos.offset_in_node == rope.tree[pos.node].piece.len() {
            if !copy.next_piece(rope) {
                return None;
            }
            (pos, copy)
        } else {
            return None;
        };
        let (l, r) = rope.tree.get2_mut(left.node, right.node);
        if mergeable(&l.piece, &r.piece) {
            let r = right.delete(rope).1;
            left.insert(rope, r.into());
        }
        Some(left)
    }

    /// Deletes `len` (in base units) from the current cursor
    ///
    /// Returns a new cursor to the current position if the rope
    /// is not empty after the deletion.
    pub fn delete_len(mut self, rope: &mut Rope<T>, len: usize) -> Option<Self> {
        if len == 0 {
            return Some(self);
        }

        if self.offset_in_node != 0
            && self.offset_in_node == rope.tree[self.node].piece.len()
            && !self.next_piece(rope) {
            return Some(self);
        }
        let start = self;
        let mut end = start.clone();
        let end = if end.navigate::<BaseMetric>(&rope.tree, len as isize) {
            end
        } else if let Some(root) = rope.tree.root() {
            let end = rope.tree.edge(root, RIGHT);
            CursorPos::new(end, rope.tree[end].piece.len())
        } else {
            return Some(start);
        };

        if start.node == end.node {
            if start.offset_in_node == 0 && end.offset_in_node == rope.tree[start.node].piece.len() {
                return start.delete(rope).0;
            }
            let summary = rope.tree[start.node].piece.delete_range(
                &mut rope.context, start.offset_in_node, end.offset_in_node,
            );
            match summary {
                DeleteResult::Updated(deleted) => {
                    // This fast path doesn't call recompute_metadata,
                    // and we need to adjust self.sum manually.
                    let delta = &deleted.negate();
                    rope.sum.add_assign(delta);
                    rope.tree.update_metadata(start.node, delta);
                }
                DeleteResult::TailSplit { mut deleted, split } => {
                    rope.sum.sub_assign(&deleted);

                    let next: Insertion<T> = split.into();
                    deleted.add_assign(&next.1);
                    rope.tree.update_metadata(start.node, &deleted.negate());
                    rope.rb_insert(Some(start.node), next.0, RIGHT);
                }
            }
            return Some(start);
        }

        let mut del_nodes = vec![];
        let del_part =
            |this: &mut Rope<T>, node: SafeRef, range: Range<usize>| {
                let piece = &mut this.tree[node].piece;
                let summary = piece.delete_range(
                    &mut this.context, range.start, if range.end == usize::MAX {
                        piece.len()
                    } else {
                        range.end
                    },
                );
                let DeleteResult::Updated(deleted) = summary else { unreachable!() };
                if deleted != T::S::identity() {
                    this.tree.update_metadata(node, &deleted.negate());
                }
            };

        let mut valid: Option<Self> = if start.offset_in_node == 0 {
            let prev = start.to_prev_piece(&rope.tree);
            del_nodes.push(start.node);
            prev
        } else {
            del_part(rope, start.node, start.offset_in_node..usize::MAX);
            Some(start.clone())
        };

        valid = valid.or(if end.offset_in_node == rope.tree[end.node].piece.len() {
            let next = end.to_next_piece(&rope.tree);
            del_nodes.push(end.node);
            next
        } else {
            del_part(rope, end.node, 0..end.offset_in_node);
            Some(end.clone())
        });

        let mut del_i = rope.tree.next(start.node, RIGHT);
        while let Some(del_i_) = del_i && del_i_ != end.node {
            del_nodes.push(del_i_);
            del_i = rope.tree.next(del_i_, RIGHT);
        }
        rope.delete_nodes(del_nodes);

        rope.recompute_metadata();

        valid
    }

    /// Navigates to the next piece, resetting in-piece offset to 0
    ///
    /// No-op if the cursor is already at the end of the rope
    pub fn next_piece(&mut self, rope: &Rope<T>) -> bool {
        self.tree_next_piece(&rope.tree)
    }
    /// Navigates to the previous piece, resetting in-piece offset to 0
    ///
    /// No-op if the cursor is already at the start of the rope
    pub fn prev_piece(&mut self, rope: &Rope<T>) -> bool {
        self.tree_prev_piece(&rope.tree)
    }

    /// Inserts a piece at the cursor position
    pub fn insert(&mut self, rope: &mut Rope<T>, value: Insertion<T>) {
        rope.sum.add_assign(&value.1);
        let result = self.insert_1(rope, value);
        self.insert_2_insert(rope, result);
    }

    /// Inserts a piece at the cursor position and merges it with adjacent pieces if possible
    pub fn insert_merging(&mut self, rope: &mut Rope<T>, value: Insertion<T>) {
        rope.sum.add_assign(&value.1);
        let result = self.insert_1(rope, value);
        self.insert_2_merging(rope, result);
    }

    /// First step of an insertion operation: try insertion
    fn insert_1(&mut self, rope: &mut Rope<T>, value: Insertion<T>) -> SplitResult<T> {
        let obj = &mut rope.tree[self.node];
        let mut summary = value.1;
        let result = obj.piece.insert_or_split(&mut rope.context, value, self.offset_in_node);
        match &result {
            SplitResult::Merged => self.node_update(rope, &summary),
            SplitResult::MiddleSplit(mid, tail) => {
                summary.sub_assign(&mid.summarize());
                summary.sub_assign(&tail.summarize());
                self.node_update(rope, &summary);
            }
            SplitResult::HeadSplit(value) => {
                let new = value.summarize();
                if new != summary {
                    summary.sub_assign(&value.summarize());
                    self.node_update(rope, &summary);
                }
            }
            SplitResult::TailSplit(value) => {
                let new = value.summarize();
                if new != summary {
                    summary.sub_assign(&value.summarize());
                    self.node_update(rope, &summary);
                }
            }
        }
        result
    }
    /// The second step of an insertion operation: handle splits
    fn insert_2_insert(&mut self, rope: &mut Rope<T>, value: SplitResult<T>) {
        match value {
            SplitResult::Merged => {}
            SplitResult::HeadSplit(head) => {
                rope.rb_insert(Some(self.node), head, LEFT);
            }
            SplitResult::TailSplit(tail) => {
                let len = tail.len();
                self.node = rope.rb_insert(Some(self.node), tail, RIGHT);
                self.offset_in_node = len;
            }
            SplitResult::MiddleSplit(mid, tail) => {
                let node = rope.rb_insert(Some(self.node), mid, RIGHT);
                let len = tail.len();
                self.node = rope.rb_insert(Some(node), tail, RIGHT);
                self.offset_in_node = len;
            }
        }
    }
    /// Similar to [Self::insert_2_insert], but tries to merge the splits with
    /// the other end if the cursor is at the start/end of a piece.
    fn insert_2_merging(&mut self, rope: &mut Rope<T>, value: SplitResult<T>) {
        let tail = match value {
            SplitResult::Merged => { return }
            SplitResult::HeadSplit(head) => {
                let mut prev = self.clone();
                if prev.prev_piece(rope) {
                    prev.offset_in_node = rope.tree[self.node].piece.len();
                    let result = prev.insert_1(rope, head.into());
                    prev.insert_2_insert(rope, result);
                } else {
                    rope.rb_insert(Some(self.node), head, LEFT);
                }
                return;
            }
            SplitResult::TailSplit(tail) => tail,
            SplitResult::MiddleSplit(mid, tail) => {
                self.node = rope.rb_insert(Some(self.node), mid, RIGHT);
                tail
            }
        };

        if self.next_piece(rope) {
            let result = self.insert_1(rope, tail.into());
            self.insert_2_insert(rope, result);
        } else {
            let len = tail.len();
            self.node = rope.rb_insert(Some(self.node), tail, RIGHT);
            self.offset_in_node = len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::tests::validate_with_cursor;
    use crate::rb_base::SENTINEL;
    use crate::roperig_test::Alphabet;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::num::NonZero;

    #[test]
    fn test_insert() {
        let mut rope: Rope<Alphabet> = Rope::default();
        rope.insert_merging(0, "aaa".into());
        assert_eq!(rope.tree.slab(1).piece, "aaa".into());
        rope.insert_merging(1, "bbb".into());
        assert_eq!(rope.tree.slab(1).piece, "a".into());
        assert_eq!(rope.tree.slab(2).piece, "bbb".into());
        assert_eq!(rope.tree.slab(3).piece, "aa".into());
    }

    #[test]
    fn test_merge() {
        fn assert_merge(ops: &[(&str, usize, usize)], result: &[Option<&str>]) {
            let mut rope: Rope<Alphabet> = Rope::default();
            for (inserted, offset, deletes) in ops {
                rope.delete_merging(*offset, *deletes, |a, b| {
                    a.is_mergeable(b)
                });
                rope.insert_merging(*offset, (*inserted).into());
            }
            for (i, node) in result.iter().enumerate() {
                let node: Option<Alphabet> = node.map(|n| n.into());
                assert_eq!(
                    node.as_ref(),
                    rope.tree.slab_get(i + 1).map(|n| &n.piece),
                );
            }
            assert!(rope.tree.slab_len() <= result.len() + 1);
        }
        // insert merge
        assert_merge(
            &[("<<<", 0, 0), (">>>", 3, 0), ("<<<", 3, 0)],
            &[Some("<<<<<<"), Some(">>>")],
        );
        assert_merge(
            &[("<<<", 0, 0), (">>>", 3, 0), (">>>", 3, 0)],
            &[Some("<<<"), Some(">>>>>>")],
        );
        // delete merge
        assert_merge(
            &[("<<<", 0, 0), (">>>", 3, 0), ("<<<", 6, 0), ("", 3, 3)],
            &[Some("<<<<<<")],
        );
        assert_merge(
            &[("<<<<<<", 0, 0), (">>>", 3, 0), ("", 3, 3)],
            &[Some("<<<<<<")],
        );
    }

    fn assert_pos(pos: Option<PiecePosition<Alphabet>>, s: &str, offset: usize) {
        assert!(pos.is_some());
        let pos = pos.unwrap();
        let expect: Alphabet = s.into();
        assert_eq!((&expect, offset), (pos.piece, pos.offset_in_piece));
    }

    #[test]
    fn test_basic_insert() {
        let mut rb: Rope<Alphabet> = Rope::default();

        rb.insert_merging(0, "1".repeat(5).into());
        rb.is_valid(); // will panic if it must
        rb.insert_merging(5, "2".into());
        rb.is_valid(); // will panic if it must
        rb.insert_merging(6, "3".into());
        rb.is_valid(); // will panic if it must

        assert_pos(rb.search(4), "11111", 4);
        assert_pos(rb.search(5), "11111", 5);
        assert_pos(rb.search(6), "2", 1);
        assert_pos(rb.search(7), "3", 1);

        rb.is_valid(); // will panic if it must
        validate_with_cursor(&rb);
    }

    fn gather(rope: &Rope<Alphabet>) -> String {
        substring(rope, 0, rope.sum.len())
    }
    fn substring(rope: &Rope<Alphabet>, start: usize, end: usize) -> String {
        let mut s = String::with_capacity(end - start);
        rope.for_range::<BaseMetric>(start..end, |_, piece, range| {
            if !range.is_empty() {
                s.push_str(&piece.c().unwrap().to_string().repeat(range.len()));
            }
            true
        });
        s
    }

    #[test]
    fn test_basic_rotation() {
        let mut r: Rope<Alphabet> = Rope::default();

        r.insert_merging(0, "x".repeat(4).into()); // x
        r.insert_merging(0, "a".into()); // alpha
        r.insert_merging(5, "y".into()); // y
        r.insert_merging(5, "bb".into()); // beta
        r.insert_merging(8, "g".into()); // gamma

        /*
         *      x
         *     / \
         *    /   y
         *   a   / \
         *      b   g
         */

        assert_eq!("axxxxbbyg", gather(&r));

        let rb = &r.tree;
        assert_eq!(rb.slab(1).piece, "xxxx".into());
        assert_eq!(rb.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab(2).piece, "a".into());
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.slab(3).piece, "y".into());
        assert_eq!(rb.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab(4).piece, "bb".into());
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.slab(5).piece, "g".into());
        assert_eq!(rb.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(5).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(5).rb.children[1], SENTINEL);

        r.tree.rotate(NonZero::new(1), 0); // left-rotate x
        assert_eq!("axxxxbbyg", gather(&r));
        let rb = &r.tree;

        /*
         *      y
         *     / \
         *    x   g
         *   / \
         *  a   b
         */

        // slab entries should be the same, but their links should reflect the new tree topology

        assert_eq!(rb.slab(1).piece, "xxxx".into());
        assert_eq!(rb.slab(2).piece, "a".into());
        assert_eq!(rb.slab(1).rb.parent, NonZero::new(3)); // x's new parent is y
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(1)); // y's left child is x
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right child is gamma
        assert_eq!(rb.slab(5).piece, "g".into());
        assert_eq!(rb.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left child is alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(4)); // x's right child is beta
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1)); // alpha's parent is x
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(1)); // beta's parent is x

        r.tree.rotate(NonZero::new(3), 1); // right-rotate y brings our tree back to the original
        assert_eq!("axxxxbbyg", gather(&r));
        let rb = &r.tree;

        assert_eq!(rb.slab(1).piece, "xxxx".into());
        assert_eq!(rb.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab(2).piece, "a".into());
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.slab(3).piece, "y".into());
        assert_eq!(rb.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab(4).piece, "bb".into());
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.slab(5).piece, "g".into());
        assert_eq!(rb.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(5).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(5).rb.children[1], SENTINEL);
    }

    #[test]
    fn test_clear() {
        let mut rb = Rope::<Alphabet>::default();
        rb.insert(0, "111".into());
        rb.insert(3, "222".into());
        rb.insert(6, "333".into());
        rb.delete(0, 9);
        rb.is_valid();
        assert_eq!("", gather(&rb));
        assert_eq!(0, rb.sum);
        rb.insert(0, "111".into());
        rb.insert(3, "222".into());
        assert_eq!("111222", gather(&rb));
    }

    #[test]
    fn test_delete() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..1000 {
            let mut expected = String::default();
            let mut rb = Rope::<Alphabet>::default();
            for c in '0'..='9' {
                let s = c.to_string().repeat(if c == '9' { 5 } else { 10 });
                let at = expected.len() / 2;
                expected.insert_str(at, &s);
                rb.insert(at, s.into());
            }
            assert_eq!(expected, gather(&rb));
            let from = rng.random_range(0..=(expected.len()/2));
            let to = rng.random_range((expected.len()/2)..=expected.len());
            rb.delete_merging(from, to - from, |a, b| a.is_mergeable(b));
            expected.drain(from..to);
            assert_eq!(expected.len(), rb.len(), "{}: from: {}, to: {}", i, from, to);
            assert_eq!(expected, gather(&rb));
        }
    }

    #[test]
    fn test_many_insert() {
        let mut rb: Rope<Alphabet> = Rope::default();
        let mut expected = String::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..10000 {
            let char: Alphabet = rng.random_range('a'..='z').into();
            let pos = if expected.is_empty() { 0 } else { rng.random_range(0..expected.len()) };
            expected.insert(pos, char.c().unwrap());
            rb.insert_merging(pos, char);
            let (mut start, mut end) = (
                rng.random_range(0..expected.len()),
                rng.random_range(0..expected.len()),
            );
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            assert_eq!(expected.len(), rb.sum.len());
            assert_eq!(expected[start..end], substring(&rb, start, end));
            validate_with_cursor(&rb);
        }

        rb.is_valid(); // will panic if it must
    }

    #[test]
    fn test_many_insert_some_delete() {
        let mut rb: Rope<Alphabet> = Rope::default();

        for i in 0..(500000 / 26 * 26) {
            rb.insert_merging(rb.sum.len(), char::from(b'a' + (i % 26) as u8).into());
            rb.insert_merging(0, char::from(b'z' - (i % 26) as u8).into());
        }
        assert_eq!(
            ('a'..='z').collect::<String>().repeat(500000 / 26 * 2),
            gather(&rb),
        );

        fn assert_alpha_off(rb: &Rope<Alphabet>, at: usize, deleted: usize) {
            assert_eq!(500000 / 26 * 26 * 2 - deleted, rb.sum.len());
            let char = ((at + deleted) % 26) as u8 + b'a';
            assert_pos(rb.search(at + 1), str::from_utf8(&[char]).unwrap(), 1);
            validate_with_cursor(rb);
        }
        fn assert_alphabet(rb: &Rope<Alphabet>, at: usize) {
            assert_alpha_off(rb, at, 0);
        }

        assert_alphabet(&rb, 5);
        assert_alphabet(&rb, 50);
        assert_alphabet(&rb, 500);
        assert_alphabet(&rb, 5000);
        assert_alphabet(&rb, 50000);
        assert_alphabet(&rb, 500000);

        rb.is_valid(); // will panic if it must
        assert_alpha_off(&rb, 5, 0);
        
        let delete_merging = |rb: &mut Rope<Alphabet>, from: usize, len: usize| {
            rb.delete_merging(from, len, |a, b| {
                a.is_mergeable(b)
            });
        };

        delete_merging(&mut rb, 5, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 5, 1);
        assert_alpha_off(&rb, 5, 2);

        delete_merging(&mut rb, 50, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50, 1);
        assert_alpha_off(&rb, 50, 4);

        delete_merging(&mut rb, 50, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50, 1);
        assert_alpha_off(&rb, 500, 6);

        delete_merging(&mut rb, 500, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 500, 1);
        assert_alpha_off(&rb, 5000, 8);

        delete_merging(&mut rb, 5000, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 5000, 1);
        assert_alpha_off(&rb, 50000, 10);

        delete_merging(&mut rb, 50000, 1);
        rb.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50000, 1);
        assert_alpha_off(&rb, 500000, 12);
    }
}

