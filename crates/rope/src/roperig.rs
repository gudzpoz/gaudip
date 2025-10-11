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
use crate::piece::{RopePiece, SplitResult, Sum};
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
    pub fn context(&mut self) -> &mut T::Context {
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
        self.cursor::<M>(measurement).map(|c| c.offset::<N>())
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
        self.insert_at(self.node_at_metric::<BaseMetric>(offset), value);
    }
    pub(crate) fn insert_at(&mut self, pos: Option<CursorPos<T>>, value: T) {
        let summary = value.summarize();
        self.sum.add_assign(&summary);
        if let Some(pos) = pos {
            self.insert_at_around(pos.node, pos.offset_in_node, value, true);
        } else {
            self.rb_insert(None, value, LEFT);
        }
    }

    fn insert_at_around(&mut self, node: SafeRef, remainder: usize, value: T, retry: bool) -> SafeRef {
        let obj = &mut self.tree[node];
        let mut summary = value.summarize();
        match obj.piece.insert_or_split(&mut self.context, value, remainder) {
            SplitResult::Merged => {
                self.tree.update_metadata(node, &summary);
                node
            }
            SplitResult::MiddleSplit(mid, tail) => {
                summary.sub_assign(&mid.summarize());
                summary.sub_assign(&tail.summarize());
                self.tree.update_metadata(node, &summary);
                let node = self.rb_insert(Some(node), mid, RIGHT);
                self.try_other_edge(node, retry, tail, RIGHT)
            }
            SplitResult::HeadSplit(value) => {
                if value.summarize() != summary {
                    summary.sub_assign(&value.summarize());
                    self.tree.update_metadata(node, &summary);
                }
                self.try_other_edge(node, retry, value, LEFT)
            }
            SplitResult::TailSplit(value) => {
                if value.summarize() != summary {
                    summary.sub_assign(&value.summarize());
                    self.tree.update_metadata(node, &summary);
                }
                self.try_other_edge(node, retry, value, RIGHT)
            }
        }
    }
    fn try_other_edge(&mut self, node: SafeRef, retry: bool, value: T, dir: usize) -> SafeRef {
        if retry {
            let other = self.tree.next(node, dir);
            if let Some(other) = other {
                let at = if dir == LEFT {
                    self.tree[other].piece.summarize().len()
                } else {
                    0
                };
                return self.insert_at_around(other, at, value, false);
            }
        }
        self.rb_insert(Some(node), value, dir)
    }

    /// Delete a substring (or sub-rope?) from the rope
    pub fn delete(&mut self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let start = self.node_at(offset).unwrap();
        let end = self.node_at(offset + len).unwrap();
        if start.node == end.node {
            let summary = self.tree[start.node].piece.delete_range(
                &mut self.context, start.offset_in_node, end.offset_in_node,
            );
            if let Some(summary) = summary {
                // This fast path doesn't call recompute_metadata,
                // and we need to adjust self.sum manually.
                self.sum.sub_assign(&summary);
                self.tree.update_metadata(start.node, &summary.negate());
            } else {
                self.tree.delete(start.node);
            }
            return;
        }

        let mut del_nodes = vec![];
        let mut del_or_keep =
            |this: &mut Self, node: SafeRef, range: Range<usize>| -> bool {
                let piece = &mut this.tree[node].piece;
                let summary = piece.delete_range(
                    &mut this.context, range.start, if range.end == usize::MAX {
                        piece.summarize().len()
                    } else {
                        range.end
                    },
                );
                if let Some(summary) = summary {
                    if summary != T::S::identity() {
                        this.tree.update_metadata(node, &summary.negate());
                    }
                    false
                } else {
                    del_nodes.push(node);
                    true
                }
            };

        let merge_head = if del_or_keep(
            self, start.node, start.offset_in_node..usize::MAX,
        ) {
            self.tree.next(start.node, LEFT)
        } else {
            Some(start.node)
        };

        let merge_end = if del_or_keep(self, end.node, 0..end.offset_in_node) {
            self.tree.next(end.node, RIGHT)
        } else {
            Some(end.node)
        };

        let mut del_i = self.tree.next(start.node, RIGHT);
        while let Some(del_i_) = del_i && del_i_ != end.node {
            del_or_keep(self, del_i_, 0..usize::MAX);
            del_i = self.tree.next(del_i_, RIGHT);
        }
        self.delete_nodes(del_nodes);

        if let Some(merge_head) = merge_head {
            let mut merge_i = merge_head;
            loop {
                let Some(next) = self.tree.next(merge_i, RIGHT) else { break; };
                if self.tree[merge_head].piece.must_try_merging(&mut self.context, &self.tree[next].piece) {
                    let tail = self.tree.delete(next);
                    self.insert_at_around(merge_head, self.tree[merge_head].piece.summarize().len(), tail, true);
                } else {
                    merge_i = next;
                }
                if Some(next) == merge_end {
                    break;
                }
            }
        }

        self.recompute_metadata();
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
    pub fn cursor<M: Metric<T>>(&self, offset: usize) -> Option<Cursor<'_, T>> {
        Self::tree_cursor::<M>(&self.tree, self, offset)
    }
    /// Get a cursor like [Self::cursor] without borrowing the whole struct
    fn tree_cursor<'a, M: Metric<T>>(tree: &'a RbSlab<T>, this: &Self, offset: usize) -> Option<Cursor<'a, T>> {
        let pos = this.node_at_metric::<M>(offset);
        pos.map(|c| c.cursor(tree))
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

    fn delete_nodes(&mut self, nodes: Vec<SafeRef>) {
        for node in nodes {
            self.tree.delete(node);
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

#[cfg(test)]
mod tests {
    use std::num::NonZero;
    use super::*;
    use crate::piece::Summable;
    use crate::rb_base::SENTINEL;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use crate::metrics::tests::validate_with_cursor;

    impl Sum for usize {
        fn len(&self) -> usize {
            *self
        }
        fn add_assign(&mut self, other: &Self) {
            *self = self.wrapping_add(*other);
        }
        fn sub_assign(&mut self, other: &Self) {
            *self = self.wrapping_sub(*other);
        }
        fn identity() -> Self {
            0
        }
    }
    impl Summable for String {
        type S = usize;
        fn summarize(&self) -> Self::S {
            self.len()
        }
    }
    impl RopePiece for String {
        type Context = ();
        fn must_try_merging(&self, _: &mut (), other: &Self) -> bool {
            other.is_empty() || self.is_empty() || self.as_bytes()[0] == other.as_bytes()[0]
        }
        fn insert_or_split(&mut self, _: &mut (), other: Self, offset: usize) -> SplitResult<Self> {
            if self.is_empty() {
                *self = other;
                SplitResult::Merged
            } else if other.is_empty() {
                SplitResult::Merged
            } else if offset == 0 {
                if other.as_bytes()[0] == self.as_bytes()[0] {
                    self.push_str(&other);
                    SplitResult::Merged
                } else {
                    SplitResult::HeadSplit(other)
                }
            } else if offset == self.len() {
                if other.as_bytes()[0] == self.as_bytes()[0] {
                    self.push_str(&other);
                    SplitResult::Merged
                } else {
                    SplitResult::TailSplit(other)
                }
            } else if other.as_bytes()[0] == self.as_bytes()[0] {
                self.push_str(&other);
                SplitResult::Merged
            } else {
                let tail = self.split_off(offset);
                SplitResult::MiddleSplit(other, tail)
            }
        }
        fn delete_range(&mut self, _: &mut (), from: usize, to: usize) -> Option<Self::S> {
            if from == 0 && to == self.len() {
                None
            } else {
                self.drain(from..to);
                Some(to - from)
            }
        }
    }

    #[test]
    fn test_insert() {
        let mut rope: Rope<String> = Rope::default();
        rope.insert(0, "aaa".to_string());
        assert_eq!("aaa", rope.tree.slab(1).piece);
        rope.insert(1, "bbb".to_string());
        assert_eq!("a", rope.tree.slab(1).piece);
        assert_eq!("bbb", rope.tree.slab(2).piece);
        assert_eq!("aa", rope.tree.slab(3).piece);
    }

    #[test]
    fn test_merge() {
        fn assert_merge(ops: &[(&str, usize, usize)], result: &[Option<&str>]) {
            let mut rope: Rope<String> = Rope::default();
            for (inserted, offset, deletes) in ops {
                rope.delete(*offset, *deletes);
                rope.insert(*offset, inserted.to_string());
            }
            for (i, node) in result.iter().enumerate() {
                assert_eq!(
                    node,
                    &rope.tree.slab_get(i + 1).map(|n| n.piece.as_str()),
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

    fn assert_pos(pos: Option<PiecePosition<String>>, s: &str, offset: usize) {
        assert!(pos.is_some());
        let pos = pos.unwrap();
        assert_eq!((s, offset), (pos.piece.as_str(), pos.offset_in_piece));
    }

    #[test]
    fn test_basic_insert() {
        let mut rb: Rope<String> = Rope::default();

        rb.insert(0, "1".repeat(5));
        rb.insert(5, "2".to_string());
        rb.insert(6, "3".to_string());

        assert_pos(rb.search(4), "11111", 4);
        assert_pos(rb.search(5), "11111", 5);
        assert_pos(rb.search(6), "2", 1);
        assert_pos(rb.search(7), "3", 1);

        rb.is_valid(); // will panic if it must
        validate_with_cursor(&rb);
    }

    fn gather(rope: &Rope<String>) -> String {
        substring(rope, 0, rope.sum.len())
    }
    fn substring(rope: &Rope<String>, start: usize, end: usize) -> String {
        let start = rope.node_at(start).unwrap();
        let end = rope.node_at(end).unwrap();
        let mut i = Some(start.node);
        let mut s = String::default();
        while let Some(idx) = i {
            let offset = if idx == start.node {
                start.offset_in_node
            } else {
                0
            };
            let piece = &rope.tree[idx].piece;
            let end_off = if idx == end.node {
                end.offset_in_node
            } else {
                piece.len()
            };
            s.push_str(&piece[offset..end_off]);
            if idx == end.node {
                break;
            }
            i = rope.tree.next(idx, RIGHT);
        }
        s
    }

    #[test]
    fn test_basic_rotation() {
        let mut r: Rope<String> = Rope::default();

        r.insert(0, "x".repeat(4)); // x
        r.insert(0, "a".to_string()); // alpha
        r.insert(5, "y".to_string()); // y
        r.insert(5, "bb".to_string()); // beta
        r.insert(8, "g".to_string()); // gamma

        /*
         *      x
         *     / \
         *    /   y
         *   a   / \
         *      b   g
         */

        assert_eq!("axxxxbbyg", gather(&r));

        let rb = &r.tree;
        assert_eq!(rb.slab(1).piece, "xxxx");
        assert_eq!(rb.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab(2).piece, "a");
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.slab(3).piece, "y");
        assert_eq!(rb.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab(4).piece, "bb");
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.slab(5).piece, "g");
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

        assert_eq!(rb.slab(1).piece, "xxxx");
        assert_eq!(rb.slab(2).piece, "a");
        assert_eq!(rb.slab(1).rb.parent, NonZero::new(3)); // x's new parent is y
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(1)); // y's left child is x
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right child is gamma
        assert_eq!(rb.slab(5).piece, "g");
        assert_eq!(rb.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left child is alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(4)); // x's right child is beta
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1)); // alpha's parent is x
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(1)); // beta's parent is x

        r.tree.rotate(NonZero::new(3), 1); // right-rotate y brings our tree back to the original
        assert_eq!("axxxxbbyg", gather(&r));
        let rb = &r.tree;

        assert_eq!(rb.slab(1).piece, "xxxx");
        assert_eq!(rb.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab(2).piece, "a");
        assert_eq!(rb.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.slab(3).piece, "y");
        assert_eq!(rb.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab(4).piece, "bb");
        assert_eq!(rb.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.slab(5).piece, "g");
        assert_eq!(rb.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.slab(5).rb.children[0], SENTINEL);
        assert_eq!(rb.slab(5).rb.children[1], SENTINEL);
    }

    #[test]
    fn test_many_insert() {
        let mut rb: Rope<String> = Rope::default();
        let mut expected = String::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..10000 {
            let char = rng.random_range('a'..='z').to_string();
            let pos = if expected.is_empty() { 0 } else { rng.random_range(0..expected.len()) };
            expected.insert_str(pos, &char);
            rb.insert(pos, char);
            let (mut start, mut end) = (
                rng.random_range(0..expected.len()),
            rng.random_range(0..expected.len()),
            );
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            assert_eq!(expected[start..end], substring(&rb, start, end));
            assert_eq!(expected.len(), rb.sum.len());
            validate_with_cursor(&rb);
        }

        rb.is_valid(); // will panic if it must
    }

    #[test]
    fn test_many_insert_some_delete() {
        let mut rb: Rope<String> = Rope::default();

        for i in 0..(500000 / 26 * 26) {
            rb.insert(rb.sum.len(), char::from(b'a' + (i % 26) as u8).to_string());
            rb.insert(0, char::from(b'z' - (i % 26) as u8).to_string());
        }
        assert_eq!(
            ('a'..='z').collect::<String>().repeat(500000 / 26 * 2),
            gather(&rb),
        );

        fn assert_alpha_off(rb: &Rope<String>, at: usize, deleted: usize) {
            assert_eq!(500000 / 26 * 26 * 2 - deleted, rb.sum.len());
            let char = ((at + deleted) % 26) as u8 + b'a';
            assert_pos(rb.search(at + 1), str::from_utf8(&[char]).unwrap(), 1);
            validate_with_cursor(rb);
        }
        fn assert_alphabet(rb: &Rope<String>, at: usize) {
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

        rb.delete(5, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(5, 1);
        assert_alpha_off(&rb, 5, 2);

        rb.delete(50, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(50, 1);
        assert_alpha_off(&rb, 50, 4);

        rb.delete(50, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(50, 1);
        assert_alpha_off(&rb, 500, 6);

        rb.delete(500, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(500, 1);
        assert_alpha_off(&rb, 5000, 8);

        rb.delete(5000, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(5000, 1);
        assert_alpha_off(&rb, 50000, 10);

        rb.delete(50000, 1);
        rb.is_valid(); // will panic if it must
        rb.delete(50000, 1);
        assert_alpha_off(&rb, 500000, 12);
    }
}

