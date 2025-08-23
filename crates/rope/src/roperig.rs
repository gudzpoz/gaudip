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

use std::collections::VecDeque;
use std::ops::Range;
use crate::metrics::{BaseMetric, Cursor, Metric};
use crate::piece::{RopePiece, SplitResult, Sum, Summable};
use crate::rb_base::{Node, RbSlab, Ref, LEFT, RIGHT};

/// The rope implementation
pub struct Rope<T: RopePiece> {
    tree: RbSlab<T>,
    sum: T::S,
    context: T::Context,
}
impl<T: RopePiece> Default for Rope<T> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

pub(crate) struct NodePosition {
    pub node: Ref,
    pub in_node_offset: usize,
}
/// Information about a particular position in tree
pub struct PiecePosition<'a, T> {
    /// The piece that the position lies in
    pub piece: &'a T,
    /// The relative offset of the position
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

    /// Batch insert multiple values at consecutive nodes
    pub fn init(&mut self, mut value: VecDeque<T>) {
        assert!(self.sum.len() == 0 && !value.is_empty());
        let mut node = self.rb_insert(None, value.pop_front().unwrap(), LEFT);
        while let Some(next) = value.pop_front() {
            node = self.rb_insert(Some(node), next, RIGHT);
        }
    }

    /// Insert a node with `value` at `offset`
    pub fn insert(&mut self, offset: usize, value: T) {
        let summary = value.summarize();
        self.sum.add_assign(&summary);
        if self.tree.root.is_sentinel() {
            assert_eq!(offset, 0);
            self.rb_insert(None, value, LEFT);
        } else {
            let NodePosition {
                node, in_node_offset: remainder, ..
            } = self.node_at(offset).unwrap();
            self.insert_at(node, remainder, value);
        }
    }

    fn insert_at(&mut self, node: Ref, remainder: usize, value: T) {
        self.insert_at_around(node, remainder, value, true);
    }
    fn insert_at_around(&mut self, node: Ref, remainder: usize, value: T, retry: bool) -> Ref {
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

    fn try_other_edge(&mut self, node: Ref, retry: bool, value: T, dir: usize) -> Ref {
        if retry {
            let other = self.tree.next(node, dir);
            if !other.is_sentinel() {
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
                &mut self.context, start.in_node_offset, end.in_node_offset,
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
            |this: &mut Self, node: Ref, range: Range<usize>| -> bool {
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
            self, start.node, start.in_node_offset..usize::MAX,
        ) {
            self.tree.next(start.node, LEFT)
        } else {
            start.node
        };

        let merge_end = if del_or_keep(self, end.node, 0..end.in_node_offset) {
            self.tree.next(end.node, RIGHT)
        } else {
            end.node
        };

        let mut del_i = self.tree.next(start.node, RIGHT);
        while !del_i.is_sentinel() && del_i != end.node {
            del_or_keep(self, del_i, 0..usize::MAX);
            del_i = self.tree.next(del_i, RIGHT);
        }
        self.delete_nodes(del_nodes);

        let mut merge_i = merge_head;
        while !merge_head.is_sentinel() && merge_i != merge_end {
            let next = self.tree.next(merge_i, RIGHT);
            if next.is_sentinel() {
                break;
            }
            if self.tree[merge_head].piece.must_try_merging(&mut self.context, &self.tree[next].piece) {
                let tail = self.tree.delete(next);
                self.insert_at(merge_head, self.tree[merge_head].piece.summarize().len(), tail);
            } else {
                merge_i = next;
            }
            if next == merge_end {
                break;
            }
        }

        self.recompute_metadata();
    }

    /// Locate an offset in the current tree
    pub fn search(&self, offset: usize) -> Option<PiecePosition<T>> {
        if let Some(NodePosition { node, in_node_offset }) = self.node_at(offset) {
            Some(PiecePosition { piece: &self.tree[node].piece, offset_in_piece: in_node_offset })
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
    pub(crate) fn node_at(&self, offset: usize) -> Option<NodePosition> {
        self.node_at_metric::<BaseMetric>(offset)
    }

    pub(crate) fn node_at_metric<M: Metric<T>>(&self, offset: usize) -> Option<NodePosition> {
        let mut x = self.tree.root;
        if offset == 0 {
            if self.tree.root.is_sentinel() {
                return None;
            }
            x = self.tree.edge(x, LEFT);
            Some(NodePosition { node: x, in_node_offset: 0 })
        } else {
            rel_node_at_metric::<T, M>(&self.tree, x, offset)
        }
    }

    /// Get a cursor into the specified offset with metric `M`
    pub fn cursor<M: Metric<T>>(&self, offset: usize) -> Cursor<T, M> {
        let pos = self.node_at_metric::<M>(offset);
        Cursor::<T, M>::new(&self.tree, pos.unwrap())
    }

    fn rb_insert(&mut self, node: Option<Ref>, piece: T, dir: usize) -> Ref {
        let z = self.tree.insert(Node::new(piece));
        match node {
            None => {
                debug_assert!(self.tree.root.is_sentinel());
                self.tree.root = z;
                self.tree[z].red = false;
            }
            Some(node) => {
                debug_assert!(!node.is_sentinel());
                let n = &mut self.tree[node];
                let n_child = n.children[dir];
                if n_child.is_sentinel() {
                    n.children[dir] = z;
                    self.tree[z].parent = node;
                } else {
                    let prev = self.tree.edge(n_child, dir ^ 1);
                    self.tree[prev].children[dir ^ 1] = z;
                    self.tree[z].parent = prev;
                }
            }
        }
        self.tree.fix_insert(z);
        z
    }

    fn delete_nodes(&mut self, nodes: Vec<Ref>) {
        for node in nodes {
            self.tree.delete(node);
        }
    }

    fn recompute_metadata(&mut self) {
        self.sum = self.tree.calculate_sum(self.tree.root);
    }
    
    #[cfg(test)]
    fn is_valid(&self) {
        let sum = self.tree.is_valid();
        assert!(sum == self.sum);
    }
}

pub(crate) fn rel_node_at_metric<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, mut x: Ref, mut offset: usize,
) -> Option<NodePosition> {
    while !x.is_sentinel() {
        let n = &tree[x];
        let left_len = M::measure(&n.left_sum);
        if left_len >= offset {
            x = n.children[0];
        } else {
            let n_len = M::measure(&n.piece.summarize());
            let pre_len = left_len + n_len;
            if pre_len >= offset {
                offset -= left_len;
                debug_assert!((offset == 0) == (n_len == 0));
                return Some(NodePosition {
                    node: x,
                    in_node_offset: offset,
                });
            } else {
                offset -= pre_len;
                x = n.children[1];
            }
        }
    }
    None
}

pub(crate) fn rel_node_at<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, from: NodePosition, offset: usize, dir: usize,
) -> Option<NodePosition> {
    let mut x = from.node;
    if dir == LEFT && offset <= from.in_node_offset {
        return Some(NodePosition { node: x, in_node_offset: from.in_node_offset - offset });
    }
    let n = &tree[x];
    let len = n.piece.summarize();
    let remaining = M::measure(&len) - from.in_node_offset - 1;
    if dir == RIGHT && offset <= remaining {
        return Some(NodePosition { node: x, in_node_offset: from.in_node_offset + offset });
    }

    let mut offset_i = M::measure(&n.left_sum) as isize
        + from.in_node_offset as isize
        + if dir == LEFT { -(offset as isize) } else { offset as isize };
    while !x.is_sentinel() {
        let n = &tree[x];
        if 0 <= offset_i {
            let subtree = rel_node_at_metric::<T, M>(tree, x, offset_i as usize);
            if subtree.is_some() {
                return subtree;
            }
        }

        let p = n.parent;
        let pn = &tree[p];
        if pn.children[1] == x {
            offset_i += M::measure(&pn.left_sum) as isize;
            offset_i += M::measure(&pn.piece.summarize()) as isize;
        }
        x = p;
    }
    None
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use crate::piece::Summable;
    use crate::rb_base::SENTINEL;
    use super::*;

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
        assert_eq!("aaa", rope.tree.slab[1].piece);
        rope.insert(1, "bbb".to_string());
        assert_eq!("a", rope.tree.slab[1].piece);
        assert_eq!("bbb", rope.tree.slab[2].piece);
        assert_eq!("aa", rope.tree.slab[3].piece);
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
                    &rope.tree.slab.get(i + 1).map(|n| n.piece.as_str()),
                );
            }
            assert!(rope.tree.slab.len() <= result.len() + 1);
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
    }

    fn gather(rope: &Rope<String>) -> String {
        substring(rope, 0, rope.sum.len())
    }
    fn substring(rope: &Rope<String>, start: usize, end: usize) -> String {
        let start = rope.node_at(start).unwrap();
        let end = rope.node_at(end).unwrap();
        let mut i = start.node;
        let mut s = String::default();
        while !i.is_sentinel() {
            let offset = if i == start.node {
                start.in_node_offset
            } else {
                0
            };
            let piece = &rope.tree[i].piece;
            let end_off = if i == end.node {
                end.in_node_offset
            } else {
                piece.len()
            };
            s.push_str(&piece[offset..end_off]);
            if i == end.node {
                break;
            }
            i = rope.tree.next(i, RIGHT);
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
        assert_eq!(rb.slab[1].piece, "xxxx");
        assert_eq!(rb.slab[1].parent, SENTINEL);
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], Ref(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].piece, "a");
        assert_eq!(rb.slab[2].parent, Ref(1));
        assert_eq!(rb.slab[2].children[0], SENTINEL);
        assert_eq!(rb.slab[2].children[1], SENTINEL);

        assert_eq!(rb.slab[3].piece, "y");
        assert_eq!(rb.slab[3].parent, Ref(1));
        assert_eq!(rb.slab[3].children[0], Ref(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].piece, "bb");
        assert_eq!(rb.slab[4].parent, Ref(3));
        assert_eq!(rb.slab[4].children[0], SENTINEL);
        assert_eq!(rb.slab[4].children[1], SENTINEL);
        assert_eq!(rb.slab[5].piece, "g");
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[5].children[0], SENTINEL);
        assert_eq!(rb.slab[5].children[1], SENTINEL);

        r.tree.rotate(Ref(1), 0); // left-rotate x
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

        assert_eq!(rb.slab[1].piece, "xxxx");
        assert_eq!(rb.slab[2].piece, "a");
        assert_eq!(rb.slab[1].parent, Ref(3)); // x's new parent is y
        assert_eq!(rb.slab[3].children[0], Ref(1)); // y's left child is x
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right child is gamma
        assert_eq!(rb.slab[5].piece, "g");
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left child is alpha
        assert_eq!(rb.slab[1].children[1], Ref(4)); // x's right child is beta
        assert_eq!(rb.slab[2].parent, Ref(1)); // alpha's parent is x
        assert_eq!(rb.slab[4].parent, Ref(1)); // beta's parent is x

        r.tree.rotate(Ref(3), 1); // right-rotate y brings our tree back to the original
        assert_eq!("axxxxbbyg", gather(&r));
        let rb = &r.tree;

        assert_eq!(rb.slab[1].piece, "xxxx");
        assert_eq!(rb.slab[1].parent, SENTINEL);
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], Ref(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].piece, "a");
        assert_eq!(rb.slab[2].parent, Ref(1));
        assert_eq!(rb.slab[2].children[0], SENTINEL);
        assert_eq!(rb.slab[2].children[1], SENTINEL);

        assert_eq!(rb.slab[3].piece, "y");
        assert_eq!(rb.slab[3].parent, Ref(1));
        assert_eq!(rb.slab[3].children[0], Ref(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].piece, "bb");
        assert_eq!(rb.slab[4].parent, Ref(3));
        assert_eq!(rb.slab[4].children[0], SENTINEL);
        assert_eq!(rb.slab[4].children[1], SENTINEL);
        assert_eq!(rb.slab[5].piece, "g");
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[5].children[0], SENTINEL);
        assert_eq!(rb.slab[5].children[1], SENTINEL);
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

