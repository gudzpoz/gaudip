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

use crate::metrics::{BaseMetric, Metric};
use crate::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use crate::rb_base::{SafeRef, LEFT, RIGHT, SENTINEL};
#[cfg(test)] use crate::rb_base::RbSlab;
use crate::ropebase::{ConvertedPosition, PartialCursorPos, RopeBase};
use delegate::delegate;
use std::collections::Bound;
use std::ops::RangeBounds;

/// A rope implementation assuming all nodes are mergeable pieces
pub struct Rope<T: RopePiece> {
    /// The inner tree
    pub tree: RopeBase<T>,
    /// The context
    pub context: T::Context,
}
impl<T: RopePiece> Default for Rope<T> where T::Context: Default {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

/// Information about a particular position in tree
pub struct PiecePosition<'a, T: Summable> {
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
            tree: Default::default(),
            context,
        }
    }

    delegate! {
        to self.tree {
            /// Returns true if the tree is empty
            pub fn is_empty(&self) -> bool;
            /// Returns the length in [BaseMetric]
            pub fn base_len(&self) -> usize;
            /// Returns the length in the given metric
            pub fn len<M: Metric<T>>(&self) -> usize;
            /// Convert metrics
            pub fn convert_metrics<'a, From: Metric<T>, To: Metric<T>>(
                &'a self, measurement: usize
            ) -> ConvertedPosition<'a, T, To, From>;
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

    /// Insert a node with `value` at `offset` in base metric
    pub fn insert(&mut self, offset: usize, value: T) {
        let offset = offset.clamp(0, self.base_len());
        if let Some(pos) = self.cursor_at(offset) {
            pos.insert(self, value);
        } else {
            assert_eq!(offset, 0);
            self.tree.init(Some(value).into_iter());
        }
    }

    /// Delete a substring (or sub-rope?) from the rope
    pub fn delete<R: RangeBounds<usize>>(&mut self, range: R) {
        let start = match range.start_bound().cloned() {
            Bound::Included(i) => i,
            Bound::Excluded(i) => i + 1,
            Bound::Unbounded => 0,
        };
        let end = self.base_len().min(match range.end_bound().cloned() {
            Bound::Included(i) => i + 1,
            Bound::Excluded(i) => i,
            Bound::Unbounded => usize::MAX,
        });
        if start >= end {
            return;
        }
        if let Some(pos) = self.cursor_at(start) {
            pos.delete_len(self, end - start);
        }
    }

    /// Returns a [CursorPos] corresponding to the supplied offset
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
    pub fn cursor_at(&self, offset: usize) -> Option<CursorPos<T>> {
        self.tree.cursor_at::<BaseMetric>(offset)
    }

    #[cfg(test)]
    pub(crate) fn compact(&mut self) -> &RbSlab<T> {
        self.tree.tree.compact(|_, _, _| true);
        &self.tree.tree
    }
}

/// Cursor in [BaseMetric]
pub type CursorPos<T> = PartialCursorPos<T, BaseMetric>;
impl<T: RopePiece> CursorPos<T> {
    /// Get the offset of the cursor in [BaseMetric]
    pub fn absolute(&self, rope: &Rope<T>) -> usize {
        self.position(&rope.tree)
    }

    /// Get the piece and offset of the cursor in [BaseMetric]
    pub fn piece_position<'a>(&self, rope: &'a Rope<T>) -> PiecePosition<'a, T> {
        let piece = self.get(&rope.tree);
        PiecePosition { piece, offset_in_piece: self.offset().value }
    }

    /// Deletes `len` (in base units) from the current cursor
    ///
    /// Returns a new cursor to the current position if the rope
    /// is not empty after the deletion.
    pub fn delete_len(mut self, rope: &mut Rope<T>, len: usize) -> Option<Self> {
        if len == 0 {
            return Some(self);
        }

        let ctx = &mut rope.context;
        let rope = &mut rope.tree;

        if self.offset().value != 0 && self.offset().value == rope.tree[self.node].piece.len() {
            let Some(next) = self.next_piece(rope) else {
                return Some(self);
            };
            self = next;
        }

        let start = self;
        let end = if let Some(end) = start.navigate_dir(&rope.tree, len, RIGHT) {
            end
        } else if let Some(root) = rope.tree.root() {
            let end = rope.tree.edge(root, RIGHT);
            PartialCursorPos::new(end, rope.tree[end].piece.len())
        } else {
            return Some(start);
        };

        if start.node == end.node {
            if start.offset().value == 0 && end.offset().value == rope.tree[start.node].piece.len() {
                let cursor = start.nearby_piece(rope);
                start.delete(rope).notify_delete(ctx);
                return cursor;
            }
            let summary = rope.tree[start.node].piece.delete_range(
                ctx, start.offset().value..end.offset().value,
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

                    deleted.add_assign(&split.summarize());
                    rope.tree.update_metadata(start.node, &deleted.negate());
                    rope.rb_insert(Some(start.node), split, RIGHT);
                }
            }
            return Some(start);
        }

        fn del_part<T: RopePiece>(
            this: &mut RopeBase<T>, ctx: &mut T::Context, node: SafeRef,
            from: usize, to: Option<usize>,
        ) {
            let piece = &mut this.tree[node].piece;
            let summary = piece.delete_range(ctx, from..to.unwrap_or(piece.len()));
            let DeleteResult::Updated(deleted) = summary else { unreachable!() };
            if deleted != T::S::identity() {
                this.tree.update_metadata(node, &deleted.negate());
            }
        }

        let mut del_range = [SENTINEL, SENTINEL];

        let start_node = start.node;
        let mut valid: Option<Self> = if start.offset().value == 0 {
            del_range[0] = Some(start.node);
            start.prev_piece(rope)
        } else {
            let del = rope.tree.next(start.node, RIGHT);
            del_range[0] = if del == Some(end.node) { SENTINEL } else { del };
            del_part(rope, ctx, start.node, start.offset().value, None);
            Some(start)
        };

        valid = valid.or(if end.offset().value == rope.tree[end.node].piece.len() {
            del_range[1] = Some(end.node);
            end.next_piece(rope)
        } else {
            let del = rope.tree.next(end.node, LEFT);
            del_range[1] = if del == Some(start_node) { SENTINEL } else { del };
            del_part(rope, ctx, end.node, 0, Some(end.offset().value));
            Some(end)
        });

        match del_range {
            [Some(l), Some(r)] => {
                CursorPos::new(l, 0).delete_many_to(
                    rope,
                    CursorPos::new(r, 0),
                    |mut piece| piece.notify_delete(ctx),
                );
            }
            other => {
                for node in other.into_iter().flatten() {
                    rope.tree.delete(node).notify_delete(ctx);
                }
            }
        }

        rope.recompute_metadata();

        valid
    }

    /// Inserts a piece at the cursor position
    pub fn insert(&self, rope: &mut Rope<T>, value: T) -> Self {
        let mut pos = self.clone();
        while pos.offset().value == 0 {
            let Some(prev) = pos.prev_piece(&rope.tree) else {
                pos.insert_left(&mut rope.tree, value);
                return pos;
            };
            pos = prev;
        }
        debug_assert_ne!(0, pos.offset().value);

        rope.tree.sum.add_assign(&value.summarize());
        let obj = &mut rope.tree.tree[self.node];
        let mut summary = value.summarize();
        let result = obj.piece.insert_or_split(&mut rope.context, value, self.offset().value);
        match result {
            SplitResult::Merged => {
                self.node_update(&mut rope.tree, &summary)
            }
            SplitResult::MiddleSplit(mid, tail) => {
                summary.sub_assign(&mid.summarize());
                summary.sub_assign(&tail.summarize());
                self.node_update(&mut rope.tree, &summary);
                let node = rope.tree.rb_insert(Some(self.node), mid, RIGHT);
                let len = tail.summarize();
                Self::new(rope.tree.rb_insert(Some(node), tail, RIGHT), len.len())
            }
            SplitResult::TailSplit(tail) => {
                let new = tail.summarize();
                if new != summary {
                    summary.sub_assign(&tail.summarize());
                    self.node_update(&mut rope.tree, &summary);
                }
                let len = tail.summarize();
                Self::new(rope.tree.rb_insert(Some(self.node), tail, RIGHT), len.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rb_base::SENTINEL;
    use crate::roperig_test::Alphabet;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::num::NonZero;

    #[test]
    fn test_insert() {
        let mut rope: Rope<Alphabet> = Rope::default();
        rope.insert(0, "aaa".into());
        assert_eq!(rope.tree.tree.slab(1).piece, "aaa".into());
        rope.insert(1, "bbb".into());
        assert_eq!(rope.tree.tree.slab(1).piece, "a".into());
        assert_eq!(rope.tree.tree.slab(2).piece, "bbb".into());
        assert_eq!(rope.tree.tree.slab(3).piece, "aa".into());
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

        rb.insert(0, "1".repeat(5).into());
        rb.tree.is_valid(); // will panic if it must
        rb.insert(5, "2".into());
        rb.tree.is_valid(); // will panic if it must
        rb.insert(6, "3".into());
        rb.tree.is_valid(); // will panic if it must

        assert_pos(rb.cursor_at(4).map(|c| c.piece_position(&rb)), "11111", 4);
        assert_pos(rb.cursor_at(5).map(|c| c.piece_position(&rb)), "11111", 5);
        assert_pos(rb.cursor_at(6).map(|c| c.piece_position(&rb)), "2", 1);
        assert_pos(rb.cursor_at(7).map(|c| c.piece_position(&rb)), "3", 1);

        rb.tree.is_valid(); // will panic if it must
    }

    fn gather(rope: &Rope<Alphabet>) -> String {
        rope.substring(0, rope.tree.sum.len())
    }

    #[test]
    fn test_basic_rotation() {
        let mut r: Rope<Alphabet> = Rope::default();

        r.insert(0, "x".repeat(4).into()); // x
        r.insert(0, "a".into()); // alpha
        r.insert(5, "y".into()); // y
        r.insert(5, "bb".into()); // beta
        r.insert(8, "g".into()); // gamma

        /*
         *      x
         *     / \
         *    /   y
         *   a   / \
         *      b   g
         */

        assert_eq!("axxxxbbyg", gather(&r));

        let rb = &r.tree;
        assert_eq!(rb.tree.slab(1).piece, "xxxx".into());
        assert_eq!(rb.tree.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.tree.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.tree.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.tree.slab(2).piece, "a".into());
        assert_eq!(rb.tree.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.tree.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.tree.slab(3).piece, "y".into());
        assert_eq!(rb.tree.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.tree.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.tree.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.tree.slab(4).piece, "bb".into());
        assert_eq!(rb.tree.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.tree.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.tree.slab(5).piece, "g".into());
        assert_eq!(rb.tree.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.tree.slab(5).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(5).rb.children[1], SENTINEL);

        r.tree.tree.rotate(NonZero::new(1), 0); // left-rotate x
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

        assert_eq!(rb.tree.slab(1).piece, "xxxx".into());
        assert_eq!(rb.tree.slab(2).piece, "a".into());
        assert_eq!(rb.tree.slab(1).rb.parent, NonZero::new(3)); // x's new parent is y
        assert_eq!(rb.tree.slab(3).rb.children[0], NonZero::new(1)); // y's left child is x
        assert_eq!(rb.tree.slab(3).rb.children[1], NonZero::new(5)); // y's right child is gamma
        assert_eq!(rb.tree.slab(5).piece, "g".into());
        assert_eq!(rb.tree.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.tree.slab(1).rb.children[0], NonZero::new(2)); // x's left child is alpha
        assert_eq!(rb.tree.slab(1).rb.children[1], NonZero::new(4)); // x's right child is beta
        assert_eq!(rb.tree.slab(2).rb.parent, NonZero::new(1)); // alpha's parent is x
        assert_eq!(rb.tree.slab(4).rb.parent, NonZero::new(1)); // beta's parent is x

        r.tree.tree.rotate(NonZero::new(3), 1); // right-rotate y brings our tree back to the original
        assert_eq!("axxxxbbyg", gather(&r));
        let rb = &r.tree;

        assert_eq!(rb.tree.slab(1).piece, "xxxx".into());
        assert_eq!(rb.tree.slab(1).rb.parent, SENTINEL);
        assert_eq!(rb.tree.slab(1).rb.children[0], NonZero::new(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.tree.slab(1).rb.children[1], NonZero::new(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.tree.slab(2).piece, "a".into());
        assert_eq!(rb.tree.slab(2).rb.parent, NonZero::new(1));
        assert_eq!(rb.tree.slab(2).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(2).rb.children[1], SENTINEL);

        assert_eq!(rb.tree.slab(3).piece, "y".into());
        assert_eq!(rb.tree.slab(3).rb.parent, NonZero::new(1));
        assert_eq!(rb.tree.slab(3).rb.children[0], NonZero::new(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.tree.slab(3).rb.children[1], NonZero::new(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.tree.slab(4).piece, "bb".into());
        assert_eq!(rb.tree.slab(4).rb.parent, NonZero::new(3));
        assert_eq!(rb.tree.slab(4).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(4).rb.children[1], SENTINEL);
        assert_eq!(rb.tree.slab(5).piece, "g".into());
        assert_eq!(rb.tree.slab(5).rb.parent, NonZero::new(3));
        assert_eq!(rb.tree.slab(5).rb.children[0], SENTINEL);
        assert_eq!(rb.tree.slab(5).rb.children[1], SENTINEL);
    }

    #[test]
    fn test_clear() {
        let mut rb = Rope::<Alphabet>::default();
        rb.insert(0, "111".into());
        rb.insert(3, "222".into());
        rb.insert(6, "333".into());
        rb.delete(0..9);
        rb.tree.is_valid();
        assert_eq!("", gather(&rb));
        assert_eq!(0, rb.tree.sum);
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
            rb.delete(from..to);
            expected.drain(from..to);
            assert_eq!(expected.len(), rb.base_len(), "{}: from: {}, to: {}", i, from, to);
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
            rb.insert(pos, char);
            let (mut start, mut end) = (
                rng.random_range(0..expected.len()),
                rng.random_range(0..expected.len()),
            );
            if start > end {
                std::mem::swap(&mut start, &mut end);
            }
            assert_eq!(expected.len(), rb.tree.sum.len());
            assert_eq!(expected[start..end], rb.substring(start, end));
        }

        rb.tree.is_valid(); // will panic if it must
    }

    #[test]
    fn test_many_insert_some_delete() {
        let mut rb: Rope<Alphabet> = Rope::default();

        for i in 0..(500000 / 26 * 26) {
            rb.insert(rb.tree.base_len(), char::from(b'a' + (i % 26) as u8).into());
            rb.insert(0, char::from(b'z' - (i % 26) as u8).into());
        }
        assert_eq!(
            ('a'..='z').collect::<String>().repeat(500000 / 26 * 2),
            gather(&rb),
        );

        fn assert_alpha_off(rb: &Rope<Alphabet>, at: usize, deleted: usize) {
            assert_eq!(500000 / 26 * 26 * 2 - deleted, rb.tree.sum.len());
            let char = ((at + deleted) % 26) as u8 + b'a';
            assert_pos(
                rb.cursor_at(at + 1).map(|c| c.piece_position(rb)),
                str::from_utf8(&[char]).unwrap(), 1,
            );
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

        rb.tree.is_valid(); // will panic if it must
        assert_alpha_off(&rb, 5, 0);

        let delete_merging = |rb: &mut Rope<Alphabet>, from: usize, len: usize| {
            rb.delete(from..from + len);
        };

        delete_merging(&mut rb, 5, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 5, 1);
        assert_alpha_off(&rb, 5, 2);

        delete_merging(&mut rb, 50, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50, 1);
        assert_alpha_off(&rb, 50, 4);

        delete_merging(&mut rb, 50, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50, 1);
        assert_alpha_off(&rb, 500, 6);

        delete_merging(&mut rb, 500, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 500, 1);
        assert_alpha_off(&rb, 5000, 8);

        delete_merging(&mut rb, 5000, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 5000, 1);
        assert_alpha_off(&rb, 50000, 10);

        delete_merging(&mut rb, 50000, 1);
        rb.tree.is_valid(); // will panic if it must
        delete_merging(&mut rb, 50000, 1);
        assert_alpha_off(&rb, 500000, 12);
    }
}

