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

use crate::piece::{Sum, Summable};
use slab::Slab;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ops::{Index, IndexMut};

/// Safe references for [Node], guaranteeing non-sentinel nodes
///
/// See [Node].
pub(crate) type SafeRef = NonZero<usize>;
/// A wrapper around indices returned by [Slab]
///
/// It's typically unsafe, because it might point to [SENTINEL] nodes,
/// whose [Node::piece] is uninitialized. See [Node].
pub(crate) type Ref = Option<SafeRef>;
pub const SENTINEL: Ref = None;
fn ref_int(node: Ref) -> usize {
    node.map(|r| r.get()).unwrap_or(0)
}

pub(crate) struct RbSlab<T: Summable> {
    /// Slab storage for nodes
    ///
    /// The first element (referenced by [SENTINEL]) is zero-initialized
    /// and is unsafe to access (unless accessing only the [RbNode] part).
    ///
    /// Please do not access this field directly, but instead add accessor
    /// methods that build on the safety assumptions:
    /// - [SafeRef] for [Node] access,
    /// - [Ref] (for [RbNode] access).
    slab: Slab<Node<T>>,
    root: Ref,
}
impl<T: Summable> RbSlab<T> {
    pub fn new() -> Self {
        let mut slab = Slab::default();
        let index = slab.insert(Node {
            rb: RbNode {
                parent: SENTINEL,
                children: [SENTINEL, SENTINEL],
                red: false,
            },
            left_sum: T::S::identity(),
            piece: unsafe { MaybeUninit::zeroed().assume_init() },
        });
        assert_eq!(index, 0);
        Self { slab, root: SENTINEL }
    }
    pub fn root(&self) -> Ref {
        self.root
    }
    pub fn set_root(&mut self, root: Ref) {
        self.root = root;
    }
    pub fn clear(&mut self) {
        self.root = SENTINEL;
        self.slab.clear();
    }

    pub fn insert(&mut self, node: Node<T>) -> SafeRef {
        NonZero::new(self.slab.insert(node)).unwrap()
    }
    pub fn remove(&mut self, idx: SafeRef) -> T {
        self.slab.remove(idx.get()).piece
    }

    pub fn get2(&mut self, idx1: SafeRef, idx2: SafeRef) -> (&mut RbNode, &mut RbNode) {
        let idx1 = idx1.get();
        let idx2 = idx2.get();
        let (n1, n2) = self.slab.get2_mut(idx1, idx2).unwrap();
        (&mut n1.rb, &mut n2.rb)
    }
    pub fn get2_mut(&mut self, idx1: SafeRef, idx2: SafeRef) -> (&mut Node<T>, &mut Node<T>) {
        let idx1 = idx1.get();
        let idx2 = idx2.get();
        let (n1, n2) = self.slab.get2_mut(idx1, idx2).unwrap();
        (n1, n2)
    }
}
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;

/// See [Node].
pub struct RbNode {
    pub parent: Ref,
    pub children: [Ref; 2],
    pub red: bool,
}
/// RB-tree node ([RbNode]) + [Summable] information.
///
/// The sentinel node has its [Node::piece] uninitialized and thus
/// its [Node] should never get accessed directly. This rb-node is
/// the portion of it can be accessed.
///
/// This is reflected in the [Index] and [IndexMut] API. Basically:
/// - [SafeRef] never points to sentinel and is used for [Index] to 
///   retrieve [Node].
/// - [Ref] can point to sentinel, and should only be used within this
///   very file manipulating rb-tree structures ([RbNode]).
pub struct Node<T: Summable> {
    pub rb: RbNode,
    pub piece: T,
    pub left_sum: T::S,
}
impl<T: Summable> Node<T> {
    pub fn new(piece: T) -> Self {
        Self {
            rb: RbNode {
                parent: SENTINEL,
                children: [SENTINEL, SENTINEL],
                red: true,
            },
            piece,
            left_sum: T::S::identity(),
        }
    }
}

impl<T: Summable> Index<SafeRef> for RbSlab<T> {
    type Output = Node<T>;

    fn index(&self, index: SafeRef) -> &Self::Output {
        &self.slab[index.get()]
    }
}
impl<T: Summable> IndexMut<SafeRef> for RbSlab<T> {
    fn index_mut(&mut self, index: SafeRef) -> &mut Self::Output {
        &mut self.slab[index.get()]
    }
}
impl<T: Summable> Index<Ref> for RbSlab<T> {
    type Output = RbNode;

    fn index(&self, index: Ref) -> &Self::Output {
        &self.slab[ref_int(index)].rb
    }
}
impl<T: Summable> IndexMut<Ref> for RbSlab<T> {
    fn index_mut(&mut self, index: Ref) -> &mut Self::Output {
        &mut self.slab[ref_int(index)].rb
    }
}

macro_rules! foreach_parent {
    (({ $p:ident: $pn:ident } of { $x:ident: $n:ident } in $tree:expr) $what:block) => {{
        let mut $x: SafeRef = $x;
        loop {
            let $n = &$tree[$x];
            let parent = $n.rb.parent;
            let Some($p) = parent else {
                debug_assert!(Some($x) == $tree.root);
                break None;
            };
            let $pn = &mut $tree[$p];
            $what;
            $x = $p;
        }
    }};
    (({ $p:ident: $pn:ident } of $x:ident in $tree:expr) $what:block) => {
        foreach_parent!(({ $p: $pn } of { $x: x_node } in $tree) $what)
    };
    (($pn:ident of $x:ident in $tree:expr) $what:block) => {
        foreach_parent!(({ parent: $pn } of $x in $tree) $what)
    };
}

impl<T: Summable> RbSlab<T> {
    pub fn next(&self, mut this: SafeRef, dir: usize) -> Ref {
        let child = self[this].rb.children[dir];
        if let Some(idx) = child {
            return Some(self.edge(idx, dir ^ 1));
        }
        let mut y = self[this].rb.parent;
        while let Some(yi) = y && Some(this) == self[yi].rb.children[dir] {
            this = yi;
            y = self[yi].rb.parent;
        }
        y
    }

    pub fn edge(&self, mut this: SafeRef, dir: usize) -> SafeRef {
        let mut node = self[this].rb.children[dir];
        while let Some(idx) = node {
            this = idx;
            node = self[idx].rb.children[dir];
        }
        this
    }

    pub fn calculate_sum(&self, mut node: Ref) -> T::S {
        let mut sum = T::S::identity();
        while let Some(idx) = node {
            let obj = &self[idx];
            sum.add_assign(&obj.left_sum);
            sum.add_assign(&obj.piece.summarize());
            node = obj.rb.children[1];
        }
        sum
    }

    fn reset_sentinel(&mut self) {
        self[SENTINEL].parent = SENTINEL;
    }

    pub fn rotate(&mut self, x: Ref, dir: usize) {
        debug_assert!(dir == 0 || dir == 1);
        let dir = dir & 1;
        let y = self[x].children[dir ^ 1];

        // fix stats
        let (left, right) = if dir == 0 { (x, y) } else { (y, x) };
        let mut delta = T::S::identity();
        if let Some(idx) = left {
            delta.add_assign(&self[idx].left_sum);
            delta.add_assign(&self[idx].piece.summarize());
            if let Some(right) = right {
                if dir == 0 {
                    self[right].left_sum.add_assign(&delta);
                } else {
                    self[right].left_sum.sub_assign(&delta);
                }
            }
        }

        self[x].children[dir ^ 1] = self[y].children[dir];
        let y_child = self[y].children[dir];
        if let Some(idx) = y_child {
            self[idx].rb.parent = x;
        }
        self.replace(x, y);
        self[y].children[dir] = x;
        self[x].parent = y;
    }

    fn replace(&mut self, x: Ref, y: Ref) {
        let parent = self[x].parent;
        self[y].parent = parent;
        if parent.is_none() {
            self.root = y;
        } else {
            let parent = &mut self[parent];
            let dir = if parent.children[0] == x { 0 } else { 1 };
            parent.children[dir] = y;
        }
    }

    pub fn delete(&mut self, z: SafeRef) -> T {
        let (x, y) = {
            let zn = &self[z];
            match zn.rb.children {
                [SENTINEL, right] => (right, z),
                [left, SENTINEL] => (left, z),
                [_, Some(right)] => {
                    let y = self.edge(right, LEFT);
                    (self[y].rb.children[1], y)
                },
            }
        };

        if Some(y) == self.root {
            self.root = x;
            let xn = &mut self[x];
            xn.red = false;
            xn.parent = SENTINEL;
            self.reset_sentinel();
            return self.remove(z);
        }

        let yn = &self[y].rb;
        let y_red = yn.red;
        let y_parent = yn.parent;

        let y_parent_i = if Some(y) == self[y_parent].children[0] {
            0
        } else {
            1
        };
        self[y_parent].children[y_parent_i] = x;

        if y == z {
            self[x].parent = y_parent;
            self.recompute_sum(x);
        } else {
            self[x].parent = if Some(z) == self[y].rb.parent { Some(y) } else { y_parent };
            self.recompute_sum(x);

            let (yn, zn) = self.get2(y, z);
            yn.children = zn.children;
            yn.parent = zn.parent;
            yn.red = zn.red;
            let yn_children = yn.children;
            self[y].left_sum = self[z].left_sum;
            let y = Some(y);

            if Some(z) == self.root {
                self.root = y;
            } else {
                let parent = self[z].rb.parent;
                let zp = &mut self[parent];
                let dir = if Some(z) == zp.children[0] { 0 } else { 1 };
                zp.children[dir] = y;
            }

            if let Some(idx) = yn_children[0] {
                self[idx].rb.parent = y;
            }
            if let Some(idx) = yn_children[1] {
                self[idx].rb.parent = y;
            }
            self.recompute_sum(y);
        }
        let ret = self.remove(z);

        let xp = self[x].parent;
        let xpn = &self[xp];
        if xpn.children[0] == x {
            let Some(xp) = xp else { unreachable!(); };
            let xpn = &self[xp];
            let new_sum = self.calculate_sum(x);
            if new_sum != xpn.left_sum {
                let mut delta = new_sum;
                delta.sub_assign(&xpn.left_sum);
                self[xp].left_sum = new_sum;
                self.update_metadata(xp, &delta);
            }
        }
        self.recompute_sum(xp);

        if y_red {
            self.reset_sentinel();
        } else {
            self.rb_insert_fixup(x);
        }
        ret
    }

    fn rb_insert_fixup(&mut self, mut x: Ref) {
        while x != self.root && !self[x].red {
            let p = self[x].parent;
            let dir = if x == self[p].children[0] { 1 } else { 0 };
            let mut w = self[p].children[dir];
            if self[w].red {
                self[w].red = false;
                self[p].red = true;
                self.rotate(p, dir ^ 1);

                // recompute w after the rotation of p
                w = self[p].children[dir];
            }
            let wl = self[w].children[0];
            let wr = self[w].children[1];
            if !self[wl].red && !self[wr].red {
                self[w].red = true;
                x = p;
            } else {
                let mut wc = self[w].children[dir]; // w child i care about
                let wo = self[w].children[dir ^ 1]; // w other child
                if !self[wc].red {
                    self[wo].red = false;
                    self[w].red = true;
                    self.rotate(w, dir);
                    w = self[p].children[dir];

                    // recompute wc after the rotation of w
                    wc = self[w].children[dir];
                }
                self[w].red = self[p].red;
                self[p].red = false;
                self[wc].red = false;
                self.rotate(p, dir ^ 1);
                x = self.root
            }
        }

        // blacken x
        self[x].red = false;
        self.reset_sentinel();
    }

    pub fn fix_insert(&mut self, mut z: Ref) {
        self.recompute_sum(z);

        let mut p = self[z].parent;

        while z != self.root && self[p].red {
            p = self[z].parent;
            let mut pp = self[p].parent;

            let dir = if self[pp].children[0] == p { 1 } else { 0 };

            let y = self[pp].children[dir];

            if self[y].red {
                self[p].red = false;
                self[y].red = false;
                self[pp].red = true;
                z = pp;

                // recompute parent and grandparent after changing z
                p = self[z].parent;
            } else {
                // y is black, or nil sentinel
                if z == self[p].children[dir] {
                    z = p;

                    self.rotate(z, dir ^ 1);

                    // recompute parent and grandparent after rotation
                    p = self[z].parent;
                    pp = self[p].parent;
                }
                self[p].red = false;
                self[pp].red = true;
                self.rotate(pp, dir);
            }
        }

        // blacken the root
        let root = self.root;
        self[root].red = false;
    }

    fn recompute_sum(&mut self, x: Ref) {
        let Some(x) = x else { return };
        let Some(x) = foreach_parent!(({ p: pn } of x in self) {
            if Some(x) != pn.rb.children[1] {
                x = p;
                break Some(x);
            }
        }) else { return };

        let xn = &self[x];
        let mut delta = self.calculate_sum(xn.rb.children[0]);
        delta.sub_assign(&xn.left_sum);

        if delta != T::S::identity() {
            self[x].left_sum.add_assign(&delta);
            self.update_metadata(x, &delta);
        }
    }

    pub fn update_metadata(&mut self, x: SafeRef, delta: &T::S) {
        let _: Option<()> = foreach_parent!((pn of x in self) {
            if pn.rb.children[0] == Some(x) {
                pn.left_sum.add_assign(delta);
            }
        });
    }
}

#[cfg(test)]
use std::collections::VecDeque;

#[cfg(test)]
/// The rb-tree implementation is actually split across two files:
/// `rb_base.rs` and `roperig.rs`, with the majority of tests put
/// in `roperig.rs`.
impl<T: Summable> RbSlab<T> {
    pub fn slab(&self, i: usize) -> &Node<T> {
        &self.slab[i]
    }
    pub fn slab_get(&self, i: usize) -> Option<&Node<T>> {
        self.slab.get(i)
    }
    pub fn slab_len(&self) -> usize {
        self.slab.len()
    }

    pub fn is_valid(&self) -> T::S {
        /*
         * properties
         * - root property: root is black
         * - leaf nodes (NULL) are black (pointless here given my sentinel is a NULL, not a real node)
         * - red property: children of a red node are black
         * - simple path from node to descendant leaf contains same number of black nodes
         */
        fn verify_black_height<T: Summable>(rb: &RbSlab<T>, x: Ref) -> i32 {
            if x.is_none() {
                return 0;
            }
            let left_height = verify_black_height(rb, rb[x].children[0]);
            let right_height = verify_black_height(rb, rb[x].children[1]);

            assert!(
                left_height != -1 && right_height != -1 && left_height == right_height,
                "red-black properties have been violated!"
            );

            let add = if rb[x].red { 0 } else { 1 };
            left_height + add
        }

        fn verify_children_color<T: Summable>(rb: &RbSlab<T>) -> bool {
            if rb.root.is_none() {
                return true;
            }
            let mut queue: VecDeque<Ref> = VecDeque::new();
            queue.push_front(rb.root);

            while !queue.is_empty() {
                let curr = queue.pop_front().unwrap();
                let Some(idx) = curr else {
                    break;
                };

                let l = rb[idx].rb.children[0];
                let r = rb[idx].rb.children[1];

                // red node must not have red children
                if rb[idx].rb.red {
                    assert!(!rb[l].red && !rb[r].red, "red node has red children");
                }

                if l.is_some() {
                    queue.push_back(l);
                }
                if r.is_some() {
                    queue.push_back(r);
                }
            }

            true
        }

        fn verify_sums<T: Summable>(rb: &RbSlab<T>, x: Ref) -> T::S {
            let Some(idx) = x else {
                return T::S::identity();
            };
            let node = &rb[idx];
            let left = verify_sums(rb, node.rb.children[0]);
            let right = verify_sums(rb, node.rb.children[1]);
            assert!(node.left_sum == left);
            let mut sum = node.piece.summarize();
            sum.add_assign(&left);
            sum.add_assign(&right);
            sum
        }

        assert!(!self[self.root].red); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
        verify_sums(self, self.root)
    }
}

#[cfg(test)]
mod tests {
    use crate::metrics::CursorPos;
    use crate::roperig_test::Alphabet;
    use super::*;

    #[test]
    fn test_option_nonzero_sizes() {
        assert_eq!(size_of::<Option<NonZero<usize>>>(), size_of::<NonZero<usize>>());
        assert_eq!(size_of::<Ref>(), size_of::<SafeRef>());
        assert_eq!(size_of::<Ref>(), size_of::<usize>());
        // Option<CursorPos<T>> should be free because CursorPos contains SafeRef,
        // which is NonZero, and can be used as Option-tagging.
        type C = CursorPos<Alphabet>;
        assert_eq!(size_of::<Option<C>>(), size_of::<C>());
    }
}
