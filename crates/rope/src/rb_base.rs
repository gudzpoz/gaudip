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
use std::mem;
use std::ops::{Index, IndexMut};

/// See [Node].
pub(crate) type SafeRef = NonZero<usize>;
/// A wrapper around indices returned by [Slab]
///
/// See [Node].
// We can potentially change from `usize` to smaller integers
// because it will be used for visible line tracking, which
// rarely exceeds even `u8::MAX`. This type wrapper might help
// if we are to make such a change.
pub(crate) type Ref = Option<SafeRef>;
pub const SENTINEL: Ref = None;

pub(crate) struct RbSlab<T: Summable> {
    pub slab: Slab<Node<T>>,
    pub root: Ref,
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
            piece: unsafe { mem::MaybeUninit::zeroed().assume_init() },
        });
        assert_eq!(index, 0);
        Self { slab, root: SENTINEL }
    }
    pub fn insert(&mut self, node: Node<T>) -> SafeRef {
        NonZero::new(self.slab.insert(node)).unwrap()
    }
    pub fn remove(&mut self, idx: SafeRef) -> T {
        self.slab.remove(idx.get()).piece
    }

    fn get2(&mut self, idx1: Ref, idx2: Ref) -> (&mut RbNode, &mut RbNode) {
        let idx1 = idx1.map(|r| r.get()).unwrap_or(0);
        let idx2 = idx2.map(|r| r.get()).unwrap_or(0);
        let (n1, n2) = self.slab.get2_mut(idx1, idx2).unwrap();
        (&mut n1.rb, &mut n2.rb)
    }
    pub(crate) fn rb(&self, node: Ref) -> &RbNode {
        &self.slab[node.map(|r| r.get()).unwrap_or(0)].rb
    }
    pub(crate) fn rb_mut(&mut self, node: Ref) -> &mut RbNode {
        &mut self.slab[node.map(|r| r.get()).unwrap_or(0)].rb
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
        self.rb_mut(SENTINEL).parent = SENTINEL;
    }

    pub fn rotate(&mut self, x: Ref, dir: usize) {
        debug_assert!(dir == 0 || dir == 1);
        let dir = dir & 1;
        let y = self.rb(x).children[dir ^ 1];

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

        self.rb_mut(x).children[dir ^ 1] = self.rb_mut(y).children[dir];
        let y_child = self.rb(y).children[dir];
        if let Some(idx) = y_child {
            self[idx].rb.parent = x;
        }
        self.replace(x, y);
        self.rb_mut(y).children[dir] = x;
        self.rb_mut(x).parent = y;
    }

    fn replace(&mut self, x: Ref, y: Ref) {
        let parent = self.rb(x).parent;
        self.rb_mut(y).parent = parent;
        if parent.is_none() {
            self.root = y;
        } else {
            let parent = &mut self.rb_mut(parent);
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
            let xn = &mut self.rb_mut(x);
            xn.red = false;
            xn.parent = SENTINEL;
            self.reset_sentinel();
            return self.remove(z);
        }

        let yn = &self[y].rb;
        let y_red = yn.red;
        let y_parent = yn.parent;

        let y_parent_i = if Some(y) == self.rb(y_parent).children[0] {
            0
        } else {
            1
        };
        self.rb_mut(y_parent).children[y_parent_i] = x;

        if y == z {
            self.rb_mut(x).parent = y_parent;
            self.recompute_sum(x);
        } else {
            self.rb_mut(x).parent = if Some(z) == self[y].rb.parent { Some(y) } else { y_parent };
            self.recompute_sum(x);

            let (yn, zn) = self.get2(Some(y), Some(z));
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
                let zp = &mut self.rb_mut(parent);
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

        let xp = self.rb(x).parent;
        let xpn = &self.rb(xp);
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
        while x != self.root && !self.rb(x).red {
            let p = self.rb(x).parent;
            let dir = if x == self.rb(p).children[0] { 1 } else { 0 };
            let mut w = self.rb(p).children[dir];
            if self.rb(w).red {
                self.rb_mut(w).red = false;
                self.rb_mut(p).red = true;
                self.rotate(p, dir ^ 1);

                // recompute w after the rotation of p
                w = self.rb(p).children[dir];
            }
            let wl = self.rb(w).children[0];
            let wr = self.rb(w).children[1];
            if !self.rb(wl).red && !self.rb(wr).red {
                self.rb_mut(w).red = true;
                x = p;
            } else {
                let mut wc = self.rb(w).children[dir]; // w child i care about
                let wo = self.rb(w).children[dir ^ 1]; // w other child
                if !self.rb(wc).red {
                    self.rb_mut(wo).red = false;
                    self.rb_mut(w).red = true;
                    self.rotate(w, dir);
                    w = self.rb(p).children[dir];

                    // recompute wc after the rotation of w
                    wc = self.rb(w).children[dir];
                }
                self.rb_mut(w).red = self.rb(p).red;
                self.rb_mut(p).red = false;
                self.rb_mut(wc).red = false;
                self.rotate(p, dir ^ 1);
                x = self.root
            }
        }

        // blacken x
        self.rb_mut(x).red = false;
        self.reset_sentinel();
    }

    pub fn fix_insert(&mut self, mut z: Ref) {
        self.recompute_sum(z);

        let mut p = self.rb(z).parent;

        while z != self.root && self.rb(p).red {
            p = self.rb(z).parent;
            let mut pp = self.rb(p).parent;

            let dir = if self.rb(pp).children[0] == p { 1 } else { 0 };

            let y = self.rb(pp).children[dir];

            if self.rb(y).red {
                self.rb_mut(p).red = false;
                self.rb_mut(y).red = false;
                self.rb_mut(pp).red = true;
                z = pp;

                // recompute parent and grandparent after changing z
                p = self.rb(z).parent;
            } else {
                // y is black, or nil sentinel
                if z == self.rb(p).children[dir] {
                    z = p;

                    self.rotate(z, dir ^ 1);

                    // recompute parent and grandparent after rotation
                    p = self.rb(z).parent;
                    pp = self.rb(p).parent;
                }
                self.rb_mut(p).red = false;
                self.rb_mut(pp).red = true;
                self.rotate(pp, dir);
            }
        }

        // blacken the root
        let root = self.root;
        self.rb_mut(root).red = false;
    }

    fn recompute_sum(&mut self, x: Ref) {
        let Some(mut x) = x else {
            return;
        };

        loop {
            let parent = self[x].rb.parent;
            let Some(pi) = parent else {
                debug_assert!(Some(x) == self.root);
                return;
            };
            let node = &self.rb(parent);
            if Some(x) != node.children[1] {
                x = pi;
                break;
            }
            x = pi;
        }

        let xn = &self[x];
        let mut delta = self.calculate_sum(xn.rb.children[0]);
        delta.sub_assign(&xn.left_sum);

        if delta != T::S::identity() {
            self[x].left_sum.add_assign(&delta);
            loop {
                let parent = self[x].rb.parent;
                let Some(pi) = parent else {
                    debug_assert!(Some(x) == self.root);
                    break;
                };
                let node = &mut self[pi];
                if node.rb.children[0] == Some(x) {
                    node.left_sum.add_assign(&delta);
                }

                x = pi;
            }
        }
    }

    pub fn update_metadata(&mut self, mut x: SafeRef, delta: &T::S) {
        loop {
            let xp = self[x].rb.parent;
            let Some(parent) = xp else {
                debug_assert!(Some(x) == self.root);
                break;
            };
            let xpn = &mut self[parent];
            if xpn.rb.children[0] == Some(x) {
                xpn.left_sum.add_assign(delta);
            }
            x = parent;
        }
    }
}

#[cfg(test)]
use std::collections::VecDeque;
use std::num::NonZero;

#[cfg(test)]
impl<T: Summable> RbSlab<T> {
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
            let left_height = verify_black_height(rb, rb.rb(x).children[0]);
            let right_height = verify_black_height(rb, rb.rb(x).children[1]);

            assert!(
                left_height != -1 && right_height != -1 && left_height == right_height,
                "red-black properties have been violated!"
            );

            let add = if rb.rb(x).red { 0 } else { 1 };
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
                    assert!(!rb.rb(l).red && !rb.rb(r).red, "red node has red children");
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

        assert!(!self.rb(self.root).red); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
        verify_sums(self, self.root)
    }
}
