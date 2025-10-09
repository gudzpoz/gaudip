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

/// A wrapper around indices returned by [Slab]
// We can potentially change from `usize` to smaller integers
// because it will be used for visible line tracking, which
// rarely exceeds even `u8::MAX`. This type wrapper might help
// if we are to make such a change.
#[derive(Eq, PartialEq, Copy, Clone)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Ref(pub usize);
impl Ref {
    pub fn is_sentinel(&self) -> bool {
        self.0 == 0
    }
}
pub const SENTINEL: Ref = Ref(0);

pub struct RbSlab<T: Summable> {
    pub slab: Slab<Node<T>>,
    pub root: Ref,
}
impl<T: Summable> RbSlab<T> {
    pub fn new() -> Self {
        let mut slab = Slab::default();
        let index = slab.insert(Node {
            parent: SENTINEL,
            children: [SENTINEL, SENTINEL],
            red: false,
            left_sum: T::S::identity(),
            piece: unsafe { mem::MaybeUninit::zeroed().assume_init() },
        });
        assert_eq!(index, 0);
        Self { slab, root: Ref(0) }
    }
    pub fn insert(&mut self, node: Node<T>) -> Ref {
        Ref(self.slab.insert(node))
    }
    pub fn remove(&mut self, idx: Ref) -> T {
        self.slab.remove(idx.0).piece
    }
    pub fn get2(&mut self, idx1: Ref, idx2: Ref) -> (&mut Node<T>, &mut Node<T>) {
        self.slab.get2_mut(idx1.0, idx2.0).unwrap()
    }
}
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;

pub struct Node<T: Summable> {
    pub parent: Ref,
    pub children: [Ref; 2],
    pub red: bool,
    pub piece: T,
    pub left_sum: T::S,
}
impl<T: Summable> Node<T> {
    pub fn new(piece: T) -> Self {
        Self {
            parent: SENTINEL,
            children: [SENTINEL, SENTINEL],
            red: true,
            piece,
            left_sum: T::S::identity(),
        }
    }
}

impl<T: Summable> Index<Ref> for RbSlab<T> {
    type Output = Node<T>;

    fn index(&self, index: Ref) -> &Self::Output {
        &self.slab[index.0]
    }
}
impl<T: Summable> IndexMut<Ref> for RbSlab<T> {
    fn index_mut(&mut self, index: Ref) -> &mut Self::Output {
        &mut self.slab[index.0]
    }
}
impl<T: Summable> RbSlab<T> {
    pub fn next(&self, mut this: Ref, dir: usize) -> Ref {
        let child = self[this].children[dir];
        if !child.is_sentinel() {
            return self.edge(child, dir ^ 1);
        }
        let mut y = self[this].parent;
        while !y.is_sentinel() && this == self[y].children[dir] {
            this = y;
            y = self[y].parent;
        }
        y
    }

    pub fn edge(&self, mut this: Ref, dir: usize) -> Ref {
        let mut node = self[this].children[dir];
        while !node.is_sentinel() {
            this = node;
            node = self[this].children[dir];
        }
        this
    }

    pub fn calculate_sum(&self, mut node: Ref) -> T::S {
        let mut sum = T::S::identity();
        while !node.is_sentinel() {
            let obj = &self[node];
            sum.add_assign(&obj.left_sum);
            sum.add_assign(&obj.piece.summarize());
            node = obj.children[1];
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
        if !left.is_sentinel() {
            delta.add_assign(&self[left].left_sum);
            delta.add_assign(&self[left].piece.summarize());
            if dir == 0 {
                self[right].left_sum.add_assign(&delta);
            } else {
                self[right].left_sum.sub_assign(&delta);
            }
        }

        self[x].children[dir ^ 1] = self[y].children[dir];
        let y_child = self[y].children[dir];
        if !y_child.is_sentinel() {
            self[y_child].parent = x;
        }
        self.replace(x, y);
        self[y].children[dir] = x;
        self[x].parent = y;
    }

    fn replace(&mut self, x: Ref, y: Ref) {
        let parent = self[x].parent;
        self[y].parent = parent;
        if parent.is_sentinel() {
            self.root = y;
        } else {
            let parent = &mut self[parent];
            let dir = if parent.children[0] == x { 0 } else { 1 };
            parent.children[dir] = y;
        }
    }

    pub fn delete(&mut self, z: Ref) -> T {
        let (x, y) = {
            let zn = &self[z];
            if zn.children[0].is_sentinel() {
                (zn.children[1], z)
            } else if zn.children[1].is_sentinel() {
                (zn.children[0], z)
            } else {
                let y = self.edge(zn.children[1], LEFT);
                (self[y].children[1], y)
            }
        };

        if y == self.root {
            self.root = x;
            let xn = &mut self[x];
            xn.red = false;
            xn.parent = SENTINEL;
            self.reset_sentinel();
            return self.remove(z);
        }

        let yn = &self[y];
        let y_red = yn.red;
        let y_parent = yn.parent;

        let y_parent_i = if y == self[y_parent].children[0] {
            0
        } else {
            1
        };
        self[y_parent].children[y_parent_i] = x;

        if y == z {
            self[x].parent = y_parent;
            self.recompute_sum(x);
        } else {
            self[x].parent = if z == self[y].parent { y } else { y_parent };
            self.recompute_sum(x);

            let (yn, zn) = self.get2(y, z);
            yn.children = zn.children;
            yn.parent = zn.parent;
            yn.red = zn.red;
            yn.left_sum = zn.left_sum;
            let yn_children = yn.children;

            if z == self.root {
                self.root = y;
            } else {
                let parent = self[z].parent;
                let zp = &mut self[parent];
                let dir = if z == zp.children[0] { 0 } else { 1 };
                zp.children[dir] = y;
            }

            if !yn_children[0].is_sentinel() {
                self[yn_children[0]].parent = y;
            }
            if !yn_children[1].is_sentinel() {
                self[yn_children[1]].parent = y;
            }
            self.recompute_sum(y);
        }
        let ret = self.remove(z);

        let xp = self[x].parent;
        let xpn = &self[xp];
        if xpn.children[0] == x {
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

    fn recompute_sum(&mut self, mut x: Ref) {
        if x == self.root {
            return;
        }

        while x != self.root {
            let parent = self[x].parent;
            let node = &self[parent];
            if x != node.children[1] {
                if x == self.root {
                    return;
                }
                x = parent;
                break;
            }
            x = parent;
        }

        let xn = &self[x];
        let mut delta = self.calculate_sum(xn.children[0]);
        delta.sub_assign(&xn.left_sum);

        if delta != T::S::identity() {
            self[x].left_sum.add_assign(&delta);
            while x != self.root {
                let parent = self[x].parent;
                let node = &mut self[parent];
                if node.children[0] == x {
                    node.left_sum.add_assign(&delta);
                }

                x = parent;
            }
        }
    }

    pub fn update_metadata(&mut self, mut x: Ref, delta: &T::S) {
        while x != self.root && !x.is_sentinel() {
            let xp = self[x].parent;
            let xpn = &mut self[xp];
            if xpn.children[0] == x {
                xpn.left_sum.add_assign(delta);
            }
            x = xp;
        }
    }
}

#[cfg(test)]
use std::collections::VecDeque;

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
            if x.is_sentinel() {
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
            if rb.root.is_sentinel() {
                return true;
            }
            let mut queue: VecDeque<Ref> = VecDeque::new();
            queue.push_front(rb.root);

            while !queue.is_empty() {
                let curr = queue.pop_front().unwrap();
                if curr.is_sentinel() {
                    break;
                }

                let l = rb[curr].children[0];
                let r = rb[curr].children[1];

                // red node must not have red children
                if rb[curr].red {
                    assert!(!rb[l].red && !rb[r].red, "red node has red children");
                }

                if !l.is_sentinel() {
                    queue.push_back(l);
                }
                if !r.is_sentinel() {
                    queue.push_back(r);
                }
            }

            true
        }

        fn verify_sums<T: Summable>(rb: &RbSlab<T>, x: Ref) -> T::S {
            if x.is_sentinel() {
                return T::S::identity();
            }
            let node = &rb[x];
            let left = verify_sums(rb, node.children[0]);
            let right = verify_sums(rb, node.children[1]);
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
