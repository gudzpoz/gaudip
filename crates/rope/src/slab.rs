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

use crate::redblack::RedBlack;
use slab::Slab;
use std::mem;
use std::ops::{Index, IndexMut};

#[cfg(test)]
use std::collections::VecDeque;

#[derive(Eq, PartialEq, Copy, Clone)]
#[cfg_attr(test, derive(Debug))]
struct Ref(usize);
impl Ref {
    const fn sentinel() -> Self {
        Self(0)
    }
    fn is_sentinel(&self) -> bool {
        self.0 == 0
    }
}
const SENTINEL: Ref = Ref::sentinel();

struct Node<T> {
    parent: Ref,
    children: [Ref; 2],
    key: T,
    red: bool,
}

impl<T> Node<T> {
    fn new(key: T) -> Node<T> {
        Node {
            parent: SENTINEL,
            children: [SENTINEL, SENTINEL],
            key,
            red: false,
        }
    }
}

pub struct SlabRedBlack<T: PartialOrd> {
    slab: Slab<Node<T>>,
    root: Ref,
}

impl<T: PartialOrd> Index<Ref> for SlabRedBlack<T> {
    type Output = Node<T>;

    fn index(&self, index: Ref) -> &Self::Output {
        &self.slab[index.0]
    }
}
impl<T: PartialOrd> IndexMut<Ref> for SlabRedBlack<T> {
    fn index_mut(&mut self, index: Ref) -> &mut Self::Output {
        &mut self.slab[index.0]
    }
}

impl<T> SlabRedBlack<T>
where
    T: PartialOrd,
{
    fn replace(&mut self, node: Ref, with: Ref) {
        let parent = self[node].parent;
        self[with].parent = parent;
        if parent.is_sentinel() {
            self.root = with;
        } else {
            let parent = &mut self[parent];
            let dir = if parent.children[0] == node { 0 } else { 1 };
            parent.children[dir] = with;
        }
    }

    fn rotate(&mut self, x: Ref, dir: usize) {
        debug_assert!(dir == 0 || dir == 1);
        let dir = dir & 1;
        let y = self[x].children[dir ^ 1];
        self[x].children[dir ^ 1] = self[y].children[dir];
        let y_child = self[y].children[dir];
        if !y_child.is_sentinel() {
            self[y_child].parent = x;
        }
        self.replace(x, y);
        self[y].children[dir] = x;
        self[x].parent = y;
    }

    fn tree_leftest(&self, mut x: Ref) -> Ref {
        let mut l = self[x].children[0];
        while !l.is_sentinel() {
            x = l;
            l = self[x].children[0];
        }
        x
    }

    fn tree_successor(&self, mut x: Ref) -> Ref {
        if !self[x].children[1].is_sentinel() {
            return self.tree_leftest(x);
        }
        let mut y = self[x].parent;
        while !y.is_sentinel() && x == self[y].children[1] {
            x = y;
            y = self[y].parent;
        }
        y
    }

    fn insert_fixup(&mut self, mut z: Ref) {
        let mut p = self[z].parent;
        let mut pp: Ref;

        while self[p].red {
            p = self[z].parent;
            pp = self[p].parent;

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

    fn delete_fixup(&mut self, mut x: Ref) {
        let mut p: Ref;
        while x != self.root && !self[x].red {
            p = self[x].parent;
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
    }

    fn search_(&mut self, key: &T) -> Option<Ref> {
        let mut curr = self.root;

        while !curr.is_sentinel() {
            if self[curr].key == *key {
                return Some(curr);
            }
            let direction = if self[curr].key < *key { 1 } else { 0 };
            curr = self[curr].children[direction];
        }
        None
    }

    #[cfg(test)]
    fn is_valid(&self) {
        /*
         * properties
         * - root property: root is black
         * - leaf nodes (NULL) are black (pointless here given my sentinel is a NULL, not a real node)
         * - red property: children of a red node are black
         * - simple path from node to descendant leaf contains same number of black nodes
         */
        fn verify_black_height<T: PartialOrd>(rb: &SlabRedBlack<T>, x: Ref) -> i32 {
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

        fn verify_children_color<T: PartialOrd>(rb: &SlabRedBlack<T>) -> bool {
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

        assert!(!self[self.root].red); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
    }
}

impl<T> RedBlack<T> for SlabRedBlack<T>
where
    T: PartialOrd,
{
    fn new() -> SlabRedBlack<T> {
        let mut rb = SlabRedBlack {
            slab: Slab::new(),
            root: SENTINEL,
        };
        let index = rb.slab.insert(Node {
            parent: SENTINEL,
            children: [SENTINEL, SENTINEL],
            key: unsafe { mem::zeroed() },
            red: false,
        });
        assert_eq!(index, 0);
        rb
    }

    fn insert(&mut self, key: T) {
        let z = Ref(self.slab.insert(Node::new(key)));

        let mut y = SENTINEL;
        let mut x = self.root;

        while x != SENTINEL {
            y = x;
            let dir = if self[z].key < self[x].key { 0 } else { 1 };
            x = self[x].children[dir];
        }

        self[z].parent = y;
        if y.is_sentinel() {
            self.root = z;
        } else {
            let dir = if self[z].key < self[y].key { 0 } else { 1 };
            self[y].children[dir] = z;
        }

        self[z].red = true;

        self.insert_fixup(z);
    }

    fn delete(&mut self, key: &T) {
        let z = match self.search_(key) {
            Some(found_idx) => found_idx,
            None => {
                return;
            }
        };

        let y = if self[z].children[0].is_sentinel() || self[z].children[1].is_sentinel() {
            z
        } else {
            self.tree_successor(z)
        };

        let dir = if self[y].children[0].is_sentinel() {
            1
        } else {
            0
        };
        let x = self[y].children[dir];
        self.replace(y, x);

        if !self[y].red {
            self.delete_fixup(x);
        }

        if y.is_sentinel() {
            return;
        }

        let mut y_removed = self.slab.remove(y.0); // remove the spliced-out node from the slab
        if y != z {
            mem::swap(&mut self[z].key, &mut y_removed.key);
        }
    }

    fn search(&mut self, key: &T) -> Option<&T> {
        if let Some(found_idx) = self.search_(key) {
            return Some(&self[found_idx].key);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert() {
        let mut rb: SlabRedBlack<i32> = SlabRedBlack::new();

        rb.insert(5);
        rb.insert(6);
        rb.insert(7);

        assert_eq!(rb.search(&5), Some(&5));
        assert_eq!(rb.search(&6), Some(&6));
        assert_eq!(rb.search(&7), Some(&7));

        rb.is_valid(); // will panic if it must
    }

    #[test]
    fn test_basic_rotation() {
        let mut rb: SlabRedBlack<i32> = SlabRedBlack::new();

        rb.insert(5); // x
        rb.insert(1); // alpha
        rb.insert(8); // y
        rb.insert(7); // beta
        rb.insert(9); // gamma

        /*
         *      x
         *     / \
         *    /   y
         *   a   / \
         *      b   g
         */

        assert_eq!(rb.slab[1].key, 5);
        assert_eq!(rb.slab[1].parent, SENTINEL);
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], Ref(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].key, 1);
        assert_eq!(rb.slab[2].parent, Ref(1));
        assert_eq!(rb.slab[2].children[0], SENTINEL);
        assert_eq!(rb.slab[2].children[1], SENTINEL);

        assert_eq!(rb.slab[3].key, 8);
        assert_eq!(rb.slab[3].parent, Ref(1));
        assert_eq!(rb.slab[3].children[0], Ref(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].key, 7);
        assert_eq!(rb.slab[4].parent, Ref(3));
        assert_eq!(rb.slab[4].children[0], SENTINEL);
        assert_eq!(rb.slab[4].children[1], SENTINEL);
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[5].children[0], SENTINEL);
        assert_eq!(rb.slab[5].children[1], SENTINEL);

        rb.rotate(Ref(1), 0); // left-rotate x

        /*
         *      y
         *     / \
         *    x   g
         *   / \
         *  a   b
         */

        // slab entries should be the same, but their links should reflect the new tree topology

        assert_eq!(rb.slab[1].key, 5);
        assert_eq!(rb.slab[2].key, 1);
        assert_eq!(rb.slab[1].parent, Ref(3)); // x's new parent is y
        assert_eq!(rb.slab[3].children[0], Ref(1)); // y's left child is x
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right child is gamma
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left child is alpha
        assert_eq!(rb.slab[1].children[1], Ref(4)); // x's right child is beta
        assert_eq!(rb.slab[2].parent, Ref(1)); // alpha's parent is x
        assert_eq!(rb.slab[4].parent, Ref(1)); // beta's parent is x

        rb.rotate(Ref(3), 1); // right-rotate y brings our tree back to the original

        assert_eq!(rb.slab[1].key, 5);
        assert_eq!(rb.slab[1].parent, SENTINEL);
        assert_eq!(rb.slab[1].children[0], Ref(2)); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], Ref(3)); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].key, 1);
        assert_eq!(rb.slab[2].parent, Ref(1));
        assert_eq!(rb.slab[2].children[0], SENTINEL);
        assert_eq!(rb.slab[2].children[1], SENTINEL);

        assert_eq!(rb.slab[3].key, 8);
        assert_eq!(rb.slab[3].parent, Ref(1));
        assert_eq!(rb.slab[3].children[0], Ref(4)); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], Ref(5)); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].key, 7);
        assert_eq!(rb.slab[4].parent, Ref(3));
        assert_eq!(rb.slab[4].children[0], SENTINEL);
        assert_eq!(rb.slab[4].children[1], SENTINEL);
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, Ref(3));
        assert_eq!(rb.slab[5].children[0], SENTINEL);
        assert_eq!(rb.slab[5].children[1], SENTINEL);
    }

    #[test]
    fn test_many_insert() {
        let mut num = 1u32;
        let mut rb: SlabRedBlack<u32> = SlabRedBlack::new();

        for _ in 0..1000000 {
            num = num.wrapping_mul(17).wrapping_add(255);
            rb.insert(num);
        }

        rb.is_valid(); // will panic if it must
    }

    #[test]
    fn test_many_insert_some_delete() {
        let mut rb: SlabRedBlack<i32> = SlabRedBlack::new();

        for i in 500000..1000000 {
            rb.insert(i);
            rb.insert(1000000 - i);
        }

        assert_eq!(rb.search(&5), Some(&5));
        assert_eq!(rb.search(&50), Some(&50));
        assert_eq!(rb.search(&500), Some(&500));
        assert_eq!(rb.search(&5000), Some(&5000));
        assert_eq!(rb.search(&50000), Some(&50000));
        assert_eq!(rb.search(&500000), Some(&500000));

        rb.is_valid(); // will panic if it must

        rb.delete(&5);
        rb.is_valid(); // will panic if it must
        rb.delete(&5);
        assert_eq!(rb.search(&5), None);

        rb.delete(&50);
        rb.is_valid(); // will panic if it must
        rb.delete(&50);
        assert_eq!(rb.search(&50), None);

        rb.delete(&500);
        rb.is_valid(); // will panic if it must
        rb.delete(&500);
        assert_eq!(rb.search(&500), None);

        rb.delete(&5000);
        rb.is_valid(); // will panic if it must
        rb.delete(&5000);
        assert_eq!(rb.search(&5000), None);

        rb.delete(&50000);
        rb.is_valid(); // will panic if it must
        rb.delete(&50000);
        assert_eq!(rb.search(&50000), None);

        rb.delete(&500000);
        rb.is_valid(); // will panic if it must
        rb.delete(&500000);
        assert_eq!(rb.search(&500000), None);
    }
}
