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

const NULL: usize = !0;

use crate::redblack::RedBlack;
use slab::Slab;
use std::mem;

#[cfg(test)]
use std::collections::VecDeque;

struct Node<T> {
    parent: usize,
    children: [usize; 2],
    key: T,
    red: bool,
}

impl<T> Node<T> {
    fn new(key: T, nil_sentinel: usize) -> Node<T> {
        Node {
            parent: nil_sentinel,
            children: [nil_sentinel, nil_sentinel],
            key,
            red: false,
        }
    }
}

pub struct SlabRedBlack<T> {
    slab: Slab<Node<T>>,
    root: usize,
    nil_sentinel: usize,
}

impl<T> SlabRedBlack<T>
where
    T: std::cmp::PartialOrd,
{
    fn rotate(&mut self, x: usize, dir: usize) {
        let y = self.slab[x].children[dir ^ 1];
        self.slab[x].children[dir ^ 1] = self.slab[y].children[dir];
        let y_chld = self.slab[y].children[dir];
        if y_chld != self.nil_sentinel {
            self.slab[y_chld].parent = x;
        }
        self.slab[y].parent = self.slab[x].parent;
        let x_parent = self.slab[x].parent;
        if x_parent == self.nil_sentinel {
            self.root = y;
        } else {
            let sib_dir = if self.slab[x_parent].children[0] == x {
                0
            } else {
                1
            };
            self.slab[x_parent].children[sib_dir] = y;
        }
        self.slab[y].children[dir] = x;
        self.slab[x].parent = y;
    }

    fn tree_minimum(&mut self, mut x: usize) -> usize {
        let mut l = self.slab[x].children[0];
        while l != self.nil_sentinel {
            x = l;
            l = self.slab[x].children[0];
        }
        x
    }

    fn tree_successor(&mut self, mut x: usize) -> usize {
        if self.slab[x].children[1] != self.nil_sentinel {
            return self.tree_minimum(x);
        }
        let mut y = self.slab[x].parent;
        while y != self.nil_sentinel && x == self.slab[y].children[1] {
            x = y;
            y = self.slab[y].parent;
        }
        y
    }

    fn insert_fixup(&mut self, mut z: usize) {
        let mut p = self.slab[z].parent;
        let mut pp: usize;

        while self.slab[p].red {
            p = self.slab[z].parent;
            pp = self.slab[p].parent;

            let dir = if self.slab[pp].children[0] == p { 1 } else { 0 };

            let y = self.slab[pp].children[dir];

            if self.slab[y].red {
                self.slab[p].red = false;
                self.slab[y].red = false;
                self.slab[pp].red = true;
                z = pp;

                // recompute parent and grandparent after changing z
                p = self.slab[z].parent;
            } else {
                // y is black, or nil sentinel
                if z == self.slab[p].children[dir] {
                    z = p;

                    self.rotate(z, dir ^ 1);

                    // recompute parent and grandparent after rotation
                    p = self.slab[z].parent;
                    pp = self.slab[p].parent;
                }
                self.slab[p].red = false;
                self.slab[pp].red = true;
                self.rotate(pp, dir);
            }
        }

        // blacken the root
        self.slab[self.root].red = false;
    }

    fn delete_fixup(&mut self, mut x: usize) {
        let mut p: usize;
        while x != self.root && !self.slab[x].red {
            p = self.slab[x].parent;
            let dir = if x == self.slab[p].children[0] { 1 } else { 0 };
            let mut w = self.slab[p].children[dir];
            if self.slab[w].red {
                self.slab[w].red = false;
                self.slab[p].red = true;
                self.rotate(p, dir ^ 1);

                // recompute w after the rotation of p
                w = self.slab[p].children[dir];
            }
            let wl = self.slab[w].children[0];
            let wr = self.slab[w].children[1];
            if !self.slab[wl].red && !self.slab[wr].red {
                self.slab[w].red = true;
                x = p;
            } else {
                let mut wc = self.slab[w].children[dir]; // w child i care about
                let wo = self.slab[w].children[dir ^ 1]; // w other child
                if !self.slab[wc].red {
                    self.slab[wo].red = false;
                    self.slab[w].red = true;
                    self.rotate(w, dir);
                    w = self.slab[p].children[dir];

                    // recompute wc after the rotation of w
                    wc = self.slab[w].children[dir];
                }
                self.slab[w].red = self.slab[p].red;
                self.slab[p].red = false;
                self.slab[wc].red = false;
                self.rotate(p, dir ^ 1);
                x = self.root
            }
        }

        // blacken x
        self.slab[x].red = false;
    }

    fn search_(&mut self, key: &T) -> Option<usize> {
        let mut curr = self.root;

        while curr != self.nil_sentinel {
            if self.slab[curr].key == *key {
                return Some(curr);
            }
            let direction = if self.slab[curr].key < *key { 1 } else { 0 };
            curr = self.slab[curr].children[direction];
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
        fn verify_black_height<T>(rb: &SlabRedBlack<T>, x: usize) -> i32 {
            if x == rb.nil_sentinel {
                return 0;
            }
            let left_height = verify_black_height(rb, rb.slab[x].children[0]);
            let right_height = verify_black_height(rb, rb.slab[x].children[1]);

            assert!(
                left_height != -1 && right_height != -1 && left_height == right_height,
                "red-black properties have been violated!"
            );

            let add = if rb.slab[x].red { 0 } else { 1 };
            left_height + add
        }

        fn verify_children_color<T>(rb: &SlabRedBlack<T>) -> bool {
            if rb.root == rb.nil_sentinel {
                return true;
            }
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_front(rb.root);

            while !queue.is_empty() {
                let curr = queue.pop_front().unwrap();
                if curr == rb.nil_sentinel {
                    break;
                }

                let l = rb.slab[curr].children[0];
                let r = rb.slab[curr].children[1];

                // red node must not have red children
                if rb.slab[curr].red {
                    assert!(
                        !rb.slab[l].red && !rb.slab[r].red,
                        "red node has red children"
                    );
                }

                if l != rb.nil_sentinel {
                    queue.push_back(l);
                }
                if r != rb.nil_sentinel {
                    queue.push_back(r);
                }
            }

            true
        }

        assert!(!self.slab[self.root].red); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
    }
}

impl<T> RedBlack<T> for SlabRedBlack<T>
where
    T: std::cmp::PartialOrd,
{
    fn new() -> SlabRedBlack<T> {
        let mut rb = SlabRedBlack {
            slab: Slab::new(),
            root: NULL,
            nil_sentinel: NULL,
        };
        unsafe {
            let nil_sentinel = rb.slab.insert(Node::new(
                mem::MaybeUninit::<T>::uninit().assume_init(),
                NULL,
            ));
            rb.nil_sentinel = nil_sentinel;
            rb.root = nil_sentinel;
        }
        rb
    }

    fn search(&mut self, key: &T) -> Option<&T> {
        if let Some(found_idx) = self.search_(key) {
            return Some(&self.slab[found_idx].key);
        }
        None
    }

    fn delete(&mut self, key: &T) {
        let z = match self.search_(key) {
            Some(found_idx) => found_idx,
            None => {
                return;
            }
        };

        let y = if self.slab[z].children[0] == self.nil_sentinel
            || self.slab[z].children[1] == self.nil_sentinel
        {
            z
        } else {
            self.tree_successor(z)
        };

        let dir = if self.slab[y].children[0] != self.nil_sentinel {
            0
        } else {
            1
        };
        let x = self.slab[y].children[dir];

        let yp = self.slab[y].parent;

        self.slab[x].parent = yp;

        if yp == self.nil_sentinel {
            self.root = x;
        } else {
            let dir = if y == self.slab[yp].children[0] { 0 } else { 1 };
            self.slab[yp].children[dir] = x;
        }

        if !self.slab[y].red {
            self.delete_fixup(x);
        }

        if y == self.nil_sentinel {
            return;
        }

        let mut y_removed = self.slab.remove(y); // remove the spliced-out node from the slab
        if y != z {
            mem::swap(&mut self.slab[z].key, &mut y_removed.key);
        }
    }

    fn insert(&mut self, key: T) {
        let z = self.slab.insert(Node::new(key, self.nil_sentinel));

        let mut y = self.nil_sentinel;
        let mut x = self.root;

        while x != self.nil_sentinel {
            y = x;
            let dir = if self.slab[z].key < self.slab[x].key {
                0
            } else {
                1
            };
            x = self.slab[x].children[dir];
        }

        self.slab[z].parent = y;
        if y == self.nil_sentinel {
            self.root = z;
        } else {
            let dir = if self.slab[z].key < self.slab[y].key {
                0
            } else {
                1
            };
            self.slab[y].children[dir] = z;
        }

        self.slab[z].red = true;

        self.insert_fixup(z);
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
        assert_eq!(rb.slab[1].parent, rb.nil_sentinel);
        assert_eq!(rb.slab[1].children[0], 2); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], 3); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].key, 1);
        assert_eq!(rb.slab[2].parent, 1);
        assert_eq!(rb.slab[2].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[2].children[1], rb.nil_sentinel);

        assert_eq!(rb.slab[3].key, 8);
        assert_eq!(rb.slab[3].parent, 1);
        assert_eq!(rb.slab[3].children[0], 4); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], 5); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].key, 7);
        assert_eq!(rb.slab[4].parent, 3);
        assert_eq!(rb.slab[4].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[4].children[1], rb.nil_sentinel);
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, 3);
        assert_eq!(rb.slab[5].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[5].children[1], rb.nil_sentinel);

        rb.rotate(1, 0); // left-rotate x

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
        assert_eq!(rb.slab[1].parent, 3); // x's new parent is y
        assert_eq!(rb.slab[3].children[0], 1); // y's left child is x
        assert_eq!(rb.slab[3].children[1], 5); // y's right child is gamma
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, 3);
        assert_eq!(rb.slab[1].children[0], 2); // x's left child is alpha
        assert_eq!(rb.slab[1].children[1], 4); // x's right child is beta
        assert_eq!(rb.slab[2].parent, 1); // alpha's parent is x
        assert_eq!(rb.slab[4].parent, 1); // beta's parent is x

        rb.rotate(3, 1); // right-rotate y brings our tree back to the original

        assert_eq!(rb.slab[1].key, 5);
        assert_eq!(rb.slab[1].parent, rb.nil_sentinel);
        assert_eq!(rb.slab[1].children[0], 2); // x's left points to 2 in the slab i.e. alpha
        assert_eq!(rb.slab[1].children[1], 3); // x's right points to 3 in the slab i.e. y

        assert_eq!(rb.slab[2].key, 1);
        assert_eq!(rb.slab[2].parent, 1);
        assert_eq!(rb.slab[2].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[2].children[1], rb.nil_sentinel);

        assert_eq!(rb.slab[3].key, 8);
        assert_eq!(rb.slab[3].parent, 1);
        assert_eq!(rb.slab[3].children[0], 4); // y's left points to 4 in the slab i.e. beta
        assert_eq!(rb.slab[3].children[1], 5); // y's right points to 5 in the slab i.e. gamma

        assert_eq!(rb.slab[4].key, 7);
        assert_eq!(rb.slab[4].parent, 3);
        assert_eq!(rb.slab[4].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[4].children[1], rb.nil_sentinel);
        assert_eq!(rb.slab[5].key, 9);
        assert_eq!(rb.slab[5].parent, 3);
        assert_eq!(rb.slab[5].children[0], rb.nil_sentinel);
        assert_eq!(rb.slab[5].children[1], rb.nil_sentinel);
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
