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

// Copyright (c) 2015 - 2025 Microsoft Corporation
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

use slab::Slab;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::num::NonZero;
use std::ops::{Index, IndexMut, Range};

use crate::interval::{
    BLACK, RED, LEFT, RIGHT, MAX_SAFE_DELTA, MIN_SAFE_DELTA,
    Interval, Ref, SENTINEL,
    interval_compare,
};

/// A red-black tree of relatively-positioned intervals.
///
/// Please note that the API mostly requires a mutable reference to the tree,
/// because it needs to constantly update outdated cached absolute metadata.
#[derive(Default, Clone)]
pub struct IntervalTree {
    /// Slab storage for nodes
    ///
    /// The first element (referenced by [SENTINEL]) is zero-initialized.
    slab: Slab<Interval>,
    root: Ref,
    request_normalize_delta: bool,
}

impl IntervalTree {
    /// Constructs a new empty tree containing a single SENTINEL node
    pub fn new() -> Self {
        let mut slab = Slab::default();
        let index = slab.insert(Interval::default());
        assert_eq!(index, 0);
        Self { slab, root: SENTINEL, request_normalize_delta: false }
    }

    /// Searches a range for overlapping intervals
    pub fn search<F>(&mut self, range: Range<isize>, cache_version_id: u32, f: F)
    where F: FnMut(&mut Interval) {
        if self.root == SENTINEL {
            return;
        }
        self.interval_search(range, cache_version_id, f);
    }

    /// Inserts a new node into the tree
    ///
    /// Returns a reference to the new node.
    pub fn insert(&mut self, node: Interval) -> NonZero<Ref> {
        let id = self.rb_tree_insert(node);
        self.normalize_delta_if_necessary();
        NonZero::new(id).expect("never sentinel")
    }

    /// Deletes a node from the tree by reference (returned by [Self::insert])
    pub fn delete(&mut self, node: Ref) -> Interval {
        self.rb_tree_delete(node);
        let node = self.slab.remove(node);
        self.normalize_delta_if_necessary();
        node
    }

    /// Gets a reference to an interval by reference
    ///
    /// Note that for absolute position, use [Self::resolve_node].
    pub fn get(&self, node: Ref) -> Option<&Interval> {
        self.slab.get(node)
    }
    /// Gets a mutable reference to an interval by reference
    ///
    /// Note that for absolute position, use [Self::resolve_node].
    ///
    /// # Safety
    ///
    /// The returned interval is an internal tree node. The user must
    /// not modify hidden fields with something like `*node = new_node;`.
    /// Currently, the user is expected to only update the metadata bits
    /// in the [Interval] struct.
    pub unsafe fn get_mut(&mut self, node: Ref) -> Option<&mut Interval> {
        self.slab.get_mut(node)
    }

    /// Updates the cached absolute position of the node
    ///
    /// Note that it does not return a mutable reference. If you want to update a node,
    /// delete and insert it.
    pub fn resolve_node(&mut self, mut node: Ref, cached_version_id: u32) -> &Interval {
        if self[node].cached_version_id == cached_version_id {
            return &mut self[node];
        }

        let initial_node = node;
        let mut delta = 0;
        while node != self.root {
            let n = &self[node];
            let p = n.parent;
            let pn = &self[p];
            if node == pn.children[RIGHT] {
                delta += pn.delta;
            }
            node = p;
        }

        let n = &mut self[initial_node];
        let node_start = n.start + delta;
        let node_end = n.end + delta;
        n.set_cached_offsets(node_start, node_end, cached_version_id);
        n
    }

    /// Notifies the tree of a change to the underlying text to let it update interval ranges
    pub fn accept_replace(&mut self, del_range: Range<isize>, insert_length: usize) {
        let nodes_of_interest = self.search_for_editing(del_range.clone());
        for node in &nodes_of_interest {
            self.rb_tree_delete(*node);
        }
        self.normalize_delta_if_necessary();

        self.no_overlap_replace(del_range.clone(), insert_length);
        self.normalize_delta_if_necessary();

        for node in nodes_of_interest {
            let n = &mut self[node];
            n.start = n.cached_range.start;
            n.end = n.cached_range.end;
            n.accept_edit(del_range.clone(), insert_length);
            n.max_end = n.end;
            self.rb_tree_insert_ref(node);
        }
        self.normalize_delta_if_necessary();
    }

    fn normalize_delta_if_necessary(&mut self) {
        if !self.request_normalize_delta {
            return;
        }
        self.request_normalize_delta = false;

        let mut node = self.root;
        let mut delta = 0;
        while node != SENTINEL {
            let n = &self[node];
            let [left, right] = n.children;
            let parent = n.parent;
            if left != SENTINEL && !self[left].is_visited() {
                node = left;
                continue;
            }
            if right != SENTINEL && !self[right].is_visited() {
                delta += n.delta;
                node = right;
                continue;
            }

            let n = &mut self[node];
            n.start += delta;
            n.end += delta;
            n.delta = 0;
            n.set_visited(true);
            self.recompute_max_end(node);

            self[left].set_visited(false);
            self[right].set_visited(false);
            let p = &self[parent];
            if node == p.children[RIGHT] {
                delta -= p.delta;
            }
            node = parent;
        }

        let root = self.root;
        self[root].set_visited(false);
    }
}

impl Index<usize> for IntervalTree {
    type Output = Interval;

    fn index(&self, index: usize) -> &Self::Output {
        &self.slab[index]
    }
}
impl IndexMut<usize> for IntervalTree {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.slab[index]
    }
}

impl IntervalTree {
    fn search_for_editing(&mut self, del_range: Range<isize>) -> SmallVec<[Ref; 4]> {
        let mut node = self.root;
        let mut delta = 0;
        let mut node_max_end;
        let mut node_start;
        let mut node_end;
        let mut result = SmallVec::new();
        while node != SENTINEL {
            let n = &self[node];
            let Interval { start, end, max_end, children: [left, right], .. } = *n;
            if n.is_visited() {
                self[left].set_visited(false);
                self[right].set_visited(false);
                let parent = self[node].parent;
                let p = &self[parent];
                if node == p.children[RIGHT] {
                    delta -= p.delta;
                }
                node = parent;
                continue;
            }

            if left != SENTINEL && !self[left].is_visited() {
                node_max_end = max_end + delta;
                if node_max_end < del_range.start {
                    self[node].set_visited(true);
                    continue;
                }
                node = left;
                continue;
            }

            node_start = start + delta;
            if node_start > del_range.end {
                self[node].set_visited(true);
                continue;
            }

            node_end = end + delta;
            if node_end >= del_range.start {
                self[node].set_cached_offsets(node_start, node_end, 0);
                result.push(node);
            }
            self[node].set_visited(true);

            if right != SENTINEL && !self[right].is_visited() {
                delta += self[node].delta;
                node = right;
                continue;
            }
        }

        let root = self.root;
        self[root].set_visited(false);
        result
    }

    fn no_overlap_replace(&mut self, del_range: Range<isize>, insert_length: usize) {
        let mut node = self.root;
        let mut delta = 0;
        let mut node_max_end;
        let mut node_start;
        let edit_delta = insert_length.checked_signed_diff(del_range.len()).unwrap();
        while node != SENTINEL {
            let n = &self[node];
            let Interval { start, max_end, children: [left, right], .. } = *n;
            if n.is_visited() {
                self[left].set_visited(false);
                self[right].set_visited(false);
                let parent = self[node].parent;
                let p = &self[parent];
                if node == p.children[RIGHT] {
                    delta -= p.delta;
                }
                self.recompute_max_end(node);
                node = parent;
                continue;
            }

            if left != SENTINEL && !self[left].is_visited() {
                node_max_end = max_end + delta;
                if node_max_end < del_range.start {
                    self[node].set_visited(true);
                    continue;
                }
                node = left;
                continue;
            }

            node_start = start + delta;
            if node_start > del_range.end {
                let n = &mut self[node];
                n.start += edit_delta;
                n.end += edit_delta;
                n.delta += edit_delta;
                n.set_visited(true);
                let n_delta = n.delta;
                self.set_request_normalize_delta(n_delta);
                continue;
            }

            self[node].set_visited(true);

            if right != SENTINEL && !self[right].is_visited() {
                delta += self[node].delta;
                node = right;
                continue;
            }
        }

        let root = self.root;
        self[root].set_visited(false);
    }
}

impl IntervalTree {
    fn interval_search<F>(
        &mut self, interval: Range<isize>, cache_version_id: u32,
        mut f: F,
    ) where F: FnMut(&mut Interval) {
        let mut node = self.root;
        let mut delta = 0;
        let mut node_max_end;
        let mut node_start;
        let mut node_end;
        while node != SENTINEL {
            let n = &self[node];
            let Interval { start, end, max_end, children: [left, right], .. } = *n;
            if n.is_visited() {
                self[left].set_visited(false);
                self[right].set_visited(false);
                let parent = self[node].parent;
                let p = &self[parent];
                if node == p.children[RIGHT] {
                    delta -= p.delta;
                }
                node = parent;
                continue;
            }

            if left != SENTINEL && !self[left].is_visited() {
                node_max_end = max_end + delta;
                if node_max_end < interval.start {
                    self[node].set_visited(true);
                    continue;
                }
                node = left;
                continue;
            }

            node_start = start + delta;
            if node_start > interval.end {
                self[node].set_visited(true);
                continue;
            }

            node_end = end + delta;
            if node_end >= interval.start {
                let n = &mut self[node];
                n.set_cached_offsets(node_start, node_end, cache_version_id);
                f(n);
            }

            self[node].set_visited(true);

            if right != SENTINEL && !self[right].is_visited() {
                delta += self[node].delta;
                node = right;
                continue;
            }
        }

        let root = self.root;
        self[root].set_visited(false);
    }

    fn rb_tree_insert(&mut self, new_node: Interval) -> Ref {
        let new_node = self.slab.insert(new_node);
        self.rb_tree_insert_ref(new_node);
        new_node
    }

    fn rb_tree_insert_ref(&mut self, new_node: Ref) {
        let n = &mut self.slab[new_node];
        n.parent = SENTINEL;
        n.children = [SENTINEL; 2];
        if self.root == SENTINEL {
            n.set_color(BLACK);
            self.root = new_node;
            return;
        }

        self.tree_insert(new_node);
        let mut z = new_node;
        let mut p = self[z].parent;
        self.recompute_max_end_walk_to_root(p);

        while z != self.root && self[p].color() == RED {
            p = self[z].parent;
            let mut pp = self[p].parent;

            let dir = if self[pp].children[LEFT] == p { RIGHT } else { LEFT };

            let y = self[pp].children[dir];

            if self[y].color() == RED {
                self[p].set_color(BLACK);
                self[y].set_color(BLACK);
                self[pp].set_color(RED);
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
                self[p].set_color(BLACK);
                self[pp].set_color(RED);
                self.rotate(pp, dir);
            }
        }

        let root = self.root;
        self[root].set_color(BLACK);
    }

    fn tree_insert(&mut self, z: Ref) {
        let mut delta = 0isize;
        let mut x = self.root;
        let Interval { start: z_abs_start, end: z_abs_end, .. } = self[z];
        loop {
            let xn = &self[x];
            let cmp = interval_compare(
                z_abs_start, z_abs_end,
                xn.start + delta, xn.end + delta,
            );
            let [left, right] = xn.children;
            if matches!(cmp, Ordering::Less) {
                if left == SENTINEL {
                    let zn = &mut self[z];
                    zn.start -= delta;
                    zn.end -= delta;
                    zn.max_end -= delta;
                    self[x].children[LEFT] = z;
                    break;
                } else {
                    x = left;
                }
            } else {
                delta += xn.delta;
                if right == SENTINEL {
                    let zn = &mut self[z];
                    zn.start -= delta;
                    zn.end -= delta;
                    zn.max_end -= delta;
                    self[x].children[RIGHT] = z;
                    break;
                } else {
                    x = right;
                }
            }
        }

        let zn = &mut self[z];
        zn.parent = x;
        zn.children = [SENTINEL; 2];
        zn.set_color(RED);
    }

    fn rb_tree_delete(&mut self, z: usize) {
        let x: Ref;
        let y: Ref;

        let [left, right] = self[z].children;
        if left == SENTINEL {
            x = right;
            y = z;

            let z_delta = self[z].delta;
            let xn = &mut self[x];
            xn.delta += z_delta;
            xn.start += z_delta;
            xn.end += z_delta;
            let delta = xn.delta;
            self.set_request_normalize_delta(delta);
        } else if right == SENTINEL {
            x = left;
            y = z;
        } else {
            y = self.leftest(right);
            x = self[y].children[RIGHT];

            let y_delta = self[y].delta;
            let xn = &mut self[x];
            xn.delta += y_delta;
            xn.start += y_delta;
            xn.end += y_delta;
            let delta = xn.delta;
            self.set_request_normalize_delta(delta);

            let z_delta = self[z].delta;
            let yn = &mut self[y];
            yn.start += z_delta;
            yn.end += z_delta;
            yn.delta = z_delta;
            self.set_request_normalize_delta(z_delta);
        }

        if y == self.root {
            self.root = x;
            self[x].set_color(BLACK);

            self.detach(z);
            self.reset_sentinel();
            self.recompute_max_end(x);
            self[x].parent = SENTINEL;
            return;
        }

        let y_was_red = self[y].color();

        let y_parent = self[y].parent;
        let ypn = &mut self[y_parent];
        if y == ypn.children[LEFT] {
            ypn.children[LEFT] = x;
        } else {
            ypn.children[RIGHT] = x;
        }

        if y == z {
            self[x].parent = y_parent;
        } else {
            self[x].parent = if y_parent == z { y } else { y_parent };

            let (yn, zn) = self.slab.get2_mut(y, z).unwrap();
            yn.children = zn.children;
            yn.parent = zn.parent;
            yn.set_color(zn.color());

            if z == self.root {
                self.root = y;
            } else {
                let zp = self[z].parent;
                let zpn = &mut self[zp];
                if z == zpn.children[LEFT] {
                    zpn.children[LEFT] = y;
                } else {
                    zpn.children[RIGHT] = y;
                }
            }

            let [left, right] = self[y].children;
            if left != SENTINEL {
                self[left].parent = y;
            }
            if right != SENTINEL {
                self[right].parent = y;
            }
        }

        self.detach(z);

        if y_was_red {
            self.recompute_max_end_walk_to_root(self[x].parent);
            if y != z {
                self.recompute_max_end_walk_to_root(y);
                self.recompute_max_end_walk_to_root(self[y].parent);
            }
            self.reset_sentinel();
            return;
        }

        self.recompute_max_end_walk_to_root(x);
        self.recompute_max_end_walk_to_root(self[x].parent);
        if y != z {
            self.recompute_max_end_walk_to_root(y);
            self.recompute_max_end_walk_to_root(self[y].parent);
        }

        // RB-DELETE-FIXUP
        let mut x = x;
        while x != self.root && self[x].color() == BLACK {
            let p = self[x].parent;
            let dir = if x == self[p].children[LEFT] { RIGHT } else { LEFT };
            let mut w = self[p].children[dir];

            if self[w].color() == RED {
                self[w].set_color(BLACK);
                self[p].set_color(RED);
                self.rotate(p, dir ^ 1);
                // recompute w after the rotation of p
                w = self[p].children[dir];
            }
            let [wl, wr] = self[w].children;
            if self[wl].color() == BLACK && self[wr].color() == BLACK {
                self[w].set_color(RED);
                x = p;
            } else {
                let mut wc = self[w].children[dir]; // w child i care about
                let wo = self[w].children[dir ^ 1]; // w other child
                if self[wc].color() == BLACK {
                    self[wo].set_color(BLACK);
                    self[w].set_color(RED);
                    self.rotate(w, dir);
                    w = self[p].children[dir];

                    // recompute wc after the rotation of w
                    wc = self[w].children[dir];
                }

                let xp_color = self[p].color();
                self[w].set_color(xp_color);
                self[p].set_color(BLACK);
                self[wc].set_color(BLACK);
                self.rotate(p, dir ^ 1);
                x = self.root;
            }
        }

        self[x].set_color(BLACK);
        self.reset_sentinel();
    }

    fn leftest(&self, x: Ref) -> Ref {
        let mut x = x;
        while self[x].children[LEFT] != SENTINEL {
            x = self[x].children[LEFT];
        }
        x
    }

    fn reset_sentinel(&mut self) {
        let sentinel = &mut self[SENTINEL];
        sentinel.parent = SENTINEL;
        sentinel.delta = 0;
        sentinel.start = 0;
        sentinel.end = 0;
    }

    fn rotate(&mut self, x: Ref, dir: usize) {
        debug_assert!(dir == 0 || dir == 1);
        let dir = dir & 1;
        let y = self[x].children[dir ^ 1];

        // fix stats
        let delta = if dir == LEFT {
            let delta = self[x].delta;
            let yn = &mut self[y];
            yn.delta += delta;
            yn.start += delta;
            yn.end += delta;
            yn.delta
        } else {
            let delta = self[y].delta;
            let xn = &mut self[x];
            xn.delta -= delta;
            xn.start -= delta;
            xn.end -= delta;
            xn.delta
        };
        self.set_request_normalize_delta(delta);

        self[x].children[dir ^ 1] = self[y].children[dir];
        let y_child = self[y].children[dir];
        if y_child != SENTINEL {
            self[y_child].parent = x;
        }
        let parent = self[x].parent;
        self[y].parent = parent;
        if parent == SENTINEL {
            self.root = y;
        } else {
            let parent = &mut self[parent];
            let dir = if parent.children[LEFT] == x { LEFT } else { RIGHT };
            parent.children[dir] = y;
        }
        self[y].children[dir] = x;
        self[x].parent = y;

        self.recompute_max_end(x);
        self.recompute_max_end(y);
    }

    //#region max end computation
    fn compute_max_end(&self, x: Ref) -> isize {
        let node = &self[x];
        let mut max_end = node.end;
        let [left, right] = node.children;
        if left != SENTINEL {
            max_end = max_end.max(self[left].max_end);
        }
        if right != SENTINEL {
            max_end = max_end.max(self[right].max_end + node.delta);
        }
        max_end
    }
    fn recompute_max_end(&mut self, x: Ref) {
        self[x].max_end = self.compute_max_end(x);
    }
    fn recompute_max_end_walk_to_root(&mut self, mut x: Ref) {
        while x != SENTINEL {
            let max_end = self.compute_max_end(x);
            let node = &mut self[x];
            if node.max_end == max_end {
                return;
            }
            node.max_end = max_end;
            x = node.parent;
        }
    }
    //#endregion

    fn detach(&mut self, x: Ref) {
        let node = &mut self[x];
        node.parent = SENTINEL;
        node.children = [SENTINEL; 2];
    }

    fn set_request_normalize_delta(&mut self, delta: isize) {
        if !(MIN_SAFE_DELTA..=MAX_SAFE_DELTA).contains(&delta) {
            self.request_normalize_delta = true;
        }
    }
}

#[cfg(test)]
impl IntervalTree {
    pub(crate) fn is_valid(&self) {
        /*
         * properties
         * - root property: root is black
         * - leaf nodes (NULL) are black (pointless here given my sentinel is a NULL, not a real node)
         * - red property: children of a red node are black
         * - simple path from node to descendant leaf contains same number of black nodes
         */
        fn verify_black_height(rb: &IntervalTree, x: Ref) -> i32 {
            if x == SENTINEL {
                return 0;
            }
            let left_height = verify_black_height(rb, rb[x].children[0]);
            let right_height = verify_black_height(rb, rb[x].children[1]);

            assert!(
                left_height != -1 && right_height != -1 && left_height == right_height,
                "red-black properties have been violated!",
            );

            let add = if rb[x].color() == RED { 0 } else { 1 };
            left_height + add
        }

        fn verify_children_color(rb: &IntervalTree) -> bool {
            if rb.root == SENTINEL {
                return true;
            }
            let mut queue: std::collections::VecDeque<Ref> = Default::default();
            queue.push_front(rb.root);

            while !queue.is_empty() {
                let curr = queue.pop_front().unwrap();
                if curr == SENTINEL {
                    break;
                };
                let idx = curr;

                let l = rb[idx].children[0];
                let r = rb[idx].children[1];

                // red node must not have red children
                if rb[idx].color() == RED {
                    assert!(!rb[l].color() && !rb[r].color(), "red node has red children");
                }

                if l != SENTINEL {
                    queue.push_back(l);
                }
                if r != SENTINEL {
                    queue.push_back(r);
                }
            }

            true
        }

        fn verify_sums(rb: &IntervalTree, x: Ref) -> usize {
            if x == SENTINEL {
                return 0;
            }
            let idx = x;
            let node = &rb[idx];
            let left = verify_sums(rb, node.children[0]);
            let right = verify_sums(rb, node.children[1]);
            1 + left + right
        }

        assert_eq!(self[self.root].color(), BLACK); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
        let size = verify_sums(self, self.root);
        assert_eq!(size + 1, self.slab.len());
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use super::*;
    use crate::interval::Interval;

    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Range {
        start: isize,
        end: isize,
        front_advance: bool,
        rear_advance: bool,
    }
    impl From<&Interval> for Range {
        fn from(interval: &Interval) -> Self {
            Self {
                start: interval.cached_range.start,
                end: interval.cached_range.end,
                front_advance: interval.is_front_advance(),
                rear_advance: interval.is_rear_advance(),
            }
        }
    }

    impl Range {
        fn new(start: isize, end: isize, front_advance: bool, rear_advance: bool) -> Self {
            Self { start, end, front_advance, rear_advance }
        }

        fn simple(start: isize, end: isize) -> Self {
            Self::new(start, end, false, false)
        }
    }
    impl Ord for Range {
        fn cmp(&self, other: &Self) -> Ordering {
            if self.start == other.start {
                self.end.cmp(&other.end)
            } else {
                self.start.cmp(&other.start)
            }
        }
    }
    impl PartialOrd for Range {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    struct Oracle {
        intervals: Vec<Rc<RefCell<Range>>>,
    }

    impl Oracle {
        fn new() -> Self {
            Self { intervals: Vec::new() }
        }

        fn insert(&mut self, interval: Rc<RefCell<Range>>) {
            self.intervals.push(interval);
            self.intervals.sort();
        }

        fn delete(&mut self, interval: Rc<RefCell<Range>>) -> Rc<RefCell<Range>> {
            let pos = self.intervals.iter()
                .position(|x| Rc::ptr_eq(x, &interval))
                .expect("No such node");
            self.intervals.remove(pos)
        }

        fn search(&self, query: &Range) -> Vec<Range> {
            let mut result = Vec::new();
            for range in &self.intervals {
                let range = range.borrow();
                if range.start <= query.end && range.end >= query.start {
                    result.push(range.clone());
                }
            }
            result
        }

        fn accept_replace(&mut self, del_range: std::ops::Range<isize>, inserted: usize) {
            let offset = del_range.start;
            let deleted = del_range.len() as isize;
            for range in &mut self.intervals {
                let mut range = range.borrow_mut();
                if range.end < offset {
                    continue;
                }

                // Simple shift logic for ranges purely after the edit
                if range.start > offset + deleted {
                    let delta = (-deleted).strict_add_unsigned(inserted);
                    range.start += delta;
                    range.end += delta;
                    continue;
                }

                // Complex logic: Delegate to Interval implementation to mirror Tree behavior
                let mut temp_interval = Interval::new(
                    range.start..range.end,
                    range.front_advance, range.rear_advance,
                );
                temp_interval.accept_edit(offset..offset + deleted, inserted);

                range.start = temp_interval.start;
                range.end = temp_interval.end;
            }
        }

        fn assert_rc_2(&self) {
            for range in &self.intervals {
                assert_eq!(2, Rc::strong_count(range));
            }
        }
    }

    #[derive(Clone, Debug)]
    enum Operation {
        Insert { begin: isize, end: isize, front_advance: bool, rear_advance: bool },
        Delete(usize), // references index in TestState.vec_ids
        Change { id: usize, begin: isize, end: isize },
        Search { begin: isize, end: isize, assert_exact: bool },
        Replace { begin: isize, end: isize, text_length: usize },
    }

    struct TestState {
        oracle: Oracle,
        tree: IntervalTree,
        // Maps the index (used in operations) to the actual Tree Node ID (usize)
        tree_ids: Vec<Option<usize>>,
        oracle_nodes: Vec<Option<Rc<RefCell<Range>>>>,
        timestamp: u32,
    }

    impl TestState {
        fn new() -> Self {
            Self {
                oracle: Oracle::new(),
                tree: IntervalTree::new(),
                tree_ids: Vec::new(),
                oracle_nodes: Vec::new(),
                timestamp: 0,
            }
        }

        fn accept_operation(&mut self, op: Operation) {
            self.timestamp += 1;
            match op {
                Operation::Insert { begin, end, front_advance, rear_advance } => {
                    let node = Interval::new(begin..end, front_advance, rear_advance);

                    let tree_id = self.tree.insert(node);
                    self.tree_ids.push(Some(tree_id.get()));

                    let range = Range::new(begin, end, front_advance, rear_advance);
                    let range = Rc::new(RefCell::new(range));
                    self.oracle.insert(range.clone());
                    self.oracle_nodes.push(Some(range));

                    assert_eq!(self.oracle_nodes.len(), self.tree_ids.len());
                }
                Operation::Delete(id_idx) => {
                    let tree_id = self.tree_ids[id_idx].take().expect("Node already deleted");
                    let range = self.oracle_nodes[id_idx].take().expect("Node already deleted");

                    self.tree.delete(tree_id);
                    self.oracle.delete(range);
                }
                Operation::Change { id, begin, end } => {
                    let tree_id = self.tree_ids[id].take().expect("Node missing for change");

                    let old = self.tree.delete(tree_id);
                    let new_tree_id = self.tree.insert(Interval::new(
                        begin..end,
                        old.is_front_advance(),
                        old.is_rear_advance(),
                    ));
                    self.tree_ids[id] = Some(new_tree_id.get());

                    let range = self.oracle_nodes[id].take().expect("Oracle node missing");
                    let range = self.oracle.delete(range);
                    let mut r = range.borrow_mut();
                    r.start = begin;
                    r.end = end;
                    drop(r);
                    self.oracle.insert(range.clone());
                    self.oracle_nodes[id] = Some(range);
                }
                Operation::Search { begin, end, assert_exact } => {
                    let mut nodes_in_range: Vec<Range> = Vec::new();
                    self.tree.interval_search(begin..end, self.timestamp, |i| {
                        nodes_in_range.push(Range::from(&*i));
                    });

                    let mut expected = self.oracle.search(&Range::simple(begin, end));

                    nodes_in_range.sort();
                    expected.sort();

                    assert_eq!(
                        expected.clone().into_iter().map(|i| (i.start, i.end)).collect::<Vec<_>>(),
                        nodes_in_range.clone().into_iter().map(|i| (i.start, i.end)).collect::<Vec<_>>(),
                    );

                    if assert_exact {
                        assert_eq!(1, nodes_in_range.len());
                        assert_eq!(begin, expected[0].start);
                        assert_eq!(end, expected[0].end);
                    }
                }
                Operation::Replace { begin, end, text_length } => {
                    self.tree.accept_replace(begin..end, text_length);
                    self.oracle.accept_replace(begin..end, text_length);
                }
            }
            self.oracle.assert_rc_2();
            self.tree.is_valid();
        }

        fn get_existing_node_idx(&self, index: usize) -> usize {
            let mut curr_index = -1isize;
            for (i, node) in self.tree_ids.iter().enumerate() {
                if node.is_none() { continue; }
                curr_index += 1;
                if curr_index == index as isize {
                    return i;
                }
            }
            panic!("Invalid index");
        }

        fn run_test(ops: Vec<Operation>) -> IntervalTree {
            let mut state = TestState::new();
            for op in ops {
                state.accept_operation(op);
            }
            state.accept_operation(Operation::Search {
                begin: 0,
                end: isize::MAX,
                assert_exact: false
            });
            state.tree
        }
    }

    #[test]
    fn test_interval_tree() {
        TestState::run_test(vec![
            Operation::Insert { begin: 28, end: 35, front_advance: false, rear_advance: false },
            Operation::Search { begin: 30, end: 31, assert_exact: false },
            Operation::Insert { begin: 52, end: 54, front_advance: false, rear_advance: false },
            Operation::Search { begin: 30, end: 31, assert_exact: false },
            Operation::Search { begin: 53, end: 54, assert_exact: false },
            Operation::Insert { begin: 63, end: 69, front_advance: false, rear_advance: false },
            Operation::Search { begin: 30, end: 31, assert_exact: false },
            Operation::Search { begin: 53, end: 54, assert_exact: false },
            Operation::Search { begin: 64, end: 65, assert_exact: false },
        ]);
    }

    #[test]
    fn test_simple_edits() {
        // insert @ to the left
        TestState::run_test(vec![
            Operation::Insert { begin: 5, end: 10, front_advance: false, rear_advance: false },
            Operation::Replace { begin: 0, end: 0, text_length: 5 },
            Operation::Search { begin: 10, end: 15, assert_exact: true },
        ]);
        // insert @ to the right
        TestState::run_test(vec![
            Operation::Insert { begin: 5, end: 10, front_advance: false, rear_advance: false },
            Operation::Replace { begin: 15, end: 15, text_length: 5 },
            Operation::Search { begin: 5, end: 10, assert_exact: true },
        ]);
        // insert @ middle
        TestState::run_test(vec![
            Operation::Insert { begin: 5, end: 10, front_advance: false, rear_advance: false },
            Operation::Replace { begin: 7, end: 7, text_length: 5 },
            Operation::Search { begin: 5, end: 15, assert_exact: true },
        ]);
        // delete @ left
        TestState::run_test(vec![
            Operation::Insert { begin: 5, end: 10, front_advance: false, rear_advance: false },
            Operation::Replace { begin: 0, end: 5, text_length: 0 },
            Operation::Search { begin: 0, end: 5, assert_exact: true },
        ]);
    }

    #[test]
    fn test_delete() {
        TestState::run_test(vec![
            Operation::Insert { begin: 0, end: 5, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 5, end: 10, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 10, end: 15, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 15, end: 20, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 20, end: 25, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 25, end: 30, front_advance: false, rear_advance: false },
            Operation::Delete(0),
            Operation::Delete(2),
            Operation::Delete(1),
        ]);
    }

    #[test]
    fn test_edge_edits() {
        struct TestCase {
            op: Operation,
            case1: Operation,
            case2: Operation,
        }

        let run_comparative = |i1: Operation, i2: Operation, cases: Vec<TestCase>| {
            for test in cases {
                TestState::run_test(vec![i1.clone(), test.op.clone(), test.case1]);
                TestState::run_test(vec![i2.clone(), test.op.clone(), test.case2]);
            }
        };

        // Front Advance vs No Front Advance
        run_comparative(
            Operation::Insert { begin: 4, end: 8, front_advance: true, rear_advance: false },
            Operation::Insert { begin: 4, end: 8, front_advance: false, rear_advance: false },
            vec![
                TestCase {
                    op: Operation::Replace { begin: 4, end: 4, text_length: 4 },
                    case1: Operation::Search { begin: 8, end: 12, assert_exact: true },
                    case2: Operation::Search { begin: 4, end: 12, assert_exact: true },
                },
                TestCase {
                    op: Operation::Replace { begin: 2, end: 4, text_length: 0 },
                    case1: Operation::Search { begin: 2, end: 6, assert_exact: true },
                    case2: Operation::Search { begin: 2, end: 6, assert_exact: true },
                },
                TestCase {
                    op: Operation::Replace { begin: 2, end: 4, text_length: 4 }, // del + ins front
                    case1: Operation::Search { begin: 6, end: 10, assert_exact: true },
                    case2: Operation::Search { begin: 4, end: 10, assert_exact: true },
                },
            ]
        );
    }

    #[test]
    fn test_strange_interval() {
        let mut tree = TestState::run_test(vec![
            Operation::Insert { begin: 4, end: 4, front_advance: true, rear_advance: false },
            Operation::Replace { begin: 4, end: 4, text_length: 4 }
        ]);

        let mut results = Vec::new();
        tree.interval_search(0..isize::MAX, 0, |i| results.push(i.clone()));

        assert_eq!(1, results.len());
        assert_eq!(4, results[0].start);
        assert_eq!(4, results[0].end);
    }

    #[test]
    fn test_get_all() {
        let mut tree = TestState::run_test(vec![
            Operation::Insert { begin: 0, end: 0, front_advance: false, rear_advance: false },
            Operation::Insert { begin: 4, end: 4, front_advance: false, rear_advance: false }
        ]);

        let mut results = Vec::new();
        tree.interval_search(0..4, 0, |i| results.push(i.clone()));

        assert_eq!(2, results.len());
        assert_eq!(0, results[0].start);
        assert_eq!(4, results[1].start);
    }

    struct AutoTest {
        history: Vec<Operation>,
        state: TestState,
        rng: ChaCha8Rng,
        insert_cnt: usize,
        delete_cnt: usize,
        change_cnt: usize,
        max_interval_end: i32,
        apply_edits: bool,
    }

    impl AutoTest {
        fn new(seed: u64, max_interval_end: i32, max_inserts: usize, max_changes: usize, apply_edits: bool) -> Self {
            use rand::SeedableRng;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let insert_cnt = rng.random_range(1..max_inserts);
            let change_cnt = rng.random_range(10..max_changes);

            Self {
                history: Vec::new(),
                state: TestState::new(),
                rng,
                insert_cnt,
                delete_cnt: 0,
                change_cnt,
                max_interval_end,
                apply_edits,
            }
        }

        fn run_op(&mut self, op: Operation) {
            self.history.push(op.clone());
            self.state.accept_operation(op);
        }

        fn run(mut self) {
            while self.insert_cnt > 0 || self.delete_cnt > 0 || self.change_cnt > 0 {
                if self.insert_cnt > 0 {
                    self.do_random_insert();
                    self.insert_cnt -= 1;
                    self.delete_cnt += 1;
                } else if self.change_cnt > 0 {
                    self.do_random_change();
                    self.change_cnt -= 1;
                } else {
                    self.do_random_delete();
                    self.delete_cnt -= 1;
                }

                let begin = self.rng.random_range(0..self.max_interval_end) as isize;
                let end = self.rng.random_range(begin as i32..=self.max_interval_end) as isize;
                self.run_op(Operation::Search { begin: 0, end: isize::MAX, assert_exact: false });
                self.run_op(Operation::Search { begin, end, assert_exact: false });
            }
        }

        fn do_random_insert(&mut self) {
            let begin = self.rng.random_range(0..self.max_interval_end) as isize;
            let end = self.rng.random_range(begin as i32..self.max_interval_end + 1) as isize;
            self.run_op(Operation::Insert { begin, end, front_advance: false, rear_advance: false });
        }

        fn do_random_delete(&mut self) {
            let idx = self.rng.random_range(0..self.delete_cnt);
            let id = self.state.get_existing_node_idx(idx);
            self.run_op(Operation::Delete(id));
        }

        fn do_random_change(&mut self) {
            if self.apply_edits && self.rng.random_bool(0.5) {
                let offset = self.rng.random_range(0..self.max_interval_end) as isize;
                let (deletes, inserts) = if self.rng.random_bool(0.2) {
                     // Large range
                     (self.rng.random_range(0..self.max_interval_end / 2),
                      self.rng.random_range(0..self.max_interval_end / 2))
                } else {
                    (self.rng.random_range(0..10), self.rng.random_range(0..10))
                };
                self.run_op(Operation::Replace {
                    begin: offset, end: offset + deletes as isize,
                    text_length: inserts as usize,
                });
            } else {
                let idx = self.rng.random_range(0..self.delete_cnt);
                let id = self.state.get_existing_node_idx(idx);
                let begin = self.rng.random_range(0..self.max_interval_end) as isize;
                let end = self.rng.random_range(begin as i32..=self.max_interval_end) as isize;
                self.run_op(Operation::Change { id, begin, end });
            }
        }
    }

    #[test]
    fn test_auto_no_edit() {
        for i in 0..100 { // Reduced loop count for example, Java used 10000
            let test = AutoTest::new(i as u64, 100, 30, 30, false);
            test.run();
        }
    }

    #[test]
    fn test_auto_edit() {
        for i in 0..100 {
            let test = AutoTest::new(i as u64, 100, 30, 30, true);
            test.run();
        }
    }
}
