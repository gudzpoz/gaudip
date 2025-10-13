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

use crate::piece::{Sum, Summable, TransientRef};
use slab::Slab;
use std::cmp::Ordering;
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

    fn adopt(&mut self, parent: SafeRef, red: bool, left: Ref, right: Ref) {
        self[parent].rb.red = red;
        self[parent].rb.children = [left, right];
        self[parent].left_sum = if let Some(left) = left {
            self[left].rb.parent = Some(parent);
            self.calculate_sum(Some(left))
        } else {
            T::S::identity()
        };
        if let Some(right) = right {
            self[right].rb.parent = Some(parent);
        }
    }

    pub fn compact<F>(&mut self, mut f: F)
    where F: FnMut(&mut T, TransientRef, TransientRef) -> bool {
        // TODO: this is unexposed API because it's bad...
        //       we expose TransientRef and compacting will ruin them.
        // We only use this duplicate mut reference to index
        // parent/children nodes and never the node itself.
        // So the aliasing should be safe.
        let slab = {
            let slab = &mut self.slab;
            unsafe { (slab as *mut Slab<Node<T>>).as_mut() }.unwrap()
        };
        let mut root = self.root;
        self.slab.compact(|node, old, new| {
            if old == 0 { unreachable!(); }
            let old = SafeRef::new(old);
            let new = SafeRef::new(new).unwrap();
            if !f(&mut node.piece, TransientRef(old.unwrap()), TransientRef(new)) {
                return false;
            }
            if root == old {
                root.replace(new);
            }
            let node = &node.rb;
            let p_children = &mut unsafe {
                slab.get_unchecked_mut(ref_int(node.parent))
            }.rb.children;
            let dir = if old == p_children[LEFT] { LEFT } else { RIGHT };
            p_children[dir] = Some(new);
            if let Some(left) = node.children[LEFT] {
                unsafe { slab.get_unchecked_mut(left.get()) }.rb.parent = Some(new);
            }
            if let Some(right) = node.children[RIGHT] {
                unsafe { slab.get_unchecked_mut(right.get()) }.rb.parent = Some(new);
            }
            true
        });
        self.root = root;
    }

    /// Constructs a valid red-black tree from a sorted iterator
    ///
    /// The constructed tree is not merged with the existing tree.
    /// And the user is expected to use this function with care.
    ///
    /// It basically follows the `buildFromSorted` method from
    /// OpenJDK `TreeMap`.
    pub fn build_from_sorted<I>(&mut self, size: usize, nodes: &mut I) -> Option<(SafeRef, T::S)>
    where I: Iterator<Item = T> {
        fn build_from_sorted_rec<T: Summable, I>(
            // manual captures
            this: &mut RbSlab<T>, red_level: usize, nodes: &mut I,
            // recursion args
            level: usize, lo: usize, hi: usize,
        ) -> Option<(SafeRef, T::S)>
        where I: Iterator<Item = T> {
            if hi < lo {
                return None;
            }
            let mid = (lo + hi) >> 1;
            let left = if lo < mid {
                build_from_sorted_rec(
                    this, red_level, nodes,
                    level + 1, lo, mid - 1,
                )
            } else {
                None
            };

            let value = nodes.next().unwrap();
            let mut size = value.summarize();
            let mut node = Node::new(value);
            node.rb.red = level == red_level;
            let middle = this.insert(node);

            if let Some((left, left_sum)) = left {
                size.add_assign(&left_sum);
                this[middle].left_sum = left_sum;
                this[middle].rb.children[LEFT] = Some(left);
                this[left].rb.parent = Some(middle);
            }

            let right = build_from_sorted_rec(
                this, red_level, nodes,
                level + 1, mid + 1, hi,
            );
            if let Some((right, right_sum)) = right {
                size.add_assign(&right_sum);
                this[middle].rb.children[RIGHT] = Some(right);
                this[right].rb.parent = Some(middle);
            }

            Some((middle, size))
        }

        // self.compact();
        let red_level = (usize::BITS - 1 - (size + 1).leading_zeros()) as usize;
        build_from_sorted_rec(self, red_level, nodes, 0, 0, size - 1)
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
                debug_assert!(Some($x) == $tree.root());
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
pub(crate) use foreach_parent;

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
/// Various batch operations
impl<T: Summable> RbSlab<T> {
    fn black_height(&self, x: Ref) -> usize {
        let mut x = x;
        let mut height = 0;
        while let Some(xi) = x {
            if !self[xi].rb.red {
                height += 1;
            }
            x = self[xi].rb.children[0];
        }
        height
    }

    /// Concatenates the three pieces
    ///
    /// The `mid` argument is expected to be a leaf node.
    /// (The function just overwrites the children of `mid`.
    /// If it is not a leaf, you risk leaking memory.)
    ///
    /// The parent field of the returned node is set to sentinel.
    pub fn join(&mut self, left: Ref, mid: SafeRef, right: Ref) -> SafeRef {
        fn join_dir<T: Summable>(
            this: &mut RbSlab<T>,
            left: Ref, mid: SafeRef, right: Ref,
            dir: usize,
        ) -> (SafeRef, T::S) {
            // if (TL.color=black) and (TL.blackHeight=TR.blackHeight):
            //    return Node(TL,⟨k,red⟩,TR)
            let mut pair = [left, right];
            let dest = pair[dir ^ 1];
            let dest_n = &this[dest];
            if !dest_n.red && this.black_height(left) == this.black_height(right) {
                this.adopt(mid, true, left, right);
                return (mid, this.calculate_sum(Some(mid)));
            }
            // - dest is red: dest is not sentinel
            // - TL.blackHeight > TR.blackHeight: dest has higher height: not sentinel
            let dest = dest.unwrap();

            // T'=Node(TL.left,⟨TL.key,TL.color⟩,joinRightRB(TL.right,k,TR))
            pair[dir ^ 1] = dest_n.children[dir];
            let joined = join_dir(this, pair[0], mid, pair[1], dir);
            this[dest].rb.children[dir] = Some(joined.0);
            this[joined.0].rb.parent = Some(dest);
            let sum = if dir == LEFT {
                this[dest].left_sum = joined.1;
                this.calculate_sum(Some(dest))
            } else {
                let mut sum = joined.1;
                sum.add_assign(&this[dest].left_sum);
                sum.add_assign(&this[dest].piece.summarize());
                sum
            };

            // if (TL.color=black) and (T'.right.color=T'.right.right.color=red):
            //    T'.right.right.color=black;
            //    return rotateLeft(T')
            let dest_n = &this[dest].rb;
            if !dest_n.red && this[joined.0].rb.red {
                let rr = this[joined.0].rb.children[dir];
                let rrn = &mut this[rr];
                if rrn.red {
                    rrn.red = false;
                    // ideally we should set the parent of all values returned from
                    // join_dir to SENTINEL, but they almost always get rewritten later
                    // so we just don't bother. however, rotation might change the parent
                    // and let's set this here just in case.
                    this[dest].rb.parent = SENTINEL;
                    this.rotate(Some(dest), dir ^ 1);
                    return (joined.0, sum);
                }
            }
            // return T'
            (dest, sum)
        }
        debug_assert!(!self.slab[0].rb.red);
        let dir = match self.black_height(left).cmp(&self.black_height(right)) {
            Ordering::Less => LEFT,
            Ordering::Greater => RIGHT,
            Ordering::Equal => {
                self.adopt(mid, false, left, right);
                self.reset_sentinel();
                return mid;
            }
        };
        self[left].parent = SENTINEL;
        self[right].parent = SENTINEL;
        let joined = join_dir(self, left, mid, right, dir).0;
        if self[joined].rb.red && self[self[joined].rb.children[dir]].red {
            self[joined].rb.red = false;
        }
        self.reset_sentinel();
        joined
    }

    /// Splits the tree into two subtrees, at the given point.
    /// The supplied `at` will be at the left tree if `dir` is `LEFT`,
    /// and at the right tree otherwise.
    fn split_dir(&mut self, mut at: SafeRef, side: usize) -> (Ref, Ref) {
        // join overwrites the parent & children field,
        // so we need to store them before the join.
        let mut parent = self[at].rb.parent;
        let mut pair = self[at].rb.children;
        {
            let mut sided = [SENTINEL, SENTINEL];
            sided[side] = pair[side];
            pair[side] = Some(self.join(sided[0], at, sided[1]));
        }

        while let Some(p) = parent {
            let pn = &self[p];
            let mut children = pn.rb.children;
            let next = pn.rb.parent;
            let dir = if children[LEFT] == Some(at) { LEFT } else { RIGHT };
            {
                children[dir] = pair[dir ^ 1];
                pair[dir ^ 1] = Some(self.join(children[0], p, children[1]));
            }
            at = p;
            parent = next;
        }
        assert!(pair[side].is_some());
        self[pair[0]].parent = SENTINEL;
        self[pair[1]].parent = SENTINEL;
        (pair[0], pair[1])
    }

    /// Batch insert a bunch of nodes (`size >= 2`)
    ///
    /// The `size` must be the length of the iterator.
    pub fn batch_insert<I>(&mut self, at: SafeRef, side: usize, size: usize, mut values: I) -> T::S
    where I: Iterator<Item = T> {
        assert!(size >= 2);

        // our join function requires a left tree, a joiner node and a right tree.
        // since we need to join twice, we use the first piece from `values` as
        // the first joiner node, and the last piece as the second joiner node.

        // first joiner
        let Some(first) = values.next() else { return T::S::identity() };
        let mut sum = first.summarize();
        let first = self.insert(Node::new(first));

        // everything in between
        let (new, mid_sum) =
            if size > 2 && let Some((new, mid_sum)) = self.build_from_sorted(
                size - 2, &mut values
            ) {
                (Some(new), mid_sum)
            }
            else {
                (SENTINEL, T::S::identity())
            };
        sum.add_assign(&mid_sum);

        // second joiner
        let last = values.next().unwrap();
        sum.add_assign(&last.summarize());
        let last = self.insert(Node::new(last));

        let (left, right) = self.split_dir(at, side ^ 1);
        let left = self.join(left, first, new);
        let root = self.join(Some(left), last, right);
        self[root].rb.parent = SENTINEL;
        self[root].rb.red = false;
        self.reset_sentinel();
        self.root = Some(root);
        sum
    }

    /// Batch-delete a bunch of nodes (`from..=to`, inclusive)
    ///
    /// Please ensure that `from` is to the left of `to`. Otherwise,
    /// it deletes `(to.next_piece())..from` (that is, both sides excluded).
    pub fn batch_delete<F>(&mut self, from: SafeRef, to: SafeRef, mut dropper: F) where F: FnMut(T) {
        let (mut left, del_) = self.split_dir(from, RIGHT);
        let (mut del, mut right) = self.split_dir(to, LEFT);
        if left != SENTINEL && (self[left].parent != SENTINEL || left == del || left == right) {
            // `from` is to the right of `to`
            left = del;
            del = right;
            right = del_;
        }
        self.drop_tree(del, &mut dropper);
        let root = match (left, right) {
            (SENTINEL, SENTINEL) => SENTINEL,
            (SENTINEL, right) => right,
            (left, SENTINEL) => left,
            (Some(left), Some(right)) => {
                let edge = self.edge(right, LEFT);
                let (joiner, right) = self.split_dir(edge, LEFT);
                debug_assert!(joiner == Some(edge));
                Some(self.join(Some(left), edge, right))
            }
        };
        self[root].parent = SENTINEL;
        self[root].red = false;
        self.root = root;
        self.reset_sentinel();
    }

    fn drop_tree<F>(&mut self, root: Ref, dropper: &mut F) where F: FnMut(T) {
        let Some(root) = root else { return };
        let children = self[root].rb.children;
        let node = self.slab.remove(root.get());
        dropper(node.piece);
        self.drop_tree(children[LEFT], dropper);
        self.drop_tree(children[RIGHT], dropper);
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

        fn verify_sums<T: Summable>(rb: &RbSlab<T>, x: Ref) -> (T::S, usize) {
            let Some(idx) = x else {
                return (T::S::identity(), 0);
            };
            let node = &rb[idx];
            let left = verify_sums(rb, node.rb.children[0]);
            let right = verify_sums(rb, node.rb.children[1]);
            assert!(node.left_sum == left.0);
            let mut sum = node.piece.summarize();
            sum.add_assign(&left.0);
            sum.add_assign(&right.0);
            (sum, 1 + left.1 + right.1)
        }

        assert!(!self[self.root].red); // root is black
        verify_children_color(self);
        verify_black_height(self, self.root);
        let (sum, size) = verify_sums(self, self.root);
        assert_eq!(size + 1, self.slab.len());
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{BaseMetric, CursorPos};
    use crate::roperig::Rope;
    use crate::roperig_test::Alphabet;
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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

    #[test]
    fn test_join_split_special_case() {
        fn new_abc() -> (RbSlab<Alphabet>, SafeRef, SafeRef, SafeRef) {
            let mut rb: RbSlab<Alphabet> = RbSlab::new();
            let left = rb.insert(Node::new('a'.into()));
            assert_eq!(1, left.get());
            let mid = rb.insert(Node::new('b'.into()));
            assert_eq!(2, mid.get());
            let right = rb.insert(Node::new('c'.into()));
            assert_eq!(3, right.get());
            let mid = rb.join(Some(left), mid, Some(right));
            rb.set_root(Some(mid));
            assert_eq!(2, mid.get());
            assert_eq!(3, rb.is_valid().len());
            (rb, left, mid, right)
        }

        // join already connected nodes
        let (mut rb, left, mid, right) = new_abc();
        let mid = rb.join(Some(left), mid, Some(right));
        assert_eq!(2, mid.get());
        assert_eq!(3, rb.is_valid().len());

        // join sentinel left
        let (mut rb, left, mid, right) = new_abc();
        let mid = rb.join(SENTINEL, mid, Some(right)); // left disconnected
        let mid = rb.join(SENTINEL, left, Some(mid)); // restored
        assert_eq!(2, mid.get());
        assert_eq!(3, rb.is_valid().len());

        // join sentinel right
        let (mut rb, left, mid, right) = new_abc();
        let mid = rb.join(Some(left), mid, SENTINEL); // right disconnected
        let mid = rb.join(Some(mid), right, SENTINEL); // restored
        assert_eq!(2, mid.get());
        assert_eq!(3, rb.is_valid().len());

        // split + join
        for i in left.get()..=right.get() {
            let mut rb = new_abc().0;
            let i = NonZero::new(i).unwrap();
            let (l, _) = rb.split_dir(i, RIGHT);
            let (mid, r) = rb.split_dir(i, LEFT);
            assert_eq!(rb[mid].children, [SENTINEL, SENTINEL]);
            let mid = rb.join(l, mid.unwrap(), r);
            assert_eq!(2, mid.get());
            rb[mid].rb.red = false;
            assert_eq!(3, rb.is_valid().len());
        }
    }

    #[test]
    fn test_many_split_join() {
        for count in [10, 100, 1000, 5000] {
            let mut rb: RbSlab<Alphabet> = RbSlab::new();
            rb.root = rb.build_from_sorted(
                count,
                &mut (0..count).map(|i| {
                    char::from_u32(('a' as usize + i) as u32).unwrap().into()
                }),
            ).unwrap().0.into();
            assert_eq!(count, rb.is_valid().len());
            assert_eq!(count + 1, rb.slab.len());
            for i in 0..count {
                let i = NonZero::new(i + 1).unwrap();
                let (l, _) = rb.split_dir(i, RIGHT);
                let (mid, r) = rb.split_dir(i, LEFT);
                assert_eq!(rb[mid].children, [SENTINEL, SENTINEL]);
                assert_eq!(mid, Some(i));
                let mid = rb.join(l, i, r);
                rb.root = Some(mid);
                rb[mid].rb.red = false;
                assert_eq!(count, rb.is_valid().len());
            }
        }
    }

    #[test]
    fn test_batch_insert_start() {
        let mut rb: Rope<Alphabet> = Rope::default();
        rb.insert(0, "a".into());
        rb.insert(1, "b".into());
        rb.insert(2, "c".into());

        let c = rb.cursor::<BaseMetric>(0).unwrap().inner();
        rb.insert_many_before(
            Some(&c),
            3,
            "def".chars().map(|c| c.to_string().into()),
        );
        assert_eq!("defabc", rb.substring(0, 6));
    }

    #[test]
    fn test_batch_insert_mid() {
        let mut rb: Rope<Alphabet> = Rope::default();
        rb.insert(0, "a".into());
        rb.insert(1, "b".into());
        rb.insert(2, "c".into());
        rb.insert(3, "d".into());

        let c = rb.cursor::<BaseMetric>(2).unwrap().inner();
        rb.insert_many_after(
            Some(&c),
            3,
            "def".chars().map(|c| c.to_string().into()),
        );
        assert_eq!("abdefc", rb.substring(0, 6));
    }

    #[test]
    fn test_batch_insert() {
        let mut rb: Rope<Alphabet> = Rope::default();
        rb.insert(0, "a".into());
        rb.insert(1, "b".into());
        rb.insert(2, "c".into());
        assert_eq!("abc", rb.substring(0, 3));
        let count = 10000;
        let chars: Vec<String> = (0..count).map(|i| {
            char::from_u32('0' as u32 + i as u32).unwrap().to_string()
        }).collect();
        let c = rb.cursor::<BaseMetric>(2).unwrap().inner();
        rb.insert_many_after(
            Some(&c),
            count,
            chars.iter().map(|c| c.clone().into()),
        );

        rb.is_valid();
        assert_eq!(
            "ab".to_string() + &chars.join("") + "c",
            rb.substring(0, 3 + count),
        );
        assert_eq!(3 + count, rb.len());

        let start = rb.cursor::<BaseMetric>(2).unwrap().inner();
        let end = rb.cursor::<BaseMetric>(rb.len() - 1).unwrap().inner();
        rb.delete_many_to(start, end);

        rb.is_valid();
        assert_eq!(2, rb.len());
        assert_eq!("ac", rb.substring(0, 2));

        let inner = rb.compact();
        assert_eq!(3, inner.slab.len());
        assert_eq!('a', inner.slab[1].piece.0);
        assert_eq!('c', inner.slab[2].piece.0);
    }

    #[test]
    fn test_batch_delete_some() {
        let mut rb: Rope<Alphabet> = Rope::default();
        rb.insert(0, "a".into());
        rb.insert(1, "b".into());
        rb.insert(2, "c".into());
        rb.insert(3, "d".into());
        rb.insert(4, "e".into());
        let start = rb.cursor::<BaseMetric>(1).unwrap().inner();
        let end = rb.cursor::<BaseMetric>(2).unwrap().inner();
        rb.is_valid();
        rb.delete_many_to(start, end);
        rb.is_valid();
        assert_eq!("cde", rb.substring(0, 3));
    }

    #[test]
    fn test_batch_delete_all() {
        for reverse in [false, true] {
            let mut rb: Rope<Alphabet> = Rope::default();
            rb.insert(0, "a".into());
            rb.insert(1, "b".into());
            rb.insert(2, "c".into());
            let start = rb.cursor::<BaseMetric>(0).unwrap().inner();
            let end = rb.cursor::<BaseMetric>(rb.len()).unwrap().inner();
            let (start, end) = if reverse {
                (end, start)
            } else {
                (start, end)
            };
            rb.is_valid();
            rb.delete_many_to(start, end);
            rb.is_valid();
            if reverse {
                assert_eq!("ac", rb.substring(0, rb.len()));
            } else {
                assert_eq!("", rb.substring(0, rb.len()));
                assert!(rb.is_empty());
            }
        }
    }

    #[test]
    fn test_batch_many() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut expected = String::new();
        let mut rb: Rope<Alphabet> = Rope::default();
        // batch api never merges nodes
        for _ in 0..20000 {
            let insert = rb.is_empty() || rng.random_bool(0.5);
            if insert {
                let s: String = (0..rng.random_range(8..32))
                    .map(|_| rng.random_range('a'..='z')).collect();
                let at = rng.random_range(0..=expected.len());
                expected.insert_str(at, &s);
                let cursor = rb.cursor::<BaseMetric>(at).map(|c| c.inner());
                let c = cursor.as_ref();
                if at == 0 {
                    rb.insert_many_before(c, s.len(), s.chars().map(|c| c.into()));
                } else {
                    rb.insert_many_after(c, s.len(), s.chars().map(|c| c.into()));
                }
            } else {
                let from = rng.random_range(0..expected.len());
                let to = rng.random_range((from+1)..=expected.len());
                expected.drain(from..to);
                let mut c1 = rb.cursor::<BaseMetric>(from).unwrap();
                if from != 0 {
                    c1.next_piece();
                }
                let c1 = c1.inner();
                let c2 = rb.cursor::<BaseMetric>(to).unwrap().inner();
                rb.delete_many_to(c1, c2);
            }
            rb.is_valid();
            if expected.is_empty() {
                assert!(rb.is_empty());
            } else {
                let from = rng.random_range(0..expected.len());
                let to = rng.random_range(from..=expected.len());
                assert_eq!(expected.len(), rb.len());
                assert_eq!(&expected[from..to], rb.substring(from, to));
            }
        }
    }
}
