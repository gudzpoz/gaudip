use crate::metrics::{Measured, Metric};
use crate::piece::{Sum, Summable};
use crate::rb_base::{foreach_parent, Node, RbSlab, Ref, SafeRef, LEFT, RIGHT};
use std::ops::Range;

/// A rope implementation with opaque pieces
///
/// Querying and modifications are mostly done through [PartialCursorPos].
pub struct RopeBase<T: Summable> {
    pub(crate) tree: RbSlab<T>,
    pub(crate) sum: T::S,
}
impl<T: Summable> Default for RopeBase<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A position in a rope, with partially converted metrics
pub struct ConvertedPosition<'a, T: Summable, Pos: Metric<T>, Rem: Metric<T>> {
    /// The piece starting at [Self::piece_position]
    pub piece: Option<&'a T>,
    /// The absolute position of the current piece, in converted metric `Pos`
    pub piece_position: Measured<T, Pos>,
    /// The relative position into the current piece, in unconverted metric `Rem`
    pub offset_in_piece: Measured<T, Rem>,
}
impl<T: Summable> RopeBase<T> {
    /// Creates an empty rope
    pub fn new() -> Self {
        Self {
            tree: RbSlab::new(),
            sum: T::S::identity(),
        }
    }

    /// Batch insert multiple values at consecutive nodes
    pub fn init<I: ExactSizeIterator<Item=T>>(&mut self, mut nodes: I) {
        assert_eq!(self.sum.len(), 0);
        assert_eq!(None, self.tree.root());
        let tree = self.tree.build_from_sorted(&mut nodes);
        let Some((root, sum)) = tree else { return };
        self.tree.set_root(Some(root));
        self.sum = sum;
    }

    /// Check if the rope contains nothing
    pub fn is_empty(&self) -> bool {
        self.tree.root().is_none()
    }

    /// Returns the length of the rope in [crate::metrics::BaseMetric]
    ///
    /// Note that we allow zero-length nodes, like markers.
    /// So zero-length does not imply an empty tree.
    /// Use [Self::is_empty] for empty checking.
    pub fn base_len(&self) -> usize {
        self.sum.len()
    }
    /// Return the measurement of the whole tree
    pub fn len<M: Metric<T>>(&self) -> usize {
        M::measure(&self.sum)
    }

    /// Converts offsets from one measurement to another
    pub fn convert_metrics<From: Metric<T>, To: Metric<T>>(
        &'_ self, measurement: usize,
    ) -> ConvertedPosition<'_, T, To, From> {
        let acc = accumulate_until_metric(
            &self.tree, self.tree.root(), measurement,
            |v, sum| v + To::measure(sum),
            0,
        );
        let Accumulation { node, remaining, value } = acc;
        ConvertedPosition {
            piece: node.map(|idx| &self.tree[idx].piece),
            piece_position: Measured::new(value),
            offset_in_piece: remaining,
        }
    }

    /// Returns a cursor pointing to the given position
    pub fn cursor_at<M: Metric<T>>(&self, measurement: usize) -> Option<PartialCursorPos<T, M>> {
        let acc = accumulate_until_metric(
            &self.tree, self.tree.root(), measurement, |_, _| (), (),
        );
        let Accumulation { node, remaining, .. } = acc;
        node.map(|node| PartialCursorPos { node, offset_in_piece: remaining })
    }

    /// Accumulates until the given position
    pub fn accumulate<M: Metric<T>, V, F: FnMut(V, &T::S) -> V>(
        &self, measurement: usize, f: F, init: V,
    ) -> (V, Option<PartialCursorPos<T, M>>) {
        let acc = accumulate_until_metric::<T, M, V, F>(
            &self.tree, self.tree.root(), measurement, f, init,
        );
        (acc.value, acc.node.map(|n| PartialCursorPos::new(n, acc.remaining.value)))
    }

    /// Returns a cursor and absolute [T::S] offset of the node
    pub fn cursor_and_pos_at<M: Metric<T>>(&self, measurement: usize) -> (Option<PartialCursorPos<T, M>>, T::S) {
        let mut sum = T::S::identity();
        let acc = accumulate_until_metric(
            &self.tree, self.tree.root(), measurement,
            |_, v| sum.add_assign(v), (),
        );
        let Accumulation { node, remaining, .. } = acc;
        (node.map(|node| PartialCursorPos { node, offset_in_piece: remaining }), sum)
    }

    pub(crate) fn rb_insert(&mut self, node: Ref, piece: T, dir: usize) -> SafeRef {
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

    pub(crate) fn recompute_metadata(&mut self) {
        self.sum = self.tree.calculate_sum(self.tree.root());
    }

    #[cfg(test)]
    pub(crate) fn is_valid(&self) {
        let sum = self.tree.is_valid();
        assert!(sum == self.sum, "len: {} != {}", sum.len(), self.sum.len());
    }
}

/// A cursor pointing into an opaque piece in a rope tree
///
/// The user is responsible for ensuring to pass the correct
/// [RopeBase] reference to the member functions.
pub struct PartialCursorPos<T: Summable, M: Metric<T>> {
    pub(crate) node: SafeRef,
    offset_in_piece: Measured<T, M>,
}
impl<T: Summable, M: Metric<T>> Clone for PartialCursorPos<T, M> {
    fn clone(&self) -> Self {
        Self::new(self.node, self.offset_in_piece.value)
    }
}
/// Node API
impl<T: Summable, M: Metric<T>> PartialCursorPos<T, M> {
    /// Returns a new cursor pointing to the given position
    pub fn new(node: SafeRef, offset_in_piece: usize) -> Self {
        Self { node, offset_in_piece: Measured::new(offset_in_piece) }
    }

    /// Returns a new cursor moved by `measurement` from the current position,
    /// or `None` if out of range
    pub fn navigate(&self, tree: &RopeBase<T>, measurement: isize) -> Option<Self> {
        self.navigate_dir(
            &tree.tree, measurement.unsigned_abs(),
            if measurement < 0 { LEFT } else { RIGHT },
        )
    }
    /// Returns a new cursor moved by `measurement` from the current position,
    /// or `None` if out of range
    pub(crate) fn navigate_dir(&self, tree: &RbSlab<T>, measurement: usize, dir: usize) -> Option<Self> {
        if dir == LEFT {
            rel_node_left::<T, M>(tree, self, measurement)
        } else {
            rel_node_right::<T, M>(tree, self, measurement)
        }
    }
    /// Returns a new cursor pointing to the next piece
    pub fn next_piece(&self, tree: &RopeBase<T>) -> Option<Self> {
        tree.tree.next(self.node, RIGHT).map(|next| PartialCursorPos {
            node: next,
            offset_in_piece: Measured::new(0),
        })
    }
    /// Returns a new cursor pointing to the previous piece
    pub fn prev_piece(&self, tree: &RopeBase<T>) -> Option<Self> {
        tree.tree.next(self.node, LEFT).map(|prev| PartialCursorPos {
            node: prev,
            offset_in_piece: Measured::new(0),
        })
    }
    /// Returns a near-by cursor
    pub fn nearby_piece(&self, tree: &RopeBase<T>) -> Option<Self> {
        if let Some(prev) = self.prev_piece(tree) {
            Some(PartialCursorPos {
                node: prev.node,
                offset_in_piece: Measured::new(tree.tree[prev.node].piece.len()),
            })
        } else {
            self.next_piece(tree)
        }
    }

    /// Returns the absolute position of the cursor in metric `M`
    pub fn position(&self, tree: &RopeBase<T>) -> usize {
        let mut sum = M::measure(&tree.tree[self.node].left_sum) + self.offset_in_piece.value;
        let x = self.node;
        let _: Option<()> = foreach_parent!(({ p: pn } of { x: xn } in tree.tree) {
            if pn.rb.children[RIGHT] == Some(x) {
                sum += M::measure(&pn.left_sum);
                sum += M::measure(&pn.piece.summarize());
            }
        });
        sum
    }
    /// Returns the offset within the current node (relative to node start)
    pub fn offset(&self) -> Measured<T, M> {
        self.offset_in_piece
    }
    /// Returns the absolute position of the starting position of the current node
    pub fn node_start(&self, tree: &RopeBase<T>) -> T::S {
        let mut sum = tree.tree[self.node].left_sum;
        let x = self.node;
        let _: Option<()> = foreach_parent!(({ p: pn } of { x: xn } in tree.tree) {
            if pn.rb.children[RIGHT] == Some(x) {
                sum.add_assign(&pn.left_sum);
                sum.add_assign(&pn.piece.summarize());
            }
        });
        sum
    }

    /// Returns a new cursor with a different offset into the same node, in metric `N`
    pub fn with_offset<N: Metric<T>>(&self, offset_in_piece: usize) -> PartialCursorPos<T, N> {
        PartialCursorPos {
            node: self.node,
            offset_in_piece: Measured::new(offset_in_piece),
        }
    }

    /// Returns the piece at the current position
    pub fn get<'a>(&self, tree: &'a RopeBase<T>) -> &'a T {
        &tree.tree[self.node].piece
    }
    /// Returns the piece at the current position as mutable
    pub fn get_mut<'a>(&self, tree: &'a mut RopeBase<T>) -> &'a mut T {
        &mut tree.tree[self.node].piece
    }
    /// Returns true if two cursor points to the same piece
    pub fn is_same_piece(&self, other: &Self) -> bool {
        self.node == other.node
    }

    /// Updates the metadata of parent nodes and moves the cursor forward
    ///
    /// Note that it does not update the metadata of the tree ([RopeBase.sum]).
    pub(crate) fn node_update(&self, tree: &mut RopeBase<T>, delta: &T::S) -> Self {
        tree.tree.update_metadata(self.node, delta);
        self.with_offset(self.offset_in_piece.value.wrapping_add(delta.len()))
    }
    /// Updates the piece metadata at the current position
    ///
    /// This must be called after modifying the piece.
    pub fn update(&self, tree: &mut RopeBase<T>, delta: &T::S) {
        self.node_update(tree, delta);
        tree.sum.add_assign(delta);
    }

    /// Inserts a new piece at the left of this piece
    pub fn insert_left(&self, tree: &mut RopeBase<T>, piece: T) -> PartialCursorPos<T, M> {
        tree.sum.add_assign(&piece.summarize());
        let new = tree.rb_insert(Some(self.node), piece, LEFT);
        PartialCursorPos { node: new, offset_in_piece: Measured::new(0) }
    }
    /// Inserts a new piece at the right of this piece
    pub fn insert_right(&self, tree: &mut RopeBase<T>, piece: T) -> PartialCursorPos<T, M> {
        tree.sum.add_assign(&piece.summarize());
        let new = tree.rb_insert(Some(self.node), piece, RIGHT);
        PartialCursorPos { node: new, offset_in_piece: Measured::new(0) }
    }

    /// Deletes the piece at the current position
    pub fn delete(self, tree: &mut RopeBase<T>) -> T {
        let value = tree.tree.delete(self.node);
        tree.sum.sub_assign(&value.summarize());
        value
    }

    /// Iterate over the next `len`
    pub fn for_range(
        &self, tree: &RopeBase<T>, mut len: usize,
        mut f: impl FnMut(&T, Range<usize>) -> bool,
    ) {
        let mut i = Some(self.node);
        while len > 0 && let Some(idx) = i {
            let offset = if idx == self.node {
                self.offset_in_piece.value
            } else {
                0
            };
            let piece = &tree.tree[idx].piece;
            let end = M::measure(&piece.summarize());
            let end_off = (offset + len).min(end);
            let range = offset..end_off;
            len -= range.len();
            if !f(piece, range) {
                break;
            }
            i = tree.tree.next(idx, RIGHT);
        }
    }
}
/// Batch API
impl<T: Summable, M: Metric<T>> PartialCursorPos<T, M> {
    /// Batch inserts nodes to one side of this node
    ///
    /// This function ignores the [Self::offset] value.
    fn insert_many<I>(
        &self, tree: &mut RopeBase<T>, dir: usize, pieces: &mut I,
    ) where I: ExactSizeIterator<Item = T> {
        let size = pieces.len();
        if size == 0 {
            return;
        }
        if size == 1 {
            let Some(piece) = pieces.next() else { return };
            tree.sum.add_assign(&piece.summarize());
            tree.rb_insert(Some(self.node), piece, dir);
            return;
        }
        let sum = tree.tree.batch_insert(self.node, dir, pieces);
        tree.sum.add_assign(&sum);
    }

    /// Batch inserts a sequence of pieces *after this piece*
    ///
    /// Please note that the insertion point is *not at this cursor*,
    /// but after this whole piece. The user might need to manually
    /// split the current piece.
    pub fn insert_many_after<I>(
        &self, tree: &mut RopeBase<T>, pieces: &mut I,
    ) where I: ExactSizeIterator<Item = T> {
        self.insert_many(tree, RIGHT, pieces);
    }

    /// Batch inserts a sequence of pieces *before this piece*
    ///
    /// See [Self::insert_many_after] for more details.
    pub fn insert_many_before<I>(
        &self, tree: &mut RopeBase<T>, pieces: &mut I,
    ) where I: ExactSizeIterator<Item = T> {
        self.insert_many(tree, LEFT, pieces);
    }

    /// Batch deletes a range of pieces from this piece to `to` (inclusive).
    pub fn delete_many_to<F>(self, tree: &mut RopeBase<T>, to: Self, dropper: F)
    where F: FnMut(T) {
        if self.node == to.node {
            self.delete(tree);
            return;
        }
        tree.tree.batch_delete(self.node, to.node, dropper);
        tree.recompute_metadata();
    }
}

/// Generic (partially-)accumulated value until a given position
pub(crate) struct Accumulation<V, T: Summable, M: Metric<T>> {
    node: Ref,
    value: V,
    remaining: Measured<T, M>,
}
impl<V, T: Summable, M: Metric<T>> Accumulation<V, T, M> {
    fn new(node: Ref, value: V, remaining: usize) -> Self {
        Self {
            node,
            value,
            remaining: Measured::new(remaining),
        }
    }
}
#[inline]
pub(crate) fn accumulate_until_metric<T: Summable, M: Metric<T>, V, F: FnMut(V, &T::S) -> V>(
    tree: &RbSlab<T>, mut x: Ref, mut offset: usize, mut f: F, init: V,
) -> Accumulation<V, T, M> {
    if let Some(x) = x && offset == 0 {
        let node = tree.edge(x, LEFT);
        return Accumulation::new(Some(node), init, 0);
    }
    let mut v = init;
    while let Some(xi) = x {
        let n = &tree[xi];
        let left_len = M::measure(&n.left_sum);
        if left_len >= offset && n.rb.children[0].is_some() {
            x = n.rb.children[0];
        } else {
            v = f(v, &n.left_sum);
            let n_len = M::measure(&n.piece.summarize());
            let pre_len = left_len + n_len;
            if pre_len >= offset {
                offset -= left_len;
                debug_assert!((offset == 0) == (n_len == 0));
                return Accumulation::new(Some(xi), v, offset);
            } else {
                offset -= pre_len;
                v = f(v, &n.piece.summarize());
                x = n.rb.children[1];
            }
        }
    }
    Accumulation::new(None, v, offset)
}

pub(crate) fn rel_node_at_metric<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, x: Ref, offset: usize,
) -> Option<PartialCursorPos<T, M>> {
    let acc = accumulate_until_metric(tree, x, offset, |_, _| {}, ());
    let Accumulation { node: Some(node), remaining, .. } = acc else {
        return None;
    };
    Some(PartialCursorPos { node, offset_in_piece: remaining })
}

pub(crate) fn rel_node_left<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, from: &PartialCursorPos<T, M>, metric_offset: usize,
) -> Option<PartialCursorPos<T, M>> {
    let x = from.node;
    if from.offset_in_piece.value > metric_offset {
        return Some(PartialCursorPos {
            node: x,
            offset_in_piece: from.offset_in_piece - metric_offset,
        });
    }

    let mut back_offset = metric_offset - from.offset_in_piece.value;
    let mut x = Some(x);
    while let Some(xi) = x {
        let n = &tree[xi];
        let left = M::measure(&n.left_sum);
        if left > back_offset {
            return rel_node_at_metric::<T, M>(tree, n.rb.children[0], left - back_offset);
        }
        back_offset -= left;

        let mut p = n.rb.parent;
        while let Some(pi) = p {
            let pn = &tree[pi];
            if pn.rb.children[RIGHT] == x {
                let len = M::measure(&pn.piece.summarize());
                if len > back_offset {
                    return Some(PartialCursorPos { node: pi, offset_in_piece: Measured::new(len - back_offset) });
                }
                back_offset -= len;
                break;
            }
            x = p;
            p = pn.rb.parent;
        }
        x = p;
    }
    if back_offset == 0 {
        return Some(PartialCursorPos {
            node: tree.edge(tree.root()?, LEFT),
            offset_in_piece: Measured::new(0),
        });
    }
    None
}
pub(crate) fn rel_node_right<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, from: &PartialCursorPos<T, M>, metric_offset: usize,
) -> Option<PartialCursorPos<T, M>> {
    let mut x = from.node;
    let n = &tree[x];
    let len = M::measure(&n.piece.summarize());
    if len >= from.offset_in_piece.value + metric_offset {
        return Some(PartialCursorPos {
            node: x,
            offset_in_piece: from.offset_in_piece + metric_offset,
        });
    }

    let mut forward_offset = metric_offset + from.offset_in_piece.value + M::measure(&n.left_sum);
    let mut p = n.rb.parent;
    while let Some(pi) = p {
        let pn = &tree[pi];
        let left = M::measure(&pn.left_sum);
        if pn.rb.children[0] == Some(x) {
            if left >= forward_offset {
                return rel_node_at_metric::<T, M>(tree, Some(x), forward_offset);
            }
        } else {
            forward_offset += left + M::measure(&pn.piece.summarize());
        }
        p = pn.rb.parent;
        x = pi;
    }
    rel_node_at_metric::<T, M>(tree, tree.root(), forward_offset)
}
