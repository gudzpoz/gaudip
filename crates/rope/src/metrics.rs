use crate::piece::{Sum, Summable};
use crate::rb_base::{RbSlab, Ref, SafeRef, LEFT, RIGHT};
use std::marker::PhantomData;
use std::ops::Neg;

/// A metric system for rope pieces
///
/// For the conceptual background, see the [rope science series].
///
/// [rope science series]: https://xi-editor.io/docs/rope_science_02.html
///
/// ## Base Metric
///
/// We provide a [BaseMetric] as the metric in base units of the rope.
pub trait Metric<T: Summable> {
    /// Returns the measurement of a [Sum]
    fn measure(sum: &T::S) -> usize;
    /// Converts from base units (as is returned by [Sum::len]) to this measurement
    ///
    /// This should truncate to the nearest previous metric unit boundary.
    fn from_base_units(piece: &T, base_units: usize) -> usize;
    /// Converts from this measurement to base units (as is returned by [Sum::len])
    #[allow(clippy::wrong_self_convention)]
    fn to_base_units(piece: &T, measurement: usize) -> usize;
    /// Returns the base offset of when moving from `base_offset` by
    /// `delta_metric` metric units.
    ///
    /// For example, for a line count metric:
    /// - `delta_metric == 0` returns the offset of the line start of the current line
    /// - `delta_metric == 1` returns the offset to the next line start
    /// - `delta_metric == -1` returns the offset to the line start of the previous line
    ///
    /// The default implementation approximates
    /// `to_base_units(from_base_units(base_offset) + delta_metric)`.
    /// But the implementer should override this method if they can optimize it further.
    fn navigate(piece: &T, base_offset: usize, delta_metric: isize) -> Option<usize> {
        let aligned = Self::from_base_units(piece, base_offset);
        if let Some(next) = aligned.checked_add_signed(delta_metric)
            && next <= Self::measure(&piece.summarize()) {
            Some(next)
        } else {
            None
        }
    }
}

/// The base metric measured by [Sum::len]
pub struct BaseMetric();

impl<T: Summable> Metric<T> for BaseMetric {
    fn measure(sum: &T::S) -> usize {
        sum.len()
    }
    fn from_base_units(_piece: &T, base_units: usize) -> usize {
        base_units
    }
    fn to_base_units(_piece: &T, measurement: usize) -> usize {
        measurement
    }
}

/// A more verbose API for cursor navigation and modification
///
/// The user is responsible for ensuring to pass the correct
/// [crate::roperig::Rope] reference to the member functions.
pub struct CursorPos<T: Summable> {
    /// Reference to an actual node
    pub(crate) node: SafeRef,
    /// Offset in node *in base unit*
    pub(crate) offset_in_node: usize,
    tree: PhantomData<T>,
}
impl<T: Summable> CursorPos<T> {
    pub(crate) fn new(node: SafeRef, offset_in_node: usize) -> Self {
        Self { node, offset_in_node, tree: PhantomData }
    }
}

impl<T: Summable> Clone for CursorPos<T> {
    fn clone(&self) -> Self {
        Self {
            offset_in_node: self.offset_in_node,
            node: self.node,
            tree: PhantomData,
        }
    }
}

// See [Cursor] for documentation.
impl<T: Summable> CursorPos<T> {
    /// Converts to an immutable cursor
    pub(crate) fn cursor(self, tree: &RbSlab<T>) -> Cursor<'_, T> {
        Cursor::new(tree, self)
    }

    pub(crate) fn get_base_units<'a>(&self, tree: &'a RbSlab<T>) -> (&'a T, usize) {
        let piece = &tree[self.node].piece;
        (piece, self.offset_in_node)
    }

    fn offset<M: Metric<T>>(&self, tree: &RbSlab<T>) -> usize {
        let mut node = self.node;
        let (mut offset, mut parent) = {
            let n = &tree[node];
            (self.offset_in_node, n.rb.parent)
        };
        while let Some(p) = parent {
            let pn = &tree[p];
            if pn.rb.children[1] == Some(node) {
                offset += M::measure(&pn.left_sum);
                offset += M::measure(&pn.piece.summarize());
            }
            node = p;
            parent = pn.rb.parent;
        }
        offset
    }

    pub(crate) fn prev<M: Metric<T>>(&mut self, tree: &RbSlab<T>, measurement: usize) -> bool {
        self.move_towards::<M>(tree, measurement, LEFT)
    }
    pub(crate) fn next<M: Metric<T>>(&mut self, tree: &RbSlab<T>, measurement: usize) -> bool {
        self.move_towards::<M>(tree, measurement, RIGHT)
    }
    fn move_towards<M: Metric<T>>(&mut self, tree: &RbSlab<T>, measurement: usize, dir: usize) -> bool {
        let next = rel_node_at::<T, M>(tree, self.clone(), measurement, dir);
        if let Some(next) = next {
            *self = next;
            true
        } else {
            false
        }
    }

    pub(crate) fn tree_next_piece(&mut self, tree: &RbSlab<T>) -> bool {
        if let Some(next) = self.to_next_piece(tree) {
            *self = next;
            true
        } else {
            false
        }
    }
    pub(crate) fn tree_prev_piece(&mut self, tree: &RbSlab<T>) -> bool {
        if let Some(prev) = self.to_prev_piece(tree) {
            *self = prev;
            true
        } else {
            false
        }
    }
    pub(crate) fn to_next_piece(&self, tree: &RbSlab<T>) -> Option<CursorPos<T>> {
        tree.next(self.node, RIGHT).map(|next| CursorPos::new(next, 0))
    }
    pub(crate) fn to_prev_piece(&self, tree: &RbSlab<T>) -> Option<CursorPos<T>> {
        tree.next(self.node, LEFT).map(|next| CursorPos::new(next, 0))
    }
}

/// An iterator-like construct to navigate in a rope with different metrics
///
/// This struct contains a reference to the tree to simplify the API.
/// If detailed lifetime control or mutation is wanted, use [CursorPos]
/// instead.
pub struct Cursor<'a, T: Summable> {
    tree: &'a RbSlab<T>,
    pos: CursorPos<T>,
}

impl<'a, T: Summable> Cursor<'a, T> {
    pub(crate) fn new(tree: &'a RbSlab<T>, pos: CursorPos<T>) -> Self {
        Self { tree, pos }
    }

    /// Returns the inner [CursorPos], used only to convert between
    /// [Cursor] and [MutCursor]
    pub fn inner(&self) -> CursorPos<T> {
        self.pos.clone()
    }

    /// Returns the current piece and relative position of the cursor
    /// to the start of the current node in base units
    pub fn get<M: Metric<T>>(&'a self) -> (&'a T, usize) {
        let (piece, base) = self.pos.get_base_units(self.tree);
        (piece, M::from_base_units(piece, base))
    }
    /// Returns the current piece and relative position of the cursor
    /// to the start of the current node in base units
    pub fn get_base_units(&'a self) -> (&'a T, usize) {
        self.pos.get_base_units(self.tree)
    }

    /// Returns the current absolute offset in specific metric
    pub fn offset<M: Metric<T>>(&self) -> usize {
        self.pos.offset::<M>(self.tree)
    }

    /// Move backwards `measurement` units
    pub fn prev<M: Metric<T>>(&mut self, measurement: usize) -> bool {
        self.pos.prev::<M>(self.tree, measurement)
    }
    /// Move forwards `measurement` units
    pub fn next<M: Metric<T>>(&mut self, measurement: usize) -> bool {
        self.pos.next::<M>(self.tree, measurement)
    }

    /// Move the cursor to the next consecutive piece
    ///
    /// Returns `false` if there are no more pieces.
    pub fn next_piece(&mut self) -> bool {
        self.pos.tree_next_piece(self.tree)
    }
    /// Move the cursor to the previous consecutive piece
    ///
    /// Returns `false` if there are no more pieces.
    pub fn prev_piece(&mut self) -> bool {
        self.pos.tree_prev_piece(self.tree)
    }
}

pub(crate) fn rel_node_at_metric<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, mut x: Ref, mut offset: usize,
) -> Option<CursorPos<T>> {
    while let Some(xi) = x {
        let n = &tree[xi];
        let left_len = M::measure(&n.left_sum);
        if left_len >= offset {
            x = n.rb.children[0];
        } else {
            let n_len = M::measure(&n.piece.summarize());
            let pre_len = left_len + n_len;
            if pre_len >= offset {
                offset -= left_len;
                debug_assert!((offset == 0) == (n_len == 0));
                let base_offset = M::to_base_units(&n.piece, offset);
                return Some(CursorPos::new(xi, M::to_base_units(&n.piece, base_offset)));
            } else {
                offset -= pre_len;
                x = n.rb.children[1];
            }
        }
    }
    None
}

pub(crate) fn rel_node_at<T: Summable, M: Metric<T>>(
    tree: &RbSlab<T>, from: CursorPos<T>, metric_offset: usize, dir: usize,
) -> Option<CursorPos<T>> {
    let x = from.node;
    let n = &tree[x];
    if dir == LEFT && let Some(next) = M::navigate(
        &n.piece, from.offset_in_node, (metric_offset as isize).neg(),
    ) {
        return Some(CursorPos::new(x, next));
    }
    if dir == RIGHT && let Some(next) = M::navigate(
        &n.piece, from.offset_in_node, metric_offset as isize,
    ) {
        return Some(CursorPos::new(x, next));
    }

    let mut offset_i = M::measure(&n.left_sum) as isize
        + from.offset::<M>(tree) as isize
        + if dir == LEFT { -(metric_offset as isize) } else { metric_offset as isize };
    let mut x = Some(x);
    while let Some(xi) = x {
        let n = &tree[xi];
        if 0 <= offset_i {
            let subtree = rel_node_at_metric::<T, M>(tree, x, offset_i as usize);
            if subtree.is_some() {
                return subtree;
            }
        }

        let p = n.rb.parent;
        let pn = &tree[p];
        if pn.children[1] == x {
            let Some(p) = p else { unreachable!() };
            offset_i += M::measure(&tree[p].left_sum) as isize;
            offset_i += M::measure(&tree[p].piece.summarize()) as isize;
        }
        x = p;
    }
    None
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::roperig::Rope;
    use crate::roperig_test::Alphabet;
    use super::*;

    pub fn validate_with_cursor(rb: &Rope<Alphabet>) {
        let Some(mut cursor) = rb.cursor::<BaseMetric>(0) else {
            assert_eq!(rb.len(), 0);
            assert!(rb.is_empty());
            return;
        };
        let mut c: Option<char> = None;
        loop {
            let (s, offset) = cursor.get_base_units();
            assert_eq!(offset, 0);
            assert!(!s.0.is_empty());
            let next = s.0.chars().next().unwrap();
            if let Some(c) = c {
                assert_ne!(next, c);
            }
            c = s.0.chars().next();
            if !cursor.next_piece() {
                break;
            }
        }
    }

    #[test]
    fn test_cursor_creation() {
        let mut rb = Rope::<Alphabet>::default();
        rb.insert(0, "111".into());
        rb.insert(3, "222".into());

        let start = rb.cursor::<BaseMetric>(0);
        assert!(start.is_some());
        let start = start.unwrap();
        assert_eq!(start.get_base_units().0.0, "111");
        assert_eq!(start.get_base_units().1, 0);

        let mid = rb.cursor::<BaseMetric>(3);
        assert!(mid.is_some());
        let mid = mid.unwrap();
        assert_eq!(mid.get_base_units().0.0, "111");
        assert_eq!(mid.get_base_units().1, 3);

        let end = rb.cursor::<BaseMetric>(6);
        assert!(end.is_some());
        let end = end.unwrap();
        assert_eq!(end.get_base_units().0.0, "222");
        assert_eq!(end.get_base_units().1, 3);

        assert!(rb.cursor::<BaseMetric>(7).is_none());
    }
}
