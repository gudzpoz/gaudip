use crate::piece::{Sum, Summable};
use crate::rb_base::{RbSlab, Ref, LEFT, RIGHT};
use crate::roperig::{rel_node_at, NodePosition};
use std::marker::PhantomData;

/// A metric system for rope pieces
///
/// For the conceptual background, see the [rope science series].
///
/// [rope science series]: https://xi-editor.io/docs/rope_science_02.html
pub trait Metric<T: Summable> {
    /// Returns the measurement of a [Sum]
    fn measure(sum: &T::S) -> usize;
    /// Converts from base units (as is returned by [Sum::len]) to this measurement
    fn from_base_units(piece: &T, base_units: usize) -> usize;
    /// Converts from this measurement to base units (as is returned by [Sum::len])
    #[allow(clippy::wrong_self_convention)]
    fn to_base_units(piece: &T, measurement: usize) -> usize;
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

/// An iterator-like construct to navigate in a rope with different metrics
pub struct Cursor<'a, T: Summable, M: Metric<T>> {
    tree: &'a RbSlab<T>,
    offset_in_node: usize,
    node: Ref,
    metric: PhantomData<M>,
}

impl<'a, T: Summable, M: Metric<T>> Cursor<'a, T, M> {
    pub(crate) fn new(tree: &'a RbSlab<T>, pos: NodePosition) -> Self {
        Self {
            tree,
            offset_in_node: M::from_base_units(&tree[pos.node].piece, pos.in_node_offset),
            node: pos.node,
            metric: PhantomData,
        }
    }

    /// Returns the current piece and offset in piece of this cursor
    pub fn get(&self) -> (&'a T, usize) {
        let piece = &self.tree[self.node].piece;
        (piece, self.offset_in_node)
    }
    /// Returns the current piece and base unit offset
    pub fn get_base_units(&self) -> (&'a T, usize) {
        let piece = &self.tree[self.node].piece;
        (piece, M::to_base_units(piece, self.offset_in_node))
    }

    /// Move backwards `measurement` units
    pub fn prev(&mut self, measurement: usize) -> bool {
        self.move_towards(measurement, LEFT)
    }
    /// Move forwards `measurement` units
    pub fn next(&mut self, measurement: usize) -> bool {
        self.move_towards(measurement, RIGHT)
    }
    fn move_towards(&mut self, measurement: usize, dir: usize) -> bool {
        if let Some(next) = rel_node_at::<T, M>(self.tree, NodePosition {
            node: self.node,
            in_node_offset: self.offset_in_node,
        }, measurement, dir) {
            self.node = next.node;
            self.offset_in_node = next.in_node_offset;
            true
        } else {
            false
        }
    }
}
