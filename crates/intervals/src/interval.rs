//! Contains the interval node implementation

use std::{cmp::Ordering, ops::Range};

use bitflags::bitflags;

bitflags! {
    /// Metadata flags
    #[derive(Default, Clone)]
    pub struct Metadata: u32 {
        /// Color info (red/black)
        const COLOR   = 0b0000_0001;
        /// Used when iterating
        const VISITED = 0b0000_0010;

        /// Should the start of the interval move when insertion happens
        /// exactly at the start of the interval?
        const FRONT_ADVANCE = 0b0000_0100;
        /// Should the end of the interval move when insertion happens
        /// exactly at the end of the interval?
        const REAR_ADVANCE  = 0b0000_1000;

        /// Several reserved bits
        const RESERVED = 0b1111_0000;

        /// The user may use the rest bits for extra metadata
        const _ = !0;
    }
}

/// A reference to an interval
pub type Ref = usize;
pub(crate) const SENTINEL: Ref = 0;
pub(crate) const LEFT: usize = 0;
pub(crate) const RIGHT: usize = 1;

pub(crate) const MIN_SAFE_DELTA: isize = -(1 << 30);
pub(crate) const MAX_SAFE_DELTA: isize = 1 << 30;

/// A node in the interval tree
///
/// ## API
///
/// The struct is directly used as a node in the interval tree, which internally
/// will transform the offsets to relative offsets. So when querying nodes, the
/// user needs some extra care to obtain absolute offsets (like calling
/// [crate::tree::IntervalTree::resolve_node])
///
/// ## Implementation
///
/// The API mostly uses `isize` instead of `usize`. It's because internally
/// it uses relative offsets to support dynamic updates, and relative offsets
/// can be negative.
#[derive(Default, Clone)]
pub struct Interval {
    /// Metadata
    pub metadata: Metadata,
    pub(crate) parent: Ref,
    pub(crate) children: [Ref; 2],

    pub(crate) start: isize,
    pub(crate) end: isize,
    pub(crate) delta: isize,
    pub(crate) max_end: isize,

    /// Cached version
    pub cached_version_id: u32,
    /// Cached bsolute interval range computed at [Self::cached_version_id]
    pub cached_range: Range<isize>,
}

pub(crate) type Color = bool;
pub(crate) const RED: Color = true;
pub(crate) const BLACK: Color = false;

impl From<Range<isize>> for Interval {
    fn from(range: Range<isize>) -> Self {
        Self::new(range, false, false)
    }
}

impl Interval {
    /// Creates a new interval with edge advance flags
    pub fn new(range: Range<isize>, front_advance: bool, rear_advance: bool) -> Interval {
        let mut interval = Interval {
            start: range.start,
            end: range.end,
            max_end: range.end,
            cached_range: range,
            ..Default::default()
        };
        interval.set_front_advance(front_advance);
        interval.set_rear_advance(rear_advance);
        interval
    }

    /// The color of the node ([RED] (`true`) or [BLACK] (`false`))
    pub(crate) fn color(&self) -> Color {
        self.metadata.contains(Metadata::COLOR)
    }
    /// Sets the color of the node
    pub(crate) fn set_color(&mut self, color: bool) {
        self.metadata.set(Metadata::COLOR, color);
    }

    /// Returns `true` if the node has been visited
    ///
    /// Used for ordered, non-recursive traversal.
    pub(crate) fn is_visited(&self) -> bool {
        self.metadata.contains(Metadata::VISITED)
    }
    /// Sets the visited flag
    pub(crate) fn set_visited(&mut self, visited: bool) {
        self.metadata.set(Metadata::VISITED, visited);
    }

    /// Sets the cached absolute offsets
    pub(crate) fn set_cached_offsets(&mut self, start: isize, end: isize, version_id: u32) {
        self.cached_version_id = version_id;
        self.cached_range = start..end;
    }

    /// Should the start of the interval be advanced when inserting at the start?
    pub fn is_front_advance(&self) -> bool {
        self.metadata.contains(Metadata::FRONT_ADVANCE)
    }
    /// Sets [Self::is_front_advance]
    pub fn set_front_advance(&mut self, front_advance: bool) {
        self.metadata.set(Metadata::FRONT_ADVANCE, front_advance);
    }
    /// Should the end of the interval be advanced when inserting at the end?
    pub fn is_rear_advance(&self) -> bool {
        self.metadata.contains(Metadata::REAR_ADVANCE)
    }
    /// Sets [Self::is_rear_advance]
    pub fn set_rear_advance(&mut self, rear_advance: bool) {
        self.metadata.set(Metadata::REAR_ADVANCE, rear_advance);
    }

    /// Applies an edit to the interval according to [Self::is_front_advance] and [Self::is_rear_advance]
    pub(crate) fn accept_edit(&mut self, del_range: Range<isize>, insert_length: usize) {
        fn adjust_marker_before_column(
            marker_offset: isize,
            should_advance: bool,
            check_offset: isize,
            force_stay: bool,
        ) -> bool {
            match marker_offset.cmp(&check_offset) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => force_stay || !should_advance,
            }
        }

        let front_advance = self.is_front_advance();
        let rear_advance = self.is_rear_advance();
        let deleting_cnt = del_range.len();
        let insert_cnt = insert_length;
        let common_length = insert_cnt.min(deleting_cnt);

        let Interval { start, end, .. } = *self;
        let mut start_done = false;
        let mut end_done = false;

        {
            let move_semantics = deleting_cnt > 0;
            if !start_done && adjust_marker_before_column(start, front_advance, del_range.start, move_semantics) {
                start_done = true;
            }
            if !end_done && adjust_marker_before_column(end, rear_advance, del_range.start, move_semantics) {
                end_done = true;
            }
        }

        if common_length > 0 {
            let move_semantics = deleting_cnt > insert_cnt;
            let del_min = del_range.start.strict_add_unsigned(common_length);
            if !start_done && adjust_marker_before_column(start, front_advance, del_min, move_semantics) {
                start_done = true;
            }
            if !end_done && adjust_marker_before_column(end, rear_advance, del_min, move_semantics) {
                end_done = true;
            }
        }

        {
            let move_semantics = false;
            if !start_done && adjust_marker_before_column(start, front_advance, del_range.end, move_semantics) {
                self.start = del_range.start.strict_add_unsigned(insert_cnt);
                start_done = true;
            }
            if !end_done && adjust_marker_before_column(end, rear_advance, del_range.end, move_semantics) {
                self.end = del_range.start.strict_add_unsigned(insert_cnt);
                end_done = true;
            }
        }

        let delta_column = insert_cnt.checked_signed_diff(deleting_cnt).unwrap();
        if !start_done {
            self.start = (self.start + delta_column).max(0);
        }
        if !end_done {
            self.end = (self.end + delta_column).max(0);
        }

        self.start = self.start.min(self.end);
    }
}

pub fn interval_compare(a_start: isize, a_end: isize, b_start: isize, b_end: isize) -> Ordering {
    if a_start == b_start {
        a_end.cmp(&b_end)
    } else {
        a_start.cmp(&b_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_edit() {
        // input data: (front_advance, rear_advance, expected_start, expected_end)
        // zero width inserts: initial: (0, 0)
        for (front_advance, rear_advance, expected_start, expected_end) in [
            (false, false, 0, 0),
            (true, false, 0, 0),
            (false, true, 0, 10),
            (true, true, 10, 10),
        ] {
            let mut interval = Interval::new(0..0, front_advance, rear_advance);
            interval.accept_edit(0..0, 10);
            assert_eq!(interval.start, expected_start);
            assert_eq!(interval.end, expected_end);
        }
        // front inserts: intial: (10, 20)
        for (front_advance, rear_advance, expected_start, expected_end) in [
            (false, false, 10, 30),
            (true, false, 20, 30),
            (false, true, 10, 30),
            (true, true, 20, 30),
        ] {
            let mut interval = Interval::new(10..20, front_advance, rear_advance);
            interval.accept_edit(10..10, 10);
            assert_eq!(interval.start, expected_start);
            assert_eq!(interval.end, expected_end);
        }
        // rear inserts: intial: (10, 20)
        for (front_advance, rear_advance, expected_start, expected_end) in [
            (false, false, 10, 20),
            (true, false, 10, 20),
            (false, true, 10, 30),
            (true, true, 10, 30),
        ] {
            let mut interval = Interval::new(10..20, front_advance, rear_advance);
            interval.accept_edit(20..20, 10);
            assert_eq!(interval.start, expected_start);
            assert_eq!(interval.end, expected_end);
        }
    }
}
