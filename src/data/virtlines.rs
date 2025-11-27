use roperig::metrics::Metric;
use roperig::piece::{Sum, Summable};
use roperig::ropebase::{PartialCursorPos, RopeBase};
use std::num::NonZero;

/// Info about a virtual line
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineInfo {
    /// Number of chars in this line, line feed char(s) included
    pub chars: usize,
    /// Number of lines in this virtual line, maybe zero
    pub lines: usize,
    /// Height of this virtual line in pango pixels
    ///
    /// This is a cached/estimated value. The user is responsible for updating it
    /// on resize.
    pub height: usize,
}
impl Sum for LineInfo {
    fn len(&self) -> usize {
        self.chars
    }

    fn add_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_add(other.chars);
        self.lines = self.lines.wrapping_add(other.lines);
        self.height = self.height.wrapping_add(other.height);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_sub(other.chars);
        self.lines = self.lines.wrapping_sub(other.lines);
        self.height = self.height.wrapping_sub(other.height);
    }

    fn identity() -> Self {
        Default::default()
    }
}

pub struct OpaqueLine<T> {
    line: Option<T>,
    info: LineInfo,
}
impl<T> Summable for OpaqueLine<T> {
    type S = LineInfo;
    fn summarize(&self) -> Self::S {
        self.info
    }
}

struct LineMetric();
impl<T> Metric<OpaqueLine<T>> for LineMetric {
    fn measure(sum: &LineInfo) -> usize {
        sum.lines
    }
}
struct HeightMetric();
impl<T> Metric<OpaqueLine<T>> for HeightMetric {
    fn measure(sum: &LineInfo) -> usize {
        sum.height
    }
}

pub type LineNumber = NonZero<usize>;
/// Representation of a collection of lines and a viewport into them
pub struct VirtualLines<T> {
    rope: RopeBase<OpaqueLine<T>>,
    current: Option<PartialCursorPos<OpaqueLine<T>, HeightMetric>>,
}
impl<T> Default for VirtualLines<T> {
    fn default() -> Self {
        Self { rope: Default::default(), current: None }
    }
}
pub struct LineIter<T> {
    current: Option<PartialCursorPos<OpaqueLine<T>, HeightMetric>>,
}
impl<T> LineIter<T> {
    /// Returns true if there is a current line
    pub fn has_current(&self) -> bool {
        self.current.is_some()
    }
    /// Returns the current line `(rel_height, line, stats)`
    ///
    /// The `rel_height` should only be non-zero for the first visible line,
    /// meaning how much the current line is above the top of the viewport.
    /// It should always be negative.
    pub fn current<'a>(
        &self, lines: &'a VirtualLines<T>,
    ) -> Option<(isize, Option<&'a T>, &'a LineInfo)> {
        let current = self.current.as_ref()?;
        let start = 0isize.saturating_sub_unsigned(current.offset().value);
        let line = current.get(&lines.rope);
        Some((start, line.line.as_ref(), &line.info))
    }
    /// Returns the current line number, 1-based
    pub fn current_line_num(&self, lines: &VirtualLines<T>) -> Option<LineNumber> {
        let Some(current) = self.current.as_ref() else {
            return LineNumber::new(lines.rope.len::<LineMetric>() + 1);
        };
        LineNumber::new(current.with_offset::<LineMetric>(0).position(&lines.rope) + 1)
    }
    /// Returns a mutable reference to the current line
    pub fn current_mut<'a>(&self, lines: &'a mut VirtualLines<T>) -> Option<&'a mut Option<T>> {
        let current = self.current.as_ref()?;
        let line = current.get_mut(&mut lines.rope);
        Some(&mut line.line)
    }
    /// Updates the stats of the current line
    pub fn update_line(&self, lines: &mut VirtualLines<T>, delta: &LineInfo) {
        let Some(current) = self.current.as_ref() else { return };
        current.get_mut(&mut lines.rope).info.add_assign(delta);
        current.update(&mut lines.rope, delta);
    }
    /// Advances to the next line
    pub fn advance(&mut self, lines: &VirtualLines<T>) {
        let Some(current) = self.current.as_ref() else { return };
        self.current = current.next_piece(&lines.rope);
    }
    /// Materializes the current line
    ///
    /// Let's call lines with `None` of its "inner" data `T` "virtual lines"
    /// and with `Some` of its "inner" data `T` "materialized lines".
    /// So this function turns:
    ///
    /// ```text
    /// --- virtual line #1 start --- (stats: chars: N, lines: 5)
    /// 1
    /// 2
    /// 3    <--- materialize(position: { lines: rel 2 }, size: { lines: 1})
    /// 4
    /// 5
    /// --- virtual line #1 end ---
    /// ```
    ///
    /// into:
    ///
    /// ```text
    /// --- virtual line #1 start --- (stats: chars: N', lines: 2)
    /// 1
    /// 2
    /// --- virtual line #1 end ---
    /// 3 (materialized line)
    /// --- virtual line #2 start --- (stats: chars: N'', lines: 2)
    /// 4
    /// 5
    /// --- virtual line #2 end ---
    /// ```
    pub fn materialize(
        &self, lines: &mut VirtualLines<T>,
        inner: T, position: &LineInfo, mut size: LineInfo,
    ) -> Option<()> {
        let current = self.current.as_ref()?;
        let line = current.get(&lines.rope);

        let offset = {
            let mut start = current.node_start(&lines.rope);
            start.height = start.height.min(position.height);
            let mut offset = position.sub(start);
            offset.height = offset.height.min(line.info.height);
            offset
        };
        // ensure heights: line.info = offset + size + tail
        if offset.height + size.height > line.info.height {
            size.height = line.info.height - offset.height;
        }

        if offset.chars == 0 {
            if size.chars == line.info.chars {
                // virtual line totally replaced
                if offset.lines != 0 || size.lines != line.info.lines {
                    // TODO: unreachable / panic?
                    return None;
                }
                current.get_mut(&mut lines.rope).line = Some(inner);
                return Some(());
            }
            // head replace: (virt#1) => (mat#2)(virt#1)
            let new_info = line.info.sub(size);
            current.get_mut(&mut lines.rope).info = new_info;
            current.update(&mut lines.rope, &size.negate());
            current.insert_left(&mut lines.rope, OpaqueLine { line: Some(inner), info: size });
        } else {
            // mid/tail replace: (virt#1) => (virt#1)(mat#2) or (virt#1)(mat#2)(virt#3)
            let tail = line.info.sub(offset).sub(size);
            let delta = offset.sub(line.info);
            current.get_mut(&mut lines.rope).info = offset;
            current.update(&mut lines.rope, &delta);
            let next = current.insert_right(&mut lines.rope, OpaqueLine { line: Some(inner), info: size });
            if tail.chars != 0 {
                next.insert_right(&mut lines.rope, OpaqueLine { line: None, info: tail });
            }
        }
        Some(())
    }
}
impl<T> VirtualLines<T> {
    /// Returns the total height
    pub fn height(&self) -> usize {
        self.rope.len::<HeightMetric>()
    }
    /// Returns the total line count
    pub fn line_count(&self) -> usize {
        self.rope.len::<LineMetric>()
    }
    /// Returns the current height
    pub fn current_height(&self) -> usize {
        self.current.as_ref().map_or(0, |c| c.position(&self.rope))
    }
    /// Get the current line number, if any
    pub fn current_line_num(&self) -> Option<LineNumber> {
        let current = self.current.as_ref()?;
        LineNumber::new(current.with_offset::<LineMetric>(1).position(&self.rope))
    }
    /// Returns an iterator starting from the current line
    pub fn iter_from_current(&self) -> LineIter<T> {
        LineIter { current: self.current.clone() }
    }
    /// Returns an iterator starting from the specified line
    pub fn iter_from_line(&self, line: LineNumber) -> LineIter<T> {
        let cursor = self.rope.cursor_at::<LineMetric>(line.get());
        LineIter { current: cursor.map(|c| c.with_offset(0)) }
    }
    /// Sets the current line by scrolling a percentage
    pub fn scroll(&mut self, percent: f64) {
        assert_eq!(self.current.is_none(), self.rope.is_empty());
        if !percent.is_finite() {
            return;
        }
        fn ensure_ratio(f: f64) -> f64 {
            f.clamp(-1.0, 1.0)
        }

        let Some(current) = self.current.as_ref() else { return };
        let at = current.position(&self.rope);
        let delta = match percent.partial_cmp(&0.0).expect("finite") {
            std::cmp::Ordering::Equal => 0isize,
            std::cmp::Ordering::Less => (ensure_ratio(percent) * at as f64).ceil() as isize,
            std::cmp::Ordering::Greater => (
                ensure_ratio(percent) * (self.height() - at) as f64
            ).ceil() as isize,
        };
        self.scroll_by(delta);
    }
    /// Sets the current line by scrolling a certain height units
    pub fn scroll_by(&mut self, delta: isize) {
        let Some(current) = self.current.as_ref() else { return };
        let Some(next) = current.navigate(&self.rope, delta) else { return };
        self.current = Some(next);
    }
    /// Sets the current line by absolute y coordinate
    pub fn scroll_to(&mut self, height: usize) {
        let Some(cursor) = self.rope.cursor_at::<HeightMetric>(height) else { return };
        self.current = Some(cursor);
    }
    /// Sets the current line by line number
    pub fn scroll_to_line(&mut self, line: LineNumber) {
        let Some(cursor) = self.rope.cursor_at::<LineMetric>(line.get()) else { return };
        self.current = Some(cursor.with_offset(0));
    }
    /// Gets a line by on-screen y coordinate
    pub fn line_at_rel_height(&self, height: usize) -> Option<(usize, LineIter<T>)> {
        let current = self.current.as_ref()?;
        let cursor = current.navigate(&self.rope, height as isize)?;
        Some((cursor.offset().value, LineIter { current: Some(cursor) }))
    }
    /// Batch initializes the lines
    pub fn init_lines<I>(&mut self, lines: &mut I)
    where I: ExactSizeIterator<Item = (LineInfo, Option<T>)> {
        self.rope.init(lines.map(|(info, line)| OpaqueLine { line, info }));
        self.current = self.rope.cursor_at::<HeightMetric>(0);
    }
    /// Inserts a new line
    pub fn insert_line(&mut self, at: LineNumber, info: LineInfo, extra: Option<T>) {
        let line = OpaqueLine { line: extra, info };
        let Some(cursor) = self.rope.cursor_at::<LineMetric>(at.get()) else {
            self.rope.init(Some(line).into_iter());
            self.current = self.rope.cursor_at::<HeightMetric>(0);
            return;
        };
        if cursor.offset().value == 0 {
            cursor.insert_left(&mut self.rope, line);
        } else {
            cursor.insert_right(&mut self.rope, line);
        }
    }
}
