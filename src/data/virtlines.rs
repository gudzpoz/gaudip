use std::num::NonZero;
use roperig::ropebase::{PartialCursorPos, RopeBase};
use roperig::metrics::Metric;
use roperig::piece::{Sum, Summable};

/// Info about a virtual line
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        Self { chars: 0, lines: 0, height: 0 }
    }
}

pub type Id = Option<NonZero<usize>>;
struct OpaqueLine {
    id: Id,
    info: LineInfo,
}
impl Summable for OpaqueLine {
    type S = LineInfo;
    fn summarize(&self) -> Self::S {
        self.info
    }
}

struct LineMetric();
impl Metric<OpaqueLine> for LineMetric {
    fn measure(sum: &LineInfo) -> usize {
        sum.lines
    }
}
struct HeightMetric();
impl Metric<OpaqueLine> for HeightMetric {
    fn measure(sum: &LineInfo) -> usize {
        sum.height
    }
}

pub type LineNumber = NonZero<usize>;
/// Representation of a collection of lines and a viewport into them
#[derive(Default)]
pub struct VirtualLines {
    rope: RopeBase<OpaqueLine>,
    current: Option<PartialCursorPos<OpaqueLine, HeightMetric>>,
}
pub struct LineIter<'a> {
    rope: &'a RopeBase<OpaqueLine>,
    current: Option<PartialCursorPos<OpaqueLine, HeightMetric>>,
    offset: usize,
}
impl<'a> Iterator for LineIter<'a> {
    type Item = (isize, Id, &'a LineInfo);
    fn next(&mut self) -> Option<Self::Item> {
        let mut offset: isize = self.offset.try_into().unwrap_or(isize::MAX);
        loop {
            let current = self.current.as_ref()?;
            let start = offset.saturating_sub_unsigned(current.offset().value);
            let line = current.get(self.rope);
            offset = start.saturating_add_unsigned(line.info.height);
            self.current = current.next_piece(self.rope);
            if offset >= 0 {
                self.offset = offset as usize;
                return Some((start, line.id, &line.info));
            }
        }
    }
}
impl VirtualLines {
    /// Returns the total height
    pub fn height(&self) -> usize {
        self.rope.len::<HeightMetric>()
    }
    /// Get the current line number, if any
    pub fn current_line_num(&self) -> Option<LineNumber> {
        let current = self.current.as_ref()?;
        LineNumber::new(current.position(&self.rope) + 1)
    }
    /// Returns an iterator starting from the current line
    pub fn iter_from_current(&'_ self) -> LineIter<'_> {
        LineIter {
            rope: &self.rope,
            current: self.current.clone(),
            offset: 0,
        }
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
                ensure_ratio(percent) * (self.rope.len::<HeightMetric>() - at) as f64
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
    /// Inserts a new line
    pub fn insert_line(&mut self, at: LineNumber, line: LineInfo, link: Id) {
        let line = OpaqueLine { id: link, info: line };
        let Some(cursor) = self.rope.cursor_at::<LineMetric>(at.get() - 1) else {
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
