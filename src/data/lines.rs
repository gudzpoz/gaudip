use crate::data::buffer::LineBuffer;
use crate::data::segment::EStrSegment;
use roperig::metrics::{BaseMetric, Cursor, Metric};
use roperig::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use roperig::roperig::Rope;
use roperig::string::RopeContainer;

#[derive(Default)]
pub struct BufferLines {
    lines: Rope<VirtualLines>,
}
impl BufferLines {
    pub fn for_each_physical<F>(&self, mut f: F)
    where F: FnMut(&LineBuffer) {
        self.lines.for_range::<BaseMetric>(0..usize::MAX, |_, piece, _, _| {
            let VirtualLines::Physical { line, .. } = piece else { return true };
            f(line);
            true
        });
    }
    pub fn insert(&mut self, at: usize, text: EStrSegment) {
        let Some(cursor) = self.lines.cursor::<BaseMetric>(at) else {
            self.lines.insert(0, VirtualLines::from(text));
            return;
        };
        let Some(physical_c) = Self::adjust_cursor(cursor) else { return };
        let (VirtualLines::Physical {
            line, ..
        }, offset, ..) = physical_c.inner().get_mut(&mut self.lines) else { return };
        line.insert_segment(offset.chars, text);
    }

    pub fn insert_line(&mut self, line: usize, text: EStrSegment) {
        let Some(cursor) = self.lines.cursor::<LineMetric>(line) else {
            return;
        };
        let node = VirtualLines::from(text);
        // TODO: handle virtual lines
        if cursor.abs_offset().lines == line {
            cursor.inner().insert_left(&mut self.lines, node);
        } else {
            cursor.inner().insert_right(&mut self.lines, node);
        }
    }

    pub fn lines(&self) -> usize {
        self.lines.measure::<LineMetric>()
    }

    fn adjust_cursor(mut cursor: Cursor<VirtualLines>) -> Option<Cursor<VirtualLines>> {
        loop {
            let (piece, offset) = cursor.get_offset();
            match piece {
                VirtualLines::Physical { line, eol } => {
                    if !*eol || offset.chars <= line.char_len() {
                        return cursor.into();
                    }
                    if cursor.next_piece() {
                        continue;
                    }
                }
                VirtualLines::Virtual(LineInfo { chars, .. }) => {
                    if offset.chars == *chars && cursor.next_piece() {
                        continue;
                    }
                }
            }
            return None;
        }
    }
}

enum VirtualLines {
    Virtual(LineInfo),
    Physical {
        line: LineBuffer,
        eol: bool
    },
}
impl From<EStrSegment> for VirtualLines {
    fn from(value: EStrSegment) -> Self {
        let mut line = LineBuffer::default();
        line.insert_segment(0, value);
        Self::Physical {
            line,
            eol: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct LineInfo {
    chars: usize,
    lines: usize,
}
impl Sum for LineInfo {
    fn len(&self) -> usize {
        self.chars
    }

    fn add_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_add(other.chars);
        self.lines = self.lines.wrapping_add(other.lines);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_sub(other.chars);
        self.lines = self.lines.wrapping_sub(other.lines);
    }

    fn identity() -> Self {
        Self::default()
    }
}

impl Summable for VirtualLines {
    type S = LineInfo;

    fn summarize(&self) -> Self::S {
        match self {
            VirtualLines::Virtual(info) => *info,
            VirtualLines::Physical { line, eol } => LineInfo {
                chars: line.char_len() + if *eol { 1 } else { 0 },
                lines: if *eol { 1 } else { 0 },
            },
        }
    }
}

impl RopePiece for VirtualLines {
    type Context = ();
    const ABS: bool = false;

    fn insert_or_split(
        &mut self, _: &mut Self::Context,
        other: Self, offset: &Self::S,
    ) -> SplitResult<Self> {
        if offset.chars == 0 {
            SplitResult::HeadSplit(other)
        } else if offset.chars == self.summarize().chars {
            SplitResult::TailSplit(other)
        } else {
            panic!("should manually insert");
        }
    }

    fn delete_range(
        &mut self, _: &mut Self::Context,
        from: &Self::S, to: &Self::S,
    ) -> DeleteResult<Self> {
        let mut delta = *to;
        delta.sub_assign(from);
        if delta.chars != 0 {
            panic!("should manually delete");
        }
        DeleteResult::Updated(delta)
    }

    fn delete(&mut self, _: &mut Self::Context) {
        // no-op (for now)
    }

    fn measure_offset(
        &self, _: &Self::Context,
        base_offset: usize, _abs_base_offset: usize,
    ) -> Self::S {
        LineInfo { chars: base_offset, lines: usize::MAX }
    }
}

struct LineMetric();

impl Metric<VirtualLines> for LineMetric {
    fn measure(sum: &LineInfo) -> usize {
        sum.lines
    }

    fn from_base_units(_context: &(), _piece: &VirtualLines, _base_units: usize, _abs: usize) -> usize {
        0
    }

    fn to_base_units(_context: &(), _piece: &VirtualLines, _measurement: usize, _abs: usize) -> usize {
        0
    }
}
