use std::cmp::Ordering;
use std::ops::Range;
use crate::metrics::WithCharMetric;
use crate::piece::{DeleteResult, Insertion, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use crate::string::RopeContainer;

#[derive(Default, Copy, Clone, Eq, PartialEq)]
struct Segment {
    length: usize,
    chars: usize,
}

impl Segment {
    pub fn from_str(s: &str) -> Self {
        Self { length: s.len(), chars: s.chars().count() }
    }
}

#[derive(Default)]
struct Buffer {
    s: Box<[u8]>,
    gap_start: usize,
    gap_end: usize,
}

impl Sum for Segment {
    fn len(&self) -> usize {
        self.length
    }

    fn add_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_add(other.chars);
        self.length = self.length.wrapping_add(other.length);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.chars = self.chars.wrapping_sub(other.chars);
        self.length = self.length.wrapping_sub(other.length);
    }

    fn identity() -> Self {
        Self::default()
    }
}

impl Summable for Segment {
    type S = Self;

    fn summarize(&self) -> Self::S {
        *self
    }
}

const MAX_PIECE_LEN: usize = 256;

impl RopePiece for Segment {
    type Context = Buffer;
    const ABS: bool = true;

    fn insert_or_split(
        &mut self, _context: &mut Self::Context,
        other: Insertion<Self>, offset: &Self::S,
    ) -> SplitResult<Self> {
        if self.length < MAX_PIECE_LEN {
            self.add_assign(&other.1);
            SplitResult::Merged
        } else if offset.length == 0 {
            SplitResult::HeadSplit(other.0)
        } else if offset.length == self.length {
            SplitResult::TailSplit(other.0)
        } else {
            let mut tail = *self;
            tail.sub_assign(offset);
            *self = *offset;
            SplitResult::MiddleSplit(other.0, tail)
        }
    }

    fn delete_range(
        &mut self, _context: &mut Self::Context,
        from: &Self::S, to: &Self::S,
    ) -> DeleteResult<Self> {
        let mut del = *to;
        del.sub_assign(from);
        self.sub_assign(&del);
        DeleteResult::Updated(del)
    }

    fn delete(&mut self, _context: &mut Self::Context) {
        // (de)allocation is handled by GapBuffer
    }

    fn measure_offset(
        &self, context: &Self::Context,
        base_offset: usize, abs_base_offset: usize,
    ) -> Self::S {
        Segment {
            length: base_offset,
            chars: context.count_chars(abs_base_offset, abs_base_offset + base_offset),
        }
    }
}
impl WithCharMetric for Segment {
    fn substring<F, R: Default>(
        &self, context: &Self::Context,
        range: Range<usize>, abs_base: usize,
        mut f: F,
    ) -> R where F: FnMut(&str, R) -> R {
        let mut r = R::default();
        let start = range.start + abs_base;
        let end = range.end + abs_base;
        if start < context.gap_start {
            r = f(unsafe {
                str::from_utf8_unchecked(&context.s[start..context.gap_start.min(end)])
            }, r);
        }
        if end > context.gap_start {
            r = f(unsafe {
                str::from_utf8_unchecked(&context.s[context.gap_end..end+context.gap_size()])
            }, r);
        }
        r
    }

    fn chars(sum: &Self::S) -> usize {
        sum.chars
    }
}

impl Buffer {
    fn gap_size(&self) -> usize {
        self.gap_end - self.gap_start
    }

    fn count_chars(&self, from: usize, to: usize) -> usize {
        if from >= to {
            return 0;
        }
        if to <= self.gap_start {
            unsafe {
                str::from_utf8_unchecked(&self.s[from..to])
            }.chars().count()
        } else if from >= self.gap_start {
            let gap = self.gap_size();
            unsafe {
                str::from_utf8_unchecked(&self.s[from+gap..to+gap])
            }.chars().count()
        } else {
            let gap = self.gap_size();
            let left = unsafe {
                str::from_utf8_unchecked(&self.s[from..self.gap_start])
            }.chars().count();
            let right = unsafe {
                str::from_utf8_unchecked(&self.s[self.gap_end..to+gap])
            }.chars().count();
            left + right
        }
    }

    fn move_gap(&mut self, to: usize) {
        if self.gap_size() == 0 {
            self.gap_start = to;
            self.gap_end = to;
            return;
        }
        match to.cmp(&self.gap_start) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.gap_end = to + self.gap_size();
                self.s.copy_within(to..self.gap_start, self.gap_end);
                self.gap_start = to;
            }
            Ordering::Greater => {
                let gap = self.gap_size();
                self.s.copy_within(self.gap_end..to+gap, self.gap_start);
                self.gap_end = to + gap;
                self.gap_start = to;
            }
        }
    }

    fn ensure_gap(&mut self, some: usize, at: usize) {
        let gap = self.gap_size();
        if some <= gap {
            self.move_gap(at);
            return;
        }
        let required = self.s.len() + some - gap;
        let mut new_size = self.s.len().max(MAX_PIECE_LEN);
        while new_size < required {
            new_size = new_size + (new_size>> 1);
        }
        let delta = new_size - self.s.len();
        let mut extended = Vec::with_capacity(new_size);
        match at.cmp(&self.gap_start) {
            Ordering::Less | Ordering::Equal => {
                extended.extend_from_slice(&self.s[..at]);
                extended.resize(at + self.gap_size() + delta, 0);
                extended.extend_from_slice(&self.s[at..self.gap_start]);
                extended.extend_from_slice(&self.s[self.gap_end..]);
            }
            Ordering::Greater => {
                extended.extend_from_slice(&self.s[..self.gap_start]);
                extended.extend_from_slice(&self.s[self.gap_end..at+self.gap_size()]);
                extended.resize(at + self.gap_size() + delta, 0);
                extended.extend_from_slice(&self.s[at+self.gap_size()..]);
            }
        }
        self.s = extended.into_boxed_slice();
        self.gap_end = at + self.gap_size() + delta;
        self.gap_start = at;
    }

    pub fn insert(&mut self, at: usize, s: &str) {
        self.ensure_gap(s.len(), at);
        self.s[self.gap_start..self.gap_start+s.len()].copy_from_slice(s.as_bytes());
        self.gap_start += s.len();
    }

    pub fn delete(&mut self, range: Range<usize>) {
        if range.end <= self.gap_start {
            self.move_gap(range.end);
            self.gap_start = range.start;
        } else if range.start >= self.gap_start {
            self.move_gap(range.start);
            self.gap_end += range.len();
        } else {
            let prior = self.gap_start - range.start;
            self.gap_start = range.start;
            self.gap_end += range.len() - prior;
        }
    }
}

/// A gap buffer implementation
///
/// Note that the implementation is not well-optimized
/// since currently [Rope] recalculate the absolute offsets
/// quite frequently.
#[derive(Default)]
pub struct GapBuffer {
    buffer: Rope<Segment>,
}

impl RopeContainer<Segment> for GapBuffer {
    fn rope(&self) -> &Rope<Segment> {
        &self.buffer
    }
    fn rope_mut(&mut self) -> &mut Rope<Segment> {
        &mut self.buffer
    }
    fn substring_store(&self, range: Range<usize>, buffer: &mut String) {
        let context = self.buffer.context();
        if range.start < context.gap_start {
            buffer.push_str(unsafe {
                str::from_utf8_unchecked(&context.s[range.start..context.gap_start.min(range.end)])
            });
        }
        if range.end > context.gap_start {
            buffer.push_str(unsafe {
                str::from_utf8_unchecked(&context.s[context.gap_end..range.end+context.gap_size()])
            });
        }
    }
    /// Deletes a range of bytes
    fn delete_bytes(&mut self, range: Range<usize>) {
        self.rope_mut().delete(range.clone());
        self.buffer.context_mut().delete(range);
    }
}

impl GapBuffer {
    /// Inserts strings
    pub fn insert(&mut self, at: usize, s: &str) {
        self.buffer.context_mut().insert(at, s);
        self.buffer.insert(at, Segment::from_str(s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_chars() {
        let mut buf = GapBuffer::default();
        buf.insert(0, "helloworld");
        buf.insert(5, " ");
        assert_eq!("hello world", buf.substring(0..buf.len()));
        buf.buffer.is_valid();
        buf.delete_bytes(1..buf.len() - 1);
        assert_eq!("hd", buf.substring(0..buf.len()));
        buf.buffer.is_valid();
    }
}
