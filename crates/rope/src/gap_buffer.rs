use crate::metrics::{BaseMetric, Metric};
use crate::piece::{Sum, Summable};
use crate::rb_base::{LEFT, RIGHT};
use crate::ropebase::{PartialCursorPos, RopeBase};
use crate::string::CharMetric;
use std::cmp::Ordering;
use std::ops::Range;

#[derive(Default, Copy, Clone, Eq, PartialEq)]
struct Segment {
    length: usize,
    chars: usize,
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
impl Metric<Segment> for CharMetric {
    fn measure(sum: &Segment) -> usize {
        sum.chars
    }
}

#[derive(Default)]
struct Buffer {
    s: Box<[u8]>,
    gap_start: usize,
    gap_end: usize,
}
impl Buffer {
    fn gap_size(&self) -> usize {
        self.gap_end - self.gap_start
    }

    fn before_gap(&self) -> &str {
        unsafe { str::from_utf8_unchecked(&self.s[..self.gap_start]) }
    }
    fn after_gap(&self) -> &str {
        unsafe { str::from_utf8_unchecked(&self.s[self.gap_end..]) }
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

const MAX_PIECE_LEN: usize = 256;

/// A gap buffer implementation
#[derive(Default)]
pub struct GapBuffer {
    rope: RopeBase<Segment>,

    buffer: Buffer,
    gap_start_chars: usize,
    gap_start_cursor: Option<Cursor>,
}
#[derive(Clone)]
struct Cursor {
    base: PartialCursorPos<Segment, BaseMetric>,
    char_offset: usize,
}
impl Cursor {
    fn next(&self, rope: &RopeBase<Segment>) -> Option<Self> {
        self.base.next_piece(rope).map(|base| Cursor { base, char_offset: 0 })
    }
    fn prev(&self, rope: &RopeBase<Segment>) -> Option<Self> {
        self.base.prev_piece(rope).map(|base| {
            let char_offset = base.get(rope).chars;
            Cursor { base, char_offset }
        })
    }
    fn is_tail(&self, rope: &RopeBase<Segment>) -> bool {
        self.base.get(rope).chars == self.char_offset
    }
    fn head(&self) -> Segment {
        Segment { length: self.base.offset().value, chars: self.char_offset }
    }
}

impl GapBuffer {
    fn char_to_byte(&self, chars: usize) -> (usize, Option<Cursor>) {
        if chars == self.gap_start_chars {
            return (self.buffer.gap_start, self.gap_start_cursor.clone());
        }
        let (bytes, pos) = self.rope.accumulate::<CharMetric, _, _>(
            chars, |acc, seg| acc + seg.length, 0,
        );
        let Some(cursor) = pos else { return (bytes, None) };
        let remaining = cursor.offset().value;
        let piece_start_chars = chars - remaining;
        let byte_offset = if self.gap_start_chars <= piece_start_chars {
            str_indices::chars::to_byte_idx(&self.buffer.after_gap()[bytes-self.buffer.gap_start..], remaining)
        } else if chars < self.gap_start_chars {
            str_indices::chars::to_byte_idx(&self.buffer.before_gap()[bytes..], remaining)
        } else {
            str_indices::chars::to_byte_idx(self.buffer.after_gap(), chars - self.gap_start_chars)
                + self.buffer.gap_start - bytes
        };
        let char_offset = cursor.offset().value;
        (bytes + byte_offset, Some(Cursor {
            base: cursor.with_offset::<BaseMetric>(byte_offset),
            char_offset,
        }))
    }

    fn move_cursor(&mut self, mut chars: usize) {
        chars = chars.min(self.rope.len::<CharMetric>());
        let (bytes, cursor) = self.char_to_byte(chars);
        self.gap_start_chars = chars;
        self.gap_start_cursor = cursor;
        self.buffer.move_gap(bytes);
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.rope.is_empty()
    }

    /// Returns the length in chars
    pub fn len(&self) -> usize {
        self.rope.len::<CharMetric>()
    }

    /// Inserts strings
    pub fn insert(&mut self, at: usize, s: &str) {
        if s.is_empty() {
            return;
        }
        if self.rope.is_empty() {
            if at == 0 {
                self.rope.init(compute_metrics(s));
                self.buffer.insert(0, s);
                debug_assert!(!self.rope.is_empty());
                let c = self.rope.cursor_at::<BaseMetric>(s.len()).expect("rope not empty");
                let c_char_offset = c.get(&self.rope).chars;
                self.gap_start_cursor = Some(Cursor {
                    base: c,
                    char_offset: c_char_offset,
                });
                self.gap_start_chars = self.rope.len::<CharMetric>();
            }
            return;
        }
        self.move_cursor(at);
        let Some(cursor) = &self.gap_start_cursor else { return };
        self.buffer.insert(at, s);
        let (delta, cursor) = update_insert(&mut self.rope, s, cursor);
        self.gap_start_cursor = Some(cursor);
        self.gap_start_chars += delta;
    }

    /// Deletes chars in range
    pub fn delete(&mut self, char_range: Range<usize>) {
        if char_range.is_empty() {
            return;
        }
        if self.rope.is_empty() {
            return;
        }
        let (byte_start, pos_start) = self.char_to_byte(char_range.start);
        let (byte_end, pos_end) = self.char_to_byte(char_range.end);
        let (Some(pos_start), Some(pos_end)) = (pos_start, pos_end) else { return };
        self.buffer.delete(byte_start..byte_end);
        self.gap_start_chars = char_range.start;

        if pos_start.base.is_same_piece(&pos_end.base) {
            if pos_start.char_offset == 0 && pos_end.is_tail(&self.rope) {
                let next = pos_start.base.nearby_piece(&self.rope);
                pos_start.base.delete(&mut self.rope);
                self.gap_start_cursor = next.map(|base| {
                    let char_offset = if base.offset().value == 0 { 0 } else { base.get(&self.rope).chars };
                    Cursor { base, char_offset }
                });
            } else {
                let delta = &Segment {
                    length: byte_end - byte_start,
                    chars: pos_end.char_offset - pos_start.char_offset,
                }.negate();
                pos_start.base.get_mut(&mut self.rope).add_assign(delta);
                pos_start.base.update(&mut self.rope, delta);
                self.gap_start_cursor = Some(pos_start);
            }
            return;
        }

        let start = if pos_start.char_offset == 0 {
            self.gap_start_cursor = pos_start.prev(&self.rope);
            pos_start
        } else {
            self.gap_start_cursor = Some(pos_start.clone());
            let mut delta = pos_start.head();
            delta.sub_assign(pos_start.base.get(&self.rope));
            pos_start.base.get_mut(&mut self.rope).add_assign(&delta);
            pos_start.base.update(&mut self.rope, &delta);
            pos_start.next(&self.rope).expect("start < end")
        };
        let end = if pos_end.is_tail(&self.rope) {
            Some(pos_end)
        } else {
            let delta = pos_end.head().negate();
            pos_end.base.get_mut(&mut self.rope).add_assign(&delta);
            pos_end.base.update(&mut self.rope, &delta);
            if pos_end.base.is_same_piece(&start.base) {
                None
            } else {
                pos_end.prev(&self.rope)
            }
        };

        if let Some(end) = end {
            start.base.delete_many_to(&mut self.rope, end.base, |_| {});
        }

        if self.gap_start_cursor.is_none() && let Some(base) = self.rope.cursor_at(0) {
            self.gap_start_cursor = Some(Cursor { base, char_offset: 0 });
        }
    }

    /// Gets substrings
    pub fn substring(&self, char_range: Range<usize>) -> String {
        if char_range.is_empty() {
            return Default::default();
        }
        let (start, _) = self.char_to_byte(char_range.start);
        let (end, _) = self.char_to_byte(char_range.end);
        debug_assert!(start < end);
        unsafe {
            if end <= self.buffer.gap_start {
                str::from_utf8_unchecked(&self.buffer.s[start..end]).to_string()
            } else if start >= self.buffer.gap_start {
                let gap = self.buffer.gap_size();
                str::from_utf8_unchecked(&self.buffer.s[start+gap..end+gap]).to_string()
            } else {
                let mut s = String::new();
                s.reserve_exact(end - start);
                s.push_str(&self.buffer.before_gap()[start..]);
                s.push_str(&self.buffer.after_gap()[..end-self.buffer.gap_start]);
                s
            }
        }
    }
}

fn update_insert(rope: &mut RopeBase<Segment>, s: &str, cursor: &Cursor) -> (usize, Cursor) {
    let mut pieces = compute_metrics(s);
    debug_assert!(!s.is_empty());
    // merging fast path
    if pieces.len() == 1 {
        let piece = pieces.next().expect("len == 1");
        debug_assert!(pieces.next().is_none());
        if cursor.base.get(rope).length < MAX_PIECE_LEN {
            cursor.base.get_mut(rope).add_assign(&piece);
            cursor.base.update(rope, &piece);
            return (pieces.chars, Cursor {
                base: cursor.base.with_offset(cursor.base.offset().value + piece.length),
                char_offset: cursor.char_offset + piece.chars,
            });
        }
        pieces.reset();
    }
    // split and predict ending cursor
    let (dir, next) = split_at(rope, cursor);
    debug_assert_eq!(0, next.as_ref().map(|c| c.base.offset().value).unwrap_or(0));
    if pieces.len() == 1 {
        let piece = pieces.next().expect("len == 1");
        debug_assert!(pieces.next().is_none());
        if dir == LEFT {
            cursor.base.insert_left(rope, piece);
        } else {
            cursor.base.insert_right(rope, piece);
        }
    } else if dir == LEFT {
        cursor.base.insert_many_before(rope, &mut pieces);
    } else {
        cursor.base.insert_many_after(rope, &mut pieces);
    }
    (pieces.chars, if let Some(next) = next {
        next
    } else {
        // rightest cursor
        let base = rope.cursor_at::<BaseMetric>(rope.base_len()).expect("in range");
        let char_offset = base.get(rope).chars;
        Cursor { base, char_offset }
    })
}
fn split_at(rope: &mut RopeBase<Segment>, cursor: &Cursor) -> (usize, Option<Cursor>) {
    if cursor.char_offset == 0 {
        return (LEFT, Some(cursor.clone()));
    }
    let piece = cursor.base.get(rope);
    if piece.chars == cursor.char_offset {
        return (RIGHT, cursor.base.next_piece(rope).map(|next| Cursor { base: next, char_offset: 0 }));
    }
    let total = piece.summarize();
    let head = cursor.head();
    let mut tail = total;
    tail.sub_assign(&head);
    *cursor.base.get_mut(rope) = head;
    cursor.base.update(rope, &tail.negate());
    let next = cursor.base.insert_right(rope, tail);
    (RIGHT, Some(Cursor { base: next, char_offset: 0 }))
}

struct ExactSegmentIterator<'a> {
    s: &'a str,
    byte_index: usize,
    /// Emitted chars
    chars: usize,
    /// Remaining item count
    count: usize,
}
fn compute_metrics<'a>(s: &'a str) -> ExactSegmentIterator<'a> {
    ExactSegmentIterator { s, byte_index: 0, chars: 0, count: s.len().div_ceil(MAX_PIECE_LEN) }
}
impl<'a> ExactSegmentIterator<'a> {
    fn reset(&mut self) {
        *self = compute_metrics(self.s);
    }
}
impl<'a> ExactSizeIterator for ExactSegmentIterator<'a> {
}
impl<'a> Iterator for ExactSegmentIterator<'a> {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next_index = if self.count == 0 {
            self.s.len()
        } else {
            let mut max_next = self.byte_index + MAX_PIECE_LEN;
            if max_next >= self.s.len() {
                self.s.len()
            } else {
                while self.s.is_char_boundary(max_next) {
                    max_next -= 1;
                }
                max_next
            }
        };
        let chars = self.s[self.byte_index..next_index].chars().count();
        self.chars += chars;
        Some(Segment { length: next_index - self.byte_index, chars })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roperig_test::{test_simple_string_fuzz, FuzzOp};
    use rand::Rng;

    #[test]
    fn test_gap_buffer() {
        fn assert_gapped(buf: &Buffer, before: &str, after: &str) {
            let gap = 256 - before.len() - after.len();
            assert_eq!(Some(before), str::from_utf8(buf.before_gap().as_bytes()).ok());
            assert_eq!(Some(after), str::from_utf8(buf.after_gap().as_bytes()).ok());
            assert_eq!(gap, buf.gap_size());
        }

        let mut buf = Buffer::default();
        assert_eq!(0, buf.gap_size());
        buf.ensure_gap(256, 0);
        assert_eq!(256, buf.gap_size());
        buf.insert(0, "abcxyz");
        assert_gapped(&buf, "abcxyz", "");
        buf.move_gap(3);
        assert_gapped(&buf, "abc", "xyz");
        buf.move_gap(0);
        assert_gapped(&buf, "", "abcxyz");
        buf.move_gap(6);
        assert_gapped(&buf, "abcxyz", "");
    }

    #[test]
    fn test_count_chars() {
        let mut buf = GapBuffer::default();
        buf.insert(0, "helloworld");
        assert_eq!("helloworld", buf.substring(0..buf.len()));
        buf.insert(5, " ");
        assert_eq!("hello world", buf.substring(0..buf.len()));
        buf.rope.is_valid();
        buf.delete(1..buf.len() - 1);
        assert_eq!("hd", buf.substring(0..buf.len()));
        buf.rope.is_valid();
    }

    #[test]
    #[allow(clippy::len_zero)]
    fn test_many_ops() {
        let mut buf = GapBuffer::default();
        test_simple_string_fuzz(move |op, expected, rng| {
            match op {
                FuzzOp::Insert(at, s) => buf.insert(at, &s),
                FuzzOp::Delete(r) => buf.delete(r),
            }
            assert_eq!(buf.is_empty(), buf.len() == 0);
            let from = rng.random_range(0..=buf.len());
            let to = rng.random_range(from..=buf.len());
            assert_eq!(expected.len(), buf.len());
            assert_eq!(buf.substring(from..to), expected[from..to]);
            buf.rope.tree.is_valid();
        }, false);
    }
}
