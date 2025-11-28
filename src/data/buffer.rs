use crate::data::segment::PangoMetric;

use super::segment::EStrSegment;
use std::ops::Range;
use roperig::piece::{Sum, Summable};
use roperig::metrics::BaseMetric;
use roperig::ropebase::{ConvertedPosition, PartialCursorPos, RopeBase};

/// A buffer for a line
///
/// All public functions use char indices (rather than byte indices).
#[derive(Default)]
pub struct LineBuffer {
    pango_str: Vec<u8>,
    metrics: RopeBase<EStrSegment>,
}

impl LineBuffer {
    pub fn pango_bytes(&self) -> &[u8] {
        &self.pango_str
    }
    pub fn chars(&self) -> usize {
        self.metrics.base_len()
    }

    pub fn byte_index_to_char(&self, bytes: usize) -> usize {
        let ConvertedPosition {
            piece, piece_position, offset_in_piece,
        } = self.metrics.convert_metrics::<PangoMetric, BaseMetric>(bytes);
        let Some(piece) = piece else { return piece_position.value };
        piece.pango_offset_to_char_offset(offset_in_piece.value) + piece_position.value
    }

    pub fn char_index_to_byte(&self, char_index: usize) -> usize {
        self.char_index_to_byte_pos(char_index).0
    }

    pub fn split(&mut self, char_index: usize) -> LineBuffer {
        let (byte_offset, cursor) = self.char_index_to_byte_pos(char_index);
        let tail_str = self.pango_str.split_off(byte_offset);

        let Some(cursor) = cursor else { return LineBuffer::default() };
        let at = cursor.get_mut(&mut self.metrics);
        let right = at.split(cursor.offset().value);
        cursor.update(&mut self.metrics, &right.summarize().negate());

        let mut pieces = vec![right];
        if let Some(rest) = cursor.next_piece(&self.metrics) {
            let end = self.metrics.cursor_at::<BaseMetric>(self.metrics.base_len()).expect("not empty");
            rest.delete_many_to(&mut self.metrics, end, |piece| pieces.push(piece));
        }

        let mut new = RopeBase::default();
        new.init(pieces.into_iter());
        LineBuffer {
            pango_str: tail_str,
            metrics: new,
        }
    }

    pub fn delete(&mut self, chars: Range<usize>) {
        if chars.is_empty() {
            return;
        }
        let (start_bytes, from) = self.char_index_to_byte_pos(chars.start);
        let (end_bytes, to) = self.char_index_to_byte_pos(chars.end);
        let (Some(mut from), Some(to)) = (from, to) else { return };
        self.pango_str.drain(start_bytes..end_bytes);
        if from.is_same_piece(&to) {
            let piece = from.get_mut(&mut self.metrics);
            let piece_prev = piece.summarize();
            let tail = piece.split(to.offset().value);
            if from.offset().value != 0 {
                piece.split(from.offset().value);
            }
            let delta = piece.summarize().sub(piece_prev);
            from.update(&mut self.metrics, &delta);
            if !tail.is_empty() {
                from.insert_right(&mut self.metrics, tail);
            }
            if from.offset().value == 0 {
                from.delete(&mut self.metrics);
            }
            return;
        }
        if from.offset().value != 0 {
            let piece = from.get_mut(&mut self.metrics);
            let delta = piece.split(from.offset().value).summarize().negate();
            from.update(&mut self.metrics, &delta);
            from = from.next_piece(&self.metrics).expect("from < to");
        }
        if to.offset().value != to.get(&self.metrics).len() {
            let piece = to.get_mut(&mut self.metrics);
            let keep = piece.split(to.offset().value);
            let delta = keep.summarize().negate();
            to.update(&mut self.metrics, &delta);
            to.insert_right(&mut self.metrics, keep);
        }
        from.delete_many_to(&mut self.metrics, to, |_| {});
    }

    pub fn insert(&mut self, char_index: usize, s: EStrSegment) {
        let (offset, cursor) = self.char_index_to_byte_pos(char_index);
        let Some(cursor) = cursor else {
            if self.metrics.is_empty() && char_index == 0 {
                let extra = s.pango_bytes();
                self.expand_str(offset, extra);
                s.write(&mut self.pango_str[offset..offset + extra]);
                self.metrics.init(Some(s).into_iter());
            }
            return;
        };
        let extra = s.pango_bytes();
        self.expand_str(offset, extra);
        s.write(&mut self.pango_str[offset..offset + extra]);

        if cursor.offset().value == 0 {
            cursor.insert_left(&mut self.metrics, s);
        } else if cursor.offset().value == cursor.get(&self.metrics).pango_bytes() {
            cursor.insert_right(&mut self.metrics, s);
        } else {
            let piece = cursor.get_mut(&mut self.metrics);
            let tail = piece.split(cursor.offset().value);
            let delta = tail.summarize().negate();
            cursor.update(&mut self.metrics, &delta);
            cursor.insert_right(&mut self.metrics, tail);
            cursor.insert_right(&mut self.metrics, s);
        }
    }

    fn expand_str(&mut self, offset: usize, extra: usize) {
        let end = self.pango_str.len();
        self.pango_str.resize(end + extra, 0);
        self.pango_str.copy_within(offset..end, offset + extra);
    }

    fn char_index_to_byte_pos(&self, index: usize) -> (usize, Option<PartialCursorPos<EStrSegment, BaseMetric>>) {
        let (acc, cursor) = self.metrics.accumulate::<BaseMetric, _, _>(
            index, |acc, s| acc + s.pango, 0,
        );
        let Some(cursor) = cursor else { return (acc, None) };
        let byte_offset = acc + cursor.get(&self.metrics).char_offset_to_pango_offset(cursor.offset().value);
        (byte_offset, Some(cursor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emacs_string() {
        let s = "Hello! 你好！ こんにちは！˚˖𓍢🌷✧˚.🎀⋆ \u{10FFFF}";
        for _ in 0..2 {
            let mut buffer = LineBuffer::default();
            buffer.insert(0, s.into());
            assert_eq!(s.as_bytes(), buffer.pango_str);
        }
    }

    #[test]
    fn test_split() {
        let mut buffer = LineBuffer::default();
        let s = "Hello!World!";
        buffer.insert(0, s.into());
        assert_eq!(s.as_bytes(), buffer.pango_bytes());
        let new_line = buffer.split(6);
        assert_eq!("Hello!".as_bytes(), buffer.pango_bytes());
        assert_eq!("World!".as_bytes(), new_line.pango_bytes());
    }
}
