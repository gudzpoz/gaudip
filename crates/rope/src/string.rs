use std::ops::Range;
use crate::metrics::{BaseMetric, Metric};
use crate::piece::RopePiece;
use crate::ropebase::ConvertedPosition;
use crate::roperig::Rope;

/// Basic methods to allow automatic implementation of [CharMetric]
pub trait WithCharMetric: RopePiece {
    /// Returns the string of this piece
    fn get<'a>(&'a self, ctx: &'a Self::Context) -> &'a str;

    /// Returns the number of characters in the piece
    fn chars(sum: &Self::S) -> usize;
}

/// Metric that measure char counts
pub struct CharMetric();
impl<T: WithCharMetric> Metric<T> for CharMetric {
    fn measure(sum: &T::S) -> usize {
        T::chars(sum)
    }
}

/// A utility trait for types that contain a rope
///
/// It contains some convenience methods for string-like ropes.
pub trait RopeContainer<T: RopePiece + WithCharMetric> {
    /// Returns a reference to the contained rope
    fn rope(&self) -> &Rope<T>;

    /// Returns a mutable reference to the contained rope
    fn rope_mut(&mut self) -> &mut Rope<T>;

    /// Returns the length of the tree, in bytes
    fn len(&self) -> usize {
        self.rope().base_len()
    }

    /// Returns true if the tree is empty
    fn is_empty(&self) -> bool {
        self.rope().is_empty()
    }

    /// Returns the length of the tree, in characters
    fn char_len(&self) -> usize {
        self.rope().len::<CharMetric>()
    }

    /// Returns a substring of the rope
    fn substring(&self, range: Range<usize>) -> String {
        let mut gather = String::with_capacity(range.len());
        self.substring_store(range, &mut gather);
        gather
    }

    /// Appends the substring of the rope into `buffer`
    fn substring_store(&self, range: Range<usize>, buffer: &mut String) {
        let rope = self.rope();
        let Some(c) = rope.cursor_at(range.start) else { return };
        c.for_range(&rope.tree, range.len(), |s, r| {
            buffer.push_str(&s.get(&rope.context)[r]);
            true
        });
    }

    /// Converts a char offset to a byte offset
    fn char_to_byte(&self, offset: usize) -> usize {
        self.rope().char_to_byte(offset)
    }

    /// Converts a byte offset to a char offset
    fn byte_to_char(&self, offset: usize) -> usize {
        self.rope().byte_to_char(offset)
    }

    /// Deletes a range of bytes
    fn delete_bytes(&mut self, range: Range<usize>) {
        self.rope_mut().delete(range)
    }
}

impl<T: RopePiece + WithCharMetric> Rope<T> {
    /// Converts a char offset to a byte offset
    pub fn char_to_byte(&self, offset: usize) -> usize {
        let ConvertedPosition {
            piece, piece_position, offset_in_piece,
        } = self.tree.convert_metrics::<CharMetric, BaseMetric>(offset);
        let Some(piece) = piece else { return if offset == 0 { 0 } else { self.tree.len::<BaseMetric>() } };
        let s = piece.get(&self.context);
        str_indices::chars::to_byte_idx(s, offset_in_piece.value) + piece_position.value
    }
    /// Converts a byte offset to a char offset
    pub fn byte_to_char(&self, offset: usize) -> usize {
        let ConvertedPosition {
            piece, piece_position, offset_in_piece,
        } = self.tree.convert_metrics::<BaseMetric, CharMetric>(offset);
        let Some(piece) = piece else { return if offset == 0 { 0 } else { self.tree.len::<CharMetric>() } };
        let s = piece.get(&self.context);
        str_indices::chars::from_byte_idx(s, offset_in_piece.value) + piece_position.value
    }
}
