use std::ops::Range;
use crate::metrics::{BaseMetric, CharMetric, Metric, WithCharMetric};
use crate::piece::{RopePiece, Sum};
use crate::roperig::Rope;

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
        self.rope().len()
    }

    /// Returns true if the tree is empty
    fn is_empty(&self) -> bool {
        self.rope().is_empty()
    }

    /// Returns the length of the tree, in characters
    fn char_len(&self) -> usize {
        self.rope().measure::<CharMetric>()
    }

    /// Returns a substring of the rope
    fn substring(&self, range: Range<usize>) -> String {
        let mut gather = String::with_capacity(range.len());
        self.substring_store(range, &mut gather);
        gather
    }

    /// Appends the substring of the rope into `buffer`
    fn substring_store(&self, range: Range<usize>, buffer: &mut String) {
        self.rope().for_range::<BaseMetric>(range, |ctx, s, range, abs| {
            s.substring(ctx, range.start.len()..range.end.len(), abs, |sub, _| buffer.push_str(sub));
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
        self.cursor::<CharMetric>(offset)
            .map(|c| <BaseMetric as Metric<T>>::measure(&c.abs_offset()))
            .unwrap_or(self.len())
    }
    /// Converts a byte offset to a char offset
    pub fn byte_to_char(&self, offset: usize) -> usize {
        self.cursor::<BaseMetric>(offset)
            .map(|c| <CharMetric as Metric<T>>::measure(&c.abs_offset()))
            .unwrap_or(self.len())
    }
}
