use crate::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use crate::string::{RopeContainer, WithCharMetric};
use std::mem;
use std::ops::Range;

#[derive(Default)]
struct TreeBuffer {
    buffer: String,
}

struct TreePiece {
    buffer: usize,
    buffer_offset: usize,
    sum: TreeSum,
}

#[derive(Default, Eq, PartialEq, Copy, Clone)]
struct TreeSum {
    length: usize,
    chars: usize,
}

/// A [piece tree] implementation
///
/// [piece tree]: https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation
pub struct PieceTree {
    buffer: usize,
    tree: Rope<TreePiece>,
}

impl TreeSum {
    fn summarize(s: &str) -> TreeSum {
        TreeSum {
            length: s.len(),
            chars: s.chars().count(),
        }
    }
}

impl Sum for TreeSum {
    fn len(&self) -> usize {
        self.length
    }

    fn add_assign(&mut self, other: &Self) {
        self.length = self.length.wrapping_add(other.length);
        self.chars = self.chars.wrapping_add(other.chars);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.length = self.length.wrapping_sub(other.length);
        self.chars = self.chars.wrapping_sub(other.chars);
    }

    fn identity() -> Self {
        Self::default()
    }
}

impl Summable for TreePiece {
    type S = TreeSum;

    fn summarize(&self) -> Self::S {
        self.sum
    }
}
impl WithCharMetric for TreePiece {
    fn get<'a>(&self, ctx: &'a Self::Context) -> &'a str {
        &ctx[self.buffer].buffer[self.buffer_offset..self.buffer_offset + self.sum.length]
    }

    fn chars(sum: &TreeSum) -> usize {
        sum.chars
    }
}

impl TreePiece {
    fn split(&mut self, context: &mut [TreeBuffer], at: usize) -> Self {
        let buffer = &context[self.buffer].buffer;
        let sum = TreeSum::summarize(
            &buffer[self.buffer_offset + at..self.buffer_offset + self.len()],
        );
        self.sum.sub_assign(&sum);
        Self {
            buffer: self.buffer,
            buffer_offset: self.buffer_offset + at,
            sum,
        }
    }
}

const MAX_PIECE_LEN: usize = 256;

impl RopePiece for TreePiece {
    type Context = Vec<TreeBuffer>;

    fn insert_or_split(
        &mut self, context: &mut Self::Context, other: Self, offset: usize,
    ) -> SplitResult<Self> {
        if offset == self.len() {
            if other.buffer == self.buffer
                && self.len() < MAX_PIECE_LEN
                && other.buffer_offset == self.buffer_offset + self.len() {
                self.sum.add_assign(&other.sum);
                SplitResult::Merged
            } else {
                SplitResult::TailSplit(other)
            }
        } else {
            let split = self.split(context, offset);
            SplitResult::MiddleSplit(other, split)
        }
    }

    fn delete_range(&mut self, context: &mut Self::Context, range: Range<usize>) -> DeleteResult<Self> {
        let from = range.start;
        let to = range.end;
        if from == 0 {
            let remaining = self.split(context, to);
            let del = mem::replace(self, remaining);
            DeleteResult::Updated(del.summarize())
        } else if to == self.len() {
            let mut del = self.split(context, from);
            del.notify_delete(context);
            DeleteResult::Updated(del.summarize())
        } else {
            let split = self.split(context, to);
            let del = self.split(context, from);
            DeleteResult::TailSplit {
                deleted: del.summarize(),
                split,
            }
        }
    }

    fn notify_delete(&mut self, context: &mut Self::Context) {
        let buffer = &mut context[self.buffer].buffer;
        if self.buffer_offset + self.len() == buffer.len() {
            buffer.truncate(self.buffer_offset);
        }
    }
}

impl RopeContainer<TreePiece> for PieceTree {
    fn rope(&self) -> &Rope<TreePiece> {
        &self.tree
    }
    fn rope_mut(&mut self) -> &mut Rope<TreePiece> {
        &mut self.tree
    }
}
impl Default for PieceTree {
    fn default() -> Self {
        Self::new()
    }
}

impl PieceTree {
    /// Create a new empty piece tree
    pub fn new() -> Self {
        Self {
            buffer: 0,
            tree: Rope::new(vec![TreeBuffer::default()]),
        }
    }

    /// Insert a string at a position
    pub fn insert(&mut self, at: usize, string: &str) {
        if string.is_empty() {
            return;
        }
        let buffer = &mut self.tree.context_mut()[self.buffer].buffer;
        let piece = TreePiece {
            buffer: self.buffer,
            buffer_offset: buffer.len(),
            sum: TreeSum::summarize(string),
        };
        buffer.push_str(string);
        self.tree.insert(at, piece);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roperig_test::{test_simple_string_fuzz, FuzzOp};
    use rand::Rng;

    #[test]
    fn test_simple_delete() {
        let mut pt = PieceTree::new();
        pt.insert(0, "helloworld");
        pt.insert(5, " ");
        pt.delete_bytes(1..pt.len() - 1);
        assert_eq!("hd", pt.substring(0..pt.len()));
        pt.delete_bytes(1..2);
        assert_eq!("h", pt.substring(0..pt.len()));

        let mut pt = PieceTree::new();
        pt.insert(0, "helloworld");
        pt.delete_bytes(5..10);
        assert_eq!("hello", pt.substring(0..pt.len()));
    }

    #[test]
    fn test_many_ops() {
        let mut pt = PieceTree::new();
        test_simple_string_fuzz(move |op, expected, rng| {
            match op {
                FuzzOp::Insert(at, s) => pt.insert(at, &s),
                FuzzOp::Delete(r) => pt.delete_bytes(r),
            }
            assert_eq!(pt.is_empty(), pt.len() == 0);
            assert_eq!(pt.is_empty(), pt.char_len() == 0);
            let from_chars = rng.random_range(0..=pt.char_len());
            let to_chars = rng.random_range(from_chars..=pt.char_len());
            let from = pt.char_to_byte(from_chars);
            let to = pt.char_to_byte(to_chars);
            assert_eq!(from_chars, pt.byte_to_char(from));
            assert_eq!(to_chars, pt.byte_to_char(to));
            assert_eq!(pt.substring(from..to), expected[from..to]);
            assert_eq!(expected.len(), pt.len());
            pt.tree.tree.is_valid();
        }, true);
    }
}
