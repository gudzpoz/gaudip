use crate::metrics::WithCharMetric;
use crate::piece::{DeleteResult, Insertion, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use crate::string::RopeContainer;
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
    fn substring<F, R: Default>(&self, ctx: &Self::Context, range: Range<usize>, _abs: usize, mut f: F) -> R
    where
        F: FnMut(&str, R) -> R
    {
        let s = &ctx[self.buffer]
            .buffer[self.buffer_offset + range.start..self.buffer_offset + range.end];
        f(s, R::default())
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
    const ABS: bool = false;

    fn insert_or_split(
        &mut self, context: &mut Self::Context, other: Insertion<Self>, offset: &Self::S,
    ) -> SplitResult<Self> {
        let offset = offset.len();
        if offset == 0 {
            SplitResult::HeadSplit(other.0)
        } else if offset == self.len() {
            if other.0.buffer == self.buffer
                && self.len() < MAX_PIECE_LEN
                && other.0.buffer_offset == self.buffer_offset + self.len() {
                self.sum.add_assign(&other.0.sum);
                SplitResult::Merged
            } else {
                SplitResult::TailSplit(other.0)
            }
        } else {
            let split = self.split(context, offset);
            SplitResult::MiddleSplit(other.0, split)
        }
    }

    fn delete_range(&mut self, context: &mut Self::Context, from: &Self::S, to: &Self::S) -> DeleteResult<Self> {
        let from = from.len();
        let to = to.len();
        if from == 0 {
            let remaining = self.split(context, to);
            let del = mem::replace(self, remaining);
            DeleteResult::Updated(del.summarize())
        } else if to == self.len() {
            let mut del = self.split(context, from);
            del.delete(context);
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

    fn delete(&mut self, context: &mut Self::Context) {
        let buffer = &mut context[self.buffer].buffer;
        if self.buffer_offset + self.len() == buffer.len() {
            buffer.truncate(self.buffer_offset);
        }
    }

    fn measure_offset(&self, context: &Self::Context, base_offset: usize, _abs: usize) -> Self::S {
        let buffer = &context[self.buffer].buffer;
        TreeSum::summarize(
            &buffer[self.buffer_offset..self.buffer_offset + base_offset],
        )
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
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rand_char_pos(rng: &mut ChaCha8Rng, s: &str) -> usize {
        let mut at = rng.random_range(0..=s.len());
        while !s.is_char_boundary(at) {
            at += 1;
        }
        at
    }

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
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut pt = PieceTree::new();
        let mut expected = String::default();
        let mut random_s = String::default();
        for _ in 0..100000 {
            let insert = pt.is_empty() || rng.random_bool(0.5);
            if insert {
                let at = rand_char_pos(&mut rng, &expected);
                let len = rng.random_range(0..=64);
                random_s.clear();
                (0..len)
                    .map(|_| if rng.random_bool(0.5) {
                        rng.random_range('a'..='z')
                    } else {
                        rng.random_range('一'..='😄')
                    }).for_each(|c| random_s.push(c));
                expected.insert_str(at, &random_s);
                pt.insert(at, &random_s);
            } else {
                let from = rand_char_pos(&mut rng, &expected);
                let len = rand_char_pos(&mut rng, &expected[from..]);
                expected.drain(from..from+len);
                pt.delete_bytes(from..from+len);
            }
            assert_eq!(pt.is_empty(), pt.len() == 0);
            assert_eq!(pt.is_empty(), pt.char_len() == 0);
            let from = rng.random_range(0..=pt.char_len());
            let to = rng.random_range(from..=pt.char_len());
            let from = pt.char_to_byte(from);
            let to = pt.char_to_byte(to);
            assert_eq!(pt.substring(from..to), expected[from..to]);
            assert_eq!(expected.len(), pt.len());
            pt.tree.is_valid();
        }
    }
}
