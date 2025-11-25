use std::ops::Range;

use crate::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use crate::string::{RopeContainer, WithCharMetric};

/// A wrapper around a [String], with precalculated stats.
pub struct StringExt<Ext: Sum + FromStr> {
    /// The string content.
    pub s: String,
    /// The number of characters in the string.
    pub chars: usize,
    /// Extra data, calculated from the string.
    pub extra: Ext,
}
/// A trait extracting stats from strings
pub trait FromStr {
    /// Extracts the stats from a string
    fn from_str(s: &str) -> Self;
}
impl FromStr for () {
    fn from_str(_s: &str) -> Self {
    }
}
impl<Ext: Sum + FromStr> From<String> for StringExt<Ext> {
    fn from(value: String) -> Self {
        let extra = Ext::from_str(&value);
        let chars = value.chars().count();
        Self {
            s: value,
            chars,
            extra,
        }
    }
}
/// Sum for [StringExt].
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StringSum<Ext: Sum> {
    bytes: usize,
    chars: usize,
    extra: Ext,
}
impl<Ext: Sum> Sum for StringSum<Ext> {
    fn len(&self) -> usize {
        self.bytes
    }

    fn add_assign(&mut self, other: &Self) {
        self.bytes = self.bytes.wrapping_add(other.bytes);
        self.chars = self.chars.wrapping_add(other.chars);
        self.extra.add_assign(&other.extra);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.bytes = self.bytes.wrapping_sub(other.bytes);
        self.chars = self.chars.wrapping_sub(other.chars);
        self.extra.sub_assign(&other.extra);
    }

    fn identity() -> Self {
        Self {
            bytes: 0,
            chars: 0,
            extra: Ext::identity(),
        }
    }
}
impl<Ext: Sum + FromStr> Summable for StringExt<Ext> {
    type S = StringSum<Ext>;

    fn summarize(&self) -> Self::S {
        Self::S {
            bytes: self.s.len(),
            chars: self.chars,
            extra: self.extra,
        }
    }
}

impl<Ext: Sum + FromStr> WithCharMetric for StringExt<Ext> {
    fn get<'a>(&'a self, _context: &'a Self::Context) -> &'a str {
        &self.s
    }

    fn chars(sum: &StringSum<Ext>) -> usize {
        sum.chars
    }
}

/// A [String] rope implementation based on [roperig].
pub struct StringRope<Ext: Sum + FromStr = ()>(Rope<StringExt<Ext>>);
const MIN_PIECE_SIZE: usize = 128;
const MAX_PIECE_SIZE: usize = 256;

impl<Ext: Sum + FromStr> RopeContainer<StringExt<Ext>> for StringRope<Ext> {
    fn rope(&self) -> &Rope<StringExt<Ext>> {
        &self.0
    }
    fn rope_mut(&mut self) -> &mut Rope<StringExt<Ext>> {
        &mut self.0
    }
}

impl<Ext: Sum + FromStr> RopePiece for StringExt<Ext> {
    type Context = ();

    fn insert_or_split(&mut self, _context: &mut Self::Context, other: Self, offset: usize) -> SplitResult<Self> {
        let len = self.s.len() + other.s.len();
        if offset == self.s.len() {
            if len > MAX_PIECE_SIZE {
                return SplitResult::TailSplit(other);
            }
        } else if len > MAX_PIECE_SIZE {
            let tail_sum = Ext::from_str(&self.s[offset..]);
            let tail_chars = self.s[offset..].chars().count();
            let mut split = other;
            return if offset > MIN_PIECE_SIZE {
                split.s.push_str(&self.s[offset..]);
                split.chars += tail_chars;
                split.extra.add_assign(&tail_sum);
                self.s.truncate(offset);
                self.chars -= tail_chars;
                self.extra.sub_assign(&tail_sum);
                SplitResult::TailSplit(split)
            } else {
                let tail = self.s.drain(offset..).as_str().to_string();
                self.chars -= tail_chars;
                self.extra.sub_assign(&tail_sum);
                SplitResult::MiddleSplit(split, StringExt {
                    s: tail,
                    chars: tail_chars,
                    extra: tail_sum,
                })
            };
        }
        self.s.insert_str(offset, &other.s);
        self.chars += other.chars;
        self.extra.add_assign(&other.extra);
        SplitResult::Merged
    }

    fn delete_range(&mut self, _context: &mut Self::Context, range: Range<usize>) -> DeleteResult<Self> {
        let from = range.start;
        let to = range.end;
        let extra = Ext::from_str(&self.s[from..to]);
        let chars = self.s[from..to].chars().count();
        self.s.drain(from..to);
        self.chars -= chars;
        self.extra.sub_assign(&extra);
        DeleteResult::Updated(StringSum {
            bytes: to - from,
            chars,
            extra,
        })
    }

    fn notify_delete(&mut self, _context: &mut Self::Context) {
    }
}

impl<Ext: Sum + FromStr> Default for StringRope<Ext> {
    fn default() -> Self {
        Self(Rope::default())
    }
}

impl<Ext: Sum + FromStr> StringRope<Ext> {
    /// Inserts a string at the given byte offset
    pub fn insert_str(&mut self, offset: usize, s: &str) {
        self.insert(offset, s.to_string())
    }
    /// Inserts a string at the given byte offset
    pub fn insert(&mut self, offset: usize, s: String) {
        self.0.insert(offset, s.to_string().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roperig_test::{test_simple_string_fuzz, FuzzOp};
    use rand::Rng;

    #[test]
    fn test_many_ops() {
        let mut rope = StringRope::<()>::default();
        test_simple_string_fuzz(move |op, expected, rng| {
            match op {
                FuzzOp::Insert(at, s) => rope.insert(at, s),
                FuzzOp::Delete(r) => rope.delete_bytes(r),
            }
            let start = rng.random_range(0..=expected.len());
            let end = rng.random_range(start..=expected.len());
            let start = next_char_boundary(expected, start);
            let start_chars = expected[..start].chars().count();
            assert_eq!(start_chars, rope.byte_to_char(start));
            assert_eq!(start, rope.char_to_byte(start_chars));
            let end = next_char_boundary(expected, end);
            assert_eq!(&expected[start..end], rope.substring(start..end));
        }, true);
    }

    fn next_char_boundary(s: &str, mut i: usize) -> usize {
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i
    }
}
