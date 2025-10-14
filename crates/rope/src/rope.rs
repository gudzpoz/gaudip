use std::ops::Range;

use crate::metrics::{CharMetric, WithCharMetric};
use crate::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use crate::string::RopeContainer;

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
    fn substring<F, R: Default>(&self, _context: &Self::Context, range: Range<usize>, _abs: usize, mut f: F) -> R
    where
        F: FnMut(&str, R) -> R
    {
        let s = &self.s[range];
        f(s, R::default())
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
    const ABS: bool = false;

    fn insert_or_split(&mut self, _context: &mut Self::Context, other: Self, offset: &Self::S) -> SplitResult<Self> {
        let offset = offset.len();
        let len = self.s.len() + other.s.len();
        if offset == 0 {
            if len > MAX_PIECE_SIZE {
                return SplitResult::HeadSplit(other);
            }
        } else if offset == self.s.len() {
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

    fn delete_range(&mut self, _context: &mut Self::Context, from: &Self::S, to: &Self::S) -> DeleteResult<Self> {
        let from = from.len();
        let to = to.len();
        let extra = Ext::from_str(&self.s[from..to]);
        let chars = self.s[from..to].chars().count();
        self.s.drain(from..to);
        DeleteResult::Updated(StringSum {
            bytes: to - from,
            chars,
            extra,
        })
    }

    fn delete(&mut self, _context: &mut Self::Context) {
    }

    fn measure_offset(&self, _context: &Self::Context, base_offset: usize, _abs: usize) -> Self::S {
        StringSum {
            bytes: base_offset,
            chars: self.s[..base_offset].chars().count(),
            extra: Ext::from_str(&self.s[..base_offset]),
        }
    }
}

impl<Ext: Sum + FromStr> Default for StringRope<Ext> {
    fn default() -> Self {
        Self(Rope::default())
    }
}

impl<Ext: Sum + FromStr> StringRope<Ext> {
    /// Returns the length of the rope in bytes
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Returns the length of the rope in characters
    pub fn chars(&self) -> usize {
        self.0.measure::<CharMetric>()
    }
    /// Returns true if the rope is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
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
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut expected = String::new();
        let mut rope = StringRope::<()>::default();
        for _ in 0..1000000 {
            if expected.is_empty() || rng.random_bool(0.5) {
                let from = next_char_boundary(&expected, rng.random_range(0..=expected.len()));
                let to = next_char_boundary(&expected, rng.random_range(from..=expected.len()));
                expected.drain(from..to);
                rope.delete_bytes(from..to);
            } else {
                let offset = next_char_boundary(&expected, rng.random_range(0..=expected.len()));
                let len = rng.random_range(0..100);
                let s = (0..len).map(|_| {
                    let range = if rng.random_bool(0.5) {
                        'A'..='z'
                    } else {
                        '一'..='😄'
                    };
                    rng.random_range(range)
                }).collect::<String>();
                expected.insert_str(offset, &s);
                rope.insert(offset, s);
            }
            let start = rng.random_range(0..=expected.len());
            let end = rng.random_range(start..=expected.len());
            let start = next_char_boundary(&expected, start);
            let start_chars = expected[..start].chars().count();
            assert_eq!(start_chars, rope.byte_to_char(start));
            assert_eq!(start, rope.char_to_byte(start_chars));
            let end = next_char_boundary(&expected, end);
            assert_eq!(&expected[start..end], rope.substring(start..end));
        }
    }

    fn next_char_boundary(s: &str, mut i: usize) -> usize {
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i
    }
}
