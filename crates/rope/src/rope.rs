use crate::metrics::{BaseMetric, Metric};
use crate::piece::{DeleteResult, Insertion, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;

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

struct CharMetric();
impl<Ext: Sum + FromStr> Metric<StringExt<Ext>> for CharMetric {
    fn measure(sum: &StringSum<Ext>) -> usize {
        sum.chars
    }

    fn from_base_units(piece: &StringExt<Ext>, base_units: usize) -> usize {
        piece.s[..base_units].chars().count()
    }

    fn to_base_units(piece: &StringExt<Ext>, measurement: usize) -> usize {
        piece.s.char_indices().nth(measurement).map(|(i, _)| i).unwrap_or(piece.s.len())
    }
}

/// A [String] rope implementation based on [roperig].
pub struct StringRope<Ext: Sum + FromStr = ()>(Rope<StringExt<Ext>>);
const MIN_PIECE_SIZE: usize = 128;
const MAX_PIECE_SIZE: usize = 256;

impl<Ext: Sum + FromStr> RopePiece for StringExt<Ext> {
    type Context = ();

    fn insert_or_split(&mut self, _context: &mut Self::Context, other: Insertion<Self>, offset: usize) -> SplitResult<Self> {
        let len = self.s.len() + other.0.s.len();
        if offset == 0 {
            if len > MAX_PIECE_SIZE {
                return SplitResult::HeadSplit(other.0);
            }
        } else if offset == self.s.len() {
            if len > MAX_PIECE_SIZE {
                return SplitResult::TailSplit(other.0);
            }
        } else if len > MAX_PIECE_SIZE {
            let tail_sum = Ext::from_str(&self.s[offset..]);
            let tail_chars = self.s[offset..].chars().count();
            let mut split = other.0;
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
        self.s.insert_str(offset, &other.0.s);
        self.chars += other.0.chars;
        self.extra.add_assign(&other.0.extra);
        SplitResult::Merged
    }

    fn delete_range(&mut self, _context: &mut Self::Context, from: usize, to: usize) -> DeleteResult<Self> {
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
    /// Deletes a range of bytes
    pub fn delete(&mut self, offset: usize, len: usize) {
        self.0.delete(offset, len)
    }
    /// Returns a substring of the rope
    pub fn substring(&self, from: usize, to: usize) -> String {
        let mut gather = String::with_capacity(to - from);
        self.0.for_range::<BaseMetric>(from..to, |_, s, range| {
            gather.push_str(&s.s[range]);
            true
        });
        gather
    }
    /// Converts a char offset to a byte offset
    pub fn char_to_byte(&self, offset: usize) -> usize {
        self.0.cursor::<CharMetric>(offset)
            .map(|c| c.abs_offset::<BaseMetric>())
            .unwrap_or(self.len())
    }
    /// Converts a byte offset to a char offset
    pub fn byte_to_char(&self, offset: usize) -> usize {
        self.0.cursor::<BaseMetric>(offset)
            .map(|c| c.abs_offset::<CharMetric>())
            .unwrap_or(self.len())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use rand_chacha::rand_core::SeedableRng;
    use super::*;

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
                rope.delete(from, to - from);
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
            assert_eq!(&expected[start..end], rope.substring(start, end));
        }
    }

    fn next_char_boundary(s: &str, mut i: usize) -> usize {
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i
    }
}
