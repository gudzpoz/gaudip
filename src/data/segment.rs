use std::io::{Cursor, Write};
use std::ops::Range;
use roperig::metrics::Metric;
use roperig::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use EStrSegment::*;

/// A segment of Emacs string
///
/// In contrast to normal strings, for which [BaseMetric] usually means bytes,
/// for [EStrSegment] (which is intended for rendering), [BaseMetric] means Emacs characters.
///
/// We don't support [CharMetric], but instead offer [PangoMetric] to measure the produced
/// Pango string length in bytes.
#[derive(Clone, Debug)]
pub enum EStrSegment {
    /// `chars`: Unicode chars; `pango`: UTF-8 bytes
    Unicode { str: String, chars: usize },
    /// `chars` and `pango`: the same
    Ascii { str: String },
    /// `chars`: char counts, i.e., raw byte count;
    /// `pango`: one byte for ASCII, 4 bytes for `0x80` and above (`\XXX`)
    RawBytes { str: Vec<u8>, pango: usize },
    /// `chars`: char counts (in `utf-8-emacs` coding), 1 or 4 character only for now;
    /// `pango`: a single byte (rendered with custom renderer)
    NonUnicode { c: u32 },
    /// `chars` and `pango`: 1
    Widget { width: usize, height: usize },
}
impl From<&str> for EStrSegment {
    fn from(value: &str) -> Self {
        Self::from_counted_utf8(value.to_string(), value.len())
    }
}

/// An enum to hopefully reduce allocations for ASCII/UTF-8 strings
pub enum OptimisticParseResult {
    Single(EStrSegment),
    Mixed(Vec<EStrSegment>),
}
impl ExactSizeIterator for OptimisticParseResult {}
impl Iterator for OptimisticParseResult {
    type Item = EStrSegment;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            OptimisticParseResult::Single(_) => {
                let s = std::mem::replace(self, OptimisticParseResult::Mixed(vec![]));
                Some(match s {
                    OptimisticParseResult::Single(s) => s,
                    OptimisticParseResult::Mixed(_) => unreachable!(),
                })
            },
            OptimisticParseResult::Mixed(estr_segments) => {
                estr_segments.pop()
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            OptimisticParseResult::Single(_) => (1, Some(1)),
            OptimisticParseResult::Mixed(v) => (v.len(), Some(v.len())),
        }
    }
}

impl EStrSegment {
    /// Creates a segment from ASCII bytes
    pub fn from_ascii(str: &[u8]) -> EStrSegment {
        debug_assert!(str.is_ascii());
        Ascii {
            str: unsafe { String::from_utf8_unchecked(str.to_owned()) }
        }
    }
    /// Creates a segment from raw bytes
    pub fn from_raw(bytes: &[u8]) -> EStrSegment {
        let len = raw_bytes_to_pango_byte_count(bytes);
        if len == bytes.len() {
            Ascii { str: unsafe { String::from_utf8_unchecked(bytes.to_vec()) } }
        } else {
            RawBytes{ str: bytes.to_vec(), pango: len }
        }
    }
    /// Creates a segment from UTF-8 bytes, with precalculated char count
    fn from_counted_utf8(str: String, chars: usize) -> EStrSegment {
        if str.len() == chars {
            Ascii { str }
        } else {
            Unicode { str, chars }
        }
    }
    /// Creates a segment from UTF-8 bytes
    pub fn from_utf8(str: &str) -> EStrSegment {
        EStrSegment::from_counted_utf8(str.to_string(), str.chars().count())
    }
    /// Creates a segment from `utf-8-emacs` bytes
    ///
    /// See `character.h` in Emacs code for details.
    pub fn from_emacs(str: &[u8]) -> OptimisticParseResult {
        let partial = match str::from_utf8(str) {
            Ok(str) => return OptimisticParseResult::Single(
                EStrSegment::from_counted_utf8(str.to_string(), str.chars().count()),
            ),
            Err(err) => err.valid_up_to(),
        };

        fn from_partial_utf8(str: &[u8], partial: Range<usize>) -> EStrSegment {
            let s = unsafe { str::from_utf8_unchecked(&str[partial]) };
            EStrSegment::from_counted_utf8(s.to_string(), s.chars().count())
        }

        let mut vec: Vec<EStrSegment> = vec![from_partial_utf8(str, 0..partial)];
        let mut from = partial;
        while from < str.len() {
            let (c, advance) = next_emacs_codepoint(str, from).unwrap_or((0x3FFF00 + str[from] as u32, 1));
            vec.push(EStrSegment::NonUnicode { c });
            from += advance;
            from += match str::from_utf8(&str[from..]) {
                Ok(str) => {
                    if !str.is_empty() {
                        vec.push(EStrSegment::from_counted_utf8(str.to_string(), str.chars().count()));
                    }
                    str.len()
                },
                Err(err) => {
                    let partial = err.valid_up_to();
                    if partial > 0 {
                        vec.push(from_partial_utf8(str, from..from + partial));
                    }
                    partial
                }
            };
        }
        OptimisticParseResult::Mixed(vec)
    }

    /// Returns the total number of chars
    pub fn chars(&self) -> usize {
        match self {
            Unicode { chars, .. } => *chars,
            Ascii { str } => str.len(),
            RawBytes { str, .. } => str.len(),
            NonUnicode { c } => if *c < 0x3FFF80 { 1 } else { 4 },
            Widget { .. } => 1,
        }
    }

    /// Returns the total number of pango bytes
    pub fn pango_bytes(&self) -> usize {
        match self {
            Unicode { str, .. } | Ascii { str } => str.len(),
            RawBytes { pango, .. } => *pango,
            NonUnicode { c } => if *c < 0x3FFF80 { 1 } else { 4 },
            Widget { .. } => 1,
        }
    }

    pub fn len(&self) -> usize {
        self.chars()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, bytes: &mut [u8]) {
        debug_assert!(bytes.len() == self.len());
        match self {
            Unicode { str, .. } => bytes.copy_from_slice(str.as_bytes()),
            Ascii { str } => bytes.copy_from_slice(str.as_bytes()),
            RawBytes { str, .. } => {
                let mut writer = Cursor::new(bytes);
                for b in str {
                    if *b < 128 {
                        write!(writer, "{}", *b as char).unwrap();
                    } else {
                        write!(writer, "\\{:3o}", *b).unwrap();
                    }
                }
                debug_assert!(self.len() as u64 == writer.position());
            }
            NonUnicode { c } => {
                debug_assert_eq!(bytes.len(), self.pango_bytes());
                if *c < 0x3FFF80 {
                    bytes.fill(b' ')
                } else {
                    let mut writer = Cursor::new(bytes);
                    write!(writer, "\\{:3o}", c - 0x3FFF00).unwrap();
                }
            },
            Widget { .. } => bytes[0] = b' ',
        }
    }

    pub fn split(&mut self, char_offset: usize) -> Self {
        debug_assert!(char_offset != 0 && char_offset <= self.chars());
        match self {
            Unicode { str, chars } => {
                let byte_offset = str_indices::chars::to_byte_idx(str, char_offset);
                let tail = str[byte_offset..].to_string();
                str.drain(byte_offset..);
                let tail_chars = *chars - char_offset;
                *chars = char_offset;
                Self::from_counted_utf8(tail, tail_chars)
            }
            Ascii { str } => {
                let tail = str[char_offset..].to_string();
                str.drain(char_offset..);
                let len = tail.len();
                Self::from_counted_utf8(tail, len)
            }
            RawBytes { str, pango } => {
                let tail = EStrSegment::from_raw(&str[char_offset..]);
                *pango -= tail.pango_bytes();
                str.drain(char_offset..);
                tail
            }
            NonUnicode { .. } | Widget { .. } => unreachable!(),
        }
    }

    pub fn char_offset_to_pango_offset(&self, char_offset: usize) -> usize {
        match self {
            Unicode { str, .. } => str_indices::chars::to_byte_idx(str, char_offset),
            Ascii { .. } => char_offset,
            RawBytes { str, .. } => raw_bytes_to_pango_byte_count(&str[..char_offset]),
            NonUnicode { .. } => if char_offset == 0 { 0 } else { self.pango_bytes() },
            Widget { .. } => char_offset,
        }
    }

    pub fn pango_offset_to_char_offset(&self, pango_offset: usize) -> usize {
        match self {
            Unicode { str, .. } => str_indices::chars::from_byte_idx(str, pango_offset),
            Ascii { .. } => pango_offset,
            RawBytes { str, .. } => raw_bytes_pango_to_char_index(str, pango_offset).0,
            NonUnicode { .. } => if pango_offset == 0 { 0 } else { 1 },
            Widget { .. } => pango_offset,
        }
    }
}

impl Summable for EStrSegment {
    type S = EStrInfo;

    fn summarize(&self) -> Self::S {
        EStrInfo { chars: self.chars(), pango: self.pango_bytes() }
    }
}

impl RopePiece for EStrSegment {
    type Context = ();

    fn insert_or_split(&mut self, _: &mut Self::Context, other: Self, offset: usize) -> SplitResult<Self> {
        assert!(!self.is_empty());
        if other.is_empty() {
            return SplitResult::Merged;
        }
        if offset == self.chars() {
            return SplitResult::TailSplit(other);
        }
        let tail = self.split(offset);
        SplitResult::MiddleSplit(other, tail)
    }

    fn delete_range(&mut self, _: &mut Self::Context, range: Range<usize>) -> DeleteResult<Self> {
        let bytes = match self {
            Unicode { str, chars } => {
                let byte_start = str_indices::chars::to_byte_idx(str, range.start);
                let byte_end = str_indices::chars::to_byte_idx(str, range.end);
                str.drain(byte_start..byte_end);
                *chars -= range.len();
                byte_end - byte_start
            }
            Ascii { str } => { str.drain(range.clone()); range.len() }
            RawBytes { str, pango } => {
                let pango_bytes = raw_bytes_to_pango_byte_count(&str[range.clone()]);
                str.drain(range.clone());
                *pango -= pango_bytes;
                pango_bytes
            },
            NonUnicode { .. } | Widget { .. } => unreachable!(),
        };
        DeleteResult::Updated(EStrInfo { chars: range.len(), pango: bytes })
    }

    fn notify_delete(&mut self, _context: &mut Self::Context) {
    }
}

fn raw_bytes_pango_to_char_index(str: &[u8], byte_index: usize) -> (usize, usize) {
    let mut remaining = byte_index;
    let mut i = 0usize;
    for c in str {
        if let Some(next) = remaining.checked_sub(if c.is_ascii() { 1 } else { 4 }) {
            remaining = next;
            i += 1;
        } else {
            break;
        }
    }
    (i, remaining)
}

/// Compute the required pango bytes for a raw byte slice
fn raw_bytes_to_pango_byte_count(str: &[u8]) -> usize {
    str.iter().map(|c| if c.is_ascii() { 1 } else { 4 }).sum()
}

fn next_emacs_codepoint(str: &[u8], i: usize) -> Option<(u32, usize)> {
    let b1 = str[i];
    let get = |i: usize| str.get(i).and_then(|c| if c & 0b1100_0000 == 0b1000_0000 {
        Some(c)
    } else {
        None
    });
    Some(match b1 {
        // ASCII
        0x00..=0x7F => (b1 as u32, 1),
        // Invalid starting byte
        0x80..=0xBF => return None,
        // Emacs eight-bit-char
        0xC0..=0xC1 => (
            0x3FFF80u32 | ((b1 & 1) << 6) as u32
                | (get(i + 1)? & 0x3F) as u32, 2,
        ),
        // UTF-8 2-byte sequence
        0xC2..=0xDF => (
            ((b1 & 0x1F) as u32) << 6
                | (get(i + 1)? & 0x3F) as u32, 2,
        ),
        // UTF-8 3-byte sequence
        0xE0..=0xEF => (
            ((b1 & 0xF) as u32) << 12
                | ((get(i + 1)? & 0x3F) as u32) << 6
                | ((get(i + 2)? & 0x3F) as u32),
            3,
        ),
        // UTF-8 4-byte sequence
        0xF0..=0xF7 => (
            ((b1 & 0x7) as u32) << 18
                | ((get(i + 1)? & 0x3F) as u32) << 12
                | ((get(i + 2)? & 0x3F) as u32) << 6
                | ((get(i + 3)? & 0x3F) as u32),
            4,
        ),
        // Emacs 5-byte sequence
        0xF8..=0xF8 => (
            (((get(i + 1)? & 0x0F) as u32) << 18)
                | ((get(i + 2)? & 0x3F) as u32) << 12
                | ((get(i + 3)? & 0x3F) as u32) << 6
                | ((get(i + 4)? & 0x3F) as u32),
            4,
        ),
        // Invalid starting byte
        _ => return None,
    })
}

/// Metrics for Emacs string and the corresponding Pango string
///
/// When displaying Emacs strings, we need to convert them to
/// a UTF-8 string that Pango accepts, while keeping it easy for
/// modification. And this rope serves exactly this purpose.
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct EStrInfo {
    /// The char count in this interval.
    ///
    /// See [str::chars].
    pub(crate) chars: usize,
    /// The encoded length in the Pango UTF-8 string
    ///
    /// For example, raw bytes are encoded as `\XXX`,
    /// so each char corresponds to 4 bytes.
    pub(crate) pango: usize,
}

impl Sum for EStrInfo {
    fn len(&self) -> usize {
        self.chars
    }

    fn add_assign(&mut self, other: &Self) {
        self.pango = self.pango.wrapping_add(other.pango);
        self.chars = self.chars.wrapping_add(other.chars);
    }

    fn sub_assign(&mut self, other: &Self) {
        self.pango = self.pango.wrapping_sub(other.pango);
        self.chars = self.chars.wrapping_sub(other.chars);
    }

    fn identity() -> Self {
        Self::default()
    }
}

pub struct PangoMetric();
impl Metric<EStrSegment> for PangoMetric {
    fn measure(sum: &<EStrSegment as Summable>::S) -> usize {
        sum.pango
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use roperig::metrics::BaseMetric;
    use roperig::ropebase::RopeBase;
    use roperig::roperig::Rope;
    use crate::data::segment::{EStrSegment, OptimisticParseResult, PangoMetric};

    fn simple_test_cases() -> Vec<(EStrSegment, usize, usize)> {
        let mut vec = Vec::new();
        let test_bytes = vec![1, 63, 128, 2, 64, 255];
        vec.push((EStrSegment::from_ascii("hello!".as_bytes()), 6, 6));
        vec.push((EStrSegment::from_raw("hello!".as_bytes()), 6, 6));
        vec.push((EStrSegment::from_raw(&test_bytes), 12, 6));
        let s = EStrSegment::from_emacs("😄😊".as_bytes());
        match s {
            OptimisticParseResult::Single(segment) =>
                vec.push((segment, 8, 2)),
            _ => panic!(),
        }

        for (_, bytes, chars) in &vec {
            assert_eq!(0, bytes % 2);
            assert_eq!(0, chars % 2);
        }
        vec
    }

    #[test]
    fn length_single() {
        let assert_len = |s: EStrSegment, len: usize| {
            let mut rope = RopeBase::default();
            assert_eq!(len, s.len());
            rope.init(Some(s).into_iter());
            assert_eq!(len, rope.base_len());
        };

        for (segment, _, chars) in simple_test_cases() {
            assert_len(segment, chars);
        }
    }

    #[test]
    fn base_metric_single() {
        for (segment, bytes, chars) in simple_test_cases() {
            let mut metrics = Rope::default();
            metrics.insert(0, segment.clone());
            assert_eq!(chars, metrics.base_len());
            assert_eq!(bytes, metrics.len::<PangoMetric>());
            assert_eq!(chars, metrics.len::<BaseMetric>());
        }
    }

    #[test]
    fn metrics_multiple() {
        let segments = simple_test_cases();

        let mut indices = [0usize, 0, 0];
        let inc = |indices: &mut [usize]| -> bool {
            for i in indices.iter_mut() {
                if *i < segments.len() - 1 {
                    *i += 1;
                    return true;
                }
                *i = 0;
            }
            false
        };

        loop {
            let mut bytes = 0;
            let mut chars = 0;
            let mut metrics = Rope::default();
            indices.iter().map(|i| &segments[*i]).for_each(
                |(segment, delta_bytes, delta_chars)| {
                    bytes += delta_bytes;
                    chars += delta_chars;
                    metrics.insert(metrics.base_len(), segment.clone());
                });
            assert_eq!(chars, metrics.base_len());
            assert_eq!(bytes, metrics.len::<PangoMetric>());
            assert_eq!(chars, metrics.len::<BaseMetric>());
            if !inc(&mut indices) {
                break;
            }
        }
    }
}
