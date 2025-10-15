use std::io::{Cursor, Write};
use std::mem::replace;
use roperig::metrics::WithCharMetric;
use roperig::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use EStrSegment::*;
use crate::data::segment::metrics::EStrMetrics;

/// A segment of Emacs string
///
/// See [EStrMetrics] for meaning of `len` and `bytes`.
#[derive(Default, Clone, Debug)]
pub enum EStrSegment {
    #[default]
    Empty,
    /// `len` and `bytes`: as is defined by UTF-8
    Unicode { str: String, len: usize },
    /// `len` and `bytes`: the same unit
    Ascii { str: String },
    /// `len`: char counts, `bytes`: one byte for ASCII, 4 bytes for `0x80` and above (`\XXX`)
    RawBytes { str: Vec<u8>, bytes: usize },
    /// `len`: char counts, `bytes`: a single byte (rendered with custom renderer)
    NonUnicode { str: Vec<u32> },
    Widget { width: usize, height: usize },
}
impl From<&str> for EStrSegment {
    fn from(value: &str) -> Self {
        Self::from_ascii_or(value.to_string(), value.len())
    }
}

const MIN_CHILD_LEN: usize = 32;
const MAX_CHILD_LEN: usize = 64;

macro_rules! from_utf32_impl {
    ($name:ident, $stride_shift:expr, $ptr_type:ty) => {
        pub fn $name(str: &[u8]) -> EStrSegment {
            let stride = 1 << $stride_shift;
            debug_assert!(str.len() % stride == 0);
            let mut s = String::with_capacity(str.len() >> $stride_shift);
            let base = str.as_ptr();
            for i in (0..str.len()).step_by(stride) {
                let c = unsafe { (base.add(i) as *const $ptr_type).read_unaligned() };
                s.push(unsafe { char::from_u32_unchecked(c as u32) });
            }
            Self::from_ascii_or(s, str.len() >> $stride_shift)
        }
    };
}

impl EStrSegment {
    pub fn from_ascii(str: &[u8]) -> EStrSegment {
        Ascii {
            str: unsafe { String::from_utf8_unchecked(str.to_owned()) }
        }
    }
    pub fn from_raw(bytes: &[u8]) -> EStrSegment {
        let len = raw_bytes_pango_bytes(bytes);
        if len == bytes.len() {
            Ascii { str: unsafe { String::from_utf8_unchecked(bytes.to_vec()) } }
        } else {
            RawBytes{ str: bytes.to_vec(), bytes: len }
        }
    }

    from_utf32_impl!(from_utf32_stride1, 0, u8);
    from_utf32_impl!(from_utf32_stride2, 1, u16);
    from_utf32_impl!(from_utf32_stride4, 2, u32);

    fn from_ascii_or(str: String, len: usize) -> EStrSegment {
        if str.len() == len {
            Ascii { str }
        } else {
            Unicode { str, len }
        }
    }
    pub fn from_emacs(str: &[u8]) -> Vec<EStrSegment> {
        struct Builder {
            vec: Vec<EStrSegment>,
            state: EStrSegment,
        }
        impl Builder {
            fn push(&mut self, c: u32) {
                match (&mut self.state, char::from_u32(c)) {
                    (Ascii { str }, Some(uni_c))
                    if uni_c.is_ascii() => {
                        str.push(uni_c)
                    }
                    (Unicode { str, ref mut len }, Some(uni_c))
                    if str.len() < MAX_CHILD_LEN => {
                        str.push(uni_c);
                        *len += 1;
                    }
                    (NonUnicode { str }, None) if c < 0x3FFF80 => {
                        str.push(c);
                    }
                    (RawBytes { str, ref mut bytes }, Some(uni_c))
                    if uni_c.is_ascii() && str.len() < MAX_CHILD_LEN => {
                        str.push(uni_c as u8);
                        *bytes += 1;
                    }
                    (RawBytes { str, ref mut bytes }, None)
                    if c >= 0x3FFF80 => {
                        str.push((c - 0x3FFF00) as u8);
                        *bytes += 4;
                    }
                    (_, Some(uni_c)) if uni_c.is_ascii() =>
                        self.commit(Ascii { str: String::from(uni_c) }),
                    (_, Some(uni_c)) =>
                        self.commit(Unicode { str: String::from(uni_c), len: 1 }),
                    (_, _) if c < 0x3FFF80 =>
                        self.commit(NonUnicode { str: vec![c] }),
                    (_, _) => self.commit(
                        RawBytes { str: vec![(c - 0x3FFF00) as u8], bytes: 4 },
                    ),
                };
            }
            fn commit(&mut self, next: EStrSegment) {
                let prev = replace(&mut self.state, next);
                if !prev.is_empty() {
                    self.vec.push(prev);
                }
            }
        }
        let mut builder = Builder{ vec: Vec::default(), state: Default::default() };
        debug_assert!(str.len().is_multiple_of(4));
        let base = str.as_ptr();
        for i in (0..str.len()).step_by(4) {
            let c = unsafe { (base.add(i) as *const u32).read_unaligned() };
            builder.push(c);
        }
        builder.commit(Empty);
        builder.vec
    }

    /// Returns the length info in the form of `(len_chars, len_bytes)`
    pub fn chars_bytes(&self) -> (usize, usize) {
        match self {
            Empty => (0, 0),
            Unicode { str, len } => (*len, str.len()),
            Ascii { str } => (str.len(), str.len()),
            RawBytes { str, bytes } => (str.len(), *bytes),
            NonUnicode { str } => (str.len(), str.len()),
            Widget { .. } => (1, 1),
        }
    }

    pub fn len(&self) -> usize {
        self.chars_bytes().1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn write(&self, bytes: &mut [u8]) {
        debug_assert!(bytes.len() == self.len());
        match self {
            Empty => (),
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
            NonUnicode { .. } => bytes.fill(b' '),
            Widget { .. } => bytes[0] = b' ',
        }
    }

    pub fn split(&mut self, offset: usize) -> Self {
        debug_assert!(offset != 0);
        match self {
            Empty => Empty,
            Unicode { str, len } => {
                let tail = str[offset..].to_string();
                str.drain(offset..);
                let tail_chars = tail.chars().count();
                *len -= tail_chars;
                Self::from_ascii_or(tail, tail_chars)
            }
            Ascii { str } => {
                let tail = str[offset..].to_string();
                str.drain(offset..);
                let len = tail.len();
                Self::from_ascii_or(tail, len)
            }
            RawBytes { str, bytes } => {
                let tail = Self::from_raw(&str[offset..]);
                *bytes -= tail.chars_bytes().0;
                str.drain(offset..);
                tail
            }
            NonUnicode { str } => {
                let tail = NonUnicode { str: str[offset..].to_vec() };
                str.drain(offset..);
                tail
            }
            Widget { .. } => unreachable!(),
        }
    }
}

impl Summable for EStrSegment {
    type S = EStrMetrics;

    fn summarize(&self) -> Self::S {
        let (chars, bytes) =self.chars_bytes();
        EStrMetrics { bytes, chars }
    }
}

impl RopePiece for EStrSegment {
    type Context = ();
    const ABS: bool = false;

    fn insert_or_split(&mut self, _: &mut Self::Context, other: Self, offset: &Self::S) -> SplitResult<Self> {
        if other.is_empty() {
            return SplitResult::Merged;
        }
        if self.is_empty() {
            *self = other;
            return SplitResult::Merged;
        }
        let offset = offset.bytes;
        match (self, &other) {
            (
                Unicode { str, len },
                Unicode { str: str_o, len: len_o },
            ) => {
                if str.len() + str_o.len() > MAX_CHILD_LEN {
                    if offset == 0 {
                        return SplitResult::HeadSplit(other);
                    }
                    if offset == str.len() {
                        return SplitResult::TailSplit(other);
                    }

                    let tail = str[offset..].to_string();
                    let tail_chars = tail.chars().count();
                    let tail = Self::from_ascii_or(tail, tail_chars);
                    str.drain(offset..);
                    return if str_o.len() >= MIN_CHILD_LEN {
                        *len -= tail_chars;
                        SplitResult::MiddleSplit(other, tail)
                    } else {
                        str.push_str(str_o);
                        *len = *len + len_o - tail_chars;
                        SplitResult::TailSplit(tail)
                    };
                }
                str.insert_str(offset, str_o);
                *len += len_o;
                SplitResult::Merged
            }
            (
                Ascii { str },
                Ascii { str: str_o },
            ) => {
                str.insert_str(offset, str_o);
                if str.len() > MAX_CHILD_LEN {
                    let mid = str.len() / 2;
                    let tail = str[mid..].to_string();
                    str.drain(mid..);
                    let len = tail.len();
                    SplitResult::TailSplit(Self::from_ascii_or(tail, len))
                } else {
                    SplitResult::Merged
                }
            }
            (this, _) => {
                if offset == 0 {
                    SplitResult::HeadSplit(other)
                } else if offset == this.len() {
                    SplitResult::TailSplit(other)
                } else {
                    SplitResult::MiddleSplit(other, this.split(offset))
                }
            }
        }
    }

    fn delete_range(&mut self, _: &mut Self::Context, from: &Self::S, to: &Self::S) -> DeleteResult<Self> {
        let range = from.bytes..to.bytes;
        match self {
            Empty => unreachable!(),
            Unicode { str, len } => {
                str.drain(range);
                *len -= to.chars - from.chars;
            }
            Ascii { str } => { str.drain(range); }
            RawBytes { str, bytes } => {
                str.drain(from.chars..to.chars);
                *bytes -= range.len();
            }
            NonUnicode { str } => { str.drain(range); }
            Widget { .. } => unreachable!(),
        }
        let mut delta = *to;
        delta.sub_assign(from);
        DeleteResult::Updated(delta)
    }

    fn delete(&mut self, _context: &mut Self::Context) {
    }

    fn measure_offset(&self, _: &Self::Context, base_offset: usize, _abs: usize) -> Self::S {
        let chars = match self {
            Unicode { str, .. } => str[..base_offset].chars().count(),
            RawBytes { str, .. } => raw_bytes_pango_to_index(str, base_offset).0,
            Empty | Ascii { .. } | NonUnicode { .. } | Widget { .. } => base_offset,
        };
        EStrMetrics { bytes: base_offset, chars }
    }
}

pub(crate) fn raw_bytes_pango_to_index(str: &[u8], byte_index: usize) -> (usize, usize) {
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

fn raw_bytes_pango_bytes(str: &[u8]) -> usize {
    str.iter().map(|c| if c.is_ascii() { 1 } else { 4 }).sum()
}

impl WithCharMetric for EStrSegment {
    fn substring<F, R: Default>(
        &self, _: &Self::Context,
        range: std::ops::Range<usize>, _abs_base: usize,
        mut f: F,
    ) -> R where F: FnMut(&str, R) -> R {
        let r = R::default();
        match self {
            Empty => r,
            Unicode { str, .. } | Ascii { str } => f(&str[range], r),
            NonUnicode { .. } => range.fold(r, |r, _| f("�", r)),
            Widget { .. } => f("￼", r),
            RawBytes { str, .. } => str[range].iter().fold(r, |r, c| {
                if c.is_ascii() {
                    let buf = [*c];
                    f(str::from_utf8(&buf[..]).unwrap(), r)
                } else {
                    f("🔢", r)
                }
            }),
        }
    }

    fn chars(sum: &Self::S) -> usize {
        sum.chars
    }
}

pub mod metrics {
    use roperig::piece::Sum;

    /// Metrics for Emacs string and the corresponding Pango string
    ///
    /// When displaying Emacs strings, we need to convert them to
    /// a UTF-8 string that Pango accepts, while keeping it easy for
    /// modification. And this rope serves exactly this purpose.
    #[derive(Default, Copy, Clone, Eq, PartialEq)]
    pub struct EStrMetrics {
        /// The encoded length in the Pango UTF-8 string
        ///
        /// For example, raw bytes are encoded as `\XXX`,
        /// so each char corresponds to 4 bytes.
        pub(crate) bytes: usize,
        /// The char count in this interval.
        ///
        /// See [str::chars].
        pub(crate) chars: usize,
    }

    impl Sum for EStrMetrics {
        fn len(&self) -> usize {
            self.bytes
        }

        fn add_assign(&mut self, other: &Self) {
            self.bytes = self.bytes.wrapping_add(other.bytes);
            self.chars = self.chars.wrapping_add(other.chars);
        }

        fn sub_assign(&mut self, other: &Self) {
            self.bytes = self.bytes.wrapping_sub(other.bytes);
            self.chars = self.chars.wrapping_sub(other.chars);
        }

        fn identity() -> Self {
            Self::default()
        }
    }

    fn char_boundary_search<const INC: bool>(str: &str, offset: usize) -> usize {
        let mut i = if INC { offset + 1 } else { offset - 1 };
        while !str.is_char_boundary(i) {
            i = if INC { i + 1 } else { i - 1 };
        }
        i
    }

    #[cfg(test)]
    pub(crate) mod tests {
        use roperig::metrics::{BaseMetric, CharMetric};
        use roperig::roperig::Rope;
        use crate::data::segment::EStrSegment;

        macro_rules! utf32_to_bytes_impl {
            ($name:ident, $stride_shift:expr, $ptr_type:ty) => {
                pub fn $name(from: &[$ptr_type]) -> &[u8] {
                    let len = from.len() << $stride_shift;
                    let ptr: *const u8 = from.as_ptr().cast();
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                }
            };
        }
        utf32_to_bytes_impl!(utf32_stride2_to_bytes, 1, u16);
        utf32_to_bytes_impl!(utf32_stride4_to_bytes, 2, u32);

        fn simple_test_cases() -> Vec<(EStrSegment, usize, usize)> {
            let mut vec = Vec::new();
            let test_bytes = vec![1, 63, 128, 2, 64, 255];
            vec.push((EStrSegment::from_ascii("hello!".as_bytes()), 6, 6));
            vec.push((EStrSegment::from_raw("hello!".as_bytes()), 6, 6));
            vec.push((EStrSegment::from_raw(&test_bytes), 12, 6));

            vec.push((EStrSegment::from_utf32_stride1("hello!".as_bytes()), 6, 6));
            vec.push((EStrSegment::from_utf32_stride1(&test_bytes), 8, 6));

            let stride2: Vec<u16> = "hello!".as_bytes().iter().map(|c| *c as u16).collect();
            vec.push((EStrSegment::from_utf32_stride2(utf32_stride2_to_bytes(&stride2)), 6, 6));
            let stride2: Vec<u16> = test_bytes.iter().map(|c| *c as u16).collect();
            vec.push((EStrSegment::from_utf32_stride2(utf32_stride2_to_bytes(&stride2)), 8, 6));
            let stride2 = vec![1, 128, 0xFFF, 2, 255, 0xFFFF];
            vec.push((EStrSegment::from_utf32_stride2(utf32_stride2_to_bytes(&stride2)), 12, 6));

            let stride4: Vec<u32> = "hello!".as_bytes().iter().map(|c| *c as u32).collect();
            vec.push((EStrSegment::from_utf32_stride4(utf32_stride4_to_bytes(&stride4)), 6, 6));
            let stride4: Vec<u32> = test_bytes.iter().map(|c| *c as u32).collect();
            vec.push((EStrSegment::from_utf32_stride4(utf32_stride4_to_bytes(&stride4)), 8, 6));
            let stride4: Vec<u32> = stride2.iter().map(|c| *c as u32).collect();
            vec.push((EStrSegment::from_utf32_stride4(utf32_stride4_to_bytes(&stride4)), 12, 6));
            let stride4 = vec![1, 128, 0xFFF, 0x10000, 2, 255, 0xFFFF, 0x10FFFF];
            vec.push((EStrSegment::from_utf32_stride4(utf32_stride4_to_bytes(&stride4)), 20, 8));

            let non_unicode = vec![0x110000, 0x3FFF00];
            let emacs = EStrSegment::from_emacs(utf32_stride4_to_bytes(&non_unicode));
            assert_eq!(1, emacs.len());
            vec.push((emacs[0].clone(), 2, 2));
            let raw_bytes = vec![0x3FFF80, 0x3FFFFF];
            let emacs = EStrSegment::from_emacs(utf32_stride4_to_bytes(&raw_bytes));
            assert_eq!(1, emacs.len());
            vec.push((emacs[0].clone(), 8, 2));

            for (_, bytes, chars) in &vec {
                assert_eq!(0, bytes % 2);
                assert_eq!(0, chars % 2);
            }
            vec
        }

        #[test]
        fn length_single() {
            let assert_len = |s: EStrSegment, len: usize| {
                let mut rope = Rope::default();
                assert_eq!(len, s.len());
                rope.insert(0, s);
                assert_eq!(len, rope.len());
            };

            for (segment, bytes, _) in simple_test_cases() {
                assert_len(segment, bytes);
            }
        }

        fn test_metrics(rope: &Rope<EStrSegment>, chars: usize) {
            for index in 0..=chars {
                assert_eq!(Some(index), rope.convert_metrics::<CharMetric, CharMetric>(index));
                let byte_index = rope.convert_metrics::<CharMetric, BaseMetric>(index);
                assert_eq!(byte_index, rope.convert_metrics::<BaseMetric, BaseMetric>(byte_index.unwrap()));
                let char_index = rope.convert_metrics::<BaseMetric, CharMetric>(byte_index.unwrap());
                assert_eq!(Some(index), char_index);
            }
        }

        type ByteMetric = BaseMetric;

        #[test]
        fn base_metric_single() {
            for (segment, bytes, chars) in simple_test_cases() {
                let mut metrics = Rope::default();
                metrics.insert(0, segment.clone());
                assert_eq!(bytes, metrics.len());
                assert_eq!(bytes, metrics.measure::<ByteMetric>());
                assert_eq!(chars, metrics.measure::<CharMetric>());
                assert_eq!(
                    Some(bytes / 2), metrics.convert_metrics::<CharMetric, ByteMetric>(chars / 2),
                    "{:?}@(char){}", segment, chars / 2,
                );
                assert_eq!(
                    Some(chars / 2), metrics.convert_metrics::<ByteMetric, CharMetric>(bytes / 2),
                    "{:?}@(byte){}", segment, bytes / 2,
                );
                test_metrics(&metrics, chars);
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
                        metrics.insert(metrics.len(), segment.clone());
                    });
                assert_eq!(bytes, metrics.len());
                assert_eq!(bytes, metrics.measure::<ByteMetric>());
                assert_eq!(chars, metrics.measure::<CharMetric>());
                test_metrics(&metrics, chars);
                if !inc(&mut indices) {
                    break;
                }
            }
        }
    }
}
