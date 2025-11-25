use crate::piece::{DeleteResult, RopePiece, SplitResult, Sum, Summable};
use crate::roperig::Rope;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::ops::Range;

impl Sum for usize {
    fn len(&self) -> usize {
        *self
    }
    fn add_assign(&mut self, other: &Self) {
        *self = self.wrapping_add(*other);
    }
    fn sub_assign(&mut self, other: &Self) {
        *self = self.wrapping_sub(*other);
    }
    fn identity() -> Self {
        0
    }
}

/// Test rope with unmergeable nodes
///
/// Each piece is a string containing a letter repeating,
/// and pieces of different letters are not mergeable.
#[derive(Debug, Eq, PartialEq)]
pub struct Alphabet(pub char, pub usize);
impl From<&str> for Alphabet {
    fn from(s: &str) -> Self {
        assert!(!s.is_empty());
        Alphabet(s.chars().next().unwrap(), s.chars().count())
    }
}
impl From<String> for Alphabet {
    fn from(s: String) -> Self {
        Alphabet::from(s.as_str())
    }
}
impl From<char> for Alphabet {
    fn from(c: char) -> Self {
        let mut arr = [0u8; 8];
        assert!(c.len_utf8() < arr.len());
        c.encode_utf8(&mut arr);
        Alphabet::from(str::from_utf8(&arr[..c.len_utf8()]).unwrap())
    }
}

impl Alphabet {
    pub fn c(&self) -> Option<char> {
        if self.is_empty() { None } else { Some(self.0) }
    }
    pub fn is_empty(&self) -> bool {
        self.1 == 0
    }
}

impl Summable for Alphabet {
    type S = usize;
    fn summarize(&self) -> Self::S {
        self.1
    }
}

impl RopePiece for Alphabet {
    type Context = ();
    fn insert_or_split(&mut self, _: &mut (), other: Self, offset: usize) -> SplitResult<Self> {
        assert!(!self.is_empty());
        assert!(!other.is_empty());
        if offset == self.1 {
            if other.c() == self.c() {
                self.1 += &other.1;
                SplitResult::Merged
            } else {
                SplitResult::TailSplit(other)
            }
        } else if other.c() == self.c() {
            self.1 += &other.1;
            SplitResult::Merged
        } else {
            let tail = self.1 - offset;
            self.1 = offset;
            SplitResult::MiddleSplit(other, Alphabet(self.c().unwrap(), tail))
        }
    }
    fn delete_range(&mut self, _: &mut (), range: Range<usize>) -> DeleteResult<Alphabet> {
        self.1 -= range.len();
        DeleteResult::Updated(range.len())
    }
    fn notify_delete(&mut self, _: &mut ()) {
    }
}

impl Rope<Alphabet> {
    /// Utility function for testing: to string
    pub fn substring(&self, start: usize, end: usize) -> String {
        let mut s = String::with_capacity(end - start);
        let Some(c) = self.cursor_at(start) else { return Default::default() };
        c.for_range(&self.tree, end - start, |piece, range| {
            if !range.is_empty() {
                s.push_str(&piece.c().unwrap().to_string().repeat(range.len()));
            }
            true
        });
        s
    }
}

pub(crate) enum FuzzOp {
    Insert(usize, String),
    Delete(Range<usize>),
}

fn rand_char_pos(rng: &mut ChaCha8Rng, s: &str) -> usize {
    let mut at = rng.random_range(0..=s.len());
    while !s.is_char_boundary(at) {
        at += 1;
    }
    at
}

pub(crate) fn test_simple_string_fuzz<F>(mut f: F, unicode: bool) where F: FnMut(FuzzOp, &str, &mut ChaCha8Rng) {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut expected = String::default();
    let mut random_s = String::default();
    for _ in 0..100000 {
        let insert = expected.is_empty() || rng.random_bool(0.5);
        let op = if insert {
            let at = rand_char_pos(&mut rng, &expected);
            let len = rng.random_range(0..=64);
            random_s.clear();
            (0..len)
                .map(|_| if !unicode || rng.random_bool(0.5) {
                    rng.random_range('a'..='z')
                } else {
                    rng.random_range('一'..='😄')
                }).for_each(|c| random_s.push(c));
            expected.insert_str(at, &random_s);
            FuzzOp::Insert(at, random_s.to_string())
        } else {
            let from = rand_char_pos(&mut rng, &expected);
            let len = rand_char_pos(&mut rng, &expected[from..]);
            expected.drain(from..from+len);
            FuzzOp::Delete(from..from+len)
        };
        f(op, &expected, &mut rng);
    }
}
