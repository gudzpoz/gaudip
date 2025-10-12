use crate::piece::{DeleteResult, Insertion, RopePiece, SplitResult, Sum, Summable};

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
        if s.is_empty() {
            return Alphabet(' ', 0)
        }
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
    pub fn is_mergeable(&self, other: &Self) -> bool {
        self.is_empty() || other.is_empty() || self.c() == other.c()
    }
    pub fn c(&self) -> Option<char> {
        if self.is_empty() { None } else { Some(self.0) }
    }
    pub fn len(&self) -> usize {
        self.1
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
    fn insert_or_split(&mut self, _: &mut (), other: Insertion<Self>, offset: usize) -> SplitResult<Self> {
        let other = other.0;
        if self.is_empty() {
            *self = other;
            SplitResult::Merged
        } else if other.is_empty() {
            SplitResult::Merged
        } else if offset == 0 {
            if other.c() == self.c() {
                self.1 += &other.1;
                SplitResult::Merged
            } else {
                SplitResult::HeadSplit(other)
            }
        } else if offset == self.1 {
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
    fn delete_range(&mut self, _: &mut (), from: usize, to: usize) -> DeleteResult<Alphabet> {
        self.1 -= to - from;
        DeleteResult::Updated(to - from)
    }
    fn delete(&mut self, _: &mut ()) {
    }
}
