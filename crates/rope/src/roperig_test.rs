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
pub struct Alphabet(pub String);
impl From<&str> for Alphabet {
    fn from(s: &str) -> Self {
        Alphabet(s.to_string())
    }
}
impl From<String> for Alphabet {
    fn from(s: String) -> Self {
        Alphabet(s)
    }
}
impl From<char> for Alphabet {
    fn from(c: char) -> Self {
        Alphabet(c.to_string())
    }
}

impl Alphabet {
    pub fn is_mergeable(&self, other: &Self) -> bool {
        match (self.0.chars().next(), other.0.chars().next()) {
            (None, _) | (_, None) => true,
            (Some(c1), Some(c2)) => c1 == c2,
        }
    }
}

impl Summable for Alphabet {
    type S = usize;
    fn summarize(&self) -> Self::S {
        self.0.len()
    }
}

impl RopePiece for Alphabet {
    type Context = ();
    fn insert_or_split(&mut self, _: &mut (), other: Insertion<Self>, offset: usize) -> SplitResult<Self> {
        let other = other.0;
        if self.0.is_empty() {
            *self = other;
            SplitResult::Merged
        } else if other.0.is_empty() {
            SplitResult::Merged
        } else if offset == 0 {
            if other.0.as_bytes()[0] == self.0.as_bytes()[0] {
                self.0.push_str(&other.0);
                SplitResult::Merged
            } else {
                SplitResult::HeadSplit(other)
            }
        } else if offset == self.0.len() {
            if other.0.as_bytes()[0] == self.0.as_bytes()[0] {
                self.0.push_str(&other.0);
                SplitResult::Merged
            } else {
                SplitResult::TailSplit(other)
            }
        } else if other.0.as_bytes()[0] == self.0.as_bytes()[0] {
            self.0.push_str(&other.0);
            SplitResult::Merged
        } else {
            let tail = self.0.split_off(offset);
            SplitResult::MiddleSplit(other, Alphabet(tail))
        }
    }
    fn delete_range(&mut self, _: &mut (), from: usize, to: usize) -> DeleteResult<Alphabet> {
        DeleteResult::Updated(to - from)
    }
    fn delete(&mut self, _: &mut ()) {
    }
}
