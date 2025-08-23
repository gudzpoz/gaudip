/// Summary for a [Summable]
///
/// For string ropes, this will typically include lengths,
/// line counts, etc.
#[allow(clippy::len_without_is_empty)]
pub trait Sum: Sized + Eq + PartialEq + Copy + Clone {
    /// Length in base units
    fn len(&self) -> usize;

    /// Adds to this sum/summary
    /// 
    /// Note that one should probably use signed numbers or
    /// [usize::wrapping_add] for this, because there will be 
    /// negative deltas when deleting nodes.
    fn add_assign(&mut self, other: &Self);
    /// The inverse of [Self::add_assign]
    /// 
    /// Similarly, usage of signed numbers or wrapping sub is
    /// recommended.
    fn sub_assign(&mut self, other: &Self);
    
    fn identity() -> Self;
    fn negate(&self) -> Self {
        let mut zero = Self::identity();
        zero.sub_assign(&self);
        zero
    }
}

pub trait Summable {
    type S: Sum;
    fn summarize(&self) -> Self::S;
}

pub enum SplitResult<T> {
    Merged,
    HeadSplit(T),
    TailSplit(T),
    MiddleSplit(T, T),
}
pub trait RopePiece: Summable + Sized {
    /// Precondition to [Self::insert_or_split].
    /// 
    /// If the user wants to exert certain invariants like "some adjacent nodes
    /// must get merged", they should return `true` in this function.
    fn must_try_merging(&self, other: &Self) -> bool;
    fn insert_or_split(&mut self, other: Self, offset: usize) -> SplitResult<Self>;
    fn delete_range(&mut self, from: usize, to: usize) -> Self::S;
}
