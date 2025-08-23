/// Summary for a [Summable]
///
/// For string ropes, this will typically include lengths,
/// line counts, etc.
///
/// It is assumed that this sum has the following properties:
/// - Associativity
/// - Commutativity
/// - Has an identity element
/// - Has inverse elements ([Sum::add_assign] versus [Sum::sub_assign])
// We allow zero-length pieces, and an is_empty function
// can be confusing at least.
#[allow(clippy::len_without_is_empty)]
pub trait Sum: Sized + Eq + PartialEq + Copy + Clone {
    /// Length in base units (see [crate::metrics::Metric])
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

    /// Returns the [identity element] of this sum, typically zero(es)
    ///
    /// [identity element]: https://en.wikipedia.org/wiki/Identity_element
    fn identity() -> Self;
    /// Returns the [inverse element] of the current sum
    ///
    /// [inverse element]: https://en.wikipedia.org/wiki/Inverse_element
    fn negate(&self) -> Self {
        let mut zero = Self::identity();
        zero.sub_assign(self);
        zero
    }
}

/// A type with some length properties
pub trait Summable {
    /// The [Sum] summary type, typically length(s)
    type S: Sum;
    /// Returns a [Sum] containing info to the current object
    fn summarize(&self) -> Self::S;
}

/// Holds the result of [RopePiece::insert_or_split]
pub enum SplitResult<T> {
    /// Meaning: the supplied piece has been merged into this node
    Merged,
    /// Meaning: The returned value is the unmerged portion and should
    /// be inserted in front of this node.
    HeadSplit(T),
    /// Similar to [SplitResult::HeadSplit], but returned only when
    /// the insertion should happen at the tail end.
    TailSplit(T),
    /// Meaning: The unmerged portion causes the node to split.
    /// As a result, the current node should be split in place,
    /// returning the result as `MiddleSplit(unmerged_portion, split_tail)`.
    MiddleSplit(T, T),
}

/// A piece to be stored in a rope node
pub trait RopePiece: Summable + Sized {
    /// Context object for rope pieces
    ///
    /// For example, for [piece tree] implementations, this can be a type
    /// holder piece buffers. This will be passed as a context argument
    /// when calling functions of this trait.
    ///
    /// [piece tree]: https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation
    type Context: Default;

    /// Precondition to [Self::insert_or_split] when deleting nodes
    /// 
    /// If the user wants to exert certain invariants like "some adjacent nodes
    /// must get merged", they should return `true` in this function.
    fn must_try_merging(&self, context: &mut Self::Context, other: &Self) -> bool;
    /// Try to insert a new piece into the current piece
    ///
    /// ## For Implementers
    ///
    /// When the piece gets merged, the method should return [SplitResult::Merged].
    ///
    /// ### Zero-width pieces with affinity
    ///
    /// The current implementation allows zero-width pieces and won't auto-delete them.
    /// (However, we don't offer a way to ensure their relative position yet.)
    /// Whether these pieces are to be retained are determined by the user:
    /// merge zero-width pieces (returning [SplitResult::Merged]) to remove them.
    ///
    /// If you want zero-width nodes with node affinity, the best you can do is to
    /// pack these nodes into a "collection" piece and handle their affinity by
    /// splitting accordingly.
    fn insert_or_split(&mut self, context: &mut Self::Context, other: Self, offset: usize) -> SplitResult<Self>;
    /// Delete a portion from the current piece and return [Sum] of the deleted part
    ///
    /// If the current node should be deleted as a whole, return `None`.
    fn delete_range(&mut self, context: &mut Self::Context, from: usize, to: usize) -> Option<Self::S>;
}
