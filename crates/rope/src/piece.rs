use crate::{metrics::Metric, rb_base::SafeRef};

/// Summary for a [Summable]
///
/// For string ropes, this will typically include lengths,
/// line counts, etc.
///
/// It is assumed that this sum has the following properties:
/// - Associativity: `(len1 + len2) + len3 == len1 + (len2 + len3)`
/// - Commutativity: `len1 + len2 == len2 + len1`
/// - Has an identity element: [Sum::identity]
/// - Has inverse elements ([Sum::add_assign] versus [Sum::sub_assign])
// We allow zero-length pieces, and an is_empty function
// can be confusing at least.
#[allow(clippy::len_without_is_empty)]
pub trait Sum: Sized + Eq + PartialEq + Copy + Clone {
    /// Length in base units (see [Metric])
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

    /// Returns a copy of the added sum
    fn add(&self, other: &Self) -> Self {
        let mut sum = *self;
        sum.add_assign(other);
        sum
    }
}
/// A simple trait to avoid generic brackets
pub trait Measured<T: RopePiece> {
    fn get<M: Metric<T>>(&self) -> usize;
}
impl<T: RopePiece> Measured<T> for T::S {
    fn get<M: Metric<T>>(&self) -> usize {
        M::measure(self)
    }
}

/// A type with some length properties
#[allow(clippy::len_without_is_empty)]
pub trait Summable: Sized {
    /// The [Sum] summary type, typically length(s)
    type S: Sum;
    /// Returns a [Sum] containing info to the current object
    fn summarize(&self) -> Self::S;
    /// Returns [Sum::len] of [Self::summarize]
    fn len(&self) -> usize {
        self.summarize().len()
    }
}

/// Holds the result of [RopePiece::insert_or_split]
pub enum SplitResult<T: RopePiece> {
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
/// Holds the result of [RopePiece::delete_range]
pub enum DeleteResult<T: RopePiece> {
    /// Meaning: the deletion is done in place, with the return value
    /// being the [Sum] of the deleted part.
    Updated(T::S),
    /// Meaning: the deletion resulted in node splitting
    TailSplit {
        /// The [Sum] of the deleted part (excluding the split part)
        deleted: T::S,
        /// The split piece after the deleted part
        split: T,
    },
}

/// A piece to be stored in a rope node
///
/// Note that this is a barebone API that has a few assumptions:
/// - All nodes are mergeable in a sense.
///   - This also means that zero-length nodes almost always get merged/deleted.
/// - Node splits are only a means of optimization.
///
/// If the user wishes for a more flexible API (for, for example, expressing
/// mergeability, zero-width nodes, etc.), they are expected to use the [Cursor]
/// API instead.
pub trait RopePiece: Summable + Sized {
    /// Context object for rope pieces
    ///
    /// For example, for [piece tree] implementations, this can be a type
    /// holder piece buffers. This will be passed as a context argument
    /// when calling functions of this trait.
    ///
    /// Methods of the same [RopePiece] instance will always receive
    /// the same context object, as is returned by [crate::roperig::Rope::context].
    ///
    /// [piece tree]: https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation
    type Context;

    /// Try to insert a new piece into the current piece
    ///
    /// ## For Implementers
    ///
    /// When the piece gets merged, the method should return [SplitResult::Merged].
    fn insert_or_split(
        &mut self, context: &mut Self::Context,
        other: Insertion<Self>, offset: &Self::S,
    ) -> SplitResult<Self>;
    /// Delete a portion from the current piece and return [Sum] of the deleted part
    fn delete_range(
        &mut self, context: &mut Self::Context,
        from: &Self::S, to: &Self::S,
    ) -> DeleteResult<Self>;
    /// Delete the entire piece
    ///
    /// This is mainly for any deallocation logic that should happen in the contexts.
    fn delete(&mut self, context: &mut Self::Context);

    /// Returns the relative metrics at the given offset
    fn measure_offset(&self, context: &Self::Context, base_offset: usize) -> Self::S;
}
/// A piece, typically to be inserted, with its [Sum] precomputed
pub struct Insertion<T: RopePiece>(pub T, pub T::S);
impl<T: RopePiece> From<T> for Insertion<T> {
    fn from(value: T) -> Self {
        let sum = value.summarize();
        Insertion(value, sum)
    }
}

/// A reference to internal nodes, inherently unsafe
///
/// ## Safety
///
/// The user (ref holder) is responsible for keeping track of
/// the lifetime of this reference by either:
/// - Do not modify the rope before dropping the reference.
/// - Make sure to drop the reference when [RopeContext::on_deletion]
///   is called with it.
///
/// Internally, this reference is used as indices, and we trust the user
/// and don't perform (or guarantee) much validation.
///
/// ## Usage
///
/// This is typically used as references to markers. For example, in Emacs,
/// you can obtain references to markers and extract their positions.
///
/// A recommended way to use transient refs is to record external references
/// in [RopeContext::on_insertion] and delete them when [RopeContext::on_deletion]
/// is called.
#[derive(Clone)]
pub struct TransientRef(pub(crate) SafeRef);

#[allow(clippy::from_over_into)]
impl Into<usize> for TransientRef {
    fn into(self) -> usize {
        self.0.get()
    }
}

impl Sum for () {
    fn len(&self) -> usize {
        0
    }

    fn add_assign(&mut self, _other: &Self) {
    }

    fn sub_assign(&mut self, _other: &Self) {
    }

    fn identity() -> Self {
    }
}
