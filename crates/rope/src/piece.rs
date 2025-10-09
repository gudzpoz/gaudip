use crate::rb_base::Ref;

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
    /// Methods of the same [RopePiece] instance will always receive
    /// the same context object, as is returned by [crate::roperig::Rope::context].
    ///
    /// [piece tree]: https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation
    type Context: RopeContext<Self>;

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
pub struct TransientRef(Ref);

#[allow(clippy::from_over_into)]
impl Into<usize> for TransientRef {
    fn into(self) -> usize {
        self.0.0
    }
}

/// Context object for a rope
///
/// Usually used by ropes to store external data, if any.
///
/// The main purpose of the [RopeContext::on_insertion] and
/// [RopeContext::on_deletion] APIs is to allow safe references to
/// certain nodes in the tree. This might be useful to markers.
pub trait RopeContext<R: RopePiece>: Default {
    /// Called when a new node is inserted into the tree
    fn on_insertion(&mut self, _piece: &mut R, _node: TransientRef) {
    }
    /// Called when a node is deleted from the tree
    fn on_deletion(&mut self, _piece: R, _node: TransientRef) {
    }
}
