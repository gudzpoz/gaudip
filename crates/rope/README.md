# `roperig`

This crate uses code from [sevagh/red-black-tree] and is based on [piece trees].

[sevagh/red-black-tree]: https://github.com/sevagh/red-black-tree

[piece trees]: https://code.visualstudio.com/blogs/2018/03/23/text-buffer-reimplementation

If you are looking for rope-like structures for text representation, you might
instead want to check out the above crates or `any-rope`, `crop`, `jumprope`,
`ropey`, (or that [`sum_tree`] used internally by Zed), or other well tested
rope implementations. There are quite a few on crates.io. This crate is not
well-optimized, and might not be suitable for text representation.

[`sum_tree`]: https://zed.dev/blog/zed-decoded-rope-sumtree

## Why?

[Rope] is a fine structure for text, in that:

- It keeps decent worst case performance of insertion / deletion operations.
- It can keep stats along with the text, making random access by line numbers,
  Unicode chars, or any other non-byte indices quite efficient.

And it goes beyond that: by always keeping relative position data, rope-like
structures can also serve as a nice representation of things associated with
strings. Normal interval trees or tree maps usually uses absolute key values and
there is no way to work around that. It means that, statistically, you will need
to shift the coordinates of *half of all the metadata objects every single time*
you insert/delete texts. Rope-like structures reduce metadata keeping cost from
`O(n)` to `O(log(n))` and can be suitable for:

- Intervals (syntax highlighting, for example)
- Markers (or cursors, if you think about multi-cursor editing)
- Or maybe anything?

However, most of the crates listed above hardcoded Rust strings as the
underlying data type; while `xi-rope` allows user-implemented leaves, it assumes
that all nodes are mergeable, which isn't the case for metadata. (I didn't look
into `sum_tree`, but it seems to me (at a glance) that it does not merge nodes,
which is fine for strings but bad for opaque nodes (see below).)

This crate ultimately aims to serve as the data structure behind the *virtual
text representation* of Juicemacs clients. It should support:

- Virtual nodes: Stats-only lines. Lazily loaded from the server when they are
  scrolled into view.
- Opaque nodes: Externally managed lines. Unlike strings, they are not supposed
  to be split (because, for example, most text layout engines expect a whole
  line) and insertion should be handled by the user.

It should be possible to implement actual string ropes (and piece tables/trees
too!) with this crate (via an externally supplied context object).

[Rope]: https://en.wikipedia.org/wiki/Rope_(data_structure)
