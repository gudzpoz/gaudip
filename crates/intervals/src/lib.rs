//! This crate contains a interval tree based relative offsets.
#![doc = include_str!("../README.md")]

#![warn(missing_docs)]

/// Contains a basic red-black tree implementation based on slab.
pub mod tree;

mod interval;
pub use interval::{Interval, Metadata};
