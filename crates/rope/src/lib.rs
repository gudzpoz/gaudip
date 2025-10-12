//! This crate contains a rope implementation based on red-black trees
//! and piece trees.
//!
//! To use it, implement the [piece::RopePiece] trait and use [roperig::Rope].

#![warn(missing_docs)]

/// Traits for using and converting between different metric systems
pub mod metrics;
/// Contains traits that the user needs to implement to use ropes.
pub mod piece;
/// The rope implementation.
pub mod roperig;

/// Contains a [String] rope implementation based on [roperig].
#[cfg(feature = "rope")]
pub mod rope;

/// Contains a basic red-black tree implementation based on slab.
mod rb_base;

#[cfg(test)]
mod roperig_test;
