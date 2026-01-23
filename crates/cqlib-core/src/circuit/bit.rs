// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! # QuBit Module
//!
//! This module defines the fundamental addressing units for quantum circuits.
//!
//! ## Architecture Design
//!
//! This library adopts a **Handle Pattern** (also known as Entity-Component style) for qubit management:
//!
//! - **[`Qubit`]** acts as a lightweight, `Copy`-able handle (a unique identifier). It does not hold state
//!   or topological information itself.
//! - State and connectivity are managed by the parent `Circuit` or `DAG`.
//!
//! This design ensures strictly $O(1)$ cloning costs and cache-friendly memory layouts during
//! complex compilation passes.

use std::{fmt, hash::Hash};

/// A lightweight handle representing a unique quantum bit (qubit).
///
/// `Qubit` is the fundamental addressing unit in this library. Unlike physical qubits,
/// this structure serves as a stable reference (index) to a node in the circuit graph or
/// a slot in the simulator's state vector.
///
/// It wraps a `u32` to optimize memory usage (4 bytes) while maintaining compatibility
/// with most underlying vector indexing systems via `[Qubit::index]`.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use cqlib_core::circuit::bit::Qubit;
///
/// // Create a qubit handle pointing to the 5th global qubit
/// let q = Qubit::new(5);
///
/// assert_eq!(q.id(), 5);
/// assert_eq!(q.index(), 5usize);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct Qubit(u32);

impl fmt::Display for Qubit {
    /// Formats the qubit identifier.
    ///
    /// The output format is `Q<id>` (e.g., `Q0`, `Q12`), representing the global unique index.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Q{}", self.0)
    }
}

impl Qubit {
    /// Creates a new `Qubit` handle.
    ///
    /// This is the only way for external users to create a Qubit instance,
    /// as the internal field is private to enforce encapsulation.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the raw underlying identifier as a `u32`.
    ///
    /// Use this when you need the compact 4-byte representation.
    #[inline]
    pub const fn id(&self) -> u32 {
        self.0
    }

    /// Returns the identifier cast to `usize` for vector indexing.
    ///
    /// This is a convenience method to avoid manual `as usize` casting
    /// when accessing `Vec` or slices.
    #[inline]
    pub const fn index(&self) -> usize {
        self.0 as usize
    }
}

#[test]
fn test_qubit_creation_and_display() {
    let q0 = Qubit(0);
    let q1 = Qubit(1);

    assert_eq!(q0.id(), 0);
    assert_eq!(q1.id(), 1);
    assert_ne!(q0, q1);

    assert_eq!(format!("{}", q0), "Q0");
    assert_eq!(format!("{}", q1), "Q1");

    let q0_1 = Qubit(0);
    assert_eq!(q0, q0_1);
}
