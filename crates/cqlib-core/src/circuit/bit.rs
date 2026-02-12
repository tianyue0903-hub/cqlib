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

macro_rules! impl_qubit_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Qubit {
                fn from(idx: $t) -> Self {
                    Self(idx as u32)
                }
            }
        )*
    };
}
macro_rules! impl_qubit_from_integer {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Qubit {
                fn from(idx: $t) -> Self {
                    assert!(idx >= 0, "Qubit index must be non-negative");
                    Self(idx as u32)
                }
            }
        )*
    };
}

// Implement From for types that cannot overflow u32
impl_qubit_from_unsigned!(u32, u16, u8);
impl_qubit_from_integer!(i32, i16, i8);

// Implement TryFrom for usize/isize which may overflow on 64-bit systems
impl TryFrom<usize> for Qubit {
    type Error = &'static str;

    fn try_from(idx: usize) -> Result<Self, Self::Error> {
        if idx > u32::MAX as usize {
            Err("Qubit index exceeds u32::MAX")
        } else {
            Ok(Self(idx as u32))
        }
    }
}

impl TryFrom<isize> for Qubit {
    type Error = &'static str;

    fn try_from(idx: isize) -> Result<Self, Self::Error> {
        if idx < 0 {
            Err("Qubit index must be non-negative")
        } else if idx > u32::MAX as isize {
            Err("Qubit index exceeds u32::MAX")
        } else {
            Ok(Self(idx as u32))
        }
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

#[test]
fn test_qubit_try_from_usize() {
    // Valid indices should work
    assert!(Qubit::try_from(0usize).is_ok());
    assert!(Qubit::try_from(100usize).is_ok());
    assert!(Qubit::try_from(u32::MAX as usize).is_ok());

    // Overflow on 64-bit systems
    if usize::BITS > 32 {
        let overflow_idx = (u32::MAX as usize) + 1;
        let result = Qubit::try_from(overflow_idx);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Qubit index exceeds u32::MAX");
    }
}

#[test]
fn test_qubit_try_from_isize() {
    // Valid indices should work
    assert!(Qubit::try_from(0isize).is_ok());
    assert!(Qubit::try_from(100isize).is_ok());
    assert!(Qubit::try_from(u32::MAX as isize).is_ok());

    // Negative index should fail
    let result = Qubit::try_from(-1isize);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Qubit index must be non-negative");

    // Overflow on 64-bit systems
    if isize::BITS > 32 {
        let overflow_idx = (u32::MAX as isize) + 1;
        let result = Qubit::try_from(overflow_idx);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Qubit index exceeds u32::MAX");
    }
}

#[test]
fn test_qubit_from_small_types() {
    // These should all work via From (infallible)
    let _: Qubit = 0u8.into();
    let _: Qubit = 100u8.into();
    let _: Qubit = 0u16.into();
    let _: Qubit = 1000u16.into();
    let _: Qubit = 0u32.into();
    let _: Qubit = 100000u32.into();

    let _: Qubit = 0i8.into();
    let _: Qubit = 100i8.into();
    let _: Qubit = 0i16.into();
    let _: Qubit = 1000i16.into();
    let _: Qubit = 0i32.into();
    let _: Qubit = 100000i32.into();
}
