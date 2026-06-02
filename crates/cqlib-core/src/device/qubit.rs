// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Strongly typed qubit identifiers for device-facing APIs.
//!
//! Circuit operations use [`Qubit`] as logical wire identifiers. Device-facing
//! code must distinguish those logical identifiers from physical hardware
//! positions. [`LogicalQubit`] and [`PhysicalQubit`] provide that distinction
//! without changing the compact representation.

use crate::circuit::Qubit;
use std::fmt;

/// Logical qubit identifier used when crossing into device-facing code.
///
/// A logical qubit identifies a circuit wire. It is distinct from a
/// [`PhysicalQubit`], even when both carry the same numeric identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct LogicalQubit(Qubit);

impl LogicalQubit {
    /// Creates a logical qubit identifier from its numeric ID.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(Qubit::new(id))
    }

    /// Wraps an existing circuit qubit as a logical qubit identifier.
    #[inline]
    pub const fn from_qubit(qubit: Qubit) -> Self {
        Self(qubit)
    }

    /// Returns the underlying circuit qubit identifier.
    #[inline]
    pub const fn qubit(self) -> Qubit {
        self.0
    }

    /// Returns the numeric qubit identifier.
    #[inline]
    pub const fn id(self) -> u32 {
        self.0.id()
    }
}

impl From<Qubit> for LogicalQubit {
    fn from(qubit: Qubit) -> Self {
        Self::from_qubit(qubit)
    }
}

impl From<LogicalQubit> for Qubit {
    fn from(qubit: LogicalQubit) -> Self {
        qubit.qubit()
    }
}

impl fmt::Display for LogicalQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.id())
    }
}

/// Physical qubit identifier representing a hardware position on a device.
///
/// A physical qubit is not a circuit wire. Layout code is responsible for
/// mapping [`LogicalQubit`] values to physical qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[repr(transparent)]
pub struct PhysicalQubit(Qubit);

impl PhysicalQubit {
    /// Creates a physical qubit identifier from its numeric ID.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(Qubit::new(id))
    }

    /// Wraps an existing qubit-shaped identifier as a physical hardware ID.
    #[inline]
    pub const fn from_qubit(qubit: Qubit) -> Self {
        Self(qubit)
    }

    /// Returns the underlying qubit-shaped identifier.
    #[inline]
    pub const fn qubit(self) -> Qubit {
        self.0
    }

    /// Returns the numeric hardware-qubit identifier.
    #[inline]
    pub const fn id(self) -> u32 {
        self.0.id()
    }
}

impl From<Qubit> for PhysicalQubit {
    fn from(qubit: Qubit) -> Self {
        Self::from_qubit(qubit)
    }
}

impl From<PhysicalQubit> for Qubit {
    fn from(qubit: PhysicalQubit) -> Self {
        qubit.qubit()
    }
}

impl fmt::Display for PhysicalQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}", self.id())
    }
}
