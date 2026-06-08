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

//! Runtime classical storage for circuits.
//!
//! This module defines the first layer of the dynamic control-flow model:
//! typed classical storage locations that exist only while a circuit executes.
//! A [`ClassicalVar`] is a circuit-local handle to a mutable runtime location.
//! Measurement and assignment operations may write it, while classical
//! expressions may read its current value.

use std::num::NonZeroU32;

/// Static type of a runtime classical variable.
///
/// `Bit` and `BitVec` are the direct targets of measurement operations.
/// `Bool` is kept distinct from `Bit` so control-flow predicates can require
/// explicit boolean expressions. `UInt` represents an unsigned integer with a
/// fixed bit width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClassicalType {
    /// A single measured or assigned bit.
    Bit,
    /// A logical boolean value.
    Bool,
    /// An unsigned integer with the given non-zero bit width.
    UInt(NonZeroU32),
    /// An ordered bit-vector with the given non-zero bit width.
    BitVec(NonZeroU32),
}

impl ClassicalType {
    /// Creates an unsigned integer type with `width` bits.
    ///
    /// Returns `None` when `width` is zero.
    pub fn uint(width: u32) -> Option<Self> {
        NonZeroU32::new(width).map(Self::UInt)
    }

    /// Creates a bit-vector type with `width` bits.
    ///
    /// Returns `None` when `width` is zero.
    pub fn bit_vec(width: u32) -> Option<Self> {
        NonZeroU32::new(width).map(Self::BitVec)
    }

    /// Returns the number of bits used to represent values of this type.
    pub fn width(self) -> u32 {
        match self {
            Self::Bit | Self::Bool => 1,
            Self::UInt(width) | Self::BitVec(width) => width.get(),
        }
    }

    /// Returns the number of measured bits accepted by this type.
    ///
    /// Only `Bit` and `BitVec` are valid direct measurement targets. `Bool`
    /// and `UInt` require explicit expression-level conversion.
    pub fn measurement_width(self) -> Option<u32> {
        match self {
            Self::Bit => Some(1),
            Self::BitVec(width) => Some(width.get()),
            Self::Bool | Self::UInt(_) => None,
        }
    }
}

/// Circuit-local handle to a mutable runtime classical storage location.
///
/// The handle carries its static type so expression and operation builders can
/// validate uses without maintaining parallel typed ID families. The `id` is
/// meaningful only inside the circuit that allocated it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassicalVar {
    id: u32,
    ty: ClassicalType,
}

impl ClassicalVar {
    /// Creates a new classical variable handle.
    ///
    /// This is crate-private because variables must be allocated by a circuit
    /// once the circuit owns a classical variable table.
    pub(crate) fn new(id: u32, ty: ClassicalType) -> Self {
        Self { id, ty }
    }

    /// Returns the circuit-local variable identifier.
    pub fn id(self) -> u32 {
        self.id
    }

    /// Returns the static type of this variable.
    pub fn ty(self) -> ClassicalType {
        self.ty
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassicalType, ClassicalVar};

    #[test]
    fn type_widths_are_reported() {
        assert_eq!(ClassicalType::Bit.width(), 1);
        assert_eq!(ClassicalType::Bool.width(), 1);
        assert_eq!(ClassicalType::uint(7).unwrap().width(), 7);
        assert_eq!(ClassicalType::bit_vec(3).unwrap().width(), 3);
    }

    #[test]
    fn zero_width_integer_and_bit_vector_are_rejected() {
        assert_eq!(ClassicalType::uint(0), None);
        assert_eq!(ClassicalType::bit_vec(0), None);
    }

    #[test]
    fn measurement_width_accepts_only_bits_and_bit_vectors() {
        assert_eq!(ClassicalType::Bit.measurement_width(), Some(1));
        assert_eq!(
            ClassicalType::bit_vec(5).unwrap().measurement_width(),
            Some(5)
        );
        assert_eq!(ClassicalType::Bool.measurement_width(), None);
        assert_eq!(ClassicalType::uint(5).unwrap().measurement_width(), None);
    }

    #[test]
    fn variables_expose_id_and_type() {
        let var = ClassicalVar::new(12, ClassicalType::bit_vec(4).unwrap());

        assert_eq!(var.id(), 12);
        assert_eq!(var.ty(), ClassicalType::bit_vec(4).unwrap());
    }

    #[test]
    fn variable_identity_includes_id_and_type() {
        let bit = ClassicalVar::new(1, ClassicalType::Bit);
        let same_bit = ClassicalVar::new(1, ClassicalType::Bit);
        let bool_var = ClassicalVar::new(1, ClassicalType::Bool);
        let other_bit = ClassicalVar::new(2, ClassicalType::Bit);

        assert_eq!(bit, same_bit);
        assert_ne!(bit, bool_var);
        assert_ne!(bit, other_bit);
    }
}
