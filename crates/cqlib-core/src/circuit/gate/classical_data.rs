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

use crate::circuit::{ClassicalExpr, ClassicalValue, ClassicalVar};
use std::fmt;

/// Runtime classical data operation.
///
/// These operations are side-effecting classical updates in the circuit
/// schedule. They produce or overwrite runtime classical storage, while
/// [`ClassicalExpr`] remains side-effect-free and only reads runtime classical
/// variables or immutable values.
#[derive(Debug, Clone)]
pub enum ClassicalDataOp {
    /// Stores an expression value into a mutable classical variable.
    Store {
        target: ClassicalVar,
        value: ClassicalExpr,
    },
    /// Measures one qubit and produces an immutable `Bit` value.
    MeasureBit { result: ClassicalValue },
    /// Measures several qubits and produces an immutable `BitVec(width)` value.
    ///
    /// The operation qubit order is the bit-vector order, with qubit index 0
    /// mapped to the least-significant bit.
    MeasureBits { result: ClassicalValue },
}

impl ClassicalDataOp {
    pub fn target(&self) -> Option<ClassicalVar> {
        match self {
            Self::Store { target, .. } => Some(*target),
            Self::MeasureBit { .. } | Self::MeasureBits { .. } => None,
        }
    }

    pub fn result(&self) -> Option<ClassicalValue> {
        match self {
            Self::Store { .. } => None,
            Self::MeasureBit { result } | Self::MeasureBits { result } => Some(*result),
        }
    }

    pub fn value(&self) -> Option<&ClassicalExpr> {
        match self {
            Self::Store { value, .. } => Some(value),
            Self::MeasureBit { .. } | Self::MeasureBits { .. } => None,
        }
    }
}

impl fmt::Display for ClassicalDataOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store { .. } => write!(f, "store"),
            Self::MeasureBit { .. } => write!(f, "measure_bit"),
            Self::MeasureBits { .. } => write!(f, "measure_bits"),
        }
    }
}
