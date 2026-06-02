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

//! Multi-controlled gate decomposition.
//!
//! This module is reserved for decomposition of
//! [`Instruction::McGate`](crate::circuit::Instruction::McGate) operations.
//! The lowering implementation has not yet been migrated into the current
//! compiler tree, so this module does not expose an active decomposition entry
//! point yet.

pub mod mcx;
pub mod pauli;

#[cfg(test)]
mod pauli_test;
