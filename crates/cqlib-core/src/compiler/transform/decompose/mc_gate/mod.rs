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
//! This module provides explicit synthesis primitives for lowering
//! [`Instruction::McGate`](crate::circuit::Instruction::McGate) operations.
//! The primitives do not choose an algorithm, allocate ancillary qubits, or
//! rewrite a circuit automatically. Those responsibilities belong to the
//! future multi-controlled-gate decomposition planner.

pub mod mc_su2;
pub mod mcx;
pub mod pauli;
pub mod phase;
pub mod rotation;
pub mod unitary;

#[cfg(test)]
mod pauli_test;
#[cfg(test)]
mod phase_test;
#[cfg(test)]
mod rotation_test;
#[cfg(test)]
mod unitary_test;
