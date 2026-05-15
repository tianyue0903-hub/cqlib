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

//! Multi-controlled standard-gate decomposition.
//!
//! The module owns decomposition for [`MCGate`] instructions whose base gate can
//! be lowered without a general controlled-unitary synthesizer. The supported
//! families are:
//!
//! - MCX-like Pauli gates (`X`, `Y`, `Z`, `CX`, `CY`, `CZ`, `CCX`),
//!   implemented by reusing MCX decompositions plus target-basis conjugation.
//! - Diagonal phase and rotation gates (`S`, `SDG`, `T`, `TDG`, `Phase`,
//!   `RZ`, `RX`, `RY`, `CRZ`, `CRX`, `CRY`), implemented either by no-ancilla
//!   parity phases, by an explicit clean flag, or by target-basis changes.
//! - Controlled `SWAP`, implemented as three controlled-`CX` style MCX
//!   decompositions.
//!
//! `decompose_mc_gate` is the single internal entry point. Ancilla availability
//! is passed as data so callers do not need separate facade functions for each
//! decomposition strategy.
//!
//! [`MCGate`]: crate::circuit::MCGate

pub(super) mod decompose;
pub(super) mod transformer;

#[cfg(test)]
mod test_utils;

#[cfg(test)]
#[path = "./decompose_test.rs"]
mod decompose_test;

mod pauli;

#[cfg(test)]
#[path = "./pauli_test.rs"]
mod pauli_test;

mod phase;
mod phase_ops;

#[cfg(test)]
#[path = "./phase_test.rs"]
mod phase_test;

mod rz;

#[cfg(test)]
#[path = "./rz_test.rs"]
mod rz_test;

mod rx_ry;

#[cfg(test)]
#[path = "./rx_ry_test.rs"]
mod rx_ry_test;

mod one_qubit;

#[cfg(test)]
#[path = "./one_qubit_test.rs"]
mod one_qubit_test;

mod pauli_interaction;

#[cfg(test)]
#[path = "./pauli_interaction_test.rs"]
mod pauli_interaction_test;

mod fsim;

#[cfg(test)]
#[path = "./fsim_test.rs"]
mod fsim_test;

mod swap;

#[cfg(test)]
#[path = "./swap_test.rs"]
mod swap_test;
