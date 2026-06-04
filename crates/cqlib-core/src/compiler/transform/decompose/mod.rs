// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Circuit decomposition building blocks.
//!
//! This module separates decomposition by the representation that is available
//! for an operation:
//!
//! - [`definition`] expands gates that already carry an implementation circuit.
//!   Use this first for [`CircuitGate`](crate::circuit::gate::CircuitGate) and
//!   circuit-backed [`UnitaryGate`](crate::circuit::UnitaryGate) operations.
//! - [`unitary`] synthesizes remaining custom `UnitaryGate` operations from
//!   fixed numeric matrices. It currently supports one- and two-qubit matrices.
//! - [`mc_gate`] provides explicit algorithmic synthesis primitives for
//!   multi-controlled gates, plus the circuit-level resource-aware
//!   `decompose_mc_gates` entry point.
//!
//! These are independent decomposition entry points rather than a complete
//! compiler pipeline. Target-basis lowering, layout, routing, and scheduling
//! belong to their respective compiler stages. A caller that needs both
//! definition expansion and matrix synthesis should run [`expand_definitions`]
//! before [`unitary::unitary::decompose_unitaries`], so circuit-backed unitary
//! gates are expanded before the matrix-only synthesis stage is reached.

pub mod definition;
pub mod mc_gate;
pub mod unitary;

pub use definition::expand_definitions;
