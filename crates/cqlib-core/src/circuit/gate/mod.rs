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

//! Quantum Gate and Instruction Definitions
//!
//! This module provides the fundamental building blocks for quantum circuits. It defines the
//! instruction set architecture (ISA) of the library, ranging from basic unitary gates to
//! complex controlled operations and non-reversible measurements.
//!
//! # Core Components
//!
//! - **[`StandardGate`]**: The set of natively optimized quantum gates (e.g., `H`, `CX`, `RX`).
//! - **[`ExtendedGate`]**: Extensions to standard gates, supporting arbitrary controls and custom unitary matrices.
//! - **[`Directive`]**: Non-unitary circuit operations like `Measure`, `Reset`, and `Barrier`.
//! - **[`Instruction`]**: The unified sum type that wraps all the above, representing a single step in a circuit.
//!
//! # Gate Matrix Generation
//!
//! The [`gate_matrix`] module provides low-level functions to generate the unitary matrices
//! for all supported gates.

pub mod circuit_gate;
pub mod directive;
pub mod gate_matrix;
pub mod instruction;
pub mod mc_gate;
pub mod standard_gate;
pub mod unitary_gate;

// Re-export key types for easier access
pub use directive::Directive;
pub use instruction::Instruction;
pub use mc_gate::MCGate;
pub use standard_gate::StandardGate;
pub use unitary_gate::UnitaryGate;
