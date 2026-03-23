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
//! - **[`UnitaryGate`]**: User-defined custom gates via matrix representation.
//! - **[`MCGate`]**: Multi-controlled gates extending standard gates with arbitrary controls.
//! - **[`Directive`]**: Non-unitary circuit operations like `Measure`, `Reset`, and `Barrier`.
//! - **[`ControlFlow`]**: Control flow operations for conditional and iterative quantum execution.
//! - **[`Instruction`]**: The unified sum type that wraps all the above, representing a single step in a circuit.
//! - **[`CircuitGate`]**: Composite gates defined by entire sub-circuits.
//!
//! # Gate Matrix Generation
//!
//! The [`gate_matrix`] module provides low-level functions to generate the unitary matrices
//! for all supported gates.
//!
//! # Examples
//!
//! ## Using Standard Gates
//!
//! ```
//! use cqlib_core::circuit::gate::StandardGate;
//!
//! // Get the Hadamard gate matrix
//! let h_matrix = StandardGate::H.matrix(&[]).unwrap();
//! assert_eq!(h_matrix.shape(), &[2, 2]);
//!
//! // Get the parametric RX gate matrix
//! let rx_matrix = StandardGate::RX.matrix(&[std::f64::consts::PI / 2.0]).unwrap();
//! assert_eq!(rx_matrix.shape(), &[2, 2]);
//!
//! // Check gate properties
//! assert_eq!(StandardGate::CX.num_qubits(), 2);
//! assert_eq!(StandardGate::RX.num_params(), 1);
//! ```
//!
//! ## Creating Multi-Controlled Gates
//!
//! ```
//! use cqlib_core::circuit::gate::{MCGate, StandardGate};
//!
//! // Create a Toffoli gate (CCX)
//! let toffoli = MCGate::new(2, StandardGate::X);
//! assert_eq!(toffoli.num_qubits(), 3);
//! ```
//!
//! ## Working with Instructions
//!
//! ```
//! use cqlib_core::circuit::gate::{Instruction, StandardGate, Directive};
//!
//! // Create instructions from gates
//! let h_inst: Instruction = StandardGate::H.into();
//! let barrier_inst: Instruction = Directive::Barrier.into();
//! ```

pub mod circuit_gate;
pub mod control_flow;
pub mod directive;
pub mod gate_matrix;
pub mod instruction;
pub mod mc_gate;
pub mod standard_gate;
pub mod unitary_gate;

// Re-export key types for easier access
pub use circuit_gate::{CircuitGate, FrozenCircuit};
pub use control_flow::{ConditionView, ControlFlow, IfElseGate, WhileLoopGate};
pub use directive::Directive;
pub use instruction::Instruction;
pub use mc_gate::MCGate;
pub use standard_gate::StandardGate;
pub use unitary_gate::UnitaryGate;
