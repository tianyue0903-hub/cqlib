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

//! # Quantum Circuit Module
//!
//! This module defines the core data structures for quantum circuit representation
//! and manipulation. It provides the [`Circuit`] struct as the primary container
//! for quantum programs, along with supporting types for qubits, gates, operations,
//! and parameters.
//!
//! ## Key Components
//!
//! - [`Circuit`]: Main quantum circuit container
//! - [`Qubit`]: Quantum bit identifier
//! - [`Operation`]: Individual circuit operation
//! - [`StandardGate`]: Standard quantum gate definitions
//! - [`Parameter`]: Symbolic parameter support for variational circuits

pub mod ansatz;
pub mod bit;
pub mod cfg;
pub mod circuit_impl;
pub mod circuit_param;
mod circuit_to_matrix;
pub mod error;
pub mod gate;
pub mod operation;
pub mod parameter;
pub mod symbolic_matrix;

pub use bit::Qubit;
pub use cfg::CircuitCFG;
pub use circuit_impl::Circuit;
pub use circuit_param::{CircuitParam, ParameterValue};
pub use circuit_to_matrix::circuit_to_matrix;
pub use error::CircuitError;
pub use gate::circuit_gate::CircuitGate;
pub use gate::control_flow::{ConditionView, ControlFlow, IfElseGate, WhileLoopGate};
pub use gate::directive::Directive;
pub use gate::instruction::Instruction;
pub use gate::mc_gate::MCGate;
pub use gate::standard_gate::StandardGate;
pub use gate::unitary_gate::UnitaryGate;
pub use operation::{Operation, ValueOperation};
pub use parameter::Parameter;
