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

//! # Circuit Operation Module
//!
//! This module defines the [`Operation`] struct, which represents a single, concrete execution step
//! within a quantum circuit.
//!
//! ## Role in Architecture
//!
//! While [`Instruction`] defines *what* operation to perform (e.g., "apply a Hadamard gate"),
//! [`Operation`] defines *where* and *how* to apply it. It binds an abstract instruction to:
//! - Specific qubits (Topology).
//! - Specific parameters (Context).
//!
//! ## Memory Optimization
//!
//! Since a circuit may contain millions of operations, this struct is heavily optimized for memory compactness:
//! - **SmallVec**: Uses `SmallVec` for qubits and parameters to store data inline on the stack for common cases
//!   (e.g., 1-2 qubit gates, 0-1 parameters), avoiding heap allocation overhead.
//! - **CircuitParam**: Uses a lightweight enum to store parameters, supporting both immediate float values
//!   and references to the circuit's global parameter table (interning).

use crate::circuit::bit::Qubit;
use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::instruction::Instruction;
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};

/// A fully resolved operation in a quantum circuit.
///
/// An `Operation` combines a gate (instruction) with the specific qubits it acts upon and its
/// parameters. It serves as the fundamental node in the circuit's execution list.
///
/// # Fields
///
/// * `instruction` - The type of operation (e.g., `StandardGate::H`, `Directive::Measure`).
/// * `qubits` - The ordered list of qubits involved in this operation.
///   - For controlled gates, control qubits usually come first, followed by target qubits.
///   - Implementation uses `SmallVec<[Qubit; 3]>` to optimize for gates acting on ≤3 qubits (covering almost all standard gates).
/// * `params` - The parameters for the operation.
///   - Implementation uses `SmallVec<[CircuitParam; 1]>` to optimize for single-parameter gates (e.g., `RX`, `RZ`).
/// * `label` - An optional human-readable label or tag for this specific operation instance.
#[derive(Debug, Clone)]
pub struct Operation {
    /// The abstract instruction definition (what to do).
    pub instruction: Instruction,
    /// The specific qubits this operation applies to (where to do it).
    pub qubits: SmallVec<[Qubit; 3]>,
    /// The concrete or symbolic parameters for this operation (how to do it).
    pub params: SmallVec<[CircuitParam; 1]>,
    /// Optional metadata label.
    pub label: Option<Box<str>>,
}

impl Operation {
    /// Computes the numerical unitary matrix for this specific operation.
    ///
    /// This method resolves any parameters associated with the operation and delegates
    /// the matrix generation to the underlying [`Instruction`].
    ///
    /// # Returns
    ///
    /// * `Ok(Cow<Array2>)` - The unitary matrix. It may be borrowed (static) or owned (computed).
    /// * `Err(CircuitError)` - If the operation is non-unitary (e.g., Measure, Barrier) or if
    ///   symbolic parameters cannot be resolved (implementation pending).
    ///
    /// # Current Limitations
    ///
    /// **Partial Implementation**: Currently, this method only supports `CircuitParam::Fixed` (concrete float values).
    /// Attempting to call this on an operation with symbolic parameters (`CircuitParam::Index`) will
    /// result in a panic ("not yet implemented").
    pub fn matrix(&self) -> Result<Cow<'_, Array2<Complex64>>, CircuitError> {
        let mut ps: SmallVec<[f64; 4]> = smallvec![];
        for p in self.params.iter() {
            match p {
                CircuitParam::Fixed(val) => {
                    ps.push(*val);
                }
                CircuitParam::Index(_index) => {
                    return Err(CircuitError::SymbolicParameterError);
                }
            }
        }
        self.instruction
            .matrix(&ps)
            .ok_or(CircuitError::NoMatrixRepresentation)
    }
}
