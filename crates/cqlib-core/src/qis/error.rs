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

use crate::circuit::CircuitError;
use thiserror::Error;

/// Comprehensive error type for the `qis` (Quantum Information Science) module.
///
/// This enum captures errors that can occur during statevector simulation,
/// density matrix simulation, observables, Hamiltonians, and Pauli operator manipulations.
#[derive(Debug, Error)]
pub enum QisError {
    /// Errors originating from quantum circuit operations, such as unsupported gates
    /// or symbolic parameters.
    #[error(transparent)]
    CircuitError(#[from] CircuitError),

    /// Thrown when the number of qubits of an operator (e.g., Hamiltonian) does not match
    /// the number of qubits of the quantum state being measured.
    #[error("Qubit count mismatch: expected {expected}, got {actual}")]
    QubitMismatch { expected: usize, actual: usize },

    /// Thrown when the dimensions of matrices or vectors do not match expectations
    /// (e.g., when multiplying matrices of incompatible shapes).
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Thrown when a state vector or density matrix has an invalid length
    /// (e.g., its length is not a power of 2).
    #[error("Invalid state dimension: length {0} is not a power of 2")]
    InvalidStateDimension(usize),

    /// Thrown when a probability value is mathematically invalid (e.g., < 0 or > 1).
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),

    /// Thrown when attempting to access a qubit or amplitude index that is out of bounds.
    #[error("Index out of bounds: index {index}, max {max}")]
    IndexOutOfBounds { index: usize, max: usize },

    /// Thrown when an operation is requested but not supported by the simulation backend.
    #[error("Unsupported simulation operation: {0}")]
    UnsupportedOperation(String),

    /// Thrown when parsing a Pauli string fails.
    #[error(transparent)]
    PauliStringParseError(#[from] PauliStringParseError),

    /// Thrown when an explicitly provided quantum state is not normalized
    /// (i.e., trace or norm is not 1).
    #[error("State is not normalized")]
    NotNormalized,

    /// Thrown when an operator is expected to be Hermitian but is not
    /// (e.g., a non-Hermitian Hamiltonian or density matrix).
    #[error("Operator is not Hermitian")]
    NotHermitian,

    /// Thrown when a parameter value is invalid.
    #[error("Invalid parameter value: {0}")]
    InvalidParameterValue(String),

    /// Thrown when a subsystem specification is invalid for entanglement calculations.
    #[error("Invalid subsystem: {0}")]
    InvalidSubsystem(String),

    /// Thrown when an operation requires a specific dimension that is not met.
    #[error("Unsupported dimension: expected {expected}, got {actual}")]
    UnsupportedDimension { expected: usize, actual: usize },
}

/// Error type for parsing PauliString from a string representation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PauliStringParseError {
    /// The string provided for parsing is empty.
    #[error("empty string")]
    EmptyString,

    /// The string contains an invalid or unrecognized character.
    #[error("invalid character '{0}'")]
    InvalidCharacter(char),

    /// The string only contains a phase but lacks any Pauli operators.
    #[error("no Pauli operators specified")]
    NoOperators,
}
