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

use thiserror::Error;

/// Errors that can occur during the numerical evaluation of symbolic parameters.
///
/// This enum captures all failure modes when resolving a symbolic expression tree ([`ExprNode`](crate::circuit::parameter::expr_node::ExprNode))
/// into a concrete floating-point value. These errors typically arise during the binding phase of a
/// Parameterized Quantum Circuit (PQC).
#[derive(Debug, Error)]
pub enum EvalError {
    /// Indicates that a symbolic variable required for evaluation was missing from the provided bindings.
    ///
    /// # Example
    /// evaluating `theta + phi` with bindings `{ "theta": 1.0 }` will raise `UndefinedSymbol("phi")`.
    #[error("Symbol not found: {0}")]
    UndefinedSymbol(String),

    /// Indicates an arithmetic error where a division or modulo operation by zero occurred.
    ///
    /// This includes both explicit division by zero (e.g., `x / 0.0`) and runtime zeros (e.g., `x / y` where `y = 0.0`).
    #[error("Division by zero")]
    DivisionByZero,

    /// Indicates that a mathematical function was called with an argument outside its defined domain.
    ///
    /// Common causes include:
    /// - `sqrt(x)` where `x < 0`
    /// - `ln(x)` where `x <= 0`
    /// - `asin(x)` or `acos(x)` where `|x| > 1`
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Indicates that an intermediate or final calculation resulted in `NaN` (Not a Number).
    ///
    /// This serves as a catch-all for undefined numerical behaviors not covered by other variants.
    #[error("Calculation resulted in NaN: {0}")]
    NaN(String),
}

/// A comprehensive error type for operations involving Quantum Circuits.
///
/// This enum aggregates errors from circuit construction, validation, and manipulation.
/// It serves as the primary error type for the `Circuit` struct and its related methods.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Thrown when attempting to add qubits that share the same unique identifier (index).
    ///
    /// Qubit indices within a circuit must be unique to ensure unambiguous addressing.
    #[error("Duplicate qubits found in circuit definition")]
    DuplicateQubits,

    /// Thrown when an operation references a qubit that is not part of the circuit.
    ///
    /// All qubits used in operations (gates, measurements) must be explicitly added to the circuit
    /// via `add_qubits` or `new` before use.
    #[error("Qubit {0} not found in circuit")]
    QubitNotFound(u32),

    /// Thrown when an operation is requested to provide a unitary matrix, but none exists.
    ///
    /// This typically happens when calling `.matrix()` on non-unitary instructions such as:
    /// - `Measure`
    /// - `Barrier`
    /// - `Reset`
    #[error("Instruction has no matrix representation")]
    NoMatrixRepresentation,

    /// Wraps errors that occur during the evaluation of circuit parameters.
    ///
    /// This variant propagates [`EvalError`]s up the stack when operations like matrix generation
    /// or gate inversion fail due to invalid parameter bindings.
    #[error("Failed to evaluate parameters: {0}")]
    ParamEvaluationFailed(#[from] EvalError),
}
