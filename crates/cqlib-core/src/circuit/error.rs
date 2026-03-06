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

    /// Indicates that the expression tree exceeded the maximum allowed recursion depth during evaluation.
    ///
    /// This typically occurs with deeply nested expressions (e.g., `((((x+1)+1)+1)...)`).
    /// Consider simplifying the expression or increasing the depth limit.
    #[error("Maximum recursion depth exceeded during evaluation (depth limit: {0})")]
    MaxRecursionDepthExceeded(usize),
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

    /// Thrown when the number of qubits provided for an operation does not match its definition.
    ///
    /// For example, applying a 2-qubit CNOT gate to 3 qubits, or a 1-qubit Custom Unitary to 2 qubits.
    #[error("Qubit count mismatch: expected {expected}, got {actual}")]
    QubitCountMismatch { expected: usize, actual: usize },

    #[error("Parameter count mismatch: expected {expected}, got {actual}")]
    ParameterCountMismatch { expected: usize, actual: usize },

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

    #[error("Operation is irreversible")]
    IrreversibleOperation,

    #[error("Operation cannot be controlled: {0}")]
    InvalidControlOperation(String),

    #[error("Symbolic parameter cannot be evaluated in this context")]
    SymbolicParameterError,

    /// Thrown when a parameter cannot be resolved during circuit decomposition.
    ///
    /// This occurs when a sub-circuit or control flow body contains symbolic parameters
    /// that are not bound to concrete values and cannot be evaluated.
    #[error("Unresolved parameter in decomposition: {0}")]
    UnresolvedParameter(String),

    /// Thrown when a parameter index is out of range.
    ///
    /// This indicates internal data corruption or an inconsistency between the
    /// circuit's parameter table and the indices stored in operations.
    #[error("Parameter index {0} out of range")]
    InvalidParameterIndex(u32),

    /// Thrown when a gate parameter has an invalid value (NaN or Infinity).
    ///
    /// This occurs when evaluating gate matrices with non-finite parameter values.
    #[error("Invalid parameter value at index {0}: {1}")]
    InvalidParameterValue(usize, f64),

    /// Thrown when the control flow graph (CFG) has an invalid or inconsistent structure.
    ///
    /// This error indicates that the DAG representation of the circuit is malformed,
    /// which can occur when:
    /// - A Branch terminator lacks required TrueBranch or FalseBranch edges
    /// - A Jump terminator references a non-existent block
    /// - The entry block is not set
    /// - Edges reference nodes that don't exist in the graph
    ///
    /// # Examples
    ///
    /// This error is returned when converting an invalid `CircuitDag` back to a `Circuit`
    /// if the DAG structure was manually modified and left in an inconsistent state.
    #[error("Invalid control flow graph structure: {0}")]
    InvalidControlFlow(String),

    #[error("Invalid Operation: {0}")]
    InvalidOperation(String),
}

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("error")]
    Error,
}
