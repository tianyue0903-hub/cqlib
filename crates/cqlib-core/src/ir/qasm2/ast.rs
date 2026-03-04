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

//! # OpenQASM 2.0 Abstract Syntax Tree (AST)
//!
//! This module defines the intermediate representation (IR) for parsing OpenQASM 2.0 quantum
//! programs. It provides data structures that represent the grammatical elements of the
//! OpenQASM 2.0 language specification.
//!
//! ## Overview
//!
//! The AST is structured as a tree of statements that correspond to the quantum circuit
//! description language defined in the OpenQASM 2.0 specification. Key elements include:
//!
//! - **Expressions**: Mathematical expressions for gate parameters (real numbers, integers,
//!   identifiers, binary/ unary operations)
//! - **Arguments**: Quantum/classical register references (named or indexed)
//! - **Statements**: Quantum operations, gate declarations, classical operations
//! - **Program**: Complete OpenQASM program containing version and statements
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::ir::qasm2::ast::{Statement, Argument, Expression};
//!
//! // Represents: qreg q[2];
//! let qreg_stmt = Statement::QReg("q".to_string(), 2);
//!
//! // Represents: h q[0];
//! let h_gate = Statement::CustomGate(
//!     "h".to_string(),
//!     vec![],
//!     vec![Argument::IndexedId("q".to_string(), 0)]
//! );
//! ```

/// Represents literal values and expressions in OpenQASM 2.0.
///
/// This enum covers all expression types defined in the OpenQASM 2.0 specification:
/// - Real and integer constants
/// - Identifier references (variables/parameters)
/// - Built-in constant `pi`
/// - Binary operations (arithmetic)
/// - Unary operations (trigonometric, exponential, etc.)
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Real number literal (e.g., `3.14159`, `0.5`)
    Real(f64),
    /// Integer literal (e.g., `1`, `2`, `-3`)
    Integer(i64),
    /// Identifier reference (parameter name or variable)
    Id(String),
    /// Built-in constant pi (π ≈ 3.14159...)
    Pi,
    /// Binary operation (e.g., `a + b`, `x * 2`, `pi/2`)
    BinaryOp(Box<Expression>, OpCode, Box<Expression>),
    /// Unary operation (e.g., `sin(theta)`, `sqrt(x)`, `-y`)
    UnaryOp(UnaryOpCode, Box<Expression>),
}

/// Binary arithmetic operators supported in OpenQASM 2.0 expressions.
///
/// These operators follow standard precedence rules and are used within
/// gate parameter expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum OpCode {
    /// Addition operator (`+`)
    Add,
    /// Subtraction operator (`-`)
    Sub,
    /// Multiplication operator (`*`)
    Mul,
    /// Division operator (`/`)
    Div,
    /// Power/exponentiation operator (`^`)
    Pow,
}

/// Unary operators (functions) supported in OpenQASM 2.0 expressions.
///
/// These include trigonometric functions, exponential/logarithmic functions,
/// and the negation operator.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOpCode {
    /// Sine function: `sin(x)`
    Sin,
    /// Cosine function: `cos(x)`
    Cos,
    /// Tangent function: `tan(x)`
    Tan,
    /// Exponential function: `exp(x)`
    Exp,
    /// Natural logarithm: `ln(x)`
    Ln,
    /// Square root: `sqrt(x)`
    Sqrt,
    /// Arcsine: `asin(x)`
    Asin,
    /// Arccosine: `acos(x)`
    Acos,
    /// Arctangent: `atan(x)`
    Atan,
    /// Negation operator: `-x`
    Neg,
}

/// Represents a quantum or classical register argument.
///
/// Arguments can be either:
/// - Named references (entire register): `q` refers to all qubits in register `q`
/// - Indexed references (single element): `q[0]` refers to the first qubit in register `q`
#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    /// Named register reference (e.g., `q` in `h q`)
    Id(String),
    /// Indexed register reference (e.g., `q[0]` in `cx q[0], q[1]`)
    IndexedId(String, i64),
}

/// Top-level statements in an OpenQASM 2.0 program.
///
/// Each variant represents a distinct type of statement as defined in the
/// OpenQASM 2.0 specification, including declarations, quantum operations,
/// gate definitions, and control flow.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Quantum register declaration: `qreg q[5];`
    QReg(String, i64),
    /// Classical register declaration: `creg c[3];`
    CReg(String, i64),
    /// Include external file: `include "qelib1.inc";`
    Include(String),
    /// Barrier operation: `barrier q[0], q[1];`
    Barrier(Vec<Argument>),
    /// Reset operation: `reset q[0];`
    Reset(Argument),
    /// Measurement: `measure q[0] -> c[0];`
    Measure(Argument, Argument),
    /// Custom gate invocation: `gate_name(params) qubits;`
    CustomGate(String, Vec<Expression>, Vec<Argument>),
    /// Opaque gate declaration (no body): `opaque gate_name q;`
    Opaque(String, Vec<String>, Vec<String>),
    /// Gate definition with body: `gate gate_name(params) q { ... }`
    GateDecl(Box<GateDeclData>),
    /// Conditional execution: `if (c[0] == 1) gate q;`
    If(String, i64, Box<Statement>),
}

/// Data structure holding gate definition content.
///
/// This represents the complete gate declaration including:
/// - Gate name
/// - Formal parameters (may be empty)
/// - Formal qubits (the qubits the gate operates on)
/// - Gate body (sequence of statements)
#[derive(Debug, Clone, PartialEq)]
pub struct GateDeclData {
    /// Name of the gate being defined
    pub name: String,
    /// List of parameter names (empty if gate takes no parameters)
    pub params: Vec<String>,
    /// List of qubit argument names (the qubits the gate operates on)
    pub qubits: Vec<String>,
    /// Gate body: sequence of quantum operations making up the gate
    pub body: Vec<Statement>,
}

/// Represents a complete OpenQASM 2.0 program.
///
/// This is the root structure containing the version number and
/// all top-level statements in the program.
#[derive(Debug, Clone)]
pub struct OpenQASMProgram {
    /// OpenQASM version (typically 2.0)
    pub version: f64,
    /// List of top-level statements in program order
    pub statements: Vec<Statement>,
}
