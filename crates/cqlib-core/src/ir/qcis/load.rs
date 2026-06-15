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

//! QCIS Parser Module
//!
//! This module provides functionality to parse QCIS (Quantum Circuit Intermediate Representation)
//! format into the internal `Circuit` representation.
//!
//! ## QCIS Format
//!
//! Each line in QCIS represents a quantum operation with the format:
//! ```text
//! GATE_NAME QUBIT_LIST [PARAMETER_LIST]
//! ```
//!
//! - **QUBIT_LIST**: Space-separated list of qubits in `Q<id>` format (e.g., `Q0`, `Q1`)
//! - **PARAMETER_LIST**: Optional space-separated parameters (numeric values or expressions)
//!
//! ## Examples
//!
//! ```rust
//! use cqlib_core::ir::qcis::loads;
//!
//! // Single-qubit gate
//! let circuit = loads("H Q0").unwrap();
//!
//! // Two-qubit gate
//! let circuit = loads("CZ Q0 Q1").unwrap();
//!
//! // Parametrized gate
//! let circuit = loads("RX Q0 1.57").unwrap();
//! ```
//!
//! ## Comments and Whitespace
//!
//! - Lines starting with `//` are treated as comments and ignored
//! - Empty lines are ignored
//! - Extra whitespace is normalized

use crate::circuit::circuit_param::ParameterValue;
use crate::circuit::{Circuit, Parameter, Qubit};
use regex::Regex;
use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;
use thiserror::Error;

static QUBIT_PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^Q\d+$").unwrap());

/// Errors that can occur during QCIS parsing.
#[derive(Debug, Error)]
pub enum QcisParseError {
    /// File system or I/O error (file not found, permission denied, etc.)
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid qubit format (e.g., not "Q123" format)
    #[error("Invalid qubit format: '{0}' (expected format: Q<id>, e.g., Q0, Q1)")]
    InvalidQubitFormat(String),

    /// Invalid qubit ID (failed to parse the number after 'Q')
    #[error("Invalid qubit ID: '{0}'")]
    InvalidQubitId(String),

    /// Qubit count mismatch for a gate
    #[error("Qubit count mismatch for gate '{gate}': expected {expected}, got {actual}")]
    QubitCountMismatch {
        gate: String,
        expected: usize,
        actual: usize,
    },

    /// Parameter count mismatch for a gate
    #[error("Parameter count mismatch for gate '{gate}': expected {expected}, got {actual}")]
    ParameterCountMismatch {
        gate: String,
        expected: usize,
        actual: usize,
    },

    /// Missing required parameter for a gate
    #[error("Missing required parameter(s) for gate '{0}'")]
    MissingParameter(String),

    /// Failed to parse a parameter expression
    #[error("Failed to parse parameter '{param}' for gate '{gate}': {reason}")]
    InvalidParameter {
        gate: String,
        param: String,
        reason: String,
    },

    /// Unknown gate name
    #[error("Unknown gate: '{0}'")]
    UnknownGate(String),

    /// Empty line or no valid content
    #[error("Empty line or no valid content")]
    EmptyLine,
}

impl PartialEq for QcisParseError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::IoError(lhs), Self::IoError(rhs)) => {
                lhs.kind() == rhs.kind() && lhs.to_string() == rhs.to_string()
            }
            (Self::InvalidQubitFormat(lhs), Self::InvalidQubitFormat(rhs)) => lhs == rhs,
            (Self::InvalidQubitId(lhs), Self::InvalidQubitId(rhs)) => lhs == rhs,
            (
                Self::QubitCountMismatch {
                    gate: lhs_gate,
                    expected: lhs_expected,
                    actual: lhs_actual,
                },
                Self::QubitCountMismatch {
                    gate: rhs_gate,
                    expected: rhs_expected,
                    actual: rhs_actual,
                },
            ) => lhs_gate == rhs_gate && lhs_expected == rhs_expected && lhs_actual == rhs_actual,
            (
                Self::ParameterCountMismatch {
                    gate: lhs_gate,
                    expected: lhs_expected,
                    actual: lhs_actual,
                },
                Self::ParameterCountMismatch {
                    gate: rhs_gate,
                    expected: rhs_expected,
                    actual: rhs_actual,
                },
            ) => lhs_gate == rhs_gate && lhs_expected == rhs_expected && lhs_actual == rhs_actual,
            (Self::MissingParameter(lhs), Self::MissingParameter(rhs)) => lhs == rhs,
            (
                Self::InvalidParameter {
                    gate: lhs_gate,
                    param: lhs_param,
                    reason: lhs_reason,
                },
                Self::InvalidParameter {
                    gate: rhs_gate,
                    param: rhs_param,
                    reason: rhs_reason,
                },
            ) => lhs_gate == rhs_gate && lhs_param == rhs_param && lhs_reason == rhs_reason,
            (Self::UnknownGate(lhs), Self::UnknownGate(rhs)) => lhs == rhs,
            (Self::EmptyLine, Self::EmptyLine) => true,
            _ => false,
        }
    }
}

/// Result type for QCIS parsing operations.
pub type Result<T> = std::result::Result<T, QcisParseError>;

/// Gate specification defining required qubit and parameter counts.
///
/// Used to validate gate invocations against the QCIS specification.
#[derive(Debug, Clone, Copy)]
struct GateSpec {
    min_qubits: usize,
    max_qubits: usize,
    min_params: usize,
    max_params: usize,
}

impl GateSpec {
    const fn new(
        min_qubits: usize,
        max_qubits: usize,
        min_params: usize,
        max_params: usize,
    ) -> Self {
        Self {
            min_qubits,
            max_qubits,
            min_params,
            max_params,
        }
    }

    /// Expects exact qubit and parameter counts.
    const fn exact(qubits: usize, params: usize) -> Self {
        Self::new(qubits, qubits, params, params)
    }

    /// Variable qubit count (at least min), exact parameters.
    const fn min_qubits(min_qubits: usize, params: usize) -> Self {
        Self::new(min_qubits, usize::MAX, params, params)
    }
}

/// Get the gate specification for a given gate name.
fn get_gate_spec(gate_name: &str) -> Option<GateSpec> {
    match gate_name {
        // Native QCIS gates
        "X2P" | "X2M" | "Y2P" | "Y2M" => Some(GateSpec::exact(1, 0)),
        "XY2P" | "XY2M" => Some(GateSpec::exact(1, 1)),
        "CZ" => Some(GateSpec::exact(2, 0)),
        "RZ" => Some(GateSpec::exact(1, 1)),
        // Delay gate with time parameter
        "I" => Some(GateSpec::exact(1, 1)),

        // Standard single-qubit gates
        "X" | "Y" | "Z" | "H" | "S" | "SD" | "SDG" | "T" | "TD" | "TDG" => {
            Some(GateSpec::exact(1, 0))
        }

        // Parameterized single-qubit gates
        "RX" | "RY" => Some(GateSpec::exact(1, 1)),
        "RXY" => Some(GateSpec::exact(1, 2)),

        // Multi-qubit gates
        "B" | "Barrier" => Some(GateSpec::min_qubits(1, 0)),

        // Measurement - supports 1 or more qubits
        "M" => Some(GateSpec::min_qubits(1, 0)),

        _ => None,
    }
}

pub fn load<P: AsRef<Path>>(path: P) -> Result<Circuit> {
    let content = std::fs::read_to_string(path)?;
    loads(&content)
}

/// Parse a QCIS file and convert it to a [`Circuit`].
///
/// Rust-style alias for [`load`]. The Python-style `load` name is retained for
/// compatibility with the rest of the IR module API.
pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Circuit> {
    load(path)
}

pub fn loads(qcis: &str) -> Result<Circuit> {
    let mut c = Circuit::new(0);
    // Maintain a set of existing qubit IDs to avoid repeated circuit.qubits() calls
    let mut existing_qubits: HashSet<u32> = HashSet::new();

    for line in qcis.lines() {
        if let Err(e) = process_line(line, &mut c, &mut existing_qubits) {
            return Err(e);
        }
    }

    Ok(c)
}

/// Parse a QCIS source string and convert it to a [`Circuit`].
///
/// Rust-style alias for [`loads`].
pub fn from_str(qcis: &str) -> Result<Circuit> {
    loads(qcis)
}

/// Parse a parameter string into a ParameterValue.
/// Supports numbers, pi, e, and basic arithmetic operations (+, -, *, /).
fn parse_param(param_str: &str) -> Result<ParameterValue> {
    let param_str = param_str.trim();
    if param_str.is_empty() {
        return Err(QcisParseError::MissingParameter("".to_string()));
    }
    let p = Parameter::try_from(param_str);
    if let Ok(p) = p {
        if let Ok(v) = p.evaluate(&None) {
            Ok(ParameterValue::Fixed(v))
        } else {
            Ok(p.into())
        }
    } else {
        Err(QcisParseError::MissingParameter("".to_string()))
    }
}

/// Ensure the circuit has enough qubits for the given qubit IDs.
///
/// Uses the provided `existing_qubits` set for O(1) membership testing
/// instead of querying the circuit repeatedly.
fn ensure_qubits(c: &mut Circuit, qubits: &[Qubit], existing_qubits: &mut HashSet<u32>) {
    let mut missing = Vec::new();

    for q in qubits {
        let id = q.id();
        if existing_qubits.insert(id) {
            // insert returns true if the value was not already present
            missing.push(*q);
        }
    }

    if !missing.is_empty() {
        // Sort to ensure consistent ordering (though HashSet iteration order is arbitrary,
        // the qubit IDs themselves are what matter)
        missing.sort_by_key(|q| q.id());
        let _ = c.add_qubits(missing);
    }
}

/// Parse a qubit string (e.g., "Q0", "Q123") into a Qubit.
fn parse_qubit(s: &str) -> Result<Qubit> {
    if s.len() > 1 && s.starts_with('Q') {
        s[1..]
            .parse::<u32>()
            .map(Qubit::new)
            .map_err(|_| QcisParseError::InvalidQubitId(s.to_string()))
    } else {
        Err(QcisParseError::InvalidQubitFormat(s.to_string()))
    }
}

/// Validate qubit and parameter counts against gate specification.
fn validate_gate_args(gate_name: &str, qubits: &[Qubit], params: &[ParameterValue]) -> Result<()> {
    let spec = match get_gate_spec(gate_name) {
        Some(s) => s,
        None => {
            // Unknown gates are allowed without validation
            return Ok(());
        }
    };

    // Validate qubit count
    let qubit_count = qubits.len();
    if qubit_count < spec.min_qubits || qubit_count > spec.max_qubits {
        return Err(QcisParseError::QubitCountMismatch {
            gate: gate_name.to_string(),
            expected: spec.min_qubits,
            actual: qubit_count,
        });
    }

    // Validate parameter count
    let param_count = params.len();
    if param_count < spec.min_params || param_count > spec.max_params {
        return Err(QcisParseError::ParameterCountMismatch {
            gate: gate_name.to_string(),
            expected: spec.min_params,
            actual: param_count,
        });
    }

    Ok(())
}

fn process_line(line: &str, c: &mut Circuit, existing_qubits: &mut HashSet<u32>) -> Result<()> {
    let clean_line = line.split("//").next().unwrap_or("").trim();
    if clean_line.is_empty() {
        return Ok(());
    }

    let parts = clean_line.split_whitespace().collect::<Vec<_>>();

    if parts.is_empty() {
        return Ok(());
    }

    // let qubit_pattern = Regex::new(r"^Q\d+$").unwrap();

    let gate_name = parts[0];
    let args = &parts[1..];

    // Validate qubit arguments first
    for &token in args.iter() {
        if (token.starts_with('Q') || token.starts_with('q')) && !QUBIT_PATTERN.is_match(token) {
            return Err(QcisParseError::InvalidQubitFormat(token.to_string()));
        }
    }

    let split_index = args
        .iter()
        .position(|&token| !QUBIT_PATTERN.is_match(token))
        .unwrap_or(args.len());

    let (qubit_slice, param_slice) = args.split_at(split_index);

    // Parse qubits with explicit error handling
    let qubits: Vec<Qubit> = qubit_slice
        .iter()
        .map(|&s| parse_qubit(s))
        .collect::<Result<Vec<_>>>()?;

    // Parse parameters
    let params: Vec<ParameterValue> = param_slice
        .iter()
        .map(|&s| parse_param(s))
        .collect::<Result<Vec<_>>>()?;

    // Validate qubit and parameter counts
    validate_gate_args(gate_name, &qubits, &params)?;

    // Ensure circuit has enough qubits (using cached set for efficiency)
    ensure_qubits(c, &qubits, existing_qubits);

    // Helper macro to apply single-qubit gates
    macro_rules! apply_single_qubit {
        ($method:ident) => {{
            if let Some(&q) = qubits.first() {
                c.$method(q)
                    .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
            }
        }};
    }

    // Helper macro to apply single-qubit gates with one parameter
    macro_rules! apply_single_qubit_with_param {
        ($method:ident) => {{
            if let Some(&q) = qubits.first() {
                if let Some(param) = params.first() {
                    c.$method(q, param.clone())
                        .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
                }
            }
        }};
    }

    match gate_name {
        // Native QCIS gates
        "X2P" => apply_single_qubit!(x2p),
        "X2M" => apply_single_qubit!(x2m),
        "Y2P" => apply_single_qubit!(y2p),
        "Y2M" => apply_single_qubit!(y2m),
        "XY2P" => apply_single_qubit_with_param!(xy2p),
        "XY2M" => apply_single_qubit_with_param!(xy2m),
        "CZ" => {
            if qubits.len() == 2 {
                c.cz(qubits[0], qubits[1])
                    .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
            }
        }
        "RZ" => apply_single_qubit_with_param!(rz),
        "I" => apply_single_qubit_with_param!(delay),

        // Barrier
        "B" | "Barrier" => {
            if !qubits.is_empty() {
                c.barrier(qubits)
                    .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
            }
        }

        // Standard single-qubit gates
        "X" => apply_single_qubit!(x),
        "Y" => apply_single_qubit!(y),
        "Z" => apply_single_qubit!(z),
        "H" => apply_single_qubit!(h),
        "S" => apply_single_qubit!(s),
        "SD" | "SDG" => apply_single_qubit!(sdg),
        "T" => apply_single_qubit!(t),
        "TD" | "TDG" => apply_single_qubit!(tdg),

        // Parameterized single-qubit gates
        "RX" => apply_single_qubit_with_param!(rx),
        "RY" => apply_single_qubit_with_param!(ry),
        "RXY" => {
            if let Some(&q) = qubits.first() {
                if params.len() >= 2 {
                    let theta = params[0].clone();
                    let phi = params[1].clone();
                    c.rxy(q, theta, phi)
                        .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
                }
            }
        }

        // Measurement - supports 1 or more qubits
        "M" => {
            for q in qubits.iter() {
                c.measure(*q)
                    .map_err(|e| QcisParseError::InvalidQubitId(format!("{:?}", e)))?;
            }
        }

        _ => {
            // Unknown gate
            return Err(QcisParseError::UnknownGate(gate_name.to_string()));
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "./load_test.rs"]
mod load_test;
