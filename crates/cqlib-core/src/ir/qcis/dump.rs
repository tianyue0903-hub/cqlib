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

//! QCIS Serializer Module
//!
//! This module provides functionality to serialize internal `Circuit` representations
//! into QCIS (Quantum Circuit Intermediate Representation) format.
//!
//! ## QCIS Output Format
//!
//! The serializer outputs one operation per line:
//! ```text
//! GATE Q0 Q1 ... [param1] [param2] ...
//! ```
//!
//! ## Gate Mapping
//!
//! Standard gates use their QCIS opcode equivalents:
//! - `SDG` (S dagger) → `SD`
//! - `TDG` (T dagger) → `TD`
//! - `Delay` → `I Qn t`, where `t` is a non-negative integer count in 0.5 ns ticks
//!
//! ## Parameter Formatting
//!
//! Common values are simplified:
//! - `pi` for π
//! - `-pi` for -π
//! - `pi/2` for π/2
//! - `-pi/2` for -π/2
//! - `pi/4` for π/4
//! - `-pi/4` for -π/4
//! - Integers without decimal when whole number
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::ir::qcis::dumps;
//! use cqlib_core::circuit::{Circuit, Qubit};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let qcis = dumps(&circuit).unwrap();
//! assert_eq!(qcis, "H Q0\nCZ Q0 Q1\n");
//! ```
//!
//! ## Limitations
//!
//! - Standard identity and global-phase gates are not represented
//! - Control flow gates (if/while) are not supported
//! - Custom gates (CircuitGate, UnitaryGate) require prior compilation

use crate::circuit::Circuit;
use crate::circuit::bit::Qubit;

use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::gate::directive::Directive;
use crate::circuit::gate::standard_gate::StandardGate;
use crate::circuit::gate::{ClassicalDataOp, Instruction};
use crate::circuit::operation::Operation;
use std::fmt::Write;
use std::path::Path;

const PI: f64 = std::f64::consts::PI;
const PI_2: f64 = std::f64::consts::PI / 2.0;
const PI_4: f64 = std::f64::consts::PI / 4.0;

/// Errors that can occur during QCIS dumping.
#[derive(Debug)]
pub enum QcisDumpError {
    IoError(std::io::Error),
    /// Gate is not represented by the QCIS circuit format
    UnsupportedGate(String),
    /// Classical data operation is not representable in QCIS.
    UnsupportedClassicalData(String),
    /// Classical control flow is not representable in QCIS.
    UnsupportedClassicalControl(String),
    /// Parameter contains symbolic values that cannot be resolved to numbers
    SymbolicParameter(String),
    /// Delay parameter is not a valid QCIS tick count.
    InvalidDelayParameter(String),
}

impl std::fmt::Display for QcisDumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QcisDumpError::IoError(e) => write!(f, "IO error: {}", e),
            QcisDumpError::UnsupportedGate(g) => {
                write!(
                    f,
                    "Unsupported gate '{g}': it is not represented by the QCIS circuit format"
                )
            }
            QcisDumpError::UnsupportedClassicalData(operation) => write!(
                f,
                "Unsupported classical data operation '{operation}': QCIS does not represent classical storage"
            ),
            QcisDumpError::UnsupportedClassicalControl(operation) => write!(
                f,
                "Unsupported classical control operation '{operation}': QCIS does not represent classical control flow"
            ),
            QcisDumpError::SymbolicParameter(p) => {
                write!(
                    f,
                    "Symbolic parameter '{}': QCIS only supports numeric parameters. Please bind all symbolic parameters to numeric values before dumping.",
                    p
                )
            }
            QcisDumpError::InvalidDelayParameter(reason) => {
                write!(f, "Invalid QCIS delay parameter: {reason}")
            }
        }
    }
}

impl std::error::Error for QcisDumpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            QcisDumpError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for QcisDumpError {
    fn from(e: std::io::Error) -> Self {
        QcisDumpError::IoError(e)
    }
}

/// Write a circuit to a QCIS file.
///
/// # Errors
///
/// Returns an error if the circuit contains instructions not represented by QCIS.
pub fn dump<P: AsRef<Path>>(circuit: &Circuit, path: P) -> Result<(), QcisDumpError> {
    let content = dumps(circuit)?;
    std::fs::write(path, content)?;
    Ok(())
}

/// Write a circuit to a QCIS file.
///
/// Rust-style alias for [`dump`]. The Python-style `dump` name is retained for
/// compatibility with the rest of the IR module API.
pub fn to_path<P: AsRef<Path>>(circuit: &Circuit, path: P) -> Result<(), QcisDumpError> {
    dump(circuit, path)
}

/// Convert a circuit to a QCIS string.
///
/// # Errors
///
/// Returns an error if the circuit contains instructions not represented by QCIS.
pub fn dumps(circuit: &Circuit) -> Result<String, QcisDumpError> {
    let mut output = String::new();
    let operations = circuit.operations();

    for op in operations {
        let line = operation_to_qcis(op, circuit)?;
        if !line.is_empty() {
            writeln!(output, "{}", line)
                .map_err(|e| QcisDumpError::IoError(std::io::Error::other(e)))?;
        }
    }

    Ok(output)
}

/// Serialize a circuit to a QCIS string.
///
/// Rust-style alias for [`dumps`].
pub fn to_string(circuit: &Circuit) -> Result<String, QcisDumpError> {
    dumps(circuit)
}

/// Convert a single operation to QCIS format.
///
/// Instructions without a QCIS circuit-level representation return an error.
fn operation_to_qcis(op: &Operation, circuit: &Circuit) -> Result<String, QcisDumpError> {
    match &op.instruction {
        Instruction::Standard(gate) => {
            standard_gate_to_qcis(*gate, &op.qubits, &op.params, circuit)
        }
        Instruction::McGate(_) => Err(QcisDumpError::UnsupportedGate(
            "Multi-controlled gate".to_string(),
        )),
        Instruction::UnitaryGate(_) => {
            Err(QcisDumpError::UnsupportedGate("UnitaryGate".to_string()))
        }
        Instruction::CircuitGate(_) => Err(QcisDumpError::UnsupportedGate(
            "CircuitGate (custom gate)".to_string(),
        )),
        Instruction::ClassicalData(ClassicalDataOp::MeasureBit { .. })
        | Instruction::ClassicalData(ClassicalDataOp::MeasureBits { .. }) => {
            Ok(format!("M {}", format_qubits(&op.qubits)))
        }
        Instruction::ClassicalData(ClassicalDataOp::Store { .. }) => {
            Err(QcisDumpError::UnsupportedClassicalData("store".to_string()))
        }
        Instruction::ClassicalControl(control) => Err(QcisDumpError::UnsupportedClassicalControl(
            format!("{control:?}"),
        )),
        Instruction::Directive(dir) => directive_to_qcis(*dir, &op.qubits),
        Instruction::Delay => delay_to_qcis(&op.qubits, &op.params, circuit),
    }
}

/// Format qubits as QCIS format (e.g., "Q0 Q1 Q2").
fn format_qubits(qubits: &[Qubit]) -> String {
    qubits
        .iter()
        .map(|q| format!("Q{}", q.id()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format parameters as QCIS format.
fn format_params(params: &[CircuitParam], circuit: &Circuit) -> Result<String, QcisDumpError> {
    let formatted: Result<Vec<_>, _> = params
        .iter()
        .map(|p| format_circuit_param(p, circuit))
        .collect();
    Ok(formatted?.join(" "))
}

/// Format a single circuit parameter.
fn format_circuit_param(param: &CircuitParam, circuit: &Circuit) -> Result<String, QcisDumpError> {
    match param {
        CircuitParam::Fixed(v) => Ok(format_float(*v)),
        CircuitParam::Index(idx) => {
            // Look up the parameter in circuit's parameter table
            if let Some(param) = circuit.parameters().iter().nth(*idx as usize) {
                // Try to evaluate the parameter to a numeric value
                match param.evaluate(&None) {
                    Ok(value) => Ok(format_float(value)),
                    Err(_) => Ok(param.to_string().replace("π", "pi")),
                }
            } else {
                // Parameter index not found
                Err(QcisDumpError::SymbolicParameter(format!("p{}", idx)))
            }
        }
    }
}

/// Format a float value, using special notation for common values.
fn format_float(v: f64) -> String {
    // Check for special values
    if (v - PI).abs() < 1e-10 {
        "pi".to_string()
    } else if (v - (-PI)).abs() < 1e-10 {
        "-pi".to_string()
    } else if (v - PI_2).abs() < 1e-10 {
        "pi/2".to_string()
    } else if (v - (-PI_2)).abs() < 1e-10 {
        "-pi/2".to_string()
    } else if (v - PI_4).abs() < 1e-10 {
        "pi/4".to_string()
    } else if (v - (-PI_4)).abs() < 1e-10 {
        "-pi/4".to_string()
    } else if v == 0.0 {
        "0".to_string()
    } else if v == 1.0 {
        "1".to_string()
    } else if v == -1.0 {
        "-1".to_string()
    } else if v.fract() == 0.0 {
        // Integer value
        format!("{:.0}", v)
    } else {
        // General float
        format!("{:.10}", v)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Convert a standard gate to QCIS format.
///
/// QCIS `I Qn t` is a delay instruction, not the cqlib identity gate. Global
/// phase operations are intentionally not represented.
fn standard_gate_to_qcis(
    gate: StandardGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    circuit: &Circuit,
) -> Result<String, QcisDumpError> {
    let qubit_str = format_qubits(qubits);
    let param_str = format_params(params, circuit)?;

    let gate_name = match gate {
        StandardGate::H => "H",
        StandardGate::RX => "RX",
        StandardGate::RXX => "RXX",
        StandardGate::RXY => "RXY",
        StandardGate::RY => "RY",
        StandardGate::RYY => "RYY",
        StandardGate::RZ => "RZ",
        StandardGate::RZX => "RZX",
        StandardGate::RZZ => "RZZ",
        StandardGate::S => "S",
        StandardGate::SDG => "SD",
        StandardGate::SWAP => "SWAP",
        StandardGate::T => "T",
        StandardGate::TDG => "TD",
        StandardGate::U => "U",
        StandardGate::X => "X",
        StandardGate::XY => "XY",
        StandardGate::X2P => "X2P",
        StandardGate::X2M => "X2M",
        StandardGate::XY2P => "XY2P",
        StandardGate::XY2M => "XY2M",
        StandardGate::Y => "Y",
        StandardGate::Y2P => "Y2P",
        StandardGate::Y2M => "Y2M",
        StandardGate::Z => "Z",
        StandardGate::Phase => "PHASE",
        StandardGate::CX => "CX",
        StandardGate::CCX => "CCX",
        StandardGate::CY => "CY",
        StandardGate::CZ => "CZ",
        StandardGate::CRX => "CRX",
        StandardGate::CRY => "CRY",
        StandardGate::CRZ => "CRZ",
        StandardGate::FSIM => "FSIM",
        StandardGate::I | StandardGate::GPhase => {
            return Err(QcisDumpError::UnsupportedGate(format!("{gate:?}")));
        }
    };

    let line = if param_str.is_empty() {
        format!("{} {}", gate_name, qubit_str)
    } else {
        format!("{} {} {}", gate_name, qubit_str, param_str)
    };

    Ok(line)
}

/// Convert a directive to QCIS format.
fn directive_to_qcis(dir: Directive, qubits: &[Qubit]) -> Result<String, QcisDumpError> {
    match dir {
        Directive::Measure => {
            let qubit_str = format_qubits(qubits);
            Ok(format!("M {}", qubit_str))
        }
        Directive::Barrier => {
            let qubit_str = format_qubits(qubits);
            Ok(format!("B {}", qubit_str))
        }
        Directive::Reset => Err(QcisDumpError::UnsupportedGate("Reset".to_string())),
    }
}

/// Convert delay (I gate) to QCIS format.
fn delay_to_qcis(
    qubits: &[Qubit],
    params: &[CircuitParam],
    circuit: &Circuit,
) -> Result<String, QcisDumpError> {
    let qubit_str = format_qubits(qubits);
    let param_str = format_delay_param(params, circuit)?;

    Ok(format!("I {} {}", qubit_str, param_str))
}

fn format_delay_param(params: &[CircuitParam], circuit: &Circuit) -> Result<String, QcisDumpError> {
    let [param] = params else {
        return Err(QcisDumpError::InvalidDelayParameter(
            "QCIS delay requires exactly one tick parameter".to_string(),
        ));
    };

    let value = match param {
        CircuitParam::Fixed(value) => *value,
        CircuitParam::Index(idx) => {
            let param = circuit
                .parameters()
                .iter()
                .nth(*idx as usize)
                .ok_or_else(|| QcisDumpError::SymbolicParameter(format!("p{}", idx)))?;
            param
                .evaluate(&None)
                .map_err(|_| QcisDumpError::SymbolicParameter(param.to_string()))?
        }
    };

    if !value.is_finite() || value < 0.0 || value.fract().abs() >= 1e-10 {
        return Err(QcisDumpError::InvalidDelayParameter(format!(
            "{value} is not a non-negative integer number of 0.5 ns ticks"
        )));
    }

    Ok(format_float(value))
}

#[cfg(test)]
#[path = "./dump_test.rs"]
mod dump_test;
