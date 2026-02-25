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

use crate::circuit::Circuit;
use crate::circuit::bit::Qubit;

use crate::circuit::gate::directive::Directive;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::gate::standard_gate::StandardGate;
use crate::circuit::operation::Operation;
use crate::circuit::param::CircuitParam;
use std::fmt::Write;

const PI: f64 = std::f64::consts::PI;
const PI_2: f64 = std::f64::consts::PI / 2.0;
const PI_4: f64 = std::f64::consts::PI / 4.0;

/// Errors that can occur during QCIS dumping.
#[derive(Debug)]
pub enum QcisDumpError {
    IoError(std::io::Error),
    /// Gate is not supported by QCIS backend and needs compilation
    UnsupportedGate(String),
    /// Parameter contains symbolic values that cannot be resolved to numbers
    SymbolicParameter(String),
}

impl std::fmt::Display for QcisDumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QcisDumpError::IoError(e) => write!(f, "IO error: {}", e),
            QcisDumpError::UnsupportedGate(g) => {
                write!(
                    f,
                    "Unsupported gate '{}': this gate is not natively supported by QCIS backend. ",
                    g
                )?;
                write!(
                    f,
                    "Please compile the circuit to QCIS basis gates (X2P, X2M, Y2P, Y2M, XY2P, XY2M, CZ, RZ, I, X, Y, Z, H, S, SD, T, TD, RX, RY, RXY) before dumping."
                )
            }
            QcisDumpError::SymbolicParameter(p) => {
                write!(
                    f,
                    "Symbolic parameter '{}': QCIS only supports numeric parameters. Please bind all symbolic parameters to numeric values before dumping.",
                    p
                )
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
/// Returns an error if the circuit contains gates that are not natively supported by QCIS.
/// The circuit must be compiled to QCIS basis gates before dumping.
pub fn dump(circuit: &Circuit, path: &std::path::PathBuf) -> Result<(), QcisDumpError> {
    let content = dumps(circuit)?;
    std::fs::write(path, content)?;
    Ok(())
}

/// Convert a circuit to a QCIS string.
///
/// # Errors
///
/// Returns an error if the circuit contains gates that are not natively supported by QCIS.
/// The circuit must be compiled to QCIS basis gates before dumping.
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

/// Convert a single operation to QCIS format.
///
/// Only QCIS natively supported gates are allowed. All other gates must be
/// compiled to QCIS basis gates before calling this function.
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
        Instruction::ControlFlowGate(_) => Err(QcisDumpError::UnsupportedGate(
            "ControlFlowGate".to_string(),
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
                    Err(_) => Ok(param.to_string()),
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
/// Only QCIS natively supported gates are allowed:
/// - Native gates: X2P, X2M, Y2P, Y2M, XY2P, XY2M, CZ, RZ, I
/// - Standard single-qubit: X, Y, Z, H, S, SD, T, TD
/// - Parameterized: RX, RY, RXY
fn standard_gate_to_qcis(
    gate: StandardGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    circuit: &Circuit,
) -> Result<String, QcisDumpError> {
    let qubit_str = format_qubits(qubits);
    let param_str = format_params(params, circuit)?;

    let gate_name = match gate {
        // Native QCIS gates
        StandardGate::X2P => "X2P",
        StandardGate::X2M => "X2M",
        StandardGate::Y2P => "Y2P",
        StandardGate::Y2M => "Y2M",
        StandardGate::XY2P => "XY2P",
        StandardGate::XY2M => "XY2M",
        StandardGate::CZ => "CZ",
        StandardGate::RZ => "RZ",

        // Standard single-qubit gates
        StandardGate::X => "X",
        StandardGate::Y => "Y",
        StandardGate::Z => "Z",
        StandardGate::H => "H",
        StandardGate::S => "S",
        // QCIS: use SD
        StandardGate::SDG => "SD",
        StandardGate::T => "T",
        // QCIS: use TD
        StandardGate::TDG => "TD",

        // Parameterized single-qubit gates
        StandardGate::RX => "RX",
        StandardGate::RY => "RY",
        StandardGate::RXY => "RXY",

        // All other gates are not natively supported by QCIS
        _ => {
            return Err(QcisDumpError::UnsupportedGate(format!("{:?}", gate)));
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
    let param_str = format_params(params, circuit)?;

    Ok(format!("I {} {}", qubit_str, param_str))
}

#[cfg(test)]
#[path = "./dump_test.rs"]
mod dump_test;
