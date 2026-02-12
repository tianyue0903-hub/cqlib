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

//! Integration tests for QASM2 to QCIS conversion.
//!
//! These tests verify that quantum circuits maintain structural consistency
//! when converted from OpenQASM 2.0 format to QCIS format, particularly
//! focusing on measurement qubit ordering.

use crate::circuit::gate::{Directive, Instruction};
use crate::ir::qasm2::load::loads as qasm2_loads;
use crate::ir::qcis::dump::dumps as qcis_dumps;

/// Test that measurement qubit order is preserved during QASM2 -> QCIS conversion.
///
/// This test uses a circuit with:
/// - Multiple single-qubit gates (X, Y, H)
/// - Two-qubit gates (CZ)
/// - Measurements in non-sequential order (q[2]->c[1], q[3]->c[0], etc.)
/// - Gates applied after measurements
///
/// The key assertion is that measurement operations in QCIS output
/// maintain the same qubit ordering as specified in the original QASM.
#[test]
fn test_measurement_qubit_order_qasm2_to_qcis() {
    // Note: Using CZ instead of CNOT since QCIS natively supports CZ
    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
x q[0];
y q[1];
y q[2];
h q[3];
cz q[0],q[1];
measure q[2] -> c[1];
measure q[3] -> c[0];
y q[0];
measure q[1] -> c[2];
measure q[0] -> c[3];
"#;

    // Parse QASM to circuit
    let circuit = qasm2_loads(qasm).expect("Failed to parse QASM");

    // Verify circuit structure before conversion
    let ops = circuit.operations();

    // Find measurement operations and verify their qubit order
    let measure_ops: Vec<_> = ops
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Directive(Directive::Measure)))
        .collect();

    assert_eq!(measure_ops.len(), 4, "Expected 4 measurement operations");

    // Verify original measurement order: q[2], q[3], q[1], q[0]
    assert_eq!(
        measure_ops[0].qubits[0].id(),
        2,
        "First measurement should be on q[2]"
    );
    assert_eq!(
        measure_ops[1].qubits[0].id(),
        3,
        "Second measurement should be on q[3]"
    );
    assert_eq!(
        measure_ops[2].qubits[0].id(),
        1,
        "Third measurement should be on q[1]"
    );
    assert_eq!(
        measure_ops[3].qubits[0].id(),
        0,
        "Fourth measurement should be on q[0]"
    );

    // Convert to QCIS
    let qcis = qcis_dumps(&circuit).expect("Failed to convert to QCIS");

    // Parse QCIS output and verify measurement lines
    let lines: Vec<_> = qcis.lines().collect();

    // Find measurement lines in QCIS output
    let measure_lines: Vec<_> = lines.iter().filter(|line| line.starts_with("M ")).collect();

    assert_eq!(
        measure_lines.len(),
        4,
        "Expected 4 measurement lines in QCIS output"
    );

    // QCIS measurements should be in the same order as QASM
    // Each measurement line format: "M Q<n>"
    assert_eq!(
        measure_lines[0], &"M Q2",
        "First QCIS measurement should be M Q2"
    );
    assert_eq!(
        measure_lines[1], &"M Q3",
        "Second QCIS measurement should be M Q3"
    );
    assert_eq!(
        measure_lines[2], &"M Q1",
        "Third QCIS measurement should be M Q1"
    );
    assert_eq!(
        measure_lines[3], &"M Q0",
        "Fourth QCIS measurement should be M Q0"
    );

    // Verify full QCIS output structure
    let expected_qcis = r#"X Q0
Y Q1
Y Q2
H Q3
CZ Q0 Q1
M Q2
M Q3
Y Q0
M Q1
M Q0
"#;
    assert_eq!(
        qcis, expected_qcis,
        "QCIS output should match expected structure with correct measurement order"
    );
}

/// Test multi-qubit measurement in a single measure statement.
#[test]
fn test_multi_qubit_measurement_order() {
    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
measure q[3] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[0] -> c[3];
"#;

    let circuit = qasm2_loads(qasm).expect("Failed to parse QASM");
    let qcis = qcis_dumps(&circuit).expect("Failed to convert to QCIS");

    // Verify measurement order in QCIS output
    let lines: Vec<_> = qcis.lines().collect();
    let measure_lines: Vec<_> = lines.iter().filter(|line| line.starts_with("M ")).collect();

    assert_eq!(measure_lines.len(), 4);
    assert_eq!(measure_lines[0], &"M Q3");
    assert_eq!(measure_lines[1], &"M Q1");
    assert_eq!(measure_lines[2], &"M Q2");
    assert_eq!(measure_lines[3], &"M Q0");
}

/// Test measurement interleaved with gates.
#[test]
fn test_interleaved_measurement_order() {
    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
measure q[0] -> c[2];
y q[1];
measure q[1] -> c[0];
z q[2];
measure q[2] -> c[1];
"#;

    let circuit = qasm2_loads(qasm).expect("Failed to parse QASM");
    let qcis = qcis_dumps(&circuit).expect("Failed to convert to QCIS");

    let expected = r#"X Q0
M Q0
Y Q1
M Q1
Z Q2
M Q2
"#;
    assert_eq!(qcis, expected);

    // Verify measurement qubit order
    let lines: Vec<_> = qcis.lines().collect();
    let measure_lines: Vec<_> = lines.iter().filter(|line| line.starts_with("M ")).collect();

    assert_eq!(measure_lines[0], &"M Q0");
    assert_eq!(measure_lines[1], &"M Q1");
    assert_eq!(measure_lines[2], &"M Q2");
}
