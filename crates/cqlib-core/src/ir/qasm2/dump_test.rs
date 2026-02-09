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

use crate::circuit::gate::StandardGate;
use crate::circuit::parameter::Parameter;
use crate::circuit::{Circuit, Qubit};
use crate::ir::qasm2::dump::dumps;

fn assert_qasm_contains(qasm: &str, expected_lines: &[&str]) {
    for &line in expected_lines {
        assert!(
            qasm.contains(line),
            "QASM output missing expected line:\n'{}'\n\nActual QASM:\n{}",
            line,
            qasm
        );
    }
}

#[test]
fn test_dump_standard_gates() {
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    circuit.h(q0).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.x(q1).unwrap();
    circuit.y(q2).unwrap();
    circuit.z(q0).unwrap();
    circuit.ccx(q0, q1, q2).unwrap(); // Toffoli

    let qasm = dumps(&circuit).expect("Dump failed");

    let expected = &[
        "OPENQASM 2.0;",
        "include \"qelib1.inc\";",
        "qreg q[3];",
        "h q[0];",
        "cx q[0],q[1];",
        "x q[1];",
        "y q[2];",
        "z q[0];",
        "ccx q[0],q[1],q[2];",
    ];
    assert_qasm_contains(&qasm, expected);
}

#[test]
fn test_dump_parametric_gates() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Fixed parameter
    circuit.rx(q0, 1.57).unwrap(); // ~pi/2
    // Symbolic parameter
    let theta = Parameter::from("theta");
    circuit.ry(q0, theta.clone()).unwrap();
    // Expression
    let phi = theta + 0.5;
    circuit.rz(q0, phi).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    let expected = &["rx(1.57) q[0];", "ry(theta) q[0];", "rz(theta + 0.5) q[0];"];
    assert_qasm_contains(&qasm, expected);
}

#[test]
fn test_dump_directives() {
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.reset(q0).unwrap();
    circuit.barrier(vec![q0, q1]).unwrap();
    circuit.measure(q0).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    let expected = &[
        "reset q[0];",
        "barrier q[0],q[1];",
        "creg c[2];", // Default classical register
        "measure q[0] -> c[0];",
    ];
    assert_qasm_contains(&qasm, expected);
}

#[test]
fn test_dump_custom_gate() {
    // Define a sub-circuit
    let mut sub_circ = Circuit::new(2);
    let sq0 = Qubit::new(0);
    let sq1 = Qubit::new(1);
    sub_circ.h(sq0).unwrap();
    sub_circ.cx(sq0, sq1).unwrap();

    // Create a gate from it
    let gate = sub_circ.to_gate("bell").unwrap();

    // Use it in main circuit
    let mut main_circ = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // Append requires manual building of the instruction call,
    // but `Circuit::append` handles the wrapping.
    main_circ.append(gate, vec![q0, q1], vec![], None).unwrap();

    let qasm = dumps(&main_circ).expect("Dump failed");

    // Check Definition
    let expected_def = &["gate bell q0,q1 {", "h q0;", "cx q0,q1;", "}"];
    assert_qasm_contains(&qasm, expected_def);

    // Check Usage
    assert_qasm_contains(&qasm, &["bell q[0],q[1];"]);
}

#[test]
fn test_dump_parameterized_custom_gate() {
    // Sub-circuit with parameter
    let mut sub_circ = Circuit::new(1);
    let sq0 = Qubit::new(0);
    let lambda = Parameter::from("lambda");
    sub_circ.rx(sq0, lambda).unwrap();

    let gate = sub_circ.to_gate("my_rot").unwrap();

    let mut main_circ = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Use with fixed value
    main_circ
        .append(gate.clone(), vec![q0], vec![1.23.into()], None)
        .unwrap();

    // Use with symbolic value
    let gamma = Parameter::from("gamma");
    main_circ
        .append(gate, vec![q0], vec![gamma.into()], None)
        .unwrap();

    let qasm = dumps(&main_circ).expect("Dump failed");

    // Definition should use the parameter name from the sub-circuit
    let expected_def = &["gate my_rot(lambda) q0 {", "rx(lambda) q0;", "}"];
    assert_qasm_contains(&qasm, expected_def);

    // Usages
    let expected_usage = &["my_rot(1.23) q[0];", "my_rot(gamma) q[0];"];
    assert_qasm_contains(&qasm, expected_usage);
}

#[test]
fn test_global_phase() {
    let circuit = Circuit::new(1);

    let qasm = dumps(&circuit).expect("Dump failed");
    assert!(!qasm.contains("// Global Phase: 0"));
}

#[test]
fn test_dump_mc_gate() {
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    // Multi-controlled X with 2 controls -> ccx
    circuit
        .multi_control(StandardGate::X, vec![q0, q1], vec![q2], vec![])
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");
    assert_qasm_contains(&qasm, &["ccx q[0],q[1],q[2];"]);
}

#[test]
fn test_gate_collision_behavior() {
    // Define Gate A: H gate
    let mut c1 = Circuit::new(1);
    c1.h(Qubit::new(0)).unwrap();
    let g1 = c1.to_gate("my_gate").unwrap();

    // Define Gate B: X gate (same name "my_gate")
    let mut c2 = Circuit::new(1);
    c2.x(Qubit::new(0)).unwrap();
    let g2 = c2.to_gate("my_gate").unwrap();

    let mut main = Circuit::new(1);
    let q0 = Qubit::new(0);

    main.append(g1, vec![q0], vec![], None).unwrap();
    main.append(g2, vec![q0], vec![], None).unwrap();

    let qasm = dumps(&main).expect("Dump failed");

    assert!(qasm.contains("gate my_gate q0 {\nh q0;"));
    assert!(!qasm.contains("x q0;")); // The definition body shouldn't have x

    let matches: Vec<_> = qasm.match_indices("my_gate q[0];").collect();
    assert_eq!(matches.len(), 2);
}

#[test]
fn test_dump_nested_custom_gates() {
    // 1. Define Leaf Gate: rz(p) q
    let mut leaf = Circuit::new(1);
    let p = Parameter::from("p");
    leaf.rz(Qubit::new(0), p).unwrap();
    let gate_leaf = leaf.to_gate("gate_leaf").unwrap();

    // 2. Define Middle Gate: calls gate_leaf(m * 2.0)
    let mut mid = Circuit::new(1);
    let m = Parameter::from("m");
    let param_expr: Parameter = m.clone() * 2.0;
    mid.append(
        gate_leaf,
        vec![Qubit::new(0)],
        vec![param_expr.into()],
        None,
    )
    .unwrap();
    let gate_mid = mid.to_gate("gate_mid").unwrap();

    // 3. Main Circuit: calls gate_mid(theta)
    let mut main = Circuit::new(1);
    let theta = Parameter::from("theta");
    main.append(gate_mid, vec![Qubit::new(0)], vec![theta.into()], None)
        .unwrap();

    let qasm = dumps(&main).expect("Dump failed");

    // Expected Output Analysis:
    // 1. gate gate_leaf(p) q0 { rz(p) q0; }
    // 2. gate gate_mid(m) q0 { gate_leaf(m * 2) q0; }  <-- Note: 2.0 might be '2' or '2.0' depending on formatting
    // 3. gate_mid(theta) q[0];

    assert_qasm_contains(
        &qasm,
        &[
            "gate gate_leaf(p) q0 {",
            "rz(p) q0;",
            "}",
            "gate gate_mid(m) q0 {",
            "gate_leaf(2 * m) q0;", // Simplified check, might need regex if float formatting varies
            "}",
            "gate_mid(theta) q[0];",
        ],
    );
}
