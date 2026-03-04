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

use crate::circuit::gate::circuit_gate::FrozenCircuit;
use crate::circuit::gate::{StandardGate, UnitaryGate};
use crate::circuit::parameter::Parameter;
use crate::circuit::{
    Circuit, ConditionView, ControlFlow, IfElseGate, Instruction, Operation, Qubit, WhileLoopGate,
};
use crate::ir::qasm2::dump::dumps;
use smallvec::smallvec;
use std::sync::Arc;

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
    let theta = Parameter::try_from("theta").unwrap();
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

    // When there are no conditional operations, no creg is generated
    // Measurement is commented out as there's no register to store to
    let expected = &[
        "reset q[0];",
        "barrier q[0],q[1];",
        "// measure q[0] -> c0[0];",
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
    let lambda = Parameter::try_from("lambda").unwrap();
    sub_circ.rx(sq0, lambda).unwrap();

    let gate = sub_circ.to_gate("my_rot").unwrap();

    let mut main_circ = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Use with fixed value
    main_circ
        .append(gate.clone(), vec![q0], vec![1.23.into()], None)
        .unwrap();

    // Use with symbolic value
    let gamma = Parameter::try_from("gamma").unwrap();
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
    let p = Parameter::try_from("p").unwrap();
    leaf.rz(Qubit::new(0), p).unwrap();
    let gate_leaf = leaf.to_gate("gate_leaf").unwrap();

    // 2. Define Middle Gate: calls gate_leaf(m * 2.0)
    let mut mid = Circuit::new(1);
    let m = Parameter::try_from("m").unwrap();
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
    let theta = Parameter::try_from("theta").unwrap();
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

#[test]
fn test_dump_unitary_gate_with_circuit() {
    // 1. Create an inner circuit for the UnitaryGate
    let mut inner_circuit = Circuit::new(2);
    inner_circuit.h(Qubit::new(0)).unwrap();
    inner_circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    // 2. Create a UnitaryGate with the inner circuit
    let frozen_circuit = FrozenCircuit::new(inner_circuit);
    let u_gate = UnitaryGate::new("MyBell", 2).with_circuit(Arc::new(frozen_circuit));

    // 3. Use the UnitaryGate in a main circuit
    let mut main_circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // Add some standard gates
    main_circuit.x(q0).unwrap();

    // Add the UnitaryGate
    main_circuit.unitary(u_gate, vec![q0, q1]).unwrap();

    // Add more gates after
    main_circuit.z(q1).unwrap();

    // 4. Dump and verify
    let qasm = dumps(&main_circuit).expect("Dump failed");
    println!("{}", qasm);

    // The UnitaryGate should be output as a gate definition and then called
    // Check gate definition
    assert_qasm_contains(&qasm, &["gate MyBell q0,q1 {", "h q0;", "cx q0,q1;", "}"]);

    // Check gate call
    assert_qasm_contains(&qasm, &["x q[0];", "MyBell q[0],q[1];", "z q[1];"]);
}

#[test]
fn test_dump_unitary_gate_with_circuit_nested() {
    // Test that UnitaryGate properly handles nested CircuitGate inside it
    // 1. Create a CircuitGate definition
    let mut sub_circ = Circuit::new(1);
    sub_circ.h(Qubit::new(0)).unwrap();
    sub_circ.s(Qubit::new(0)).unwrap();
    let gate_def = sub_circ.to_gate("hs_gate").unwrap();

    // 2. Create an inner circuit that uses the CircuitGate
    let mut inner_circuit = Circuit::new(2);
    inner_circuit
        .append(gate_def.clone(), vec![Qubit::new(0)], vec![], None)
        .unwrap();
    inner_circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    inner_circuit
        .append(gate_def, vec![Qubit::new(1)], vec![], None)
        .unwrap();

    // 3. Create a UnitaryGate with this inner circuit
    let frozen_circuit = FrozenCircuit::new(inner_circuit);
    let u_gate = UnitaryGate::new("CustomOp", 2).with_circuit(Arc::new(frozen_circuit));

    // 4. Use in main circuit
    let mut main_circuit = Circuit::new(2);
    main_circuit
        .unitary(u_gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();

    // 5. Dump and verify
    let qasm = dumps(&main_circuit).expect("Dump failed");

    // Should contain the CircuitGate definition
    assert_qasm_contains(&qasm, &["gate hs_gate q0 {", "h q0;", "s q0;", "}"]);

    // Should contain the UnitaryGate definition that calls the CircuitGate
    assert_qasm_contains(
        &qasm,
        &[
            "gate CustomOp q0,q1 {",
            "hs_gate q0;",
            "cx q0,q1;",
            "hs_gate q1;",
            "}",
        ],
    );

    // Should call the UnitaryGate
    assert_qasm_contains(&qasm, &["CustomOp q[0],q[1];"]);
}

#[test]
fn test_dump_unitary_gate_without_circuit() {
    // Test that UnitaryGate without circuit (only matrix) outputs opaque declaration
    use ndarray::Array2;
    use num_complex::Complex64;

    // Create a UnitaryGate with only matrix (no circuit)
    let mat = Array2::eye(2).mapv(|x| Complex64::new(x, 0.0));
    let u_gate = UnitaryGate::new("MatrixOnly", 1).with_matrix(mat).unwrap();

    let mut circuit = Circuit::new(1);
    circuit.unitary(u_gate, vec![Qubit::new(0)]).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");
    println!("{}", qasm);
    // Should output opaque declaration and gate call
    assert_qasm_contains(&qasm, &["opaque MatrixOnly q0;", "MatrixOnly q[0];"]);
}

#[test]
fn test_dump_extended_gates() {
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = 1.0;

    circuit.crx(q0, q1, theta).unwrap();
    circuit.cry(q0, q1, theta).unwrap();
    circuit.rzz(q0, q1, theta).unwrap();
    circuit.rxx(q0, q1, theta).unwrap();
    circuit.ryy(q0, q1, theta).unwrap();
    circuit.rzx(q0, q1, theta).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");
    println!("{}", qasm);

    // Check Definitions
    assert_qasm_contains(
        &qasm,
        &[
            "gate crx(theta) a,b { h b;rz(theta/2) b; cx a,b; rz(-theta/2) b; cx a,b; h b;}",
            "gate cry(theta) a,b { ry(theta/2) b; cx a,b; ry(-theta/2) b; cx a,b; }",
            "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }",
            "gate rxx(theta) a,b { h a; h b; cx a,b; rz(theta) b; cx a,b; h a; h b; }",
            "gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; cx a,b; rz(theta) b; cx a,b; rx(-pi/2) a; rx(-pi/2) b; }",
            "gate rzx(theta) a,b { h b; cx a,b; rz(theta) b; cx a,b; h b; }",
        ],
    );

    // Check Usages
    assert_qasm_contains(
        &qasm,
        &[
            "crx(1) q[0],q[1];",
            "cry(1) q[0],q[1];",
            "rzz(1) q[0],q[1];",
            "rxx(1) q[0],q[1];",
            "ryy(1) q[0],q[1];",
            "rzx(1) q[0],q[1];",
        ],
    );
}

#[test]
fn test_dump_if_statement() {
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // Build a circuit with if statement
    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    // Create if-else gate manually
    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    let if_else_gate = IfElseGate::new(condition, true_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![q0],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump should succeed");

    // Check that the output contains the if statement with OpenQASM 2.0 compliant format
    assert!(
        qasm.contains("creg c0[1];"),
        "Expected 'creg c0[1];' in output, got:\n{}",
        qasm
    );
    assert!(
        qasm.contains("measure q[0] -> c0[0];"),
        "Expected 'measure q[0] -> c0[0];' in output, got:\n{}",
        qasm
    );
    assert!(
        qasm.contains("if (c0 == 1) x q[1];"),
        "Expected 'if (c0 == 1) x q[1];' in output, got:\n{}",
        qasm
    );
}

#[test]
fn test_dump_if_statement_with_cx() {
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    // Create if-else gate with CX
    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![q1, q2],
        params: smallvec![],
        label: None,
    }];
    let if_else_gate = IfElseGate::new(condition, true_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![q0],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump should succeed");

    // Check that the output contains the if statement with OpenQASM 2.0 compliant format
    assert!(
        qasm.contains("creg c0[1];"),
        "Expected 'creg c0[1];' in output, got:\n{}",
        qasm
    );
    assert!(
        qasm.contains("if (c0 == 1) cx q[1],q[2];"),
        "Expected 'if (c0 == 1) cx q[1],q[2];' in output, got:\n{}",
        qasm
    );
}

#[test]
fn test_dump_simple_if_else() {
    // Create a circuit with if-else
    let mut circuit = Circuit::new(2);

    // Add H gate
    circuit.h(Qubit::new(0)).unwrap();

    // Add measurement
    circuit.measure(Qubit::new(0)).unwrap();

    // Add if-else: if (q0 == 1) x q1
    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let if_else_gate = IfElseGate::new(condition, true_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    // Dump to QASM
    let qasm_result = dumps(&circuit);
    assert!(
        qasm_result.is_ok(),
        "Dump should succeed: {:?}",
        qasm_result.err()
    );

    let qasm = qasm_result.unwrap();
    println!("Generated QASM:\n{}", qasm);

    // Verify the QASM contains OpenQASM 2.0 compliant if statement
    assert!(
        qasm.contains("creg c0[1];"),
        "QASM should declare single-bit register c0"
    );
    assert!(
        qasm.contains("measure q[0] -> c0[0];"),
        "QASM should measure to c0[0]"
    );
    assert!(
        qasm.contains("if (c0 == 1)"),
        "QASM should contain if statement with c0 (not c[0])"
    );
    assert!(qasm.contains("x q[1]"), "QASM should contain x q[1]");
}

#[test]
fn test_dump_if_else_detailed() {
    // Create a circuit with if-else with false branch
    let mut circuit = Circuit::new(2);

    circuit.h(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();

    // if (q0 == 1) x q1 else z q1
    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let if_else_gate = IfElseGate::new(condition, true_body, Some(false_body));
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Should dump");
    println!("Generated QASM with else branch:\n{}", qasm);

    // Verify OpenQASM 2.0 compliant format
    assert!(
        qasm.contains("creg c0[1];"),
        "QASM should declare single-bit register c0"
    );
    assert!(
        qasm.contains("if (c0 == 1) x q[1];"),
        "QASM should contain true branch with c0"
    );
    assert!(
        qasm.contains("if (c0 == 0) z q[1];"),
        "QASM should contain false branch with inverted condition"
    );
}

#[test]
fn test_dump_while_loop() {
    let mut circuit = Circuit::new(2);

    // while (q0 == 1) h q1
    let condition = ConditionView::new(Qubit::new(0), 1);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let while_gate = WhileLoopGate::new(condition, body);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_gate)),
            vec![Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    // While loop is not supported in OpenQASM 2.0 - should return error
    let result = dumps(&circuit);
    assert!(
        result.is_err(),
        "While loop dump should fail because OpenQASM 2.0 doesn't support it"
    );
}
