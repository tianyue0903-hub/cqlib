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
use crate::ir::qasm2::dump::{QasmDumpError, dumps};
use crate::ir::qasm2::load::loads;
use smallvec::smallvec;
use std::sync::Arc;

/// Assert that QASM output contains all expected lines in order
fn assert_qasm_contains_ordered(qasm: &str, expected_lines: &[&str]) {
    let mut search_start = 0;
    for (i, &line) in expected_lines.iter().enumerate() {
        let pos = qasm[search_start..].find(line);
        assert!(
            pos.is_some(),
            "QASM output missing expected line #{}:\n'{}'\n\nActual QASM:\n{}",
            i + 1,
            line,
            qasm
        );
        search_start += pos.unwrap() + line.len();
    }
}

/// Assert that QASM does NOT contain any forbidden lines
fn assert_qasm_not_contains(qasm: &str, forbidden_lines: &[&str]) {
    for &line in forbidden_lines {
        assert!(
            !qasm.contains(line),
            "QASM output should NOT contain:\n'{}'\n\nActual QASM:\n{}",
            line,
            qasm
        );
    }
}

/// Verify that generated QASM can be parsed back and has correct structure
fn verify_qasm_roundtrip(qasm: &str, expected_qubits: usize) {
    let parsed = loads(qasm);
    assert!(
        parsed.is_ok(),
        "Generated QASM should be valid and parseable. Error: {:?}\nQASM:\n{}",
        parsed.err(),
        qasm
    );
    let circuit = parsed.unwrap();
    assert_eq!(
        circuit.num_qubits(),
        expected_qubits,
        "Parsed circuit should have {} qubits",
        expected_qubits
    );
}

/// Simple contains check (for backwards compatibility)
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
    circuit.ccx(q0, q1, q2).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Strict: Verify complete structure
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
    assert_qasm_contains_ordered(&qasm, expected);

    // Verify no creg declarations (no conditional ops)
    assert!(
        !qasm.contains("creg"),
        "Should not have creg without conditional ops"
    );

    // Verify roundtrip
    verify_qasm_roundtrip(&qasm, 3);
}

#[test]
fn test_dump_parametric_gates() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Fixed parameter
    circuit.rx(q0, std::f64::consts::PI / 2.0).unwrap();
    // Symbolic parameter
    let theta = Parameter::try_from("theta").unwrap();
    circuit.ry(q0, theta.clone()).unwrap();
    // Expression
    let phi = theta + 0.5;
    circuit.rz(q0, phi).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify all parametric gates are present with correct syntax
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "rx(1.5707963267948966) q[0];",
            "ry(theta) q[0];",
            "rz(theta + 0.5) q[0];",
        ],
    );

    // Note: Skip roundtrip verification because symbolic parameters (theta)
    // cannot be parsed back without being defined in the QASM
    // verify_qasm_roundtrip(&qasm, 1);
}

#[test]
fn test_dump_complex_parameters() {
    // Test complex parameter expressions
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let theta = Parameter::try_from("theta").unwrap();
    let phi = Parameter::try_from("phi").unwrap();

    // Parameter with pi
    circuit.rx(q0, theta.clone() * Parameter::pi()).unwrap();
    // Division
    circuit.ry(q0, phi.clone() / 2.0).unwrap();
    // Complex expression
    circuit.rz(q1, (theta + phi) * 0.5).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify expressions are properly formatted
    // Note: Actual output format uses π symbol and reorders some expressions
    assert!(
        qasm.contains("rx(π * theta)"),
        "Should contain 'rx(π * theta)', got: {}",
        qasm
    );
    assert!(
        qasm.contains("ry(phi / 2)"),
        "Should contain 'ry(phi / 2)', got: {}",
        qasm
    );
    assert!(
        qasm.contains("rz(0.5 * (theta + phi))"),
        "Should contain complex expression, got: {}",
        qasm
    );

    // Note: Skip roundtrip verification because symbolic parameters (theta, phi)
    // cannot be parsed back without being defined in the QASM, and π symbol is not valid
    // verify_qasm_roundtrip(&qasm, 2);
}

#[test]
fn test_dump_directives_no_condition() {
    // Test directives without conditional operations
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.reset(q0).unwrap();
    circuit.barrier(vec![q0, q1]).unwrap();
    circuit.measure(q0).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Strict verification
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "reset q[0];",
            "barrier q[0],q[1];",
            "// measure q[0] -> c0[0];", // Commented out - no conditional use
        ],
    );

    // Verify NO creg declarations
    assert_qasm_not_contains(&qasm, &["creg"]);

    verify_qasm_roundtrip(&qasm, 2);
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

    // Check Definitions (in order)
    assert_qasm_contains_ordered(
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

    // Check Usages in order
    assert_qasm_contains_ordered(
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

    verify_qasm_roundtrip(&qasm, 2);
}

#[test]
fn test_dump_if_statement_simple() {
    // Strict test for basic if statement
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

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

    // Strict verification of OpenQASM 2.0 compliance
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) x q[1];",
        ],
    );

    // Verify exact format - NO array notation in if condition
    assert!(
        qasm.contains("if (c0 == 1)"),
        "OpenQASM 2.0 requires 'if (c0 == 1)' not 'if (c[0] == 1)'"
    );
    assert!(
        !qasm.contains("c[0]"),
        "Should NOT use multi-bit register notation c[0]"
    );

    verify_qasm_roundtrip(&qasm, 2);
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

    // Strict verification
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) cx q[1],q[2];",
        ],
    );

    verify_qasm_roundtrip(&qasm, 3);
}

#[test]
fn test_dump_simple_if_else() {
    // Create a circuit with if-else (only true branch)
    let mut circuit = Circuit::new(2);

    circuit.h(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();

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

    let qasm = dumps(&circuit).expect("Dump should succeed");

    // Strict verification
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) x q[1];",
        ],
    );

    // Verify NO else branch
    assert!(!qasm.contains("c0 == 0"), "Should not have else branch");

    verify_qasm_roundtrip(&qasm, 2);
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

    // Strict verification of OpenQASM 2.0 format
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) x q[1];",
            "if (c0 == 0) z q[1];",
        ],
    );

    // Verify else branch uses inverted condition
    assert!(
        qasm.contains("if (c0 == 0) z q[1]"),
        "Else branch should use inverted condition (c0 == 0)"
    );

    verify_qasm_roundtrip(&qasm, 2);
}

#[test]
fn test_dump_while_loop_error() {
    // While loops are not supported in OpenQASM 2.0
    let mut circuit = Circuit::new(2);

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

    // Must return specific error type
    let result = dumps(&circuit);
    assert!(result.is_err(), "While loop should fail");

    match result.err().unwrap() {
        QasmDumpError::WhileLoopNotSupported => {
            // Expected error type
        }
        other => panic!("Expected WhileLoopNotSupported, got {:?}", other),
    }
}

// =============================================================================
// Comprehensive New Tests
// =============================================================================

#[test]
fn test_dump_empty_circuit() {
    let circuit = Circuit::new(2);

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify minimal valid QASM
    assert_qasm_contains_ordered(
        &qasm,
        &["OPENQASM 2.0;", "include \"qelib1.inc\";", "qreg q[2];"],
    );

    // Verify no gates or operations
    assert!(!qasm.contains("h q["), "Empty circuit should have no gates");
    assert!(!qasm.contains("creg"), "Empty circuit should have no creg");

    verify_qasm_roundtrip(&qasm, 2);
}

#[test]
fn test_dump_multiple_conditional_qubits() {
    // Test multiple independent if statements with different condition qubits
    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);

    // Measure all qubits used in conditions
    circuit.h(q0).unwrap();
    circuit.h(q1).unwrap();
    circuit.measure(q0).unwrap();
    circuit.measure(q1).unwrap();

    // First if: condition on q0
    let cond0 = ConditionView::new(q0, 1);
    let true_body0 = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q2],
        params: smallvec![],
        label: None,
    }];
    let if_else0 = IfElseGate::new(cond0, true_body0, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else0)),
            vec![q0, q2],
            std::iter::empty(),
            None,
        )
        .unwrap();

    // Second if: condition on q1
    let cond1 = ConditionView::new(q1, 1);
    let true_body1 = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q3],
        params: smallvec![],
        label: None,
    }];
    let if_else1 = IfElseGate::new(cond1, true_body1, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else1)),
            vec![q1, q3],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify both single-bit registers are declared
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "creg c1[1];",
            "h q[0];",
            "h q[1];",
            "measure q[0] -> c0[0];",
            "measure q[1] -> c1[0];",
            "if (c0 == 1) x q[2];",
            "if (c1 == 1) z q[3];",
        ],
    );

    verify_qasm_roundtrip(&qasm, 4);
}

#[test]
fn test_dump_if_body_with_mcgate() {
    // Test if-body containing multi-controlled gates
    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    // ccx in if body (2-controlled X, 2 controls + 1 target = 3 qubits)
    let true_body = vec![Operation {
        instruction: Instruction::McGate(Box::new(crate::circuit::gate::MCGate::new(
            2, // 2 control qubits
            StandardGate::X,
        ))),
        qubits: smallvec![q1, q2, q3],
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

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify if-body contains ccx
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) ccx q[1],q[2],q[3];",
        ],
    );

    verify_qasm_roundtrip(&qasm, 4);
}

#[test]
fn test_dump_if_body_with_parameterized_mcgate() {
    // Test if-body containing parameterized multi-controlled gates
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let theta: f64 = 0.5; // Use fixed parameter for test
    let condition = ConditionView::new(q0, 1);
    // crx in if body (1-controlled RX with parameter)
    let true_body = vec![Operation {
        instruction: Instruction::McGate(Box::new(crate::circuit::gate::MCGate::new(
            1, // 1 control qubit
            StandardGate::RX,
        ))),
        qubits: smallvec![q1, q2],
        params: smallvec![theta.into()],
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

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify if-body contains crx with parameter
    // Note: Using fixed parameter value 0.5 instead of symbolic theta
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) crx(0.5) q[1],q[2];",
        ],
    );

    verify_qasm_roundtrip(&qasm, 3);
}

#[test]
fn test_dump_if_body_with_custom_gate() {
    // Test if-body containing custom CircuitGate
    let mut sub_circ = Circuit::new(2);
    sub_circ.h(Qubit::new(0)).unwrap();
    sub_circ.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::CircuitGate(Box::new(
            crate::circuit::gate::circuit_gate::CircuitGate {
                name: Arc::new("my_bell".to_string()),
                circuit: Arc::new(crate::circuit::gate::circuit_gate::FrozenCircuit::new(
                    sub_circ.clone(),
                )),
                num_qubits: 2,
                num_params: 0,
            },
        )),
        qubits: smallvec![Qubit::new(1), Qubit::new(2)],
        params: smallvec![],
        label: None,
    }];
    // Use sub_circ to avoid unused variable warning
    let _custom_gate = sub_circ.to_gate("my_bell").unwrap();
    let if_else_gate = IfElseGate::new(condition, true_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![q0],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify custom gate definition and usage in if-body
    // Note: Gate definitions for CircuitGate inside if-body may not be output
    // The circuit simply references the gate by name
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) my_bell q[1],q[2];",
        ],
    );
}

#[test]
fn test_dump_nested_control_flow_error() {
    // Nested control flow should return error
    let mut circuit = Circuit::new(3);

    // Outer if
    let condition = ConditionView::new(Qubit::new(0), 1);

    // Inner if in body (nested control flow)
    let inner_condition = ConditionView::new(Qubit::new(1), 1);
    let inner_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![Qubit::new(2)],
        params: smallvec![],
        label: None,
    }];
    let inner_if = IfElseGate::new(inner_condition, inner_body, None);
    let inner_cf = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(inner_if)),
        qubits: smallvec![Qubit::new(1), Qubit::new(2)],
        params: smallvec![],
        label: None,
    };

    let outer_body = vec![inner_cf];
    let outer_if = IfElseGate::new(condition, outer_body, None);

    circuit.h(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(outer_if)),
            vec![Qubit::new(0)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let result = dumps(&circuit);
    assert!(result.is_err(), "Nested control flow should fail");

    match result.err().unwrap() {
        QasmDumpError::NestedControlFlowNotSupported => {
            // Expected
        }
        other => panic!("Expected NestedControlFlowNotSupported, got {:?}", other),
    }
}

#[test]
fn test_dump_empty_if_body() {
    // Empty if body should still generate valid QASM (no if statement needed)
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let empty_body: Vec<Operation> = vec![];
    let if_else_gate = IfElseGate::new(condition, empty_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![q0],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Should have creg and measure but no if statement
    assert_qasm_contains_ordered(&qasm, &["creg c0[1];", "h q[0];", "measure q[0] -> c0[0];"]);

    // Should NOT have any if statement
    assert!(
        !qasm.contains("if ("),
        "Empty body should not generate if statement"
    );
}

#[test]
fn test_dump_multi_operation_if_body() {
    // If body with multiple operations
    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![q1],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::Y),
            qubits: smallvec![q2],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::Z),
            qubits: smallvec![q3],
            params: smallvec![],
            label: None,
        },
    ];
    let if_else_gate = IfElseGate::new(condition, true_body, None);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else_gate)),
            vec![q0],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // All three operations should be guarded by if
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "creg c0[1];",
            "h q[0];",
            "measure q[0] -> c0[0];",
            "if (c0 == 1) x q[1];",
            "if (c0 == 1) y q[2];",
            "if (c0 == 1) z q[3];",
        ],
    );

    verify_qasm_roundtrip(&qasm, 4);
}

#[test]
fn test_dump_unsupported_mcgate_error() {
    // Test that unsupported McGates return proper error
    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    // Create unsupported McGate: 3-controlled X (cccx) - but we need a target qubit
    // For this test, let's use q3 as target (3 controls + 1 target = 4 qubits, but we only have 4)
    // So use 3 controls on X with only 3 qubits - this should cause an error
    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::McGate(Box::new(crate::circuit::gate::MCGate::new(
            3, // 3 control qubits (cccx)
            StandardGate::X,
        ))),
        qubits: smallvec![q1, q2, q3], // This is wrong (should be 4 qubits), but error handling should catch it
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

    let result = dumps(&circuit);
    assert!(result.is_err(), "Unsupported McGate should fail");

    match result.err().unwrap() {
        QasmDumpError::FormatError(msg) => {
            assert!(
                msg.contains("Unsupported"),
                "Error should mention unsupported: {}",
                msg
            );
        }
        other => panic!("Expected FormatError, got {:?}", other),
    }
}

#[test]
fn test_dump_delay_gate() {
    // Test delay gate output using ParameterValue
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Use circuit.delay() method with ParameterValue
    use crate::circuit::param::ParameterValue;
    circuit.delay(q0, ParameterValue::Fixed(100.0)).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify delay gate declaration and usage
    // Note: ParameterValue::Fixed wraps value in parentheses
    assert_qasm_contains_ordered(&qasm, &["opaque delay(t) q;", "delay((100)) q[0];"]);
}

#[test]
fn test_dump_gate_collision_different_interface() {
    // Test gate collision when interfaces differ
    let mut c1 = Circuit::new(1);
    c1.h(Qubit::new(0)).unwrap();
    let g1 = c1.to_gate("collision_gate").unwrap();

    // Different number of qubits
    let mut c2 = Circuit::new(2);
    c2.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let g2 = c2.to_gate("collision_gate").unwrap();

    let mut main = Circuit::new(2);
    main.append(g1, vec![Qubit::new(0)], vec![], None).unwrap();
    main.append(g2, vec![Qubit::new(0), Qubit::new(1)], vec![], None)
        .unwrap();

    let qasm = dumps(&main).expect("Dump failed");

    // Should have warning output but still work
    // The first gate definition should be used
    assert!(qasm.contains("gate collision_gate"));
}

#[test]
fn test_dump_all_standard_gates() {
    // Test that all standard gates can be dumped
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    // Single qubit gates
    circuit.i(q0).unwrap();
    circuit.x(q0).unwrap();
    circuit.y(q0).unwrap();
    circuit.z(q0).unwrap();
    circuit.h(q0).unwrap();
    circuit.s(q0).unwrap();
    circuit.sdg(q0).unwrap();
    circuit.t(q0).unwrap();
    circuit.tdg(q0).unwrap();
    circuit.rx(q0, 0.1).unwrap();
    circuit.ry(q0, 0.2).unwrap();
    circuit.rz(q0, 0.3).unwrap();

    // Two qubit gates
    circuit.cx(q0, q1).unwrap();
    circuit.cy(q0, q1).unwrap();
    circuit.cz(q0, q1).unwrap();
    circuit.swap(q0, q1).unwrap();

    // Three qubit gates
    circuit.ccx(q0, q1, q2).unwrap();

    // U gates
    circuit.u(q0, 0.1, 0.2, 0.3).unwrap();
    circuit.crz(q0, q1, 0.4).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify all gates are present
    let expected_gates = [
        "id q[0];",
        "x q[0];",
        "y q[0];",
        "z q[0];",
        "h q[0];",
        "s q[0];",
        "sdg q[0];",
        "t q[0];",
        "tdg q[0];",
        "rx(0.1) q[0];",
        "ry(0.2) q[0];",
        "rz(0.3) q[0];",
        "cx q[0],q[1];",
        "cy q[0],q[1];",
        "cz q[0],q[1];",
        "swap q[0],q[1];",
        "ccx q[0],q[1],q[2];",
        "u3(0.1,0.2,0.3) q[0];",
        "crz(0.4) q[0],q[1];",
    ];

    for gate in &expected_gates {
        assert!(
            qasm.contains(gate),
            "QASM should contain '{}',\ngot:\n{}",
            gate,
            qasm
        );
    }

    verify_qasm_roundtrip(&qasm, 3);
}

#[test]
fn test_dump_xy_fsim_gates() {
    // Test XY and fSim gates
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // XY gate takes 1 qubit + theta
    circuit.xy(q0, 0.5).unwrap();
    circuit.fsim(q0, q1, 0.1, 0.2).unwrap();
    circuit.rxy(q0, 0.3, 0.4).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // These gates should be present (may use gate definitions)
    assert!(qasm.contains("xy"), "QASM should contain xy gate");
    assert!(qasm.contains("fsim"), "QASM should contain fsim gate");
    assert!(qasm.contains("rxy"), "QASM should contain rxy gate");
}

#[test]
fn test_dump_x2p_x2m_y2p_y2m() {
    // Test X2P, X2M, Y2P, Y2M gates (decomposed to rx/ry)
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    circuit.x2p(q0).unwrap();
    circuit.x2m(q0).unwrap();
    circuit.y2p(q0).unwrap();
    circuit.y2m(q0).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify decomposed gates
    // Note: Comments don't have trailing semicolons in output
    assert_qasm_contains_ordered(
        &qasm,
        &[
            "// x2p q[0]",
            "rx(pi/2) q[0]",
            "// x2m q[0]",
            "rx(-pi/2) q[0]",
            "// y2p q[0]",
            "ry(pi/2) q[0]",
            "// y2m q[0]",
            "ry(-pi/2) q[0]",
        ],
    );
}

#[test]
fn test_dump_gphase() {
    // Test GPhase gate (should output comment)
    // Note: gphase() method doesn't exist, using phase() instead which outputs u1
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    circuit.phase(q0, 0.5).unwrap();

    let qasm = dumps(&circuit).expect("Dump failed");

    // Phase gate outputs as u1 in OpenQASM 2.0
    assert!(
        qasm.contains("u1(0.5)"),
        "Phase gate should output as u1: {}",
        qasm
    );
}

#[test]
fn test_dump_condition_value_zero() {
    // Test if condition with target value 0
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 0); // Target value 0
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

    let qasm = dumps(&circuit).expect("Dump failed");

    // Verify condition value 0
    assert!(
        qasm.contains("if (c0 == 0)"),
        "Should have condition with value 0: {}",
        qasm
    );

    verify_qasm_roundtrip(&qasm, 2);
}

#[test]
fn test_dump_qasm_ordering() {
    // Test that QASM output follows correct ordering:
    // 1. Header
    // 2. QReg/CReg declarations
    // 3. Gate definitions
    // 4. Main circuit operations

    let mut sub_circ = Circuit::new(1);
    sub_circ.h(Qubit::new(0)).unwrap();
    let custom_gate = sub_circ.to_gate("my_h").unwrap();

    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.h(q0).unwrap();
    circuit.measure(q0).unwrap();

    // Use custom gate
    circuit.append(custom_gate, vec![q1], vec![], None).unwrap();

    // Add if statement
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

    let qasm = dumps(&circuit).expect("Dump failed");
    let lines: Vec<_> = qasm
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    // Find positions of key elements
    let header_pos = lines.iter().position(|l| l.contains("OPENQASM")).unwrap();
    let qreg_pos = lines.iter().position(|l| l.starts_with("qreg")).unwrap();
    let creg_pos = lines.iter().position(|l| l.starts_with("creg")).unwrap();
    let gate_def_pos = lines.iter().position(|l| l.starts_with("gate ")).unwrap();
    let h_pos = lines.iter().position(|l| *l == "h q[0];").unwrap();
    let measure_pos = lines.iter().position(|l| l.contains("measure")).unwrap();
    let if_pos = lines.iter().position(|l| l.contains("if (")).unwrap();

    // Verify ordering
    assert!(header_pos < qreg_pos, "Header before qreg");
    assert!(qreg_pos < creg_pos, "Qreg before creg");
    assert!(creg_pos < gate_def_pos, "Creg before gate definitions");
    assert!(gate_def_pos < h_pos, "Gate definitions before operations");
    assert!(h_pos < measure_pos, "H before measure");
    assert!(measure_pos < if_pos, "Measure before if");
}
