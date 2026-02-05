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

use super::*;
use crate::circuit::gate::{Directive, Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use crate::circuit::{Circuit, Qubit};

// --- Helper Functions ---

fn assert_standard_gate(
    circuit: &Circuit,
    op_idx: usize,
    expected_gate: StandardGate,
    expected_qubits: &[u32],
    expected_params: &[f64],
) {
    let ops = circuit.operations();
    assert!(
        op_idx < ops.len(),
        "Op index {} out of bounds (len {})",
        op_idx,
        ops.len()
    );
    let op = &ops[op_idx];

    match &op.instruction {
        Instruction::Standard(g) => assert_eq!(
            *g, expected_gate,
            "Op {}: Expected gate {:?}, found {:?}",
            op_idx, expected_gate, g
        ),
        _ => panic!(
            "Op {}: Expected StandardGate, found {:?}",
            op_idx, op.instruction
        ),
    }

    assert_eq!(
        op.qubits.len(),
        expected_qubits.len(),
        "Op {}: Qubit count mismatch",
        op_idx
    );
    for (i, &q_idx) in expected_qubits.iter().enumerate() {
        assert_eq!(
            op.qubits[i],
            Qubit::new(q_idx),
            "Op {}: Qubit mismatch at index {}",
            op_idx,
            i
        );
    }

    assert_eq!(
        op.params.len(),
        expected_params.len(),
        "Op {}: Param count mismatch",
        op_idx
    );
    for (i, &expected_val) in expected_params.iter().enumerate() {
        let val = match &op.params[i] {
            CircuitParam::Fixed(v) => *v,
            CircuitParam::Index(idx) => {
                let p = &circuit.parameters()[*idx as usize];
                p.evaluate(&None)
                    .unwrap_or_else(|_| panic!("Failed to evaluate param {}", idx))
            }
        };
        assert!(
            (val - expected_val).abs() < 1e-10,
            "Op {}: Param {} mismatch. Expected {}, got {}",
            op_idx,
            i,
            expected_val,
            val
        );
    }
}

fn assert_directive(
    circuit: &Circuit,
    op_idx: usize,
    expected_directive: Directive,
    expected_qubits: &[u32],
) {
    let ops = circuit.operations();
    assert!(op_idx < ops.len(), "Op index {} out of bounds", op_idx);
    let op = &ops[op_idx];

    match &op.instruction {
        Instruction::Directive(d) => assert_eq!(
            *d, expected_directive,
            "Op {}: Expected {:?}, found {:?}",
            op_idx, expected_directive, d
        ),
        _ => panic!(
            "Op {}: Expected Directive, found {:?}",
            op_idx, op.instruction
        ),
    }

    assert_eq!(
        op.qubits.len(),
        expected_qubits.len(),
        "Op {}: Qubit count mismatch",
        op_idx
    );
    for (i, &q_idx) in expected_qubits.iter().enumerate() {
        assert_eq!(
            op.qubits[i],
            Qubit::new(q_idx),
            "Op {}: Qubit mismatch at index {}",
            op_idx,
            i
        );
    }
}

// --- Tests ---

#[test]
fn test_parse_simple_qasm() {
    let qasm = r#"
            OPENQASM 2.0;
            qreg q[2];
            h q[0];
            cx q[0], q[1];
        "#;
    let result = loads(qasm);
    assert!(result.is_ok(), "Parse failed: {:?}", result.err());
    let circuit = result.unwrap();

    assert_eq!(circuit.num_qubits(), 2);
    assert_eq!(circuit.operations().len(), 2);

    assert_standard_gate(&circuit, 0, StandardGate::H, &[0], &[]);
    assert_standard_gate(&circuit, 1, StandardGate::CX, &[0, 1], &[]);
}

#[test]
fn test_scientific_notation_no_dot() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx(1e-5) q[0];
    "#;
    let result = loads(qasm);
    assert!(result.is_ok(), "Failed to parse 1e-5: {:?}", result.err());
    let circuit = result.unwrap();

    assert_standard_gate(&circuit, 0, StandardGate::RX, &[0], &[1e-5]);
}

#[test]
fn test_power_associativity() {
    // 2^3^2 should be 2^(3^2) = 512, not (2^3)^2 = 64
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx(2^3^2) q[0];
    "#;
    let result = loads(qasm);
    assert!(result.is_ok(), "Parse failed: {:?}", result.err());
    let circuit = result.unwrap();

    assert_standard_gate(&circuit, 0, StandardGate::RX, &[0], &[512.0]);
}

#[test]
fn test_identifier_case_compliance() {
    // OpenQASM 2.0 identifiers must start with lowercase
    let qasm = r#"
        OPENQASM 2.0;
        qreg Q[1];
    "#;
    let result = loads(qasm);
    assert!(
        result.is_err(),
        "Should fail for uppercase identifier start 'Q'"
    );
}

#[test]
fn test_uppercase_identifier() {
    // MyGate starts with uppercase, should fail
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate MyGate a { h a; }
        MyGate q[0];
    "#;
    let result = loads(qasm);
    assert!(
        result.is_err(),
        "Should fail for uppercase gate name 'MyGate'"
    );
}

#[test]
fn test_u_cx_special_cases() {
    // U and CX are allowed as exceptions
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        U(pi/2, 0, pi) q[0];
        CX q[0], q[1];
    "#;
    let result = loads(qasm);
    assert!(
        result.is_ok(),
        "Failed to parse U and CX: {:?}",
        result.err()
    );
    let circuit = result.unwrap();

    assert_standard_gate(
        &circuit,
        0,
        StandardGate::U,
        &[0],
        &[std::f64::consts::PI / 2.0, 0.0, std::f64::consts::PI],
    );
    assert_standard_gate(&circuit, 1, StandardGate::CX, &[0, 1], &[]);
}

#[test]
fn test_measure_reset() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        reset q[0];
        barrier q;
        measure q[0] -> c[0];
    "#;
    let result = loads(qasm);
    assert!(result.is_ok(), "Parse failed: {:?}", result.err());
    let circuit = result.unwrap();

    // 1. Reset q[0]
    assert_directive(&circuit, 0, Directive::Reset, &[0]);
    // 2. Barrier q[0], q[1] (q expanded)
    assert_directive(&circuit, 1, Directive::Barrier, &[0, 1]);
    // 3. Measure q[0]
    assert_directive(&circuit, 2, Directive::Measure, &[0]);
}

#[test]
fn test_param_gate() {
    let qasm = r#"
            OPENQASM 2.0;
            qreg q[1];
            gate my_rx(theta) a { rx(theta) a; }
            my_rx(3.14) q[0];
        "#;
    let result = loads(qasm);
    assert!(result.is_ok());

    let circuit = result.unwrap();
    let ops = circuit.operations();
    assert_eq!(ops.len(), 1);

    // Verify CircuitGate structure
    if let Instruction::CircuitGate(cg) = &ops[0].instruction {
        assert_eq!(cg.name.as_str(), "my_rx");

        let inner_ops = cg.circuit.circuit.operations();
        assert_eq!(inner_ops.len(), 1);
        if let Instruction::Standard(StandardGate::RX) = &inner_ops[0].instruction {
            // Check if parameter is mapped correctly (Index 0 in inner circuit)
            match &inner_ops[0].params[0] {
                CircuitParam::Index(idx) => assert_eq!(*idx, 0), // 0th param of inner circuit
                _ => panic!("Expected Index parameter in inner gate"),
            }
        } else {
            panic!("Inner gate should be RX");
        }
    } else {
        panic!("Top level gate should be CircuitGate");
    }
}

#[test]
fn test_nested_gate() {
    let qasm = r#"
            OPENQASM 2.0;
            qreg q[2];
            gate my_h a { h a; }
            gate my_hh a, b { my_h a; my_h b; }
            my_hh q[0], q[1];
        "#;
    let result = loads(qasm);
    assert!(result.is_ok(), "Nested gate failed: {:?}", result.err());
    let circuit = result.unwrap();
    let ops = circuit.operations();
    assert_eq!(ops.len(), 1);

    if let Instruction::CircuitGate(cg) = &ops[0].instruction {
        assert_eq!(cg.name.as_str(), "my_hh");
        let inner_ops = cg.circuit.circuit.operations();
        assert_eq!(inner_ops.len(), 2);
        // Verify inner ops are ALSO CircuitGates (my_h)
        for op in inner_ops {
            if let Instruction::CircuitGate(inner_cg) = &op.instruction {
                assert_eq!(inner_cg.name.as_str(), "my_h");
            } else {
                panic!("Expected inner gate to be CircuitGate(my_h)");
            }
        }
    } else {
        panic!("Top level gate should be CircuitGate(my_hh)");
    }
}

#[test]
fn test_recursion_limit() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate recursive a { recursive a; }
        recursive q[0];
    "#;
    let result = loads(qasm);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Recursion limit exceeded")
            || err.contains("stack overflow")
            || err.contains("recursion"),
        "Got error: {}",
        err
    );
}
