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
use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::gate::{Directive, Instruction, StandardGate};
use crate::circuit::{Circuit, Qubit};
use crate::ir::qasm2::dump::dumps;

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
    // This test checks for circular gate dependency detection
    // gate recursive a { recursive a; } is a self-referential gate (circular dependency)
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
        err.contains("Circular gate dependency") || err.contains("recursion"),
        "Got error: {}",
        err
    );
}

#[test]
fn test_circular_gate_dependency() {
    // Test A calls B, B calls A (circular dependency)
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate a q {
            b q;
        }
        gate b q {
            a q;
        }
        a q[0];
    "#;
    let result = loads(qasm);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Circular gate dependency"),
        "Got error: {}",
        err
    );
}

#[test]
fn test_dumps() {
    let qs = (0..3).map(Qubit::new).collect::<Vec<_>>();
    let mut c = Circuit::new(3);
    c.h(qs[0]).unwrap();
    c.cx(qs[1], qs[2]).unwrap();
    let g = c.to_gate("g").unwrap();

    let qs = (0..4).map(Qubit::new).collect::<Vec<_>>();
    let mut c = Circuit::new(4);
    c.h(qs[0]).unwrap();
    c.append(g, vec![qs[1], qs[0], qs[3]], vec![], None)
        .unwrap();

    let qasm = dumps(&c);
    println!("{}", qasm.unwrap());

    // let c = load("/Users/gaojianjian/work/code/jianjian001/cqlib2/tests/qft_n18.qasm").unwrap();
    // let qasm = dumps(&c);
    // println!("{:?}", qasm);
}

#[test]
fn test_parameter_count_mismatch() {
    // Test U2 gate with only 1 parameter (should fail, needs 2)
    let qasm_u2_wrong = r#"
        OPENQASM 2.0;
        qreg q[1];
        u2(0.5) q[0];
    "#;

    let result = loads(qasm_u2_wrong);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Mismatched parameter count"),
        "Expected parameter count error, got: {}",
        err
    );

    // Test U3 gate with only 2 parameters (should fail, needs 3)
    let qasm_u3_wrong = r#"
        OPENQASM 2.0;
        qreg q[1];
        u3(0.5, 0.3) q[0];
    "#;

    let result = loads(qasm_u3_wrong);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Mismatched parameter count"),
        "Expected parameter count error, got: {}",
        err
    );

    // Test RX gate with no parameters (should fail, needs 1)
    let qasm_rx_wrong = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx q[0];
    "#;

    let result = loads(qasm_rx_wrong);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Mismatched parameter count"),
        "Expected parameter count error, got: {}",
        err
    );

    // Test U2 gate with correct 2 parameters (should succeed)
    let qasm_u2_correct = r#"
        OPENQASM 2.0;
        qreg q[1];
        u2(0.5, 0.3) q[0];
    "#;

    let result = loads(qasm_u2_correct);
    assert!(result.is_ok(), "U2 with 2 params should succeed");

    // Test U3 gate with correct 3 parameters (should succeed)
    let qasm_u3_correct = r#"
        OPENQASM 2.0;
        qreg q[1];
        u3(0.5, 0.3, 0.2) q[0];
    "#;

    let result = loads(qasm_u3_correct);
    assert!(result.is_ok(), "U3 with 3 params should succeed");
}

#[test]
fn test_if_statement() {
    // Test if statement with measurement
    let qasm_with_if = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) x q[1];
    "#;

    let result = loads(qasm_with_if);
    assert!(
        result.is_ok(),
        "If statement should succeed: {:?}",
        result.err()
    );
    let circuit = result.unwrap();

    // Verify the circuit has 3 operations: H, Measure, IfElse
    let ops = circuit.operations();
    assert_eq!(ops.len(), 3, "Expected 3 operations, got {}", ops.len());

    // Op 0: H gate on q[0]
    assert_standard_gate(&circuit, 0, StandardGate::H, &[0], &[]);

    // Op 1: Measure q[0] -> c[0]
    match &ops[1].instruction {
        Instruction::Directive(Directive::Measure) => {
            assert_eq!(ops[1].qubits.len(), 1);
            assert_eq!(ops[1].qubits[0], Qubit::new(0));
        }
        _ => panic!("Expected Measure directive, got {:?}", ops[1].instruction),
    }

    // Op 2: IfElse gate
    match &ops[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
            // Verify condition: qubit 0, target value 1
            let condition = if_else.condition();
            assert_eq!(condition.qubit, Qubit::new(0));
            assert_eq!(condition.target, 1);

            // Verify true_body has one operation: X on q[1]
            let true_body = if_else.true_body();
            assert_eq!(true_body.len(), 1, "Expected 1 operation in true_body");
            match &true_body[0].instruction {
                Instruction::Standard(StandardGate::X) => {
                    assert_eq!(true_body[0].qubits.len(), 1);
                    assert_eq!(true_body[0].qubits[0], Qubit::new(1));
                }
                _ => panic!(
                    "Expected X gate in true_body, got {:?}",
                    true_body[0].instruction
                ),
            }

            // Verify false_body is None
            assert!(if_else.false_body().is_none());
        }
        _ => panic!("Expected IfElse control flow, got {:?}", ops[2].instruction),
    }
}

#[test]
fn test_if_statement_without_measurement_error() {
    // Test if statement without prior measurement (should fail)
    let qasm_no_measure = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        if (c[0] == 1) x q[1];
    "#;

    let result = loads(qasm_no_measure);
    assert!(result.is_err(), "If without measurement should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("No measurement found"),
        "Expected measurement error, got: {}",
        err
    );
}

#[test]
fn test_if_statement_simple_creg_ref() {
    // Test if statement with simple creg reference (c == 1 means c[0] == 1)
    use crate::circuit::gate::control_flow::ControlFlow;

    let qasm_simple = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c == 1) x q[1];
    "#;

    let result = loads(qasm_simple);
    assert!(
        result.is_ok(),
        "Simple creg reference should work: {:?}",
        result.err()
    );
    let circuit = result.unwrap();

    // Verify the circuit has 3 operations
    let ops = circuit.operations();
    assert_eq!(ops.len(), 3, "Expected 3 operations");

    // Op 2: IfElse gate (same as above, since c == 1 is equivalent to c[0] == 1)
    match &ops[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
            let condition = if_else.condition();
            assert_eq!(condition.qubit, Qubit::new(0));
            assert_eq!(condition.target, 1);
        }
        _ => panic!("Expected IfElse control flow"),
    }
}

#[test]
fn test_if_statement_with_cx_gate() {
    // Test if statement with CX (controlled-X) gate
    use crate::circuit::gate::control_flow::ControlFlow;

    let qasm_cx = r#"
        OPENQASM 2.0;
        qreg q[3];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) cx q[1], q[2];
    "#;

    let result = loads(qasm_cx);
    assert!(
        result.is_ok(),
        "If with CX should succeed: {:?}",
        result.err()
    );
    let circuit = result.unwrap();

    let ops = circuit.operations();
    assert_eq!(ops.len(), 3);

    // Op 2: IfElse with CX
    match &ops[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
            let true_body = if_else.true_body();
            assert_eq!(true_body.len(), 1);
            match &true_body[0].instruction {
                Instruction::Standard(StandardGate::CX) => {
                    assert_eq!(true_body[0].qubits.len(), 2);
                    assert_eq!(true_body[0].qubits[0], Qubit::new(1));
                    assert_eq!(true_body[0].qubits[1], Qubit::new(2));
                }
                _ => panic!("Expected CX gate in true_body"),
            }
        }
        _ => panic!("Expected IfElse"),
    }
}

#[test]
fn test_if_statement_with_symbolic_params() {
    // Test if statement with symbolic parameters in the body
    // This verifies that parameters like 'theta' in rx(theta) are correctly handled
    use crate::circuit::circuit_param::CircuitParam;
    use crate::circuit::gate::control_flow::ControlFlow;

    let qasm_with_symbolic = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) rx(pi/2) q[1];
    "#;

    let result = loads(qasm_with_symbolic);
    assert!(
        result.is_ok(),
        "If with symbolic param should succeed: {:?}",
        result.err()
    );
    let circuit = result.unwrap();

    let ops = circuit.operations();
    assert_eq!(ops.len(), 3);

    // Op 2: IfElse with RX(pi/2)
    match &ops[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
            let true_body = if_else.true_body();
            assert_eq!(true_body.len(), 1, "Expected 1 operation in true_body");
            match &true_body[0].instruction {
                Instruction::Standard(StandardGate::RX) => {
                    assert_eq!(true_body[0].qubits.len(), 1);
                    assert_eq!(true_body[0].qubits[0], Qubit::new(1));

                    // Check parameter - should be Fixed(pi/2), not 0.0
                    assert_eq!(true_body[0].params.len(), 1, "Expected 1 parameter");
                    match &true_body[0].params[0] {
                        CircuitParam::Fixed(val) => {
                            // pi/2 ≈ 1.5708
                            assert!(
                                (val - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
                                "Expected pi/2 ({}), got {}",
                                std::f64::consts::FRAC_PI_2,
                                val
                            );
                        }
                        CircuitParam::Index(_) => {
                            // Index is also acceptable if the parameter was interned
                        }
                    }
                }
                _ => panic!(
                    "Expected RX gate in true_body, got {:?}",
                    true_body[0].instruction
                ),
            }
        }
        _ => panic!("Expected IfElse control flow"),
    }
}

#[test]
fn test_if_statement_with_unevaluated_symbolic_param() {
    // Test that unevaluated symbolic parameters don't become 0.0
    // This is a regression test for the issue where symbolic params were replaced with 0.0
    use crate::circuit::gate::control_flow::ControlFlow;

    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) rx(0.5) q[1];
    "#;

    let result = loads(qasm);
    assert!(result.is_ok(), "Should parse: {:?}", result.err());
    let circuit = result.unwrap();

    let ops = circuit.operations();
    match &ops[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
            let true_body = if_else.true_body();
            match &true_body[0].instruction {
                Instruction::Standard(StandardGate::RX) => {
                    match &true_body[0].params[0] {
                        CircuitParam::Fixed(val) => {
                            // Should be 0.5, not 0.0
                            assert!(
                                (val - 0.5).abs() < 1e-10,
                                "Parameter should be 0.5, got {} (regression: was incorrectly 0.0)",
                                val
                            );
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

#[test]
fn test_if_statement_with_true_symbolic_param() {
    // Test that true symbolic parameters (like undefined 'theta') are handled
    // This test documents current behavior: undefined symbols fall back to 0.0
    use crate::circuit::gate::control_flow::ControlFlow;

    // This qasm uses an undefined parameter 'theta'
    // According to OpenQASM spec, this should either:
    // 1. Error out (parameter not defined)
    // 2. Be handled as a symbolic parameter
    //
    // Current behavior: falls back to 0.0 (may need improvement)

    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) rx(theta) q[1];
    "#;

    let result = loads(qasm);
    // Currently this may either fail or succeed with theta=0.0
    // The test documents the current behavior
    if let Ok(circuit) = result {
        let ops = circuit.operations();
        match &ops[2].instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
                let true_body = if_else.true_body();
                if let Instruction::Standard(StandardGate::RX) = &true_body[0].instruction {
                    match &true_body[0].params[0] {
                        CircuitParam::Fixed(val) => {
                            // Document current behavior: undefined symbols become 0.0
                            // This is a known limitation that may be improved in the future
                            println!("Undefined symbolic parameter 'theta' evaluated to: {}", val);
                        }
                        CircuitParam::Index(_) => {
                            // If this is an Index, it means the parameter was properly interned
                            // which would be an improvement
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Test passes either way - it just documents the behavior
}

#[test]
fn test_if_statement_param_evaluation() {
    // Verify that parameters are correctly evaluated in if-body context
    use crate::circuit::gate::control_flow::ControlFlow;

    // Test with expressions that should evaluate correctly
    let test_cases = vec![
        ("pi/2", std::f64::consts::FRAC_PI_2),
        ("pi", std::f64::consts::PI),
        ("0.5", 0.5),
        ("1.0", 1.0),
        ("2*pi", 2.0 * std::f64::consts::PI),
    ];

    for (expr, expected) in test_cases {
        let qasm = format!(
            r#"
            OPENQASM 2.0;
            qreg q[2];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
            if (c[0] == 1) rx({}) q[1];
        "#,
            expr
        );

        let result = loads(&qasm);
        assert!(
            result.is_ok(),
            "Should parse rx({}): {:?}",
            expr,
            result.err()
        );

        let circuit = result.unwrap();
        let ops = circuit.operations();
        match &ops[2].instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
                let true_body = if_else.true_body();
                match &true_body[0].instruction {
                    Instruction::Standard(StandardGate::RX) => {
                        match &true_body[0].params[0] {
                            CircuitParam::Fixed(val) => {
                                assert!(
                                    (val - expected).abs() < 1e-9,
                                    "Expression '{}' should evaluate to {}, got {}",
                                    expr,
                                    expected,
                                    val
                                );
                            }
                            CircuitParam::Index(_) => {
                                // Index is acceptable if parameter was interned
                            }
                        }
                    }
                    _ => panic!("Expected RX gate"),
                }
            }
            _ => panic!("Expected IfElse"),
        }
    }
}

#[test]
fn test_if_statement_undefined_symbol_fails() {
    // OpenQASM 2.0 does not support global variables or parameters.
    // Using undefined symbols like 'theta' should be an error.
    let qasm_undefined = r#"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        if (c==1) rx(theta) q[0];
    "#;

    let result = loads(qasm_undefined);
    assert!(
        result.is_err(),
        "Undefined symbol 'theta' should cause an error"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Unknown parameter") || err.contains("Evaluation error"),
        "Expected 'Unknown parameter' or 'Evaluation error', got: {}",
        err
    );
}

#[test]
fn test_if_statement_multibit_register_fails() {
    // The backend only supports single-bit conditions.
    // Using a multi-bit register should fail with a clear error.
    let qasm_multibit = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c_reg[3];
        h q[0];
        measure q[0] -> c_reg[0];
        if (c_reg==1) x q[1];
    "#;

    let result = loads(qasm_multibit);
    assert!(
        result.is_err(),
        "Multi-bit register condition should cause an error"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("single-bit conditions") || err.contains("not supported"),
        "Expected error about single-bit conditions, got: {}",
        err
    );
}

#[test]
fn test_memory_resolver_include() {
    // Test the source resolver abstraction using a mock memory resolver
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    /// A mock resolver that serves files from memory
    struct MemoryResolver {
        files: HashMap<PathBuf, String>,
    }

    impl MemoryResolver {
        fn new() -> Self {
            let mut files = HashMap::new();
            // Add a mock include file
            files.insert(
                PathBuf::from("mylib.inc"),
                r#"
                gate my_gate a {
                    h a;
                    x a;
                }
                "#
                .to_string(),
            );
            Self { files }
        }
    }

    impl QasmSourceResolver for MemoryResolver {
        fn resolve_source(&self, path: &Path) -> Result<String, String> {
            self.files
                .get(path)
                .cloned()
                .ok_or_else(|| format!("File not found: {:?}", path))
        }
    }

    // Test parsing with the mock resolver
    let qasm_with_include = r#"
        OPENQASM 2.0;
        include "mylib.inc";
        qreg q[1];
        my_gate q[0];
    "#;

    // Use the internal parser with our mock resolver
    let resolver = Box::new(MemoryResolver::new());
    let result = parse_qasm_with_context(qasm_with_include, None, resolver);

    assert!(
        result.is_ok(),
        "Memory resolver should work: {:?}",
        result.err()
    );
    let circuit = result.unwrap();
    assert_eq!(circuit.num_qubits(), 1);
}

#[test]
fn test_null_resolver_recludes_includes() {
    // Test that NullResolver rejects include statements
    let qasm_with_include = r#"
        OPENQASM 2.0;
        include "somefile.inc";
        qreg q[1];
    "#;

    // loads uses NullResolver, so includes should fail
    let result = loads(qasm_with_include);
    assert!(result.is_err(), "Include in raw string mode should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("raw string mode") || err.contains("Cannot include"),
        "Expected error about raw string mode, got: {}",
        err
    );
}
