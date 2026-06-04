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

use crate::circuit::Qubit;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::circuit_param::{CircuitParam, ParameterValue};
use crate::circuit::error::CircuitError;
use crate::circuit::gate::control_flow::ConditionView;
use crate::circuit::gate::{Instruction, StandardGate, UnitaryGate};
use crate::circuit::operation::{Operation, ValueOperation};
use crate::circuit::parameter::Parameter;
use smallvec::smallvec;
use std::collections::HashSet;
use std::f64::consts::PI;

#[test]
fn test_circuit_basic_construction() {
    let mut circuit = Circuit::new(3);
    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.width(), 3);

    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    // Test simple append
    let _ = circuit.h(q0);
    let _ = circuit.cx(q0, q1);
    let _ = circuit.rx(q2, 1.5);

    assert_eq!(circuit.data.len(), 3);

    // Check operation details
    let op0 = &circuit.data[0];
    assert!(matches!(
        op0.instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(op0.qubits[0], q0);

    let op1 = &circuit.data[1];
    assert!(matches!(
        op1.instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    let expected: smallvec::SmallVec<[Qubit; 3]> = smallvec![q0, q1];
    assert_eq!(op1.qubits, expected);
}

#[test]
fn test_circuit_qubit_validation() {
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q3 = Qubit::new(3); // Does not exist

    let res = circuit.h(q3);
    assert!(matches!(res, Err(CircuitError::QubitNotFound(3))));

    let res = circuit.cx(q0, q3);
    assert!(matches!(res, Err(CircuitError::QubitNotFound(3))));
}

#[test]
fn test_from_qubits_and_add() {
    let q0 = Qubit::new(0);
    let q2 = Qubit::new(2);

    // Non-contiguous qubits
    let mut circuit = Circuit::from_qubits(vec![q0, q2]).unwrap();
    assert_eq!(circuit.num_qubits(), 2);
    assert!(circuit.qubits.contains(&q0));
    assert!(circuit.qubits.contains(&q2));
    assert!(!circuit.qubits.contains(&Qubit::new(1)));

    // Duplicate check
    let res = Circuit::from_qubits(vec![q0, q0]);
    assert!(matches!(res, Err(CircuitError::DuplicateQubits)));

    // Add qubits
    let q1 = Qubit::new(1);
    circuit.add_qubits(vec![q1]).unwrap();
    assert_eq!(circuit.num_qubits(), 3);
    assert!(circuit.qubits.contains(&q1));

    // Add duplicate
    let res = circuit.add_qubits(vec![q0]);
    assert!(matches!(res, Err(CircuitError::DuplicateQubits)));
}

#[test]
fn from_operations_builds_circuit_and_interns_symbolic_parameters() {
    let theta = Parameter::symbol("theta");
    let operations = vec![
        ValueOperation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(2)],
            params: smallvec![],
            label: Some("prepare".into()),
        },
        ValueOperation {
            instruction: Instruction::Standard(StandardGate::RX),
            qubits: smallvec![Qubit::new(4)],
            params: smallvec![ParameterValue::Param(theta.clone())],
            label: None,
        },
        ValueOperation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(2)],
            params: smallvec![ParameterValue::Fixed(0.25)],
            label: None,
        },
    ];

    let circuit = Circuit::from_operations(vec![Qubit::new(2), Qubit::new(4)], operations).unwrap();

    assert_eq!(circuit.qubits(), vec![Qubit::new(2), Qubit::new(4)]);
    assert_eq!(circuit.operations().len(), 3);
    assert_eq!(circuit.operations()[0].label.as_deref(), Some("prepare"));
    assert_eq!(circuit.parameters().len(), 1);
    assert!(circuit.parameters().contains(&theta));
    assert!(circuit.symbols().contains("theta"));
    assert!(matches!(
        circuit.operations()[1].params.as_slice(),
        [CircuitParam::Index(0)]
    ));
    assert!(matches!(
        circuit.operations()[2].params.as_slice(),
        [CircuitParam::Fixed(value)] if value.to_bits() == 0.25f64.to_bits()
    ));
}

#[test]
fn from_operations_rejects_duplicate_qubit_declarations() {
    let result = Circuit::from_operations(vec![Qubit::new(0), Qubit::new(0)], Vec::new());

    assert!(matches!(result, Err(CircuitError::DuplicateQubits)));
}

#[test]
fn from_operations_rejects_unknown_operation_qubits() {
    let operations = vec![ValueOperation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];

    let result = Circuit::from_operations(vec![Qubit::new(0)], operations);

    assert!(matches!(result, Err(CircuitError::QubitNotFound(1))));
}

#[test]
fn test_multi_control_logic() {
    let mut circuit = Circuit::new(4);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);

    // X -> CX (1 control)
    circuit
        .multi_control(StandardGate::X, [q0], [q1], [])
        .unwrap();
    let op = &circuit.data[0];
    assert!(matches!(
        op.instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    let expected: smallvec::SmallVec<[Qubit; 3]> = smallvec![q0, q1];
    assert_eq!(op.qubits, expected);

    // X -> CCX (2 controls)
    circuit
        .multi_control(StandardGate::X, [q0, q1], [q2], [])
        .unwrap();
    let op = &circuit.data[1];
    assert!(matches!(
        op.instruction,
        Instruction::Standard(StandardGate::CCX)
    ));
    let expected: smallvec::SmallVec<[Qubit; 3]> = smallvec![q0, q1, q2];
    assert_eq!(op.qubits, expected);

    // X -> MCX (3 controls, extended)
    circuit
        .multi_control(StandardGate::X, [q0, q1, q2], [q3], [])
        .unwrap();
    let op = &circuit.data[2];
    if let Instruction::McGate(ext) = &op.instruction {
        assert_eq!(ext.base_gate(), &StandardGate::X);
        assert_eq!(ext.num_ctrl_qubits(), 3);
    } else {
        panic!("Expected Extended instruction for 3-control X");
    }
}

#[test]
fn test_parametric_operations() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    // Fixed parameter
    circuit.rx(q0, PI).unwrap();
    if let CircuitParam::Fixed(val) = circuit.data[0].params[0] {
        assert!((val - PI).abs() < 1e-10);
    } else {
        panic!("Expected fixed parameter");
    }

    // Symbolic parameter
    let theta = Parameter::symbol("theta");
    circuit.rz(q0, theta.clone()).unwrap();

    // Check it was interned
    assert_eq!(circuit.parameters.len(), 1);
    assert!(circuit.symbols.contains("theta"));

    if let CircuitParam::Index(idx) = circuit.data[1].params[0] {
        assert_eq!(idx, 0);
    } else {
        panic!("Expected indexed parameter");
    }
}

#[test]
fn test_inverse_basic() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);

    circuit.h(q0).unwrap();
    circuit.s(q0).unwrap(); // S -> Sdg

    let inv_circuit = circuit.inverse().unwrap();

    assert_eq!(inv_circuit.data.len(), 2);

    // Order should be reversed: Sdg then H
    let op0 = &inv_circuit.data[0];
    assert!(matches!(
        op0.instruction,
        Instruction::Standard(StandardGate::SDG)
    ));

    let op1 = &inv_circuit.data[1];
    assert!(matches!(
        op1.instruction,
        Instruction::Standard(StandardGate::H)
    )); // H inv is H
}

#[test]
fn test_inverse_parametric() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);
    let theta = Parameter::symbol("theta");

    // RX(theta)
    circuit.rx(q0, theta.clone()).unwrap();

    let inv_circuit = circuit.inverse().unwrap();
    let op = &inv_circuit.data[0];

    // Should be RX(-theta)
    assert!(matches!(
        op.instruction,
        Instruction::Standard(StandardGate::RX)
    ));

    if let CircuitParam::Index(idx) = op.params[0] {
        let _ = &inv_circuit.parameters[idx as usize];
    } else {
        panic!("Expected indexed parameter for inverted rotation");
    }
}

#[test]
fn test_to_gate_and_circuit_gate() {
    // 1. Define a sub-circuit: H(q0), CX(q0, q1)
    let mut sub = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    sub.h(q0).unwrap();
    sub.cx(q0, q1).unwrap();

    // 2. Convert to Gate
    let bell_gate_inst = sub.to_gate("bell").unwrap();

    // 3. Use in main circuit
    let mut main = Circuit::new(3);
    // Apply bell gate to q1, q2
    // to_gate produced an instruction, we need to append it
    // Note: append expects params. bell_gate has 0 params.
    main.append(
        bell_gate_inst.clone(),
        [Qubit::new(1), Qubit::new(2)],
        [],
        Some("my_bell"),
    )
    .unwrap();

    assert_eq!(main.data.len(), 1);
    if let Instruction::CircuitGate(gate) = &main.data[0].instruction {
        assert_eq!(gate.name.as_ref(), "bell");
        assert_eq!(gate.num_qubits(), 2);
        assert_eq!(gate.num_params(), 0);
    } else {
        panic!("Expected Circuit instruction");
    }
}

#[test]
fn test_assign_parameters() {
    use std::collections::HashMap;

    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let a = Parameter::symbol("a");
    let b = Parameter::symbol("b");

    // rx(a) q0
    circuit.rx(q0, a.clone()).unwrap();
    // rz(a + b) q0
    circuit.rz(q0, a.clone() + b.clone()).unwrap();

    // Case 1: Partial assignment a = 1.0
    // Expected:
    // rx(1.0) q0 -> Fixed(1.0)
    // rz(1.0 + b) q0 -> Index(new_param)
    let mut bindings = HashMap::new();
    bindings.insert("a", 1.0);

    let assigned_circuit = circuit.assign_parameters(&Some(bindings)).unwrap();

    assert_eq!(assigned_circuit.data.len(), 2);

    // Check first op: rx(1.0)
    if let CircuitParam::Fixed(val) = assigned_circuit.data[0].params[0] {
        assert!((val - 1.0).abs() < 1e-10);
    } else {
        panic!(
            "Expected Fixed(1.0) for rx, got {:?}",
            assigned_circuit.data[0].params[0]
        );
    }

    // Check second op: rz(1.0 + b)
    if let CircuitParam::Index(idx) = assigned_circuit.data[1].params[0] {
        let param = &assigned_circuit.parameters[idx as usize];
        let symbols = param.get_symbols();
        assert!(symbols.contains(&"b".to_string()));
        assert!(!symbols.contains(&"a".to_string()));
    } else {
        panic!(
            "Expected Index for rz, got {:?}",
            assigned_circuit.data[1].params[0]
        );
    }

    // Case 2: Full assignment a = 1.0, b = 2.0
    let mut bindings = HashMap::new();
    bindings.insert("a", 1.0);
    bindings.insert("b", 2.0);

    let assigned_circuit = circuit.assign_parameters(&Some(bindings)).unwrap();

    // Check second op: rz(1.0 + 2.0) -> rz(3.0) -> Fixed(3.0)
    if let CircuitParam::Fixed(val) = assigned_circuit.data[1].params[0] {
        assert!((val - 3.0).abs() < 1e-10);
    } else {
        panic!(
            "Expected Fixed(3.0) for rz, got {:?}",
            assigned_circuit.data[1].params[0]
        );
    }
}

#[test]
fn test_parameterized_unitary_appends_fixed_and_symbolic_params() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    let gate = UnitaryGate::new("CustomU", 1, 2);

    circuit
        .unitary_with_params(
            gate,
            vec![Qubit::new(0)],
            vec![ParameterValue::Fixed(0.25), ParameterValue::Param(theta)],
        )
        .unwrap();

    assert_eq!(circuit.data.len(), 1);
    assert!(matches!(
        circuit.data[0].instruction,
        Instruction::UnitaryGate(_)
    ));
    assert!(matches!(
        circuit.data[0].params[0],
        CircuitParam::Fixed(0.25)
    ));
    assert!(matches!(circuit.data[0].params[1], CircuitParam::Index(_)));
    assert_eq!(circuit.parameters().len(), 1);
    assert!(circuit.symbols().contains("theta"));
}

#[test]
fn test_parameterized_unitary_validates_param_count() {
    let mut circuit = Circuit::new(1);
    let gate = UnitaryGate::new("CustomU", 1, 2);

    let err = circuit
        .unitary_with_params(gate.clone(), vec![Qubit::new(0)], vec![0.1.into()])
        .unwrap_err();
    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 2,
            actual: 1
        }
    ));

    let err = circuit.unitary(gate, vec![Qubit::new(0)]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 2,
            actual: 0
        }
    ));
}

#[test]
fn test_decompose() {
    use std::collections::HashMap;

    let mut inner = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");
    let beta = Parameter::symbol("beta");

    inner.h(q0).unwrap();
    inner.rx(q0, theta).unwrap();
    inner.rz(q1, beta.clone() + 1.0).unwrap();

    let gate = inner.to_gate("c1").unwrap();

    let mut outer = Circuit::new(2);
    let qa = Qubit::new(0);
    let qb = Qubit::new(1);
    let gamma = Parameter::symbol("gamma");
    let delta = Parameter::symbol("delta");

    outer
        .append(
            gate,
            vec![qb, qa],
            vec![
                ParameterValue::Param(gamma.clone()),
                ParameterValue::Param(gamma.clone() + delta.clone()),
            ],
            None,
        )
        .unwrap();

    // 3. Decompose
    let decomposed = outer.decompose().unwrap();

    // 4. Verify
    assert_eq!(decomposed.num_qubits(), 2);
    assert_eq!(
        decomposed.symbols().get_index(0),
        Some(&"gamma".to_string())
    );
    assert_eq!(
        decomposed.symbols().get_index(1),
        Some(&"delta".to_string())
    );

    assert_eq!(decomposed.data.len(), 3);

    assert!(matches!(
        decomposed.data[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(decomposed.data[0].qubits[0], qb);

    assert!(matches!(
        decomposed.data[1].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert_eq!(decomposed.data[1].qubits[0], qb);
    if let CircuitParam::Index(idx) = decomposed.data[1].params[0] {
        let p = &decomposed.parameters[idx as usize];
        // Should evaluate to same as gamma with bindings
        let mut bind = HashMap::new();
        bind.insert("gamma", 2.0);
        bind.insert("delta", 3.0);
        assert_eq!(p.evaluate(&Some(bind)).unwrap(), 2.0);
    } else {
        panic!("Expected Index param for RX");
    }

    // Op 3: RZ(gamma+delta+1, qA)
    assert!(matches!(
        decomposed.data[2].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert_eq!(decomposed.data[2].qubits[0], qa);
    if let CircuitParam::Index(idx) = decomposed.data[2].params[0] {
        let p = &decomposed.parameters[idx as usize];
        // gamma=2, delta=3 -> 2+3+1 = 6.0
        let mut bind = HashMap::new();
        bind.insert("gamma", 2.0);
        bind.insert("delta", 3.0);
        assert_eq!(p.evaluate(&Some(bind)).unwrap(), 6.0);
    } else {
        panic!("Expected Index param for RZ");
    }
}

#[test]
fn test_decompose_nested() {
    let mut l1 = Circuit::new(1);
    l1.h(Qubit::new(0)).unwrap();
    let g1 = l1.to_gate("g1").unwrap();

    let mut l2 = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    l2.rx(Qubit::new(0), theta.clone()).unwrap();
    l2.append(g1, [Qubit::new(0)], [], None).unwrap();
    let g2 = l2.to_gate("g2").unwrap();

    let mut top = Circuit::new(1);
    let phi = Parameter::symbol("phi");
    top.append(g2, [Qubit::new(0)], [ParameterValue::Param(phi)], None)
        .unwrap();

    let flat = top.decompose().unwrap();
    assert_eq!(flat.data.len(), 2);

    assert!(matches!(
        flat.data[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    if let CircuitParam::Index(idx) = flat.data[0].params[0] {
        let p = &flat.parameters[idx as usize];
        let set1: HashSet<String> = ["phi".to_string()].into_iter().collect();
        assert_eq!(p.get_symbols(), set1);
    }

    assert!(matches!(
        flat.data[1].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn test_if_else_basic() {
    // Test basic if-else construction
    let mut circuit = Circuit::new(2);

    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];

    circuit.if_else(condition, true_body, None).unwrap();

    // Verify the circuit has 1 operation
    assert_eq!(circuit.data.len(), 1);

    // Verify the operation is a ControlFlowGate
    let op = &circuit.data[0];
    assert!(matches!(op.instruction, Instruction::ControlFlowGate(_)));
}

#[test]
fn test_if_else_with_false_branch() {
    // Test if-else with both true and false branches
    let mut circuit = Circuit::new(2);

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

    circuit
        .if_else(condition, true_body, Some(false_body))
        .unwrap();

    assert_eq!(circuit.data.len(), 1);

    let op = &circuit.data[0];
    if let Instruction::ControlFlowGate(cf) = &op.instruction {
        use crate::circuit::gate::control_flow::ControlFlow;
        if let ControlFlow::IfElse(gate) = cf {
            assert_eq!(gate.true_body().len(), 1);
            assert_eq!(gate.false_body().unwrap().len(), 1);
        }
    }
}

#[test]
fn test_while_loop_basic() {
    // Test basic while loop construction
    let mut circuit = Circuit::new(2);

    let condition = ConditionView::new(Qubit::new(0), 1);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];

    circuit.while_loop(condition, body).unwrap();

    assert_eq!(circuit.data.len(), 1);

    let op = &circuit.data[0];
    assert!(matches!(op.instruction, Instruction::ControlFlowGate(_)));
}

#[test]
fn test_control_flow_multiple_operations() {
    // Test control flow with multiple operations in body
    let mut circuit = Circuit::new(3);

    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
    ];

    circuit.if_else(condition, true_body, None).unwrap();

    assert_eq!(circuit.data.len(), 1);

    let op = &circuit.data[0];
    if let Instruction::ControlFlowGate(cf) = &op.instruction {
        use crate::circuit::gate::control_flow::ControlFlow;
        if let ControlFlow::IfElse(gate) = cf {
            assert_eq!(gate.true_body().len(), 2);
        }
    }
}

#[test]
fn test_control_flow_inverse_error() {
    // Test that inverse() returns error for circuits with control flow
    let mut circuit = Circuit::new(2);

    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];

    circuit.if_else(condition, true_body, None).unwrap();

    // inverse() should return error for circuits with control flow
    let result = circuit.inverse();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        CircuitError::IrreversibleOperation
    ));
}

#[test]
fn test_control_flow_matrix_returns_none() {
    // Test that ControlFlowGate matrix() returns None
    use crate::circuit::gate::control_flow::{ControlFlow, IfElseGate};

    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];

    let gate = IfElseGate::new(condition, true_body, None);
    let cf = ControlFlow::IfElse(gate);

    // matrix() should return None for control flow
    let matrix = cf.matrix();
    assert!(matrix.is_none());
}

#[test]
fn test_decompose_preserves_control_flow() {
    // Test that decompose() preserves control flow structure
    let mut circuit = Circuit::new(2);

    // Add control flow with same qubit as condition
    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(0)], // Use same qubit as condition
        params: smallvec![],
        label: None,
    }];

    circuit.if_else(condition, true_body, None).unwrap();

    // Decompose the circuit
    let decomposed = circuit.decompose().unwrap();

    // Should have the control flow preserved
    assert_eq!(decomposed.data.len(), 1);

    // Control flow should still be present
    let has_control_flow = decomposed
        .data
        .iter()
        .any(|op| matches!(op.instruction, Instruction::ControlFlowGate(_)));
    assert!(
        has_control_flow,
        "ControlFlowGate should be preserved after decompose"
    );
}

#[test]
fn test_decompose_control_flow_multiple_qubits() {
    // Test decompose with control flow body using multiple qubits
    let mut circuit = Circuit::new(3);

    // Add control flow with body using multiple qubits (0, 1, 2)
    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
    ];

    circuit.if_else(condition, true_body, None).unwrap();

    // Decompose should work without error
    let decomposed = circuit.decompose().unwrap();

    // Control flow should be preserved
    let has_control_flow = decomposed
        .data
        .iter()
        .any(|op| matches!(op.instruction, Instruction::ControlFlowGate(_)));
    assert!(
        has_control_flow,
        "ControlFlowGate should be preserved after decompose"
    );
}

#[test]
fn test_compose_basic_with_mapping() {
    // Create qc1 with qubits 1, 3, 5
    let mut qc1 = Circuit::new(0);
    let q1 = Qubit::new(1);
    let q3 = Qubit::new(3);
    let q5 = Qubit::new(5);
    qc1.add_qubits(vec![q1, q3, q5]).unwrap();
    qc1.h(q1).unwrap();

    // Create qc2 with qubits 1, 2
    let mut qc2 = Circuit::new(0);
    let q2 = Qubit::new(2);
    qc2.add_qubits(vec![q1, q2]).unwrap();
    qc2.x(q1).unwrap();

    // Compose: map qc2's q1 -> qc1's q3, qc2's q2 -> qc1's q1
    let result = qc1.compose(&qc2, Some(&[q3, q1]));
    assert!(result.is_ok(), "compose should succeed");

    // Verify: qc1 qubit count unchanged, qubits are {1, 3, 5}
    assert_eq!(qc1.num_qubits(), 3);
    let expected: Vec<Qubit> = vec![q1, q3, q5];
    assert_eq!(qc1.qubits(), expected);

    // Verify: 2 operations (1 from qc1, 1 from qc2)
    assert_eq!(qc1.data.len(), 2);
}

#[test]
fn test_compose_without_mapping() {
    // Create qc1 with qubits 1, 3, 5
    let mut qc1 = Circuit::new(0);
    let q1 = Qubit::new(1);
    let q3 = Qubit::new(3);
    let q5 = Qubit::new(5);
    qc1.add_qubits(vec![q1, q3, q5]).unwrap();
    qc1.h(q1).unwrap();

    // Create qc2 with qubits 6, 7
    let mut qc2 = Circuit::new(0);
    let q6 = Qubit::new(6);
    let q7 = Qubit::new(7);
    qc2.add_qubits(vec![q6, q7]).unwrap();
    qc2.cx(q6, q7).unwrap();

    // Compose without mapping: qc2's qubits are appended
    let result = qc1.compose(&qc2, None);
    assert!(result.is_ok(), "compose should succeed");

    // Verify: 3 + 2 = 5 qubits
    assert_eq!(qc1.num_qubits(), 5);

    // Verify: all qubits present
    let expected: Vec<Qubit> = vec![q1, q3, q5, q6, q7];
    assert_eq!(qc1.qubits(), expected);

    // Verify: 2 operations (1 from qc1, 1 from qc2)
    assert_eq!(qc1.data.len(), 2);
}

#[test]
fn test_compose_empty_circuit() {
    let mut qc1 = Circuit::new(2);
    qc1.h(Qubit::new(0)).unwrap();

    let qc2 = Circuit::new(0);

    let result = qc1.compose(&qc2, None);
    assert!(result.is_ok());

    assert_eq!(qc1.num_qubits(), 2);
    assert_eq!(qc1.data.len(), 1);
}

#[test]
fn test_compose_qubit_count_mismatch() {
    let mut qc1 = Circuit::new(0);
    let q1 = Qubit::new(1);
    let q3 = Qubit::new(3);
    qc1.add_qubits(vec![q1, q3]).unwrap();

    let mut qc2 = Circuit::new(0);
    let q2 = Qubit::new(2);
    qc2.add_qubits(vec![q1, q2, q3]).unwrap();

    // Mapping only 2 qubits, but qc2 has 3
    let result = qc1.compose(&qc2, Some(&[q1, q3]));
    assert!(matches!(
        result,
        Err(CircuitError::QubitCountMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[test]
fn test_compose_nonexistent_target_qubit() {
    let mut qc1 = Circuit::new(0);
    let q1 = Qubit::new(1);
    qc1.add_qubits(vec![q1]).unwrap();

    let mut qc2 = Circuit::new(0);
    let q2 = Qubit::new(2);
    qc2.add_qubits(vec![q2]).unwrap();

    // Try to map qc2's q2 to q3, which does not exist in qc1
    let q3 = Qubit::new(3);
    let result = qc1.compose(&qc2, Some(&[q3]));
    assert!(matches!(result, Err(CircuitError::QubitNotFound(3))));
}

#[test]
fn test_compose_with_parameters() {
    // qc1 with parameterized gate
    let mut qc1 = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    qc1.rx(Qubit::new(0), theta.clone()).unwrap();

    // qc2 with different parameterized gate
    let mut qc2 = Circuit::new(1);
    let phi = Parameter::symbol("phi");
    qc2.ry(Qubit::new(0), phi.clone()).unwrap();

    let result = qc1.compose(&qc2, None);
    assert!(result.is_ok());

    // Both parameters should be present
    assert_eq!(qc1.parameters().len(), 2);
    assert!(qc1.symbols().contains("theta"));
    assert!(qc1.symbols().contains("phi"));
}

#[test]
fn test_compose_preserves_operation_order() {
    // qc1: Bell state preparation
    let mut qc1 = Circuit::new(2);
    qc1.h(Qubit::new(0)).unwrap();
    qc1.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    // qc2: Bell measurement
    let mut qc2 = Circuit::new(2);
    qc2.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    qc2.h(Qubit::new(0)).unwrap();

    let result = qc1.compose(&qc2, None);
    assert!(result.is_ok());

    // Total: 2 + 2 = 4 operations
    assert_eq!(qc1.data.len(), 4);

    // Verify order: h, cx (from qc1), cx, h (from qc2)
    assert!(matches!(
        qc1.data[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        qc1.data[1].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert!(matches!(
        qc1.data[2].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert!(matches!(
        qc1.data[3].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}
