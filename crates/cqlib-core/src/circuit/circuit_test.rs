// This code is part of Cqlib.

// (C) Copyright China Telecom Quantum Group 2026

// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::Qubit;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::parameter::impls::Parameter;
use smallvec::smallvec;
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

    // 4. Test Inversion of CircuitGate
    // main.inverse() should call bell_gate.inverse()
    // let inv_main = main.inverse().unwrap();
    // let op = &inv_main.data[0];

    //     if let Instruction::Circuit(gate) = &op.instruction {
    //         assert_eq!(gate.name.as_ref(), "bell_dg"); // Name should have suffix
    //         // Inner circuit should be inverted: CX(q0, q1) -> H(q0) (Order reversed and inverted)
    //         // CX inv is CX, H inv is H. So: CX then H.
    //         let inner_ops = &gate.circuit().circuit.data;
    //         assert_eq!(inner_ops.len(), 2);
    //         assert!(matches!(
    //             inner_ops[0].instruction,
    //             Instruction::Standard(StandardGate::CX)
    //         ));
    //         assert!(matches!(
    //             inner_ops[1].instruction,
    //             Instruction::Standard(StandardGate::H)
    //         ));
    //     } else {
    //         panic!("Expected Circuit instruction after inverse");
    //     }
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
    bindings.insert("a".to_string(), 1.0);

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
    bindings.insert("a".to_string(), 1.0);
    bindings.insert("b".to_string(), 2.0);

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
    let decomposed = outer.decompose();

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
        bind.insert("gamma".to_string(), 2.0);
        bind.insert("delta".to_string(), 3.0);
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
        bind.insert("gamma".to_string(), 2.0);
        bind.insert("delta".to_string(), 3.0);
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

    let flat = top.decompose();
    assert_eq!(flat.data.len(), 2);

    assert!(matches!(
        flat.data[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    if let CircuitParam::Index(idx) = flat.data[0].params[0] {
        let p = &flat.parameters[idx as usize];
        assert_eq!(p.get_symbols(), vec!["phi"]);
    }

    assert!(matches!(
        flat.data[1].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}
