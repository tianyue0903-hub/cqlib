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

use crate::compile::optimization::commutative::CommutativeOptimization;
use std::f64::consts::PI;

use crate::circuit::gate::control_flow::ConditionView;
use crate::circuit::gate::{ControlFlow, Instruction, MCGate, StandardGate, UnitaryGate};
use crate::circuit::operation::Operation;
use crate::circuit::{Circuit, CircuitParam, Parameter, Qubit};
use ndarray::array;
use num_complex::Complex64;
use smallvec::smallvec;

/// Create complex number from real and imaginary parts
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Helper constructor since the `Operation` struct is public but has no
/// dedicated `new` method.  Most tests deal with simple, parameterless
/// standard gates.
fn make_op(gate: StandardGate, qubits: &[u32]) -> Operation {
    Operation {
        instruction: Instruction::Standard(gate),
        qubits: qubits.iter().map(|&i| Qubit::new(i)).collect(),
        params: smallvec![],
        label: None,
    }
}

#[test]
fn identity_commutes_with_everything() {
    let id = make_op(StandardGate::I, &[]);
    let x = make_op(StandardGate::X, &[0]);
    assert!(CommutativeOptimization::is_commutative(&id, &x));
    assert!(CommutativeOptimization::is_commutative(&x, &id));
}

#[test]
fn disjoint_qubits_are_commutative() {
    let h = make_op(StandardGate::H, &[0]);
    let x = make_op(StandardGate::X, &[1]);
    assert!(CommutativeOptimization::is_commutative(&h, &x));
    assert!(CommutativeOptimization::is_commutative(&x, &h));
}

#[test]
fn same_qubit_non_commuting_gates() {
    // H and X do not commute on the same qubit (HX = - XH up to phase).
    let h = make_op(StandardGate::H, &[0]);
    let x = make_op(StandardGate::X, &[0]);
    assert!(!CommutativeOptimization::is_commutative(&h, &x));
    assert!(!CommutativeOptimization::is_commutative(&x, &h));
}

#[test]
fn identical_operations_always_commute() {
    let x1 = make_op(StandardGate::X, &[0]);
    let x2 = make_op(StandardGate::X, &[0]);
    assert!(CommutativeOptimization::is_commutative(&x1, &x2));
}

#[test]
fn overlapping_but_commuting_examples() {
    // Z on target commutes with CZ (both are diagonal).
    let cz = make_op(StandardGate::CZ, &[0, 1]);
    let z1 = make_op(StandardGate::Z, &[1]);
    assert!(CommutativeOptimization::is_commutative(&cz, &z1));
    assert!(CommutativeOptimization::is_commutative(&z1, &cz));

    // Z with Z on same qubit trivially commutes.
    let z0 = make_op(StandardGate::Z, &[0]);
    let z0b = make_op(StandardGate::Z, &[0]);
    assert!(CommutativeOptimization::is_commutative(&z0, &z0b));
}

#[test]
fn commutative_support_unitary_gate() {
    // Create a custom unitary (Pauli X for simplicity)
    let mat = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];
    let u_gate = UnitaryGate::new("CustomX", 1).with_matrix(mat).unwrap();
    let u_op = Operation {
        instruction: Instruction::UnitaryGate(Box::new(u_gate)),
        qubits: vec![Qubit::new(0)].into(),
        params: smallvec![],
        label: None,
    };

    let x = make_op(StandardGate::X, &[0]);
    assert!(CommutativeOptimization::is_commutative(&u_op, &x));
    assert!(CommutativeOptimization::is_commutative(&x, &u_op));
}

#[test]
fn commutative_support_mc_gate() {
    // Create a custom unitary (Pauli X for simplicity)
    let ccx = MCGate::new(2, StandardGate::X);
    let u_op = Operation {
        instruction: Instruction::McGate(Box::new(ccx)),
        qubits: vec![0, 1, 2].iter().map(|&i| Qubit::new(i)).collect(),
        params: smallvec![],
        label: None,
    };

    let x = make_op(StandardGate::X, &[0]);
    assert!(!CommutativeOptimization::is_commutative(&u_op, &x));
    assert!(!CommutativeOptimization::is_commutative(&x, &u_op));
}

#[test]
fn commutative_support_circuit_gate() {
    let mut x_cir = Circuit::new(1);
    x_cir.x(x_cir.qubits()[0]).unwrap();
    let x_cg = x_cir.to_gate("X_CG").unwrap();
    let u_op = Operation {
        instruction: x_cg,
        qubits: vec![0].iter().map(|&i| Qubit::new(i)).collect(),
        params: smallvec![],
        label: None,
    };

    let x = make_op(StandardGate::X, &[0]);
    assert!(CommutativeOptimization::is_commutative(&u_op, &x));
    assert!(CommutativeOptimization::is_commutative(&x, &u_op));
}

#[test]
fn unsupported_control_flow_returns_false() {
    // ControlFlowGate is not unitary and should return false
    use crate::circuit::gate::control_flow::{ConditionView, ControlFlow, IfElseGate};

    let cond = ConditionView::new(Qubit::new(0), 0);
    let cond_gate = IfElseGate::new(cond, vec![], None);
    let ctrl_op = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(cond_gate)),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };

    let x = make_op(StandardGate::X, &[0]);
    assert!(!CommutativeOptimization::is_commutative(&ctrl_op, &x));
    assert!(!CommutativeOptimization::is_commutative(&x, &ctrl_op));
}

#[test]
fn unsupported_delay_returns_false() {
    // Delay is not unitary and should return false
    let delay_op = Operation {
        instruction: Instruction::Delay,
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };

    let x = make_op(StandardGate::X, &[0]);
    assert!(!CommutativeOptimization::is_commutative(&delay_op, &x));
    assert!(!CommutativeOptimization::is_commutative(&x, &delay_op));
}

#[test]
fn two_x_gates_cancelled() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    cir.x(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['x']), true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 0);
}

#[test]
fn x_and_y_gates_not_cancelled() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    cir.y(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['x', 'y']), true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 2);
}

#[test]
fn rx_gate_deparametered() {
    let mut cir = Circuit::new(1);
    cir.rx(cir.qubits()[0], PI).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['x']), true, false);
    let optimized_cir = co.execute(&cir);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::X)
    ));
}

#[test]
fn rx_gate_not_deparametered() {
    let mut cir = Circuit::new(1);
    cir.rx(cir.qubits()[0], PI).unwrap();
    let mut co = CommutativeOptimization::new(None, None, true, false);
    let optimized_cir = co.execute(&cir);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
}

#[test]
fn two_x_gates_cancelled_with_rx_inbetween() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    cir.rx(cir.qubits()[0], 0.1).unwrap();
    cir.x(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, None, true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 2);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(matches!(
        optimized_cir.operations()[1].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
}

#[test]
fn two_x_gates_not_cancelled_with_y_inbetween() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    cir.y(cir.qubits()[0]).unwrap();
    cir.x(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['x', 'y']), true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 3);
}

#[test]
fn y2p_and_y2m_gates_cancelled() {
    let mut cir = Circuit::new(1);
    cir.y2p(cir.qubits()[0]).unwrap();
    cir.y2m(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['y']), true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 0);
}

#[test]
fn y2p_and_y2m_gates_not_cancelled() {
    let mut cir = Circuit::new(1);
    cir.y2p(cir.qubits()[0]).unwrap();
    cir.y2m(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(Some(vec!['x']), None, true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 2);
}

#[test]
fn x_gate_parametered_with_keep_phase() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, None, true, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 2);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(matches!(
        optimized_cir.operations()[1].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    assert!(matches!(
        optimized_cir.operations()[1].params[0],
        CircuitParam::Fixed(a) if (a - PI / 2.0).abs() <= f64::EPSILON
    ));
}

#[test]
fn x_gate_parametered_without_keep_phase() {
    let mut cir = Circuit::new(1);
    cir.x(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, None, false, false);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 1);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(optimized_cir.global_phase() == Parameter::try_from(0.0).unwrap());
}

#[test]
fn y2p_and_y2m_gates_not_cancelled_under_keep_order() {
    let mut cir = Circuit::new(1);
    cir.y2p(cir.qubits()[0]).unwrap();
    cir.y2m(cir.qubits()[0]).unwrap();
    let mut co = CommutativeOptimization::new(None, None, true, true);
    let optimized_cir = co.execute(&cir);
    assert!(optimized_cir.operations().len() == 2);
}

#[test]
fn rx_gate_not_deparametered_with_keep_order() {
    let mut cir = Circuit::new(1);
    cir.rx(cir.qubits()[0], PI).unwrap();
    let mut co = CommutativeOptimization::new(None, Some(vec!['x']), true, true);
    let optimized_cir = co.execute(&cir);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
}

#[test]
fn support_symbolic_param() {
    let mut cir = Circuit::new(1);
    // Symbolic parameter
    let theta = Parameter::symbol("theta");
    cir.rx(cir.qubits()[0], theta.clone()).unwrap();
    cir.rx(cir.qubits()[0], PI).unwrap();

    let mut co = CommutativeOptimization::new(None, None, true, false);
    let optimized_cir = co.execute(&cir);
    assert!(matches!(
        optimized_cir.operations()[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(matches!(
        optimized_cir.operations()[0].params[0],
        CircuitParam::Index(0)
    ));
}

#[test]
fn while_loop_body_supported() {
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);
    let condition2 = ConditionView::new(q0, 1);
    let while_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q0],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q0],
            params: smallvec![],
            label: None,
        },
    ];

    circuit.while_loop(condition2, while_body).unwrap();

    let mut co = CommutativeOptimization::new(None, None, true, false);
    let optimized_cir = co.execute(&circuit);
    assert!(matches!(
        &optimized_cir.operations()[0].instruction,
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) if gate.body().len() == 0
    ));
}
