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
use crate::circuit::gate::control_flow::{ConditionView, IfElseGate};
use crate::circuit::gate::{ControlFlow, FrozenCircuit, UnitaryGate};
use crate::circuit::operation::Operation;
use crate::circuit::{Circuit, Directive, Instruction, ParameterValue, Qubit};
use indexmap::IndexSet;
use ndarray::{Array2, array};
use num_complex::Complex64;
use smallvec::smallvec;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

fn assert_matrix_approx_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>, eps: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (*a - *e).norm();
        assert!(
            diff < eps,
            "matrix element {idx} differs: got {a}, expected {e}, diff {diff}"
        );
    }
}

#[test]
fn test_fixed_standard_gate_matches_numeric_matrix() {
    let symbolic = standard_gate_symbolic_matrix(StandardGate::H, &[]).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let numeric = StandardGate::H.matrix(&[]).unwrap();

    assert_matrix_approx_eq(&evaluated, numeric.as_ref(), 1e-12);
}

#[test]
fn test_parametric_rx_symbolic_matrix_evaluates() {
    let theta = Parameter::symbol("theta");
    let symbolic = standard_gate_symbolic_matrix(StandardGate::RX, &[theta]).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", PI / 2.0);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
    let expected = StandardGate::RX.matrix(&[PI / 2.0]).unwrap();

    assert_matrix_approx_eq(&evaluated, expected.as_ref(), 1e-12);
}

#[test]
fn test_parametric_u_symbolic_matrix_evaluates() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let lambda = Parameter::symbol("lambda");
    let symbolic = standard_gate_symbolic_matrix(StandardGate::U, &[theta, phi, lambda]).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.25);
    bindings.insert("phi", -0.5);
    bindings.insert("lambda", 0.75);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
    let expected = StandardGate::U.matrix(&[0.25, -0.5, 0.75]).unwrap();

    assert_matrix_approx_eq(&evaluated, expected.as_ref(), 1e-12);
}

#[test]
fn test_symbolic_circuit_matches_bound_numeric_circuit() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta).unwrap();
    circuit.rz(Qubit::new(0), phi).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.37);
    bindings.insert("phi", -0.91);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();

    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_entangling_circuit_matches_numeric_circuit() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.rz(Qubit::new(1), theta).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.42);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();
    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_single_qubit_optimized_path_matches_numeric_circuit() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.rx(Qubit::new(2), theta).unwrap();
    circuit.rz(Qubit::new(1), 0.25).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, Some(&[2, 0, 1])).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", -0.73);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();
    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, Some(&[2, 0, 1])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_two_qubit_optimized_path_matches_numeric_circuit() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(3);
    circuit.rxx(Qubit::new(2), Qubit::new(0), theta).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.ry(Qubit::new(0), 0.41).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, Some(&[1, 2, 0])).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.64);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();
    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, Some(&[1, 2, 0])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_general_gate_path_matches_numeric_circuit() {
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(1)).unwrap();
    circuit
        .ccx(Qubit::new(2), Qubit::new(0), Qubit::new(1))
        .unwrap();
    circuit.rz(Qubit::new(2), 0.17).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, Some(&[2, 0, 1])).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&circuit, Some(&[2, 0, 1])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_global_phase_is_included() {
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(1);
    circuit.x(Qubit::new(0)).unwrap();
    circuit.set_global_phase(phi);

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("phi", PI);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
    let expected = array![
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_qubits_order_matches_numeric_path() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.ry(Qubit::new(0), theta).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, Some(&[1, 0])).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", -0.33);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();
    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, Some(&[1, 0])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_circuit_gate_replaces_inner_symbols() {
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
    let gate = inner.to_gate("InnerRx").unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [ParameterValue::from(Parameter::symbol("x") * 2.0)],
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("x", 0.2);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();
    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_unitary_gate_numeric_matrix() {
    let x = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let gate = UnitaryGate::new("XLike", 1).with_matrix(x.clone()).unwrap();
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(gate)),
            [Qubit::new(0)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();

    assert_matrix_approx_eq(&evaluated, &x, 1e-12);
}

#[test]
fn test_symbolic_unitary_gate_circuit_definition() {
    let mut inner = Circuit::new(1);
    inner.h(Qubit::new(0)).unwrap();
    let gate = UnitaryGate::new("HInner", 1).with_circuit(Arc::new(FrozenCircuit::new(inner)));
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(gate)),
            [Qubit::new(0)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&circuit, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-12);
}

#[test]
fn test_symbolic_invalid_qubits_order_errors() {
    let circuit = Circuit::new(2);
    let err = circuit_to_symbolic_matrix(&circuit, Some(&[0, 0])).unwrap_err();

    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_symbolic_measure_has_no_matrix() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Directive(Directive::Measure),
            [Qubit::new(0)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let err = circuit_to_symbolic_matrix(&circuit, None).unwrap_err();

    assert!(matches!(err, CircuitError::NoMatrixRepresentation));
}

#[test]
fn test_symbolic_circuit_gate_reversed_bits() {
    // Inner circuit: CNOT(q0 -> q1)  (asymmetric, so bit-order matters)
    let mut inner = Circuit::new(2);
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let gate = inner.to_gate("CnotGate").unwrap();

    // Apply as CircuitGate to (q1, q0) in a 2-qubit circuit
    let mut circuit = Circuit::new(2);
    circuit
        .append(gate, [Qubit::new(1), Qubit::new(0)], [], None)
        .unwrap();
    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();

    // Direct CNOT(q1 -> q0) should produce the same matrix
    let mut expected_circuit = Circuit::new(2);
    expected_circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&expected_circuit, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_symbolic_circuit_gate_simultaneous_substitution() {
    // Inner circuit: RX(a + b)
    let mut inner = Circuit::new(1);
    inner
        .rx(
            Qubit::new(0),
            Parameter::symbol("a") + Parameter::symbol("b"),
        )
        .unwrap();
    let gate = inner.to_gate("AddGate").unwrap();

    // Outer call swaps a <-> b: parameters are [b, a]
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [
                ParameterValue::from(Parameter::symbol("b")),
                ParameterValue::from(Parameter::symbol("a")),
            ],
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("a", 1.0);
    bindings.insert("b", 2.0);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();

    // After simultaneous substitution a->b, b->a, the expression becomes b + a = 2 + 1 = 3
    let expected = StandardGate::RX.matrix(&[3.0]).unwrap();
    assert_matrix_approx_eq(&evaluated, expected.as_ref(), 1e-10);
}

#[test]
fn test_symbolic_circuit_gate_param_count_mismatch() {
    let theta = Parameter::symbol("theta");
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), theta).unwrap();
    let gate = inner.to_gate("RxGate").unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [ParameterValue::Fixed(1.0), ParameterValue::Fixed(2.0)],
            None,
        )
        .unwrap();

    let err = circuit_to_symbolic_matrix(&circuit, None).unwrap_err();
    assert!(matches!(err, CircuitError::ParameterCountMismatch { .. }));
}

#[test]
fn test_circuit_gate_matches_standard_sequence() {
    let mut inner = Circuit::new(2);
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let gate = inner.to_gate("CnotGate").unwrap();

    let mut cg_circuit = Circuit::new(2);
    cg_circuit
        .append(gate.clone(), [Qubit::new(0), Qubit::new(1)], [], None)
        .unwrap();

    let mut native_circuit = Circuit::new(2);
    native_circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let cg_symbolic = circuit_to_symbolic_matrix(&cg_circuit, None).unwrap();
    let native_symbolic = circuit_to_symbolic_matrix(&native_circuit, None).unwrap();
    let cg_eval = evaluate_symbolic_matrix(&cg_symbolic, &None).unwrap();
    let native_eval = evaluate_symbolic_matrix(&native_symbolic, &None).unwrap();

    assert_matrix_approx_eq(&cg_eval, &native_eval, 1e-12);
}

#[test]
fn test_circuit_gate_different_qubits_order() {
    let mut inner = Circuit::new(2);
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let gate = inner.to_gate("CnotGate").unwrap();

    let mut cg_circuit = Circuit::new(2);
    cg_circuit
        .append(gate, [Qubit::new(0), Qubit::new(1)], [], None)
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&cg_circuit, Some(&[1, 0])).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&cg_circuit, Some(&[1, 0])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_unitary_gate_circuit_2qubit_no_rev_bug() {
    let mut inner = Circuit::new(2);
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let gate = UnitaryGate::new("CnotU", 2).with_circuit(Arc::new(FrozenCircuit::new(inner)));

    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(gate)),
            [Qubit::new(1), Qubit::new(0)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();

    let mut expected_circuit = Circuit::new(2);
    expected_circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&expected_circuit, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_unitary_gate_circuit_3qubit_asymmetric() {
    let mut inner = Circuit::new(3);
    inner
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();
    let gate = UnitaryGate::new("CCXU", 3).with_circuit(Arc::new(FrozenCircuit::new(inner)));

    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(gate)),
            [Qubit::new(2), Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, Some(&[2, 0, 1])).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();

    let mut expected_circuit = Circuit::new(3);
    expected_circuit
        .ccx(Qubit::new(2), Qubit::new(0), Qubit::new(1))
        .unwrap();
    let expected = crate::circuit::circuit_to_matrix(&expected_circuit, Some(&[2, 0, 1])).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_serial_parallel_consistency_single_qubit() {
    let mut small = Circuit::new(1);
    small.h(Qubit::new(0)).unwrap();
    let small_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&small, None).unwrap(), &None)
            .unwrap();

    let mut large = Circuit::new(11);
    large.h(Qubit::new(5)).unwrap();
    let large_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&large, None).unwrap(), &None)
            .unwrap();

    let small_numeric = crate::circuit::circuit_to_matrix(&small, None).unwrap();
    let large_numeric = crate::circuit::circuit_to_matrix(&large, None).unwrap();

    assert_matrix_approx_eq(&small_eval, &small_numeric, 1e-10);
    assert_matrix_approx_eq(&large_eval, &large_numeric, 1e-10);
}

#[test]
fn test_serial_parallel_consistency_two_qubit() {
    let mut small = Circuit::new(2);
    small.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let small_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&small, None).unwrap(), &None)
            .unwrap();

    let mut large = Circuit::new(11);
    large.cx(Qubit::new(3), Qubit::new(7)).unwrap();
    let large_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&large, None).unwrap(), &None)
            .unwrap();

    let small_numeric = crate::circuit::circuit_to_matrix(&small, None).unwrap();
    let large_numeric = crate::circuit::circuit_to_matrix(&large, None).unwrap();

    assert_matrix_approx_eq(&small_eval, &small_numeric, 1e-10);
    assert_matrix_approx_eq(&large_eval, &large_numeric, 1e-10);
}

#[test]
fn test_serial_parallel_consistency_general_gate() {
    let mut small = Circuit::new(3);
    small
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();
    let small_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&small, None).unwrap(), &None)
            .unwrap();

    let mut large = Circuit::new(11);
    large
        .ccx(Qubit::new(2), Qubit::new(5), Qubit::new(8))
        .unwrap();
    let large_eval =
        evaluate_symbolic_matrix(&circuit_to_symbolic_matrix(&large, None).unwrap(), &None)
            .unwrap();

    let small_numeric = crate::circuit::circuit_to_matrix(&small, None).unwrap();
    let large_numeric = crate::circuit::circuit_to_matrix(&large, None).unwrap();

    assert_matrix_approx_eq(&small_eval, &small_numeric, 1e-10);
    assert_matrix_approx_eq(&large_eval, &large_numeric, 1e-10);
}

#[test]
fn test_random_small_circuit_consistency() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");

    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.rx(Qubit::new(1), theta.clone()).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.rz(Qubit::new(2), phi.clone()).unwrap();
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();
    circuit.ry(Qubit::new(0), 0.31).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();

    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.73);
    bindings.insert("phi", -0.42);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();

    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}

#[test]
fn test_substitution_collision_detection() {
    let mut inner = Circuit::new(1);
    inner
        .rx(
            Qubit::new(0),
            Parameter::symbol("__cqlib_internal_sub_theta"),
        )
        .unwrap();
    let gate = inner.to_gate("BadGate").unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [ParameterValue::from(Parameter::symbol("x"))],
            None,
        )
        .unwrap();

    let err = circuit_to_symbolic_matrix(&circuit, None).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_substitution_cross_dependency() {
    let mut inner = Circuit::new(1);
    inner
        .rx(
            Qubit::new(0),
            Parameter::symbol("a") + Parameter::symbol("b"),
        )
        .unwrap();
    let gate = inner.to_gate("AddGate").unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [
                ParameterValue::from(Parameter::symbol("b")),
                ParameterValue::from(Parameter::symbol("a")),
            ],
            None,
        )
        .unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("a", 1.5);
    bindings.insert("b", 2.5);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();

    let expected = StandardGate::RX.matrix(&[4.0]).unwrap();
    assert_matrix_approx_eq(&evaluated, expected.as_ref(), 1e-10);
}

#[test]
fn test_empty_circuit_identity() {
    let circuit = Circuit::new(2);
    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let expected = Array2::eye(4);
    assert_matrix_approx_eq(&evaluated, &expected, 1e-12);
}

#[test]
fn test_invalid_parameter_index_error() {
    let qubits: IndexSet<Qubit> = [Qubit::new(0)].into_iter().collect();
    let op = Operation {
        instruction: Instruction::Standard(StandardGate::RX),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![CircuitParam::Index(999)],
        label: None,
    };
    let circuit = Circuit::from_parts(
        qubits,
        IndexSet::new(),
        IndexSet::new(),
        vec![op],
        CircuitParam::Fixed(0.0),
    );

    let err = circuit_to_symbolic_matrix(&circuit, None).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidParameterIndex(999)));
}

#[test]
fn test_control_flow_no_matrix() {
    let condition = ConditionView::new(Qubit::new(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let gate = IfElseGate::new(condition, true_body, None);
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)),
            [Qubit::new(0), Qubit::new(1)],
            [],
            None,
        )
        .unwrap();

    let err = circuit_to_symbolic_matrix(&circuit, None).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_delay_is_skipped() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit
        .append(Instruction::Delay, [Qubit::new(0)], [], None)
        .unwrap();
    circuit.x(Qubit::new(0)).unwrap();

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let evaluated = evaluate_symbolic_matrix(&symbolic, &None).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&circuit, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-12);
}

#[test]
fn test_global_phase_with_bindings() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.set_global_phase(theta);

    let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("theta", PI / 3.0);
    let evaluated = evaluate_symbolic_matrix(&symbolic, &Some(bindings.clone())).unwrap();

    let bound = circuit.assign_parameters(&Some(bindings)).unwrap();
    let expected = crate::circuit::circuit_to_matrix(&bound, None).unwrap();

    assert_matrix_approx_eq(&evaluated, &expected, 1e-10);
}
