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

//! Tests for circuit_to_matrix function
//!
//! These tests verify the correctness of converting quantum circuits to their
//! unitary matrix representations.

use super::*;
use crate::circuit::Qubit;
use crate::circuit::gate::{StandardGate, UnitaryGate};
use crate::circuit::param::ParameterValue;
use crate::circuit::parameter::Parameter;
use ndarray::array;
use num_complex::Complex64;
use std::f64::consts::{PI, SQRT_2};

/// Assert that two complex matrices are approximately equal
fn assert_matrix_approx_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>, eps: f64) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "Matrix shapes differ: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).norm();
        assert!(
            diff < eps,
            "Matrix element [{}] differs: got {}, expected {}, diff = {} > {}",
            i,
            a,
            e,
            diff,
            eps
        );
    }
}

/// Assert that a matrix is unitary: U† * U = I
fn assert_is_unitary(matrix: &Array2<Complex64>, eps: f64) {
    let n = matrix.nrows();
    let conj_t = matrix.t().mapv(|x| x.conj());
    let product = conj_t.dot(matrix);

    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[i, j]] - expected).norm();
            assert!(
                diff < eps,
                "Matrix not unitary at [{}, {}]: got {}, expected {}, diff = {}",
                i,
                j,
                product[[i, j]],
                expected,
                diff
            );
        }
    }
}

/// Create identity matrix of given dimension
fn eye(n: usize) -> Array2<Complex64> {
    Array2::eye(n)
}

/// Create complex number from real and imaginary parts
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn test_single_qubit_identity() {
    let mut circuit = Circuit::new(1);
    circuit.i(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = eye(2);

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_pauli_x() {
    let mut circuit = Circuit::new(1);
    circuit.x(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_pauli_y() {
    let mut circuit = Circuit::new(1);
    circuit.y(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = array![[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_pauli_z() {
    let mut circuit = Circuit::new(1);
    circuit.z(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_hadamard() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_s_gate() {
    let mut circuit = Circuit::new(1);
    circuit.s(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(0.0, 1.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_t_gate() {
    let mut circuit = Circuit::new(1);
    circuit.t(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let s = 1.0 / SQRT_2;
    let expected = array![[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(s, s)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_rx() {
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), PI / 2.0).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, 0.0), c(0.0, -s)], [c(0.0, -s), c(s, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_ry() {
    let mut circuit = Circuit::new(1);
    circuit.ry(Qubit::new(0), PI / 2.0).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, 0.0), c(-s, 0.0)], [c(s, 0.0), c(s, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_rz() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), PI / 2.0).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, -s), c(0.0, 0.0)], [c(0.0, 0.0), c(s, s)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_single_qubit_rx_pi() {
    // RX(pi) = -iX, should be close to X up to global phase
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), PI).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    // RX(pi) = [[0, -i], [-i, 0]] = -i * X
    let expected = array![[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, -1.0), c(0.0, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_two_qubit_cnot_control_low() {
    // CNOT with control=0, target=1 (natural order)
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // Expected CNOT matrix (control=0, target=1)
    // |00> -> |00>
    // |01> -> |01>
    // |10> -> |11>
    // |11> -> |10>
    let expected = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_two_qubit_cnot_control_high() {
    // CNOT with control=1, target=0 (reversed order)
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // Expected CNOT matrix (control=1, target=0)
    // |00> -> |00>
    // |01> -> |11>  (control|1>, target|0> -> target flipped)
    // |10> -> |10>
    // |11> -> |01>  (control|1>, target|1> -> target flipped)
    let expected = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_two_qubit_cz() {
    // CZ gate
    let mut circuit = Circuit::new(2);
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    let expected = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_two_qubit_swap() {
    let mut circuit = Circuit::new(2);
    circuit.swap(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // SWAP: |01> <-> |10>
    let expected = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_two_qubit_rxx() {
    let mut circuit = Circuit::new(2);
    circuit.rxx(Qubit::new(0), Qubit::new(1), PI / 2.0).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    let s = 1.0 / SQRT_2;
    let expected = array![
        [c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, -s)],
        [c(0.0, 0.0), c(s, 0.0), c(0.0, -s), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, -s), c(s, 0.0), c(0.0, 0.0)],
        [c(0.0, -s), c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

// ============================================================================
// Three Qubit Gate Tests
// ============================================================================

#[test]
fn test_three_qubit_ccx() {
    // CCX (Toffoli) gate
    let mut circuit = Circuit::new(3);
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // CCX flips target (q2) when both controls (q0, q1) are |1>
    // Only |110> <-> |111> should be swapped
    let mut expected = eye(8);
    expected[[6, 6]] = c(0.0, 0.0); // |110>
    expected[[6, 7]] = c(1.0, 0.0);
    expected[[7, 6]] = c(1.0, 0.0); // |111>
    expected[[7, 7]] = c(0.0, 0.0);

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_three_qubit_non_adjacent_cnot() {
    // CNOT with non-adjacent qubits (control=0, target=2)
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // Verify specific transformations
    // |100> (q0=1, q2=0) -> |101> (q0=1, q2=1)
    // The state |100> is index 4, |101> is index 5
    // Check that matrix[5, 4] ≈ 1 and matrix[4, 4] ≈ 0
    assert!((matrix[[5, 4]] - c(1.0, 0.0)).norm() < 1e-10);
    assert!((matrix[[4, 4]] - c(0.0, 0.0)).norm() < 1e-10);
}

// ============================================================================
// Multi-Control Gate Tests (MCGate)
// ============================================================================

#[test]
fn test_mcgate_three_control_x() {
    // 3-control X gate (CCC-X)
    let mut circuit = Circuit::new(4);
    circuit
        .multi_control(
            StandardGate::X,
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            [Qubit::new(3)],
            [],
        )
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // Should flip q3 only when q0=q1=q2=1
    // |1110> (index 14) <-> |1111> (index 15)
    let mut expected = eye(16);
    expected[[14, 14]] = c(0.0, 0.0);
    expected[[14, 15]] = c(1.0, 0.0);
    expected[[15, 14]] = c(1.0, 0.0);
    expected[[15, 15]] = c(0.0, 0.0);

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_mcgate_control_higher_than_target() {
    // Control qubit with higher index than target
    let mut circuit = Circuit::new(3);
    circuit
        .multi_control(StandardGate::X, [Qubit::new(2)], [Qubit::new(0)], [])
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // With Big-Endian mapping (q0=MSB, q1, q2=LSB):
    // |101> (index 5) means q0=1, q1=0, q2=1
    // Control (q2) = 1, so target (q0) should flip: 1 -> 0
    // Result: |001> (index 1)
    // Verify |101> (index 5) maps to |001> (index 1)
    assert!((matrix[[1, 5]] - c(1.0, 0.0)).norm() < 1e-10);

    // Also verify that when control=0, target is unchanged
    // |100> (q0=1, q1=0, q2=0) -> |100> (unchanged)
    assert!((matrix[[4, 4]] - c(1.0, 0.0)).norm() < 1e-10);
}

#[test]
fn test_unitary_gate_single_qubit() {
    // Create a custom unitary (Pauli X for simplicity)
    let mat = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];
    let u_gate = UnitaryGate::new("CustomX", 1).with_matrix(mat).unwrap();

    let mut circuit = Circuit::new(1);
    circuit.unitary(u_gate, vec![Qubit::new(0)]).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_unitary_gate_two_qubit() {
    // Custom two-qubit gate (SWAP-like)
    let mat = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
    ];
    let u_gate = UnitaryGate::new("CustomSwap", 2).with_matrix(mat).unwrap();

    let mut circuit = Circuit::new(2);
    circuit
        .unitary(u_gate, vec![Qubit::new(0), Qubit::new(1)])
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // Verify SWAP-like behavior
    // |01> -> |10>
    assert!((matrix[[2, 1]] - c(1.0, 0.0)).norm() < 1e-10);
    // |10> -> |01>
    assert!((matrix[[1, 2]] - c(1.0, 0.0)).norm() < 1e-10);
}

// ============================================================================
// CircuitGate (Nested Circuit) Tests
// ============================================================================

#[test]
fn test_circuit_gate_bell_state() {
    // Create a Bell state preparation circuit as a gate
    let mut inner = Circuit::new(2);
    inner.h(Qubit::new(0)).unwrap();
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let gate = inner.to_gate("Bell").unwrap();

    let mut circuit = Circuit::new(2);
    circuit
        .append(gate, [Qubit::new(0), Qubit::new(1)], [], None)
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // Verify Bell state preparation
    // |00> -> (|00> + |11>) / sqrt(2)
    let s = 1.0 / SQRT_2;
    assert!((matrix[[0, 0]] - c(s, 0.0)).norm() < 1e-10);
    assert!((matrix[[3, 0]] - c(s, 0.0)).norm() < 1e-10);
    assert!((matrix[[1, 0]]).norm() < 1e-10);
    assert!((matrix[[2, 0]]).norm() < 1e-10);
}

#[test]
fn test_circuit_gate_with_params() {
    // Circuit gate with parameters
    let theta = Parameter::symbol("theta");
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), theta).unwrap();

    let gate = inner.to_gate("RxGate").unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .append(
            gate,
            [Qubit::new(0)],
            [ParameterValue::Fixed(PI / 2.0)],
            None,
        )
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, 0.0), c(0.0, -s)], [c(0.0, -s), c(s, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

// ============================================================================
// Composite Circuit Tests
// ============================================================================

#[test]
fn test_composite_h_cnot() {
    // H on q0, then CNOT(q0, q1) - Bell state preparation
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // Expected: (|00> + |11>) / sqrt(2) from |00>
    let s = 1.0 / SQRT_2;
    assert!((matrix[[0, 0]] - c(s, 0.0)).norm() < 1e-10);
    assert!((matrix[[3, 0]] - c(s, 0.0)).norm() < 1e-10);
}

#[test]
fn test_composite_x_h() {
    // X then H: |0> -> |1> -> |->
    let mut circuit = Circuit::new(1);
    circuit.x(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // H * X = [[1,1],[1,-1]]/sqrt(2) * [[0,1],[1,0]] = [[1,-1],[1,1]]/sqrt(2) (up to column ordering)
    // Actually: H * X = [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]] @ [[0, 1], [1, 0]]
    //                = [[1/sqrt(2), 1/sqrt(2)], [-1/sqrt(2), 1/sqrt(2)]]
    let s = 1.0 / SQRT_2;
    let expected = array![[c(s, 0.0), c(s, 0.0)], [c(-s, 0.0), c(s, 0.0)],];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_composite_swap_via_cnots() {
    // SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // Expected SWAP matrix
    let expected = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_empty_circuit() {
    let circuit = Circuit::new(2);

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let expected = eye(4);

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_custom_qubit_order() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();

    // Test with reversed qubit order
    let custom_order = vec![1, 0];
    let matrix = circuit_to_matrix(&circuit, Some(&custom_order)).unwrap();

    // H on qubit 0 in original, but with order [1,0], qubit 0 becomes LSB (index 0)
    // q1 becomes MSB (index 1)
    // So H should act on the LSB (q0), and I acts on MSB (q1)
    // Matrix should be I ⊗ H
    // I ⊗ H = [[1,0],[0,1]] ⊗ [[s,s],[s,-s]] = [[s,s,0,0],[s,-s,0,0],[0,0,s,s],[0,0,s,-s]]
    let s = 1.0 / SQRT_2;
    let expected = array![
        [c(s, 0.0), c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(s, 0.0), c(-s, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0), c(s, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0), c(-s, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
#[should_panic(expected = "qubits_order mismatch")]
fn test_invalid_qubit_order() {
    let circuit = Circuit::new(2);
    let invalid_order = vec![0, 2]; // qubit 2 doesn't exist
    let _ = circuit_to_matrix(&circuit, Some(&invalid_order));
}

// ============================================================================
// Parameterized Circuit Tests
// ============================================================================

#[test]
fn test_parameterized_circuit_error() {
    // Circuit with symbolic parameters should return error
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta).unwrap();

    let result = circuit_to_matrix(&circuit, None);
    assert!(result.is_err());
}

// ============================================================================
// Large Circuit Performance Test
// ============================================================================

#[test]
fn test_five_qubit_ghz() {
    // GHZ state preparation on 5 qubits
    let mut circuit = Circuit::new(5);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(2), Qubit::new(3)).unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(4)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);

    // Verify GHZ: |00000> -> (|00000> + |11111>) / sqrt(2)
    let s = 1.0 / SQRT_2;
    assert!((matrix[[0, 0]] - c(s, 0.0)).norm() < 1e-10);
    assert!((matrix[[31, 0]] - c(s, 0.0)).norm() < 1e-10);
}

// ============================================================================
// Special Cases
// ============================================================================

#[test]
fn test_multiple_single_qubit_on_different_qubits() {
    // X on q0, H on q1 - should be X ⊗ H
    let mut circuit = Circuit::new(2);
    circuit.x(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    // X ⊗ H = [[0, 0, H00, H01], [0, 0, H10, H11], [H00, H01, 0, 0], [H10, H11, 0, 0]]
    // But with standard ordering: |00>, |01>, |10>, |11>
    // X on q0: swaps |0> and |1> on q0
    // H on q1: applies H on q1
    // Combined: X ⊗ H
    let s = 1.0 / SQRT_2;
    let expected = array![
        [c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0), c(s, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(s, 0.0), c(-s, 0.0)],
        [c(s, 0.0), c(s, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(s, 0.0), c(-s, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
}

#[test]
fn test_rzz_gate() {
    let mut circuit = Circuit::new(2);
    circuit.rzz(Qubit::new(0), Qubit::new(1), PI / 2.0).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    let s = 1.0 / SQRT_2;
    let expected = array![
        [c(s, -s), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(s, s), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(s, s), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s, -s)],
    ];

    assert_matrix_approx_eq(&matrix, &expected, 1e-10);
    assert_is_unitary(&matrix, 1e-10);
}

#[test]
fn test_fsim_gate() {
    let mut circuit = Circuit::new(2);
    circuit
        .fsim(Qubit::new(0), Qubit::new(1), PI / 4.0, PI / 2.0)
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_is_unitary(&matrix, 1e-10);
}
