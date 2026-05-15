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

use super::one_qubit_unitary::{OneQubitUDecomposition, synthesize_one_qubit_unitary_as_u};
use crate::circuit::StandardGate;
use crate::compiler::error::CompilerError;
use ndarray::{Array2, array};
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_3, PI};

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn matrix_from_u(theta: f64, phi: f64, lambda: f64, global_phase: f64) -> Array2<Complex64> {
    let u_matrix = StandardGate::U
        .matrix(&[theta, phi, lambda])
        .unwrap()
        .into_owned();
    let phase = Complex64::from_polar(1.0, global_phase);
    u_matrix.mapv(|value| phase * value)
}

fn reconstructed_matrix(decomposition: OneQubitUDecomposition) -> Array2<Complex64> {
    matrix_from_u(
        decomposition.theta,
        decomposition.phi,
        decomposition.lambda,
        decomposition.global_phase,
    )
}

fn assert_matrix_close(actual: &Array2<Complex64>, expected: &Array2<Complex64>) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());
    for ((row, col), actual_value) in actual.indexed_iter() {
        let expected_value = expected[[row, col]];
        let diff = (*actual_value - expected_value).norm();
        assert!(
            diff < 1e-9,
            "matrix mismatch at ({row}, {col}): actual {actual_value}, expected {expected_value}, diff {diff}"
        );
    }
}

fn assert_angle_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "angle mismatch: actual {actual}, expected {expected}"
    );
}

#[test]
fn synthesizes_identity() {
    let matrix = matrix_from_u(0.0, 0.0, 0.0, 0.0);

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_angle_close(decomposition.theta, 0.0);
    assert_angle_close(decomposition.phi, 0.0);
    assert_angle_close(decomposition.lambda, 0.0);
    assert_angle_close(decomposition.global_phase, 0.0);
    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_global_phase_identity() {
    let matrix = matrix_from_u(0.0, 0.0, 0.0, 0.37);

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_angle_close(decomposition.theta, 0.0);
    assert_angle_close(decomposition.global_phase, 0.37);
    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_x_gate() {
    let matrix = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_angle_close(decomposition.theta, PI);
    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_h_gate() {
    let matrix = array![
        [c(FRAC_1_SQRT_2, 0.0), c(FRAC_1_SQRT_2, 0.0)],
        [c(FRAC_1_SQRT_2, 0.0), c(-FRAC_1_SQRT_2, 0.0)]
    ];

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_generic_u_with_global_phase() {
    let matrix = matrix_from_u(0.73, -1.2, 2.4, -0.41);

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_theta_zero_branch() {
    let matrix = matrix_from_u(0.0, FRAC_PI_3, 0.0, -0.2);

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_angle_close(decomposition.theta, 0.0);
    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn synthesizes_theta_pi_branch() {
    let matrix = matrix_from_u(PI, -0.7, 1.1, 0.29);

    let decomposition = synthesize_one_qubit_unitary_as_u(&matrix).unwrap();

    assert_angle_close(decomposition.theta, PI);
    assert_matrix_close(&reconstructed_matrix(decomposition), &matrix);
}

#[test]
fn rejects_non_2x2_matrix() {
    let matrix = Array2::from_elem((3, 3), c(0.0, 0.0));

    let err = synthesize_one_qubit_unitary_as_u(&matrix).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("expected 2x2 matrix")
    ));
}

#[test]
fn rejects_non_finite_matrix() {
    let matrix = array![[c(f64::NAN, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];

    let err = synthesize_one_qubit_unitary_as_u(&matrix).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("non-finite element")
    ));
}

#[test]
fn rejects_non_unitary_matrix() {
    let matrix = array![[c(1.0, 0.0), c(1.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];

    let err = synthesize_one_qubit_unitary_as_u(&matrix).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("matrix is not unitary")
    ));
}
