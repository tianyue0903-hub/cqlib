// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Numeric single-qubit unitary synthesis.
//!
//! This module is deliberately circuit-agnostic. It decomposes one concrete
//! 2x2 unitary matrix into Cqlib's `U(theta, phi, lambda)` convention plus a
//! scalar global phase. Circuit traversal and parameter-table rebuilding live
//! in `unitary.rs`.
//!
//! The input must be a finite 2x2 unitary matrix. The implementation verifies
//! `M^dagger M = I` within `UNITARY_EPS` and rejects matrices with an invalid
//! determinant. The returned values satisfy:
//!
//! ```text
//! M = exp(i * global_phase) * U(theta, phi, lambda)
//! ```
//!
//! The angles are not canonicalized beyond the formulas needed for
//! reconstruction. Callers should treat the returned global phase as part of
//! the decomposition contract, including for identity-equivalent matrices.

use crate::compile::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;

const UNITARY_EPS: f64 = 1e-8;

pub(super) fn synthesize_numeric_1q_unitary(
    matrix: &Array2<Complex64>,
) -> Result<([f64; 3], f64), CompilerError> {
    if matrix.shape() != [2, 2] {
        return Err(CompilerError::InvalidInput(format!(
            "1q unitary synthesis expects a 2x2 matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }

    for ((row, col), value) in matrix.indexed_iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CompilerError::InvalidInput(format!(
                "1q unitary matrix contains non-finite element at ({row}, {col}): {value}"
            )));
        }
    }

    let product = matrix.t().mapv(|value| value.conj()).dot(matrix);
    for row in 0..2 {
        for col in 0..2 {
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[row, col]] - expected).norm();
            if diff > UNITARY_EPS {
                return Err(CompilerError::InvalidInput(format!(
                    "1q unitary matrix is not unitary: (Mdag M)[{row},{col}] differs by {diff}"
                )));
            }
        }
    }

    let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
    if det.norm() <= UNITARY_EPS || !det.re.is_finite() || !det.im.is_finite() {
        return Err(CompilerError::InvalidInput(
            "1q unitary matrix has invalid determinant".to_string(),
        ));
    }

    let det_arg = det.arg();
    let theta = 2.0 * matrix[[1, 0]].norm().atan2(matrix[[0, 0]].norm());
    let ang1 = matrix[[1, 1]].arg();
    let ang2 = matrix[[1, 0]].arg();
    let phi = ang1 + ang2 - det_arg;
    let lambda = ang1 - ang2;
    let global_phase = 0.5 * det_arg - 0.5 * (phi + lambda);

    Ok(([theta, phi, lambda], global_phase))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{StandardGate, gate::gate_matrix};
    use approx::assert_abs_diff_eq;

    fn assert_reconstructs(source: &Array2<Complex64>) {
        let ([theta, phi, lambda], global_phase) = synthesize_numeric_1q_unitary(source).unwrap();
        let reconstructed =
            gate_matrix::u_gate(theta, phi, lambda) * Complex64::from_polar(1.0, global_phase);
        assert_abs_diff_eq!(source, &reconstructed, epsilon = 1e-10);
    }

    #[test]
    fn numeric_synthesis_reconstructs_matrix() {
        let phase = Complex64::from_polar(1.0, -0.2);
        let source = gate_matrix::u_gate(0.8, -0.3, 0.6) * phase;

        let ([theta, phi, lambda], global_phase) = synthesize_numeric_1q_unitary(&source).unwrap();
        let reconstructed =
            gate_matrix::u_gate(theta, phi, lambda) * Complex64::from_polar(1.0, global_phase);

        assert_abs_diff_eq!(source, reconstructed, epsilon = 1e-10);
    }

    #[test]
    fn numeric_synthesis_reconstructs_singular_angle_cases() {
        for (theta, phi, lambda, global_phase) in [
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.7, -0.2, 0.37),
            (std::f64::consts::PI, -0.7, 1.1, 0.29),
        ] {
            let source =
                gate_matrix::u_gate(theta, phi, lambda) * Complex64::from_polar(1.0, global_phase);
            let ([theta, phi, lambda], global_phase) =
                synthesize_numeric_1q_unitary(&source).unwrap();
            let reconstructed =
                gate_matrix::u_gate(theta, phi, lambda) * Complex64::from_polar(1.0, global_phase);

            assert_abs_diff_eq!(source, reconstructed, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_non_unitary_matrix() {
        let matrix = ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)]
        ];

        let err = synthesize_numeric_1q_unitary(&matrix).unwrap_err();
        assert!(err.to_string().contains("not unitary"));
    }

    #[test]
    fn reconstructs_identity_with_global_phase() {
        let source =
            Array2::eye(2).mapv(|value: Complex64| value * Complex64::from_polar(1.0, 0.73));

        assert_reconstructs(&source);
    }

    #[test]
    fn reconstructs_common_standard_gate_matrices() {
        for matrix in [
            StandardGate::I.matrix(&[]).unwrap().into_owned(),
            StandardGate::X.matrix(&[]).unwrap().into_owned(),
            StandardGate::Y.matrix(&[]).unwrap().into_owned(),
            StandardGate::Z.matrix(&[]).unwrap().into_owned(),
            StandardGate::H.matrix(&[]).unwrap().into_owned(),
            StandardGate::S.matrix(&[]).unwrap().into_owned(),
            StandardGate::SDG.matrix(&[]).unwrap().into_owned(),
            StandardGate::T.matrix(&[]).unwrap().into_owned(),
            StandardGate::TDG.matrix(&[]).unwrap().into_owned(),
            StandardGate::RX.matrix(&[0.37]).unwrap().into_owned(),
            StandardGate::RY.matrix(&[-0.91]).unwrap().into_owned(),
            StandardGate::RZ.matrix(&[1.23]).unwrap().into_owned(),
            StandardGate::Phase.matrix(&[-0.44]).unwrap().into_owned(),
        ] {
            assert_reconstructs(&matrix);
        }
    }

    #[test]
    fn reconstructs_u_gate_angle_grid_with_global_phase() {
        for (theta, phi, lambda, phase) in [
            (1.0e-12, 0.2, -0.3, 0.4),
            (0.3, 0.0, 0.0, -0.5),
            (1.2, -0.8, 0.6, 0.75),
            (std::f64::consts::PI - 1.0e-12, 0.9, -1.1, -0.2),
        ] {
            let source =
                gate_matrix::u_gate(theta, phi, lambda) * Complex64::from_polar(1.0, phase);
            assert_reconstructs(&source);
        }
    }

    #[test]
    fn rejects_invalid_shape_and_non_finite_elements() {
        let bad_shape = Array2::<Complex64>::eye(3);
        let err = synthesize_numeric_1q_unitary(&bad_shape).unwrap_err();
        assert!(err.to_string().contains("expects a 2x2 matrix"));

        let non_finite = ndarray::array![
            [Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
        ];
        let err = synthesize_numeric_1q_unitary(&non_finite).unwrap_err();
        assert!(err.to_string().contains("non-finite element"));
    }
}
