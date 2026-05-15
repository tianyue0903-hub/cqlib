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

//! Numeric single-qubit unitary synthesis.
//!
//! This module converts an arbitrary numeric 2×2 unitary into Cqlib's
//! `U(theta, phi, lambda)` convention plus a separately tracked global phase.
//! It is intentionally a numerical primitive: callers own circuit insertion,
//! parameter symbolic handling, and any target-basis lowering after synthesis.

use crate::circuit::StandardGate;
use crate::compiler::error::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

const SYNTHESIS_NAME: &str = "decompose.one_qubit_unitary";
/// Threshold for treating a matrix entry magnitude or angle as zero.
const ZERO_EPS: f64 = 1e-12;
/// Accepted residual for the input unitarity check.
const UNITARY_EPS: f64 = 1e-10;
/// Entrywise tolerance used to reject internally inconsistent decompositions.
const RECONSTRUCTION_EPS: f64 = 1e-9;

/// Decomposition of a numeric 2x2 unitary into Cqlib's `U` gate convention
/// plus a circuit-level global phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OneQubitUDecomposition {
    /// Polar angle in Cqlib's `U(theta, phi, lambda)` convention.
    pub(crate) theta: f64,
    /// Phase on the lower-left amplitude in the `U` convention.
    pub(crate) phi: f64,
    /// Phase on the upper-right amplitude in the `U` convention.
    pub(crate) lambda: f64,
    /// Circuit-level phase satisfying `matrix = exp(i*g) * U(...)`.
    pub(crate) global_phase: f64,
}

/// Synthesizes a numeric 2x2 unitary matrix as:
///
/// `matrix = exp(i * global_phase) * U(theta, phi, lambda)`.
pub(crate) fn synthesize_one_qubit_unitary_as_u(
    matrix: &Array2<Complex64>,
) -> Result<OneQubitUDecomposition, CompilerError> {
    validate_one_qubit_unitary(matrix)?;

    let a = matrix[[0, 0]];
    let b = matrix[[0, 1]];
    let c = matrix[[1, 0]];
    let d = matrix[[1, 1]];
    let a_norm = a.norm();
    let c_norm = c.norm();

    // Cqlib's U gate has |a| = cos(theta/2) and |c| = sin(theta/2).
    // Deriving theta from magnitudes avoids phase-branch instability.
    let theta = normalize_angle(2.0 * c_norm.atan2(a_norm));
    let decomposition = if c_norm <= ZERO_EPS {
        // Diagonal unitary: theta=0 makes lambda unobservable, so fold the
        // relative diagonal phase into phi and keep lambda at a stable zero.
        OneQubitUDecomposition {
            theta: 0.0,
            phi: normalize_angle(d.im.atan2(d.re) - a.im.atan2(a.re)),
            lambda: 0.0,
            global_phase: normalize_angle(a.im.atan2(a.re)),
        }
    } else if a_norm <= ZERO_EPS {
        // Anti-diagonal unitary: theta=π makes the top-left entry vanish, so
        // choose a zero global phase and read phases from the non-zero entries.
        OneQubitUDecomposition {
            theta: PI,
            phi: normalize_angle(c.im.atan2(c.re)),
            lambda: normalize_angle((-b).im.atan2((-b).re)),
            global_phase: 0.0,
        }
    } else {
        // Generic case: use the phase of the top-left entry as the global
        // phase, then express the remaining two observable phases relative to it.
        let global_phase = a.im.atan2(a.re);
        OneQubitUDecomposition {
            theta,
            phi: normalize_angle(c.im.atan2(c.re) - global_phase),
            lambda: normalize_angle((-b).im.atan2((-b).re) - global_phase),
            global_phase: normalize_angle(global_phase),
        }
    };

    validate_reconstruction(matrix, decomposition)?;
    Ok(decomposition)
}

/// Checks shape, finite entries, and `U†U ≈ I` for the input matrix.
fn validate_one_qubit_unitary(matrix: &Array2<Complex64>) -> Result<(), CompilerError> {
    if matrix.nrows() != 2 || matrix.ncols() != 2 {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "expected 2x2 matrix, got {}x{}",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }

    for ((row, col), value) in matrix.indexed_iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!("matrix contains non-finite element at ({row}, {col}): {value}"),
            });
        }
    }

    for row in 0..2 {
        for col in 0..2 {
            let value = matrix[[0, row]].conj() * matrix[[0, col]]
                + matrix[[1, row]].conj() * matrix[[1, col]];
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (value - expected).norm();
            if diff > UNITARY_EPS {
                return Err(CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!("matrix is not unitary: (U†U)[{row},{col}] differs by {diff}"),
                });
            }
        }
    }

    Ok(())
}

/// Reconstructs `exp(i*g) * U(theta, phi, lambda)` and compares it entrywise.
fn validate_reconstruction(
    matrix: &Array2<Complex64>,
    decomposition: OneQubitUDecomposition,
) -> Result<(), CompilerError> {
    let u_matrix =
        StandardGate::U.matrix(&[decomposition.theta, decomposition.phi, decomposition.lambda])?;
    let phase = Complex64::from_polar(1.0, decomposition.global_phase);

    let mut max_diff = 0.0_f64;
    for ((row, col), expected) in matrix.indexed_iter() {
        let actual = phase * u_matrix[[row, col]];
        max_diff = max_diff.max((actual - expected).norm());
    }

    if max_diff > RECONSTRUCTION_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "failed to reconstruct unitary within tolerance: max entry difference {max_diff}"
            ),
        });
    }

    Ok(())
}

/// Normalizes angles to `(-π, π]` and snaps tiny roundoff to exact zero.
fn normalize_angle(angle: f64) -> f64 {
    let normalized = (angle + PI).rem_euclid(TAU) - PI;
    let normalized = if normalized <= -PI + f64::EPSILON {
        PI
    } else {
        normalized
    };

    if normalized.abs() < ZERO_EPS {
        0.0
    } else {
        normalized
    }
}
