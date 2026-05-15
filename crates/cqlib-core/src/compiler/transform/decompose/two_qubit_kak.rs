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

//! Numeric two-qubit KAK (Cartan / Weyl) decomposition primitive.
//!
//! Decomposes an arbitrary 4×4 unitary into
//!
//! ```text
//! U = exp(i·g) · (K1l ⊗ K1r) · exp(i·(a·XX + b·YY + c·ZZ)) · (K2l ⊗ K2r)
//! ```
//!
//! where `g` is a global phase, `K1l, K1r, K2l, K2r ∈ SU(2)`, and
//! `a, b, c` are the Cartan (Weyl) coordinates satisfying
//! `π/4 ≥ a ≥ b ≥ |c|`.
//!
//! This module intentionally exposes only the numerical primitive. It does not
//! lower the interaction core into a target-native gate set or wire the
//! primitive into the compiler workflow.
//!
//! The implementation follows the standard magic-basis route:
//!
//! 1. remove the global phase and work in `SU(4)`;
//! 2. transform into the magic basis, where local `SU(2) ⊗ SU(2)` factors are
//!    represented by real orthogonal matrices;
//! 3. diagonalize `M2 = M^T M` by a real Autonne factorization;
//! 4. fold the raw interaction coordinates into the Weyl chamber while
//!    applying the matching local Clifford corrections;
//! 5. recover the four single-qubit local factors from `SO(4)`.

use crate::circuit::StandardGate;
use crate::compiler::error::CompilerError;
use faer::Mat;
use faer::Side::Lower;
use ndarray::{Array1, Array2, array};
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

/// Name used in compiler errors emitted by this synthesis primitive.
const SYNTHESIS_NAME: &str = "decompose.two_qubit_kak";

/// Input unitarity tolerance.
///
/// This primitive currently accepts numerically unitary matrices rather than
/// projecting arbitrary near-unitary inputs onto the unitary group.
const UNITARY_EPS: f64 = 1e-8;

/// Final entrywise reconstruction tolerance.
///
/// The richer Frobenius diagnostics in `validate_reconstruction` are reported
/// on failure, while this entrywise bound remains the pass/fail threshold.
const RECONSTRUCTION_EPS: f64 = 1e-7;

/// Autonne-stage tolerance derived from the accepted input unitarity tolerance.
const AUTONNE_EPS: f64 = 10.0 * UNITARY_EPS;

/// Rank and reconstruction tolerance for extracting `SU(2) ⊗ SU(2)` factors.
const KRON_RANK_EPS: f64 = 1e-8;

/// Result of a two-qubit KAK decomposition.
///
/// Represents:
///
/// ```text
/// U = exp(i*global_phase)
///   * (K1l ⊗ K1r)
///   * RXX(-2a) * RYY(-2b) * RZZ(-2c)
///   * (K2l ⊗ K2r)
/// ```
///
/// The Cartan coordinates satisfy the Weyl chamber constraint
/// `pi/4 >= a >= b >= |c|`, and if `a = pi/4` then `c >= 0`.
#[derive(Debug, Clone)]
pub(crate) struct KakDecomposition {
    /// Circuit-level global phase multiplying the local and interaction parts.
    pub(crate) global_phase: f64,
    /// Left local gate on qubit 1 (applied after the interaction).
    pub(crate) k1l: Array2<Complex64>,
    /// Left local gate on qubit 0 (applied after the interaction).
    pub(crate) k1r: Array2<Complex64>,
    /// Right local gate on qubit 1 (applied before the interaction).
    pub(crate) k2l: Array2<Complex64>,
    /// Right local gate on qubit 0 (applied before the interaction).
    pub(crate) k2r: Array2<Complex64>,
    /// Largest Cartan coordinate (corresponds to XX interaction).
    pub(crate) a: f64,
    /// Middle Cartan coordinate (corresponds to YY interaction).
    pub(crate) b: f64,
    /// Smallest Cartan coordinate (corresponds to ZZ interaction).
    pub(crate) c: f64,
}

// ---------------------------------------------------------------------------
// Magic basis constants (un-normalized)
// ---------------------------------------------------------------------------

/// Un-normalized magic basis matrix B.
///
/// ```text
/// B = [[1,  i, 0,  0],
///      [0,  0, i,  1],
///      [0,  0, i, -1],
///      [1, -i, 0,  0]]
/// ```
static B_RAW: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ],
    [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(1.0, 0.0),
    ],
    [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
    ],
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ],
];

/// B† with 0.5 normalization factor so that B†·B = I.
///
/// This matches Qiskit's `B_NON_NORMALIZED_DAGGER`. Without the 0.5 factor,
/// B†·B = 2I, which introduces spurious scaling that only cancels incidentally.
static B_DAG_RAW: [[Complex64; 4]; 4] = [
    [
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.5, 0.0),
    ],
    [
        Complex64::new(0.0, -0.5),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.5),
    ],
    [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, -0.5),
        Complex64::new(0.0, -0.5),
        Complex64::new(0.0, 0.0),
    ],
    [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(-0.5, 0.0),
        Complex64::new(0.0, 0.0),
    ],
];

// i·X, i·Y, i·Z are special-unitary Pauli factors used by the Weyl chamber
// folding step. Multiplying these into the local gates changes signs or folds
// Cartan coordinates by local equivalence while preserving the represented
// two-qubit unitary up to the tracked global phase.
const IPX: [[Complex64; 2]; 2] = [
    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
];
const IPY: [[Complex64; 2]; 2] = [
    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
];
const IPZ: [[Complex64; 2]; 2] = [
    [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
];

// ---------------------------------------------------------------------------
// Fixed-size matrix helpers
// ---------------------------------------------------------------------------

/// Returns `0.5 * (A + A^T)`.
///
/// `M^T M` is theoretically complex symmetric. Symmetrizing after a checked
/// tolerance removes harmless roundoff asymmetry before the real Autonne step.
fn mat4_symmetrized(a: &Array2<Complex64>) -> Array2<Complex64> {
    Array2::from_shape_fn(a.dim(), |(row, col)| (a[[row, col]] + a[[col, row]]) * 0.5)
}

/// Computes the maximum elementwise absolute difference between two matrices.
fn max_entry_diff(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    let mut max_diff = 0.0_f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            max_diff = max_diff.max((a[[i, j]] - b[[i, j]]).norm());
        }
    }
    max_diff
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Performs the KAK decomposition of a 4×4 unitary matrix.
///
/// Returns a [`KakDecomposition`] such that
/// `U ≈ exp(i·g) · (K1l ⊗ K1r) · RXX(-2a) · RYY(-2b) · RZZ(-2c) · (K2l ⊗ K2r)`.
pub(crate) fn kak_decompose(matrix: &Array2<Complex64>) -> Result<KakDecomposition, CompilerError> {
    validate_input(matrix)?;

    // Remove the global phase by taking a fourth root of det(U). The KAK
    // interaction and local factors are then computed for an `SU(4)` matrix.
    let det_u_raw = Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        let value = matrix[[row, col]];
        faer::c64::new(value.re, value.im)
    })
    .as_ref()
    .determinant();
    let det_u = Complex64::new(det_u_raw.re, det_u_raw.im);
    let det_pow = det_u.powf(-0.25);
    let u_su4 = matrix.mapv(|value| det_pow * value);
    let mut global_phase = det_u.arg() / 4.0;

    // Transform into the magic basis. In this basis, local gates become real
    // orthogonal matrices, allowing the non-local diagonal factor to be found
    // from a symmetric Autonne decomposition.
    let b = Array2::from_shape_fn((4, 4), |(row, col)| B_RAW[row][col]);
    let b_dag = Array2::from_shape_fn((4, 4), |(row, col)| B_DAG_RAW[row][col]);
    let u_p = b_dag.dot(&u_su4.dot(&b));

    // Build the symmetric matrix used to recover the right orthogonal factor.
    // KAK uses `M^T M` in the magic basis; replacing this with `M†M` would
    // destroy the Autonne structure.
    let u_p_t = u_p.t().to_owned();
    let m2_raw = u_p_t.dot(&u_p);
    validate_complex_symmetric("U_p^T U_p", &m2_raw)?;
    let m2 = mat4_symmetrized(&m2_raw);
    validate_unitary_4x4("U_p^T U_p", &m2, AUTONNE_EPS)?;

    // Autonne decomposition: M2 = P * diag(d) * P^T, where P is real
    // orthogonal and each d[i] lies on the unit circle for exact inputs.
    let (p, d) = autonne_decompose(&m2)?;

    // Principal-branch angle extraction. The subsequent Weyl folding logic is
    // responsible for mapping these raw interaction parameters into the
    // canonical chamber while preserving local equivalence.
    let mut d_angles = [0.0f64; 4];
    for i in 0..4 {
        d_angles[i] = -d[i].arg() / 2.0;
    }
    // Enforce the zero-trace interaction convention by deriving the fourth
    // diagonal phase from the first three principal-branch phases.
    d_angles[3] = -(d_angles[0] + d_angles[1] + d_angles[2]);

    // Convert diagonal magic-basis phases into the three raw Cartan
    // coordinates used by the subsequent Weyl chamber folding logic.
    let mut cs = [0.0f64; 3];
    for i in 0..3 {
        cs[i] = ((d_angles[i] + d_angles[3]) / 2.0).rem_euclid(TAU);
    }

    // Sort raw interaction coordinates by folded magnitude. The same
    // permutation must be applied to the Autonne columns and phase angles so
    // that local factors remain synchronized with the Cartan core.
    let cstemp: [f64; 3] = cs.map(|x| {
        let y = x.rem_euclid(FRAC_PI_2);
        y.min(FRAC_PI_2 - y)
    });
    let mut sort_order = [0usize, 1, 2];
    sort_order.sort_by(|&a, &b| cstemp[a].partial_cmp(&cstemp[b]).unwrap());
    // Qiskit permutation: [second, third, first]
    let order = [sort_order[1], sort_order[2], sort_order[0]];

    let cs_orig = cs;
    let d_orig = d_angles;
    let p_orig = p.clone();
    for i in 0..3 {
        cs[i] = cs_orig[order[i]];
        d_angles[i] = d_orig[order[i]];
    }
    let mut p = p;
    for i in 0..3 {
        for row in 0..4 {
            p[[row, i]] = p_orig[[row, order[i]]];
        }
    }

    // Ensure P has det = +1 (SO(4)).
    let det_p = Mat::from_fn(p.nrows(), p.ncols(), |row, col| p[[row, col]].re)
        .as_ref()
        .determinant();
    if (det_p.abs() - 1.0).abs() > AUTONNE_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Autonne P determinant {det_p}, expected ±1"),
        });
    }
    if det_p < 0.0 {
        for row in 0..4 {
            p[[row, 3]] = -p[[row, 3]];
        }
    }

    // Recover local `SU(2) ⊗ SU(2)` factors from the magic basis before final
    // Weyl reflections. The reflections below are easier to express directly
    // on the extracted single-qubit factors.
    //
    // With M2 = P * D^2 * P^T and K2 = P^T, this matrix is D^{-1}; the
    // sign convention follows the `d_angles = -arg(d) / 2` branch above.
    let diag_d_inv: Array2<Complex64> = Array2::from_diag(&Array1::from_vec(
        d_angles.map(|x| Complex64::from_polar(1.0, x)).to_vec(),
    ));
    let u_p_p_d_inv = u_p.dot(&p.dot(&diag_d_inv));
    let p_t = p.t().to_owned();

    // K1 = B * U_p * P * D^{-1} * B†
    let k1_magic = b.dot(&u_p_p_d_inv.dot(&b_dag));
    // K2 = B * P^T * B†
    let k2_magic = b.dot(&p_t.dot(&b_dag));

    // Decompose K1 and K2 as Kronecker products of single-qubit SU(2) gates.
    let (mut k1l, mut k1r, phase_l) = decompose_two_qubit_product_gate(&k1_magic)?;
    let (k2l, mut k2r, phase_r) = decompose_two_qubit_product_gate(&k2_magic)?;
    global_phase += phase_l + phase_r;

    // Weyl chamber reflections.
    //
    // The invariant through this block is:
    //
    // (K1l ⊗ K1r) · A(cs) · (K2l ⊗ K2r)
    //
    // represents the same two-qubit unitary up to `global_phase`, where
    // A(cs) = exp(i(cs[1] XX + cs[0] YY + cs[2] ZZ)) in the coordinate order
    // used by the Qiskit-style folding logic below. Multiplying by iX, iY, or
    // iZ on selected local factors applies local Clifford equivalences that
    // either shift one coordinate by π/2 or reflect it back across a Weyl
    // chamber wall.
    let ipx = Array2::from_shape_fn((2, 2), |(row, col)| IPX[row][col]);
    let ipy = Array2::from_shape_fn((2, 2), |(row, col)| IPY[row][col]);
    let ipz = Array2::from_shape_fn((2, 2), |(row, col)| IPZ[row][col]);

    // Period-fold the first two coordinates when they lie past π/2. The
    // paired `iY`/`iX` updates are local Clifford factors that absorb the
    // extra Cartan-period contribution.
    if cs[0] > FRAC_PI_2 {
        cs[0] -= 3.0 * FRAC_PI_2;
        k1l = k1l.dot(&ipy);
        k1r = k1r.dot(&ipy);
        global_phase += FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_2 {
        cs[1] -= 3.0 * FRAC_PI_2;
        k1l = k1l.dot(&ipx);
        k1r = k1r.dot(&ipx);
        global_phase += FRAC_PI_2;
    }

    // Reflect coordinates above π/4 back into the principal Weyl chamber.
    // `conjs` tracks whether an odd number of these reflections has occurred;
    // the third coordinate must be handled differently in that case to keep
    // the chamber boundary convention consistent.
    let mut conjs = 0u32;
    if cs[0] > FRAC_PI_4 {
        cs[0] = FRAC_PI_2 - cs[0];
        k1l = k1l.dot(&ipy);
        k2r = ipy.dot(&k2r);
        conjs += 1;
        global_phase -= FRAC_PI_2;
    }
    if cs[1] > FRAC_PI_4 {
        cs[1] = FRAC_PI_2 - cs[1];
        k1l = k1l.dot(&ipx);
        k2r = ipx.dot(&k2r);
        conjs += 1;
        global_phase += FRAC_PI_2;
        if conjs == 1 {
            global_phase -= PI;
        }
    }

    // Fold the third coordinate. Its sign is allowed to remain negative, but
    // its absolute value must not exceed the middle Weyl coordinate.
    if cs[2] > FRAC_PI_2 {
        cs[2] -= 3.0 * FRAC_PI_2;
        k1l = k1l.dot(&ipz);
        k1r = k1r.dot(&ipz);
        global_phase += FRAC_PI_2;
        if conjs == 1 {
            global_phase -= PI;
        }
    }
    if conjs == 1 {
        cs[2] = FRAC_PI_2 - cs[2];
        k1l = k1l.dot(&ipz);
        k2r = ipz.dot(&k2r);
        global_phase += FRAC_PI_2;
    }
    if cs[2] > FRAC_PI_4 {
        cs[2] -= FRAC_PI_2;
        k1l = k1l.dot(&ipz);
        k1r = k1r.dot(&ipz);
        global_phase -= FRAC_PI_2;
    }

    // Final Weyl coordinates: a = cs[1], b = cs[0], c = cs[2]
    let a = cs[1];
    let b = cs[0];
    let c = cs[2];

    let decomp = KakDecomposition {
        global_phase,
        k1l,
        k1r,
        k2l,
        k2r,
        a,
        b,
        c,
    };

    validate_weyl_chamber(a, b, c)?;
    validate_local_su2(&decomp)?;
    // Reconstruction is the final guard against sign-convention or branch
    // mistakes in the numerical decomposition path.
    validate_reconstruction(matrix, &decomp)?;
    Ok(decomp)
}

/// Checks the canonical Weyl chamber convention:
/// `π/4 >= a >= b >= |c|`, with `c >= 0` on the `a = π/4` boundary.
fn validate_weyl_chamber(a: f64, b: f64, c: f64) -> Result<(), CompilerError> {
    let eps = 1e-8;
    if a < -eps || b < -eps {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Weyl chamber violation: a={a}, b={b} must be non-negative"),
        });
    }
    if a > FRAC_PI_4 + eps {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Weyl chamber violation: a={a} > π/4"),
        });
    }
    if a + eps < b {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Weyl chamber violation: a={a} < b={b}"),
        });
    }
    if b + eps < c.abs() {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Weyl chamber violation: b={b} < |c|={}", c.abs()),
        });
    }
    if (a - FRAC_PI_4).abs() < eps && c < -eps {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Weyl chamber violation: a=π/4 but c={c} < 0"),
        });
    }
    Ok(())
}

/// Checks that the recovered local factors are 2×2 special-unitary matrices.
fn validate_local_su2(decomp: &KakDecomposition) -> Result<(), CompilerError> {
    let eps = 1e-8;
    for (name, m) in [
        ("K1l", &decomp.k1l),
        ("K1r", &decomp.k1r),
        ("K2l", &decomp.k2l),
        ("K2r", &decomp.k2r),
    ] {
        // Check 2×2 shape before performing fixed-size algebraic checks.
        if m.nrows() != 2 || m.ncols() != 2 {
            return Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!("{name} is not 2x2"),
            });
        }
        // Check unitarity: M†M ≈ I.
        let m_dag = m.t().mapv(|value| value.conj());
        let product = m_dag.dot(m);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (product[[i, j]] - expected).norm();
                if diff > eps {
                    return Err(CompilerError::TransformFailed {
                        name: SYNTHESIS_NAME,
                        reason: format!("{name} is not unitary: (M†M)[{i},{j}] differs by {diff}"),
                    });
                }
            }
        }
        // Check det ≈ 1 so the extracted factors are special-unitary.
        let det = det_2x2(m);
        if (det - Complex64::new(1.0, 0.0)).norm() > eps {
            return Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!("{name} det={det}, expected 1"),
            });
        }
    }
    Ok(())
}

/// Validates the public numerical input contract for this primitive.
fn validate_input(matrix: &Array2<Complex64>) -> Result<(), CompilerError> {
    if matrix.nrows() != 4 || matrix.ncols() != 4 {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "expected 4x4 matrix, got {}x{}",
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

    validate_unitary_4x4("input matrix", matrix, UNITARY_EPS)
}

/// Verifies 4×4 unitarity using the max entry residual of `M†M - I`.
fn validate_unitary_4x4(
    matrix_name: &'static str,
    matrix: &Array2<Complex64>,
    eps: f64,
) -> Result<(), CompilerError> {
    let matrix_dag = matrix.t().mapv(|value| value.conj());
    let product = matrix_dag.dot(matrix);
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[row, col]] - expected).norm();
            if diff > eps {
                return Err(CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!(
                        "{matrix_name} is not unitary: (M†M)[{row},{col}] differs by {diff}"
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Validates the complex-symmetric precondition used by the Autonne path.
fn validate_complex_symmetric(
    name: &'static str,
    matrix: &Array2<Complex64>,
) -> Result<(), CompilerError> {
    let max_diff = max_entry_diff(matrix, &matrix.t().to_owned());

    if max_diff > AUTONNE_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("{name} is not complex symmetric: max transpose difference {max_diff}"),
        });
    }

    Ok(())
}

/// Rebuilds the decomposition with Cqlib's `RXX/RYY/RZZ` sign convention and
/// rejects results that do not match the original matrix entrywise.
fn validate_reconstruction(
    original: &Array2<Complex64>,
    decomp: &KakDecomposition,
) -> Result<(), CompilerError> {
    let k1 = ndarray::linalg::kron(&decomp.k1l, &decomp.k1r);
    let k2 = ndarray::linalg::kron(&decomp.k2l, &decomp.k2r);

    // Cqlib's rotation gates use the `RPP(theta) = exp(-i theta/2 PP)`
    // convention, so the Cartan interaction `exp(i a XX)` is reconstructed as
    // `RXX(-2a)`, and similarly for YY and ZZ.
    let rxx = StandardGate::RXX.matrix(&[-2.0 * decomp.a]).map_err(|e| {
        CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("failed to get RXX matrix: {e}"),
        }
    })?;
    let ryy = StandardGate::RYY.matrix(&[-2.0 * decomp.b]).map_err(|e| {
        CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("failed to get RYY matrix: {e}"),
        }
    })?;
    let rzz = StandardGate::RZZ.matrix(&[-2.0 * decomp.c]).map_err(|e| {
        CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("failed to get RZZ matrix: {e}"),
        }
    })?;

    let core = rxx.dot(&ryy.dot(&rzz.dot(&k2)));
    let left = k1.dot(&core);
    let phase = Complex64::from_polar(1.0, decomp.global_phase);
    let reconstructed = left.mapv(|value| phase * value);

    let error = reconstruction_error(original, &reconstructed);

    if error.max_entry > RECONSTRUCTION_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "KAK reconstruction failed: max_entry={}, frobenius={}, phase_invariant_frobenius={}, global_phase={}, cartan=({}, {}, {})",
                error.max_entry,
                error.frobenius,
                error.phase_invariant_frobenius,
                decomp.global_phase,
                decomp.a,
                decomp.b,
                decomp.c
            ),
        });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ReconstructionError {
    /// Maximum absolute entrywise residual.
    max_entry: f64,
    /// Frobenius norm of the direct matrix difference.
    frobenius: f64,
    /// Frobenius residual after allowing one best-fit global phase.
    phase_invariant_frobenius: f64,
}

/// Computes several reconstruction diagnostics used for failure messages.
fn reconstruction_error(
    expected: &Array2<Complex64>,
    actual: &Array2<Complex64>,
) -> ReconstructionError {
    let mut max_entry = 0.0_f64;
    let mut frobenius_sq = 0.0_f64;
    let mut expected_norm_sq = 0.0_f64;
    let mut actual_norm_sq = 0.0_f64;
    let mut inner = Complex64::new(0.0, 0.0);

    for row in 0..expected.nrows() {
        for col in 0..expected.ncols() {
            let expected_value = expected[[row, col]];
            let actual_value = actual[[row, col]];
            let diff = actual_value - expected_value;
            max_entry = max_entry.max(diff.norm());
            frobenius_sq += diff.norm_sqr();
            expected_norm_sq += expected_value.norm_sqr();
            actual_norm_sq += actual_value.norm_sqr();
            inner += expected_value.conj() * actual_value;
        }
    }

    ReconstructionError {
        max_entry,
        frobenius: frobenius_sq.sqrt(),
        phase_invariant_frobenius: (expected_norm_sq + actual_norm_sq - 2.0 * inner.norm())
            .max(0.0)
            .sqrt(),
    }
}

/// Deterministic real Autonne decomposition for the symmetric unitary `M2`.
///
/// `M2 = A + iB` with real symmetric, commuting `A` and `B`. We first
/// diagonalize `A`; when `A` has a degenerate eigenspace, `B` is diagonalized
/// inside that eigenspace to obtain a common real orthogonal eigenbasis.
fn autonne_decompose(
    m2: &Array2<Complex64>,
) -> Result<(Array2<Complex64>, [Complex64; 4]), CompilerError> {
    validate_autonne_input(m2)?;

    // For symmetric unitary `M2`, real and imaginary parts commute. A real
    // common eigenbasis therefore gives the orthogonal Autonne factor.
    let real_part = Mat::from_fn(4, 4, |i, j| m2[[i, j]].re);
    let real_decomp =
        real_part
            .self_adjoint_eigen(Lower)
            .map_err(|e| CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!("failed to diagonalize Re(M2): {e:?}"),
            })?;

    let real_basis = real_decomp.U();
    let real_eigenvalues = real_decomp.S();
    let mut p_real = [[0.0_f64; 4]; 4];
    let mut start = 0usize;

    while start < 4 {
        // Group numerically degenerate eigenvalues of Re(M2). Inside such a
        // block, Re(M2) alone does not choose a deterministic basis.
        let mut end = start + 1;
        while end < 4 && (real_eigenvalues[end] - real_eigenvalues[start]).abs() <= AUTONNE_EPS {
            end += 1;
        }

        let width = end - start;
        if width == 1 {
            for row in 0..4 {
                p_real[row][start] = real_basis[(row, start)];
            }
        } else {
            // Resolve the degenerate real eigenspace by diagonalizing Im(M2)
            // projected into that eigenspace, yielding a shared real basis.
            let imag_block = Mat::from_fn(width, width, |row, col| {
                let mut value = 0.0_f64;
                for i in 0..4 {
                    for j in 0..4 {
                        value += real_basis[(i, start + row)]
                            * m2[[i, j]].im
                            * real_basis[(j, start + col)];
                    }
                }
                value
            });
            let imag_decomp = imag_block.self_adjoint_eigen(Lower).map_err(|e| {
                CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!("failed to diagonalize Im(M2) degeneracy block: {e:?}"),
                }
            })?;
            let block_basis = imag_decomp.U();

            for local_col in 0..width {
                for row in 0..4 {
                    let mut value = 0.0_f64;
                    for basis_col in 0..width {
                        value += real_basis[(row, start + basis_col)]
                            * block_basis[(basis_col, local_col)];
                    }
                    p_real[row][start + local_col] = value;
                }
            }
        }

        start = end;
    }

    let p = Array2::from_shape_fn((4, 4), |(row, col)| Complex64::new(p_real[row][col], 0.0));
    validate_real_orthogonal("Autonne P", &p)?;

    let p_t = p.t().to_owned();
    let p_t_m2 = p_t.dot(m2);
    let p_t_m2_p = p_t_m2.dot(&p);
    // In the Autonne form, the diagonal entries are the squared interaction
    // phases; off-diagonal entries are checked below as a numerical residual.
    let d: [Complex64; 4] = [
        p_t_m2_p[[0, 0]],
        p_t_m2_p[[1, 1]],
        p_t_m2_p[[2, 2]],
        p_t_m2_p[[3, 3]],
    ];

    let mut max_off_diag = 0.0_f64;
    for row in 0..4 {
        for col in 0..4 {
            if row != col {
                max_off_diag = max_off_diag.max(p_t_m2_p[[row, col]].norm());
            }
        }
    }
    if max_off_diag > AUTONNE_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Autonne diagonalization left off-diagonal residual {max_off_diag}"),
        });
    }

    let diag_d = Array2::from_diag(&Array1::from_vec(d.to_vec()));
    let compare = p.dot(&diag_d.dot(&p_t));
    let mut max_err = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            max_err = max_err.max((compare[[i, j]] - m2[[i, j]]).norm());
        }
    }
    if max_err > AUTONNE_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Autonne reconstruction failed: max entry difference {max_err}"),
        });
    }

    Ok((p, d))
}

/// Validates algebraic preconditions for the real Autonne route.
fn validate_autonne_input(m2: &Array2<Complex64>) -> Result<(), CompilerError> {
    let mut max_commutator = 0.0_f64;
    for row in 0..4 {
        for col in 0..4 {
            let mut ab = 0.0_f64;
            let mut ba = 0.0_f64;
            for k in 0..4 {
                ab += m2[[row, k]].re * m2[[k, col]].im;
                ba += m2[[row, k]].im * m2[[k, col]].re;
            }
            max_commutator = max_commutator.max((ab - ba).abs());
        }
    }

    if max_commutator > AUTONNE_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "Autonne input Re(M2) and Im(M2) do not commute: max commutator {max_commutator}"
            ),
        });
    }

    Ok(())
}

/// Verifies that a matrix is real orthogonal to Autonne tolerance.
fn validate_real_orthogonal(
    name: &'static str,
    m: &Array2<Complex64>,
) -> Result<(), CompilerError> {
    for row in 0..4 {
        for col in 0..4 {
            if m[[row, col]].im.abs() > AUTONNE_EPS {
                return Err(CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!("{name} has non-real entry at ({row}, {col})"),
                });
            }
        }
    }

    let product = m.t().dot(m);
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[row, col]] - expected).norm();
            if diff > AUTONNE_EPS {
                return Err(CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!(
                        "{name} is not orthogonal: (P^T P)[{row},{col}] differs by {diff}"
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Given an SU(4) matrix that is a Kronecker product `L ⊗ R`,
/// extract L, R ∈ SU(2) and a phase such that `M = exp(i*phase) * L ⊗ R`.
fn decompose_two_qubit_product_gate(
    m: &Array2<Complex64>,
) -> Result<(Array2<Complex64>, Array2<Complex64>, f64), CompilerError> {
    // Rearrange K_{(i,j),(k,l)} into R_{(i,k),(j,l)}. For an exact product
    // `L ⊗ R`, this matrix is `vec(L) vec(R)^T` and therefore rank one.
    let rearranged = Array2::from_shape_fn((4, 4), |(row, col)| {
        let left_row = row / 2;
        let left_col = row % 2;
        let right_row = col / 2;
        let right_col = col % 2;
        m[[2 * left_row + right_row, 2 * left_col + right_col]]
    });

    let svd = Mat::from_fn(rearranged.nrows(), rearranged.ncols(), |row, col| {
        let value = rearranged[[row, col]];
        faer::c64::new(value.re, value.im)
    })
    .thin_svd()
    .map_err(|e| CompilerError::TransformFailed {
        name: SYNTHESIS_NAME,
        reason: format!("Kronecker rank-1 SVD failed: {e:?}"),
    })?;
    let singular_values = svd.S();
    let leading = singular_values[0].re;
    let trailing = if singular_values.dim() > 1 {
        singular_values[1].re.abs()
    } else {
        0.0
    };

    if leading <= KRON_RANK_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: "Kronecker factorization has zero leading singular value".to_string(),
        });
    }
    if trailing > KRON_RANK_EPS * leading.max(1.0) {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "Kronecker factorization is not rank-1: second singular value {trailing}"
            ),
        });
    }

    // The leading singular vectors recover the two vectorized factors. `faer`
    // returns `A = U S V†`, so the right factor uses `conj(V[:, 0])`.
    let scale = leading.sqrt();
    let u = svd.U();
    let v = svd.V();
    let l_vec: [Complex64; 4] = std::array::from_fn(|i| {
        let value = u[(i, 0)];
        scale * Complex64::new(value.re, value.im)
    });
    let r_vec: [Complex64; 4] = std::array::from_fn(|i| {
        let value = v[(i, 0)];
        scale * Complex64::new(value.re, value.im).conj()
    });

    let l_raw = array![[l_vec[0], l_vec[1]], [l_vec[2], l_vec[3]]];
    let r_raw = array![[r_vec[0], r_vec[1]], [r_vec[2], r_vec[3]]];

    let det_l = det_2x2(&l_raw);
    let det_r = det_2x2(&r_raw);
    if det_l.norm() < KRON_RANK_EPS || det_r.norm() < KRON_RANK_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!(
                "Kronecker factors have near-zero determinants: det_l={det_l}, det_r={det_r}"
            ),
        });
    }

    // Normalize both factors to determinant one. Each complex square root has
    // two valid branches, so try all sign choices and keep the branch with the
    // smallest product reconstruction error.
    let det_l_sqrt = det_l.sqrt();
    let det_r_sqrt = det_r.sqrt();
    let branch_one = Complex64::new(1.0, 0.0);
    let branch_minus_one = Complex64::new(-1.0, 0.0);

    let mut best: Option<(Array2<Complex64>, Array2<Complex64>, f64, f64)> = None;
    for sign_l in [branch_one, branch_minus_one] {
        for sign_r in [branch_one, branch_minus_one] {
            let sqrt_l = sign_l * det_l_sqrt;
            let sqrt_r = sign_r * det_r_sqrt;
            let l = l_raw.mapv(|value| value / sqrt_l);
            let r = r_raw.mapv(|value| value / sqrt_r);
            let phase = (sqrt_l * sqrt_r).arg();
            let phase_factor = Complex64::from_polar(1.0, phase);
            let reconstructed = ndarray::linalg::kron(&l, &r).mapv(|value| phase_factor * value);
            let error = max_entry_diff(&reconstructed, m);

            let is_better = match best.as_ref() {
                Some((_, _, _, best_error)) => error < *best_error,
                None => true,
            };
            if is_better {
                best = Some((l, r, phase, error));
            }
        }
    }

    let Some((l, r, phase, max_diff)) = best else {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: "Kronecker product decomposition produced no phase branch".to_string(),
        });
    };

    if max_diff > KRON_RANK_EPS {
        return Err(CompilerError::TransformFailed {
            name: SYNTHESIS_NAME,
            reason: format!("Kronecker product decomposition failed: max difference {max_diff}"),
        });
    }

    Ok((l, r, phase))
}

/// Determinant of a 2×2 complex matrix.
fn det_2x2(m: &Array2<Complex64>) -> Complex64 {
    m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]
}
