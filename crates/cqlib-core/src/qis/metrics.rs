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

//! Quantum information metrics and measures for quantum states.
//!
//! This module provides a comprehensive collection of functions to calculate
//! various quantum information-theoretic quantities including:
//!
//! - **Purity**: Measures how "pure" a quantum state is (1.0 for pure states).
//! - **Fidelity**: Quantifies the similarity between two quantum states.
//! - **Entropy**: Von Neumann entropy measuring the mixedness of a state.
//! - **Trace Distance**: A metric on quantum states related to distinguishability.
//! - **Entanglement Measures**: Logarithmic negativity and related quantities.
//!
//! # Key Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`purity_pure`] | Purity of a statevector |
//! | [`purity_mixed`] | Purity of a density matrix |
//! | [`state_fidelity_pure`] | Fidelity between two statevectors |
//! | [`state_fidelity_mixed`] | Fidelity between two density matrices |
//! | [`state_fidelity_pure_mixed`] | Fidelity between pure and mixed states |
//! | [`trace_distance_pure`] | Trace distance for pure states |
//! | [`trace_distance_mixed`] | Trace distance for mixed states |
//! | [`entropy`] | Von Neumann entropy |
//! | [`partial_transpose`] | Partial transpose operation |
//! | [`logarithmic_negativity`] | Logarithmic negativity entanglement measure |
//!
//! # Mathematical Background
//!
//! Most functions in this module operate on either [`Statevector`] (pure states)
//! or [`DensityMatrix`] (mixed states). The implementations leverage efficient
//! linear algebra routines from the `faer` crate for eigendecompositions and
//! matrix operations.
//!
//! # Examples
//!
//! ```
//! use cqlib_core::qis::metrics::{purity_pure, state_fidelity_pure};
//! use cqlib_core::qis::state::Statevector;
//!
//! // Create two orthogonal states
//! let psi = Statevector::from_vec(vec![1.0, 0.0]).unwrap();
//! let phi = Statevector::from_vec(vec![0.0, 1.0]).unwrap();
//!
//! // Calculate purity (should be 1.0 for normalized pure states)
//! let purity = purity_pure(&psi).unwrap();
//! assert!((purity - 1.0).abs() < 1e-10);
//!
//! // Calculate fidelity (should be 0.0 for orthogonal states)
//! let fidelity = state_fidelity_pure(&psi, &phi).unwrap();
//! assert!(fidelity < 1e-10);
//! ```

use crate::qis::QisError;
use crate::qis::state::{DensityMatrix, Statevector};
use faer::{c64, mat::Mat};
use num_complex::Complex64;
use rayon::prelude::*;

/// Calculates the purity of a pure quantum state (Statevector).
/// For a valid, normalized Statevector, the purity is theoretically 1.0.
/// This function calculates the norm squared of the inner product <psi|psi>
/// to serve as a numerical validation of the state's normalization.
///
/// Returns 1.0 within numerical precision for normalized states.
pub fn purity_pure(sv: &Statevector) -> Result<f64, QisError> {
    let norm_sqr: f64 = sv.data.par_iter().map(|c| c.norm_sqr()).sum();
    Ok(norm_sqr)
}

/// Calculates the purity of a mixed quantum state (DensityMatrix).
/// Purity is defined as Tr(rho^2). Since rho is a Hermitian matrix,
/// Tr(rho^2) is equivalent to the sum of the squared magnitudes of all its elements.
///
/// Returns a value between 1/2^N (maximally mixed) and 1.0 (pure state).
pub fn purity_mixed(dm: &DensityMatrix) -> Result<f64, QisError> {
    let purity: f64 = dm.data.par_iter().map(|c| c.norm_sqr()).sum();
    Ok(purity)
}

/// Calculates the state fidelity between two pure quantum states (Statevectors).
/// Fidelity F(psi, phi) = |<psi|phi>|^2.
///
/// Returns a value between 0.0 (orthogonal) and 1.0 (identical).
pub fn state_fidelity_pure(sv1: &Statevector, sv2: &Statevector) -> Result<f64, QisError> {
    if sv1.num_qubits != sv2.num_qubits {
        return Err(QisError::QubitMismatch {
            expected: sv1.num_qubits,
            actual: sv2.num_qubits,
        });
    }

    let inner_product: Complex64 = sv1
        .data
        .par_iter()
        .zip(sv2.data.par_iter())
        .map(|(a, b)| a.conj() * b)
        .sum();

    Ok(inner_product.norm_sqr())
}

/// Calculates the trace distance between two pure quantum states (Statevectors).
/// For pure states, trace distance D(psi, phi) = sqrt(1 - |<psi|phi>|^2).
///
/// Returns a value between 0.0 (identical) and 1.0 (orthogonal).
pub fn trace_distance_pure(sv1: &Statevector, sv2: &Statevector) -> Result<f64, QisError> {
    let fidelity = state_fidelity_pure(sv1, sv2)?;
    // Ensure the value under sqrt is slightly positive even with float inaccuracy
    let val = (1.0 - fidelity).max(0.0);
    Ok(val.sqrt())
}

/// Calculates the state fidelity between a pure state (Statevector) and a mixed state (DensityMatrix).
/// Fidelity F(psi, rho) = <psi|rho|psi>.
///
/// Returns a value between 0.0 and 1.0.
pub fn state_fidelity_pure_mixed(sv: &Statevector, dm: &DensityMatrix) -> Result<f64, QisError> {
    if sv.num_qubits != dm.num_qubits {
        return Err(QisError::QubitMismatch {
            expected: sv.num_qubits,
            actual: dm.num_qubits,
        });
    }

    let dim = 1 << sv.num_qubits;
    // Calculate rho|psi>
    let mut rho_psi = vec![Complex64::new(0.0, 0.0); dim];

    // DensityMatrix data is flattened row-major.
    rho_psi.par_iter_mut().enumerate().for_each(|(i, res)| {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..dim {
            sum += dm.data[i * dim + j] * sv.data[j];
        }
        *res = sum;
    });

    // Calculate <psi|rho_psi>
    let exp_val: Complex64 = sv
        .data
        .par_iter()
        .zip(rho_psi.par_iter())
        .map(|(a, b)| a.conj() * b)
        .sum();

    // The result should be strictly real and positive for a valid density matrix
    Ok(exp_val.re.max(0.0))
}

fn to_c64(c: Complex64) -> c64 {
    c64::new(c.re, c.im)
}

/// Helper to convert a DensityMatrix to a faer Mat<c64>
pub fn density_matrix_to_faer(dm: &DensityMatrix) -> Mat<c64> {
    let dim = 1 << dm.num_qubits;
    Mat::from_fn(dim, dim, |row, col| to_c64(dm.data[row * dim + col]))
}

/// Calculates the von Neumann entropy of a mixed state S(rho) = -Tr(rho log2 rho).
/// The entropy is returned in units of bits (base-2 logarithm).
pub fn entropy(dm: &DensityMatrix) -> Result<f64, QisError> {
    let mat = density_matrix_to_faer(dm);
    // Density matrix is Hermitian
    let s: Vec<f64> = mat
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|e| {
            QisError::UnsupportedOperation(format!("Eigendecomposition failed: {:?}", e))
        })?;

    let mut ent = 0.0;
    for &eigval in &s {
        // Ignore zero or slightly negative eigenvalues due to numerical noise
        if eigval > 1e-12 {
            ent -= eigval * eigval.log2();
        }
    }

    Ok(ent)
}

/// Calculates the trace distance between two mixed quantum states (DensityMatrices).
/// D(rho, sigma) = 1/2 Tr|rho - sigma| = 1/2 \sum |lambda_i|
/// Since Tr(rho) = Tr(sigma) = 1, this is equivalent to the sum of positive eigenvalues of (rho - sigma).
pub fn trace_distance_mixed(dm1: &DensityMatrix, dm2: &DensityMatrix) -> Result<f64, QisError> {
    if dm1.num_qubits != dm2.num_qubits {
        return Err(QisError::QubitMismatch {
            expected: dm1.num_qubits,
            actual: dm2.num_qubits,
        });
    }

    let mat1 = density_matrix_to_faer(dm1);
    let mat2 = density_matrix_to_faer(dm2);
    let diff = mat1 - mat2;

    // The difference of two Hermitian matrices is Hermitian
    let s: Vec<f64> = diff
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|e| {
            QisError::UnsupportedOperation(format!("Eigendecomposition failed: {:?}", e))
        })?;

    let mut sum_pos = 0.0;
    for &eigval in &s {
        if eigval > 0.0 {
            sum_pos += eigval;
        }
    }

    Ok(sum_pos)
}

/// Calculates the state fidelity between two mixed states.
/// Fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
pub fn state_fidelity_mixed(dm1: &DensityMatrix, dm2: &DensityMatrix) -> Result<f64, QisError> {
    if dm1.num_qubits != dm2.num_qubits {
        return Err(QisError::QubitMismatch {
            expected: dm1.num_qubits,
            actual: dm2.num_qubits,
        });
    }

    let rho = density_matrix_to_faer(dm1);
    let sigma = density_matrix_to_faer(dm2);

    // 1. Calculate sqrt(rho)
    // rho = U D U^dag -> sqrt(rho) = U sqrt(D) U^dag
    let decomp_rho = rho.self_adjoint_eigen(faer::Side::Lower).map_err(|e| {
        QisError::UnsupportedOperation(format!("Eigendecomposition failed: {:?}", e))
    })?;
    let u = &decomp_rho.U();
    let s = decomp_rho.S();
    let s_vec: Vec<f64> = (0..s.dim()).map(|i| s[i].re).collect();

    let dim = 1 << dm1.num_qubits;
    let mut sqrt_d = Mat::<c64>::zeros(dim, dim);
    for i in 0..dim {
        let val = s_vec[i].max(0.0).sqrt();
        sqrt_d[(i, i)] = c64::new(val, 0.0);
    }

    // sqrt(rho) = U * sqrt_D * U^dag
    let sqrt_rho = u * &sqrt_d * u.adjoint();

    // 2. Calculate M = sqrt(rho) * sigma * sqrt(rho)
    let m = &sqrt_rho * sigma * &sqrt_rho;

    // 3. M is positive semi-definite (Hermitian), find its eigenvalues
    let m_s: Vec<f64> = m.self_adjoint_eigenvalues(faer::Side::Lower).map_err(|e| {
        QisError::UnsupportedOperation(format!("Eigendecomposition failed: {:?}", e))
    })?;

    // 4. Trace(sqrt(M)) = sum of square roots of eigenvalues of M
    let mut trace_sqrt_m = 0.0;
    for &eigval in &m_s {
        trace_sqrt_m += eigval.max(0.0).sqrt();
    }

    Ok(trace_sqrt_m * trace_sqrt_m)
}

/// Performs the partial transpose operation on a density matrix.
///
/// The partial transpose is a fundamental operation in quantum information theory
/// used to detect entanglement (via the Peres-Horodecki criterion). For the
/// specified target qubits (subsystem A), this operation transposes only the
/// indices corresponding to that subsystem, leaving other qubits unchanged.
///
/// # Arguments
///
/// * `dm` - The input density matrix.
/// * `target_qubits` - The qubit indices specifying subsystem A to be transposed.
///
/// # Returns
///
/// * `Ok(DensityMatrix)` - The partially transposed density matrix.
/// * `Err(QisError)` - If any target qubit index is out of bounds.
///
/// # Notes
///
/// The indexing convention assumes `idx = (bra << n) | ket`, where `bra` is the
/// column index and `ket` is the row index. This is consistent with the
/// flattened representation where the upper `n` bits encode the bra state.
pub fn partial_transpose(
    dm: &DensityMatrix,
    target_qubits: &[usize],
) -> Result<DensityMatrix, QisError> {
    let n = dm.num_qubits;
    for &q in target_qubits {
        if q >= n {
            return Err(QisError::IndexOutOfBounds {
                index: q,
                max: n - 1,
            });
        }
    }

    // Construct a bit mask for the qubits to be swapped (lower n bits for bra).
    let mut swap_mask = 0usize;
    for &q in target_qubits {
        swap_mask |= 1 << q;
    }

    // Create a new density matrix to store the result.
    let mut new_dm = DensityMatrix::zeros(n);

    // Perform the partial transpose using parallel bit manipulation.
    // This is extremely cache-efficient as it only involves index arithmetic.
    new_dm
        .data
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, val)| {
            // 1. Extract the bra (column) and ket (row) states on target qubits.
            let lower_swap_bits = idx & swap_mask;
            let upper_swap_bits = (idx >> n) & swap_mask;

            // 2. Clear the bits to be swapped from the current index.
            let mut src_idx = idx & !(swap_mask | (swap_mask << n));

            // 3. Cross-assign: original ket becomes bra, original bra becomes ket.
            src_idx |= upper_swap_bits; // Original ket state -> new bra state
            src_idx |= lower_swap_bits << n; // Original bra state -> new ket state

            // 4. Read the corresponding element from the original matrix.
            *val = dm.data[src_idx];
        });

    Ok(new_dm)
}

/// Calculates the logarithmic negativity of a quantum state.
///
/// The logarithmic negativity is an entanglement measure defined as:
/// E_N(ρ) = log₂ ||ρ^{T_A}||_1
///
/// where ρ^{T_A} is the partial transpose of the density matrix with respect to
/// subsystem A, and ||·||_1 denotes the trace norm (sum of singular values).
///
/// For separable states, E_N = 0. For entangled states, E_N > 0.
/// This quantity serves as an upper bound on distillable entanglement.
///
/// # Arguments
///
/// * `dm` - The density matrix of the bipartite quantum state.
/// * `sys_a` - The qubit indices comprising subsystem A.
///
/// # Returns
///
/// * `Ok(f64)` - The logarithmic negativity value (≥ 0).
/// * `Err(QisError)` - If eigendecomposition fails.
///
/// # Mathematical Background
///
/// The partial transpose of a separable state remains positive semi-definite
/// (Peres-Horodecki criterion). For entangled states, the partial transpose
/// develops negative eigenvalues, causing the trace norm to exceed 1.
pub fn logarithmic_negativity(dm: &DensityMatrix, sys_a: &[usize]) -> Result<f64, QisError> {
    // Step 1: Perform partial transpose on subsystem A.
    let pt_dm = partial_transpose(dm, sys_a)?;

    // Step 2: Convert the partially transposed matrix to faer format.
    let mat = density_matrix_to_faer(&pt_dm);

    // Step 3: Compute eigenvalues of the Hermitian matrix.
    // Note: The partial transpose remains Hermitian but may lose positive semi-definiteness.
    let eigenvalues: Vec<f64> = mat
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|e| {
            QisError::UnsupportedOperation(format!("Eigendecomposition failed: {:?}", e))
        })?;

    // Step 4: Compute the trace norm as the sum of absolute eigenvalues.
    let mut trace_norm = 0.0;
    for &eigval in &eigenvalues {
        trace_norm += eigval.abs();
    }

    // Step 5: Compute the base-2 logarithm.
    // For separable states, all eigenvalues are positive, trace_norm = 1, log2(1) = 0.
    // For entangled states, negative eigenvalues appear, trace_norm > 1, log2(>1) > 0.
    // The max(1.0) guards against numerical errors that might yield trace_norm < 1.
    let log_neg = trace_norm.max(1.0).log2();

    Ok(log_neg)
}

#[cfg(test)]
#[path = "metrics_test.rs"]
mod metrics_test;
