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

//! Global-phase equivalence checking for symbolic matrices and circuits.
//!
//! Provides [`symbolic_matrices_equivalent()`] and
//! [`circuits_equivalent()`] for comparing quantum
//! operations that may differ by a global phase factor. The check is
//! conservative and depends on the simplification power of
//! [`Parameter::simplify`](crate::circuit::Parameter::simplify).

use crate::circuit::error::ParameterError;
use crate::circuit::symbolic_matrix::gate::circuit_to_symbolic_matrix;
use crate::circuit::symbolic_matrix::matrix::SymbolicMatrix;
use crate::circuit::{Circuit, CircuitError};

/// Returns whether two symbolic matrices are equivalent up to a global phase.
///
/// This is a conservative symbolic checker, not a complete decision procedure.
/// It may return `false` for mathematically equivalent matrices if the required
/// identity cannot be reduced by `Parameter::simplify`.
pub fn symbolic_matrices_equivalent(
    lhs: &SymbolicMatrix,
    rhs: &SymbolicMatrix,
) -> Result<bool, ParameterError> {
    if lhs.shape() != rhs.shape() {
        return Ok(false);
    }

    let mut pivot = None;
    for ((row, col), lhs_value) in lhs.indexed_iter() {
        let rhs_value = &rhs[[row, col]];
        let lhs_zero = lhs_value.simplifies_to_zero()?;
        let rhs_zero = rhs_value.simplifies_to_zero()?;
        if lhs_zero != rhs_zero {
            return Ok(false);
        }
        if !lhs_zero && pivot.is_none() {
            pivot = Some((row, col));
        }
    }

    let Some((pivot_row, pivot_col)) = pivot else {
        return Ok(true);
    };
    let lhs_pivot = &lhs[[pivot_row, pivot_col]];
    let rhs_pivot = &rhs[[pivot_row, pivot_col]];

    for ((row, col), lhs_value) in lhs.indexed_iter() {
        let rhs_value = &rhs[[row, col]];
        if lhs_value.simplifies_to_zero()? && rhs_value.simplifies_to_zero()? {
            continue;
        }

        let residual = lhs_value * rhs_pivot - rhs_value * lhs_pivot;
        if !residual.simplifies_to_zero()? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Builds symbolic matrices for two circuits and compares them up to a global
/// phase in the requested qubit order.
pub fn circuits_equivalent(
    lhs: &Circuit,
    rhs: &Circuit,
    qubits_order: Option<&[usize]>,
) -> Result<bool, CircuitError> {
    let lhs_matrix = circuit_to_symbolic_matrix(lhs, qubits_order)?;
    let rhs_matrix = circuit_to_symbolic_matrix(rhs, qubits_order)?;
    symbolic_matrices_equivalent(&lhs_matrix, &rhs_matrix)
        .map_err(|err| CircuitError::InvalidOperation(err.to_string()))
}

#[cfg(test)]
#[path = "./equivalence_test.rs"]
mod equivalence_test;
