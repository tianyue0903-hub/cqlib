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

//! Multi-controlled RZZ synthesis primitives.
//!
//! This module reduces a multi-controlled `RZZ` interaction to a central
//! multi-controlled `RZ` flanked by parity-computing `CX` gates:
//!
//! ```text
//! MC-RZZ(theta, controls, first, second)
//!   = CX(first, second)
//!     MC-RZ(theta, controls, second)
//!     CX(first, second)
//! ```
//!
//! The `CX` operations are unconditional — the parity bit is computed
//! regardless of the control states. Only the central `RZ` rotation is
//! subject to the multi-controlled condition. This is the canonical building
//! block for all two-qubit Pauli interaction rotations.

use super::rotation::{decompose_rotation_n_clean, decompose_rotation_no_aux};
use crate::circuit::{ParameterValue, Qubit, StandardGate, operation::ValueOperation};
use crate::compiler::error::CompilerError;
use crate::util::operation::push_standard_gate;

const DECOMPOSE_RZZ_NAME: &str = "decompose.rzz";

/// Decomposes a multi-controlled RZZ rotation without ancillary qubits.
///
/// `controls` are applied only to the central `RZ` rotation; the flanking
/// `CX` gates are always unconditional. Inputs with no controls emit a
/// bare `RZZ` standard gate.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when a qubit is repeated in
/// controls, `first`, or `second`, or when the underlying MC-RZ synthesis
/// fails.
pub fn decompose_mc_rzz_no_aux(
    theta: &ParameterValue,
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    if controls.is_empty() {
        if first == second {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_RZZ_NAME,
                reason: format!("RZZ interaction qubits must be distinct; both are {first}"),
            });
        }
        let mut operations = vec![];
        push_standard_gate(&mut operations, StandardGate::RZZ, [first, second]);
        operations[0].params.push(theta.clone());
        return Ok(operations);
    }

    check_rzz_qubits(controls, first, second, &[])?;

    let mut operations = Vec::new();
    push_standard_gate(&mut operations, StandardGate::CX, [first, second]);
    operations.extend(decompose_rotation_no_aux(
        StandardGate::RZ,
        theta,
        controls,
        second,
    )?);
    push_standard_gate(&mut operations, StandardGate::CX, [first, second]);
    Ok(operations)
}

/// Decomposes a multi-controlled RZZ rotation using clean ancillas.
///
/// `controls` are applied only to the central `RZ` rotation. The ancillary-
/// qubit contract is inherited from the underlying MC-RZ decomposition:
/// consumed ancillas must enter in `|0>` and are restored to `|0>`.
/// Extra ancillas are ignored.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when a qubit is repeated or
/// the underlying clean-accumulator MC-RZ synthesis fails.
pub fn decompose_mc_rzz_n_clean(
    theta: &ParameterValue,
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<ValueOperation>, CompilerError> {
    check_rzz_qubits(controls, first, second, clean_ancillas)?;

    let mut operations = Vec::new();
    push_standard_gate(&mut operations, StandardGate::CX, [first, second]);
    operations.extend(decompose_rotation_n_clean(
        StandardGate::RZ,
        theta,
        controls,
        second,
        clean_ancillas,
    )?);
    push_standard_gate(&mut operations, StandardGate::CX, [first, second]);
    Ok(operations)
}

fn check_rzz_qubits(
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
    ancillas: &[Qubit],
) -> Result<(), CompilerError> {
    if first == second {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_RZZ_NAME,
            reason: format!("RZZ interaction qubits must be distinct; both are {first}"),
        });
    }
    if let Some(qubit) = controls
        .iter()
        .find(|&qubit| *qubit == first || *qubit == second)
    {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_RZZ_NAME,
            reason: format!(
                "RZZ interaction qubits must not appear in controls; duplicate {qubit}"
            ),
        });
    }
    if let Some(qubit) = ancillas
        .iter()
        .find(|&qubit| *qubit == first || *qubit == second)
    {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_RZZ_NAME,
            reason: format!(
                "RZZ interaction qubits must not appear in ancillas; duplicate {qubit}"
            ),
        });
    }
    Ok(())
}
