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

//! Trivial MCX decompositions for zero, one, or two controls.

use crate::circuit::{Instruction, Qubit, StandardGate, operation::ValueOperation};
use crate::compiler::error::CompilerError;
use crate::util::qubit::find_duplicate_qubit;
use smallvec::smallvec;

use super::DECOMPOSE_MCX_NAME;

/// Decomposes an MCX with at most two controls into one `X`, `CX`, or `CCX`
/// operation.
///
/// The returned operation has no parameters or label. Its qubits are ordered
/// with controls first and the target last. The `CCX` case is preserved rather
/// than decomposed into a lower-level basis.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when any qubit is repeated or
/// when more than two controls are provided. Callers must explicitly use a
/// general MCX synthesis algorithm for larger inputs.
pub fn decompose_mcx_small(
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    let target_group = [target];
    if let Some(qubit) = find_duplicate_qubit(&[controls, &target_group]) {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_MCX_NAME,
            reason: format!(
                "MCX controls, target, and ancillas must be distinct; duplicate {qubit}"
            ),
        });
    }

    let operation = match controls {
        [] => ValueOperation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![target],
            params: smallvec![],
            label: None,
        },
        [control] => ValueOperation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![*control, target],
            params: smallvec![],
            label: None,
        },
        [first_control, second_control] => ValueOperation {
            instruction: Instruction::Standard(StandardGate::CCX),
            qubits: smallvec![*first_control, *second_control, target],
            params: smallvec![],
            label: None,
        },
        _ => {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: format!(
                    "trivial MCX decomposition supports at most 2 controls, got {}",
                    controls.len()
                ),
            });
        }
    };

    Ok(vec![operation])
}
