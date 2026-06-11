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

//! Shared helpers for compositional MCX synthesis.

use super::DECOMPOSE_MCX_NAME;
use crate::circuit::operation::ValueOperation;
use crate::circuit::value_instruction::ValueInstruction;
use crate::compile::error::CompilerError;

/// Returns the inverse of a parameter-free operation sequence.
///
/// Requiring empty parameters keeps the inversion boundary explicit: this
/// helper does not translate [`crate::circuit::ParameterValue`] expressions
/// into [`crate::circuit::Parameter`] expressions. Qubit ordering and labels
/// are preserved while operation ordering is reversed.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when an input operation has
/// parameters, its instruction is not invertible, or its inverse unexpectedly
/// produces parameters.
pub(super) fn invert_parameter_free_operations(
    operations: &[ValueOperation],
) -> Result<Vec<ValueOperation>, CompilerError> {
    let mut inverse_operations = Vec::with_capacity(operations.len());
    for operation in operations.iter().rev() {
        if !operation.params.is_empty() {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: "MCX operation inversion requires parameter-free operations".to_string(),
            });
        }

        let ValueInstruction::Instruction(inner_inst) = &operation.instruction else {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: format!(
                    "MCX operation inversion does not support instruction {:?}",
                    operation.instruction
                ),
            });
        };
        let (inverse_inst, inverse_params) =
            inner_inst
                .inverse(&[])
                .ok_or_else(|| CompilerError::TransformFailed {
                    name: DECOMPOSE_MCX_NAME,
                    reason: format!(
                        "MCX operation inversion does not support instruction {:?}",
                        inner_inst
                    ),
                })?;
        if !inverse_params.is_empty() {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: "MCX operation inversion unexpectedly produced instruction parameters"
                    .to_string(),
            });
        }

        inverse_operations.push(ValueOperation {
            instruction: ValueInstruction::Instruction(inverse_inst),
            qubits: operation.qubits.clone(),
            params: Default::default(),
            label: operation.label.clone(),
        });
    }

    Ok(inverse_operations)
}
