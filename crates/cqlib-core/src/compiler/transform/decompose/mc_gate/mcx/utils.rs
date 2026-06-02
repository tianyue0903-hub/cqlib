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

//! Shared helpers for compositional MCX synthesis.

use super::DECOMPOSE_MCX_NAME;
use crate::circuit::Qubit;
use crate::circuit::operation::ValueOperation;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::decompose::mc_gate::mcx::relative_phase::emit_relative_phase_toffoli;
use crate::qis::Statevector;
#[cfg(test)]
use crate::util::test_utils::assert_value_operations_equal;
use num_complex::Complex64;

pub const EPSILON: f64 = 1e-10;

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

        let (instruction, inverse_params) =
            operation
                .instruction
                .inverse(&[])
                .ok_or_else(|| CompilerError::TransformFailed {
                    name: DECOMPOSE_MCX_NAME,
                    reason: format!(
                        "MCX operation inversion does not support instruction {:?}",
                        operation.instruction
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
            instruction,
            qubits: operation.qubits.clone(),
            params: Default::default(),
            label: operation.label.clone(),
        });
    }

    Ok(inverse_operations)
}

#[cfg(test)]
pub fn selected_basis_states(total_width: usize) -> Vec<usize> {
    let mask = (1_usize << total_width) - 1;
    let alternating_low = (0..total_width)
        .filter(|index| index % 2 == 0)
        .fold(0_usize, |state, index| state | (1_usize << index));
    let mut states = vec![
        0,
        1,
        1_usize << (total_width - 1),
        mask,
        alternating_low,
        mask ^ alternating_low,
    ];
    states.sort_unstable();
    states.dedup();
    states
}

#[cfg(test)]
pub fn single_nonzero_statevector_output(statevector: &Statevector) -> (usize, Complex64) {
    let outputs: Vec<_> = statevector
        .data()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, amplitude)| amplitude.norm() >= EPSILON)
        .collect();
    assert_eq!(outputs.len(), 1, "statevector has outputs {outputs:?}");
    outputs[0]
}

#[cfg(test)]
pub fn assert_rccx_expansion(
    operations: &[ValueOperation],
    first_control: Qubit,
    second_control: Qubit,
    target: Qubit,
) {
    let mut expected = vec![];
    emit_relative_phase_toffoli(&mut expected, first_control, second_control, target).unwrap();
    assert_value_operations_equal(operations, &expected);
}
