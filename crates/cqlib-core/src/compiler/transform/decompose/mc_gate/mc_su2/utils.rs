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

//! Shared MC-SU(2) synthesis helpers.

use super::{DECOMPOSE_MC_SU2_NAME, Su2RotationAxis};
use crate::circuit::{Instruction, ParameterValue, Qubit, StandardGate, operation::ValueOperation};
use crate::compiler::error::CompilerError;
use crate::util::qubit::find_duplicate_qubit;
use smallvec::smallvec;

/// Appends a single-parameter standard gate operation.
pub(super) fn push_parameterized_gate(
    operations: &mut Vec<ValueOperation>,
    gate: StandardGate,
    qubits: impl IntoIterator<Item = Qubit>,
    theta: &ParameterValue,
) {
    operations.push(ValueOperation {
        instruction: Instruction::Standard(gate),
        qubits: qubits.into_iter().collect(),
        params: smallvec![theta.clone()],
        label: None,
    });
}

/// Returns a scaled clone while preserving symbolic expressions.
pub(super) fn scale_parameter(theta: &ParameterValue, factor: f64) -> ParameterValue {
    match theta {
        ParameterValue::Fixed(value) => ParameterValue::Fixed(value * factor),
        ParameterValue::Param(parameter) => ParameterValue::Param(parameter.clone() * factor),
    }
}

/// Rejects a repeated qubit across the consumed input groups.
pub(super) fn validate_distinct_qubits(qubit_groups: &[&[Qubit]]) -> Result<(), CompilerError> {
    if let Some(qubit) = find_duplicate_qubit(qubit_groups) {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_MC_SU2_NAME,
            reason: format!(
                "MC-SU(2) controls, target, and ancillas must be distinct; duplicate {qubit}"
            ),
        });
    }

    Ok(())
}

/// Maps a rotation axis to its standard single-qubit gate.
pub(super) fn standard_rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::RX,
        Su2RotationAxis::Y => StandardGate::RY,
        Su2RotationAxis::Z => StandardGate::RZ,
    }
}

/// Maps a rotation axis to its standard controlled gate.
pub(super) fn standard_controlled_rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::CRX,
        Su2RotationAxis::Y => StandardGate::CRY,
        Su2RotationAxis::Z => StandardGate::CRZ,
    }
}
