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

//! Multi-controlled `U(theta, phi, lambda)` synthesis primitives.
//!
//! The standard single-qubit unitary satisfies
//!
//! ```text
//! U(theta, phi, lambda)
//!   = exp(i * (phi + lambda) / 2)
//!     RZ(phi) RY(theta) RZ(lambda).
//! ```
//!
//! With controls, the scalar factor becomes an observable conditional phase.
//! This module emits that phase before delegating the three rotations to
//! [`super::rotation`].

use super::{
    phase::{decompose_phase_n_clean, decompose_phase_no_aux},
    rotation::{decompose_rotation_n_clean, decompose_rotation_no_aux},
};
use crate::circuit::operation::ValueOperation;
use crate::circuit::{Instruction, Parameter, ParameterValue, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

/// Decomposes a multi-controlled standard `U(theta, phi, lambda)` gate
/// without ancillary qubits.
///
/// Inputs with no controls emit the original standard `U` directly. Inputs
/// with controls emit the observable conditional phase followed by the
/// multi-controlled Z-Y-Z Euler decomposition.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when an input qubit is repeated
/// or an underlying phase or MC-SU(2) synthesis fails.
pub fn decompose_unitary_no_aux(
    theta: &ParameterValue,
    phi: &ParameterValue,
    lambda: &ParameterValue,
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_unitary_with(
        theta,
        phi,
        lambda,
        controls,
        target,
        |phase, controls, target| {
            decompose_phase_no_aux(StandardGate::Phase, Some(phase), controls, target)
        },
        |rotation, angle, controls, target| {
            decompose_rotation_no_aux(rotation, angle, controls, target)
        },
    )
}

/// Decomposes a multi-controlled standard `U(theta, phi, lambda)` gate using
/// clean ancillas.
///
/// Inputs with no controls emit the original standard `U` directly and ignore
/// `clean_ancillas`. Inputs with controls reuse the same ancillary qubits
/// sequentially for the conditional phase and all three rotations. The
/// ancillary-qubit contract is inherited from the largest emitted
/// multi-controlled rotation.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when an input qubit is repeated
/// or an underlying clean-accumulator synthesis fails.
pub fn decompose_unitary_n_clean(
    theta: &ParameterValue,
    phi: &ParameterValue,
    lambda: &ParameterValue,
    controls: &[Qubit],
    target: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_unitary_with(
        theta,
        phi,
        lambda,
        controls,
        target,
        |phase, controls, target| {
            decompose_phase_n_clean(
                StandardGate::Phase,
                Some(phase),
                controls,
                target,
                clean_ancillas,
            )
        },
        |rotation, angle, controls, target| {
            decompose_rotation_n_clean(rotation, angle, controls, target, clean_ancillas)
        },
    )
}

fn decompose_unitary_with(
    theta: &ParameterValue,
    phi: &ParameterValue,
    lambda: &ParameterValue,
    controls: &[Qubit],
    target: Qubit,
    mut decompose_phase: impl FnMut(
        &ParameterValue,
        &[Qubit],
        Qubit,
    ) -> Result<Vec<ValueOperation>, CompilerError>,
    mut decompose_rotation: impl FnMut(
        StandardGate,
        &ParameterValue,
        &[Qubit],
        Qubit,
    ) -> Result<Vec<ValueOperation>, CompilerError>,
) -> Result<Vec<ValueOperation>, CompilerError> {
    if controls.is_empty() {
        return Ok(vec![ValueOperation {
            instruction: Instruction::Standard(StandardGate::U),
            qubits: smallvec![target],
            params: smallvec![theta.clone(), phi.clone(), lambda.clone()],
            label: None,
        }]);
    }

    let Some((phase_target, phase_controls)) = controls.split_last() else {
        unreachable!("the zero-control case returns the standard U gate")
    };
    let conditional_phase =
        ParameterValue::from((Parameter::from(phi) + Parameter::from(lambda)) * 0.5);
    let mut operations = decompose_phase(&conditional_phase, phase_controls, *phase_target)?;
    operations.extend(decompose_rotation(
        StandardGate::RZ,
        lambda,
        controls,
        target,
    )?);
    operations.extend(decompose_rotation(
        StandardGate::RY,
        theta,
        controls,
        target,
    )?);
    operations.extend(decompose_rotation(StandardGate::RZ, phi, controls, target)?);
    Ok(operations)
}
