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

//! Multi-controlled Hadamard synthesis primitives.
//!
//! Hadamard is not special-unitary: `det(H) = -1`. The scalar phase in
//!
//! ```text
//! H = exp(i * pi / 2) RY(pi / 2) RZ(pi)
//! ```
//!
//! becomes observable after controls are added. This module emits that
//! conditional phase explicitly before delegating the rotations to
//! [`super::rotation`].

use super::{
    phase::{decompose_phase_n_clean, decompose_phase_no_aux},
    rotation::{decompose_rotation_n_clean, decompose_rotation_no_aux},
};
use crate::circuit::{ParameterValue, Qubit, StandardGate, operation::ValueOperation};
use crate::compile::error::CompilerError;
use crate::util::operation::push_standard_gate;
use std::f64::consts::PI;

/// Decomposes a multi-controlled Hadamard gate without ancillary qubits.
///
/// Inputs with no controls emit the original standard `H` directly.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when an input qubit is repeated
/// or an underlying phase or MC-SU(2) synthesis fails.
pub fn decompose_hadamard_no_aux(
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_hadamard_with(
        controls,
        target,
        |phase_controls, phase_target| {
            decompose_phase_no_aux(
                StandardGate::Phase,
                Some(&ParameterValue::Fixed(PI / 2.0)),
                phase_controls,
                phase_target,
            )
        },
        |rotation, theta| {
            decompose_rotation_no_aux(rotation, &ParameterValue::Fixed(theta), controls, target)
        },
    )
}

/// Decomposes a multi-controlled Hadamard gate using clean ancillas.
///
/// Inputs with no controls emit the original standard `H` directly and ignore
/// `clean_ancillas`. Recursive operations reuse the same ancillary qubits
/// sequentially.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when an input qubit is repeated
/// or an underlying clean-accumulator synthesis fails.
pub fn decompose_hadamard_n_clean(
    controls: &[Qubit],
    target: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_hadamard_with(
        controls,
        target,
        |phase_controls, phase_target| {
            decompose_phase_n_clean(
                StandardGate::Phase,
                Some(&ParameterValue::Fixed(PI / 2.0)),
                phase_controls,
                phase_target,
                clean_ancillas,
            )
        },
        |rotation, theta| {
            decompose_rotation_n_clean(
                rotation,
                &ParameterValue::Fixed(theta),
                controls,
                target,
                clean_ancillas,
            )
        },
    )
}

fn decompose_hadamard_with(
    controls: &[Qubit],
    target: Qubit,
    mut decompose_phase: impl FnMut(&[Qubit], Qubit) -> Result<Vec<ValueOperation>, CompilerError>,
    mut decompose_rotation: impl FnMut(StandardGate, f64) -> Result<Vec<ValueOperation>, CompilerError>,
) -> Result<Vec<ValueOperation>, CompilerError> {
    if controls.is_empty() {
        let mut operations = vec![];
        push_standard_gate(&mut operations, StandardGate::H, [target]);
        return Ok(operations);
    }

    let Some((phase_target, phase_controls)) = controls.split_last() else {
        unreachable!("the zero-control case returns the standard H gate")
    };
    let mut operations = decompose_phase(phase_controls, *phase_target)?;
    operations.extend(decompose_rotation(StandardGate::RZ, PI)?);
    operations.extend(decompose_rotation(StandardGate::RY, PI / 2.0)?);
    Ok(operations)
}
