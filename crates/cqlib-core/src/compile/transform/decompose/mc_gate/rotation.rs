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

//! Multi-controlled rotation synthesis primitives.
//!
//! This module normalizes standard rotation gates, including their intrinsic
//! single-control forms, before delegating to [`super::mc_su2`]. Callers must
//! provide flattened controls: a control intrinsic to `CRX`, `CRY`, or `CRZ`
//! is already present in `controls` and is not added again here.

use super::mc_su2::{Su2RotationAxis, decompose_mc_su2_n_clean, decompose_mc_su2_no_aux};
use crate::circuit::{ParameterValue, Qubit, StandardGate, operation::ValueOperation};
use crate::compile::error::CompilerError;

const DECOMPOSE_ROTATION_NAME: &str = "decompose.rotation";

/// Decomposes a multi-controlled standard rotation without ancillary qubits.
///
/// `rotation` may be `RX`, `RY`, `RZ`, `CRX`, `CRY`, or `CRZ`. `controls`
/// must contain all flattened controls, including any control intrinsic to
/// `rotation`.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when `rotation` is unsupported,
/// an input qubit is repeated, or the underlying MC-SU(2) synthesis fails.
pub fn decompose_rotation_no_aux(
    rotation: StandardGate,
    theta: &ParameterValue,
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_mc_su2_no_aux(rotation_axis(rotation)?, theta, controls, target)
}

/// Decomposes a multi-controlled standard rotation using clean ancillas.
///
/// `rotation` may be `RX`, `RY`, `RZ`, `CRX`, `CRY`, or `CRZ`. `controls`
/// must contain all flattened controls, including any control intrinsic to
/// `rotation`. The ancillary-qubit contract is inherited from
/// [`decompose_mc_su2_n_clean`].
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when `rotation` is unsupported
/// or the underlying clean-accumulator MC-SU(2) synthesis fails.
pub fn decompose_rotation_n_clean(
    rotation: StandardGate,
    theta: &ParameterValue,
    controls: &[Qubit],
    target: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_mc_su2_n_clean(
        rotation_axis(rotation)?,
        theta,
        controls,
        target,
        clean_ancillas,
    )
}

fn rotation_axis(rotation: StandardGate) -> Result<Su2RotationAxis, CompilerError> {
    match rotation {
        StandardGate::RX | StandardGate::CRX => Ok(Su2RotationAxis::X),
        StandardGate::RY | StandardGate::CRY => Ok(Su2RotationAxis::Y),
        StandardGate::RZ | StandardGate::CRZ => Ok(Su2RotationAxis::Z),
        _ => Err(CompilerError::TransformFailed {
            name: DECOMPOSE_ROTATION_NAME,
            reason: format!(
                "multi-controlled rotation decomposition supports only RX, CRX, RY, CRY, RZ, or CRZ, got {rotation}"
            ),
        }),
    }
}
