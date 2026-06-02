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

//! Multi-controlled Pauli rotation synthesis primitives.
//!
//! This module provides a uniform entry point for the two-qubit Pauli
//! interaction rotations `RXX`, `RYY`, `RZZ`, and `RZX`. Each rotation is
//! reduced to a multi-controlled `RZZ` decomposition via basis changes,
//! so that only one core synthesis algorithm (`MC-RZZ`) is maintained.
//!
//! # Basis-change summary
//!
//! | Gate | Pre-conjugation                          | Post-conjugation                         |
//! |------|------------------------------------------|------------------------------------------|
//! | RXX  | H(first), H(second)                      | H(first), H(second)                      |
//! | RYY  | RX(π/2)(first), RX(π/2)(second)          | RX(−π/2)(first), RX(−π/2)(second)        |
//! | RZZ  | (none)                                   | (none)                                   |
//! | RZX  | H(second)                                | H(second)                                |
//!
//! For `RZX`, `first` carries the Z-axis interaction and `second` the X-axis
//! interaction (matches the standard definition `exp(-i θ/2 · Z⊗X)`).

use super::rzz::{decompose_mc_rzz_n_clean, decompose_mc_rzz_no_aux};
use crate::circuit::{Instruction, ParameterValue, Qubit, StandardGate, operation::ValueOperation};
use crate::compiler::error::CompilerError;
use crate::util::operation::push_standard_gate;
use smallvec::smallvec;
use std::f64::consts::PI;

const DECOMPOSE_PAULI_ROTATION_NAME: &str = "decompose.pauli_rotation";

/// Decomposes a multi-controlled two-qubit Pauli rotation without ancillary
/// qubits.
///
/// `rotation` must be one of `RXX`, `RYY`, `RZZ`, or `RZX`. `controls` are
/// applied only to the central `RZ` rotation inside the underlying MC-RZZ
/// decomposition. Inputs with no controls emit a bare standard gate.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when `rotation` is unsupported
/// or the underlying MC-RZZ synthesis fails.
pub fn decompose_pauli_rotation_no_aux(
    rotation: StandardGate,
    theta: &ParameterValue,
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_pauli_rotation_with(
        rotation,
        theta,
        controls,
        first,
        second,
        |theta, controls| decompose_mc_rzz_no_aux(theta, controls, first, second),
    )
}

/// Decomposes a multi-controlled two-qubit Pauli rotation using clean
/// ancillas.
///
/// `rotation` must be one of `RXX`, `RYY`, `RZZ`, or `RZX`. The ancillary-
/// qubit contract is inherited from the underlying MC-RZZ decomposition.
///
/// # Errors
///
/// Returns [`CompilerError::TransformFailed`] when `rotation` is unsupported
/// or the underlying clean-ancilla MC-RZZ synthesis fails.
pub fn decompose_pauli_rotation_n_clean(
    rotation: StandardGate,
    theta: &ParameterValue,
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<ValueOperation>, CompilerError> {
    decompose_pauli_rotation_with(
        rotation,
        theta,
        controls,
        first,
        second,
        |theta, controls| decompose_mc_rzz_n_clean(theta, controls, first, second, clean_ancillas),
    )
}

fn decompose_pauli_rotation_with(
    rotation: StandardGate,
    theta: &ParameterValue,
    controls: &[Qubit],
    first: Qubit,
    second: Qubit,
    mut decompose_rzz: impl FnMut(
        &ParameterValue,
        &[Qubit],
    ) -> Result<Vec<ValueOperation>, CompilerError>,
) -> Result<Vec<ValueOperation>, CompilerError> {
    let basis = basis_change_for(rotation)?;

    if controls.is_empty() {
        let mut operations = vec![];
        push_pauli_rotation_gate(&mut operations, rotation, theta, first, second);
        return Ok(operations);
    }

    let mut operations = Vec::new();
    emit_conjugations(&mut operations, &basis.pre, first, second);
    operations.extend(decompose_rzz(theta, controls)?);
    emit_conjugations(&mut operations, &basis.post, first, second);
    Ok(operations)
}

struct BasisChange {
    pre: Vec<Conjugation>,
    post: Vec<Conjugation>,
}

#[derive(Debug, Clone, Copy)]
struct Conjugation {
    gate: ConjugationGate,
    which: WhichQubit,
}

#[derive(Debug, Clone, Copy)]
enum ConjugationGate {
    H,
    RxPi2,
    RxNegPi2,
}

#[derive(Debug, Clone, Copy)]
enum WhichQubit {
    First,
    Second,
}

fn basis_change_for(rotation: StandardGate) -> Result<BasisChange, CompilerError> {
    match rotation {
        StandardGate::RZZ => Ok(BasisChange {
            pre: vec![],
            post: vec![],
        }),
        StandardGate::RXX => {
            let both = vec![
                Conjugation {
                    gate: ConjugationGate::H,
                    which: WhichQubit::First,
                },
                Conjugation {
                    gate: ConjugationGate::H,
                    which: WhichQubit::Second,
                },
            ];
            Ok(BasisChange {
                pre: both.clone(),
                post: both,
            })
        }
        StandardGate::RYY => {
            let pre = vec![
                Conjugation {
                    gate: ConjugationGate::RxPi2,
                    which: WhichQubit::First,
                },
                Conjugation {
                    gate: ConjugationGate::RxPi2,
                    which: WhichQubit::Second,
                },
            ];
            let post = vec![
                Conjugation {
                    gate: ConjugationGate::RxNegPi2,
                    which: WhichQubit::First,
                },
                Conjugation {
                    gate: ConjugationGate::RxNegPi2,
                    which: WhichQubit::Second,
                },
            ];
            Ok(BasisChange { pre, post })
        }
        StandardGate::RZX => {
            let conjugation = vec![Conjugation {
                gate: ConjugationGate::H,
                which: WhichQubit::Second,
            }];
            Ok(BasisChange {
                pre: conjugation.clone(),
                post: conjugation,
            })
        }
        _ => Err(CompilerError::TransformFailed {
            name: DECOMPOSE_PAULI_ROTATION_NAME,
            reason: format!(
                "multi-controlled Pauli rotation decomposition supports only RXX, RYY, RZZ, or RZX, got {rotation}"
            ),
        }),
    }
}

fn emit_conjugations(
    operations: &mut Vec<ValueOperation>,
    conjugations: &[Conjugation],
    first: Qubit,
    second: Qubit,
) {
    for conjugation in conjugations {
        let qubit = match conjugation.which {
            WhichQubit::First => first,
            WhichQubit::Second => second,
        };
        match conjugation.gate {
            ConjugationGate::H => push_standard_gate(operations, StandardGate::H, [qubit]),
            ConjugationGate::RxPi2 => push_rx(operations, qubit, PI / 2.0),
            ConjugationGate::RxNegPi2 => push_rx(operations, qubit, -PI / 2.0),
        }
    }
}

fn push_rx(operations: &mut Vec<ValueOperation>, qubit: Qubit, angle: f64) {
    operations.push(ValueOperation {
        instruction: Instruction::Standard(StandardGate::RX),
        qubits: smallvec![qubit],
        params: smallvec![ParameterValue::Fixed(angle)],
        label: None,
    });
}

fn push_pauli_rotation_gate(
    operations: &mut Vec<ValueOperation>,
    rotation: StandardGate,
    theta: &ParameterValue,
    first: Qubit,
    second: Qubit,
) {
    operations.push(ValueOperation {
        instruction: Instruction::Standard(rotation),
        qubits: smallvec![first, second],
        params: smallvec![theta.clone()],
        label: None,
    });
}
