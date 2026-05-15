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

//! Multi-controlled X decomposition.
//!
//! The clean-ancilla implementation expands MCX into a V-chain of Toffoli
//! gates. The chain computes the progressive conjunction of controls into clean
//! ancillas, uses the final conjunction to flip the target, then runs the
//! compute chain in reverse to restore every ancilla to `|0>`.
//!
//! The no-ancilla implementation lowers MCX through an MCZ phase polynomial:
//! it applies a parity-controlled `RZ` phase for every non-empty subset of
//! `controls + target`, then conjugates the target with `H`.
//!
//! The dirty-ancilla implementation uses one borrowed work qubit with unknown
//! initial state. It recursively flips the borrowed qubit by the prefix
//! conjunction, uses it with the last control, uncomputes the borrowed qubit,
//! then repeats the last Toffoli to cancel the borrowed qubit's unknown
//! contribution on the target.
//!
//! # Examples
//!
//! Clean-ancilla decomposition uses `k - 2` work qubits for an MCX with `k`
//! controls and returns a Toffoli V-chain:
//!
//! ```rust,ignore
//! use crate::circuit::Qubit;
//! use crate::compiler::transform::decompose::mcx::decompose_clean_ancilla_mcx;
//!
//! let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
//! let target = Qubit::new(3);
//! let clean_ancillas = [Qubit::new(4)];
//! let operations = decompose_clean_ancilla_mcx(&controls, target, &clean_ancillas)?;
//! ```
//!
//! No-ancilla decomposition consumes only the controls and target. For three or
//! more controls it emits `H`, `CX`, and fixed-angle `RZ` operations:
//!
//! ```rust,ignore
//! use crate::circuit::Qubit;
//! use crate::compiler::transform::decompose::mcx::decompose_no_ancilla_mcx;
//!
//! let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
//! let target = Qubit::new(4);
//! let operations = decompose_no_ancilla_mcx(&controls, target)?;
//! ```
//!
//! Dirty-ancilla decomposition uses one borrowed qubit. The borrowed qubit may
//! be in either computational basis state, and is restored by the returned
//! sequence:
//!
//! ```rust,ignore
//! use crate::circuit::Qubit;
//! use crate::compiler::transform::decompose::mcx::decompose_dirty_ancilla_mcx;
//!
//! let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
//! let target = Qubit::new(3);
//! let borrow = Qubit::new(4);
//! let operations = decompose_dirty_ancilla_mcx(&controls, target, borrow)?;
//! ```

use crate::circuit::{CircuitParam, Instruction, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;
use std::collections::HashSet;
use std::f64::consts::PI;

const DECOMPOSE_MCX_NAME: &str = "decompose.mcx";
pub(crate) const MAX_NO_ANCILLA_PHASE_POLY_QUBITS: usize = 10;
pub(crate) const MAX_DIRTY_RECURSIVE_CONTROLS: usize = 10;

/// Returns the clean work qubits required by the clean-ancilla MCX primitive.
pub(crate) fn clean_ancilla_mcx_required_ancillas(control_count: usize) -> usize {
    control_count.saturating_sub(2)
}

/// Returns the exact operation count emitted by the clean-ancilla MCX primitive.
pub(crate) fn clean_ancilla_mcx_operation_count(control_count: usize) -> usize {
    match control_count {
        0..=2 => 1,
        _ => 2 * control_count - 3,
    }
}

/// Returns the exact operation count emitted by the no-ancilla MCX primitive.
pub(crate) fn no_ancilla_mcx_operation_count(control_count: usize) -> Result<usize, CompilerError> {
    if control_count <= 2 {
        return Ok(1);
    }

    let qubit_count = control_count + 1;
    if qubit_count > MAX_NO_ANCILLA_PHASE_POLY_QUBITS {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_MCX_NAME,
            reason: format!(
                "no-ancilla phase-polynomial MCX over {qubit_count} qubits would be exponential; max supported qubits is {MAX_NO_ANCILLA_PHASE_POLY_QUBITS}"
            ),
        });
    }

    let subset_count =
        1usize
            .checked_shl(qubit_count as u32)
            .ok_or_else(|| CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: format!("no-ancilla MCX over {qubit_count} qubits is too large"),
            })?;

    (qubit_count - 1)
        .checked_mul(subset_count)
        .and_then(|value| value.checked_add(3))
        .ok_or_else(|| CompilerError::TransformFailed {
            name: DECOMPOSE_MCX_NAME,
            reason: "no-ancilla MCX operation count overflow".to_string(),
        })
}

/// Returns the exact operation count emitted by the dirty-ancilla MCX primitive.
pub(crate) fn dirty_ancilla_mcx_operation_count(
    control_count: usize,
) -> Result<usize, CompilerError> {
    if control_count > MAX_DIRTY_RECURSIVE_CONTROLS {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_MCX_NAME,
            reason: format!(
                "dirty-ancilla recursive MCX with {control_count} controls would be exponential; max supported controls is {MAX_DIRTY_RECURSIVE_CONTROLS}"
            ),
        });
    }

    match control_count {
        0..=2 => Ok(1),
        _ => {
            let recursive_leaf_count =
                1usize
                    .checked_shl((control_count - 2) as u32)
                    .ok_or_else(|| CompilerError::TransformFailed {
                        name: DECOMPOSE_MCX_NAME,
                        reason: "dirty-ancilla recursive MCX operation count overflow".to_string(),
                    })?;
            recursive_leaf_count
                .checked_mul(3)
                .and_then(|value| value.checked_sub(2))
                .ok_or_else(|| CompilerError::TransformFailed {
                    name: DECOMPOSE_MCX_NAME,
                    reason: "dirty-ancilla recursive MCX operation count overflow".to_string(),
                })
        }
    }
}

/// Decomposes a multi-controlled X gate using clean-ancilla V-chain / AND-ladder.
///
/// The controls are ordered as supplied and the target is flipped iff all
/// controls are `|1>`. For `k >= 3`, the first `k - 2` entries of
/// `clean_ancillas` are used as clean work qubits. They are assumed to be
/// initialized to `|0>` and are uncomputed back to `|0>` by the returned
/// sequence.
///
/// The returned sequence intentionally stops at `X`, `CX`, and `CCX`; further
/// Toffoli lowering is owned by later rewrite/decomposition stages.
pub(crate) fn decompose_clean_ancilla_mcx(
    controls: &[Qubit],
    target: Qubit,
    clean_ancillas: &[Qubit],
) -> Result<Vec<Operation>, CompilerError> {
    let control_count = controls.len();
    let required_ancillas = clean_ancilla_mcx_required_ancillas(control_count);

    if clean_ancillas.len() < required_ancillas {
        return Err(CompilerError::TransformFailed {
            name: DECOMPOSE_MCX_NAME,
            reason: format!(
                "MCX with {control_count} controls requires {required_ancillas} clean ancillas, got {}",
                clean_ancillas.len()
            ),
        });
    }
    let used_ancillas = &clean_ancillas[..required_ancillas];
    validate_distinct_qubits(controls, target, used_ancillas)?;

    let operations = match control_count {
        0 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![target],
            params: smallvec![],
            label: None,
        }],
        1 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![controls[0], target],
            params: smallvec![],
            label: None,
        }],
        2 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::CCX),
            qubits: smallvec![controls[0], controls[1], target],
            params: smallvec![],
            label: None,
        }],
        _ => decompose_v_chain(controls, target, used_ancillas),
    };

    Ok(operations)
}

/// Decomposes a multi-controlled X gate without consuming any auxiliary qubits.
///
/// For three or more controls this uses an exact parity phase-polynomial MCZ
/// decomposition conjugated by `H(target)`. The construction is ancilla-free but
/// exponential in the number of participating qubits, so it is intended as a
/// fallback for small MCX gates or situations where no clean work qubits exist.
pub(crate) fn decompose_no_ancilla_mcx(
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<Operation>, CompilerError> {
    let control_count = controls.len();
    validate_distinct_qubits(controls, target, &[])?;

    let operations = match control_count {
        0 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![target],
            params: smallvec![],
            label: None,
        }],
        1 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![controls[0], target],
            params: smallvec![],
            label: None,
        }],
        2 => vec![Operation {
            instruction: Instruction::Standard(StandardGate::CCX),
            qubits: smallvec![controls[0], controls[1], target],
            params: smallvec![],
            label: None,
        }],
        _ => decompose_phase_polynomial_mcx(controls, target)?,
    };

    Ok(operations)
}

/// Decomposes a multi-controlled X gate using one dirty borrowed ancilla.
///
/// The dirty ancilla may start in an unknown computational basis state. The
/// returned exact sequence preserves every control, applies
/// `target ^= AND(controls)`, and restores `dirty_ancilla` to its input state.
///
/// This recursive construction uses only `X`, `CX`, and `CCX`, but its Toffoli
/// count is exponential in the number of controls. It is capped to avoid
/// accidentally materializing very large circuits.
pub(crate) fn decompose_dirty_ancilla_mcx(
    controls: &[Qubit],
    target: Qubit,
    dirty_ancilla: Qubit,
) -> Result<Vec<Operation>, CompilerError> {
    let control_count = controls.len();
    let used_ancillas = if control_count >= 3 {
        std::slice::from_ref(&dirty_ancilla)
    } else {
        &[]
    };
    validate_distinct_qubits(controls, target, used_ancillas)?;

    let operation_count = dirty_ancilla_mcx_operation_count(control_count)?;
    let mut operations = Vec::with_capacity(operation_count);
    emit_dirty_ancilla_mcx(&mut operations, controls, target, dirty_ancilla);
    Ok(operations)
}

/// Builds the Toffoli V-chain for the `control_count >= 3` case.
fn decompose_v_chain(controls: &[Qubit], target: Qubit, ancillas: &[Qubit]) -> Vec<Operation> {
    // Compute stage: ancilla[i] stores the AND of controls[0..=i + 1].
    let mut compute = Vec::with_capacity(ancillas.len());
    compute.push(Operation {
        instruction: Instruction::Standard(StandardGate::CCX),
        qubits: smallvec![controls[0], controls[1], ancillas[0]],
        params: smallvec![],
        label: None,
    });

    for ancilla_index in 1..ancillas.len() {
        compute.push(Operation {
            instruction: Instruction::Standard(StandardGate::CCX),
            qubits: smallvec![
                ancillas[ancilla_index - 1],
                controls[ancilla_index + 1],
                ancillas[ancilla_index]
            ],
            params: smallvec![],
            label: None,
        });
    }

    // Target stage followed by uncompute. Replaying the compute ladder in
    // reverse clears the clean ancillas without changing the target flip.
    let mut operations = Vec::with_capacity(2 * compute.len() + 1);
    operations.extend(compute.iter().cloned());
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::CCX),
        qubits: smallvec![
            ancillas[ancillas.len() - 1],
            controls[controls.len() - 1],
            target
        ],
        params: smallvec![],
        label: None,
    });
    operations.extend(compute.into_iter().rev());
    operations
}

/// Emits the borrowed-ancilla recursive MCX construction.
fn emit_dirty_ancilla_mcx(
    operations: &mut Vec<Operation>,
    controls: &[Qubit],
    target: Qubit,
    dirty_ancilla: Qubit,
) {
    match controls.len() {
        0 => operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![target],
            params: smallvec![],
            label: None,
        }),
        1 => operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![controls[0], target],
            params: smallvec![],
            label: None,
        }),
        2 => operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::CCX),
            qubits: smallvec![controls[0], controls[1], target],
            params: smallvec![],
            label: None,
        }),
        control_count => {
            let prefix = &controls[..control_count - 1];
            let last = controls[control_count - 1];

            emit_dirty_ancilla_mcx(operations, prefix, dirty_ancilla, target);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::CCX),
                qubits: smallvec![dirty_ancilla, last, target],
                params: smallvec![],
                label: None,
            });
            emit_dirty_ancilla_mcx(operations, prefix, dirty_ancilla, target);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::CCX),
                qubits: smallvec![dirty_ancilla, last, target],
                params: smallvec![],
                label: None,
            });
        }
    }
}

/// Builds the ancilla-free parity phase-polynomial decomposition.
fn decompose_phase_polynomial_mcx(
    controls: &[Qubit],
    target: Qubit,
) -> Result<Vec<Operation>, CompilerError> {
    let mut qubits = Vec::with_capacity(controls.len() + 1);
    qubits.extend_from_slice(controls);
    qubits.push(target);

    let estimated_ops = no_ancilla_mcx_operation_count(controls.len())?;
    let subset_count =
        1usize
            .checked_shl(qubits.len() as u32)
            .ok_or_else(|| CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: format!(
                    "no-ancilla MCX over {} qubits is too large to enumerate",
                    qubits.len()
                ),
            })?;
    let base_angle = PI / 2.0_f64.powi((qubits.len() - 1) as i32);

    let mut operations = Vec::with_capacity(estimated_ops);
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![target],
        params: smallvec![],
        label: None,
    });

    for subset_mask in 1..subset_count {
        let mut subset = Vec::new();
        for (index, qubit) in qubits.iter().copied().enumerate() {
            if subset_mask & (1usize << index) != 0 {
                subset.push(qubit);
            }
        }

        let angle = if subset.len() % 2 == 0 {
            -base_angle
        } else {
            base_angle
        };
        emit_parity_rz(&mut operations, &subset, angle);
    }

    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![target],
        params: smallvec![],
        label: None,
    });
    Ok(operations)
}

/// Emits an `RZ` phase on the XOR parity of `qubits` without using ancillas.
fn emit_parity_rz(operations: &mut Vec<Operation>, qubits: &[Qubit], angle: f64) {
    if qubits.len() == 1 {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![qubits[0]],
            params: smallvec![CircuitParam::Fixed(angle)],
            label: None,
        });
        return;
    }

    let accumulator = qubits[qubits.len() - 1];
    for qubit in &qubits[..qubits.len() - 1] {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![*qubit, accumulator],
            params: smallvec![],
            label: None,
        });
    }

    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![accumulator],
        params: smallvec![CircuitParam::Fixed(angle)],
        label: None,
    });

    for qubit in qubits[..qubits.len() - 1].iter().rev() {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![*qubit, accumulator],
            params: smallvec![],
            label: None,
        });
    }
}

/// Ensures controls, target, and every consumed ancilla are disjoint.
fn validate_distinct_qubits(
    controls: &[Qubit],
    target: Qubit,
    ancillas: &[Qubit],
) -> Result<(), CompilerError> {
    let mut seen = HashSet::with_capacity(controls.len() + ancillas.len() + 1);
    for qubit in controls
        .iter()
        .copied()
        .chain(std::iter::once(target))
        .chain(ancillas.iter().copied())
    {
        if !seen.insert(qubit) {
            return Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MCX_NAME,
                reason: format!(
                    "MCX controls, target, and ancillas must be distinct; duplicate {qubit}"
                ),
            });
        }
    }

    Ok(())
}
