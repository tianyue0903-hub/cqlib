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

//! Shared exact `C^k Phase(theta)` primitive for MCGate lowering.

use super::super::mcx::{
    MAX_NO_ANCILLA_PHASE_POLY_QUBITS, clean_ancilla_mcx_operation_count,
    decompose_clean_ancilla_mcx,
};
use super::decompose::{AncillaMode, McGateDecomposeConfig, McGateOperandView, mc_gate_view_error};
use crate::circuit::{CircuitParam, Instruction, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

pub(super) const CONTROLLED_PHASE_OPERATION_COUNT: usize = 5;

/// Estimates the exact operation count for the shared `C^k Phase(theta)` primitive.
pub(super) fn controlled_phase_component_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    control_count: usize,
    family_name: &str,
) -> Result<usize, CompilerError> {
    match config.ancilla_mode {
        AncillaMode::NoAncilla => {
            estimate_no_ancilla_phase_operation_count(view, config, control_count + 1)
        }
        AncillaMode::CleanAncilla => {
            if control_count == 0 {
                return Ok(1);
            }
            if control_count == 1 {
                return Ok(CONTROLLED_PHASE_OPERATION_COUNT);
            }

            let required_clean_ancillas = control_count - 1;
            if config.clean_ancillas.len() < required_clean_ancillas {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "clean-ancilla {family_name} decomposition with {control_count} controls requires {required_clean_ancillas} clean ancillas, got {}",
                        config.clean_ancillas.len()
                    ),
                ));
            }

            let compute_operation_count = clean_ancilla_mcx_operation_count(control_count);
            compute_operation_count
                .checked_mul(2)
                .and_then(|count| count.checked_add(CONTROLLED_PHASE_OPERATION_COUNT))
                .ok_or_else(|| {
                    mc_gate_view_error(
                        view,
                        config,
                        format!("{family_name} controlled-phase operation count overflow"),
                    )
                })
        }
        AncillaMode::DirtyAncilla => Err(mc_gate_view_error(
            view,
            config,
            format!("dirty-ancilla {family_name} decomposition is not supported"),
        )),
    }
}

/// Emits the shared `C^k Phase(theta)` primitive.
pub(super) fn emit_controlled_phase_component(
    operations: &mut Vec<Operation>,
    view: &McGateOperandView<'_>,
    controls: &[Qubit],
    target: Qubit,
    theta: f64,
    config: &McGateDecomposeConfig,
    family_name: &str,
) -> Result<(), CompilerError> {
    match config.ancilla_mode {
        AncillaMode::NoAncilla => {
            let mut qubits = Vec::with_capacity(controls.len() + 1);
            qubits.extend_from_slice(controls);
            qubits.push(target);
            emit_no_ancilla_phase(operations, view, config, &qubits, theta)
        }
        AncillaMode::CleanAncilla if controls.is_empty() => {
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::Phase),
                qubits: smallvec![target],
                params: smallvec![CircuitParam::Fixed(theta)],
                label: None,
            });
            Ok(())
        }
        AncillaMode::CleanAncilla if controls.len() == 1 => {
            emit_controlled_phase_pair(operations, controls[0], target, theta);
            Ok(())
        }
        AncillaMode::CleanAncilla => {
            controlled_phase_component_operation_count(view, config, controls.len(), family_name)?;

            let required_clean_ancillas = controls.len() - 1;
            let flag = config.clean_ancillas[0];
            let work_ancillas = &config.clean_ancillas[1..required_clean_ancillas];
            // The flag is promised clean. Compute controls conjunction into it,
            // apply one controlled phase, then uncompute the MCX body so every
            // clean ancilla returns to |0>.
            let compute = decompose_clean_ancilla_mcx(controls, flag, work_ancillas).map_err(
                |source| {
                    mc_gate_view_error(
                        view,
                        config,
                        format!(
                            "MCX primitive failed while computing {family_name} clean flag: {source}"
                        ),
                    )
                },
            )?;

            operations.extend(compute.iter().cloned());
            emit_controlled_phase_pair(operations, flag, target, theta);
            operations.extend(compute.into_iter().rev());
            Ok(())
        }
        AncillaMode::DirtyAncilla => Err(mc_gate_view_error(
            view,
            config,
            format!("dirty-ancilla {family_name} decomposition is not supported"),
        )),
    }
}

fn estimate_no_ancilla_phase_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    qubit_count: usize,
) -> Result<usize, CompilerError> {
    let subset_count = checked_phase_subset_count(view, config, qubit_count)?;

    (qubit_count - 1)
        .checked_mul(subset_count)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| {
            mc_gate_view_error(
                view,
                config,
                "no-ancilla controlled Phase operation count overflow".to_string(),
            )
        })
}

fn emit_no_ancilla_phase(
    operations: &mut Vec<Operation>,
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    qubits: &[Qubit],
    theta: f64,
) -> Result<(), CompilerError> {
    let subset_count = checked_phase_subset_count(view, config, qubits.len())?;
    let base_angle = theta / 2.0_f64.powi((qubits.len() - 1) as i32);

    // Inclusion-exclusion over XOR parities:
    // AND(q0..qn) = 1 / 2^(n - 1) * sum((-1)^(|S|-1) * XOR(S)).
    // Applying Phase(angle) to each parity realizes the exact conditional phase
    // on controls AND target without introducing ancillas.
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
        emit_parity_phase(operations, &subset, angle);
    }

    Ok(())
}

fn checked_phase_subset_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    qubit_count: usize,
) -> Result<usize, CompilerError> {
    if qubit_count > MAX_NO_ANCILLA_PHASE_POLY_QUBITS {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "no-ancilla controlled Phase decomposition over {qubit_count} qubits would be exponential; max supported qubits is {MAX_NO_ANCILLA_PHASE_POLY_QUBITS}"
            ),
        ));
    }

    1usize.checked_shl(qubit_count as u32).ok_or_else(|| {
        mc_gate_view_error(
            view,
            config,
            format!(
                "no-ancilla controlled Phase decomposition over {qubit_count} qubits is too large"
            ),
        )
    })
}

fn emit_controlled_phase_pair(
    operations: &mut Vec<Operation>,
    control: Qubit,
    target: Qubit,
    theta: f64,
) {
    let half_theta = theta / 2.0;
    // CP(theta) = P(theta/2) on both qubits, then a CX-parity correction
    // P(-theta/2) on the target, followed by uncomputing the CX.
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::Phase),
        qubits: smallvec![control],
        params: smallvec![CircuitParam::Fixed(half_theta)],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::Phase),
        qubits: smallvec![target],
        params: smallvec![CircuitParam::Fixed(half_theta)],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![control, target],
        params: smallvec![],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::Phase),
        qubits: smallvec![target],
        params: smallvec![CircuitParam::Fixed(-half_theta)],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![control, target],
        params: smallvec![],
        label: None,
    });
}

fn emit_parity_phase(operations: &mut Vec<Operation>, qubits: &[Qubit], angle: f64) {
    if qubits.len() == 1 {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::Phase),
            qubits: smallvec![qubits[0]],
            params: smallvec![CircuitParam::Fixed(angle)],
            label: None,
        });
        return;
    }

    let accumulator = qubits[qubits.len() - 1];
    // Use the last qubit as a temporary parity accumulator and immediately
    // uncompute it, so the block is a diagonal parity phase as seen by callers.
    for qubit in &qubits[..qubits.len() - 1] {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![*qubit, accumulator],
            params: smallvec![],
            label: None,
        });
    }

    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::Phase),
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
