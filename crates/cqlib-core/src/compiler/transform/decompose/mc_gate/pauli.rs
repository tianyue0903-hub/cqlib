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

//! Pauli-family lowering for multi-controlled standard gates.
//!
//! This module handles only `X`, `Y`, `Z`, `CX`, `CY`, `CZ`, and `CCX` base
//! gates. The operand partition is supplied by [`McGateOperandView`], so this
//! layer preserves the IR order
//! `[added_controls..., base_inherent_controls..., base_targets...]` and never
//! rewrites or reorders operands.
//!
//! `X`, `CX`, and `CCX` are already MCX instances after the view merges added
//! and inherent controls. `Y` and `CY` are lowered by `SDG; MCX; S` on the
//! target, while `Z` and `CZ` are lowered by `H; MCX; H`. The MCX body is
//! delegated to `mcx.rs`, and this module only selects the configured ancilla
//! strategy, checks the local expansion budget, and wraps resource diagnostics
//! with MCGate context.

use super::super::mcx::{
    clean_ancilla_mcx_operation_count, clean_ancilla_mcx_required_ancillas,
    decompose_clean_ancilla_mcx, decompose_dirty_ancilla_mcx, decompose_no_ancilla_mcx,
    dirty_ancilla_mcx_operation_count, no_ancilla_mcx_operation_count,
};
use super::decompose::{AncillaMode, McGateDecomposeConfig, McGateOperandView, mc_gate_view_error};
use crate::circuit::{Instruction, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

/// Decomposes Pauli-family MCGates using the shared MCX primitive.
pub(super) fn decompose_pauli_family(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "Pauli-family gate {} must have exactly one target, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    check_expansion_budget(view, config)?;

    let mcx_operations = decompose_mcx_for_mode(view, *target, config)?;
    match view.base_gate() {
        StandardGate::X | StandardGate::CX | StandardGate::CCX => Ok(mcx_operations),
        StandardGate::Y | StandardGate::CY => {
            let mut operations = Vec::with_capacity(mcx_operations.len() + 2);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::SDG),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
            operations.extend(mcx_operations);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::S),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
            Ok(operations)
        }
        StandardGate::Z | StandardGate::CZ => {
            let mut operations = Vec::with_capacity(mcx_operations.len() + 2);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
            operations.extend(mcx_operations);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
            Ok(operations)
        }
        base_gate => Err(mc_gate_view_error(
            view,
            config,
            format!("gate {base_gate} is not in the Pauli family"),
        )),
    }
}

fn decompose_mcx_for_mode(
    view: &McGateOperandView<'_>,
    target: Qubit,
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let controls = view.all_controls();
    let result = match config.ancilla_mode {
        AncillaMode::NoAncilla => decompose_no_ancilla_mcx(controls, target),
        AncillaMode::CleanAncilla => {
            decompose_clean_ancilla_mcx(controls, target, &config.clean_ancillas)
        }
        AncillaMode::DirtyAncilla if controls.len() <= 2 => {
            decompose_no_ancilla_mcx(controls, target)
        }
        AncillaMode::DirtyAncilla => {
            let dirty_ancilla = config.dirty_ancillas.first().copied().ok_or_else(|| {
                mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla Pauli decomposition with {} controls requires one dirty ancilla",
                        controls.len()
                    ),
                )
            })?;
            decompose_dirty_ancilla_mcx(controls, target, dirty_ancilla)
        }
    };

    result.map_err(|source| {
        mc_gate_view_error(
            view,
            config,
            format!("MCX primitive failed while lowering Pauli-family gate: {source}"),
        )
    })
}

fn check_expansion_budget(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    let basis_operation_count = match view.base_gate() {
        StandardGate::Y | StandardGate::CY | StandardGate::Z | StandardGate::CZ => 2,
        _ => 0,
    };
    let mcx_operation_count = estimate_mcx_operation_count(view, config)?;
    let operation_count = mcx_operation_count
        .checked_add(basis_operation_count)
        .ok_or_else(|| {
            mc_gate_view_error(
                view,
                config,
                "Pauli-family expansion operation count overflow".to_string(),
            )
        })?;

    if operation_count > config.max_expansion_ops {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "Pauli-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}

fn estimate_mcx_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<usize, CompilerError> {
    let control_count = view.total_control_count();
    match config.ancilla_mode {
        AncillaMode::NoAncilla => no_ancilla_mcx_operation_count(control_count).map_err(|source| {
            mc_gate_view_error(
                view,
                config,
                format!(
                    "MCX primitive failed while estimating Pauli-family no-ancilla cost: {source}"
                ),
            )
        }),
        AncillaMode::CleanAncilla => {
            let required_ancillas = clean_ancilla_mcx_required_ancillas(control_count);
            if config.clean_ancillas.len() < required_ancillas {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "clean-ancilla Pauli decomposition with {control_count} controls requires {required_ancillas} clean ancillas, got {}",
                        config.clean_ancillas.len()
                    ),
                ));
            }
            Ok(clean_ancilla_mcx_operation_count(control_count))
        }
        AncillaMode::DirtyAncilla => {
            if control_count <= 2 {
                return no_ancilla_mcx_operation_count(control_count).map_err(|source| {
                    mc_gate_view_error(
                        view,
                        config,
                        format!(
                            "MCX primitive failed while estimating Pauli-family small dirty-mode cost: {source}"
                        ),
                    )
                });
            }
            if config.dirty_ancillas.is_empty() {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla Pauli decomposition with {control_count} controls requires one dirty ancilla"
                    ),
                ));
            }
            let recursion_depth = control_count - 2;
            if recursion_depth > config.max_recursion_depth {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla Pauli decomposition recursion depth {recursion_depth} exceeds max_recursion_depth={}",
                        config.max_recursion_depth
                    ),
                ));
            }
            dirty_ancilla_mcx_operation_count(control_count).map_err(|source| {
                mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "MCX primitive failed while estimating Pauli-family dirty-ancilla cost: {source}"
                    ),
                )
            })
        }
    }
}
