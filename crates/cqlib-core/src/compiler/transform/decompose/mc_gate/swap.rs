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

//! SWAP-family lowering for multi-controlled standard gates.
//!
//! `C^k SWAP(a, b)` is emitted as three controlled-X style blocks:
//! `C^k CX(a, b); C^k CX(b, a); C^k CX(a, b)`. Each block is an MCX whose
//! controls are the original controls plus the active `CX` control.

use super::super::mcx::{
    clean_ancilla_mcx_operation_count, clean_ancilla_mcx_required_ancillas,
    decompose_clean_ancilla_mcx, decompose_dirty_ancilla_mcx, decompose_no_ancilla_mcx,
    dirty_ancilla_mcx_operation_count, no_ancilla_mcx_operation_count,
};
use super::decompose::{AncillaMode, McGateDecomposeConfig, McGateOperandView, mc_gate_view_error};
use crate::circuit::{Operation, Qubit};
use crate::compiler::error::CompilerError;

/// Decomposes controlled-SWAP gates through three controlled-X style blocks.
pub(super) fn decompose_swap_family(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [first_target, second_target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "SWAP-family gate {} must have exactly two targets, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    check_expansion_budget(view, config)?;

    let controls = view.all_controls();
    let segment_operation_count = estimate_controlled_x_operation_count(view, config)?;
    let mut operations = Vec::with_capacity(segment_operation_count * 3);
    // Each MCX block restores its own ancillas, so the configured work pool can
    // be reused across the three CX blocks.
    operations.extend(decompose_controlled_x(
        view,
        controls,
        *first_target,
        *second_target,
        config,
    )?);
    operations.extend(decompose_controlled_x(
        view,
        controls,
        *second_target,
        *first_target,
        config,
    )?);
    operations.extend(decompose_controlled_x(
        view,
        controls,
        *first_target,
        *second_target,
        config,
    )?);

    Ok(operations)
}

fn decompose_controlled_x(
    view: &McGateOperandView<'_>,
    outer_controls: &[Qubit],
    cx_control: Qubit,
    cx_target: Qubit,
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let mut controls = Vec::with_capacity(outer_controls.len() + 1);
    controls.extend_from_slice(outer_controls);
    // The active SWAP endpoint is the inherent control of the temporary CX.
    controls.push(cx_control);

    let result = match config.ancilla_mode {
        AncillaMode::NoAncilla => decompose_no_ancilla_mcx(&controls, cx_target),
        AncillaMode::CleanAncilla => {
            decompose_clean_ancilla_mcx(&controls, cx_target, &config.clean_ancillas)
        }
        AncillaMode::DirtyAncilla if controls.len() <= 2 => {
            // Small MCX stops at CX/CCX, so no borrowed qubit is needed.
            decompose_no_ancilla_mcx(&controls, cx_target)
        }
        AncillaMode::DirtyAncilla => {
            let dirty_ancilla = config.dirty_ancillas.first().copied().ok_or_else(|| {
                mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla SWAP decomposition with {} effective controls requires one dirty ancilla",
                        controls.len()
                    ),
                )
            })?;
            decompose_dirty_ancilla_mcx(&controls, cx_target, dirty_ancilla)
        }
    };

    result.map_err(|source| {
        mc_gate_view_error(
            view,
            config,
            format!("MCX primitive failed while lowering SWAP-family gate: {source}"),
        )
    })
}

fn check_expansion_budget(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    let segment_operation_count = estimate_controlled_x_operation_count(view, config)?;
    // Budget is checked before materializing any of the three MCX blocks.
    let operation_count = segment_operation_count.checked_mul(3).ok_or_else(|| {
        mc_gate_view_error(
            view,
            config,
            "SWAP-family expansion operation count overflow".to_string(),
        )
    })?;

    if operation_count > config.max_expansion_ops {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "SWAP-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}

fn estimate_controlled_x_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<usize, CompilerError> {
    // A controlled-CX block uses the original controls plus the CX control endpoint.
    let effective_control_count = view.total_control_count() + 1;
    match config.ancilla_mode {
        AncillaMode::NoAncilla => no_ancilla_mcx_operation_count(effective_control_count).map_err(
            |source| {
                mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "MCX primitive failed while estimating SWAP-family no-ancilla cost: {source}"
                    ),
                )
            },
        ),
        AncillaMode::CleanAncilla => {
            let required_ancillas = clean_ancilla_mcx_required_ancillas(effective_control_count);
            if config.clean_ancillas.len() < required_ancillas {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "clean-ancilla SWAP decomposition with {effective_control_count} effective controls requires {required_ancillas} clean ancillas, got {}",
                        config.clean_ancillas.len()
                    ),
                ));
            }
            Ok(clean_ancilla_mcx_operation_count(effective_control_count))
        }
        AncillaMode::DirtyAncilla => {
            if effective_control_count <= 2 {
                return no_ancilla_mcx_operation_count(effective_control_count).map_err(|source| {
                    mc_gate_view_error(
                        view,
                        config,
                        format!(
                            "MCX primitive failed while estimating SWAP-family small dirty-mode cost: {source}"
                        ),
                    )
                });
            }
            if config.dirty_ancillas.is_empty() {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla SWAP decomposition with {effective_control_count} effective controls requires one dirty ancilla"
                    ),
                ));
            }
            let recursion_depth = effective_control_count - 2;
            if recursion_depth > config.max_recursion_depth {
                return Err(mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "dirty-ancilla SWAP decomposition recursion depth {recursion_depth} exceeds max_recursion_depth={}",
                        config.max_recursion_depth
                    ),
                ));
            }
            dirty_ancilla_mcx_operation_count(effective_control_count).map_err(|source| {
                mc_gate_view_error(
                    view,
                    config,
                    format!(
                        "MCX primitive failed while estimating SWAP-family dirty-ancilla cost: {source}"
                    ),
                )
            })
        }
    }
}
