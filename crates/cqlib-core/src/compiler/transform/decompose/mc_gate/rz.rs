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

//! `RZ` and `CRZ` lowering for multi-controlled standard gates.
//!
//! `RZ(theta)` differs from `Phase(theta)` by a global phase:
//! `RZ(theta) = exp(-i theta / 2) * Phase(theta)`. Once controls are added,
//! that global phase becomes conditional on all controls being `|1>`, so
//! multi-controlled `RZ` is emitted as a controlled `Phase(theta)` on the
//! target plus a controlled `Phase(-theta / 2)` on one control qubit.

use super::decompose::{McGateDecomposeConfig, McGateOperandView, mc_gate_view_error};
use super::phase_ops::{
    controlled_phase_component_operation_count, emit_controlled_phase_component,
};
use crate::circuit::{CircuitParam, Instruction, Operation, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

/// Decomposes `RZ` and `CRZ` MCGates, preserving the conditional phase term.
pub(super) fn decompose_rz_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "RZ-family gate {} must have exactly one target, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    match view.base_gate() {
        StandardGate::RZ => {}
        StandardGate::CRZ => {}
        base_gate => {
            return Err(mc_gate_view_error(
                view,
                config,
                format!("rotation-family gate {base_gate} is not implemented by RZ lowering"),
            ));
        }
    }

    match view.total_control_count() {
        0 => {
            check_expansion_budget(view, config, 1)?;
            Ok(vec![Operation {
                instruction: Instruction::Standard(StandardGate::RZ),
                qubits: smallvec![*target],
                params: smallvec![params[0].clone()],
                label: None,
            }])
        }
        1 => {
            check_expansion_budget(view, config, 1)?;
            Ok(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CRZ),
                qubits: smallvec![view.all_controls()[0], *target],
                params: smallvec![params[0].clone()],
                label: None,
            }])
        }
        control_count => {
            let theta = fixed_rz_angle(view, params, config)?;
            check_multi_control_rz_budget(view, config, control_count)?;

            let mut operations = Vec::new();
            emit_controlled_phase_component(
                &mut operations,
                view,
                view.all_controls(),
                *target,
                theta,
                config,
                "RZ",
            )?;

            let compensation_target = view.all_controls()[control_count - 1];
            let compensation_controls = &view.all_controls()[..control_count - 1];
            // The global phase of `RZ` is conditional after control lifting.
            // Applying Phase(-theta/2) to one participating control, controlled
            // by the remaining controls, realizes exactly that missing term.
            emit_controlled_phase_component(
                &mut operations,
                view,
                compensation_controls,
                compensation_target,
                -theta / 2.0,
                config,
                "RZ",
            )?;

            Ok(operations)
        }
    }
}

fn fixed_rz_angle(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<f64, CompilerError> {
    match params[0] {
        CircuitParam::Fixed(theta) => Ok(theta),
        CircuitParam::Index(_) => Err(mc_gate_view_error(
            view,
            config,
            "symbolic RZ-family parameters require theta/2 arithmetic for conditional phase compensation, which is not supported yet".to_string(),
        )),
    }
}

fn check_multi_control_rz_budget(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    control_count: usize,
) -> Result<(), CompilerError> {
    let target_phase_count =
        controlled_phase_component_operation_count(view, config, control_count, "RZ")?;
    let compensation_count =
        controlled_phase_component_operation_count(view, config, control_count - 1, "RZ")?;
    let operation_count = target_phase_count
        .checked_add(compensation_count)
        .ok_or_else(|| {
            mc_gate_view_error(
                view,
                config,
                "RZ-family expansion operation count overflow".to_string(),
            )
        })?;

    check_expansion_budget(view, config, operation_count)
}

fn check_expansion_budget(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    operation_count: usize,
) -> Result<(), CompilerError> {
    if operation_count > config.max_expansion_ops {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "RZ-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}
