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

//! One-qubit-family lowering for multi-controlled standard gates.
//!
//! Named square-root gates are lowered through the existing controlled
//! rotation paths. General numeric one-qubit gates are synthesized as
//! `exp(i*g) * U(theta, phi, lambda)`, then emitted as a conditional phase
//! plus controlled `RZ(lambda); RY(theta); RZ(phi)` in circuit order.

use super::decompose::{
    AncillaMode, McGateDecomposeConfig, McGateOperandView, decompose_mc_gate, mc_gate_view_error,
};
use super::phase_ops::{
    controlled_phase_component_operation_count, emit_controlled_phase_component,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use crate::compiler::transform::decompose::one_qubit_unitary::OneQubitUDecomposition;
use crate::compiler::transform::decompose::one_qubit_unitary::synthesize_one_qubit_unitary_as_u;
use smallvec::smallvec;
use std::f64::consts::FRAC_PI_2;

const ANGLE_EPS: f64 = 1e-12;

/// Decomposes non-Pauli, non-rotation one-qubit MCGates.
pub(super) fn decompose_one_qubit_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "OneQubit-family gate {} must have exactly one target, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    ensure_one_qubit_family_gate(view, config)?;

    if view.total_control_count() == 0 {
        check_expansion_budget(view, config, 1)?;
        return Ok(vec![Operation {
            instruction: Instruction::Standard(view.base_gate()),
            qubits: smallvec![*target],
            params: params.iter().cloned().collect(),
            label: None,
        }]);
    }

    if config.ancilla_mode == AncillaMode::DirtyAncilla {
        return Err(mc_gate_view_error(
            view,
            config,
            "dirty-ancilla OneQubit decomposition is not supported".to_string(),
        ));
    }

    match view.base_gate() {
        StandardGate::X2P => {
            decompose_named_half_rotation(view, *target, StandardGate::RX, FRAC_PI_2, config, "X2P")
        }
        StandardGate::X2M => decompose_named_half_rotation(
            view,
            *target,
            StandardGate::RX,
            -FRAC_PI_2,
            config,
            "X2M",
        ),
        StandardGate::Y2P => {
            decompose_named_half_rotation(view, *target, StandardGate::RY, FRAC_PI_2, config, "Y2P")
        }
        StandardGate::Y2M => decompose_named_half_rotation(
            view,
            *target,
            StandardGate::RY,
            -FRAC_PI_2,
            config,
            "Y2M",
        ),
        _ => decompose_synthesized_one_qubit(view, *target, params, config),
    }
}

fn ensure_one_qubit_family_gate(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    match view.base_gate() {
        StandardGate::H
        | StandardGate::U
        | StandardGate::X2P
        | StandardGate::X2M
        | StandardGate::Y2P
        | StandardGate::Y2M
        | StandardGate::RXY
        | StandardGate::XY
        | StandardGate::XY2P
        | StandardGate::XY2M => Ok(()),
        base_gate => Err(mc_gate_view_error(
            view,
            config,
            format!("gate {base_gate} is not in the OneQubit family"),
        )),
    }
}

fn decompose_named_half_rotation(
    view: &McGateOperandView<'_>,
    target: Qubit,
    rotation_gate: StandardGate,
    theta: f64,
    config: &McGateDecomposeConfig,
    gate_name: &str,
) -> Result<Vec<Operation>, CompilerError> {
    let operation_count = controlled_xy_rotation_operation_count(view, config)?;
    check_expansion_budget(view, config, operation_count)?;
    decompose_lifted_rotation(view, target, rotation_gate, theta, config, gate_name)
}

fn decompose_synthesized_one_qubit(
    view: &McGateOperandView<'_>,
    target: Qubit,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let fixed_params = fixed_parameters(view, params, config)?;
    let base_gate = view.base_gate();
    let matrix = base_gate.matrix(&fixed_params).map_err(|source| {
        mc_gate_view_error(
            view,
            config,
            format!(
                "failed to build numeric matrix for OneQubit-family gate {}: {source}",
                base_gate
            ),
        )
    })?;
    let decomposition =
        synthesize_one_qubit_unitary_as_u(matrix.as_ref()).map_err(|source| match source {
            CompilerError::TransformFailed { reason, .. } => mc_gate_view_error(
                view,
                config,
                format!("numeric one-qubit synthesis failed: {reason}"),
            ),
            other => other,
        })?;
    let conditional_phase =
        decomposition.global_phase + (decomposition.phi + decomposition.lambda) / 2.0;
    let operation_count =
        synthesized_one_qubit_operation_count(view, config, decomposition, conditional_phase)?;
    check_expansion_budget(view, config, operation_count)?;

    let mut operations = Vec::with_capacity(operation_count);
    emit_conditional_phase_on_controls(&mut operations, view, conditional_phase, config)?;
    extend_lifted_rotation_if_needed(
        &mut operations,
        view,
        target,
        StandardGate::RZ,
        decomposition.lambda,
        config,
        "U.RZ(lambda)",
    )?;
    extend_lifted_rotation_if_needed(
        &mut operations,
        view,
        target,
        StandardGate::RY,
        decomposition.theta,
        config,
        "U.RY(theta)",
    )?;
    extend_lifted_rotation_if_needed(
        &mut operations,
        view,
        target,
        StandardGate::RZ,
        decomposition.phi,
        config,
        "U.RZ(phi)",
    )?;

    Ok(operations)
}

fn fixed_parameters(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<f64>, CompilerError> {
    params
        .iter()
        .map(|param| match param {
            CircuitParam::Fixed(value) => Ok(*value),
            CircuitParam::Index(_) => Err(mc_gate_view_error(
                view,
                config,
                format!(
                    "symbolic OneQubit-family parameters for gate {} require numeric synthesis, which is not supported yet",
                    view.base_gate()
                ),
            )),
        })
        .collect()
}

fn synthesized_one_qubit_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    decomposition: OneQubitUDecomposition,
    conditional_phase: f64,
) -> Result<usize, CompilerError> {
    let mut operation_count = conditional_phase_operation_count(view, config, conditional_phase)?;
    if !is_effectively_zero(decomposition.lambda) {
        operation_count = checked_add_operation_count(
            view,
            config,
            operation_count,
            controlled_rz_operation_count(view, config)?,
        )?;
    }
    if !is_effectively_zero(decomposition.theta) {
        operation_count = checked_add_operation_count(
            view,
            config,
            operation_count,
            controlled_xy_rotation_operation_count(view, config)?,
        )?;
    }
    if !is_effectively_zero(decomposition.phi) {
        operation_count = checked_add_operation_count(
            view,
            config,
            operation_count,
            controlled_rz_operation_count(view, config)?,
        )?;
    }

    Ok(operation_count)
}

fn conditional_phase_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    theta: f64,
) -> Result<usize, CompilerError> {
    if is_effectively_zero(theta) {
        return Ok(0);
    }

    let control_count = view.total_control_count();
    debug_assert!(control_count > 0);
    if control_count == 1 {
        return Ok(1);
    }

    controlled_phase_component_operation_count(view, config, control_count - 1, "OneQubit")
}

fn controlled_rz_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<usize, CompilerError> {
    let control_count = view.total_control_count();
    if control_count <= 1 {
        return Ok(1);
    }

    let target_phase_count =
        controlled_phase_component_operation_count(view, config, control_count, "OneQubit")?;
    let compensation_count =
        controlled_phase_component_operation_count(view, config, control_count - 1, "OneQubit")?;
    checked_add_operation_count(view, config, target_phase_count, compensation_count)
}

fn controlled_xy_rotation_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<usize, CompilerError> {
    if view.total_control_count() <= 1 {
        return Ok(1);
    }

    checked_add_operation_count(
        view,
        config,
        controlled_rz_operation_count(view, config)?,
        2,
    )
}

fn checked_add_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    lhs: usize,
    rhs: usize,
) -> Result<usize, CompilerError> {
    lhs.checked_add(rhs).ok_or_else(|| {
        mc_gate_view_error(
            view,
            config,
            "OneQubit-family expansion operation count overflow".to_string(),
        )
    })
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
                "OneQubit-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}

fn emit_conditional_phase_on_controls(
    operations: &mut Vec<Operation>,
    view: &McGateOperandView<'_>,
    theta: f64,
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    if is_effectively_zero(theta) {
        return Ok(());
    }

    let controls = view.all_controls();
    debug_assert!(!controls.is_empty());
    if controls.len() == 1 {
        operations.push(Operation {
            instruction: Instruction::Standard(StandardGate::Phase),
            qubits: smallvec![controls[0]],
            params: smallvec![CircuitParam::Fixed(theta)],
            label: None,
        });
        return Ok(());
    }

    let phase_target = controls[controls.len() - 1];
    let phase_controls = &controls[..controls.len() - 1];
    // The determinant/global phase of a controlled one-qubit unitary is a
    // conditional phase on the control conjunction, not a circuit global phase.
    emit_controlled_phase_component(
        operations,
        view,
        phase_controls,
        phase_target,
        theta,
        config,
        "OneQubit",
    )
}

fn extend_lifted_rotation_if_needed(
    operations: &mut Vec<Operation>,
    view: &McGateOperandView<'_>,
    target: Qubit,
    rotation_gate: StandardGate,
    theta: f64,
    config: &McGateDecomposeConfig,
    context: &str,
) -> Result<(), CompilerError> {
    if is_effectively_zero(theta) {
        return Ok(());
    }

    operations.extend(decompose_lifted_rotation(
        view,
        target,
        rotation_gate,
        theta,
        config,
        context,
    )?);
    Ok(())
}

fn decompose_lifted_rotation(
    view: &McGateOperandView<'_>,
    target: Qubit,
    rotation_gate: StandardGate,
    theta: f64,
    config: &McGateDecomposeConfig,
    context: &str,
) -> Result<Vec<Operation>, CompilerError> {
    let added_control_count = u8::try_from(view.total_control_count()).map_err(|_| {
        mc_gate_view_error(
            view,
            config,
            format!(
                "OneQubit-family control count {} exceeds supported MCGate arity",
                view.total_control_count()
            ),
        )
    })?;
    let gate = MCGate::new(added_control_count, rotation_gate);
    let mut qubits = Vec::with_capacity(view.total_control_count() + 1);
    qubits.extend_from_slice(view.all_controls());
    qubits.push(target);
    let params = [CircuitParam::Fixed(theta)];

    decompose_mc_gate(&gate, &qubits, &params, config).map_err(|source| match source {
        CompilerError::TransformFailed { name, reason } => CompilerError::TransformFailed {
            name,
            reason: format!("OneQubit-family {context} lowering failed: {reason}"),
        },
        other => other,
    })
}

fn is_effectively_zero(theta: f64) -> bool {
    theta.abs() <= ANGLE_EPS
}
