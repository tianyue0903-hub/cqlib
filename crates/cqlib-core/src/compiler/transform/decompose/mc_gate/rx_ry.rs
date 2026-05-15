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

//! `RX`, `RY`, `CRX`, and `CRY` lowering for multi-controlled standard gates.
//!
//! The lowering reuses the verified `RZ`/`CRZ` implementation. `RX` is
//! `H; RZ; H`, while `RY` is `RX(pi/2); RZ; RX(-pi/2)` in circuit order.

use super::decompose::{
    McGateDecomposeConfig, McGateOperandView, decompose_mc_gate, mc_gate_view_error,
};
use super::phase_ops::controlled_phase_component_operation_count;
use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;
use std::f64::consts::FRAC_PI_2;

/// Decomposes `RX`, `RY`, `CRX`, and `CRY` MCGates through the `RZ` family.
pub(super) fn decompose_rx_ry_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "RX/RY-family gate {} must have exactly one target, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    let direct_gate = match view.base_gate() {
        StandardGate::RX => match view.total_control_count() {
            0 => Some(StandardGate::RX),
            1 => Some(StandardGate::CRX),
            _ => None,
        },
        StandardGate::RY => match view.total_control_count() {
            0 => Some(StandardGate::RY),
            1 => Some(StandardGate::CRY),
            _ => None,
        },
        StandardGate::CRX => match view.total_control_count() {
            1 => Some(StandardGate::CRX),
            _ => None,
        },
        StandardGate::CRY => match view.total_control_count() {
            1 => Some(StandardGate::CRY),
            _ => None,
        },
        base_gate => {
            return Err(mc_gate_view_error(
                view,
                config,
                format!("gate {base_gate} is not in the RX/RY family"),
            ));
        }
    };

    if let Some(gate) = direct_gate {
        check_expansion_budget(view, config, 1)?;
        return Ok(vec![Operation {
            instruction: Instruction::Standard(gate),
            qubits: direct_qubits(view, *target, gate),
            params: smallvec![params[0].clone()],
            label: None,
        }]);
    }

    ensure_fixed_multi_control_parameter(view, params, config)?;
    let rz_operation_count = lifted_rz_operation_count(view, config)?;
    let operation_count = rz_operation_count.checked_add(2).ok_or_else(|| {
        mc_gate_view_error(
            view,
            config,
            "RX/RY-family expansion operation count overflow".to_string(),
        )
    })?;
    check_expansion_budget(view, config, operation_count)?;

    let rz_operations = decompose_lifted_rz(view, params, config)?;
    let mut operations = Vec::with_capacity(operation_count);
    match view.base_gate() {
        StandardGate::RX | StandardGate::CRX => {
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
            operations.extend(rz_operations);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![*target],
                params: smallvec![],
                label: None,
            });
        }
        StandardGate::RY | StandardGate::CRY => {
            // For inactive controls the two RX basis changes cancel. When the
            // lifted RZ fires, RX(-pi/2) * RZ(theta) * RX(pi/2) equals RY(theta).
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![*target],
                params: smallvec![CircuitParam::Fixed(FRAC_PI_2)],
                label: None,
            });
            operations.extend(rz_operations);
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![*target],
                params: smallvec![CircuitParam::Fixed(-FRAC_PI_2)],
                label: None,
            });
        }
        _ => unreachable!("RX/RY family was validated before basis-change emission"),
    }

    Ok(operations)
}

fn direct_qubits(
    view: &McGateOperandView<'_>,
    target: Qubit,
    direct_gate: StandardGate,
) -> smallvec::SmallVec<[Qubit; 3]> {
    match direct_gate {
        StandardGate::RX | StandardGate::RY => smallvec![target],
        StandardGate::CRX | StandardGate::CRY => smallvec![view.all_controls()[0], target],
        _ => unreachable!("direct RX/RY family gate must be single-control or target-only"),
    }
}

fn ensure_fixed_multi_control_parameter(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    match params[0] {
        CircuitParam::Fixed(_) => Ok(()),
        CircuitParam::Index(_) => Err(mc_gate_view_error(
            view,
            config,
            "symbolic RZ-family parameters require theta/2 arithmetic for RX/RY basis-change compensation, which is not supported yet".to_string(),
        )),
    }
}

fn lifted_rz_operation_count(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
) -> Result<usize, CompilerError> {
    let control_count = view.total_control_count();
    debug_assert!(control_count >= 2);

    let target_phase_count =
        controlled_phase_component_operation_count(view, config, control_count, "RX/RY")?;
    let compensation_count =
        controlled_phase_component_operation_count(view, config, control_count - 1, "RX/RY")?;
    target_phase_count
        .checked_add(compensation_count)
        .ok_or_else(|| {
            mc_gate_view_error(
                view,
                config,
                "RX/RY-family lifted RZ operation count overflow".to_string(),
            )
        })
}

fn decompose_lifted_rz(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        unreachable!("RX/RY family target arity was validated before RZ lifting")
    };
    let control_count = view.total_control_count();
    debug_assert!(control_count >= 2);

    let added_control_count = u8::try_from(control_count - 1).map_err(|_| {
        mc_gate_view_error(
            view,
            config,
            format!("RX/RY-family control count {control_count} exceeds supported MCGate arity"),
        )
    })?;
    let rz_gate = MCGate::new(added_control_count, StandardGate::CRZ);
    let mut rz_qubits = Vec::with_capacity(control_count + 1);
    rz_qubits.extend_from_slice(view.all_controls());
    rz_qubits.push(*target);

    decompose_mc_gate(&rz_gate, &rz_qubits, params, config).map_err(|source| match source {
        CompilerError::TransformFailed { name, reason } => CompilerError::TransformFailed {
            name,
            reason: format!("RX/RY-family RZ basis-change failed: {reason}"),
        },
        other => other,
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
                "RX/RY-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}
