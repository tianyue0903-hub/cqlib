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

//! Phase-family lowering for multi-controlled standard gates.
//!
//! `S`, `SDG`, `T`, `TDG`, and `Phase(lambda)` are treated as
//! `Phase(theta)`. A controlled phase is a conditional phase on the conjunction
//! of all controls and the target qubit, so it is emitted either as an
//! ancilla-free parity phase polynomial or through an explicit clean flag.

use super::decompose::{McGateDecomposeConfig, McGateOperandView, mc_gate_view_error};
use super::phase_ops::{
    controlled_phase_component_operation_count, emit_controlled_phase_component,
};
use crate::circuit::{CircuitParam, Instruction, Operation, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

/// Decomposes phase-family MCGates using fixed-angle exact decompositions.
pub(super) fn decompose_phase_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "Phase-family gate {} must have exactly one target, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    if view.total_control_count() == 0 {
        let operation_params = match view.base_gate() {
            StandardGate::Phase => smallvec![params[0].clone()],
            _ => smallvec![],
        };
        return Ok(vec![Operation {
            instruction: Instruction::Standard(view.base_gate()),
            qubits: smallvec![*target],
            params: operation_params,
            label: None,
        }]);
    }

    let theta = fixed_phase_angle(view, params, config)?;
    let operation_count = controlled_phase_component_operation_count(
        view,
        config,
        view.total_control_count(),
        "Phase",
    )?;
    check_expansion_budget(view, config, operation_count)?;

    let mut operations = Vec::with_capacity(operation_count);
    emit_controlled_phase_component(
        &mut operations,
        view,
        view.all_controls(),
        *target,
        theta,
        config,
        "Phase",
    )?;
    Ok(operations)
}

fn fixed_phase_angle(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<f64, CompilerError> {
    match view.base_gate() {
        StandardGate::S => Ok(FRAC_PI_2),
        StandardGate::SDG => Ok(-FRAC_PI_2),
        StandardGate::T => Ok(FRAC_PI_4),
        StandardGate::TDG => Ok(-FRAC_PI_4),
        StandardGate::Phase => match &params[0] {
            CircuitParam::Fixed(theta) => Ok(*theta),
            CircuitParam::Index(_) => Err(mc_gate_view_error(
                view,
                config,
                "symbolic Phase-family parameters require parameter arithmetic, which is not supported yet".to_string(),
            )),
        },
        base_gate => Err(mc_gate_view_error(
            view,
            config,
            format!("gate {base_gate} is not in the Phase family"),
        )),
    }
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
                "Phase-family expansion would emit {operation_count} operations, exceeding max_expansion_ops={}",
                config.max_expansion_ops
            ),
        ));
    }

    Ok(())
}
