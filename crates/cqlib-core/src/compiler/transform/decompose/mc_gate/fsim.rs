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

//! FSIM lowering for multi-controlled standard gates.
//!
//! The controlled path uses a fixed whitelist decomposition:
//! `RXX(theta); RYY(theta); Phase(-phi / 2) on q0; CRZ(-phi) q0,q1`.
//! This decomposition contains no `GPhase` or target-less operation, so every
//! base operation can be safely lifted by the outer controls and recursively
//! lowered by `mc_gate`.

use super::decompose::{
    McGateDecomposeConfig, McGateOperandView, decompose_mc_gate, mc_gate_view_error,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

/// Decomposes FSIM MCGates through a no-GPhase whitelist decomposition.
pub(super) fn decompose_fsim_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [first_target, second_target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "FSIM-family gate {} must have exactly two targets, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    if view.base_gate() != StandardGate::FSIM {
        return Err(mc_gate_view_error(
            view,
            config,
            format!("gate {} is not in the FSIM family", view.base_gate()),
        ));
    }

    if view.total_control_count() == 0 {
        if config.max_expansion_ops == 0 {
            return Err(mc_gate_view_error(
                view,
                config,
                format!(
                    "FSIM-family expansion would emit 1 operations, exceeding max_expansion_ops={}",
                    config.max_expansion_ops
                ),
            ));
        }
        return Ok(vec![Operation {
            instruction: Instruction::Standard(StandardGate::FSIM),
            qubits: smallvec![*first_target, *second_target],
            params: params.iter().cloned().collect(),
            label: None,
        }]);
    }

    let [theta, phi] = fixed_fsim_parameters(view, params, config)?;
    let base_operations = base_fsim_decomposition(theta, phi, *first_target, *second_target);
    lift_and_decompose_base_operations(view, &base_operations, config)
}

fn fixed_fsim_parameters(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<[f64; 2], CompilerError> {
    match (&params[0], &params[1]) {
        (CircuitParam::Fixed(theta), CircuitParam::Fixed(phi)) => Ok([*theta, *phi]),
        _ => Err(mc_gate_view_error(
            view,
            config,
            "symbolic FSIM-family parameters require phi/2 arithmetic for the no-GPhase whitelist decomposition, which is not supported yet".to_string(),
        )),
    }
}

fn base_fsim_decomposition(
    theta: f64,
    phi: f64,
    first_target: Qubit,
    second_target: Qubit,
) -> Vec<Operation> {
    // This is the reviewed no-GPhase FSIM identity from the decomposition
    // rules. It is intentionally kept as a local whitelist instead of selected
    // dynamically from the knowledge-rule library.
    vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::RXX),
            qubits: smallvec![first_target, second_target],
            params: smallvec![CircuitParam::Fixed(theta)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::RYY),
            qubits: smallvec![first_target, second_target],
            params: smallvec![CircuitParam::Fixed(theta)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::Phase),
            qubits: smallvec![first_target],
            params: smallvec![CircuitParam::Fixed(-phi / 2.0)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CRZ),
            qubits: smallvec![first_target, second_target],
            params: smallvec![CircuitParam::Fixed(-phi)],
            label: None,
        },
    ]
}

fn lift_and_decompose_base_operations(
    view: &McGateOperandView<'_>,
    base_operations: &[Operation],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let mut remaining_budget = config.max_expansion_ops;
    let mut operations = Vec::new();

    for base_operation in base_operations {
        let nested_operations =
            lift_and_decompose_base_operation(view, base_operation, config, remaining_budget)?;
        remaining_budget = remaining_budget
            .checked_sub(nested_operations.len())
            .ok_or_else(|| {
                mc_gate_view_error(
                    view,
                    config,
                    "FSIM-family expansion budget underflow".to_string(),
                )
            })?;
        operations.extend(nested_operations);
    }

    Ok(operations)
}

fn lift_and_decompose_base_operation(
    view: &McGateOperandView<'_>,
    base_operation: &Operation,
    config: &McGateDecomposeConfig,
    remaining_budget: usize,
) -> Result<Vec<Operation>, CompilerError> {
    let Instruction::Standard(base_gate) = base_operation.instruction else {
        unreachable!("FSIM base decomposition emits only standard gates")
    };

    let added_control_count = u8::try_from(view.total_control_count()).map_err(|_| {
        mc_gate_view_error(
            view,
            config,
            format!(
                "FSIM-family control count {} exceeds supported MCGate arity",
                view.total_control_count()
            ),
        )
    })?;
    let gate = MCGate::new(added_control_count, base_gate);
    let mut qubits = Vec::with_capacity(view.total_control_count() + base_operation.qubits.len());
    qubits.extend_from_slice(view.all_controls());
    qubits.extend_from_slice(&base_operation.qubits);

    let nested_config = McGateDecomposeConfig {
        max_expansion_ops: remaining_budget,
        ..config.clone()
    };
    decompose_mc_gate(&gate, &qubits, &base_operation.params, &nested_config).map_err(|source| {
        match source {
            CompilerError::TransformFailed { name, reason } => CompilerError::TransformFailed {
                name,
                reason: format!("FSIM-family control-lifting of {base_gate} failed: {reason}"),
            },
            other => other,
        }
    })
}
