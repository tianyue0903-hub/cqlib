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

//! Pauli-interaction lowering for multi-controlled standard gates.
//!
//! `RZZ` is lowered to `CX; RZ; CX`. `RXX`, `RYY`, and `RZX` add local
//! basis changes around that `RZZ` body. Each base operation is immediately
//! lifted by the outer controls and recursively decomposed by `mc_gate`, so
//! this module never leaves a temporary `Instruction::McGate` in its output.

use super::decompose::{
    AncillaMode, McGateDecomposeConfig, McGateOperandView, decompose_mc_gate, mc_gate_view_error,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;
use std::f64::consts::FRAC_PI_2;

/// Decomposes Pauli-interaction MCGates through controlled base decompositions.
pub(super) fn decompose_pauli_interaction_family(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let [first_target, second_target] = view.targets() else {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "PauliInteraction-family gate {} must have exactly two targets, got {}",
                view.base_gate(),
                view.targets().len()
            ),
        ));
    };

    match view.base_gate() {
        StandardGate::RXX | StandardGate::RYY | StandardGate::RZZ | StandardGate::RZX => {}
        base_gate => {
            return Err(mc_gate_view_error(
                view,
                config,
                format!("gate {base_gate} is not in the PauliInteraction family"),
            ));
        }
    }

    if view.total_control_count() == 0 {
        if config.max_expansion_ops == 0 {
            return Err(mc_gate_view_error(
                view,
                config,
                format!(
                    "PauliInteraction-family expansion would emit 1 operations, exceeding max_expansion_ops={}",
                    config.max_expansion_ops
                ),
            ));
        }
        return Ok(vec![Operation {
            instruction: Instruction::Standard(view.base_gate()),
            qubits: smallvec![*first_target, *second_target],
            params: params.iter().cloned().collect(),
            label: None,
        }]);
    }

    if config.ancilla_mode == AncillaMode::NoAncilla && view.total_control_count() >= 2 {
        return Err(mc_gate_view_error(
            view,
            config,
            format!(
                "no-ancilla PauliInteraction decomposition with {} controls would rely on non-exact no-ancilla MCX global phase; use clean ancillas for exact lowering",
                view.total_control_count()
            ),
        ));
    }

    let base_operations =
        base_interaction_decomposition(view, params, *first_target, *second_target);
    lift_and_decompose_base_operations(view, &base_operations, config)
}

fn base_interaction_decomposition(
    view: &McGateOperandView<'_>,
    params: &[CircuitParam],
    first_target: Qubit,
    second_target: Qubit,
) -> Vec<Operation> {
    let mut operations = Vec::new();

    match view.base_gate() {
        StandardGate::RXX => {
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![first_target],
                params: smallvec![],
                label: None,
            });
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![second_target],
                params: smallvec![],
                label: None,
            });
            emit_rzz_body(
                &mut operations,
                first_target,
                second_target,
                params[0].clone(),
            );
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![first_target],
                params: smallvec![],
                label: None,
            });
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![second_target],
                params: smallvec![],
                label: None,
            });
        }
        StandardGate::RYY => {
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![first_target],
                params: smallvec![CircuitParam::Fixed(FRAC_PI_2)],
                label: None,
            });
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![second_target],
                params: smallvec![CircuitParam::Fixed(FRAC_PI_2)],
                label: None,
            });
            emit_rzz_body(
                &mut operations,
                first_target,
                second_target,
                params[0].clone(),
            );
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![first_target],
                params: smallvec![CircuitParam::Fixed(-FRAC_PI_2)],
                label: None,
            });
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::RX),
                qubits: smallvec![second_target],
                params: smallvec![CircuitParam::Fixed(-FRAC_PI_2)],
                label: None,
            });
        }
        StandardGate::RZZ => {
            emit_rzz_body(
                &mut operations,
                first_target,
                second_target,
                params[0].clone(),
            );
        }
        StandardGate::RZX => {
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![second_target],
                params: smallvec![],
                label: None,
            });
            emit_rzz_body(
                &mut operations,
                first_target,
                second_target,
                params[0].clone(),
            );
            operations.push(Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![second_target],
                params: smallvec![],
                label: None,
            });
        }
        _ => unreachable!("PauliInteraction gate was validated before base decomposition"),
    }

    operations
}

fn emit_rzz_body(
    operations: &mut Vec<Operation>,
    first_target: Qubit,
    second_target: Qubit,
    theta: CircuitParam,
) {
    // RZZ(theta) is exactly CX(a,b); RZ(theta) on b; CX(a,b), with no
    // target-less global phase that would become illegal under controls.
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![first_target, second_target],
        params: smallvec![],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![second_target],
        params: smallvec![theta],
        label: None,
    });
    operations.push(Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![first_target, second_target],
        params: smallvec![],
        label: None,
    });
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
                    "PauliInteraction-family expansion budget underflow".to_string(),
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
        unreachable!("PauliInteraction base decomposition emits only standard gates")
    };

    let added_control_count = u8::try_from(view.total_control_count()).map_err(|_| {
        mc_gate_view_error(
            view,
            config,
            format!(
                "PauliInteraction-family control count {} exceeds supported MCGate arity",
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
                reason: format!(
                    "PauliInteraction-family control-lifting of {base_gate} failed: {reason}"
                ),
            },
            other => other,
        }
    })
}
