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

//! Parameter and global-phase canonicalization for Task 2.
//!
//! This file intentionally does not perform structural cleanup such as barrier
//! merging or no-op deletion. Its only job is to normalize symbolic parameter
//! storage and the circuit global phase by reusing existing `Circuit`
//! infrastructure.
//!
//! After that rebuild, we still call `set_global_phase(...)` explicitly so the
//! global phase is normalized through the same public API the rest of the
//! circuit layer uses.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, ParameterValue,
    WhileLoopGate,
};
use crate::compiler::error::CompilerError;
use std::collections::HashMap;

/// Canonicalizes symbolic parameters and global phase.
///
/// This function rebuilds the circuit rather than mutating it in place because
/// parameter simplification can change the parameter table (folding constants,
/// deduplicating expressions, and rebuilding the symbol table). The rebuild is
/// performed recursively so that control-flow bodies stay consistent with the
/// parent circuit's new parameter pool.
///
/// Global phase is normalized separately after the operations are remapped.
pub fn canonicalize_parameter_phase(circuit: &Circuit) -> Result<Circuit, CompilerError> {
    let empty_bindings: HashMap<&str, f64> = HashMap::new();
    let mut canonical = Circuit::from_qubits(circuit.qubits())?;
    let index_map = build_parameter_index_map(circuit, &mut canonical, &empty_bindings)?;

    for op in circuit.operations() {
        append_remapped_operation(op, &index_map, &mut canonical)?;
    }

    let canonical_phase = circuit
        .global_phase()
        .simplify()
        .map_err(|e| CompilerError::InvalidContextState(format!("{:?}", e)))?;
    canonical.set_global_phase(canonical_phase);

    Ok(canonical)
}

/// Builds a mapping from old parameter indices to new `CircuitParam` values.
///
/// For each parameter in the source circuit:
/// - If it evaluates to a fixed number under `bindings`, map it to
///   `CircuitParam::Fixed(value)`.
/// - Otherwise, simplify the symbolic expression, intern it into `target`'s
///   parameter table, and map it to `CircuitParam::Index(new_index)`.
///
/// The returned vector is indexed by the old parameter index; its elements are
/// the remapped forms.
fn build_parameter_index_map(
    circuit: &Circuit,
    target: &mut Circuit,
    bindings: &HashMap<&str, f64>,
) -> Result<Vec<CircuitParam>, CompilerError> {
    let mut index_map = Vec::with_capacity(circuit.parameters().len());

    for param in circuit.parameters().iter() {
        if let Ok(val) = param.evaluate(&Some(bindings.clone())) {
            index_map.push(CircuitParam::Fixed(val));
            continue;
        }

        let simplified = param
            .clone()
            .simplify()
            .map_err(|e| CompilerError::InvalidContextState(format!("{:?}", e)))?;
        let (idx, _) = target.add_parameter(simplified);
        index_map.push(CircuitParam::Index(idx as u32));
    }

    Ok(index_map)
}

/// Appends a remapped top-level operation to `target`.
fn append_remapped_operation(
    operation: &Operation,
    index_map: &[CircuitParam],
    target: &mut Circuit,
) -> Result<(), CompilerError> {
    let instruction = remap_instruction(&operation.instruction, index_map, target)?;
    let params = remap_operation_params(&operation.params, index_map, target)?;

    target.append(
        instruction,
        operation.qubits.iter().copied(),
        params,
        operation.label.as_deref(),
    )?;

    Ok(())
}

/// Remaps parameters inside an instruction, recursing into control-flow bodies.
fn remap_instruction(
    instruction: &Instruction,
    index_map: &[CircuitParam],
    target: &mut Circuit,
) -> Result<Instruction, CompilerError> {
    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            let true_body = remap_body_operations(gate.true_body(), index_map, target)?;
            let false_body = gate
                .false_body()
                .map(|body| remap_body_operations(body, index_map, target))
                .transpose()?;
            Ok(Instruction::ControlFlowGate(ControlFlow::IfElse(
                IfElseGate::new(gate.condition(), true_body, false_body),
            )))
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            let body = remap_body_operations(gate.body(), index_map, target)?;
            Ok(Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                WhileLoopGate::new(gate.condition(), body),
            )))
        }
        _ => Ok(instruction.clone()),
    }
}

/// Remaps parameters across a slice of body operations.
fn remap_body_operations(
    operations: &[Operation],
    index_map: &[CircuitParam],
    target: &mut Circuit,
) -> Result<Vec<Operation>, CompilerError> {
    operations
        .iter()
        .map(|op| remap_body_operation(op, index_map, target))
        .collect()
}

/// Remaps parameters inside a single body operation.
fn remap_body_operation(
    operation: &Operation,
    index_map: &[CircuitParam],
    target: &mut Circuit,
) -> Result<Operation, CompilerError> {
    Ok(Operation {
        instruction: remap_instruction(&operation.instruction, index_map, target)?,
        qubits: operation.qubits.clone(),
        params: remap_operation_circuit_params(&operation.params, index_map)?,
        label: operation.label.clone(),
    })
}

/// Remaps operation parameters into `ParameterValue`s for appending to a circuit.
fn remap_operation_params(
    params: &[CircuitParam],
    index_map: &[CircuitParam],
    target: &Circuit,
) -> Result<Vec<ParameterValue>, CompilerError> {
    params
        .iter()
        .map(|param| remap_circuit_param(param, index_map))
        .map(|mapped| mapped.and_then(|param| circuit_param_to_value(param, target)))
        .collect()
}

/// Remaps operation parameters into `CircuitParam`s for a body operation.
fn remap_operation_circuit_params(
    params: &[CircuitParam],
    index_map: &[CircuitParam],
) -> Result<smallvec::SmallVec<[CircuitParam; 1]>, CompilerError> {
    params
        .iter()
        .map(|param| remap_circuit_param(param, index_map))
        .collect()
}

/// Remaps a single `CircuitParam` through the index map.
fn remap_circuit_param(
    param: &CircuitParam,
    index_map: &[CircuitParam],
) -> Result<CircuitParam, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(CircuitParam::Fixed(*value)),
        CircuitParam::Index(index) => index_map.get(*index as usize).cloned().ok_or_else(|| {
            CompilerError::InvalidContextState(format!(
                "invalid control-flow body parameter index {} during parameter normalization",
                index
            ))
        }),
    }
}

/// Converts a remapped `CircuitParam` into a `ParameterValue` using the circuit's table.
fn circuit_param_to_value(
    param: CircuitParam,
    circuit: &Circuit,
) -> Result<ParameterValue, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(value)),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(index as usize)
            .cloned()
            .map(ParameterValue::from)
            .ok_or_else(|| {
                CompilerError::InvalidContextState(format!(
                    "parameter normalization produced invalid parameter index {}",
                    index
                ))
            }),
    }
}

/// Detects whether parameter-phase canonicalization changed the circuit.
///
/// This is a conservative structural check, not full quantum semantic
/// equivalence. It compares:
/// - qubit sets
/// - symbol tables
/// - parameter tables
/// - global phase representations
/// - operation counts and parameter references
///
/// It intentionally does *not* compare instructions structurally beyond
/// recursing into control-flow bodies; this guarantees that parameter-phase
/// logic cannot accidentally mutate circuit structure.
pub fn parameter_phase_changed(before: &Circuit, after: &Circuit) -> bool {
    if before.qubits() != after.qubits() {
        return true;
    }

    if before.symbols() != after.symbols() {
        return true;
    }

    if before.parameters() != after.parameters() {
        return true;
    }

    if before.global_phase() != after.global_phase() {
        return true;
    }

    if before.operations().len() != after.operations().len() {
        return true;
    }

    before
        .operations()
        .iter()
        .zip(after.operations())
        .any(|(lhs, rhs)| !operations_equal_for_parameter_phase(lhs, rhs))
}

/// Returns `true` if two operations are equal for parameter-phase tracking.
fn operations_equal_for_parameter_phase(lhs: &Operation, rhs: &Operation) -> bool {
    instructions_equal_for_parameter_phase(&lhs.instruction, &rhs.instruction)
        && lhs.qubits == rhs.qubits
        && circuit_params_equal(&lhs.params, &rhs.params)
        && lhs.label.as_deref() == rhs.label.as_deref()
}

/// Returns `true` if two instructions are equal for parameter-phase tracking,
/// recursing into control-flow bodies and nested gates.
fn instructions_equal_for_parameter_phase(lhs: &Instruction, rhs: &Instruction) -> bool {
    match (lhs, rhs) {
        (
            Instruction::ControlFlowGate(ControlFlow::IfElse(lhs)),
            Instruction::ControlFlowGate(ControlFlow::IfElse(rhs)),
        ) => {
            lhs.condition() == rhs.condition()
                && operation_slices_equal_for_parameter_phase(lhs.true_body(), rhs.true_body())
                && match (lhs.false_body(), rhs.false_body()) {
                    (Some(lhs), Some(rhs)) => operation_slices_equal_for_parameter_phase(lhs, rhs),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(lhs)),
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(rhs)),
        ) => {
            lhs.condition() == rhs.condition()
                && operation_slices_equal_for_parameter_phase(lhs.body(), rhs.body())
        }
        (Instruction::Standard(lhs), Instruction::Standard(rhs)) => lhs == rhs,
        (Instruction::McGate(lhs), Instruction::McGate(rhs)) => lhs == rhs,
        (Instruction::Directive(lhs), Instruction::Directive(rhs)) => lhs == rhs,
        (Instruction::Delay, Instruction::Delay) => true,
        (Instruction::CircuitGate(lhs), Instruction::CircuitGate(rhs)) => {
            lhs.name() == rhs.name()
                && lhs.num_qubits() == rhs.num_qubits()
                && lhs.num_params() == rhs.num_params()
                && !parameter_phase_changed(lhs.circuit().circuit(), rhs.circuit().circuit())
        }
        (Instruction::UnitaryGate(lhs), Instruction::UnitaryGate(rhs)) => {
            lhs.label() == rhs.label()
                && lhs.num_qubits() == rhs.num_qubits()
                && lhs.matrix_repr() == rhs.matrix_repr()
                && lhs.matrix_params() == rhs.matrix_params()
                && match (lhs.circuit(), rhs.circuit()) {
                    (Some(lhs), Some(rhs)) => {
                        !parameter_phase_changed(lhs.circuit(), rhs.circuit())
                    }
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => false,
    }
}

/// Returns `true` if two operation slices are equal for parameter-phase tracking.
fn operation_slices_equal_for_parameter_phase(lhs: &[Operation], rhs: &[Operation]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(lhs, rhs)| operations_equal_for_parameter_phase(lhs, rhs))
}

/// Returns `true` if two circuit parameter lists are equal index-for-index.
fn circuit_params_equal(lhs: &[CircuitParam], rhs: &[CircuitParam]) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(lhs, rhs)| match (lhs, rhs) {
            (CircuitParam::Index(a), CircuitParam::Index(b)) => a == b,
            (CircuitParam::Fixed(a), CircuitParam::Fixed(b)) => a == b,
            _ => false,
        })
}
