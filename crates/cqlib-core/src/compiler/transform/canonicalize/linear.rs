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

//! Linear-structure canonicalization.
//!
//! This module rebuilds a circuit's top-level operation sequence (and any
//! control-flow bodies) by applying structural canonicalization rules:
//! - collapsing multi-controlled gates into standard forms
//! - merging adjacent barriers
//! - dropping trivial no-ops
//! - sorting/deduplicating barrier qubit lists
//!
//! The rebuild uses the public `Circuit` construction API so that parameter
//! interning, qubit validation, and control-flow gate construction all flow
//! through the same code paths as hand-written circuits.

use crate::circuit::{
    Circuit, ControlFlow, IfElseGate, Instruction, Operation, ParameterValue, WhileLoopGate,
};
use crate::compiler::error::CompilerError;
use smallvec::SmallVec;

use super::config::CanonicalizeConfig;
use super::equivalence::{operations_equivalent, pending_operations_equivalent};
use super::ops::{
    canonicalize_barrier_qubits, is_barrier_instruction, merge_operation_labels,
    resolve_parameter_value, should_drop_operation,
};

/// Result of a structural canonicalization pass over a full circuit.
#[derive(Debug, Clone)]
pub(crate) struct StructuralCanonicalizeResult {
    /// Rebuilt circuit, or `None` if nothing changed.
    pub(crate) circuit: Option<Circuit>,
    /// Whether the pass made any changes.
    pub(crate) changed: bool,
}

/// Result of canonicalizing a linear operation sequence into pending operations.
#[derive(Debug, Clone)]
pub(crate) struct PendingSequenceResult {
    /// Canonicalized operations.
    pub(crate) operations: Vec<PendingOperation>,
    /// Whether the sequence differs from the original.
    pub(crate) changed: bool,
}

/// Result of canonicalizing a control-flow body.
#[derive(Debug, Clone)]
pub(crate) struct BodySequenceResult {
    /// Canonicalized body operations.
    pub(crate) operations: Vec<Operation>,
    /// Whether the body differs from the original.
    pub(crate) changed: bool,
}

/// A canonicalized operation with resolved parameter values, ready to append to a circuit.
#[derive(Debug, Clone)]
pub(crate) struct PendingOperation {
    pub(crate) instruction: Instruction,
    pub(crate) qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    pub(crate) params: SmallVec<[ParameterValue; 3]>,
    pub(crate) label: Option<Box<str>>,
}

/// Rebuilds a circuit by canonicalizing its linear operation sequences.
///
/// Instead of mutating the internal `data` vector directly, this function
/// rebuilds the circuit using the public `Circuit` construction API. This
/// ensures that parameter interning, qubit validation, and control-flow gate
/// construction all follow the same paths as hand-written circuits.
pub(crate) fn canonicalize_linear_structure(
    circuit: &Circuit,
    config: &CanonicalizeConfig,
) -> Result<StructuralCanonicalizeResult, CompilerError> {
    let sequence = canonicalize_operations(circuit, circuit.operations(), config)?;

    if !sequence.changed {
        return Ok(StructuralCanonicalizeResult {
            circuit: None,
            changed: false,
        });
    }

    let mut rebuilt = Circuit::from_qubits(circuit.qubits())?;
    rebuilt.set_global_phase(circuit.global_phase());

    for operation in sequence.operations {
        rebuilt.append(
            operation.instruction,
            operation.qubits,
            operation.params,
            operation.label.as_deref(),
        )?;
    }

    Ok(StructuralCanonicalizeResult {
        circuit: Some(rebuilt),
        changed: true,
    })
}

/// Canonicalizes a linear operation sequence.
///
/// Used for both the top-level circuit and control-flow bodies so that the
/// canonicalization contract is identical everywhere. Each operation is
/// canonicalized individually, then pushed into the output vector via the
/// barrier-merge logic.
pub(crate) fn canonicalize_operations(
    circuit: &Circuit,
    operations: &[Operation],
    config: &CanonicalizeConfig,
) -> Result<PendingSequenceResult, CompilerError> {
    let mut out = Vec::with_capacity(operations.len());

    for operation in operations {
        let canonical = canonicalize_operation(circuit, operation, config)?;
        if let Some(canonical) = canonical {
            push_canonical_operation(&mut out, canonical, config);
        }
    }

    Ok(PendingSequenceResult {
        changed: !pending_operations_equivalent(operations, &out, circuit),
        operations: out,
    })
}

/// Canonicalizes a single top-level operation.
///
/// Steps: canonicalize instruction form, sort/deduplicate barrier qubits,
/// drop if it's a trivial no-op, and resolve parameters into `ParameterValue`s.
fn canonicalize_operation(
    circuit: &Circuit,
    operation: &Operation,
    config: &CanonicalizeConfig,
) -> Result<Option<PendingOperation>, CompilerError> {
    let instruction = canonicalize_instruction(circuit, operation.instruction.clone(), config)?;
    let qubits = canonicalize_barrier_qubits(&instruction, &operation.qubits);

    if should_drop_operation(circuit, operation, &instruction, &qubits, config)? {
        return Ok(None);
    }

    let pending_params: SmallVec<[ParameterValue; 3]> = operation
        .params
        .iter()
        .map(|param| resolve_parameter_value(circuit, param))
        .collect::<Result<_, _>>()?;

    Ok(Some(PendingOperation {
        instruction,
        qubits,
        params: pending_params,
        label: operation.label.clone(),
    }))
}

/// Canonicalizes an instruction and optionally recurses into control-flow bodies.
///
/// Steps:
/// 1. Collapse multi-controlled gates into standard forms if enabled.
/// 2. If `recurse_control_flow` is enabled, canonicalize the bodies of
///    `IfElse` and `WhileLoop` gates and rebuild the gate only when something
///    inside changed.
fn canonicalize_instruction(
    circuit: &Circuit,
    instruction: Instruction,
    config: &CanonicalizeConfig,
) -> Result<Instruction, CompilerError> {
    let instruction = if config.canonicalizes_instruction_form() {
        instruction.canonicalize_form()
    } else {
        instruction
    };

    if !config.recurses_control_flow() {
        return Ok(instruction);
    }

    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            let true_body = canonicalize_control_flow_body(gate.true_body(), circuit, config)?;
            let false_body = gate
                .false_body()
                .map(|body| canonicalize_control_flow_body(body, circuit, config))
                .transpose()?;
            if !true_body.changed && false_body.as_ref().is_none_or(|body| !body.changed) {
                return Ok(Instruction::ControlFlowGate(ControlFlow::IfElse(gate)));
            }
            Ok(Instruction::ControlFlowGate(ControlFlow::IfElse(
                IfElseGate::new(
                    gate.condition(),
                    true_body.operations,
                    false_body.map(|body| body.operations),
                ),
            )))
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            let body = canonicalize_control_flow_body(gate.body(), circuit, config)?;
            if !body.changed {
                return Ok(Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)));
            }
            Ok(Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                WhileLoopGate::new(gate.condition(), body.operations),
            )))
        }
        _ => Ok(instruction),
    }
}

/// Canonicalizes a control-flow body directly as an operation sequence.
///
/// Control-flow bodies are stored as raw `Vec<Operation>`, not full
/// `Circuit` objects. Rebuilding them through temporary circuits would
/// re-index symbolic parameters against a body-local pool and add unnecessary
/// allocation overhead. This function preserves the original `CircuitParam`
/// references into `parent_circuit` while applying the same structural rules
/// used at the top level.
fn canonicalize_control_flow_body(
    body: &[Operation],
    parent_circuit: &Circuit,
    config: &CanonicalizeConfig,
) -> Result<BodySequenceResult, CompilerError> {
    // Control-flow bodies are stored as naked `Vec<Operation>` values rather
    // than full `Circuit` objects. Rebuilding them through temporary circuits
    // adds avoidable allocation churn and also risks re-indexing symbolic
    // parameters against a body-local parameter pool. We therefore canonicalize
    // bodies directly as operation sequences and preserve their original
    // `CircuitParam` references into the parent circuit.
    let mut out = Vec::with_capacity(body.len());

    for operation in body {
        let canonical = canonicalize_body_operation(parent_circuit, operation, config)?;
        if let Some(canonical) = canonical {
            push_canonical_body_operation(&mut out, canonical, config);
        }
    }

    Ok(BodySequenceResult {
        changed: !operations_equivalent(body, &out, parent_circuit, parent_circuit),
        operations: out,
    })
}

/// Canonicalizes a single body operation, preserving `CircuitParam` references.
///
/// Unlike top-level operations, parameters are *not* resolved into
/// `ParameterValue`s so that the body continues to index into the parent
/// circuit's parameter pool.
fn canonicalize_body_operation(
    parent_circuit: &Circuit,
    operation: &Operation,
    config: &CanonicalizeConfig,
) -> Result<Option<Operation>, CompilerError> {
    let instruction =
        canonicalize_instruction(parent_circuit, operation.instruction.clone(), config)?;
    let qubits = canonicalize_barrier_qubits(&instruction, &operation.qubits);

    if should_drop_operation(parent_circuit, operation, &instruction, &qubits, config)? {
        return Ok(None);
    }

    Ok(Some(Operation {
        instruction,
        qubits,
        params: operation.params.clone(),
        label: operation.label.clone(),
    }))
}

/// Pushes a canonicalized body operation into the output vector.
fn push_canonical_body_operation(
    out: &mut Vec<Operation>,
    operation: Operation,
    config: &CanonicalizeConfig,
) {
    push_canonical_merged(out, operation, config);
}

/// Pushes a canonicalized top-level operation into the output vector.
fn push_canonical_operation(
    out: &mut Vec<PendingOperation>,
    operation: PendingOperation,
    config: &CanonicalizeConfig,
) {
    push_canonical_merged(out, operation, config);
}

/// Merges a new operation into the output vector, collapsing adjacent barriers.
///
/// Barrier merge rules:
/// - **Equal scope**: keep the left barrier, merge labels.
/// - **Left superset**: the left barrier already covers the right one; merge labels and drop the right.
/// - **Right superset**: the right barrier covers more qubits; replace the left with the right and merge labels.
/// - **Disjoint or partial overlap**: barriers cannot be merged, push the new one.
///
/// Adjacent barriers are treated as one synchronization boundary only when the
/// merged scope is identical to, or strictly covers, the absorbed scope.
/// Operation labels are preserved by joining unique label fragments in order.
///
/// This logic is shared between top-level `PendingOperation` and body `Operation`
/// via the `BarrierMergeOp` trait.
fn push_canonical_merged<O>(out: &mut Vec<O>, mut operation: O, config: &CanonicalizeConfig)
where
    O: BarrierMergeOp,
{
    if !config.merges_adjacent_barriers() || !is_barrier_instruction(operation.instruction()) {
        out.push(operation);
        return;
    }

    if let Some(last) = out.last_mut() {
        if is_barrier_instruction(last.instruction()) {
            match super::ops::compare_barrier_scope(last.qubits(), operation.qubits()) {
                super::ops::BarrierRelation::Equal | super::ops::BarrierRelation::LeftSuperset => {
                    let merged = merge_operation_labels(last.take_label(), operation.take_label());
                    last.set_label(merged);
                    return;
                }
                super::ops::BarrierRelation::RightSuperset => {
                    let merged = merge_operation_labels(operation.take_label(), last.take_label());
                    operation.set_label(merged);
                    *last = operation;
                    return;
                }
                super::ops::BarrierRelation::DisjointOrOverlap => {}
            }
        }
    }

    out.push(operation);
}

/// Trait that lets `push_canonical_merged` operate over both `Operation`
/// (used inside control-flow bodies) and `PendingOperation` (used at the top
/// level) without duplicating the barrier-merge logic.
trait BarrierMergeOp {
    fn instruction(&self) -> &Instruction;
    fn qubits(&self) -> &[crate::circuit::Qubit];
    fn take_label(&mut self) -> Option<Box<str>>;
    fn set_label(&mut self, label: Option<Box<str>>);
}

impl BarrierMergeOp for Operation {
    fn instruction(&self) -> &Instruction {
        &self.instruction
    }
    fn qubits(&self) -> &[crate::circuit::Qubit] {
        &self.qubits
    }
    fn take_label(&mut self) -> Option<Box<str>> {
        self.label.take()
    }
    fn set_label(&mut self, label: Option<Box<str>>) {
        self.label = label;
    }
}

impl BarrierMergeOp for PendingOperation {
    fn instruction(&self) -> &Instruction {
        &self.instruction
    }
    fn qubits(&self) -> &[crate::circuit::Qubit] {
        &self.qubits
    }
    fn take_label(&mut self) -> Option<Box<str>> {
        self.label.take()
    }
    fn set_label(&mut self, label: Option<Box<str>>) {
        self.label = label;
    }
}
