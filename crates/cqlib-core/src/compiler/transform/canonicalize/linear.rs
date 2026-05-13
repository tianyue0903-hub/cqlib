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
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, ParameterValue,
    StandardGate, WhileLoopGate,
};
use crate::compiler::error::CompilerError;
use smallvec::SmallVec;

use super::config::CanonicalizeConfig;
use super::equivalence::{operations_equivalent, pending_operations_equivalent};
use super::ops::{
    canonicalize_barrier_qubits, is_barrier_instruction, merge_operation_labels,
    resolve_operation_param, resolve_parameter_value, should_drop_operation,
};
use super::standard_gate_normalize::{
    GlobalPhasePolicy, NormalizedStandardOp, normalize_standard_gate,
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

#[derive(Debug, Clone)]
struct CanonicalInstruction {
    instruction: Instruction,
    qubits: Option<SmallVec<[crate::circuit::Qubit; 3]>>,
}

impl CanonicalInstruction {
    fn new(instruction: Instruction) -> Self {
        Self {
            instruction,
            qubits: None,
        }
    }

    fn with_qubits(instruction: Instruction, qubits: SmallVec<[crate::circuit::Qubit; 3]>) -> Self {
        Self {
            instruction,
            qubits: Some(qubits),
        }
    }
}

struct StructuralCanonicalizer<'a> {
    circuit: &'a Circuit,
    config: &'a CanonicalizeConfig,
}

impl<'a> StructuralCanonicalizer<'a> {
    fn new(circuit: &'a Circuit, config: &'a CanonicalizeConfig) -> Self {
        Self { circuit, config }
    }

    fn run(&self) -> Result<StructuralCanonicalizeResult, CompilerError> {
        let sequence = self.canonicalize_operations(self.circuit.operations())?;

        if !sequence.changed {
            return Ok(StructuralCanonicalizeResult {
                circuit: None,
                changed: false,
            });
        }

        let mut rebuilt = Circuit::from_qubits(self.circuit.qubits())?;
        rebuilt.set_global_phase(self.circuit.global_phase());

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

    fn canonicalize_operations(
        &self,
        operations: &[Operation],
    ) -> Result<PendingSequenceResult, CompilerError> {
        let mut out = Vec::with_capacity(operations.len());

        for operation in operations {
            for canonical in self.canonicalize_operation(operation)? {
                push_canonical_operation(&mut out, canonical, self.config);
            }
        }

        Ok(PendingSequenceResult {
            changed: !pending_operations_equivalent(operations, &out, self.circuit),
            operations: out,
        })
    }

    fn canonicalize_operation(
        &self,
        operation: &Operation,
    ) -> Result<Vec<PendingOperation>, CompilerError> {
        let canonical_instruction = self.canonicalize_instruction(operation.instruction.clone())?;
        let instruction = canonical_instruction.instruction;
        let qubits = canonical_instruction
            .qubits
            .unwrap_or_else(|| canonicalize_barrier_qubits(&instruction, &operation.qubits));

        if should_drop_operation(self.circuit, operation, &instruction, &qubits, self.config)? {
            return Ok(vec![]);
        }

        if let Instruction::Standard(gate) = instruction {
            return self.canonicalize_standard_operation(operation, gate, qubits);
        }

        let pending_params: SmallVec<[ParameterValue; 3]> = operation
            .params
            .iter()
            .map(|param| resolve_parameter_value(self.circuit, param))
            .collect::<Result<_, _>>()?;

        Ok(vec![PendingOperation {
            instruction,
            qubits,
            params: pending_params,
            label: operation.label.clone(),
        }])
    }

    fn canonicalize_standard_operation(
        &self,
        operation: &Operation,
        gate: StandardGate,
        qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    ) -> Result<Vec<PendingOperation>, CompilerError> {
        let semantic_params: SmallVec<[crate::circuit::Parameter; 3]> = operation
            .params
            .iter()
            .map(|param| resolve_operation_param(self.circuit, param))
            .collect::<Result<_, _>>()?;

        let normalized = if self.config.normalizes_parameters() {
            normalize_standard_gate(gate, &semantic_params, GlobalPhasePolicy::Preserve)
        } else {
            vec![NormalizedStandardOp {
                gate,
                params: operation
                    .params
                    .iter()
                    .map(|param| resolve_parameter_value(self.circuit, param))
                    .collect::<Result<_, _>>()?,
            }]
        };

        if normalized.is_empty() && operation.label.is_some() {
            return Ok(vec![PendingOperation {
                instruction: Instruction::Standard(gate),
                qubits,
                params: operation
                    .params
                    .iter()
                    .map(|param| resolve_parameter_value(self.circuit, param))
                    .collect::<Result<_, _>>()?,
                label: operation.label.clone(),
            }]);
        }

        Ok(build_pending_standard_ops(
            normalized,
            qubits,
            operation.label.clone(),
        ))
    }

    fn canonicalize_instruction(
        &self,
        instruction: Instruction,
    ) -> Result<CanonicalInstruction, CompilerError> {
        let instruction = if self.config.canonicalizes_instruction_form() {
            instruction.canonicalize_form()
        } else {
            instruction
        };

        if !self.config.recurses_control_flow() {
            return Ok(CanonicalInstruction::new(instruction));
        }

        match instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                let true_body = self.canonicalize_control_flow_body(gate.true_body())?;
                let false_body = gate
                    .false_body()
                    .map(|body| self.canonicalize_control_flow_body(body))
                    .transpose()?;
                if !true_body.changed && false_body.as_ref().is_none_or(|body| !body.changed) {
                    return Ok(CanonicalInstruction::new(Instruction::ControlFlowGate(
                        ControlFlow::IfElse(gate),
                    )));
                }

                let instruction =
                    Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                        gate.condition(),
                        true_body.operations,
                        false_body.map(|body| body.operations),
                    )));
                let qubits = control_flow_operation_qubits(&instruction);
                Ok(CanonicalInstruction::with_qubits(instruction, qubits))
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                let body = self.canonicalize_control_flow_body(gate.body())?;
                if !body.changed {
                    return Ok(CanonicalInstruction::new(Instruction::ControlFlowGate(
                        ControlFlow::WhileLoop(gate),
                    )));
                }

                let instruction = Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                    WhileLoopGate::new(gate.condition(), body.operations),
                ));
                let qubits = control_flow_operation_qubits(&instruction);
                Ok(CanonicalInstruction::with_qubits(instruction, qubits))
            }
            _ => Ok(CanonicalInstruction::new(instruction)),
        }
    }

    fn canonicalize_control_flow_body(
        &self,
        body: &[Operation],
    ) -> Result<BodySequenceResult, CompilerError> {
        let mut out = Vec::with_capacity(body.len());

        for operation in body {
            for canonical in self.canonicalize_body_operation(operation)? {
                push_canonical_body_operation(&mut out, canonical, self.config);
            }
        }

        Ok(BodySequenceResult {
            changed: !operations_equivalent(body, &out, self.circuit, self.circuit),
            operations: out,
        })
    }

    fn canonicalize_body_operation(
        &self,
        operation: &Operation,
    ) -> Result<Vec<Operation>, CompilerError> {
        let canonical_instruction = self.canonicalize_instruction(operation.instruction.clone())?;
        let instruction = canonical_instruction.instruction;
        let qubits = canonical_instruction
            .qubits
            .unwrap_or_else(|| canonicalize_barrier_qubits(&instruction, &operation.qubits));

        if should_drop_operation(self.circuit, operation, &instruction, &qubits, self.config)? {
            return Ok(vec![]);
        }

        if let Instruction::Standard(gate) = instruction {
            return self.canonicalize_body_standard_operation(operation, gate, qubits);
        }

        Ok(vec![Operation {
            instruction,
            qubits,
            params: operation.params.clone(),
            label: operation.label.clone(),
        }])
    }

    fn canonicalize_body_standard_operation(
        &self,
        operation: &Operation,
        gate: StandardGate,
        qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    ) -> Result<Vec<Operation>, CompilerError> {
        let semantic_params: SmallVec<[crate::circuit::Parameter; 3]> = operation
            .params
            .iter()
            .map(|param| resolve_operation_param(self.circuit, param))
            .collect::<Result<_, _>>()?;

        let all_params_fixed = semantic_params
            .iter()
            .all(|param| param.evaluate(&None).is_ok());
        if !self.config.normalizes_parameters() || !all_params_fixed {
            return Ok(vec![Operation {
                instruction: Instruction::Standard(gate),
                qubits,
                params: operation.params.clone(),
                label: operation.label.clone(),
            }]);
        }

        let normalized =
            normalize_standard_gate(gate, &semantic_params, GlobalPhasePolicy::Preserve);

        if normalized.is_empty() && operation.label.is_some() {
            return Ok(vec![Operation {
                instruction: Instruction::Standard(gate),
                qubits,
                params: operation.params.clone(),
                label: operation.label.clone(),
            }]);
        }

        Ok(build_body_standard_ops(
            normalized,
            qubits,
            operation.label.clone(),
        ))
    }
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
    StructuralCanonicalizer::new(circuit, config).run()
}

fn build_pending_standard_ops(
    normalized: Vec<NormalizedStandardOp>,
    qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    label: Option<Box<str>>,
) -> Vec<PendingOperation> {
    let label_index = standard_label_index(&normalized);
    normalized
        .into_iter()
        .enumerate()
        .map(|(index, op)| PendingOperation {
            instruction: Instruction::Standard(op.gate),
            qubits: normalized_qubits(op.gate, &qubits),
            params: op.params,
            label: if label_index == Some(index) {
                label.clone()
            } else {
                None
            },
        })
        .collect()
}

fn build_body_standard_ops(
    normalized: Vec<NormalizedStandardOp>,
    qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    label: Option<Box<str>>,
) -> Vec<Operation> {
    let label_index = standard_label_index(&normalized);
    normalized
        .into_iter()
        .enumerate()
        .map(|(index, op)| Operation {
            instruction: Instruction::Standard(op.gate),
            qubits: normalized_qubits(op.gate, &qubits),
            params: op
                .params
                .into_iter()
                .map(parameter_value_to_circuit_param)
                .collect(),
            label: if label_index == Some(index) {
                label.clone()
            } else {
                None
            },
        })
        .collect()
}

fn standard_label_index(normalized: &[NormalizedStandardOp]) -> Option<usize> {
    normalized
        .iter()
        .position(|op| op.gate != StandardGate::GPhase)
        .or_else(|| (!normalized.is_empty()).then_some(0))
}

fn normalized_qubits(
    gate: StandardGate,
    original: &SmallVec<[crate::circuit::Qubit; 3]>,
) -> SmallVec<[crate::circuit::Qubit; 3]> {
    if gate.num_qubits() == 0 {
        SmallVec::new()
    } else {
        original.clone()
    }
}

fn control_flow_operation_qubits(
    instruction: &Instruction,
) -> SmallVec<[crate::circuit::Qubit; 3]> {
    let mut qubits = SmallVec::new();
    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            collect_operation_qubits(gate.true_body(), &mut qubits);
            if let Some(false_body) = gate.false_body() {
                collect_operation_qubits(false_body, &mut qubits);
            }
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            collect_operation_qubits(gate.body(), &mut qubits);
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        _ => {}
    }
    qubits
}

fn collect_operation_qubits(
    operations: &[Operation],
    output: &mut SmallVec<[crate::circuit::Qubit; 3]>,
) {
    for operation in operations {
        for &qubit in &operation.qubits {
            push_unique_qubit(output, qubit);
        }
    }
}

fn push_unique_qubit(
    output: &mut SmallVec<[crate::circuit::Qubit; 3]>,
    qubit: crate::circuit::Qubit,
) {
    if !output.contains(&qubit) {
        output.push(qubit);
    }
}

fn parameter_value_to_circuit_param(value: ParameterValue) -> CircuitParam {
    match value {
        ParameterValue::Fixed(value) => CircuitParam::Fixed(value),
        ParameterValue::Param(_) => unreachable!("fixed-only normalization produced a symbol"),
    }
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
