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

//! Structured preprocessing and rebuild helpers for control-flow-aware compile passes.

use crate::circuit::gate::control_flow::{ConditionView, ControlFlow, IfElseGate, WhileLoopGate};
use crate::circuit::gate::{Directive, Instruction};
use crate::circuit::{Circuit, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::compile::prepared::{PreparedCircuit, PreparedOperation};
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct PreparedSegment {
    pub(crate) operations: Vec<PreparedOperation>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedPassthroughOp {
    pub(crate) op: Operation,
    pub(crate) logical_qubits: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedIfElse {
    pub(crate) condition: ConditionView,
    pub(crate) condition_logical: usize,
    pub(crate) true_body: PreparedProgram,
    pub(crate) false_body: Option<PreparedProgram>,
    pub(crate) label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedWhileLoop {
    pub(crate) condition: ConditionView,
    pub(crate) condition_logical: usize,
    pub(crate) body: PreparedProgram,
    pub(crate) label: Option<Box<str>>,
}

#[derive(Debug, Clone)]
pub(crate) enum PreparedProgramItem {
    Segment(PreparedSegment),
    Passthrough(PreparedPassthroughOp),
    IfElse(PreparedIfElse),
    WhileLoop(PreparedWhileLoop),
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedProgram {
    pub(crate) logical_qubits: Vec<Qubit>,
    pub(crate) parameters: Vec<Parameter>,
    pub(crate) items: Vec<PreparedProgramItem>,
}

impl PreparedProgram {
    pub(crate) fn is_plain_linear(&self) -> bool {
        matches!(self.items.as_slice(), [PreparedProgramItem::Segment(_)])
    }

    pub(crate) fn flatten_interaction_circuit(&self) -> PreparedCircuit {
        let mut operations = Vec::new();
        self.collect_interaction_ops(&mut operations);
        PreparedCircuit {
            logical_qubits: self.logical_qubits.clone(),
            parameters: self.parameters.clone(),
            operations,
        }
    }

    fn collect_interaction_ops(&self, out: &mut Vec<PreparedOperation>) {
        for item in &self.items {
            match item {
                PreparedProgramItem::Segment(segment) => {
                    out.extend(segment.operations.iter().cloned())
                }
                PreparedProgramItem::Passthrough(_) => {}
                PreparedProgramItem::IfElse(node) => {
                    node.true_body.collect_interaction_ops(out);
                    if let Some(false_body) = &node.false_body {
                        false_body.collect_interaction_ops(out);
                    }
                }
                PreparedProgramItem::WhileLoop(node) => node.body.collect_interaction_ops(out),
            }
        }
    }
}

impl PreparedSegment {
    pub(crate) fn to_prepared_circuit(
        &self,
        logical_qubits: &[Qubit],
        parameters: &[Parameter],
    ) -> PreparedCircuit {
        PreparedCircuit {
            logical_qubits: logical_qubits.to_vec(),
            parameters: parameters.to_vec(),
            operations: self.operations.clone(),
        }
    }
}

pub(crate) fn preprocess_program(circuit: &Circuit) -> Result<PreparedProgram, CompileError> {
    let logical_qubits = circuit.qubits();
    let parameters: Vec<Parameter> = circuit.parameters().iter().cloned().collect();
    let logical_index_map: HashMap<Qubit, usize> = logical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, q)| (q, idx))
        .collect();

    let items = preprocess_operations(
        circuit.operations(),
        &logical_index_map,
        &logical_qubits,
        &parameters,
    )?;
    Ok(PreparedProgram {
        logical_qubits,
        parameters,
        items,
    })
}

fn preprocess_operations(
    operations: &[Operation],
    logical_index_map: &HashMap<Qubit, usize>,
    logical_qubits: &[Qubit],
    parameters: &[Parameter],
) -> Result<Vec<PreparedProgramItem>, CompileError> {
    let mut items = Vec::new();
    let mut segment_ops = Vec::new();

    for (op_index, op) in operations.iter().enumerate() {
        match &op.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                flush_segment(&mut items, &mut segment_ops);
                let Some(&condition_logical) = logical_index_map.get(&gate.condition().qubit)
                else {
                    return Err(CompileError::Internal(format!(
                        "condition qubit {} not found in circuit logical ordering",
                        gate.condition().qubit
                    )));
                };
                let true_body = PreparedProgram {
                    logical_qubits: logical_qubits.to_vec(),
                    parameters: parameters.to_vec(),
                    items: preprocess_operations(
                        gate.true_body(),
                        logical_index_map,
                        logical_qubits,
                        parameters,
                    )?,
                };
                let false_body = gate
                    .false_body()
                    .map(|ops| {
                        Ok::<PreparedProgram, CompileError>(PreparedProgram {
                            logical_qubits: logical_qubits.to_vec(),
                            parameters: parameters.to_vec(),
                            items: preprocess_operations(
                                ops,
                                logical_index_map,
                                logical_qubits,
                                parameters,
                            )?,
                        })
                    })
                    .transpose()?;
                items.push(PreparedProgramItem::IfElse(PreparedIfElse {
                    condition: gate.condition(),
                    condition_logical,
                    true_body,
                    false_body,
                    label: op.label.clone(),
                }));
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                flush_segment(&mut items, &mut segment_ops);
                let Some(&condition_logical) = logical_index_map.get(&gate.condition().qubit)
                else {
                    return Err(CompileError::Internal(format!(
                        "condition qubit {} not found in circuit logical ordering",
                        gate.condition().qubit
                    )));
                };
                let body = PreparedProgram {
                    logical_qubits: logical_qubits.to_vec(),
                    parameters: parameters.to_vec(),
                    items: preprocess_operations(
                        gate.body(),
                        logical_index_map,
                        logical_qubits,
                        parameters,
                    )?,
                };
                items.push(PreparedProgramItem::WhileLoop(PreparedWhileLoop {
                    condition: gate.condition(),
                    condition_logical,
                    body,
                    label: op.label.clone(),
                }));
            }
            Instruction::Directive(Directive::Barrier | Directive::Measure | Directive::Reset) => {
                flush_segment(&mut items, &mut segment_ops);
                let logical_qubits = op
                    .qubits
                    .iter()
                    .map(|q| {
                        logical_index_map.get(q).copied().ok_or_else(|| {
                            CompileError::Internal(format!(
                                "qubit {q} not found in circuit logical ordering"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                items.push(PreparedProgramItem::Passthrough(PreparedPassthroughOp {
                    op: op.clone(),
                    logical_qubits,
                }));
            }
            Instruction::Delay => {
                return Err(CompileError::UnsupportedInstruction {
                    op_index,
                    instruction: "Delay".to_string(),
                });
            }
            _ => {
                let arity = op.qubits.len();
                if arity != 1 && arity != 2 {
                    return Err(CompileError::UnsupportedArity { op_index, arity });
                }
                let mut logical = SmallVec::<[usize; 2]>::with_capacity(arity);
                for &q in &op.qubits {
                    let Some(&logical_idx) = logical_index_map.get(&q) else {
                        return Err(CompileError::Internal(format!(
                            "qubit {q} not found in circuit logical ordering"
                        )));
                    };
                    logical.push(logical_idx);
                }
                segment_ops.push(PreparedOperation {
                    op: op.clone(),
                    logical_qubits: logical,
                });
            }
        }
    }

    flush_segment(&mut items, &mut segment_ops);
    Ok(items)
}

fn flush_segment(items: &mut Vec<PreparedProgramItem>, segment_ops: &mut Vec<PreparedOperation>) {
    if segment_ops.is_empty() {
        return;
    }
    items.push(PreparedProgramItem::Segment(PreparedSegment {
        operations: std::mem::take(segment_ops),
    }));
}

pub(crate) fn map_program_static(
    program: &PreparedProgram,
    mapping_idx: &[usize],
    physical_qubits: &[Qubit],
) -> Vec<Operation> {
    map_program_items_static(&program.items, mapping_idx, physical_qubits)
}

fn map_program_items_static(
    items: &[PreparedProgramItem],
    mapping_idx: &[usize],
    physical_qubits: &[Qubit],
) -> Vec<Operation> {
    let mut mapped = Vec::new();
    for item in items {
        match item {
            PreparedProgramItem::Segment(segment) => {
                for prep_op in &segment.operations {
                    let mapped_qubits: Vec<Qubit> = prep_op
                        .logical_qubits
                        .iter()
                        .map(|&logical| physical_qubits[mapping_idx[logical]])
                        .collect();
                    mapped.push(remap_operation_qubits(&prep_op.op, &mapped_qubits));
                }
            }
            PreparedProgramItem::Passthrough(op) => {
                let mapped_qubits: Vec<Qubit> = op
                    .logical_qubits
                    .iter()
                    .map(|&logical| physical_qubits[mapping_idx[logical]])
                    .collect();
                mapped.push(remap_operation_qubits(&op.op, &mapped_qubits));
            }
            PreparedProgramItem::IfElse(node) => {
                let condition = ConditionView::new(
                    physical_qubits[mapping_idx[node.condition_logical]],
                    node.condition.target,
                );
                let true_body =
                    map_program_items_static(&node.true_body.items, mapping_idx, physical_qubits);
                let false_body = node.false_body.as_ref().map(|body| {
                    map_program_items_static(&body.items, mapping_idx, physical_qubits)
                });
                mapped.push(build_if_else_operation(
                    condition,
                    true_body,
                    false_body,
                    node.label.clone(),
                ));
            }
            PreparedProgramItem::WhileLoop(node) => {
                let condition = ConditionView::new(
                    physical_qubits[mapping_idx[node.condition_logical]],
                    node.condition.target,
                );
                let body = map_program_items_static(&node.body.items, mapping_idx, physical_qubits);
                mapped.push(build_while_loop_operation(
                    condition,
                    body,
                    node.label.clone(),
                ));
            }
        }
    }
    mapped
}

pub(crate) fn build_if_else_operation(
    condition: ConditionView,
    true_body: Vec<Operation>,
    false_body: Option<Vec<Operation>>,
    label: Option<Box<str>>,
) -> Operation {
    let qubits = collect_control_flow_qubits(condition.qubit, &true_body, false_body.as_deref());
    Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
            condition, true_body, false_body,
        ))),
        qubits: qubits.into_iter().collect(),
        params: smallvec![],
        label,
    }
}

pub(crate) fn build_while_loop_operation(
    condition: ConditionView,
    body: Vec<Operation>,
    label: Option<Box<str>>,
) -> Operation {
    let qubits = collect_control_flow_qubits(condition.qubit, &body, None);
    Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
            condition, body,
        ))),
        qubits: qubits.into_iter().collect(),
        params: smallvec![],
        label,
    }
}

fn collect_control_flow_qubits(
    condition_qubit: Qubit,
    true_body: &[Operation],
    false_body: Option<&[Operation]>,
) -> Vec<Qubit> {
    let mut out = Vec::new();
    for op in true_body {
        out.extend(op.qubits.iter().copied());
    }
    if let Some(false_body) = false_body {
        for op in false_body {
            out.extend(op.qubits.iter().copied());
        }
    }
    out.push(condition_qubit);
    out
}

fn remap_operation_qubits(op: &Operation, mapped_qubits: &[Qubit]) -> Operation {
    let mut mapped = op.clone();
    mapped.qubits = mapped_qubits.iter().copied().collect();
    mapped
}
