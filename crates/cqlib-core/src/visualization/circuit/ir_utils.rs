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

use crate::visualization::circuit::model::{
    VisualChildren, VisualCircuit, VisualControlFlowKind, VisualOpStyle, VisualOperation,
};

/// Expand control-flow operations into timeline markers and body operations.
///
/// This mirrors text rendering semantics: `If/Else/End` and `While/End` markers
/// become explicit operations in the resulting visual circuit.
pub(crate) fn flatten_control_flow_visual(visual: &VisualCircuit) -> VisualCircuit {
    let mut flat_ops = Vec::new();
    let mut next_cf_id = 0usize;
    flatten_ops_recursive(visual, &mut next_cf_id, &mut flat_ops);

    let mut next_free = vec![0usize; visual.num_qubits()];
    let mut num_columns = 0usize;
    for op in &mut flat_ops {
        let covered = effective_covered_lanes(op, visual.num_qubits());
        let column = covered
            .iter()
            .filter_map(|lane| next_free.get(*lane).copied())
            .max()
            .unwrap_or(0);
        op.column = column;
        let span = op.span_cols.max(1);
        for lane in covered {
            if let Some(slot) = next_free.get_mut(lane) {
                *slot = column + span;
            }
        }
        num_columns = num_columns.max(column + span);
    }

    VisualCircuit {
        qubits: visual.qubits.clone(),
        operations: flat_ops,
        num_columns,
    }
}

/// Reverse displayed qubit order for a pre-built visual circuit.
pub(crate) fn reverse_visual_lanes(mut visual: VisualCircuit) -> VisualCircuit {
    let n = visual.num_qubits();
    if n == 0 {
        return visual;
    }
    visual.qubits.reverse();
    for op in &mut visual.operations {
        for lane in &mut op.lanes {
            *lane = n - 1 - *lane;
        }
        for lane in &mut op.covered_lanes {
            *lane = n - 1 - *lane;
        }
    }
    visual
}

fn flatten_ops_recursive(
    visual: &VisualCircuit,
    next_cf_id: &mut usize,
    out: &mut Vec<VisualOperation>,
) {
    for op in &visual.operations {
        match op.style {
            VisualOpStyle::ControlFlow {
                kind: VisualControlFlowKind::IfElseBlock { condition, .. },
            } => {
                let cf_id = *next_cf_id;
                *next_cf_id += 1;
                let covered = effective_covered_lanes(op, visual.num_qubits());
                let if_label = format!("If q{}={}-{}", condition.qubit_id, condition.target, cf_id);
                out.push(make_control_marker(
                    if_label,
                    VisualControlFlowKind::IfStart,
                    covered.clone(),
                ));

                if let Some(VisualChildren::IfElse {
                    then_circuit,
                    else_circuit,
                }) = op.children.as_ref()
                {
                    flatten_ops_recursive(then_circuit, next_cf_id, out);
                    if let Some(else_body) = else_circuit {
                        out.push(make_control_marker(
                            format!("Else-{cf_id}"),
                            VisualControlFlowKind::ElseStart,
                            covered.clone(),
                        ));
                        flatten_ops_recursive(else_body, next_cf_id, out);
                    }
                }

                out.push(make_control_marker(
                    format!("End-{cf_id}"),
                    VisualControlFlowKind::End,
                    covered,
                ));
            }
            VisualOpStyle::ControlFlow {
                kind: VisualControlFlowKind::WhileBlock { condition },
            } => {
                let cf_id = *next_cf_id;
                *next_cf_id += 1;
                let covered = effective_covered_lanes(op, visual.num_qubits());
                let while_label = format!(
                    "While q{}={}-{}",
                    condition.qubit_id, condition.target, cf_id
                );
                out.push(make_control_marker(
                    while_label,
                    VisualControlFlowKind::WhileStart,
                    covered.clone(),
                ));
                if let Some(VisualChildren::While { body_circuit }) = op.children.as_ref() {
                    flatten_ops_recursive(body_circuit, next_cf_id, out);
                }
                out.push(make_control_marker(
                    format!("End-{cf_id}"),
                    VisualControlFlowKind::End,
                    covered,
                ));
            }
            _ => {
                let mut clone = op.clone();
                clone.children = None;
                clone.span_cols = 1;
                clone.column = 0;
                out.push(clone);
            }
        }
    }
}

fn make_control_marker(
    label: String,
    kind: VisualControlFlowKind,
    covered_lanes: Vec<usize>,
) -> VisualOperation {
    VisualOperation {
        column: 0,
        lanes: covered_lanes.clone(),
        covered_lanes,
        label,
        params: Vec::new(),
        style: VisualOpStyle::ControlFlow { kind },
        span_box: false,
        children: None,
        span_cols: 1,
    }
}

fn effective_covered_lanes(op: &VisualOperation, num_qubits: usize) -> Vec<usize> {
    if !op.covered_lanes.is_empty() {
        return op.covered_lanes.clone();
    }
    if !op.lanes.is_empty() {
        return op.lanes.clone();
    }
    (0..num_qubits).collect()
}