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

//! CFG block-level summaries for block-local transform decisions.
//!
//! This module derives per-block metadata from [`crate::circuit::CircuitCFG`],
//! including operation mix, touched qubits, and block suitability flags for
//! downstream transform stages (rewrite, routing, scheduling, resynthesis).
//!
//! It is intended as a cheap structural view for orchestration decisions, not as
//! a replacement for detailed pass-specific legality checks.

use crate::circuit::cfg::Terminator;
use crate::circuit::{CircuitCFG, Directive, Instruction, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::collections::{BTreeMap, BTreeSet};

/// CFG-level summary for block-local compiler transforms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockSummary {
    entry_block: Option<NodeIndex>,
    blocks: BTreeMap<NodeIndex, BlockSummaryEntry>,
    block_order: Vec<NodeIndex>,
    num_branch_blocks: usize,
    num_loop_condition_blocks: usize,
}

impl BlockSummary {
    /// Builds block summaries from the current control-flow graph.
    pub fn from_cfg(cfg: &CircuitCFG) -> Self {
        let mut blocks = BTreeMap::new();
        let mut block_order = Vec::new();
        let mut num_branch_blocks = 0;
        let mut num_loop_condition_blocks = 0;

        for (block_id, block) in cfg.blocks() {
            let entry = BlockSummaryEntry::from_block(block);

            if entry.has_control_flow_terminator {
                num_branch_blocks += 1;
            }
            if entry.is_loop_condition_block {
                num_loop_condition_blocks += 1;
            }

            block_order.push(block_id);
            blocks.insert(block_id, entry);
        }

        Self {
            entry_block: cfg.entry_block(),
            blocks,
            block_order,
            num_branch_blocks,
            num_loop_condition_blocks,
        }
    }

    /// Returns the entry block of the summarized CFG.
    pub fn entry_block(&self) -> Option<NodeIndex> {
        self.entry_block
    }

    /// Returns the total number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of blocks terminated by a conditional branch.
    pub fn num_branch_blocks(&self) -> usize {
        self.num_branch_blocks
    }

    /// Returns the number of loop-condition blocks.
    pub fn num_loop_condition_blocks(&self) -> usize {
        self.num_loop_condition_blocks
    }

    /// Returns a block summary entry by block id.
    pub fn get(&self, block_id: NodeIndex) -> Option<&BlockSummaryEntry> {
        self.blocks.get(&block_id)
    }

    /// Returns block summaries in stable block iteration order.
    pub fn entries(&self) -> impl Iterator<Item = (NodeIndex, &BlockSummaryEntry)> {
        self.block_order
            .iter()
            .copied()
            .filter_map(|block_id| self.blocks.get(&block_id).map(|entry| (block_id, entry)))
    }
}

/// Summary of a single basic block and its suitability for later transforms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockSummaryEntry {
    pub label: Option<String>,
    pub op_count: usize,
    pub single_qubit_ops: usize,
    pub two_qubit_ops: usize,
    pub multi_qubit_ops: usize,
    pub measure_ops: usize,
    pub reset_ops: usize,
    pub barrier_ops: usize,
    pub delay_ops: usize,
    pub parameterized_ops: usize,
    pub touched_qubits: BTreeSet<Qubit>,
    pub has_control_flow_terminator: bool,
    pub has_non_unitary_ops: bool,
    pub is_empty: bool,
    pub is_pure_quantum_block: bool,
    pub is_rewrite_candidate: bool,
    pub is_resynthesis_candidate: bool,
    pub is_routing_candidate: bool,
    pub is_schedule_candidate: bool,
    pub is_loop_condition_block: bool,
}

impl BlockSummaryEntry {
    fn from_block(block: &crate::circuit::cfg::BasicBlock) -> Self {
        let mut entry = Self {
            label: block.label().map(ToOwned::to_owned),
            has_control_flow_terminator: matches!(block.terminator, Some(Terminator::Branch(_))),
            is_loop_condition_block: block
                .label()
                .is_some_and(|label| label.starts_with("while_cond_")),
            ..Self::default()
        };

        for operation in &block.operations {
            entry.op_count += 1;
            entry
                .touched_qubits
                .extend(operation.qubits.iter().copied());

            match operation.qubits.len() {
                0 => {}
                1 => entry.single_qubit_ops += 1,
                2 => entry.two_qubit_ops += 1,
                _ => entry.multi_qubit_ops += 1,
            }

            if !operation.params.is_empty() {
                entry.parameterized_ops += 1;
            }

            match &operation.instruction {
                Instruction::Directive(directive) => {
                    entry.has_non_unitary_ops = true;
                    match directive {
                        Directive::Barrier => entry.barrier_ops += 1,
                        Directive::Measure => entry.measure_ops += 1,
                        Directive::Reset => entry.reset_ops += 1,
                    }
                }
                Instruction::Delay => {
                    entry.has_non_unitary_ops = true;
                    entry.delay_ops += 1;
                }
                Instruction::ControlFlowGate(_) => {
                    entry.has_non_unitary_ops = true;
                }
                Instruction::Standard(_)
                | Instruction::McGate(_)
                | Instruction::UnitaryGate(_)
                | Instruction::CircuitGate(_) => {}
            }
        }

        entry.is_empty = entry.op_count == 0;
        entry.is_pure_quantum_block =
            !entry.is_empty && !entry.has_non_unitary_ops && !entry.has_control_flow_terminator;
        entry.is_rewrite_candidate = entry.is_pure_quantum_block;
        entry.is_resynthesis_candidate = entry.is_pure_quantum_block;
        entry.is_routing_candidate =
            entry.is_pure_quantum_block && (entry.two_qubit_ops > 0 || entry.multi_qubit_ops > 0);
        entry.is_schedule_candidate =
            !entry.is_empty && !entry.has_non_unitary_ops && !entry.has_control_flow_terminator;

        entry
    }
}

#[cfg(test)]
mod tests {
    use super::BlockSummary;
    use crate::circuit::{Circuit, ConditionView, Qubit};

    #[test]
    fn block_summary_for_linear_circuit_has_single_optimizable_block() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let summary = BlockSummary::from_cfg(&cfg);

        assert_eq!(summary.num_blocks(), 1);
        assert_eq!(summary.num_branch_blocks(), 0);
        assert_eq!(summary.num_loop_condition_blocks(), 0);

        let (_, entry) = summary.entries().next().unwrap();
        assert_eq!(entry.op_count, 2);
        assert_eq!(entry.single_qubit_ops, 1);
        assert_eq!(entry.two_qubit_ops, 1);
        assert!(entry.is_pure_quantum_block);
        assert!(entry.is_rewrite_candidate);
        assert!(entry.is_resynthesis_candidate);
        assert!(entry.is_routing_candidate);
        assert!(entry.is_schedule_candidate);
    }

    #[test]
    fn block_summary_marks_if_else_condition_and_merge_blocks() {
        let mut circuit = Circuit::new(2);
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![crate::circuit::Operation {
                    instruction: crate::circuit::StandardGate::X.into(),
                    qubits: smallvec::smallvec![Qubit::new(1)],
                    params: smallvec::smallvec![],
                    label: None,
                }],
                None,
            )
            .unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let summary = BlockSummary::from_cfg(&cfg);

        assert_eq!(summary.num_branch_blocks(), 1);
        assert_eq!(summary.num_loop_condition_blocks(), 0);

        let mut saw_condition = false;
        let mut saw_true_block = false;
        let mut saw_merge = false;

        for (_, entry) in summary.entries() {
            match entry.label.as_deref() {
                Some("entry") => {
                    assert!(entry.has_control_flow_terminator);
                    assert!(entry.is_empty);
                    saw_condition = true;
                }
                Some(label) if label.starts_with("if_true_") => {
                    assert_eq!(entry.op_count, 1);
                    assert!(entry.is_pure_quantum_block);
                    saw_true_block = true;
                }
                Some(label) if label.starts_with("if_merge_") => {
                    assert!(entry.is_empty);
                    assert!(!entry.is_rewrite_candidate);
                    saw_merge = true;
                }
                _ => {}
            }
        }

        assert!(saw_condition);
        assert!(saw_true_block);
        assert!(saw_merge);
    }

    #[test]
    fn block_summary_marks_loop_condition_and_non_unitary_blocks() {
        let mut circuit = Circuit::new(2);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .while_loop(
                ConditionView::new(Qubit::new(0), 1),
                vec![crate::circuit::Operation {
                    instruction: crate::circuit::StandardGate::X.into(),
                    qubits: smallvec::smallvec![Qubit::new(1)],
                    params: smallvec::smallvec![],
                    label: None,
                }],
            )
            .unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let summary = BlockSummary::from_cfg(&cfg);

        assert_eq!(summary.num_branch_blocks(), 1);
        assert_eq!(summary.num_loop_condition_blocks(), 1);

        let entry_block = summary.get(summary.entry_block().unwrap()).unwrap();
        assert_eq!(entry_block.measure_ops, 1);
        assert!(entry_block.has_non_unitary_ops);
        assert!(!entry_block.is_resynthesis_candidate);

        let loop_cond = summary
            .entries()
            .find(|(_, entry)| entry.is_loop_condition_block)
            .map(|(_, entry)| entry)
            .unwrap();
        assert!(loop_cond.has_control_flow_terminator);
        assert!(loop_cond.is_empty);
        assert!(!loop_cond.is_schedule_candidate);
    }
}
