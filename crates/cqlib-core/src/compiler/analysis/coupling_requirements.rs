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

//! Logical two-qubit coupling demand extracted from the CFG.
//!
//! This module summarizes directed logical interaction demand for two-qubit
//! operations. It tracks both global counts and block-local demand so routing and
//! placement stages can prioritize the most constrained couplings.
//!
//! Multi-qubit operations (`arity > 2`) are counted separately but not expanded
//! into pairwise requirements because decomposition policy is workflow-dependent.

use crate::circuit::{CircuitCFG, Directive, Instruction, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::collections::{BTreeMap, BTreeSet};

/// Directed logical interaction key required by the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CouplingKey {
    pub source: Qubit,
    pub target: Qubit,
}

impl CouplingKey {
    pub const fn new(source: Qubit, target: Qubit) -> Self {
        Self { source, target }
    }
}

/// Aggregated usage information for one directed logical interaction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CouplingRequirement {
    pub occurrences: usize,
    pub first_op_index: Option<usize>,
    pub last_op_index: Option<usize>,
    pub blocks: BTreeSet<NodeIndex>,
}

impl CouplingRequirement {
    fn record(&mut self, block_id: NodeIndex, op_index: usize) {
        self.occurrences += 1;
        self.first_op_index = Some(
            self.first_op_index
                .map_or(op_index, |first| first.min(op_index)),
        );
        self.last_op_index = Some(
            self.last_op_index
                .map_or(op_index, |last| last.max(op_index)),
        );
        self.blocks.insert(block_id);
    }
}

/// Global and block-local logical coupling demand extracted from the CFG.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CouplingRequirements {
    pairs: BTreeMap<CouplingKey, CouplingRequirement>,
    block_pairs: BTreeMap<NodeIndex, BTreeMap<CouplingKey, CouplingRequirement>>,
    total_two_qubit_ops: usize,
    total_multi_qubit_ops: usize,
}

impl CouplingRequirements {
    /// Builds coupling requirements from the current control-flow graph.
    ///
    /// Only 2-qubit unitary-like operations contribute explicit coupling pairs.
    /// Operations with arity greater than 2 only contribute to the aggregate
    /// multi-qubit counter because their decomposition is policy-dependent.
    pub fn from_cfg(cfg: &CircuitCFG) -> Self {
        let mut requirements = Self::default();
        let mut global_op_index = 0usize;

        for (block_id, block) in cfg.blocks() {
            for operation in &block.operations {
                let arity = operation.qubits.len();
                let is_ignored = matches!(
                    operation.instruction,
                    Instruction::Directive(
                        Directive::Barrier | Directive::Measure | Directive::Reset
                    ) | Instruction::Delay
                        | Instruction::ControlFlowGate(_)
                );

                if is_ignored {
                    global_op_index += 1;
                    continue;
                }

                match arity {
                    2 => {
                        requirements.total_two_qubit_ops += 1;
                        let key = CouplingKey::new(operation.qubits[0], operation.qubits[1]);

                        requirements
                            .pairs
                            .entry(key)
                            .or_default()
                            .record(block_id, global_op_index);

                        requirements
                            .block_pairs
                            .entry(block_id)
                            .or_default()
                            .entry(key)
                            .or_default()
                            .record(block_id, global_op_index);
                    }
                    3.. => {
                        requirements.total_multi_qubit_ops += 1;
                    }
                    _ => {}
                }

                global_op_index += 1;
            }
        }

        requirements
    }

    /// Returns the total number of two-qubit operations that contribute pairs.
    pub fn total_two_qubit_ops(&self) -> usize {
        self.total_two_qubit_ops
    }

    /// Returns the total number of multi-qubit operations ignored for pair expansion.
    pub fn total_multi_qubit_ops(&self) -> usize {
        self.total_multi_qubit_ops
    }

    /// Returns the aggregated requirement for a directed pair, if present.
    pub fn get(&self, key: CouplingKey) -> Option<&CouplingRequirement> {
        self.pairs.get(&key)
    }

    /// Returns all global pair requirements in stable key order.
    pub fn entries(&self) -> impl Iterator<Item = (CouplingKey, &CouplingRequirement)> {
        self.pairs
            .iter()
            .map(|(&key, requirement)| (key, requirement))
    }

    /// Returns all pair requirements for a block, if any.
    pub fn block_entries(
        &self,
        block_id: NodeIndex,
    ) -> Option<impl Iterator<Item = (CouplingKey, &CouplingRequirement)>> {
        self.block_pairs
            .get(&block_id)
            .map(|pairs| pairs.iter().map(|(&key, requirement)| (key, requirement)))
    }
}

#[cfg(test)]
mod tests {
    use super::{CouplingKey, CouplingRequirements};
    use crate::circuit::{Circuit, ConditionView, Operation, Qubit, StandardGate};
    use smallvec::smallvec;

    #[test]
    fn coupling_requirements_track_directed_pairs_in_linear_circuit() {
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let requirements = CouplingRequirements::from_cfg(&cfg);

        assert_eq!(requirements.total_two_qubit_ops(), 3);
        assert_eq!(requirements.total_multi_qubit_ops(), 0);

        let forward = requirements
            .get(CouplingKey::new(Qubit::new(0), Qubit::new(1)))
            .unwrap();
        assert_eq!(forward.occurrences, 2);
        assert_eq!(forward.first_op_index, Some(0));
        assert_eq!(forward.last_op_index, Some(1));

        let reverse = requirements
            .get(CouplingKey::new(Qubit::new(1), Qubit::new(0)))
            .unwrap();
        assert_eq!(reverse.occurrences, 1);
        assert_eq!(reverse.first_op_index, Some(2));
        assert_eq!(reverse.last_op_index, Some(2));
    }

    #[test]
    fn coupling_requirements_ignore_single_qubit_and_non_unitary_ops() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
        circuit.measure(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let requirements = CouplingRequirements::from_cfg(&cfg);

        assert_eq!(requirements.total_two_qubit_ops(), 1);
        assert_eq!(requirements.entries().count(), 1);
    }

    #[test]
    fn coupling_requirements_preserve_block_local_pairs_for_control_flow() {
        let mut circuit = Circuit::new(3);
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: StandardGate::CX.into(),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                }],
                Some(vec![Operation {
                    instruction: StandardGate::CZ.into(),
                    qubits: smallvec![Qubit::new(2), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let requirements = CouplingRequirements::from_cfg(&cfg);

        assert_eq!(requirements.total_two_qubit_ops(), 2);

        let true_key = CouplingKey::new(Qubit::new(1), Qubit::new(2));
        let false_key = CouplingKey::new(Qubit::new(2), Qubit::new(1));

        assert_eq!(requirements.get(true_key).unwrap().occurrences, 1);
        assert_eq!(requirements.get(false_key).unwrap().occurrences, 1);

        let blocks_with_true: Vec<_> = requirements
            .get(true_key)
            .unwrap()
            .blocks
            .iter()
            .copied()
            .collect();
        let blocks_with_false: Vec<_> = requirements
            .get(false_key)
            .unwrap()
            .blocks
            .iter()
            .copied()
            .collect();

        assert_eq!(blocks_with_true.len(), 1);
        assert_eq!(blocks_with_false.len(), 1);
        assert_ne!(blocks_with_true[0], blocks_with_false[0]);
    }

    #[test]
    fn coupling_requirements_count_multi_qubit_ops_without_expanding_pairs() {
        let mut circuit = Circuit::new(3);
        circuit
            .multi_control(
                StandardGate::X,
                vec![Qubit::new(0), Qubit::new(1)],
                vec![Qubit::new(2)],
                [],
            )
            .unwrap();

        let cfg = crate::circuit::CircuitCFG::from_circuit(&circuit).unwrap();
        let requirements = CouplingRequirements::from_cfg(&cfg);

        assert_eq!(requirements.total_two_qubit_ops(), 0);
        assert_eq!(requirements.total_multi_qubit_ops(), 1);
        assert_eq!(requirements.entries().count(), 0);
    }
}
