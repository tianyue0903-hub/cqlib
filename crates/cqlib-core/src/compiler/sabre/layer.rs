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

use crate::compiler::CompilerError;
use crate::device::PhysicalQubit;
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub(crate) struct Layer {
    nodes: BTreeMap<NodeIndex, [PhysicalQubit; 2]>,
    active: BTreeMap<PhysicalQubit, (NodeIndex, PhysicalQubit)>,
}

impl Layer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn insert(&mut self, node: NodeIndex, qubits: [PhysicalQubit; 2]) {
        if let Some(previous) = self.nodes.insert(node, qubits) {
            self.remove_active_entry(node, previous);
        }
        self.insert_active_entry(node, qubits);
    }

    pub(crate) fn remove(&mut self, node: NodeIndex) {
        if let Some(qubits) = self.nodes.remove(&node) {
            self.remove_active_entry(node, qubits);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.active.clear();
    }

    pub(crate) fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) {
        for node in self.swap_affected_nodes(swap).into_iter().flatten() {
            let before = self.nodes[&node];
            let after = before.map(|physical| {
                if physical == swap[0] {
                    swap[1]
                } else if physical == swap[1] {
                    swap[0]
                } else {
                    physical
                }
            });
            self.remove_active_entry(node, before);
            self.nodes.insert(node, after);
            self.insert_active_entry(node, after);
        }
    }

    pub(crate) fn routable_node_on_qubit(
        &self,
        qubit: PhysicalQubit,
        neighbors: &BTreeMap<PhysicalQubit, Vec<PhysicalQubit>>,
    ) -> Option<NodeIndex> {
        let (node, other) = self.active.get(&qubit).copied()?;
        neighbors
            .get(&qubit)
            .is_some_and(|items| items.contains(&other))
            .then_some(node)
    }

    pub(crate) fn active_qubits(&self) -> impl Iterator<Item = PhysicalQubit> + '_ {
        self.active.keys().copied()
    }

    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.nodes.keys().copied()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (NodeIndex, [PhysicalQubit; 2])> + '_ {
        self.nodes.iter().map(|(node, qubits)| (*node, *qubits))
    }

    pub(crate) fn total_score(
        &self,
        distances: &impl Fn(PhysicalQubit, PhysicalQubit) -> Result<f64, CompilerError>,
    ) -> Result<f64, CompilerError> {
        let mut total = 0.0;
        for qubits in self.nodes.values() {
            total += distances(qubits[0], qubits[1])?;
        }
        Ok(total)
    }

    pub(crate) fn swap_delta_score(
        &self,
        swap: [PhysicalQubit; 2],
        distances: &impl Fn(PhysicalQubit, PhysicalQubit) -> Result<f64, CompilerError>,
    ) -> Result<f64, CompilerError> {
        let mut delta = 0.0;
        for node in self.swap_affected_nodes(swap).into_iter().flatten() {
            let before = self.nodes[&node];
            let after = before.map(|physical| {
                if physical == swap[0] {
                    swap[1]
                } else if physical == swap[1] {
                    swap[0]
                } else {
                    physical
                }
            });
            delta += distances(after[0], after[1])? - distances(before[0], before[1])?;
        }
        Ok(delta)
    }

    fn insert_active_entry(&mut self, node: NodeIndex, [left, right]: [PhysicalQubit; 2]) {
        self.active.insert(left, (node, right));
        self.active.insert(right, (node, left));
    }

    fn remove_active_entry(&mut self, node: NodeIndex, [left, right]: [PhysicalQubit; 2]) {
        if self.active.get(&left).is_some_and(|entry| entry.0 == node) {
            self.active.remove(&left);
        }
        if self.active.get(&right).is_some_and(|entry| entry.0 == node) {
            self.active.remove(&right);
        }
    }

    fn swap_affected_nodes(&self, swap: [PhysicalQubit; 2]) -> [Option<NodeIndex>; 2] {
        let first = self.active.get(&swap[0]).map(|entry| entry.0);
        let second = self
            .active
            .get(&swap[1])
            .map(|entry| entry.0)
            .filter(|node| Some(*node) != first);
        [first, second]
    }
}
