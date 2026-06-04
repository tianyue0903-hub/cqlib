// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::compiler::CompilerError;
use rustworkx_core::petgraph::prelude::NodeIndex;

#[derive(Debug, Clone, Default)]
pub(crate) struct Layer {
    nodes: Vec<Option<[usize; 2]>>,
    occupied_node_indices: Vec<usize>,
    active: Vec<Option<(NodeIndex, usize)>>,
    total_score: f64,
}

impl Layer {
    pub(crate) fn new(node_count: usize, physical_count: usize) -> Self {
        Self {
            nodes: vec![None; node_count],
            occupied_node_indices: Vec::new(),
            active: vec![None; physical_count],
            total_score: 0.0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.occupied_node_indices.is_empty()
    }

    pub(crate) fn insert(
        &mut self,
        node: NodeIndex,
        qubits: [usize; 2],
        distances: &impl Fn(usize, usize) -> Result<f64, CompilerError>,
    ) -> Result<(), CompilerError> {
        self.ensure_node_capacity(node);
        if let Some(previous) = self.nodes[node.index()].replace(qubits) {
            self.total_score -= distances(previous[0], previous[1])?;
            self.remove_active_entry(node, previous);
        } else {
            self.insert_occupied_node_index(node.index());
        }
        self.total_score += distances(qubits[0], qubits[1])?;
        self.insert_active_entry(node, qubits);
        Ok(())
    }

    pub(crate) fn remove(
        &mut self,
        node: NodeIndex,
        distances: &impl Fn(usize, usize) -> Result<f64, CompilerError>,
    ) -> Result<(), CompilerError> {
        if node.index() >= self.nodes.len() {
            return Ok(());
        }
        if let Some(qubits) = self.nodes[node.index()].take() {
            self.total_score -= distances(qubits[0], qubits[1])?;
            self.remove_active_entry(node, qubits);
            self.remove_occupied_node_index(node.index());
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        for index in self.occupied_node_indices.drain(..) {
            self.nodes[index] = None;
        }
        self.active.fill(None);
        self.total_score = 0.0;
    }

    pub(crate) fn apply_swap(
        &mut self,
        swap: [usize; 2],
        distances: &impl Fn(usize, usize) -> Result<f64, CompilerError>,
    ) -> Result<(), CompilerError> {
        let affected = self.swap_affected_nodes(swap);
        let mut updates = Vec::with_capacity(2);
        for node in affected.into_iter().flatten() {
            let before = self.nodes[node.index()].ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "sabre layer active node {} has no node entry",
                    node.index()
                ))
            })?;
            let after = before.map(|physical| {
                if physical == swap[0] {
                    swap[1]
                } else if physical == swap[1] {
                    swap[0]
                } else {
                    physical
                }
            });
            let delta = distances(after[0], after[1])? - distances(before[0], before[1])?;
            updates.push((node, before, after, delta));
        }

        for (node, before, after, delta) in updates {
            self.total_score += delta;
            self.remove_active_entry(node, before);
            self.nodes[node.index()] = Some(after);
            self.insert_active_entry(node, after);
        }
        Ok(())
    }

    pub(crate) fn routable_node_on_index(
        &self,
        qubit: usize,
        adjacent: &impl Fn(usize, usize) -> bool,
    ) -> Option<NodeIndex> {
        let (node, other) = self.active.get(qubit).copied().flatten()?;
        adjacent(qubit, other).then_some(node)
    }

    pub(crate) fn active_indices_in_order<'a>(
        &'a self,
        order: &'a [usize],
    ) -> impl Iterator<Item = usize> + 'a {
        order
            .iter()
            .copied()
            .filter(|&index| self.active.get(index).is_some_and(Option::is_some))
    }

    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.occupied_node_indices
            .iter()
            .copied()
            .map(NodeIndex::new)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (NodeIndex, [usize; 2])> + '_ {
        self.occupied_node_indices
            .iter()
            .copied()
            .filter_map(|index| self.nodes[index].map(|qubits| (NodeIndex::new(index), qubits)))
    }

    pub(crate) fn total_score(&self) -> f64 {
        self.total_score
    }

    pub(crate) fn swap_delta_score(
        &self,
        swap: [usize; 2],
        distances: &impl Fn(usize, usize) -> Result<f64, CompilerError>,
    ) -> Result<f64, CompilerError> {
        let mut delta = 0.0;
        for node in self.swap_affected_nodes(swap).into_iter().flatten() {
            let before = self.nodes[node.index()].ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "sabre layer active node {} has no node entry",
                    node.index()
                ))
            })?;
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

    fn ensure_node_capacity(&mut self, node: NodeIndex) {
        if node.index() >= self.nodes.len() {
            self.nodes.resize(node.index() + 1, None);
        }
    }

    fn insert_occupied_node_index(&mut self, index: usize) {
        match self.occupied_node_indices.binary_search(&index) {
            Ok(_) => {}
            Err(position) => self.occupied_node_indices.insert(position, index),
        }
    }

    fn remove_occupied_node_index(&mut self, index: usize) {
        if let Ok(position) = self.occupied_node_indices.binary_search(&index) {
            self.occupied_node_indices.remove(position);
        }
    }

    fn insert_active_entry(&mut self, node: NodeIndex, [left, right]: [usize; 2]) {
        self.active[left] = Some((node, right));
        self.active[right] = Some((node, left));
    }

    fn remove_active_entry(&mut self, node: NodeIndex, [left, right]: [usize; 2]) {
        if self.active[left].is_some_and(|entry| entry.0 == node) {
            self.active[left] = None;
        }
        if self.active[right].is_some_and(|entry| entry.0 == node) {
            self.active[right] = None;
        }
    }

    fn swap_affected_nodes(&self, swap: [usize; 2]) -> [Option<NodeIndex>; 2] {
        let first = self.active[swap[0]].map(|entry| entry.0);
        let second = self.active[swap[1]]
            .map(|entry| entry.0)
            .filter(|node| Some(*node) != first);
        [first, second]
    }
}
