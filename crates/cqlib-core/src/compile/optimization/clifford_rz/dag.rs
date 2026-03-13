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

use super::canonical::{CanonicalGate, CanonicalOp};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireLink {
    pub(crate) node_id: usize,
    pub(crate) wire: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DagNode {
    pub(crate) id: usize,
    pub(crate) op: CanonicalOp,
    pub(crate) predecessors: SmallVec<[Option<WireLink>; 2]>,
    pub(crate) successors: SmallVec<[Option<WireLink>; 2]>,
    pub(crate) erased: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentDag {
    nodes: Vec<DagNode>,
}

impl SegmentDag {
    pub(crate) fn from_ops(ops: &[CanonicalOp]) -> Self {
        let mut nodes = Vec::with_capacity(ops.len());
        let mut current_on_qubit: HashMap<usize, WireLink> = HashMap::new();

        for (id, op) in ops.iter().cloned().enumerate() {
            let arity = op.logical_qubits.len();
            let mut predecessors = SmallVec::<[Option<WireLink>; 2]>::with_capacity(arity);
            let mut successors = SmallVec::<[Option<WireLink>; 2]>::with_capacity(arity);
            for _ in 0..arity {
                predecessors.push(None);
                successors.push(None);
            }

            nodes.push(DagNode {
                id,
                op: op.clone(),
                predecessors,
                successors,
                erased: false,
            });

            for (wire, &logical) in op.logical_qubits.iter().enumerate() {
                if let Some(prev_link) = current_on_qubit.get(&logical).copied() {
                    nodes[prev_link.node_id].successors[prev_link.wire] =
                        Some(WireLink { node_id: id, wire });
                    nodes[id].predecessors[wire] = Some(prev_link);
                }
                current_on_qubit.insert(logical, WireLink { node_id: id, wire });
            }
        }

        Self { nodes }
    }

    pub(crate) fn node(&self, node_id: usize) -> &DagNode {
        &self.nodes[node_id]
    }

    pub(crate) fn node_mut(&mut self, node_id: usize) -> &mut DagNode {
        &mut self.nodes[node_id]
    }

    pub(crate) fn topological_ids(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|node| !node.erased)
            .map(|node| node.id)
            .collect()
    }

    pub(crate) fn to_ops(&self) -> Vec<CanonicalOp> {
        self.topological_ids()
            .into_iter()
            .map(|node_id| self.nodes[node_id].op.clone())
            .collect()
    }

    pub(crate) fn erase_node(&mut self, node_id: usize) {
        if self.nodes[node_id].erased {
            return;
        }

        let predecessor_links = self.nodes[node_id].predecessors.clone();
        let successor_links = self.nodes[node_id].successors.clone();
        self.nodes[node_id].erased = true;

        for wire in 0..self.nodes[node_id].op.logical_qubits.len() {
            let pred = predecessor_links.get(wire).copied().flatten();
            let succ = successor_links.get(wire).copied().flatten();
            if let Some(pred_link) = pred {
                self.nodes[pred_link.node_id].successors[pred_link.wire] = succ;
            }
            if let Some(succ_link) = succ {
                self.nodes[succ_link.node_id].predecessors[succ_link.wire] = pred;
            }
            self.nodes[node_id].predecessors[wire] = None;
            self.nodes[node_id].successors[wire] = None;
        }
    }

    pub(crate) fn is_exposed_pair(&self, first: usize, second: usize) -> bool {
        let node_a = &self.nodes[first];
        let node_b = &self.nodes[second];
        if node_a.erased || node_b.erased {
            return false;
        }
        if node_a.op.gate != node_b.op.gate || node_a.op.logical_qubits != node_b.op.logical_qubits
        {
            return false;
        }
        if node_a.op.logical_qubits.len() != node_b.op.logical_qubits.len() {
            return false;
        }

        for wire in 0..node_a.op.logical_qubits.len() {
            let succ = node_a.successors.get(wire).copied().flatten();
            let pred = node_b.predecessors.get(wire).copied().flatten();
            if succ
                != Some(WireLink {
                    node_id: second,
                    wire,
                })
                || pred
                    != Some(WireLink {
                        node_id: first,
                        wire,
                    })
            {
                return false;
            }
        }
        true
    }

    pub(crate) fn h_free_components(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut visited = HashSet::new();

        for start in self.topological_ids() {
            if visited.contains(&start) {
                continue;
            }
            let node = &self.nodes[start];
            if node.op.gate == CanonicalGate::H {
                continue;
            }

            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            visited.insert(start);

            while let Some(current) = queue.pop_front() {
                component.push(current);
                let current_node = &self.nodes[current];
                for link in current_node
                    .predecessors
                    .iter()
                    .chain(current_node.successors.iter())
                {
                    let Some(link) = link else {
                        continue;
                    };
                    let next = link.node_id;
                    if visited.contains(&next) {
                        continue;
                    }
                    let next_node = &self.nodes[next];
                    if next_node.erased || next_node.op.gate == CanonicalGate::H {
                        continue;
                    }
                    visited.insert(next);
                    queue.push_back(next);
                }
            }

            component.sort_unstable();
            out.push(component);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_dag_roundtrip_and_erase() {
        let ops = vec![CanonicalOp::h(0), CanonicalOp::cx(0, 1), CanonicalOp::x(0)];
        let mut dag = SegmentDag::from_ops(&ops);
        assert_eq!(dag.to_ops(), ops);

        dag.erase_node(1);
        assert_eq!(dag.to_ops(), vec![CanonicalOp::h(0), CanonicalOp::x(0)]);
    }

    #[test]
    fn test_h_free_components_skip_h_boundaries() {
        let ops = vec![
            CanonicalOp::x(0),
            CanonicalOp::h(1),
            CanonicalOp::rz(0, 0.2),
            CanonicalOp::cx(0, 2),
        ];
        let dag = SegmentDag::from_ops(&ops);
        let components = dag.h_free_components();
        assert_eq!(components, vec![vec![0, 2, 3]]);
    }
}
