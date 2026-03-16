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
use std::collections::HashMap;

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
        #[derive(Debug, Clone, Copy)]
        enum WireState {
            Unused,
            Open(usize),
            Blocked,
        }

        fn find_root(parent: &mut [usize], id: usize) -> usize {
            if parent[id] == id {
                return id;
            }
            let root = find_root(parent, parent[id]);
            parent[id] = root;
            root
        }

        fn merge_roots(
            parent: &mut [usize],
            components: &mut [Vec<usize>],
            left: usize,
            right: usize,
        ) -> usize {
            let left_root = find_root(parent, left);
            let right_root = find_root(parent, right);
            if left_root == right_root {
                return left_root;
            }
            let keep = left_root.min(right_root);
            let drop = left_root.max(right_root);
            let drained = std::mem::take(&mut components[drop]);
            components[keep].extend(drained);
            parent[drop] = keep;
            keep
        }

        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut parent: Vec<usize> = Vec::new();
        let mut wire_state: HashMap<usize, WireState> = HashMap::new();

        for node_id in self.topological_ids() {
            let node = &self.nodes[node_id];
            if node.erased {
                continue;
            }

            if node.op.gate == CanonicalGate::H {
                for &logical in &node.op.logical_qubits {
                    wire_state.insert(logical, WireState::Blocked);
                }
                continue;
            }

            let mut blocked = false;
            let mut roots = Vec::new();
            for &logical in &node.op.logical_qubits {
                match wire_state
                    .get(&logical)
                    .copied()
                    .unwrap_or(WireState::Unused)
                {
                    WireState::Unused => {}
                    WireState::Blocked => blocked = true,
                    WireState::Open(component_id) => {
                        let root = find_root(&mut parent, component_id);
                        if !roots.contains(&root) {
                            roots.push(root);
                        }
                    }
                }
            }

            let component_id = if blocked || roots.is_empty() {
                let id = components.len();
                components.push(Vec::new());
                parent.push(id);
                id
            } else {
                let mut root = roots[0];
                for &other in &roots[1..] {
                    root = merge_roots(&mut parent, &mut components, root, other);
                }
                root
            };

            let root = find_root(&mut parent, component_id);
            components[root].push(node_id);
            for &logical in &node.op.logical_qubits {
                wire_state.insert(logical, WireState::Open(root));
            }
        }

        let mut out = Vec::new();
        for (component_id, mut component) in components.into_iter().enumerate() {
            if find_root(&mut parent, component_id) != component_id || component.is_empty() {
                continue;
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

    #[test]
    fn test_h_free_components_do_not_cross_h_boundary_on_other_wire() {
        let ops = vec![
            CanonicalOp::rz(1, -std::f64::consts::FRAC_PI_2),
            CanonicalOp::cx(0, 1),
            CanonicalOp::rz(1, std::f64::consts::FRAC_PI_2),
            CanonicalOp::h(0),
            CanonicalOp::cx(1, 0),
            CanonicalOp::h(0),
            CanonicalOp::cx(0, 1),
            CanonicalOp::cx(1, 0),
            CanonicalOp::cx(0, 1),
            CanonicalOp::rz(0, 0.125),
        ];
        let dag = SegmentDag::from_ops(&ops);
        let components = dag.h_free_components();
        assert_eq!(components, vec![vec![0, 1, 2], vec![4], vec![6, 7, 8, 9]]);
    }
}
