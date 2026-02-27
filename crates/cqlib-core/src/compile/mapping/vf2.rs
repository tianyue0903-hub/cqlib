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

//! Structural VF2 mapping pass.

use super::{
    FidelityMap, PreparedCircuit, TopologyAdapter, build_output_circuit, map_operation_qubits,
    normalize_index_pair, preprocess_circuit,
};
use crate::circuit::{Circuit, Qubit};
use crate::compile::error::CompileError;
use crate::device::Topology;
use rustworkx_core::petgraph::algo::isomorphism::{
    is_isomorphic_subgraph_matching, subgraph_isomorphisms_iter,
};
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct InteractionGraph {
    active_nodes: Vec<usize>,
    adjacency: Vec<HashSet<usize>>,
    edges: Vec<(usize, usize)>,
}

/// VF2 mapping implementation.
#[derive(Debug, Clone)]
pub struct Vf2Mapping {
    /// Logical -> physical mapping after the latest `execute` call.
    pub logic2phy: Vec<Qubit>,
    /// Physical -> logical mapping after the latest `execute` call.
    pub phy2logic: HashMap<Qubit, usize>,
    topology: TopologyAdapter,
    topology_graph: UnGraph<usize, ()>,
}

impl Vf2Mapping {
    /// Creates a VF2 mapper using topology and optional fidelity map.
    ///
    /// Note: fidelity values are only validated here for interface consistency.
    /// The VF2 matching itself is purely structural and does not score by fidelity.
    pub fn new(topology: Topology, fidelity_map: Option<FidelityMap>) -> Result<Self, CompileError> {
        let topology = TopologyAdapter::new(&topology, fidelity_map.as_ref())?;
        Ok(Self::from_adapter(topology))
    }

    pub(crate) fn from_adapter(topology: TopologyAdapter) -> Self {
        let topology_graph = build_topology_graph(&topology);
        Self {
            logic2phy: Vec::new(),
            phy2logic: HashMap::new(),
            topology,
            topology_graph,
        }
    }

    /// Returns whether a subgraph-isomorphic mapping exists.
    pub fn is_subgraph_isomorphic(&self, circuit: &Circuit) -> Result<bool, CompileError> {
        let prepared = preprocess_circuit(circuit)?;
        let mapping = self.solve_prepared_initial_layout(&prepared)?;
        Ok(mapping.is_some())
    }

    /// Finds a logical->physical initial layout without inserting routing gates.
    pub fn find_initial_layout(&self, circuit: &Circuit) -> Result<Option<Vec<Qubit>>, CompileError> {
        let prepared = preprocess_circuit(circuit)?;
        let mapping = self.solve_prepared_initial_layout(&prepared)?;
        Ok(mapping.map(|idx_map| {
            idx_map
                .into_iter()
                .map(|idx| self.topology.physical_qubits[idx])
                .collect()
        }))
    }

    pub(crate) fn solve_prepared_initial_layout(
        &self,
        prepared: &PreparedCircuit,
    ) -> Result<Option<Vec<usize>>, CompileError> {
        self.find_best_mapping(prepared)
    }

    /// Executes VF2 mapping and returns mapped circuit.
    pub fn execute(&mut self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let prepared = preprocess_circuit(circuit)?;
        let mapping_idx = self
            .find_best_mapping(&prepared)?
            .ok_or(CompileError::Vf2NoMapping)?;

        let logic2phy: Vec<Qubit> = mapping_idx
            .iter()
            .map(|&idx| self.topology.physical_qubits[idx])
            .collect();

        let mut phy2logic = HashMap::new();
        for (logical, &physical) in logic2phy.iter().enumerate() {
            phy2logic.insert(physical, logical);
        }

        let mut mapped_ops = Vec::with_capacity(prepared.operations.len());
        for prep_op in &prepared.operations {
            let mapped_qubits: Vec<Qubit> = prep_op
                .logical_qubits
                .iter()
                .map(|&l| self.topology.physical_qubits[mapping_idx[l]])
                .collect();
            mapped_ops.push(map_operation_qubits(&prep_op.op, &mapped_qubits));
        }

        self.logic2phy = logic2phy;
        self.phy2logic = phy2logic;

        build_output_circuit(&mapped_ops, &prepared.parameters)
    }

    fn find_best_mapping(
        &self,
        prepared: &PreparedCircuit,
    ) -> Result<Option<Vec<usize>>, CompileError> {
        let logical_width = prepared.logical_qubits.len();
        let physical_width = self.topology.num_qubits();

        if logical_width > physical_width {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: physical_width,
            });
        }

        let interaction = self.build_interaction_graph(prepared);

        // No 2q interactions: assign deterministically by physical degree.
        if interaction.active_nodes.is_empty() {
            return Ok(Some(self.assign_isolates(logical_width, &HashMap::new())?));
        }

        let (pattern_graph, local_to_logical) = self.build_pattern_graph(&interaction);

        if !is_isomorphic_subgraph_matching(
            &pattern_graph,
            &self.topology_graph,
            |_, _| true,
            |_, _| true,
        ) {
            return Ok(None);
        }

        let mut node_match = |_: &usize, _: &usize| true;
        let mut edge_match = |_: &(), _: &()| true;
        let pattern_ref = &pattern_graph;
        let topology_ref = &self.topology_graph;
        let Some(matches) = subgraph_isomorphisms_iter(
            &pattern_ref,
            &topology_ref,
            &mut node_match,
            &mut edge_match,
        )
        else {
            return Ok(None);
        };

        let mut best_partial: Option<HashMap<usize, usize>> = None;
        let mut best_score = i64::MIN;

        for mapping in matches {
            if mapping.len() != local_to_logical.len() {
                continue;
            }

            let mut partial = HashMap::with_capacity(mapping.len());
            let mut valid = true;
            for (local_idx, topo_node_idx) in mapping.into_iter().enumerate() {
                let Some(&physical_idx) = self.topology_graph.node_weight(NodeIndex::new(topo_node_idx))
                else {
                    valid = false;
                    break;
                };
                let logical_idx = local_to_logical[local_idx];
                partial.insert(logical_idx, physical_idx);
            }

            if !valid {
                continue;
            }

            let score = self.score_partial_mapping(&interaction, &partial);
            let update = if score > best_score {
                true
            } else if score == best_score {
                match &best_partial {
                    None => true,
                    Some(prev) => self.lexicographically_better(&partial, prev),
                }
            } else {
                false
            };

            if update {
                best_score = score;
                best_partial = Some(partial);
            }
        }

        let Some(best_partial) = best_partial else {
            return Ok(None);
        };

        let full_mapping = self.assign_isolates(logical_width, &best_partial)?;
        Ok(Some(full_mapping))
    }

    fn build_pattern_graph(
        &self,
        interaction: &InteractionGraph,
    ) -> (UnGraph<usize, ()>, Vec<usize>) {
        let mut graph = UnGraph::<usize, ()>::new_undirected();
        let mut local_to_logical = interaction.active_nodes.clone();
        local_to_logical.sort_unstable();

        let mut logical_to_local = HashMap::with_capacity(local_to_logical.len());
        let mut local_nodes = Vec::with_capacity(local_to_logical.len());
        for (local_idx, &logical_idx) in local_to_logical.iter().enumerate() {
            local_nodes.push(graph.add_node(logical_idx));
            logical_to_local.insert(logical_idx, local_idx);
        }

        for &(u, v) in &interaction.edges {
            let Some(&lu) = logical_to_local.get(&u) else {
                continue;
            };
            let Some(&lv) = logical_to_local.get(&v) else {
                continue;
            };
            graph.add_edge(local_nodes[lu], local_nodes[lv], ());
        }

        (graph, local_to_logical)
    }

    fn score_partial_mapping(
        &self,
        interaction: &InteractionGraph,
        mapping: &HashMap<usize, usize>,
    ) -> i64 {
        let mut score = 0i64;
        for (&logical, &physical) in mapping {
            let logical_degree = interaction.adjacency[logical].len() as i64;
            let physical_degree = self.topology.neighbors[physical].len() as i64;
            score += logical_degree * physical_degree;
        }
        score
    }

    fn assign_isolates(
        &self,
        logical_width: usize,
        seeded_mapping: &HashMap<usize, usize>,
    ) -> Result<Vec<usize>, CompileError> {
        let mut result = vec![usize::MAX; logical_width];
        let mut used = vec![false; self.topology.num_qubits()];

        for (&logical, &physical) in seeded_mapping {
            if logical < logical_width {
                result[logical] = physical;
                used[physical] = true;
            }
        }

        let mut remaining_physical: Vec<usize> = (0..self.topology.num_qubits())
            .filter(|idx| !used[*idx])
            .collect();

        remaining_physical.sort_by(|a, b| {
            let deg_a = self.topology.neighbors[*a].len();
            let deg_b = self.topology.neighbors[*b].len();
            deg_b.cmp(&deg_a).then_with(|| {
                self.topology.physical_qubits[*a]
                    .id()
                    .cmp(&self.topology.physical_qubits[*b].id())
            })
        });

        let mut cursor = 0usize;
        for logical in 0..logical_width {
            if result[logical] != usize::MAX {
                continue;
            }
            let Some(&physical) = remaining_physical.get(cursor) else {
                return Err(CompileError::TopologyTooSmall {
                    required: logical_width,
                    available: self.topology.num_qubits(),
                });
            };
            cursor += 1;
            result[logical] = physical;
        }

        Ok(result)
    }

    fn lexicographically_better(
        &self,
        current: &HashMap<usize, usize>,
        best: &HashMap<usize, usize>,
    ) -> bool {
        let mut curr_pairs: Vec<(usize, u32)> = current
            .iter()
            .map(|(&l, &p)| (l, self.topology.physical_qubits[p].id()))
            .collect();
        let mut best_pairs: Vec<(usize, u32)> = best
            .iter()
            .map(|(&l, &p)| (l, self.topology.physical_qubits[p].id()))
            .collect();
        curr_pairs.sort_unstable_by_key(|(l, _)| *l);
        best_pairs.sort_unstable_by_key(|(l, _)| *l);

        for ((_, a), (_, b)) in curr_pairs.iter().zip(best_pairs.iter()) {
            match a.cmp(b) {
                Ordering::Less => return true,
                Ordering::Greater => return false,
                Ordering::Equal => {}
            }
        }
        false
    }

    fn build_interaction_graph(&self, prepared: &PreparedCircuit) -> InteractionGraph {
        let logical_width = prepared.logical_qubits.len();
        let mut adjacency = vec![HashSet::new(); logical_width];
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

        for prep_op in &prepared.operations {
            if prep_op.logical_qubits.len() != 2 {
                continue;
            }
            let u = prep_op.logical_qubits[0];
            let v = prep_op.logical_qubits[1];
            if u == v {
                continue;
            }
            let edge = normalize_index_pair(u, v);
            edge_set.insert(edge);
            adjacency[u].insert(v);
            adjacency[v].insert(u);
        }

        let mut active_nodes: Vec<usize> = adjacency
            .iter()
            .enumerate()
            .filter_map(|(idx, neigh)| if neigh.is_empty() { None } else { Some(idx) })
            .collect();
        active_nodes.sort_unstable();

        let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        edges.sort_unstable();

        InteractionGraph {
            active_nodes,
            adjacency,
            edges,
        }
    }
}

fn build_topology_graph(topology: &TopologyAdapter) -> UnGraph<usize, ()> {
    let mut graph = UnGraph::<usize, ()>::new_undirected();
    let mut nodes = Vec::with_capacity(topology.num_qubits());

    for idx in 0..topology.num_qubits() {
        nodes.push(graph.add_node(idx));
    }

    for u in 0..topology.num_qubits() {
        for &v in &topology.neighbors[u] {
            if u < v {
                graph.add_edge(nodes[u], nodes[v], ());
            }
        }
    }

    graph
}
