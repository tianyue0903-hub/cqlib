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

//! Topology module for quantum device coupling graphs.
//!
//! This module provides the `Topology` struct which represents the connectivity
//! of a quantum device using a graph structure. Each node represents a qubit
//! and edges represent the coupling (entanglement capability) between qubits.

use crate::device::PhysicalQubit;
use crate::device::error::TopologyError;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::prelude::{NodeIndex, StableDiGraph, StableGraph};
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Represents the coupling/connectivity of a quantum device.
///
/// # Example
///
/// ```rust
/// use cqlib_core::device::{PhysicalQubit, Topology};
///
/// let qubits = vec![PhysicalQubit::new(0), PhysicalQubit::new(1), PhysicalQubit::new(2)];
/// let couplings = vec![
///     (PhysicalQubit::new(0), PhysicalQubit::new(1), "G0".to_string()),
///     (PhysicalQubit::new(1), PhysicalQubit::new(2), "G1".to_string()),
/// ];
///
/// let topology = Topology::new(qubits, couplings).unwrap();
/// assert_eq!(topology.num_qubits(), 3);
/// assert!(topology.supports_directed_coupling(PhysicalQubit::new(0), PhysicalQubit::new(1)));
/// ```
#[derive(Debug, Clone)]
pub struct Topology {
    /// Mapping from physical qubits to graph nodes.
    node_indices: HashMap<PhysicalQubit, NodeIndex>,
    /// The underlying graph: nodes are physical qubits, edges are couplings.
    graph: StableGraph<PhysicalQubit, String>,
}

impl Topology {
    /// Creates a new Topology with given qubits and coupling map.
    pub fn new(
        qubits: Vec<PhysicalQubit>,
        coupling_map: Vec<(PhysicalQubit, PhysicalQubit, String)>,
    ) -> Result<Self, TopologyError> {
        let mut topology = Self {
            node_indices: HashMap::new(),
            graph: StableDiGraph::<PhysicalQubit, String>::new(),
        };
        topology.add_qubits(qubits)?;
        topology.add_couplings(coupling_map)?;
        Ok(topology)
    }

    /// Creates a directed line topology in the supplied qubit order.
    pub fn line(qubits: Vec<PhysicalQubit>) -> Result<Self, TopologyError> {
        let couplings = qubits
            .windows(2)
            .map(|qs| (qs[0], qs[1], String::new()))
            .collect();
        Self::new(qubits, couplings)
    }

    /// Returns a reference to the underlying graph.
    pub fn graph(&self) -> &StableGraph<PhysicalQubit, String> {
        &self.graph
    }

    /// Returns the number of qubits in the topology.
    pub fn num_qubits(&self) -> usize {
        self.graph.node_indices().count()
    }

    /// Returns the number of coupling edges in the topology.
    pub fn num_couplings(&self) -> usize {
        self.graph.edge_indices().count()
    }

    /// Returns all qubits in the topology.
    ///
    /// Returns an iterator. Call `.collect()` to get a Vec if needed.
    pub fn qubits(&self) -> impl Iterator<Item = PhysicalQubit> + '_ {
        self.graph.node_indices().map(|i| self.graph[i])
    }

    /// Adds qubits to the topology.
    ///
    /// Accepts any iterable: Vec, array, or iterator.
    /// Returns error if any qubit already exists.
    pub fn add_qubits(
        &mut self,
        qubits: impl IntoIterator<Item = PhysicalQubit>,
    ) -> Result<(), TopologyError> {
        let qubits: Vec<PhysicalQubit> = qubits.into_iter().collect();
        let mut seen = HashSet::with_capacity(qubits.len());

        for qubit in &qubits {
            if self.node_indices.contains_key(qubit) || !seen.insert(*qubit) {
                return Err(TopologyError::QubitAlreadyExists(*qubit));
            }
        }
        for qubit in qubits {
            let node_index = self.graph.add_node(qubit);
            self.node_indices.insert(qubit, node_index);
        }
        Ok(())
    }

    /// Adds coupling edges to the topology.
    ///
    /// Accepts any iterable: Vec, array, or iterator.
    /// Returns error if any qubit doesn't exist, an edge already exists, or a
    /// self-coupling is requested.
    pub fn add_couplings(
        &mut self,
        couplings: impl IntoIterator<Item = (PhysicalQubit, PhysicalQubit, String)>,
    ) -> Result<(), TopologyError> {
        let couplings: Vec<(PhysicalQubit, PhysicalQubit, String)> =
            couplings.into_iter().collect();
        let mut seen = HashSet::with_capacity(couplings.len());

        // Validate the full request before mutating the graph.
        for (control, target, _) in &couplings {
            if !self.node_indices.contains_key(control) {
                return Err(TopologyError::QubitNotFound(*control));
            }
            if !self.node_indices.contains_key(target) {
                return Err(TopologyError::QubitNotFound(*target));
            }
            if control == target {
                return Err(TopologyError::SelfCoupling { qubit: *control });
            }
            if !seen.insert((*control, *target)) {
                return Err(TopologyError::CouplingAlreadyExists {
                    control: *control,
                    target: *target,
                });
            }
            let c_idx = self.node_indices[control];
            let t_idx = self.node_indices[target];
            if self.graph.find_edge(c_idx, t_idx).is_some() {
                return Err(TopologyError::CouplingAlreadyExists {
                    control: *control,
                    target: *target,
                });
            }
        }
        for (control, target, name) in couplings {
            let c_idx = self.node_indices[&control];
            let t_idx = self.node_indices[&target];
            self.graph.add_edge(c_idx, t_idx, name);
        }
        Ok(())
    }

    /// Removes qubits from the topology.
    ///
    /// This also removes all coupling edges connected to these qubits.
    /// Accepts any iterable: Vec, array, or iterator.
    /// Returns error if any qubit doesn't exist.
    pub fn remove_qubits(
        &mut self,
        qubits: impl IntoIterator<Item = PhysicalQubit>,
    ) -> Result<(), TopologyError> {
        let qubits: Vec<PhysicalQubit> = qubits.into_iter().collect();
        let mut seen = HashSet::with_capacity(qubits.len());

        for qubit in &qubits {
            if !self.node_indices.contains_key(qubit) {
                return Err(TopologyError::QubitNotFound(*qubit));
            }
            if !seen.insert(*qubit) {
                return Err(TopologyError::DuplicateQubitRemoval(*qubit));
            }
        }
        for qubit in qubits {
            if let Some(node_index) = self.node_indices.remove(&qubit) {
                self.graph.remove_node(node_index);
            }
        }
        Ok(())
    }

    /// Removes coupling edges from the topology.
    ///
    /// Uses O(1) `find_edge` instead of O(E) linear scan.
    /// Accepts any iterable: Vec, array, or iterator.
    /// Returns error if qubits don't exist or coupling doesn't exist.
    pub fn remove_couplings(
        &mut self,
        couplings: impl IntoIterator<Item = (PhysicalQubit, PhysicalQubit)>,
    ) -> Result<(), TopologyError> {
        let collected: Vec<_> = couplings.into_iter().collect();
        let mut seen = HashSet::with_capacity(collected.len());

        // Validate the full request before mutating the graph.
        for (control, target) in &collected {
            if !self.node_indices.contains_key(control) {
                return Err(TopologyError::QubitNotFound(*control));
            }
            if !self.node_indices.contains_key(target) {
                return Err(TopologyError::QubitNotFound(*target));
            }
            if !seen.insert((*control, *target)) {
                return Err(TopologyError::DuplicateCouplingRemoval {
                    control: *control,
                    target: *target,
                });
            }
            let c_idx = self.node_indices[control];
            let t_idx = self.node_indices[target];
            if self.graph.find_edge(c_idx, t_idx).is_none() {
                return Err(TopologyError::CouplingNotFound {
                    control: *control,
                    target: *target,
                });
            }
        }

        for (control, target) in collected {
            let c_idx = self.node_indices[&control];
            let t_idx = self.node_indices[&target];
            if let Some(edge_idx) = self.graph.find_edge(c_idx, t_idx) {
                self.graph.remove_edge(edge_idx);
            }
        }
        Ok(())
    }

    /// Checks whether the directed coupling `control -> target` exists.
    pub fn supports_directed_coupling(
        &self,
        control: PhysicalQubit,
        target: PhysicalQubit,
    ) -> bool {
        if let (Some(&c_idx), Some(&t_idx)) = (
            self.node_indices.get(&control),
            self.node_indices.get(&target),
        ) {
            self.graph.find_edge(c_idx, t_idx).is_some()
        } else {
            false
        }
    }

    /// Checks whether a coupling exists between two qubits in either direction.
    pub fn supports_coupling_either_direction(&self, a: PhysicalQubit, b: PhysicalQubit) -> bool {
        self.supports_directed_coupling(a, b) || self.supports_directed_coupling(b, a)
    }

    /// Gets qubits reachable through outgoing couplings from `qubit`.
    ///
    /// Returns an iterator. Call `.collect()` to get a Vec if needed.
    pub fn successors(&self, qubit: PhysicalQubit) -> impl Iterator<Item = PhysicalQubit> + '_ {
        self.node_indices
            .get(&qubit)
            .into_iter()
            .flat_map(|&node_idx| self.graph.edges(node_idx).map(|e| self.graph[e.target()]))
    }

    /// Gets qubits with incoming couplings to `qubit`.
    ///
    /// Returns an iterator. Call `.collect()` to get a Vec if needed.
    pub fn predecessors(&self, qubit: PhysicalQubit) -> impl Iterator<Item = PhysicalQubit> + '_ {
        self.node_indices
            .get(&qubit)
            .into_iter()
            .flat_map(|&node_idx| {
                self.graph
                    .edges_directed(node_idx, Direction::Incoming)
                    .map(|e| self.graph[e.source()])
            })
    }

    /// Gets qubits coupled to `qubit` in either direction.
    ///
    /// Bidirectional couplings are returned once.
    pub fn neighbors_undirected(
        &self,
        qubit: PhysicalQubit,
    ) -> impl Iterator<Item = PhysicalQubit> + '_ {
        let mut neighbors = BTreeSet::new();
        neighbors.extend(self.successors(qubit));
        neighbors.extend(self.predecessors(qubit));
        neighbors.into_iter()
    }

    /// Gets all unique coupling edges without direction.
    ///
    /// Each returned pair is ordered by physical qubit ID, and bidirectional
    /// couplings collapse to one pair.
    pub fn undirected_edges(&self) -> impl Iterator<Item = (PhysicalQubit, PhysicalQubit)> + '_ {
        self.graph
            .edge_references()
            .map(|edge| {
                let source = self.graph[edge.source()];
                let target = self.graph[edge.target()];
                if source <= target {
                    (source, target)
                } else {
                    (target, source)
                }
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    /// Gets the coupling name between two qubits.
    ///
    /// Uses O(1) `find_edge` instead of O(degree) linear scan.
    pub fn get_coupling_name(
        &self,
        control: PhysicalQubit,
        target: PhysicalQubit,
    ) -> Option<String> {
        let c_idx = self.node_indices.get(&control)?;
        let t_idx = self.node_indices.get(&target)?;
        self.graph
            .find_edge(*c_idx, *t_idx)
            .map(|e| self.graph[e].clone())
    }

    /// Checks if a qubit exists in the topology.
    pub fn contains_qubit(&self, qubit: &PhysicalQubit) -> bool {
        self.node_indices.contains_key(qubit)
    }

    /// Gets the number of outgoing couplings from a qubit.
    pub fn out_degree(&self, qubit: &PhysicalQubit) -> usize {
        if let Some(&node_idx) = self.node_indices.get(qubit) {
            self.graph.edges(node_idx).count()
        } else {
            0
        }
    }

    /// Gets the number of incoming couplings to a qubit.
    pub fn in_degree(&self, qubit: &PhysicalQubit) -> usize {
        if let Some(&node_idx) = self.node_indices.get(qubit) {
            self.graph
                .edges_directed(node_idx, Direction::Incoming)
                .count()
        } else {
            0
        }
    }
}

#[cfg(test)]
#[path = "./topology_test.rs"]
mod topology_test;
