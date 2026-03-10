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

use crate::circuit::Qubit;
use crate::device::error::TopologyError;
use rustworkx_core::petgraph::prelude::{NodeIndex, StableDiGraph, StableGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Represents the coupling/connectivity of a quantum device.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::Qubit;
/// use cqlib_core::device::Topology;
///
/// let qubits = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
/// let couplings = vec![
///     (Qubit::new(0), Qubit::new(1), "G0".to_string()),
///     (Qubit::new(1), Qubit::new(2), "G1".to_string()),
/// ];
///
/// let topology = Topology::new(qubits, couplings).unwrap();
/// assert_eq!(topology.num_qubits(), 3);
/// assert!(topology.is_connected(Qubit::new(0), Qubit::new(1)));
/// ```
#[derive(Debug, Clone)]
pub struct Topology {
    /// Mapping from Qubit to NodeIndex in the graph
    node_indices: HashMap<Qubit, NodeIndex>,
    /// The underlying graph: nodes are qubits, edges are couplings
    graph: StableGraph<Qubit, String>,
}

impl Topology {
    /// Creates a new Topology with given qubits and coupling map.
    pub fn new(
        qubits: Vec<Qubit>,
        coupling_map: Vec<(Qubit, Qubit, String)>,
    ) -> Result<Self, TopologyError> {
        let mut graph = StableDiGraph::<Qubit, String>::new();
        let mut node_indices = HashMap::new();

        for qubit in qubits {
            let node_index = graph.add_node(qubit);
            node_indices.insert(qubit, node_index);
        }

        for (control, target, name) in coupling_map {
            if !node_indices.contains_key(&target) {
                return Err(TopologyError::QubitNotFound(target));
            }
            if !node_indices.contains_key(&control) {
                return Err(TopologyError::QubitNotFound(control));
            }
            graph.add_edge(node_indices[&control], node_indices[&target], name.clone());
        }

        Ok(Self {
            node_indices,
            graph,
        })
    }

    pub fn line(qubits: Vec<Qubit>) -> Self {
        let mut graph = StableDiGraph::<Qubit, String>::new();
        let mut node_indices = HashMap::new();
        for qubit in &qubits {
            node_indices.insert(*qubit, graph.add_node(qubit.clone()));
        }
        for qs in qubits.windows(2) {
            graph.add_edge(node_indices[&qs[0]], node_indices[&qs[1]], "".to_string());
        }

        Self {
            node_indices,
            graph,
        }
    }

    /// Returns a reference to the underlying graph.
    pub fn graph(&self) -> &StableGraph<Qubit, String> {
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
    pub fn qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.graph.node_indices().map(|i| self.graph[i])
    }

    /// Adds qubits to the topology.
    ///
    /// Accepts any iterable: Vec, array, or iterator.
    /// Returns error if any qubit already exists.
    pub fn add_qubits(
        &mut self,
        qubits: impl IntoIterator<Item = Qubit>,
    ) -> Result<(), TopologyError> {
        let qubits: Vec<Qubit> = qubits.into_iter().collect();

        for qubit in &qubits {
            if self.node_indices.contains_key(qubit) {
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
    /// Returns error if any qubit doesn't exist.
    pub fn add_couplings(
        &mut self,
        couplings: impl IntoIterator<Item = (Qubit, Qubit, String)>,
    ) -> Result<(), TopologyError> {
        let couplings: Vec<(Qubit, Qubit, String)> = couplings.into_iter().collect();

        // First pass: validate all couplings exist
        for (control, target, _) in &couplings {
            if !self.node_indices.contains_key(control) {
                return Err(TopologyError::QubitNotFound(*control));
            }
            if !self.node_indices.contains_key(target) {
                return Err(TopologyError::QubitNotFound(*target));
            }
        }
        // Second pass: add edges
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
        qubits: impl IntoIterator<Item = Qubit>,
    ) -> Result<(), TopologyError> {
        let qubits: Vec<Qubit> = qubits.into_iter().collect();

        for qubit in &qubits {
            if !self.node_indices.contains_key(qubit) {
                return Err(TopologyError::QubitNotFound(*qubit));
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
        couplings: impl IntoIterator<Item = (Qubit, Qubit)>,
    ) -> Result<(), TopologyError> {
        let collected: Vec<_> = couplings.into_iter().collect();

        // First pass: validate
        for (control, target) in &collected {
            if !self.node_indices.contains_key(control) {
                return Err(TopologyError::QubitNotFound(*control));
            }
            if !self.node_indices.contains_key(target) {
                return Err(TopologyError::QubitNotFound(*target));
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

        // Second pass: remove
        for (control, target) in collected {
            let c_idx = self.node_indices[&control];
            let t_idx = self.node_indices[&target];
            if let Some(edge_idx) = self.graph.find_edge(c_idx, t_idx) {
                self.graph.remove_edge(edge_idx);
            }
        }
        Ok(())
    }

    /// Checks if two qubits are connected.
    pub fn is_connected(&self, control: Qubit, target: Qubit) -> bool {
        if let (Some(&c_idx), Some(&t_idx)) = (
            self.node_indices.get(&control),
            self.node_indices.get(&target),
        ) {
            self.graph.edges(c_idx).any(|e| e.target() == t_idx)
        } else {
            false
        }
    }

    /// Gets the neighbors (coupled qubits) of a given qubit.
    ///
    /// Returns an iterator. Call `.collect()` to get a Vec if needed.
    pub fn neighbors(&self, qubit: Qubit) -> impl Iterator<Item = Qubit> + '_ {
        self.node_indices
            .get(&qubit)
            .into_iter()
            .flat_map(|&node_idx| self.graph.edges(node_idx).map(|e| self.graph[e.target()]))
    }

    /// Gets the coupling name between two qubits.
    ///
    /// Uses O(1) `find_edge` instead of O(degree) linear scan.
    pub fn get_coupling_name(&self, control: Qubit, target: Qubit) -> Option<String> {
        let c_idx = self.node_indices.get(&control)?;
        let t_idx = self.node_indices.get(&target)?;
        self.graph
            .find_edge(*c_idx, *t_idx)
            .map(|e| self.graph[e].clone())
    }

    /// Checks if a qubit exists in the topology.
    pub fn contains_qubit(&self, qubit: &Qubit) -> bool {
        self.node_indices.contains_key(qubit)
    }

    /// Gets the degree (number of connections) of a qubit.
    pub fn degree(&self, qubit: &Qubit) -> usize {
        if let Some(&node_idx) = self.node_indices.get(qubit) {
            self.graph.edges(node_idx).count()
        } else {
            0
        }
    }
}

#[cfg(test)]
#[path = "./topology_test.rs"]
mod topology_test;
