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

//! Physical topology view used by layout methods.

use crate::compile::CompilerError;
use crate::device::{Device, PhysicalQubit, Topology};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Compiler-local physical graph with usable qubits and distance data.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalLayoutGraph {
    physical_qubits: Vec<PhysicalQubit>,
    physical_index: BTreeMap<PhysicalQubit, usize>,
    adjacency: Vec<Vec<usize>>,
    distances: DistanceTable,
    directed_couplings: BTreeSet<(PhysicalQubit, PhysicalQubit)>,
    readout_errors: BTreeMap<PhysicalQubit, f64>,
    two_qubit_errors: BTreeMap<(PhysicalQubit, PhysicalQubit), f64>,
    has_readout_error_data: bool,
    has_two_qubit_error_data: bool,
}

impl PhysicalLayoutGraph {
    /// Builds a layout physical graph from a device, excluding invalid qubits.
    pub fn from_device(device: &Device) -> Result<Self, CompilerError> {
        let physical_qubits: Vec<_> = device.usable_qubits().collect();
        if physical_qubits.is_empty() {
            return Err(CompilerError::InvalidInput(
                "layout requires at least one usable physical qubit".to_string(),
            ));
        }

        let usable: BTreeSet<_> = physical_qubits.iter().copied().collect();
        let physical_index = build_physical_index(&physical_qubits);
        let adjacency = build_undirected_adjacency(device.topology(), &physical_qubits, &usable);
        let distances =
            DistanceTable::from_adjacency(&physical_qubits, &physical_index, &adjacency);
        let directed_couplings = collect_directed_couplings(device.topology(), &usable);
        let (readout_errors, has_readout_error_data) =
            collect_readout_errors(device, &physical_qubits)?;
        let (two_qubit_errors, has_two_qubit_error_data) =
            collect_two_qubit_errors(device, &usable)?;

        Ok(Self {
            physical_qubits,
            physical_index,
            adjacency,
            distances,
            directed_couplings,
            readout_errors,
            two_qubit_errors,
            has_readout_error_data,
            has_two_qubit_error_data,
        })
    }

    /// Returns usable physical qubits in deterministic order.
    pub fn physical_qubits(&self) -> &[PhysicalQubit] {
        &self.physical_qubits
    }

    /// Returns the distance table over usable physical qubits.
    pub fn distances(&self) -> &DistanceTable {
        &self.distances
    }

    pub(super) fn physical_index(&self, qubit: PhysicalQubit) -> Option<usize> {
        self.physical_index.get(&qubit).copied()
    }

    pub(super) fn physical_at(&self, index: usize) -> Option<PhysicalQubit> {
        self.physical_qubits.get(index).copied()
    }

    pub(super) fn distance_by_index(&self, a: usize, b: usize) -> Option<u32> {
        self.distances.distance_by_index(a, b)
    }

    /// Returns the undirected shortest-path distance between two physical qubits.
    pub fn distance(&self, a: PhysicalQubit, b: PhysicalQubit) -> Option<u32> {
        let a_index = self.physical_index(a)?;
        let b_index = self.physical_index(b)?;
        self.distance_by_index(a_index, b_index)
    }

    /// Returns whether two physical qubits are adjacent in either direction.
    pub fn is_adjacent_undirected(&self, a: PhysicalQubit, b: PhysicalQubit) -> bool {
        let Some(a_index) = self.physical_index(a) else {
            return false;
        };
        let Some(b_index) = self.physical_index(b) else {
            return false;
        };
        self.adjacency[a_index].contains(&b_index)
    }

    /// Returns the readout error for a physical qubit, if known.
    pub fn readout_error(&self, qubit: PhysicalQubit) -> Option<f64> {
        self.readout_errors.get(&qubit).copied()
    }

    pub(super) fn readout_error_by_index(&self, index: usize) -> Option<f64> {
        self.physical_at(index)
            .and_then(|qubit| self.readout_error(qubit))
    }

    /// Returns the directed two-qubit error for a coupling, if known.
    pub fn two_qubit_error_directed(
        &self,
        control: PhysicalQubit,
        target: PhysicalQubit,
    ) -> Option<f64> {
        self.two_qubit_errors.get(&(control, target)).copied()
    }

    /// Returns the lowest known two-qubit error in either coupling direction.
    pub fn two_qubit_error_undirected(&self, a: PhysicalQubit, b: PhysicalQubit) -> Option<f64> {
        match (
            self.two_qubit_error_directed(a, b),
            self.two_qubit_error_directed(b, a),
        ) {
            (Some(ab), Some(ba)) => Some(ab.min(ba)),
            (Some(ab), None) => Some(ab),
            (None, Some(ba)) => Some(ba),
            (None, None) => None,
        }
    }

    pub(super) fn two_qubit_error_undirected_by_index(&self, a: usize, b: usize) -> Option<f64> {
        let a = self.physical_at(a)?;
        let b = self.physical_at(b)?;
        self.two_qubit_error_undirected(a, b)
    }

    /// Returns whether there is a directed coupling from `control` to `target`.
    pub fn supports_directed_coupling(
        &self,
        control: PhysicalQubit,
        target: PhysicalQubit,
    ) -> bool {
        self.directed_couplings.contains(&(control, target))
    }

    pub(super) fn supports_directed_coupling_by_index(
        &self,
        control: usize,
        target: usize,
    ) -> bool {
        match (self.physical_at(control), self.physical_at(target)) {
            (Some(control), Some(target)) => self.supports_directed_coupling(control, target),
            _ => false,
        }
    }

    /// Returns whether any calibration/error data is available.
    pub fn has_fidelity_data(&self) -> bool {
        self.has_readout_error_data || self.has_two_qubit_error_data
    }

    /// Returns whether readout-error data is available.
    pub fn has_readout_error_data(&self) -> bool {
        self.has_readout_error_data
    }

    /// Returns whether two-qubit error data is available.
    pub fn has_two_qubit_error_data(&self) -> bool {
        self.has_two_qubit_error_data
    }
}

/// All-pairs undirected shortest-path distances over usable physical qubits.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceTable {
    qubits: Vec<PhysicalQubit>,
    index: BTreeMap<PhysicalQubit, usize>,
    distances: Vec<Vec<Option<u32>>>,
}

impl DistanceTable {
    fn from_adjacency(
        qubits: &[PhysicalQubit],
        index: &BTreeMap<PhysicalQubit, usize>,
        adjacency: &[Vec<usize>],
    ) -> Self {
        let mut distances = vec![vec![None; qubits.len()]; qubits.len()];

        for start_index in 0..qubits.len() {
            let mut queue = VecDeque::new();
            distances[start_index][start_index] = Some(0);
            queue.push_back(start_index);

            while let Some(current_index) = queue.pop_front() {
                let current_distance = distances[start_index][current_index]
                    .expect("queued nodes have assigned distances");

                for neighbor_index in adjacency[current_index].iter().copied() {
                    if distances[start_index][neighbor_index].is_none() {
                        distances[start_index][neighbor_index] = Some(current_distance + 1);
                        queue.push_back(neighbor_index);
                    }
                }
            }
        }

        Self {
            qubits: qubits.to_vec(),
            index: index.clone(),
            distances,
        }
    }

    /// Returns the physical qubits covered by this table.
    pub fn qubits(&self) -> &[PhysicalQubit] {
        &self.qubits
    }

    /// Returns the shortest-path distance between two physical qubits.
    pub fn distance(&self, a: PhysicalQubit, b: PhysicalQubit) -> Option<u32> {
        let a_index = self.index.get(&a)?;
        let b_index = self.index.get(&b)?;
        self.distance_by_index(*a_index, *b_index)
    }

    pub(super) fn distance_by_index(&self, a: usize, b: usize) -> Option<u32> {
        self.distances.get(a)?.get(b).copied().flatten()
    }
}

/// Builds a physical layout graph from a device.
pub fn build_physical_layout_graph(device: &Device) -> Result<PhysicalLayoutGraph, CompilerError> {
    PhysicalLayoutGraph::from_device(device)
}

fn build_physical_index(physical_qubits: &[PhysicalQubit]) -> BTreeMap<PhysicalQubit, usize> {
    physical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(index, qubit)| (qubit, index))
        .collect()
}

fn build_undirected_adjacency(
    topology: &Topology,
    physical_qubits: &[PhysicalQubit],
    usable: &BTreeSet<PhysicalQubit>,
) -> Vec<Vec<usize>> {
    let index = build_physical_index(physical_qubits);
    let mut adjacency = vec![Vec::new(); physical_qubits.len()];

    for qubit in physical_qubits {
        let qubit_index = index[qubit];
        for neighbor in topology.successors(*qubit) {
            if usable.contains(&neighbor) {
                adjacency[qubit_index].push(index[&neighbor]);
            }
        }
        for neighbor in topology.predecessors(*qubit) {
            if usable.contains(&neighbor) {
                adjacency[qubit_index].push(index[&neighbor]);
            }
        }
    }

    for neighbors in &mut adjacency {
        neighbors.sort_unstable_by_key(|index| physical_qubits[*index]);
        neighbors.dedup();
    }

    adjacency
}

fn collect_readout_errors(
    device: &Device,
    physical_qubits: &[PhysicalQubit],
) -> Result<(BTreeMap<PhysicalQubit, f64>, bool), CompilerError> {
    let mut errors = BTreeMap::new();
    for qubit in physical_qubits {
        if let Some(error) = device.get_readout_error(*qubit) {
            validate_probability(error, "readout error")?;
            errors.insert(*qubit, error);
        }
    }
    let has_data = !errors.is_empty();
    Ok((errors, has_data))
}

fn collect_directed_couplings(
    topology: &Topology,
    usable: &BTreeSet<PhysicalQubit>,
) -> BTreeSet<(PhysicalQubit, PhysicalQubit)> {
    let mut couplings = BTreeSet::new();
    for control in usable {
        for target in topology.successors(*control) {
            if usable.contains(&target) {
                couplings.insert((*control, target));
            }
        }
    }
    couplings
}

fn collect_two_qubit_errors(
    device: &Device,
    usable: &BTreeSet<PhysicalQubit>,
) -> Result<(BTreeMap<(PhysicalQubit, PhysicalQubit), f64>, bool), CompilerError> {
    let default_error = device.default_two_qubit_error();
    if let Some(error) = default_error {
        validate_probability(error, "default two-qubit error")?;
    }

    let mut errors = BTreeMap::new();
    let mut has_specific_data = false;
    for control in usable {
        for target in device.topology().successors(*control) {
            if !usable.contains(&target) {
                continue;
            }
            let specific = device.edge_properties(*control, target).and_then(|edge| {
                edge.native_instructions()
                    .iter()
                    .map(|instruction| instruction.error_rate())
                    .min_by(|a, b| a.total_cmp(b))
            });
            if let Some(error) = specific {
                validate_probability(error, "edge two-qubit error")?;
                has_specific_data = true;
                errors.insert((*control, target), error);
            } else if let Some(error) = default_error {
                errors.insert((*control, target), error);
            }
        }
    }

    let has_data = has_specific_data || default_error.is_some();
    Ok((errors, has_data))
}

fn validate_probability(value: f64, name: &str) -> Result<(), CompilerError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(CompilerError::InvalidInput(format!(
            "device {} must be a finite probability in [0, 1], got {}",
            name, value
        )))
    }
}
