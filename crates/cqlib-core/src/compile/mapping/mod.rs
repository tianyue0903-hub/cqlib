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

//! Routing/mapping algorithms and shared utilities.
//!
//! This module hosts core shared data structures used by both VF2 and SABRE,
//! plus the hybrid `map_with_vf2_sabre` orchestration function.
//!
//! Design highlights:
//! - a normalized `TopologyAdapter` for dense adjacency/fidelity lookups
//! - a preprocessing pass that enforces 1q/2q, control-flow-free constraints
//! - lightweight helpers for rebuilding mapped output circuits
//! - deterministic canonicalization of undirected edge keys
//!
//! These helpers are intentionally kept in one place to avoid duplicated
//! validation and conversion logic between algorithms.

/// Genetic Algorithm-based mapping optimizer.
pub mod ga_mapping;
/// SABRE mapper implementation and its configuration model.
pub mod sabre;
/// VF2-based structural mapper and candidate-layout search.
pub mod vf2;

pub(crate) use crate::compile::structured::{
    PreparedIfElse, PreparedPassthroughOp, PreparedProgram, PreparedProgramItem, PreparedSegment,
    PreparedWhileLoop, build_if_else_operation, build_while_loop_operation, map_program_static,
    preprocess_program,
};
pub use ga_mapping::{GaConfig, GeneticAlgMapping};
pub use sabre::{SabreConfig, SabreMapping, Vf2Policy};
pub use vf2::{
    Vf2CandidateOptions, Vf2CandidateScore, Vf2LayoutCandidate, Vf2Mapping, Vf2ScoreWeights,
};

use crate::circuit::gate::control_flow::ControlFlow;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use crate::circuit::{Circuit, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
pub(crate) use crate::compile::prepared::{PreparedCircuit, preprocess_circuit};
use crate::device::Topology;
use indexmap::IndexSet;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::collections::{HashMap, HashSet};

/// Optional fidelity map keyed by physical qubit edge.
///
/// The mapping is treated as undirected by normalizing `(u, v)` and `(v, u)`
/// to the same canonical key.
pub type FidelityMap = HashMap<(Qubit, Qubit), f64>;

#[derive(Debug, Clone)]
/// Internal struct `TopologyAdapter` used by compile mapping workflows.
pub(crate) struct TopologyAdapter {
    /// Physical qubits sorted by `Qubit::id`.
    pub(crate) physical_qubits: Vec<Qubit>,
    /// Adjacency list over physical-qubit indices.
    pub(crate) neighbors: Vec<Vec<usize>>,
    /// Symmetric adjacency matrix.
    pub(crate) adj_matrix: Vec<Vec<bool>>,
    /// All-pairs shortest path lengths.
    pub(crate) dist: Vec<Vec<u32>>,
    /// Symmetric edge-fidelity matrix.
    pub(crate) fidelity: Vec<Vec<f64>>,
    /// Indices in the largest connected component.
    pub(crate) largest_component: Vec<usize>,
    /// Physical -> Index mapping.
    pub(crate) qubit_to_index: HashMap<Qubit, usize>,
}

impl TopologyAdapter {
    /// Builds a dense topology adapter from sparse topology/fidelity inputs.
    ///
    /// Validates fidelity range, known qubits, and existence of referenced edges.
    pub(crate) fn new(
        topology: &Topology,
        fidelity_map: Option<&FidelityMap>,
    ) -> Result<Self, CompileError> {
        let mut physical_qubits: Vec<Qubit> = topology.qubits().collect();
        physical_qubits.sort_by_key(Qubit::id);

        let qubit_to_index: HashMap<Qubit, usize> = physical_qubits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, q)| (q, idx))
            .collect();

        let n = physical_qubits.len();
        let mut adj_matrix = vec![vec![false; n]; n];
        let mut neighbors_set: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        for edge in topology.graph().edge_references() {
            let u = topology.graph()[edge.source()];
            let v = topology.graph()[edge.target()];
            let Some(&u_idx) = qubit_to_index.get(&u) else {
                continue;
            };
            let Some(&v_idx) = qubit_to_index.get(&v) else {
                continue;
            };

            if u_idx == v_idx {
                continue;
            }

            adj_matrix[u_idx][v_idx] = true;
            adj_matrix[v_idx][u_idx] = true;
            neighbors_set[u_idx].insert(v_idx);
            neighbors_set[v_idx].insert(u_idx);
        }

        let mut fidelity_overrides: HashMap<(usize, usize), f64> = HashMap::new();
        if let Some(fidelity_map) = fidelity_map {
            for (&(u, v), &value) in fidelity_map {
                if !(0.0..=1.0).contains(&value) {
                    return Err(CompileError::InvalidFidelity { u, v, value });
                }

                let Some(&u_idx) = qubit_to_index.get(&u) else {
                    return Err(CompileError::UnknownFidelityQubit { u, v });
                };
                let Some(&v_idx) = qubit_to_index.get(&v) else {
                    return Err(CompileError::UnknownFidelityQubit { u, v });
                };

                let key = normalize_index_pair(u_idx, v_idx);
                fidelity_overrides.insert(key, value);
            }
        }

        for &(u_idx, v_idx) in fidelity_overrides.keys() {
            if !adj_matrix[u_idx][v_idx] {
                return Err(CompileError::FidelityEdgeNotFound {
                    u: physical_qubits[u_idx],
                    v: physical_qubits[v_idx],
                });
            }
        }

        let mut fidelity = vec![vec![1.0; n]; n];
        for i in 0..n {
            fidelity[i][i] = 1.0;
        }
        for i in 0..n {
            for &j in &neighbors_set[i] {
                let key = normalize_index_pair(i, j);
                let edge_fidelity = fidelity_overrides.get(&key).copied().unwrap_or(1.0);
                fidelity[i][j] = edge_fidelity;
                fidelity[j][i] = edge_fidelity;
            }
        }

        let mut neighbors = vec![Vec::new(); n];
        for i in 0..n {
            let mut ns: Vec<usize> = neighbors_set[i].iter().copied().collect();
            ns.sort_by_key(|idx| physical_qubits[*idx].id());
            neighbors[i] = ns;
        }

        let dist = compute_all_pairs_shortest_path(&adj_matrix);
        let largest_component = compute_largest_component(&neighbors);

        Ok(Self {
            physical_qubits,
            neighbors,
            adj_matrix,
            dist,
            fidelity,
            largest_component,
            qubit_to_index,
        })
    }

    /// Returns number of physical qubits in this adapter.
    pub(crate) fn num_qubits(&self) -> usize {
        self.physical_qubits.len()
    }

    /// Returns whether two physical indices are adjacent.
    pub(crate) fn is_adjacent(&self, u_idx: usize, v_idx: usize) -> bool {
        self.adj_matrix[u_idx][v_idx]
    }

    /// Returns stored edge fidelity for two physical indices.
    pub(crate) fn edge_fidelity(&self, u_idx: usize, v_idx: usize) -> f64 {
        self.fidelity[u_idx][v_idx]
    }
}

/// Normalizes an undirected index pair so `(a, b)` and `(b, a)` share one key.
pub(crate) fn normalize_index_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Returns whether an operation is a standard `CX` gate.
pub(crate) fn is_cx(op: &Operation) -> bool {
    matches!(op.instruction, Instruction::Standard(StandardGate::CX))
}

/// Clones an operation while replacing its qubit list with mapped qubits.
pub(crate) fn map_operation_qubits(op: &Operation, mapped_qubits: &[Qubit]) -> Operation {
    let mut mapped = op.clone();
    mapped.qubits = mapped_qubits.iter().copied().collect();
    mapped
}

/// Builds mapped output from source metadata while preserving recursive control-flow qubits.
pub(crate) fn build_output_circuit_from_source(
    source: &Circuit,
    mapped_ops: Vec<Operation>,
) -> Circuit {
    let mut used_set = HashSet::new();
    collect_program_qubits(&mapped_ops, &mut used_set);
    let mut used_qubits: Vec<Qubit> = used_set.into_iter().collect();
    used_qubits.sort_by_key(Qubit::id);

    let mut symbols = source.symbols().clone();
    let mut parameters = source.parameters().clone();
    let global_phase = preserve_global_phase(source, &mut symbols, &mut parameters);

    Circuit::from_parts(
        used_qubits.into_iter().collect::<IndexSet<Qubit>>(),
        symbols,
        parameters,
        mapped_ops,
        global_phase,
    )
}

fn preserve_global_phase(
    source: &Circuit,
    symbols: &mut IndexSet<String>,
    parameters: &mut IndexSet<Parameter>,
) -> CircuitParam {
    let phase_param = source.global_phase();
    if let Ok(value) = phase_param.evaluate(&None) {
        CircuitParam::Fixed(value)
    } else {
        let (index, is_new) = parameters.insert_full(phase_param.clone());
        if is_new {
            for sym in phase_param.get_symbols() {
                symbols.insert(sym);
            }
        }
        CircuitParam::Index(index as u32)
    }
}

fn collect_program_qubits(ops: &[Operation], out: &mut HashSet<Qubit>) {
    for op in ops {
        for &q in &op.qubits {
            out.insert(q);
        }
        if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
            match control_flow {
                ControlFlow::IfElse(gate) => {
                    out.insert(gate.condition().qubit);
                    collect_program_qubits(gate.true_body(), out);
                    if let Some(false_body) = gate.false_body() {
                        collect_program_qubits(false_body, out);
                    }
                }
                ControlFlow::WhileLoop(gate) => {
                    out.insert(gate.condition().qubit);
                    collect_program_qubits(gate.body(), out);
                }
            }
        }
    }
}

/// Maps a logical circuit onto topology using VF2-first + SABRE fallback policy.
///
/// # Arguments
///
/// * `circuit` - Logical input circuit.
/// * `topology` - Target device topology.
/// * `fidelity_map` - Optional undirected edge-fidelity overrides.
/// * `config` - SABRE configuration, including VF2 policy mode.
///
/// # Returns
///
/// * `Ok(Circuit)` - A mapped circuit with operations constrained to topology edges.
///
/// # Errors
///
/// Returns [`CompileError`] when validation fails, when VF2 strict mapping
/// fails under required policy, or when SABRE cannot progress.
pub fn map_with_vf2_sabre(
    circuit: &Circuit,
    topology: &Topology,
    fidelity_map: Option<&FidelityMap>,
    config: &SabreConfig,
) -> Result<Circuit, CompileError> {
    let fidelity_owned = fidelity_map.cloned();

    if matches!(config.vf2_policy, Vf2Policy::DirectThenSabre) {
        let mut vf2 = Vf2Mapping::new(topology.clone(), fidelity_owned.clone())?;
        if vf2.is_subgraph_isomorphic(circuit)? {
            return vf2.execute(circuit);
        }
    }

    let mut sabre = SabreMapping::new(topology.clone(), fidelity_owned, config.clone())?;
    sabre.execute(circuit)
}

pub fn map_with_ga(
    circuit: &Circuit,
    topology: &Topology,
    config: &GaConfig,
    fidelity_map: Option<&FidelityMap>,
    invalid_qubits: Option<HashSet<usize>>,
) -> Result<Circuit, CompileError> {
    let fidelity_owned = fidelity_map.cloned();
    let mut ga = GeneticAlgMapping::new(
        topology.clone(),
        config.clone(),
        fidelity_owned.clone(),
        invalid_qubits.clone(),
    )
    .unwrap();
    ga.execute(circuit)
}

/// Computes all-pairs shortest path on an unweighted adjacency matrix.
fn compute_all_pairs_shortest_path(adj_matrix: &[Vec<bool>]) -> Vec<Vec<u32>> {
    let n = adj_matrix.len();
    if n == 0 {
        return vec![];
    }

    let inf = (n as u32).saturating_mul(2).saturating_add(1);
    let mut dist = vec![vec![inf; n]; n];

    for i in 0..n {
        dist[i][i] = 0;
        for j in 0..n {
            if adj_matrix[i][j] {
                dist[i][j] = 1;
            }
        }
    }

    for k in 0..n {
        for i in 0..n {
            let dik = dist[i][k];
            if dik == inf {
                continue;
            }
            for j in 0..n {
                let dkj = dist[k][j];
                if dkj == inf {
                    continue;
                }
                let cand = dik.saturating_add(dkj);
                if cand < dist[i][j] {
                    dist[i][j] = cand;
                }
            }
        }
    }

    dist
}

/// Returns index set of the largest connected component in the topology graph.
fn compute_largest_component(neighbors: &[Vec<usize>]) -> Vec<usize> {
    let n = neighbors.len();
    let mut visited = vec![false; n];
    let mut best = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        let mut component = Vec::new();

        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            component.push(node);
            for &next in &neighbors[node] {
                if !visited[next] {
                    stack.push(next);
                }
            }
        }

        component.sort_unstable();
        if component.len() > best.len() {
            best = component;
        }
    }

    best
}

#[cfg(test)]
#[path = "mapping_test.rs"]
mod mapping_test;
