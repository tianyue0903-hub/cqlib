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

/// SABRE mapper implementation and its configuration model.
pub mod sabre;
mod structured;
/// VF2-based structural mapper and candidate-layout search.
pub mod vf2;
/// Genetic Algorithm-based mapping optimizer.
pub mod ga_mapping;


pub use sabre::{SabreConfig, SabreMapping, Vf2Policy};
pub(crate) use structured::{
    build_if_else_operation, build_while_loop_operation, map_program_static, preprocess_program,
    PreparedIfElse, PreparedPassthroughOp, PreparedProgram, PreparedProgramItem, PreparedSegment,
    PreparedWhileLoop,
};
pub use vf2::{
    Vf2CandidateOptions, Vf2CandidateScore, Vf2LayoutCandidate, Vf2Mapping, Vf2ScoreWeights,
};
pub use ga_mapping::{GaConfig, GeneticAlgMapping};


use crate::circuit::dag::Terminator;
use crate::circuit::gate::control_flow::ControlFlow;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use crate::circuit::param::ParameterValue;
use crate::circuit::{Circuit, CircuitDag, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::device::Topology;
use indexmap::IndexSet;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};

/// Optional fidelity map keyed by physical qubit edge.
///
/// The mapping is treated as undirected by normalizing `(u, v)` and `(v, u)`
/// to the same canonical key.
pub type FidelityMap = HashMap<(Qubit, Qubit), f64>;

#[derive(Debug, Clone)]
/// Internal struct `PreparedOperation` used by compile mapping workflows.
pub(crate) struct PreparedOperation {
    /// Original operation from the source circuit.
    pub(crate) op: Operation,
    /// Logical-qubit indices corresponding to `op.qubits`.
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
}

#[derive(Debug, Clone)]
/// Internal struct `PreparedCircuit` used by compile mapping workflows.
pub(crate) struct PreparedCircuit {
    /// Logical qubits in circuit ordering.
    pub(crate) logical_qubits: Vec<Qubit>,
    /// Parameter pool copied from source circuit.
    pub(crate) parameters: Vec<Parameter>,
    /// Validated operations with cached logical indices.
    pub(crate) operations: Vec<PreparedOperation>,
}

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
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
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

/// Appends one operation to an output circuit while resolving parameter references.
pub(crate) fn append_operation(
    output: &mut Circuit,
    op: &Operation,
    parameter_pool: &[Parameter],
) -> Result<(), CompileError> {
    let mut params: SmallVec<[ParameterValue; 3]> = smallvec![];
    for p in &op.params {
        match p {
            CircuitParam::Fixed(v) => params.push(ParameterValue::Fixed(*v)),
            CircuitParam::Index(index) => {
                let idx = *index as usize;
                let Some(param) = parameter_pool.get(idx) else {
                    return Err(CompileError::Internal(format!(
                        "operation references missing parameter index {}",
                        idx
                    )));
                };
                params.push(ParameterValue::Param(param.clone()));
            }
        }
    }

    output.append(
        op.instruction.clone(),
        op.qubits.clone(),
        params,
        op.label.as_deref(),
    )?;
    Ok(())
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

/// Validates and flattens a circuit into compile-friendly internal form.
///
/// The pass currently accepts only single-block, return-terminated DAGs and
/// only 1q/2q operations with no control-flow nodes.
pub(crate) fn preprocess_circuit(circuit: &Circuit) -> Result<PreparedCircuit, CompileError> {
    let dag = CircuitDag::from_circuit(circuit)
        .map_err(|err| CompileError::DagBuildFailed(err.to_string()))?;

    if dag.num_blocks() != 1 {
        return Err(CompileError::UnsupportedControlFlow);
    }

    let entry = dag.entry_block().ok_or(CompileError::MissingEntryBlock)?;
    let block = dag
        .data
        .node_weight(entry)
        .ok_or(CompileError::MissingEntryBlock)?;

    if !matches!(block.terminator, None | Some(Terminator::Return)) {
        return Err(CompileError::UnsupportedControlFlow);
    }

    let logical_qubits = circuit.qubits();
    let parameters = circuit.parameters().iter().cloned().collect();
    let logical_index_map: HashMap<Qubit, usize> = logical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, q)| (q, idx))
        .collect();

    let mut operations = Vec::with_capacity(block.operations.len());

    for (op_index, op) in block.operations.iter().enumerate() {
        match &op.instruction {
            Instruction::ControlFlowGate(_) => return Err(CompileError::UnsupportedControlFlow),
            Instruction::Directive(d) => {
                return Err(CompileError::UnsupportedInstruction {
                    op_index,
                    instruction: format!("Directive::{d}"),
                });
            }
            Instruction::Delay => {
                return Err(CompileError::UnsupportedInstruction {
                    op_index,
                    instruction: "Delay".to_string(),
                });
            }
            _ => {}
        }

        let arity = op.qubits.len();
        if arity != 1 && arity != 2 {
            return Err(CompileError::UnsupportedArity { op_index, arity });
        }

        let mut logical = SmallVec::<[usize; 2]>::with_capacity(arity);
        for &q in &op.qubits {
            let Some(&logical_idx) = logical_index_map.get(&q) else {
                return Err(CompileError::Internal(format!(
                    "qubit {q} not found in circuit logical ordering"
                )));
            };
            logical.push(logical_idx);
        }

        operations.push(PreparedOperation {
            op: op.clone(),
            logical_qubits: logical,
        });
    }

    Ok(PreparedCircuit {
        logical_qubits,
        parameters,
        operations,
    })
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
    invalid_qubits: Option<HashSet<usize>>
) -> Result<Circuit, CompileError> {

    let fidelity_owned = fidelity_map.cloned();
    let mut ga = GeneticAlgMapping::new(topology.clone(), config.clone(), fidelity_owned.clone(), invalid_qubits.clone()).unwrap();
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
mod tests {
    use super::*;
    use crate::circuit::gate::control_flow::{ConditionView, ControlFlow};
    use crate::circuit::gate::{Directive, Instruction, StandardGate};
    use crate::circuit::param::ParameterValue;
    use crate::circuit::{Circuit, Operation, Parameter, Qubit};
    use crate::compile::error::CompileError;
    use smallvec::smallvec;
    use std::collections::HashSet;
    use std::convert::TryFrom;

    fn line_topology(ids: &[u32]) -> Topology {
        let qubits: Vec<Qubit> = ids.iter().copied().map(Qubit::new).collect();
        let couplings = ids
            .windows(2)
            .map(|w| (Qubit::new(w[0]), Qubit::new(w[1]), "CX".to_string()))
            .collect();
        Topology::new(qubits, couplings)
    }

    fn connected_undirected(topology: &Topology, a: Qubit, b: Qubit) -> bool {
        topology.is_connected(a, b) || topology.is_connected(b, a)
    }

    fn assert_mapped_ops_2q_edges(ops: &[Operation], topology: &Topology) {
        for op in ops {
            if op.qubits.len() == 2 {
                assert!(
                    connected_undirected(topology, op.qubits[0], op.qubits[1]),
                    "2q op is not on a topology edge: {:?}",
                    op.qubits
                );
            }
            if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
                match control_flow {
                    ControlFlow::IfElse(gate) => {
                        assert_mapped_ops_2q_edges(gate.true_body(), topology);
                        if let Some(false_body) = gate.false_body() {
                            assert_mapped_ops_2q_edges(false_body, topology);
                        }
                    }
                    ControlFlow::WhileLoop(gate) => {
                        assert_mapped_ops_2q_edges(gate.body(), topology);
                    }
                }
            }
        }
    }

    fn assert_mapped_2q_edges(mapped: &Circuit, topology: &Topology) {
        assert_mapped_ops_2q_edges(mapped.operations(), topology);
    }

    fn count_swaps(circuit: &Circuit) -> usize {
        fn count_ops(ops: &[Operation]) -> usize {
            ops.iter()
                .map(|op| {
                    let mut total = usize::from(matches!(
                        op.instruction,
                        Instruction::Standard(StandardGate::SWAP)
                    ));
                    if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
                        match control_flow {
                            ControlFlow::IfElse(gate) => {
                                total += count_ops(gate.true_body());
                                if let Some(false_body) = gate.false_body() {
                                    total += count_ops(false_body);
                                }
                            }
                            ControlFlow::WhileLoop(gate) => {
                                total += count_ops(gate.body());
                            }
                        }
                    }
                    total
                })
                .sum()
        }

        count_ops(circuit.operations())
    }

    fn append_test_operation(circuit: &mut Circuit, op: Operation) {
        circuit
            .append(
                op.instruction,
                op.qubits,
                std::iter::empty::<ParameterValue>(),
                op.label.as_deref(),
            )
            .unwrap();
    }

    fn collect_directives_recursive(ops: &[Operation]) -> Vec<Directive> {
        let mut out = Vec::new();
        for op in ops {
            if let Instruction::Directive(directive) = op.instruction {
                out.push(directive);
            }
            if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
                match control_flow {
                    ControlFlow::IfElse(gate) => {
                        out.extend(collect_directives_recursive(gate.true_body()));
                        if let Some(false_body) = gate.false_body() {
                            out.extend(collect_directives_recursive(false_body));
                        }
                    }
                    ControlFlow::WhileLoop(gate) => {
                        out.extend(collect_directives_recursive(gate.body()));
                    }
                }
            }
        }
        out
    }

    fn fingerprint_ops(ops: &[Operation], out: &mut Vec<String>) {
        for op in ops {
            let mut qids: Vec<u32> = op.qubits.iter().map(Qubit::id).collect();
            qids.sort_unstable();
            match &op.instruction {
                Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                    out.push(format!(
                        "if:{:?}:{:?}:{:?}",
                        gate.condition(),
                        qids,
                        op.label.as_deref()
                    ));
                    fingerprint_ops(gate.true_body(), out);
                    if let Some(false_body) = gate.false_body() {
                        out.push("if:false:some".into());
                        fingerprint_ops(false_body, out);
                    } else {
                        out.push("if:false:none".into());
                    }
                    out.push("if:end".into());
                }
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                    out.push(format!(
                        "while:{:?}:{:?}:{:?}",
                        gate.condition(),
                        qids,
                        op.label.as_deref()
                    ));
                    fingerprint_ops(gate.body(), out);
                    out.push("while:end".into());
                }
                _ => out.push(format!(
                    "{:?}:{:?}:{:?}",
                    op.instruction,
                    qids,
                    op.label.as_deref()
                )),
            }
        }
    }

    fn fingerprint(circuit: &Circuit) -> Vec<String> {
        let mut out = Vec::new();
        fingerprint_ops(circuit.operations(), &mut out);
        out
    }

    #[test]
    fn test_module_exports_compile_and_device() {
        let _cfg = crate::compile::SabreConfig::default();
        let _topology = crate::device::Topology::new(vec![Qubit::new(0)], vec![]);
    }

    #[test]
    fn test_vf2_map_while_loop_preserves_structure_and_condition() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        let q2 = Qubit::new(2);
        circuit.measure(q0).unwrap();

        let body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![q1, q2],
            params: smallvec![],
            label: None,
        }];
        circuit.while_loop(ConditionView::new(q0, 1), body).unwrap();

        let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        let mapped = vf2.execute(&circuit).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);

        assert!(matches!(
            mapped.operations()[1].instruction,
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(_))
        ));
        match &mapped.operations()[1].instruction {
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
                assert_eq!(gate.body().len(), 1);
                assert!(matches!(
                    gate.body()[0].instruction,
                    Instruction::Standard(StandardGate::CX)
                ));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_vf2_preserves_control_flow_metadata_and_empty_else() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.set_global_phase(Parameter::from(0.25));
        circuit.measure(Qubit::new(0)).unwrap();

        let labeled_if = build_if_else_operation(
            ConditionView::new(Qubit::new(0), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }],
            None,
            Some("branch_label".into()),
        );
        append_test_operation(&mut circuit, labeled_if);

        let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        let mapped = vf2.execute(&circuit).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
        assert_eq!(mapped.global_phase(), Parameter::from(0.25));

        match &mapped.operations()[1] {
            Operation {
                instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(gate)),
                label,
                ..
            } => {
                assert_eq!(label.as_deref(), Some("branch_label"));
                assert_eq!(gate.condition().target, 1);
                assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
                assert!(gate.false_body().is_none());
            }
            _ => panic!("expected labeled mapped if_else operation"),
        }
    }

    #[test]
    fn test_preserve_measure_barrier_and_reset() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
        circuit.measure(Qubit::new(0)).unwrap();
        circuit.reset(Qubit::new(1)).unwrap();

        let topology = line_topology(&[0, 1]);
        let mapped =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
        let directives: Vec<Directive> = mapped
            .operations()
            .iter()
            .filter_map(|op| match op.instruction {
                Instruction::Directive(d) => Some(d),
                _ => None,
            })
            .collect();
        assert_eq!(
            directives,
            vec![Directive::Barrier, Directive::Measure, Directive::Reset]
        );
    }

    #[test]
    fn test_vf2_map_if_else_preserves_structure_and_condition() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        let true_body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        }];
        let false_body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        }];
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                true_body,
                Some(false_body),
            )
            .unwrap();

        let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        let mapped = vf2.execute(&circuit).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);

        assert!(matches!(
            mapped.operations()[1].instruction,
            Instruction::ControlFlowGate(ControlFlow::IfElse(_))
        ));
        match &mapped.operations()[1].instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
                assert_eq!(gate.true_body().len(), 1);
                assert_eq!(gate.false_body().unwrap().len(), 1);
                assert!(matches!(
                    gate.false_body().unwrap()[0].instruction,
                    Instruction::Standard(StandardGate::CX)
                ));
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_vf2_isomorphic_on_nested_if_else() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(4);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();

        let inner_true = vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(2), Qubit::new(3)],
            params: smallvec![],
            label: None,
        }];
        let nested_if = Operation {
            instruction: Instruction::ControlFlowGate(ControlFlow::if_else(
                ConditionView::new(Qubit::new(1), 1),
                inner_true,
                None,
            )),
            qubits: smallvec![Qubit::new(1), Qubit::new(2), Qubit::new(3)],
            params: smallvec![],
            label: None,
        };
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![nested_if],
                Some(vec![Operation {
                    instruction: Instruction::Standard(StandardGate::X),
                    qubits: smallvec![Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        assert!(vf2.is_subgraph_isomorphic(&circuit).unwrap());
    }

    #[test]
    fn test_vf2_maps_nested_if_else_inside_while_loop() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();

        let nested_if = build_if_else_operation(
            ConditionView::new(Qubit::new(1), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                params: smallvec![],
                label: None,
            }],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
            None,
        );
        circuit
            .while_loop(
                ConditionView::new(Qubit::new(0), 1),
                vec![
                    nested_if,
                    Operation {
                        instruction: Instruction::Standard(StandardGate::X),
                        qubits: smallvec![Qubit::new(2)],
                        params: smallvec![],
                        label: None,
                    },
                ],
            )
            .unwrap();

        let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        let mapped = vf2.execute(&circuit).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);

        match &mapped.operations()[2].instruction {
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                assert!(matches!(
                    gate.body()[0].instruction,
                    Instruction::ControlFlowGate(ControlFlow::IfElse(_))
                ));
            }
            _ => panic!("expected mapped while_loop with nested if_else"),
        }
    }

    #[test]
    fn test_map_with_vf2_sabre_routes_if_else_and_continuation() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }],
                Some(vec![Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            repeat_iterations: 0,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();

        assert_mapped_2q_edges(&mapped, &topology);
        assert!(matches!(
            mapped.operations()[1].instruction,
            Instruction::ControlFlowGate(ControlFlow::IfElse(_))
        ));
        assert!(count_swaps(&mapped) > 0);
        assert!(matches!(
            mapped.operations().last().unwrap().instruction,
            Instruction::Standard(StandardGate::CX)
        ));
    }

    #[test]
    fn test_map_with_vf2_sabre_routes_while_loop_and_continuation() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .while_loop(
                ConditionView::new(Qubit::new(0), 1),
                vec![
                    Operation {
                        instruction: Instruction::Standard(StandardGate::CX),
                        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                        params: smallvec![],
                        label: None,
                    },
                    Operation {
                        instruction: Instruction::Standard(StandardGate::CX),
                        qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                        params: smallvec![],
                        label: None,
                    },
                    Operation {
                        instruction: Instruction::Standard(StandardGate::CX),
                        qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                        params: smallvec![],
                        label: None,
                    },
                ],
            )
            .unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            repeat_iterations: 0,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();

        assert_mapped_2q_edges(&mapped, &topology);
        match &mapped.operations()[1].instruction {
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                assert!(gate.body().len() > 3);
            }
            _ => panic!("expected mapped while_loop operation"),
        }
        assert!(matches!(
            mapped.operations().last().unwrap().instruction,
            Instruction::Standard(StandardGate::CX)
        ));
    }

    #[test]
    fn test_sabre_preserves_symbolic_global_phase_and_directives_inside_control_flow() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        let theta = Parameter::try_from("theta").unwrap();
        circuit.set_global_phase(theta.clone());
        circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .while_loop(
                ConditionView::new(Qubit::new(0), 1),
                vec![
                    Operation {
                        instruction: Instruction::Directive(Directive::Measure),
                        qubits: smallvec![Qubit::new(1)],
                        params: smallvec![],
                        label: None,
                    },
                    Operation {
                        instruction: Instruction::Directive(Directive::Reset),
                        qubits: smallvec![Qubit::new(2)],
                        params: smallvec![],
                        label: None,
                    },
                    Operation {
                        instruction: Instruction::Standard(StandardGate::CX),
                        qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                        params: smallvec![],
                        label: None,
                    },
                ],
            )
            .unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            repeat_iterations: 0,
            seed: 9,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
        assert_eq!(mapped.global_phase(), theta);

        match &mapped.operations()[1].instruction {
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                assert_eq!(
                    collect_directives_recursive(gate.body()),
                    vec![Directive::Measure, Directive::Reset]
                );
            }
            _ => panic!("expected mapped while_loop operation"),
        }
    }

    #[test]
    fn test_sabre_maps_nested_while_loop_inside_if_else() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();

        let nested_while = build_while_loop_operation(
            ConditionView::new(Qubit::new(1), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }],
            None,
        );
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![nested_while],
                Some(vec![Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            repeat_iterations: 0,
            seed: 17,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);

        match &mapped.operations()[2].instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                assert!(matches!(
                    gate.true_body()[0].instruction,
                    Instruction::ControlFlowGate(ControlFlow::WhileLoop(_))
                ));
            }
            _ => panic!("expected mapped if_else with nested while_loop"),
        }
    }

    #[test]
    fn test_reject_unsupported_arity() {
        let mut circuit = Circuit::new(3);
        circuit
            .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
            .unwrap();

        let topology = line_topology(&[0, 1, 2, 3]);
        let err =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            CompileError::UnsupportedArity {
                arity: 3,
                op_index: 0
            }
        ));
    }

    #[test]
    fn test_invalid_fidelity_rejected() {
        let topology = line_topology(&[0, 1, 2]);
        let mut fidelity = FidelityMap::new();
        fidelity.insert((Qubit::new(0), Qubit::new(1)), 1.2);
        let err = Vf2Mapping::new(topology, Some(fidelity)).unwrap_err();
        assert!(matches!(err, CompileError::InvalidFidelity { .. }));
    }

    #[test]
    fn test_missing_fidelity_defaults_to_one() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20)]).unwrap();
        circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();

        let mut fidelity = FidelityMap::new();
        fidelity.insert((Qubit::new(0), Qubit::new(1)), 0.2);

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::Disabled,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, Some(&fidelity), &cfg).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
    }

    #[test]
    fn test_fidelity_pair_normalization() {
        let topology = line_topology(&[0, 1, 2]);
        let mut fidelity = FidelityMap::new();
        fidelity.insert((Qubit::new(2), Qubit::new(1)), 0.9);
        let _ = SabreMapping::new(topology, Some(fidelity), SabreConfig::default()).unwrap();
    }

    #[test]
    fn test_vf2_fast_path_no_overhead() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit =
            Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
        circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
        circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

        let mapped =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
        assert_eq!(mapped.operations().len(), circuit.operations().len());
        assert_eq!(count_swaps(&mapped), 0);
        assert_mapped_2q_edges(&mapped, &topology);
    }

    #[test]
    fn test_vf2_standalone_initial_layout_api() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit =
            Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
        circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
        circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let layout = vf2.find_initial_layout(&circuit).unwrap().unwrap();
        assert_eq!(layout.len(), 3);
    }

    #[test]
    fn test_vf2_find_initial_layout_fallback_top1() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

        let layout = vf2.find_initial_layout(&circuit).unwrap();
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().len(), 3);
    }

    #[test]
    fn test_vf2_map_remains_strict_no_fallback() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let mut vf2 = Vf2Mapping::new(topology, None).unwrap();
        let err = vf2.execute(&circuit).unwrap_err();
        assert!(matches!(err, CompileError::Vf2NoMapping));
    }

    #[test]
    fn test_vf2_candidates_topk_and_score_range() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(3);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
        circuit.x(Qubit::new(2)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let options = Vf2CandidateOptions {
            top_k: 3,
            ..Vf2CandidateOptions::default()
        };
        let candidates = vf2
            .find_initial_layout_candidates(&circuit, Some(options))
            .unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 3);
        for c in candidates {
            assert_eq!(c.logic2phy.len(), 3);
            assert_eq!(c.region.len(), 3);
            assert!((0.0..=1.0).contains(&c.score.total));
            assert!((0.0..=1.0).contains(&c.score.fidelity));
            assert!((0.0..=1.0).contains(&c.score.topology_fit));
            assert!((0.0..=1.0).contains(&c.score.gate_distribution));
        }
    }

    #[test]
    fn test_vf2_candidates_deterministic_order() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let options = Vf2CandidateOptions {
            top_k: 5,
            ..Vf2CandidateOptions::default()
        };
        let c1 = vf2
            .find_initial_layout_candidates(&circuit, Some(options.clone()))
            .unwrap();
        let c2 = vf2
            .find_initial_layout_candidates(&circuit, Some(options))
            .unwrap();

        let l1: Vec<Vec<u32>> = c1
            .iter()
            .map(|c| c.logic2phy.iter().map(Qubit::id).collect())
            .collect();
        let l2: Vec<Vec<u32>> = c2
            .iter()
            .map(|c| c.logic2phy.iter().map(Qubit::id).collect())
            .collect();
        assert_eq!(l1, l2);
    }

    #[test]
    fn test_vf2_candidates_topk_zero() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let options = Vf2CandidateOptions {
            top_k: 0,
            ..Vf2CandidateOptions::default()
        };
        let candidates = vf2
            .find_initial_layout_candidates(&circuit, Some(options))
            .unwrap();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_vf2_candidates_topk_effective_when_strict_isomorphic() {
        let topology = Topology::new(
            vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
            vec![
                (Qubit::new(0), Qubit::new(1), "CX".to_string()),
                (Qubit::new(1), Qubit::new(2), "CX".to_string()),
                (Qubit::new(2), Qubit::new(3), "CX".to_string()),
                (Qubit::new(3), Qubit::new(0), "CX".to_string()),
            ],
        );
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let options = Vf2CandidateOptions {
            top_k: 4,
            max_matches_per_subgraph: 16,
            ..Vf2CandidateOptions::default()
        };
        let candidates = vf2
            .find_initial_layout_candidates(&circuit, Some(options))
            .unwrap();
        assert!(candidates.len() > 1);
        assert!(candidates.len() <= 4);
    }

    #[test]
    fn test_vf2_candidates_respect_max_matches_per_subgraph() {
        let topology = Topology::new(
            vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
            vec![
                (Qubit::new(0), Qubit::new(1), "CX".to_string()),
                (Qubit::new(1), Qubit::new(2), "CX".to_string()),
                (Qubit::new(2), Qubit::new(3), "CX".to_string()),
                (Qubit::new(3), Qubit::new(0), "CX".to_string()),
            ],
        );
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let options = Vf2CandidateOptions {
            top_k: 8,
            max_matches_per_subgraph: 1,
            ..Vf2CandidateOptions::default()
        };
        let candidates = vf2
            .find_initial_layout_candidates(&circuit, Some(options))
            .unwrap();
        assert!(candidates.len() <= 1);
    }

    #[test]
    fn test_vf2_find_initial_layout_fallback_none_when_no_candidate() {
        let topology = Topology::new(vec![Qubit::new(0), Qubit::new(1)], vec![]);
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());
        let layout = vf2.find_initial_layout(&circuit).unwrap();
        assert!(layout.is_none());
    }

    #[test]
    fn test_vf2_isomorphic_on_dense_topology_non_induced_case() {
        let topology = Topology::new(
            vec![
                Qubit::new(0),
                Qubit::new(1),
                Qubit::new(2),
                Qubit::new(3),
                Qubit::new(4),
            ],
            vec![
                (Qubit::new(0), Qubit::new(1), "CX".to_string()),
                (Qubit::new(0), Qubit::new(2), "CX".to_string()),
                (Qubit::new(0), Qubit::new(3), "CX".to_string()),
                (Qubit::new(0), Qubit::new(4), "CX".to_string()),
                (Qubit::new(1), Qubit::new(2), "CX".to_string()),
                (Qubit::new(1), Qubit::new(3), "CX".to_string()),
                (Qubit::new(1), Qubit::new(4), "CX".to_string()),
                (Qubit::new(2), Qubit::new(3), "CX".to_string()),
                (Qubit::new(2), Qubit::new(4), "CX".to_string()),
                (Qubit::new(3), Qubit::new(4), "CX".to_string()),
            ],
        );
        let mut circuit = Circuit::new(5);
        circuit.cx(Qubit::new(2), Qubit::new(4)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(4)).unwrap();
        circuit.cx(Qubit::new(3), Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(4), Qubit::new(3)).unwrap();
        circuit.cx(Qubit::new(3), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(3)).unwrap();

        let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        assert!(vf2.is_subgraph_isomorphic(&circuit).unwrap());
        let mapped = vf2.execute(&circuit).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
    }

    #[test]
    fn test_policy_initial_only_routes_with_sabre() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            seed: 12345,
            initial_iterations: 2,
            repeat_iterations: 1,
            ..SabreConfig::default()
        };
        let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
    }

    #[test]
    fn test_sabre_fallback_and_state_exposure() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
        assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

        let mapped =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
        assert!(mapped.operations().len() > circuit.operations().len());
        assert_mapped_2q_edges(&mapped, &topology);

        let mut sabre = SabreMapping::new(topology, None, SabreConfig::default()).unwrap();
        let _ = sabre.execute(&circuit).unwrap();
        assert_eq!(sabre.logic2phy.len(), circuit.qubits().len());
    }

    #[test]
    fn test_output_uses_only_physical_qubits_in_use() {
        let topology = line_topology(&[0, 1, 2, 3, 4]);
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let mapped =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
        assert_eq!(mapped.qubits().len(), 2);
        assert_mapped_2q_edges(&mapped, &topology);
    }

    #[test]
    fn test_sabre_determinism_with_fixed_seed() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            seed: 12345,
            initial_iterations: 3,
            repeat_iterations: 2,
            swap_iterations: 3,
            ..SabreConfig::default()
        };

        let mut sabre1 = SabreMapping::new(topology.clone(), None, cfg.clone()).unwrap();
        let mut sabre2 = SabreMapping::new(topology, None, cfg).unwrap();

        let out1 = sabre1.execute(&circuit).unwrap();
        let out2 = sabre2.execute(&circuit).unwrap();
        assert_eq!(fingerprint(&out1), fingerprint(&out2));
    }

    #[test]
    fn test_sabre_control_flow_determinism_with_fixed_seed() {
        let topology = line_topology(&[0, 1, 2]);
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }],
                Some(vec![Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

        let cfg = SabreConfig {
            vf2_policy: Vf2Policy::InitialOnly,
            seed: 12345,
            initial_iterations: 3,
            repeat_iterations: 0,
            swap_iterations: 3,
            ..SabreConfig::default()
        };

        let mut sabre1 = SabreMapping::new(topology.clone(), None, cfg.clone()).unwrap();
        let mut sabre2 = SabreMapping::new(topology, None, cfg).unwrap();

        let out1 = sabre1.execute(&circuit).unwrap();
        let out2 = sabre2.execute(&circuit).unwrap();
        assert_eq!(fingerprint(&out1), fingerprint(&out2));
    }

    #[test]
    fn test_non_contiguous_qubit_ids_supported() {
        let topology = line_topology(&[100, 200, 300, 400]);
        let mut circuit =
            Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(30), Qubit::new(70)]).unwrap();
        circuit.cx(Qubit::new(10), Qubit::new(30)).unwrap();
        circuit.cx(Qubit::new(30), Qubit::new(70)).unwrap();

        let mapped =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();

        let topo_set: HashSet<Qubit> = topology.qubits().collect();
        for q in mapped.qubits() {
            assert!(topo_set.contains(&q));
        }
        assert_mapped_2q_edges(&mapped, &topology);
    }

    fn test_circuit(width: usize) -> Circuit {
        let mut circuit = Circuit::new(width);
        if width > 1 {
            circuit.cx(Qubit::new(0), Qubit::new((width as u32) - 1)).unwrap();
        }
        circuit
    }

    fn fast_ga_config(seed: i64) -> GaConfig {
        let mut sabre_config = SabreConfig::default();
        sabre_config.repeat_iterations = 0; 
        sabre_config.seed = seed;

        GaConfig {
            population: 4,
            update_iters: 2,
            seed,
            sabre_config,
            ..GaConfig::default()
        }
    }


    #[test]
    fn test_map_with_ga_basic_success() {
        
        let topology = line_topology(&[0, 1, 2, 3]);
        let circuit = test_circuit(3);
        let config = fast_ga_config(42);

        let result = map_with_ga(&circuit, &topology, &config, None, None);
        
        assert!(result.is_ok(), "GA mapping failed in basic scenario");
        let mapped_circuit = result.unwrap();
        
        assert!(mapped_circuit.operations().len() >= circuit.operations().len());
    }

    #[test]
    fn test_map_with_ga_invalid_qubits_avoidance() {
        
        let topology = line_topology(&[0, 1, 2, 3, 4, 5]);
        let circuit = test_circuit(3); 
        
        let mut invalid_qubits = HashSet::new();
        invalid_qubits.insert(2);

        let config = fast_ga_config(42);
        
        let result = map_with_ga(&circuit, &topology, &config, None, Some(invalid_qubits));
        assert!(result.is_ok(), "Failed to find mapping in partitioned topology");
        
        let mapped_circuit = result.unwrap();
        
        for op in mapped_circuit.operations() {
            for q in &op.qubits {
                let id = q.id();
                assert!(
                    id == 3 || id == 4 || id == 5,
                    "Algorithm mapped to an invalid or disconnected qubit: {}",
                    id
                );
            }
        }
    }

    #[test]
    fn test_map_with_ga_invalid_qubits_causes_too_small() {
        
        let topology = line_topology(&[0, 1, 2, 3]);
        let circuit = test_circuit(3); 
        
        let mut invalid_qubits = HashSet::new();
        invalid_qubits.insert(1);
        invalid_qubits.insert(2);

        let config = fast_ga_config(42);
        let result = map_with_ga(&circuit, &topology, &config, None, Some(invalid_qubits));
        
        assert!(
            matches!(result, Err(CompileError::TopologyTooSmall { .. })),
            "Expected TopologyTooSmall error due to fragmentation"
        );
    }

    #[test]
    fn test_map_with_ga_fidelity_map_integration() {
        
        let topology = line_topology(&[0, 1, 2, 3]);
        let circuit = test_circuit(2);
        
        let mut fidelity_map = HashMap::new();
        fidelity_map.insert((Qubit::new(0), Qubit::new(1)), 0.5);
        fidelity_map.insert((Qubit::new(1), Qubit::new(2)), 0.99);
        fidelity_map.insert((Qubit::new(2), Qubit::new(3)), 0.99);

        let config = fast_ga_config(1024);

        let result = map_with_ga(&circuit, &topology, &config, Some(&fidelity_map), None);
        assert!(result.is_ok(), "Mapping failed with fidelity map provided");
    }

    #[test]
    fn test_map_with_ga_determinism() {
        let topology = line_topology(&[0, 1, 2, 3, 4]);
        let circuit = test_circuit(4);
        
        let seed = 999;
        let config = fast_ga_config(seed);

        let result1 = map_with_ga(&circuit, &topology, &config, None, None).unwrap();
        let result2 = map_with_ga(&circuit, &topology, &config, None, None).unwrap();
        
        let fp1 = fingerprint(&result1);
        let fp2 = fingerprint(&result2);

        assert_eq!(
            fp1, fp2, 
            "GA mapping should be deterministic given the same seed. Run 1: {:?}, Run 2: {:?}", 
            fp1, fp2
        );
    }

    #[test]
    fn test_map_with_ga_ghz_circuit_on_star_topology() {
        
        let topology = Topology::new(
            vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3), Qubit::new(4)],
            vec![
                (Qubit::new(0), Qubit::new(1), "CX".to_string()),
                (Qubit::new(0), Qubit::new(2), "CX".to_string()),
                (Qubit::new(0), Qubit::new(3), "CX".to_string()),
                (Qubit::new(0), Qubit::new(4), "CX".to_string()),
            ],
        );

        let mut circuit = Circuit::new(5);
        circuit.h(Qubit::new(0)).unwrap();
        for i in 0..4 {
            circuit.cx(Qubit::new(i as u32), Qubit::new((i + 1) as u32)).unwrap();
        }

        let config = fast_ga_config(100);

        let result = map_with_ga(&circuit, &topology, &config, None, None);
        assert!(result.is_ok(), "GA failed to map GHZ circuit on star topology");
        
        let mapped = result.unwrap();
        
        assert_mapped_2q_edges(&mapped, &topology);
        
        assert!(mapped.operations().len() == 6);
    }

    #[test]
    fn test_map_with_ga_all_to_all_heavy_routing() {
        
        let topology = line_topology(&[0, 1, 2, 3, 4]);

        let mut circuit = Circuit::new(5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                circuit.cx(Qubit::new(i as u32), Qubit::new(j as u32)).unwrap();
            }
        }

        let config = GaConfig {
            population: 10,   
            update_iters: 5,  
            seed: 2024,
            ..fast_ga_config(2024)
        };

        let result = map_with_ga(&circuit, &topology, &config, None, None);
        assert!(result.is_ok(), "GA failed to map all-to-all circuit");
        
        let mapped = result.unwrap();
        assert_mapped_2q_edges(&mapped, &topology);
        
        println!("All-to-All 5-qubit mapped operations count: {}", mapped.operations().len());

    }
}
