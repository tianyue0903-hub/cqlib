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

pub mod sabre;
pub mod vf2;

pub use sabre::{SabreConfig, SabreMapping, Vf2Policy};
pub use vf2::Vf2Mapping;

use crate::circuit::dag::Terminator;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use crate::circuit::param::ParameterValue;
use crate::circuit::{Circuit, CircuitDag, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::device::Topology;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

/// Optional fidelity map keyed by physical qubit edge.
///
/// The mapping is treated as undirected by normalizing `(u, v)` and `(v, u)`
/// to the same canonical key.
pub type FidelityMap = HashMap<(Qubit, Qubit), f64>;

#[derive(Debug, Clone)]
pub(crate) struct PreparedOperation {
    pub(crate) op: Operation,
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedCircuit {
    pub(crate) logical_qubits: Vec<Qubit>,
    pub(crate) parameters: Vec<Parameter>,
    pub(crate) operations: Vec<PreparedOperation>,
}

#[derive(Debug, Clone)]
pub(crate) struct TopologyAdapter {
    pub(crate) physical_qubits: Vec<Qubit>,
    pub(crate) neighbors: Vec<Vec<usize>>,
    pub(crate) adj_matrix: Vec<Vec<bool>>,
    pub(crate) dist: Vec<Vec<u32>>,
    pub(crate) fidelity: Vec<Vec<f64>>,
    pub(crate) largest_component: Vec<usize>,
}

impl TopologyAdapter {
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

    pub(crate) fn num_qubits(&self) -> usize {
        self.physical_qubits.len()
    }

    pub(crate) fn is_adjacent(&self, u_idx: usize, v_idx: usize) -> bool {
        self.adj_matrix[u_idx][v_idx]
    }

    pub(crate) fn edge_fidelity(&self, u_idx: usize, v_idx: usize) -> f64 {
        self.fidelity[u_idx][v_idx]
    }
}

pub(crate) fn normalize_index_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

pub(crate) fn is_cx(op: &Operation) -> bool {
    matches!(op.instruction, Instruction::Standard(StandardGate::CX))
}

pub(crate) fn map_operation_qubits(op: &Operation, mapped_qubits: &[Qubit]) -> Operation {
    let mut mapped = op.clone();
    mapped.qubits = mapped_qubits.iter().copied().collect();
    mapped
}

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

pub(crate) fn build_output_circuit(
    mapped_ops: &[Operation],
    parameter_pool: &[Parameter],
) -> Result<Circuit, CompileError> {
    let mut used_set: HashSet<Qubit> = HashSet::new();
    for op in mapped_ops {
        for &q in &op.qubits {
            used_set.insert(q);
        }
    }

    let mut used_qubits: Vec<Qubit> = used_set.into_iter().collect();
    used_qubits.sort_by_key(Qubit::id);

    let mut output = Circuit::from_qubits(used_qubits)?;
    for op in mapped_ops {
        append_operation(&mut output, op, parameter_pool)?;
    }

    Ok(output)
}

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

    if !matches!(
        block.terminator,
        None | Some(Terminator::Return)
    ) {
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
    use crate::circuit::gate::control_flow::ConditionView;
    use crate::circuit::gate::{Instruction, StandardGate};
    use crate::circuit::{Circuit, Operation, Qubit};
    use crate::compile::error::CompileError;
    use smallvec::smallvec;
    use std::collections::HashSet;

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

    fn assert_mapped_2q_edges(mapped: &Circuit, topology: &Topology) {
        for op in mapped.operations() {
            if op.qubits.len() == 2 {
                assert!(
                    connected_undirected(topology, op.qubits[0], op.qubits[1]),
                    "2q op is not on a topology edge: {:?}",
                    op.qubits
                );
            }
        }
    }

    fn count_swaps(circuit: &Circuit) -> usize {
        circuit
            .operations()
            .iter()
            .filter(|op| matches!(op.instruction, Instruction::Standard(StandardGate::SWAP)))
            .count()
    }

    fn fingerprint(circuit: &Circuit) -> Vec<String> {
        circuit
            .operations()
            .iter()
            .map(|op| {
                let mut qids: Vec<u32> = op.qubits.iter().map(Qubit::id).collect();
                qids.sort_unstable();
                format!("{:?}:{:?}", op.instruction, qids)
            })
            .collect()
    }

    #[test]
    fn test_module_exports_compile_and_device() {
        let _cfg = crate::compile::SabreConfig::default();
        let _topology = crate::device::Topology::new(vec![Qubit::new(0)], vec![]);
    }

    #[test]
    fn test_reject_control_flow() {
        let mut circuit = Circuit::new(2);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        circuit.measure(q0).unwrap();

        let true_body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![q1],
            params: smallvec![],
            label: None,
        }];
        circuit
            .if_else(ConditionView::new(q0, 1), true_body, None)
            .unwrap();

        let topology = line_topology(&[0, 1, 2]);
        let err =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
        assert!(matches!(err, CompileError::UnsupportedControlFlow));
    }

    #[test]
    fn test_reject_directive_and_delay() {
        let mut circuit = Circuit::new(1);
        circuit.barrier(vec![Qubit::new(0)]).unwrap();
        let topology = line_topology(&[0, 1]);
        let err =
            map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            CompileError::UnsupportedInstruction {
                instruction: _,
                op_index: _
            }
        ));
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
        let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
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
        let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
        circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
        circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

        let vf2 = Vf2Mapping::new(topology, None).unwrap();
        let layout = vf2.find_initial_layout(&circuit).unwrap().unwrap();
        assert_eq!(layout.len(), 3);
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
    fn test_non_contiguous_qubit_ids_supported() {
        let topology = line_topology(&[100, 200, 300, 400]);
        let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(30), Qubit::new(70)]).unwrap();
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
}
