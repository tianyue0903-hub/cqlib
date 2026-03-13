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

//! Shared compile-time gate graph utilities.
//!
//! This module centralizes gate-node construction from preprocessed circuits and
//! exposes two edge semantics over the same nodes:
//! - [`DependencyView`], used by routing/mapping (SABRE).
//! - [`CommutationView`], used by template matching/optimization.
//!
//! The node model stores instruction metadata plus resolved parameter handles so
//! downstream passes can choose strict parameter matching and conservative
//! symbolic handling.

use crate::circuit::param::CircuitParam;
use crate::circuit::{Operation, Parameter};
use crate::compile::error::CompileError;
use crate::compile::prepared::{PreparedCircuit, PreparedOperation};
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;
use std::collections::{HashSet, VecDeque};

/// Resolved parameter handle used in compile graph nodes.
///
/// Fixed values can be consumed by matrix-based checks, while symbolic
/// parameters are carried for exact-structure matching.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResolvedParam {
    /// Numerically resolved scalar parameter.
    Fixed(f64),
    /// Symbolic parameter expression.
    Symbolic(Parameter),
}

/// One gate node in the shared compile graph.
#[derive(Debug, Clone)]
pub(crate) struct GateNode {
    /// Zero-based node id in topological order.
    pub(crate) node_id: usize,
    /// Operation index in preprocessed circuit order.
    pub(crate) op_index: usize,
    /// Original operation payload.
    pub(crate) op: Operation,
    /// Logical-qubit indices for this operation.
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
    /// Resolved parameters aligned with `op.params`.
    pub(crate) resolved_params: SmallVec<[ResolvedParam; 3]>,
}

/// Shared gate graph used by compile passes.
#[derive(Debug, Clone)]
pub(crate) struct GateGraph {
    nodes: Vec<GateNode>,
}

impl GateGraph {
    /// Builds gate nodes from a preprocessed compile circuit.
    pub(crate) fn from_prepared(prepared: &PreparedCircuit) -> Result<Self, CompileError> {
        let mut nodes = Vec::with_capacity(prepared.operations.len());
        for (node_id, prep_op) in prepared.operations.iter().enumerate() {
            let resolved_params = resolve_params(prep_op, prepared.parameters.as_slice())?;
            nodes.push(GateNode {
                node_id,
                op_index: node_id,
                op: prep_op.op.clone(),
                logical_qubits: prep_op.logical_qubits.clone(),
                resolved_params,
            });
        }
        Ok(Self { nodes })
    }

    /// Returns one node by zero-based id.
    pub(crate) fn node(&self, node_id: usize) -> Option<&GateNode> {
        self.nodes.get(node_id)
    }

    /// Returns graph node count.
    pub(crate) fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Builds routing dependency view from shared nodes.
    pub(crate) fn dependency_view(&self, logical_width: usize) -> DependencyView {
        let mut nodes: Vec<DependencyNode> = Vec::with_capacity(self.nodes.len());
        let mut front_layer = HashSet::new();
        let mut initial_single_qubit_ops = Vec::new();
        let mut pre_gate: Vec<Option<usize>> = vec![None; logical_width];

        for gateid_zero in 0..self.nodes.len() {
            let gateid = gateid_zero + 1;
            let node = &self.nodes[gateid_zero];

            let mut dep_node = DependencyNode {
                op_index: node.op_index,
                attach_ids: Vec::new(),
                next_ids: Vec::new(),
                logical_qubits: node.logical_qubits.clone(),
                indegree: 0,
            };

            if dep_node.logical_qubits.len() == 1 {
                let u = dep_node.logical_qubits[0];
                if let Some(prev) = pre_gate[u] {
                    nodes[prev - 1].attach_ids.push(gateid);
                } else {
                    initial_single_qubit_ops.push((dep_node.op_index, u));
                }
                nodes.push(dep_node);
                continue;
            }

            for &u in &dep_node.logical_qubits {
                if let Some(prev) = pre_gate[u]
                    && !nodes[prev - 1].next_ids.contains(&gateid)
                {
                    nodes[prev - 1].next_ids.push(gateid);
                    dep_node.indegree += 1;
                }
            }

            for &u in &dep_node.logical_qubits {
                pre_gate[u] = Some(gateid);
            }

            if dep_node.indegree == 0 {
                front_layer.insert(gateid);
            }

            nodes.push(dep_node);
        }

        DependencyView {
            nodes,
            front_layer,
            initial_single_qubit_ops,
        }
    }

    /// Builds non-commutation DAG view from shared nodes.
    pub(crate) fn commutation_view(&self) -> Result<CommutationView, CompileError> {
        let n = self.nodes.len();
        let mut predecessors = vec![Vec::<usize>::new(); n];
        let mut successors = vec![Vec::<usize>::new(); n];
        let mut reachable = vec![false; n];

        for idx in 0..n {
            for item in reachable.iter_mut().take(idx) {
                *item = true;
            }
            for prev in (0..idx).rev() {
                if !reachable[prev] {
                    continue;
                }
                let commute = operations_commute(&self.nodes[prev], &self.nodes[idx])?;
                if commute {
                    continue;
                }

                predecessors[idx].push(prev);
                successors[prev].push(idx);

                // Remove transitive predecessors of `prev` from current reachability.
                let mut queue = VecDeque::new();
                for &p in &predecessors[prev] {
                    queue.push_back(p);
                }
                while let Some(node) = queue.pop_front() {
                    if !reachable[node] {
                        continue;
                    }
                    reachable[node] = false;
                    for &p in &predecessors[node] {
                        queue.push_back(p);
                    }
                }
            }
        }

        for preds in &mut predecessors {
            preds.sort_unstable();
        }
        for succs in &mut successors {
            succs.sort_unstable();
        }

        Ok(CommutationView {
            predecessors,
            successors,
        })
    }
}

/// One node in routing dependency view.
#[derive(Debug, Clone)]
pub(crate) struct DependencyNode {
    /// Operation index in preprocessed circuit order.
    pub(crate) op_index: usize,
    /// Single-qubit operations attached to this gate.
    pub(crate) attach_ids: Vec<usize>,
    /// Successor 2q dependency edges.
    pub(crate) next_ids: Vec<usize>,
    /// Logical qubits used by this gate.
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
    /// Number of unresolved predecessors.
    pub(crate) indegree: usize,
}

/// Routing dependency view built from shared graph nodes.
#[derive(Debug, Clone)]
pub(crate) struct DependencyView {
    /// Dependency nodes in one-based gate id order.
    pub(crate) nodes: Vec<DependencyNode>,
    /// Initial executable 2q gate ids.
    pub(crate) front_layer: HashSet<usize>,
    /// Initial 1q operations with no predecessor.
    pub(crate) initial_single_qubit_ops: Vec<(usize, usize)>,
}

/// Non-commutation DAG view for template matching.
#[derive(Debug, Clone)]
pub(crate) struct CommutationView {
    /// Immediate predecessors for each zero-based node id.
    pub(crate) predecessors: Vec<Vec<usize>>,
    /// Immediate successors for each zero-based node id.
    pub(crate) successors: Vec<Vec<usize>>,
}

impl CommutationView {
    /// Returns immediate successors of a node.
    pub(crate) fn successors(&self, node_id: usize) -> &[usize] {
        self.successors.get(node_id).map_or(&[], Vec::as_slice)
    }

    /// Returns all transitive successors of a node.
    pub(crate) fn all_successors(&self, start: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut visited = HashSet::from([start]);

        while let Some(node) = queue.pop_front() {
            for &succ in self.successors(node) {
                if visited.insert(succ) {
                    result.insert(succ);
                    queue.push_back(succ);
                }
            }
        }
        result
    }

    /// Returns whether `target` is reachable from `source`.
    pub(crate) fn is_reachable(&self, source: usize, target: usize) -> bool {
        if source == target {
            return true;
        }
        if self
            .predecessors
            .get(target)
            .is_some_and(|preds| preds.contains(&source))
        {
            return true;
        }
        self.all_successors(source).contains(&target)
    }
}

/// Resolves operation parameters into fixed or symbolic handles.
fn resolve_params(
    prep_op: &PreparedOperation,
    parameter_pool: &[Parameter],
) -> Result<SmallVec<[ResolvedParam; 3]>, CompileError> {
    let mut resolved = SmallVec::<[ResolvedParam; 3]>::with_capacity(prep_op.op.params.len());
    for param in &prep_op.op.params {
        match param {
            CircuitParam::Fixed(v) => resolved.push(ResolvedParam::Fixed(*v)),
            CircuitParam::Index(index) => {
                let idx = *index as usize;
                let Some(symbolic) = parameter_pool.get(idx) else {
                    return Err(CompileError::Internal(format!(
                        "operation references missing parameter index {}",
                        idx
                    )));
                };
                match symbolic.evaluate(&None) {
                    Ok(v) => resolved.push(ResolvedParam::Fixed(v)),
                    Err(_) => resolved.push(ResolvedParam::Symbolic(symbolic.clone())),
                }
            }
        }
    }
    Ok(resolved)
}

/// Computes operation commutation under strict symbolic policy.
///
/// When any shared-qubit operation carries symbolic parameters, this function
/// conservatively returns `false` to avoid unsound optimizations.
fn operations_commute(a: &GateNode, b: &GateNode) -> Result<bool, CompileError> {
    let set_a: HashSet<usize> = a.logical_qubits.iter().copied().collect();
    let set_b: HashSet<usize> = b.logical_qubits.iter().copied().collect();
    let overlap: HashSet<usize> = set_a.intersection(&set_b).copied().collect();
    if overlap.is_empty() {
        return Ok(true);
    }

    let Some(a_matrix) = op_matrix(a)? else {
        return Ok(false);
    };
    let Some(b_matrix) = op_matrix(b)? else {
        return Ok(false);
    };

    let mut combined_qubits: Vec<usize> = set_a.union(&set_b).copied().collect();
    combined_qubits.sort_unstable();
    let total_qubits = combined_qubits.len();
    if total_qubits == 0 {
        return Ok(true);
    }

    let mut pos_a = Vec::with_capacity(a.logical_qubits.len());
    for &q in &a.logical_qubits {
        let Some(pos) = combined_qubits.iter().position(|&x| x == q) else {
            return Err(CompileError::Internal(
                "failed to map operation qubit into combined set".to_string(),
            ));
        };
        pos_a.push(pos);
    }

    let mut pos_b = Vec::with_capacity(b.logical_qubits.len());
    for &q in &b.logical_qubits {
        let Some(pos) = combined_qubits.iter().position(|&x| x == q) else {
            return Err(CompileError::Internal(
                "failed to map operation qubit into combined set".to_string(),
            ));
        };
        pos_b.push(pos);
    }

    let expand_a = expand_unitary(&a_matrix, &pos_a, total_qubits)?;
    let expand_b = expand_unitary(&b_matrix, &pos_b, total_qubits)?;
    let ab = expand_a.dot(&expand_b);
    let ba = expand_b.dot(&expand_a);
    Ok(approx_matrix_eq(&ab, &ba, 1e-10))
}

/// Returns resolved unitary matrix for an operation.
fn op_matrix(node: &GateNode) -> Result<Option<Array2<Complex64>>, CompileError> {
    let mut params = Vec::with_capacity(node.resolved_params.len());
    for p in &node.resolved_params {
        match p {
            ResolvedParam::Fixed(v) => params.push(*v),
            ResolvedParam::Symbolic(_) => return Ok(None),
        }
    }

    let Some(matrix) = node.op.instruction.matrix(&params) else {
        return Ok(None);
    };
    Ok(Some(matrix.into_owned()))
}

/// Expands a k-qubit gate matrix into an `n`-qubit matrix space.
fn expand_unitary(
    matrix: &Array2<Complex64>,
    gate_positions: &[usize],
    total_qubits: usize,
) -> Result<Array2<Complex64>, CompileError> {
    if gate_positions.is_empty() {
        return Err(CompileError::Internal(
            "cannot expand unitary with empty gate positions".to_string(),
        ));
    }
    if total_qubits < gate_positions.len() {
        return Err(CompileError::Internal(
            "total_qubits is smaller than gate positions".to_string(),
        ));
    }

    let expected = 1usize << gate_positions.len();
    if matrix.nrows() != expected || matrix.ncols() != expected {
        return Err(CompileError::Internal(format!(
            "gate matrix shape {}x{} does not match {} qubits",
            matrix.nrows(),
            matrix.ncols(),
            gate_positions.len()
        )));
    }

    let dim = 1usize << total_qubits;
    let mut xor_mask = dim - 1;
    for &pos in gate_positions {
        if pos >= total_qubits {
            return Err(CompileError::Internal(format!(
                "gate position {} out of range {}",
                pos, total_qubits
            )));
        }
        xor_mask ^= 1usize << (total_qubits - 1 - pos);
    }

    let mut expand_vec = vec![0usize; dim];
    for (idx, item) in expand_vec.iter_mut().enumerate().take(dim) {
        let mut mapped = 0usize;
        for (q_idx, &pos) in gate_positions.iter().enumerate() {
            let bit = total_qubits - 1 - pos;
            if (idx & (1usize << bit)) != 0 {
                mapped |= 1usize << (gate_positions.len() - 1 - q_idx);
            }
        }
        *item = mapped;
    }

    let mut expanded = Array2::<Complex64>::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            if (i & xor_mask) == (j & xor_mask) {
                expanded[[i, j]] = matrix[[expand_vec[i], expand_vec[j]]];
            }
        }
    }
    Ok(expanded)
}

/// Compares two matrices with absolute tolerance.
fn approx_matrix_eq(lhs: &Array2<Complex64>, rhs: &Array2<Complex64>, eps: f64) -> bool {
    if lhs.dim() != rhs.dim() {
        return false;
    }
    lhs.iter()
        .zip(rhs.iter())
        .all(|(a, b)| (*a - *b).norm() <= eps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::gate::{Instruction, StandardGate};
    use crate::circuit::{Circuit, Parameter, Qubit};
    use crate::compile::prepared::preprocess_circuit;

    /// Builds a small circuit used by graph tests.
    fn sample_circuit() -> Circuit {
        let mut circuit = Circuit::new(3);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.h(Qubit::new(2)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit
    }

    /// Checks dependency view reproduces SABRE-style gate dependencies.
    #[test]
    fn test_dependency_view_builds_expected_layers() {
        let prepared = preprocess_circuit(&sample_circuit()).unwrap();
        let graph = GateGraph::from_prepared(&prepared).unwrap();
        let dep = graph.dependency_view(prepared.logical_qubits.len());

        assert_eq!(dep.nodes.len(), 4);
        assert!(dep.front_layer.contains(&2));
        assert_eq!(dep.front_layer.len(), 1);
        assert_eq!(dep.initial_single_qubit_ops, vec![(0, 0), (2, 2)]);
        assert_eq!(dep.nodes[1].next_ids, vec![4]);
        assert!(dep.nodes[0].attach_ids.is_empty());
    }

    /// Checks commutation view keeps non-overlapping gates disconnected.
    #[test]
    fn test_commutation_view_skips_disjoint_gates() {
        let prepared = preprocess_circuit(&sample_circuit()).unwrap();
        let graph = GateGraph::from_prepared(&prepared).unwrap();
        let comm = graph.commutation_view().unwrap();

        // H(q2) and CX(q0,q1) are disjoint and should commute.
        assert!(!comm.is_reachable(1, 2));
        assert!(!comm.is_reachable(2, 1));
    }

    /// Checks symbolic parameters are treated conservatively as non-commuting.
    #[test]
    fn test_symbolic_parameter_is_non_commuting() {
        let mut circuit = Circuit::new(1);
        circuit
            .append(
                Instruction::Standard(StandardGate::RZ),
                [Qubit::new(0)],
                [crate::circuit::param::ParameterValue::Param(
                    Parameter::symbol("theta"),
                )],
                None,
            )
            .unwrap();
        circuit
            .append(
                Instruction::Standard(StandardGate::RZ),
                [Qubit::new(0)],
                [crate::circuit::param::ParameterValue::Fixed(0.1)],
                None,
            )
            .unwrap();

        let prepared = preprocess_circuit(&circuit).unwrap();
        let graph = GateGraph::from_prepared(&prepared).unwrap();
        let comm = graph.commutation_view().unwrap();
        assert!(comm.is_reachable(0, 1));
    }
}
