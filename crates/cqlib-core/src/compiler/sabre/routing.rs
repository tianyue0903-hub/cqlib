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

use super::dag::{SabreControlFlow, SabreDag, SabreNodeKind};
use super::heuristic::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
use super::layer::Layer;
use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, Instruction, Operation, ParameterValue, Qubit,
};
use crate::compiler::CompilerError;
use crate::compiler::transform::layout::{PhysicalLayoutGraph, build_physical_layout_graph};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::token_swapper::token_swapper;
use smallvec::{SmallVec, smallvec};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const CONTROL_FLOW_EPILOGUE_TRIALS: usize = 4;

/// Routed circuit and layout metadata produced by [`sabre_route`].
#[derive(Debug, Clone)]
pub struct SabreRoutingResult {
    /// Physical circuit with inserted SWAP operations.
    pub circuit: Circuit,
    /// Initial logical-to-physical layout used by the selected trial.
    pub initial_layout: Layout,
    /// Final logical-to-physical layout after all routed operations.
    pub final_layout: Layout,
    /// Number of inserted SWAP operations, including control-flow epilogues.
    pub swap_count: usize,
    /// Diagnostics describing routing search behavior.
    pub diagnostics: SabreRoutingDiagnostics,
}

/// Diagnostics emitted by SABRE routing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SabreRoutingDiagnostics {
    /// Number of routing trials evaluated.
    pub trials_evaluated: usize,
    /// Zero-based index of the selected routing trial.
    pub selected_trial_index: usize,
    /// Number of times the shortest-path fallback was used.
    pub fallback_count: usize,
    /// Number of recursively routed control-flow blocks.
    pub control_flow_blocks_routed: usize,
    /// ASAP two-qubit depth of the selected routed operation stream.
    pub two_qubit_depth: usize,
    /// Total number of operations in the selected routed operation stream.
    pub operation_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct TrialResult {
    pub(crate) operations: Vec<Operation>,
    pub(crate) final_layout: Layout,
    pub(crate) swap_count: usize,
    pub(crate) fallback_count: usize,
    pub(crate) control_flow_blocks_routed: usize,
    pub(crate) quality: TrialQuality,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TrialQuality {
    pub(crate) swap_count: usize,
    pub(crate) two_qubit_depth: usize,
    pub(crate) operation_count: usize,
}

/// Routes `circuit` from `initial_layout` by inserting SWAP operations.
///
/// The returned circuit uses physical qubit identifiers as its circuit qubits.
/// Two-qubit operations in the routed circuit are adjacent in the usable
/// physical topology.  Control-flow bodies are routed recursively and restored
/// to their entry layout before leaving the block.
pub fn sabre_route(
    circuit: &Circuit,
    device: &Device,
    initial_layout: &Layout,
    config: &SabreConfig,
) -> Result<SabreRoutingResult, CompilerError> {
    validate_config(config)?;
    let physical = build_physical_layout_graph(device)?;
    let target = RoutingTarget::from_physical(&physical)?;
    let logical_qubits = circuit
        .qubits()
        .into_iter()
        .map(LogicalQubit::from_qubit)
        .collect::<Vec<_>>();
    let initial_layout = normalize_initial_layout(&logical_qubits, &target, initial_layout)?;
    let sabre = SabreDag::from_operations(circuit.operations())?;
    validate_reachable_interactions(&sabre, &target, &initial_layout)?;

    let trial_results = trial_seeds(config.seed, config.routing_trials)
        .into_par_iter()
        .enumerate()
        .map(|(index, seed)| {
            route_trial_unchecked(&sabre, &target, &initial_layout, &config.heuristic, seed)
                .map(|result| (index, result))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?;
    let (best_index, best) = trial_results
        .into_iter()
        .min_by(|(left_index, left), (right_index, right)| {
            compare_trial_quality(
                config.trial_objective,
                left.quality,
                *left_index,
                right.quality,
                *right_index,
            )
        })
        .expect("routing_trials is validated to be non-zero");

    let mut routed = Circuit::from_qubits(
        target
            .physical_qubits
            .iter()
            .copied()
            .map(PhysicalQubit::qubit)
            .collect(),
    )?;
    routed.set_global_phase(circuit.global_phase());
    for operation in &best.operations {
        append_operation_to_circuit(&mut routed, operation, circuit)?;
    }

    Ok(SabreRoutingResult {
        circuit: routed,
        initial_layout,
        final_layout: best.final_layout,
        swap_count: best.swap_count,
        diagnostics: SabreRoutingDiagnostics {
            trials_evaluated: config.routing_trials,
            selected_trial_index: best_index,
            fallback_count: best.fallback_count,
            control_flow_blocks_routed: best.control_flow_blocks_routed,
            two_qubit_depth: best.quality.two_qubit_depth,
            operation_count: best.quality.operation_count,
        },
    })
}

pub(crate) fn validate_config(config: &SabreConfig) -> Result<(), CompilerError> {
    if config.layout_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_trials must be greater than zero".to_string(),
        ));
    }
    if config.routing_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre routing_trials must be greater than zero".to_string(),
        ));
    }
    if config.layout_scoring_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_scoring_trials must be greater than zero".to_string(),
        ));
    }
    config.heuristic.validate()
}

pub(crate) fn route_trial(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    validate_reachable_interactions(sabre, target, initial_layout)?;
    route_trial_unchecked(sabre, target, initial_layout, heuristic, seed)
}

pub(crate) fn route_trial_unchecked(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    let mut output = TrialOutput::new(seed);
    let mut state = RoutingState::new(sabre, target, initial_layout.clone(), heuristic, seed);

    for operation in &sabre.initial {
        output
            .operations
            .push(map_operation(operation, &state.layout)?);
    }

    state.update_route(
        sabre,
        target,
        heuristic,
        &mut output,
        &sabre.first_layer,
        None,
    )?;
    state.populate_extended_set(sabre);

    let mut routable_nodes = Vec::with_capacity(2);
    let mut search_steps_since_decay_reset = 0usize;
    while !state.front_layer.is_empty() {
        let mut current_swaps = Vec::new();
        while routable_nodes.is_empty() && current_swaps.len() <= heuristic.attempt_limit {
            let best_swap = state.choose_best_swap(target, heuristic)?;
            state.apply_swap(best_swap)?;
            current_swaps.push(best_swap);
            push_unique(
                &mut routable_nodes,
                state
                    .front_layer
                    .routable_node_on_qubit(best_swap[0], &target.neighbors),
            );
            push_unique(
                &mut routable_nodes,
                state
                    .front_layer
                    .routable_node_on_qubit(best_swap[1], &target.neighbors),
            );

            if let Some(increment) = heuristic.decay_increment {
                search_steps_since_decay_reset += 1;
                if search_steps_since_decay_reset >= heuristic.decay_reset {
                    for value in state.decay.values_mut() {
                        *value = 1.0;
                    }
                    search_steps_since_decay_reset = 0;
                } else {
                    *state.decay.entry(best_swap[0]).or_insert(1.0) += increment;
                    *state.decay.entry(best_swap[1]).or_insert(1.0) += increment;
                }
            }
        }

        if routable_nodes.is_empty() {
            for swap in current_swaps.drain(..).rev() {
                state.apply_swap(swap)?;
            }
            output.fallback_count += 1;
            let forced = state.force_enable_closest_node(target, &mut current_swaps)?;
            routable_nodes.extend(forced);
        }

        for node in &routable_nodes {
            state.front_layer.remove(*node);
        }
        state.update_route(
            sabre,
            target,
            heuristic,
            &mut output,
            &routable_nodes,
            Some(current_swaps),
        )?;
        state.lookahead_layers.iter_mut().for_each(Layer::clear);
        state.populate_extended_set(sabre);
        if heuristic.decay_increment.is_some() {
            for value in state.decay.values_mut() {
                *value = 1.0;
            }
        }
        routable_nodes.clear();
    }

    let quality = trial_quality(&output.operations, output.swap_count);
    Ok(TrialResult {
        operations: output.operations,
        final_layout: state.layout,
        swap_count: output.swap_count,
        fallback_count: output.fallback_count,
        control_flow_blocks_routed: output.control_flow_blocks_routed,
        quality,
    })
}

pub(crate) fn trial_seeds(seed: Option<u64>, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random));
    (0..count).map(|_| rng.random()).collect()
}

pub(crate) fn normalize_initial_layout(
    logical_qubits: &[LogicalQubit],
    target: &RoutingTarget,
    initial_layout: &Layout,
) -> Result<Layout, CompilerError> {
    let mut mapping = BTreeMap::new();
    for logical in logical_qubits {
        let physical = initial_layout.get_physical(*logical).ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "sabre initial layout does not map logical qubit {logical}"
            ))
        })?;
        if !target.physical_set.contains(&physical) {
            return Err(CompilerError::InvalidInput(format!(
                "sabre initial layout maps logical qubit {logical} to unusable physical qubit {physical}"
            )));
        }
        mapping.insert(*logical, physical);
    }
    Layout::new(
        logical_qubits.to_vec(),
        target.physical_qubits.clone(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvalidInput(format!("sabre initial layout is invalid: {error}"))
    })
}

#[derive(Debug, Clone)]
pub(crate) struct RoutingTarget {
    pub(crate) physical_qubits: Vec<PhysicalQubit>,
    physical_set: BTreeSet<PhysicalQubit>,
    neighbors: BTreeMap<PhysicalQubit, Vec<PhysicalQubit>>,
    distances: Vec<Vec<Option<f64>>>,
    graph: UnGraph<(), ()>,
    graph_index: BTreeMap<PhysicalQubit, NodeIndex>,
    physical_by_index: Vec<PhysicalQubit>,
}

impl RoutingTarget {
    pub(crate) fn from_physical(physical: &PhysicalLayoutGraph) -> Result<Self, CompilerError> {
        let physical_qubits = physical.physical_qubits().to_vec();
        let physical_set = physical_qubits.iter().copied().collect::<BTreeSet<_>>();
        let mut neighbors = physical_qubits
            .iter()
            .copied()
            .map(|qubit| (qubit, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        let mut graph = UnGraph::with_capacity(physical_qubits.len(), 0);
        let mut graph_index = BTreeMap::new();
        let mut physical_by_index = Vec::with_capacity(physical_qubits.len());

        for physical in &physical_qubits {
            let index = graph.add_node(());
            graph_index.insert(*physical, index);
            physical_by_index.push(*physical);
        }
        let mut distances = vec![vec![None; physical_qubits.len()]; physical_qubits.len()];
        for (index, row) in distances.iter_mut().enumerate() {
            row[index] = Some(0.0);
        }

        for (left_index, left) in physical_qubits.iter().copied().enumerate() {
            for (right_index, right) in physical_qubits
                .iter()
                .copied()
                .enumerate()
                .skip(left_index + 1)
            {
                if let Some(distance) = physical.distance(left, right) {
                    distances[left_index][right_index] = Some(f64::from(distance));
                    distances[right_index][left_index] = Some(f64::from(distance));
                }
                if physical.is_adjacent_undirected(left, right) {
                    neighbors
                        .get_mut(&left)
                        .expect("neighbor map exists")
                        .push(right);
                    neighbors
                        .get_mut(&right)
                        .expect("neighbor map exists")
                        .push(left);
                    graph.add_edge(graph_index[&left], graph_index[&right], ());
                }
            }
        }
        for items in neighbors.values_mut() {
            items.sort_unstable();
        }

        Ok(Self {
            physical_qubits,
            physical_set,
            neighbors,
            distances,
            graph,
            graph_index,
            physical_by_index,
        })
    }

    fn distance(&self, left: PhysicalQubit, right: PhysicalQubit) -> Result<f64, CompilerError> {
        let Some(left_index) = self.graph_index.get(&left).map(|index| index.index()) else {
            return Err(CompilerError::InvalidInput(format!(
                "physical qubit {left} is not usable in the target topology"
            )));
        };
        let Some(right_index) = self.graph_index.get(&right).map(|index| index.index()) else {
            return Err(CompilerError::InvalidInput(format!(
                "physical qubit {right} is not usable in the target topology"
            )));
        };
        self.distances[left_index][right_index].ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "physical qubits {left} and {right} are disconnected in the usable topology"
            ))
        })
    }
}

#[derive(Debug)]
struct RoutingState {
    layout: Layout,
    front_layer: Layer,
    lookahead_layers: Vec<Layer>,
    required_predecessors: Vec<u32>,
    decay: BTreeMap<PhysicalQubit, f64>,
    rng: StdRng,
}

impl RoutingState {
    fn new(
        sabre: &SabreDag,
        target: &RoutingTarget,
        layout: Layout,
        heuristic: &SabreHeuristicConfig,
        seed: u64,
    ) -> Self {
        let mut required_predecessors = vec![0; sabre.graph.node_count()];
        for edge in sabre.graph.edge_references() {
            required_predecessors[edge.target().index()] += 1;
        }

        Self {
            layout,
            front_layer: Layer::new(),
            lookahead_layers: vec![Layer::new(); heuristic.lookahead_weights.len()],
            required_predecessors,
            decay: target
                .physical_qubits
                .iter()
                .copied()
                .map(|physical| (physical, 1.0))
                .collect(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn apply_swap(&mut self, swap: [PhysicalQubit; 2]) -> Result<(), CompilerError> {
        self.front_layer.apply_swap(swap);
        for layer in &mut self.lookahead_layers {
            layer.apply_swap(swap);
        }
        self.layout
            .swap_physical(swap[0], swap[1])
            .map_err(|error| {
                CompilerError::InvariantViolation(format!(
                    "sabre attempted an invalid physical swap {swap:?}: {error}"
                ))
            })
    }

    fn update_route(
        &mut self,
        sabre: &SabreDag,
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
        output: &mut TrialOutput,
        nodes: &[NodeIndex],
        initial_swaps: Option<Vec<[PhysicalQubit; 2]>>,
    ) -> Result<(), CompilerError> {
        let mut to_visit = nodes.iter().copied().collect::<VecDeque<_>>();
        let mut pending_swaps = initial_swaps;

        while let Some(node_id) = to_visit.pop_front() {
            let node = &sabre.graph[node_id];
            match &node.kind {
                SabreNodeKind::TwoQ(pair) => {
                    let physical = [
                        physical_for(&self.layout, pair[0])?,
                        physical_for(&self.layout, pair[1])?,
                    ];
                    if !target
                        .neighbors
                        .get(&physical[0])
                        .is_some_and(|neighbors| neighbors.contains(&physical[1]))
                    {
                        self.front_layer.insert(node_id, physical);
                        continue;
                    }
                    output.apply_pending_swaps(pending_swaps.take());
                    for operation in &node.operations {
                        output
                            .operations
                            .push(map_operation(operation, &self.layout)?);
                    }
                }
                SabreNodeKind::Synchronize => {
                    output.apply_pending_swaps(pending_swaps.take());
                    for operation in &node.operations {
                        output
                            .operations
                            .push(map_operation(operation, &self.layout)?);
                    }
                }
                SabreNodeKind::ControlFlow(flow) => {
                    output.apply_pending_swaps(pending_swaps.take());
                    self.route_control_flow_node(
                        flow,
                        &node.operations,
                        target,
                        heuristic,
                        output,
                    )?;
                }
            }

            for edge in sabre.graph.edges_directed(node_id, Direction::Outgoing) {
                let successor = edge.target();
                self.required_predecessors[successor.index()] -= 1;
                if self.required_predecessors[successor.index()] == 0 {
                    to_visit.push_back(successor);
                }
            }
        }

        if pending_swaps.is_some() {
            return Err(CompilerError::InvariantViolation(
                "sabre selected swaps that did not route any front-layer node".to_string(),
            ));
        }
        Ok(())
    }

    fn route_control_flow_node(
        &mut self,
        flow: &SabreControlFlow,
        operations: &[Operation],
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
        output: &mut TrialOutput,
    ) -> Result<(), CompilerError> {
        let Some((first, rest)) = operations.split_first() else {
            return Ok(());
        };
        let routed = match flow {
            SabreControlFlow::IfElse {
                condition,
                true_body,
                false_body,
            } => {
                let mapped_condition = map_condition(*condition, &self.layout)?;
                let true_result = route_control_flow_body(
                    true_body,
                    target,
                    &self.layout,
                    heuristic,
                    output.next_nested_seed(),
                )?;
                let false_result = false_body
                    .as_ref()
                    .map(|body| {
                        route_control_flow_body(
                            body,
                            target,
                            &self.layout,
                            heuristic,
                            output.next_nested_seed(),
                        )
                    })
                    .transpose()?;
                output.merge_nested(&true_result);
                if let Some(result) = &false_result {
                    output.merge_nested(result);
                }
                let qubits = collect_control_flow_qubits(
                    mapped_condition.qubit,
                    &true_result.operations,
                    false_result
                        .as_ref()
                        .map(|result| result.operations.as_slice()),
                );
                let true_ops = true_result.operations;
                let false_ops = false_result.map(|result| result.operations);
                Operation {
                    instruction: Instruction::ControlFlowGate(ControlFlow::if_else(
                        mapped_condition,
                        true_ops,
                        false_ops,
                    )),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
            SabreControlFlow::WhileLoop { condition, body } => {
                let mapped_condition = map_condition(*condition, &self.layout)?;
                let body_result = route_control_flow_body(
                    body,
                    target,
                    &self.layout,
                    heuristic,
                    output.next_nested_seed(),
                )?;
                output.merge_nested(&body_result);
                let qubits = collect_control_flow_qubits(
                    mapped_condition.qubit,
                    &body_result.operations,
                    None,
                );
                let body_ops = body_result.operations;
                Operation {
                    instruction: Instruction::ControlFlowGate(ControlFlow::while_loop(
                        mapped_condition,
                        body_ops,
                    )),
                    qubits,
                    params: SmallVec::new(),
                    label: first.label.clone(),
                }
            }
        };
        output.operations.push(routed);
        for operation in rest {
            output
                .operations
                .push(map_operation(operation, &self.layout)?);
        }
        Ok(())
    }

    fn populate_extended_set(&mut self, sabre: &SabreDag) {
        // Build fixed-depth lookahead layers from the current front layer. Synchronize
        // and control-flow nodes are transparent for depth counting because they do
        // not add a parent-level two-qubit adjacency constraint.
        let mut next_visit = self.front_layer.iter_nodes().collect::<Vec<_>>();
        let mut to_visit = Vec::new();
        let mut decremented = BTreeMap::<NodeIndex, u32>::new();

        for layer in &mut self.lookahead_layers {
            for node in next_visit.drain(..) {
                for edge in sabre.graph.edges_directed(node, Direction::Outgoing) {
                    let successor = edge.target();
                    *decremented.entry(successor).or_insert(0) += 1;
                    self.required_predecessors[successor.index()] -= 1;
                    if self.required_predecessors[successor.index()] == 0 {
                        to_visit.push(successor);
                    }
                }
            }

            let mut index = 0;
            while index < to_visit.len() {
                let node = to_visit[index];
                match &sabre.graph[node].kind {
                    SabreNodeKind::TwoQ(pair) => {
                        if let (Ok(left), Ok(right)) = (
                            physical_for(&self.layout, pair[0]),
                            physical_for(&self.layout, pair[1]),
                        ) {
                            layer.insert(node, [left, right]);
                            next_visit.push(node);
                        }
                        // Missing physical mappings are ignored defensively here.
                        // Normal routing entrypoints normalize complete layouts before
                        // creating state, so this only affects future partial-layout use.
                    }
                    SabreNodeKind::Synchronize | SabreNodeKind::ControlFlow(_) => {
                        for edge in sabre.graph.edges_directed(node, Direction::Outgoing) {
                            let successor = edge.target();
                            *decremented.entry(successor).or_insert(0) += 1;
                            self.required_predecessors[successor.index()] -= 1;
                            if self.required_predecessors[successor.index()] == 0 {
                                to_visit.push(successor);
                            }
                        }
                    }
                }
                index += 1;
            }
            to_visit.clear();
        }

        // Lookahead exploration temporarily relaxes predecessor counts; restore
        // them before the real routing state advances.
        for (node, amount) in decremented {
            self.required_predecessors[node.index()] += amount;
        }
    }

    fn choose_best_swap(
        &mut self,
        target: &RoutingTarget,
        heuristic: &SabreHeuristicConfig,
    ) -> Result<[PhysicalQubit; 2], CompilerError> {
        let mut candidates = Vec::new();
        for active in self.front_layer.active_qubits() {
            if let Some(neighbors) = target.neighbors.get(&active) {
                for neighbor in neighbors {
                    candidates.push(if active <= *neighbor {
                        [active, *neighbor]
                    } else {
                        [*neighbor, active]
                    });
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        if candidates.is_empty() {
            return Err(CompilerError::TransformFailed {
                name: "sabre_route",
                reason: "no candidate SWAP can affect the front layer".to_string(),
            });
        }

        // SABRE score = weighted front-layer distance + weighted lookahead
        // distance, with optional multiplicative decay on recently swapped
        // physical qubits.
        let distance = |left, right| target.distance(left, right);
        let mut absolute = heuristic.basic_weight * self.front_layer.total_score(&distance)?;
        for (layer, weight) in self
            .lookahead_layers
            .iter()
            .zip(heuristic.lookahead_weights.iter().copied())
        {
            absolute += weight * layer.total_score(&distance)?;
        }

        let mut best_score = f64::INFINITY;
        let mut best_swaps = Vec::new();
        for swap in candidates {
            let mut score = absolute
                + heuristic.basic_weight * self.front_layer.swap_delta_score(swap, &distance)?;
            for (layer, weight) in self
                .lookahead_layers
                .iter()
                .zip(heuristic.lookahead_weights.iter().copied())
            {
                score += weight * layer.swap_delta_score(swap, &distance)?;
            }
            if heuristic.decay_increment.is_some() {
                let decay = self.decay[&swap[0]].max(self.decay[&swap[1]]);
                score *= decay;
            }

            if score - best_score < -heuristic.best_epsilon {
                best_score = score;
                best_swaps.clear();
                best_swaps.push(swap);
            } else if (score - best_score).abs() <= heuristic.best_epsilon {
                best_swaps.push(swap);
            }
        }

        best_swaps.choose(&mut self.rng).copied().ok_or_else(|| {
            CompilerError::InvariantViolation("sabre found no best SWAP".to_string())
        })
    }

    fn force_enable_closest_node(
        &mut self,
        target: &RoutingTarget,
        current_swaps: &mut Vec<[PhysicalQubit; 2]>,
    ) -> Result<Vec<NodeIndex>, CompilerError> {
        // Fallback guarantees progress when the heuristic is stuck: choose the
        // closest front-layer interaction and walk one endpoint along a shortest
        // path until that interaction becomes adjacent.
        let (closest_node, qubits) = self
            .front_layer
            .iter()
            .min_by(|(_, a), (_, b)| {
                target
                    .distance(a[0], a[1])
                    .and_then(|ad| target.distance(b[0], b[1]).map(|bd| ad.total_cmp(&bd)))
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or_else(|| {
                CompilerError::InvariantViolation(
                    "sabre fallback called with an empty front layer".to_string(),
                )
            })?;
        let path = shortest_path(&target.neighbors, qubits[0], qubits[1]).ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "physical qubits {} and {} are disconnected in the usable topology",
                qubits[0], qubits[1]
            ))
        })?;
        if path.len() < 3 {
            return Ok(vec![closest_node]);
        }
        for window in path.windows(2).take(path.len() - 2) {
            let swap = [window[0], window[1]];
            self.apply_swap(swap)?;
            current_swaps.push(swap);
        }

        let mut routed = Vec::new();
        if self.front_layer.iter().any(|(node, current)| {
            node == closest_node
                && target
                    .neighbors
                    .get(&current[0])
                    .is_some_and(|neighbors| neighbors.contains(&current[1]))
        }) {
            routed.push(closest_node);
        }

        for swap in current_swaps.iter().copied() {
            push_unique(
                &mut routed,
                self.front_layer
                    .routable_node_on_qubit(swap[0], &target.neighbors),
            );
            push_unique(
                &mut routed,
                self.front_layer
                    .routable_node_on_qubit(swap[1], &target.neighbors),
            );
        }

        if routed.is_empty() {
            routed.push(closest_node);
        }
        Ok(routed)
    }
}

#[derive(Debug, Default)]
struct TrialOutput {
    operations: Vec<Operation>,
    swap_count: usize,
    fallback_count: usize,
    control_flow_blocks_routed: usize,
    nested_seed_counter: u64,
}

impl TrialOutput {
    fn new(seed: u64) -> Self {
        Self {
            nested_seed_counter: seed,
            ..Self::default()
        }
    }

    fn apply_pending_swaps(&mut self, swaps: Option<Vec<[PhysicalQubit; 2]>>) {
        if let Some(swaps) = swaps {
            self.swap_count += swaps.len();
            self.operations
                .extend(swaps.into_iter().map(swap_operation));
        }
    }

    fn next_nested_seed(&mut self) -> u64 {
        let seed = self.nested_seed_counter;
        self.nested_seed_counter = self.nested_seed_counter.wrapping_add(1);
        seed
    }

    fn merge_nested(&mut self, nested: &TrialResult) {
        self.swap_count += nested.swap_count;
        self.fallback_count += nested.fallback_count;
        self.control_flow_blocks_routed += nested.control_flow_blocks_routed + 1;
    }
}

fn route_control_flow_body(
    sabre: &SabreDag,
    target: &RoutingTarget,
    entry_layout: &Layout,
    heuristic: &SabreHeuristicConfig,
    seed: u64,
) -> Result<TrialResult, CompilerError> {
    let mut result = route_trial(sabre, target, entry_layout, heuristic, seed)?;
    let epilogue_swaps = restore_layout_swaps(target, &result.final_layout, entry_layout, seed)?;
    let mut layout = result.final_layout.clone();
    for swap in &epilogue_swaps {
        layout.swap_physical(swap[0], swap[1]).map_err(|error| {
            CompilerError::InvariantViolation(format!(
                "sabre control-flow epilogue generated an invalid SWAP: {error}"
            ))
        })?;
    }
    result
        .operations
        .extend(epilogue_swaps.iter().copied().map(swap_operation));
    result.swap_count += epilogue_swaps.len();
    result.final_layout = layout;
    result.quality = trial_quality(&result.operations, result.swap_count);
    if result.final_layout.l2p_map() != entry_layout.l2p_map() {
        return Err(CompilerError::InvariantViolation(
            "sabre control-flow epilogue did not restore the entry layout".to_string(),
        ));
    }
    Ok(result)
}

fn restore_layout_swaps(
    target: &RoutingTarget,
    current: &Layout,
    desired: &Layout,
    seed: u64,
) -> Result<Vec<[PhysicalQubit; 2]>, CompilerError> {
    let mapping = desired
        .physical_qubits()
        .filter_map(|physical| {
            let logical = current.get_logical(physical)?;
            let desired_physical = desired
                .get_physical(logical)
                .expect("desired layout maps logical qubits it reports");
            Some((
                target.graph_index[&physical],
                target.graph_index[&desired_physical],
            ))
        })
        .collect();

    let swaps = token_swapper(
        &target.graph,
        mapping,
        Some(CONTROL_FLOW_EPILOGUE_TRIALS),
        Some(seed),
        None,
    )
    .map_err(|error| CompilerError::TransformFailed {
        name: "sabre_route",
        reason: format!("failed to restore control-flow layout: {error}"),
    })?;
    swaps
        .into_iter()
        .map(|(left, right)| {
            Ok([
                target.physical_by_index[left.index()],
                target.physical_by_index[right.index()],
            ])
        })
        .collect()
}

fn append_operation_to_circuit(
    circuit: &mut Circuit,
    operation: &Operation,
    context: &Circuit,
) -> Result<(), CompilerError> {
    let params = operation
        .params
        .iter()
        .map(|param| match param {
            CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(*value)),
            CircuitParam::Index(index) => context
                .parameters()
                .get_index(*index as usize)
                .cloned()
                .map(ParameterValue::Param)
                .ok_or_else(|| {
                    CompilerError::InvalidInput(format!(
                        "operation references parameter index {} outside the source circuit",
                        index
                    ))
                }),
        })
        .collect::<Result<Vec<_>, CompilerError>>()?;
    circuit
        .append(
            operation.instruction.clone(),
            operation.qubits.iter().copied(),
            params,
            operation.label.as_deref(),
        )
        .map_err(CompilerError::from)
}

fn map_operation(operation: &Operation, layout: &Layout) -> Result<Operation, CompilerError> {
    Ok(Operation {
        instruction: operation.instruction.clone(),
        qubits: operation
            .qubits
            .iter()
            .copied()
            .map(|qubit| {
                physical_for(layout, LogicalQubit::from_qubit(qubit)).map(PhysicalQubit::qubit)
            })
            .collect::<Result<SmallVec<[crate::circuit::Qubit; 3]>, _>>()?,
        params: operation.params.clone(),
        label: operation.label.clone(),
    })
}

fn map_condition(
    condition: crate::circuit::ConditionView,
    layout: &Layout,
) -> Result<crate::circuit::ConditionView, CompilerError> {
    Ok(crate::circuit::ConditionView::new(
        physical_for(layout, LogicalQubit::from_qubit(condition.qubit))?.qubit(),
        condition.target,
    ))
}

fn physical_for(layout: &Layout, logical: LogicalQubit) -> Result<PhysicalQubit, CompilerError> {
    layout.get_physical(logical).ok_or_else(|| {
        CompilerError::InvariantViolation(format!(
            "sabre layout does not map logical qubit {logical}"
        ))
    })
}

pub(crate) fn validate_reachable_interactions(
    sabre: &SabreDag,
    target: &RoutingTarget,
    layout: &Layout,
) -> Result<(), CompilerError> {
    for node in sabre.graph.node_weights() {
        match &node.kind {
            SabreNodeKind::TwoQ(pair) => {
                let left = physical_for(layout, pair[0])?;
                let right = physical_for(layout, pair[1])?;
                target.distance(left, right)?;
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::IfElse {
                true_body,
                false_body,
                ..
            }) => {
                validate_reachable_interactions(true_body, target, layout)?;
                if let Some(false_body) = false_body {
                    validate_reachable_interactions(false_body, target, layout)?;
                }
            }
            SabreNodeKind::ControlFlow(SabreControlFlow::WhileLoop { body, .. }) => {
                validate_reachable_interactions(body, target, layout)?;
            }
            SabreNodeKind::Synchronize => {}
        }
    }
    Ok(())
}

fn swap_operation(swap: [PhysicalQubit; 2]) -> Operation {
    Operation {
        instruction: Instruction::Standard(crate::circuit::StandardGate::SWAP),
        qubits: smallvec![swap[0].qubit(), swap[1].qubit()],
        params: SmallVec::new(),
        label: None,
    }
}

fn collect_control_flow_qubits(
    condition: crate::circuit::Qubit,
    true_body: &[Operation],
    false_body: Option<&[Operation]>,
) -> SmallVec<[crate::circuit::Qubit; 3]> {
    let mut qubits = BTreeSet::new();
    qubits.insert(condition);
    for operation in true_body {
        qubits.extend(operation.qubits.iter().copied());
    }
    if let Some(false_body) = false_body {
        for operation in false_body {
            qubits.extend(operation.qubits.iter().copied());
        }
    }
    qubits.into_iter().collect()
}

fn push_unique(nodes: &mut Vec<NodeIndex>, node: Option<NodeIndex>) {
    if let Some(node) = node {
        if !nodes.contains(&node) {
            nodes.push(node);
        }
    }
}

pub(crate) fn compare_trial_quality(
    objective: SabreTrialObjective,
    left: TrialQuality,
    left_index: usize,
    right: TrialQuality,
    right_index: usize,
) -> Ordering {
    match objective {
        SabreTrialObjective::SwapCount => left
            .swap_count
            .cmp(&right.swap_count)
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::Depth => left
            .two_qubit_depth
            .cmp(&right.two_qubit_depth)
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::SwapThenDepth => left
            .swap_count
            .cmp(&right.swap_count)
            .then_with(|| left.two_qubit_depth.cmp(&right.two_qubit_depth))
            .then_with(|| left.operation_count.cmp(&right.operation_count))
            .then_with(|| left_index.cmp(&right_index)),
        SabreTrialObjective::DepthThenSwap => left
            .two_qubit_depth
            .cmp(&right.two_qubit_depth)
            .then_with(|| left.swap_count.cmp(&right.swap_count))
            .then_with(|| left.operation_count.cmp(&right.operation_count))
            .then_with(|| left_index.cmp(&right_index)),
    }
}

fn trial_quality(operations: &[Operation], swap_count: usize) -> TrialQuality {
    TrialQuality {
        swap_count,
        two_qubit_depth: two_qubit_depth(operations),
        operation_count: operation_count(operations),
    }
}

fn two_qubit_depth(operations: &[Operation]) -> usize {
    let mut qubit_depths = BTreeMap::<Qubit, usize>::new();
    let mut max_depth = 0usize;

    for operation in operations {
        let local_depth = operation_local_two_qubit_depth(operation);
        if local_depth == 0 {
            continue;
        }

        let base = operation
            .qubits
            .iter()
            .map(|qubit| qubit_depths.get(qubit).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let depth = base + local_depth;
        for qubit in &operation.qubits {
            qubit_depths.insert(*qubit, depth);
        }
        max_depth = max_depth.max(depth);
    }

    max_depth
}

fn operation_local_two_qubit_depth(operation: &Operation) -> usize {
    match &operation.instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            let true_depth = two_qubit_depth(gate.true_body());
            let false_depth = gate.false_body().map(two_qubit_depth).unwrap_or(0);
            true_depth.max(false_depth)
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => two_qubit_depth(gate.body()),
        _ if operation.qubits.len() == 2 => 1,
        _ => 0,
    }
}

fn operation_count(operations: &[Operation]) -> usize {
    operations
        .iter()
        .map(|operation| {
            1 + match &operation.instruction {
                Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                    operation_count(gate.true_body())
                        + gate.false_body().map(operation_count).unwrap_or(0)
                }
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                    operation_count(gate.body())
                }
                _ => 0,
            }
        })
        .sum()
}

fn shortest_path(
    neighbors: &BTreeMap<PhysicalQubit, Vec<PhysicalQubit>>,
    start: PhysicalQubit,
    goal: PhysicalQubit,
) -> Option<Vec<PhysicalQubit>> {
    if start == goal {
        return Some(vec![start]);
    }
    let mut queue = VecDeque::new();
    let mut predecessor = BTreeMap::new();
    let mut seen = BTreeSet::new();
    queue.push_back(start);
    seen.insert(start);

    while let Some(current) = queue.pop_front() {
        for neighbor in neighbors.get(&current).into_iter().flatten().copied() {
            if !seen.insert(neighbor) {
                continue;
            }
            predecessor.insert(neighbor, current);
            if neighbor == goal {
                let mut path = vec![goal];
                let mut cursor = goal;
                while cursor != start {
                    cursor = predecessor[&cursor];
                    path.push(cursor);
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(neighbor);
        }
    }
    None
}
