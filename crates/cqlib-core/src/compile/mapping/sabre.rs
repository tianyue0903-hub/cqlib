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

//! Fidelity-aware SABRE mapper.
//!
//! This implementation routes 2q interactions onto sparse hardware topologies
//! using an adaptive SWAP-search heuristic inspired by SABRE, plus optional VF2
//! assistance for initial layout quality.
//!
//! High-level flow:
//! 1. Preprocess and validate source circuit (1q/2q only, no control flow).
//! 2. Build gate dependency DAG and maintain front-layer execution state.
//! 3. Score SWAP candidates using distance, fidelity, and decay terms.
//! 4. Emit mapped operations and reconstruct a topology-compliant circuit.

use super::vf2::{Vf2CandidateOptions, Vf2Mapping, Vf2ScoreWeights};
use super::{
    build_if_else_operation, build_output_circuit_from_source, build_while_loop_operation, is_cx,
    map_operation_qubits, normalize_index_pair, preprocess_circuit, preprocess_program,
    FidelityMap, PreparedCircuit, PreparedIfElse, PreparedPassthroughOp, PreparedProgram,
    PreparedProgramItem, PreparedSegment, PreparedWhileLoop, TopologyAdapter,
};
use crate::circuit::gate::control_flow::ConditionView;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::compile::graph::{DependencyNode, GateGraph};
use crate::device::Topology;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
/// Internal struct `GateDependencyDag` used by compile mapping workflows.
struct GateDependencyDag {
    nodes: Vec<DependencyNode>,
    indegree: Vec<usize>,
    initial_single_qubit_ops: Vec<(usize, usize)>,
    front_layer: HashSet<usize>,
}

impl GateDependencyDag {
    /// Internal helper for node.
    fn node(&self, gate_id: usize) -> Option<&DependencyNode> {
        self.nodes.get(gate_id.saturating_sub(1))
    }
}

#[derive(Debug, Clone)]
/// Internal enum `AnsStep` used by compile mapping workflows.
enum AnsStep {
    Op(Operation),
    Swap { u: usize, v: usize },
    Bridge { u: usize, v: usize, bridge: usize },
}

impl AnsStep {
    /// Internal helper for cost.
    fn cost(&self) -> usize {
        match self {
            Self::Op(_) => 1,
            Self::Swap { .. } => 3,
            Self::Bridge { .. } => 4,
        }
    }
}

#[derive(Debug, Clone)]
/// Internal struct `AnsGroup` used by compile mapping workflows.
struct AnsGroup {
    initial_l2p: Vec<usize>,
    final_l2p: Vec<usize>,
    steps: Vec<AnsStep>,
    cost: usize,
    log_fidelity: f64,
    objective: f64,
}

#[derive(Debug, Clone)]
/// Internal struct `RatedSwap` used by compile mapping workflows.
struct RatedSwap {
    u: usize,
    v: usize,
    score: f64,
    distance_term: f64,
    fidelity_penalty: f64,
}

#[derive(Debug, Clone)]
/// Internal struct `RoutingState` used by compile mapping workflows.
struct RoutingState {
    logic2phy: Vec<usize>,
    phy2logic: Vec<Option<usize>>,
    pre_number: Vec<usize>,
    front_layer: HashSet<usize>,
    ans_steps: Vec<AnsStep>,
    decay: Vec<f64>,
    decay_time: usize,
    weight_gates: Vec<Vec<(usize, f64)>>,
    preprocessing_h: f64,
}

#[derive(Debug, Clone)]
struct StructuredRoute {
    exit_l2p: Vec<usize>,
    ops: Vec<Operation>,
    cost: usize,
    log_fidelity: f64,
    objective: f64,
}

#[derive(Debug, Clone)]
struct LayoutTransition {
    ops: Vec<Operation>,
    cost: usize,
    log_fidelity: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StructuredLayoutState {
    l2p: Vec<usize>,
}

impl StructuredLayoutState {
    fn new(l2p: &[usize]) -> Self {
        Self { l2p: l2p.to_vec() }
    }

    fn as_slice(&self) -> &[usize] {
        &self.l2p
    }

    fn condition_view(
        &self,
        topology: &TopologyAdapter,
        logical_qubit: usize,
        target: u8,
    ) -> ConditionView {
        ConditionView::new(topology.physical_qubits[self.l2p[logical_qubit]], target)
    }
}

#[derive(Debug, Clone)]
struct StructuredBranchMerge {
    true_body: Vec<Operation>,
    false_body: Option<Vec<Operation>>,
    cost: usize,
    log_fidelity: f64,
}

#[derive(Debug, Clone)]
struct StructuredLoopClosure {
    body_ops: Vec<Operation>,
    cost: usize,
    log_fidelity: f64,
}

/// Policy controlling how VF2 is used around SABRE routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vf2Policy {
    /// Try direct VF2 mapping first; if it fails, run SABRE.
    DirectThenSabre,
    /// Skip direct mapping and only use VF2 as an optional SABRE initial layout seed.
    InitialOnly,
    /// Disable VF2 completely.
    Disabled,
}

impl Default for Vf2Policy {
    /// Returns the default policy (`DirectThenSabre`).
    fn default() -> Self {
        Self::DirectThenSabre
    }
}

/// Configuration for SABRE mapping.
///
/// Defaults are chosen to match the current baseline behavior used by Python
/// bindings and compile tests.
#[derive(Debug, Clone)]
pub struct SabreConfig {
    /// VF2 usage policy for direct mapping and seeding.
    pub vf2_policy: Vf2Policy,
    /// Number of ranked VF2 seed layouts considered before random fill.
    pub vf2_seed_top_k: usize,
    /// Scoring weights for VF2 seed-candidate ranking.
    pub vf2_seed_weights: Vf2ScoreWeights,
    /// Enables field-aware ranking term in swap scoring.
    pub field_mode: bool,
    /// Look-ahead gate window size.
    pub size_e: usize,
    /// Weight coefficient for look-ahead term.
    pub w: f64,
    /// Decay coefficient for repeated SWAP penalties.
    pub decay_coff: f64,
    /// Number of steps before decay values are reset.
    pub decay_reset_time: usize,
    /// Internal strategy selector for greedy SWAP selection.
    pub greedy_strategy: usize,
    /// Number of random initial-layout trials.
    pub initial_iterations: usize,
    /// Number of forward/reverse refinement rounds.
    pub repeat_iterations: usize,
    /// Number of SWAP-sampling iterations in each routing pass.
    pub swap_iterations: usize,
    /// Weight applied to SWAP-edge fidelity penalty in local swap scoring.
    pub swap_fidelity_weight: f64,
    /// Weight applied to gate cost in global candidate objective.
    pub gate_cost_weight: f64,
    /// Weight applied to predicted fidelity loss in global objective.
    pub predicted_fidelity_weight: f64,
    /// Random seed (`-1` means auto-seeded).
    pub seed: i64,
}

impl Default for SabreConfig {
    /// Returns baseline SABRE defaults used across compile APIs.
    fn default() -> Self {
        Self {
            vf2_policy: Vf2Policy::DirectThenSabre,
            vf2_seed_top_k: 8,
            vf2_seed_weights: Vf2ScoreWeights::default(),
            field_mode: true,
            size_e: 20,
            w: 0.5,
            decay_coff: 0.001,
            decay_reset_time: 5,
            greedy_strategy: 3,
            initial_iterations: 1,
            repeat_iterations: 1,
            swap_iterations: 1,
            swap_fidelity_weight: 0.25,
            gate_cost_weight: 1.0,
            predicted_fidelity_weight: 0.1,
            seed: -1,
        }
    }
}

/// SABRE mapping implementation.
#[derive(Debug, Clone)]
pub struct SabreMapping {
    /// Logical -> physical mapping after latest execution.
    pub logic2phy: Vec<Qubit>,
    /// Physical -> logical mapping after latest execution.
    pub phy2logic: HashMap<Qubit, usize>,

    topology: TopologyAdapter,
    config: SabreConfig,
    sample_eps: f64,
    fidelity_eps: f64,
    objective_eps: f64,
    rng: StdRng,
}

impl SabreMapping {
    /// Creates a SABRE mapper with optional fidelity map and explicit config.
    pub fn new(
        topology: Topology,
        fidelity_map: Option<FidelityMap>,
        config: SabreConfig,
    ) -> Result<Self, CompileError> {
        let topology = TopologyAdapter::new(&topology, fidelity_map.as_ref())?;

        let seed = if config.seed >= 0 {
            config.seed as u64
        } else {
            let mut trng = rand::rng();
            trng.random::<u64>()
        };

        Ok(Self {
            logic2phy: Vec::new(),
            phy2logic: HashMap::new(),
            topology,
            config,
            sample_eps: 1e-10,
            fidelity_eps: 1e-12,
            objective_eps: 1e-12,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Executes SABRE routing on a validated 1q/2q, control-flow-free circuit.
    pub fn execute(&mut self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let program = preprocess_program(circuit)?;
        if !program.is_plain_linear() {
            return self.execute_structured(circuit, &program);
        }
        self.execute_linear(circuit)
    }

    fn execute_linear(&mut self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let prepared = preprocess_circuit(circuit)?;
        let reverse_circuit = circuit.inverse()?;
        let reverse_prepared = preprocess_circuit(&reverse_circuit)?;

        let logical_width = prepared.logical_qubits.len();
        let available_nodes = self.usable_nodes();

        if logical_width > available_nodes.len() {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: available_nodes.len(),
            });
        }

        let original_info = self.build_circuit_info(&prepared, logical_width)?;
        let reverse_info = self.build_circuit_info(&reverse_prepared, logical_width)?;

        let initial_iters = self.config.initial_iterations.max(1);
        let repeat_iters = self.config.repeat_iterations;
        let swap_iters = self.config.swap_iterations.max(1);
        let mut initial_layouts = self.initial_layout_candidates(
            &prepared,
            &available_nodes,
            logical_width,
            initial_iters,
        )?;

        let mut best_group: Option<AnsGroup> = None;

        for mut initial_mapping in initial_layouts.drain(..) {
            for iter in 0..=repeat_iters {
                let forward_group =
                    self.execute_routing(&original_info, &prepared, &initial_mapping, swap_iters)?;

                if best_group
                    .as_ref()
                    .map(|g| self.group_better(&forward_group, g))
                    .unwrap_or(true)
                {
                    best_group = Some(forward_group.clone());
                }

                if iter == repeat_iters {
                    break;
                }

                let best_ref = best_group
                    .as_ref()
                    .ok_or_else(|| CompileError::Internal("missing best SABRE group".into()))?;

                let reverse_group = self.execute_routing(
                    &reverse_info,
                    &reverse_prepared,
                    &best_ref.final_l2p,
                    swap_iters,
                )?;
                initial_mapping = reverse_group.final_l2p;
            }
        }

        let best_group = best_group.ok_or(CompileError::SabreRoutingStuck)?;

        self.logic2phy = best_group
            .final_l2p
            .iter()
            .map(|&p| self.topology.physical_qubits[p])
            .collect();
        self.phy2logic.clear();
        for (logical, &physical) in self.logic2phy.iter().enumerate() {
            self.phy2logic.insert(physical, logical);
        }

        let mapped_ops = self.replay_ops(&prepared, &original_info, &best_group);
        Ok(build_output_circuit_from_source(circuit, mapped_ops))
    }

    fn execute_structured(
        &mut self,
        circuit: &Circuit,
        program: &PreparedProgram,
    ) -> Result<Circuit, CompileError> {
        let prepared = program.flatten_interaction_circuit();
        let logical_width = prepared.logical_qubits.len();
        let available_nodes = self.usable_nodes();

        if logical_width > available_nodes.len() {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: available_nodes.len(),
            });
        }

        let initial_iters = self.config.initial_iterations.max(1);
        let mut initial_layouts = self.initial_layout_candidates(
            &prepared,
            &available_nodes,
            logical_width,
            initial_iters,
        )?;

        let mut best_route: Option<StructuredRoute> = None;
        for initial_layout in initial_layouts.drain(..) {
            let candidate = self.route_items(program, &program.items, &initial_layout)?;
            if best_route
                .as_ref()
                .map(|best| self.structured_route_better(&candidate, best))
                .unwrap_or(true)
            {
                best_route = Some(candidate);
            }
        }

        let best_route = best_route.ok_or(CompileError::SabreRoutingStuck)?;
        self.logic2phy = best_route
            .exit_l2p
            .iter()
            .map(|&p| self.topology.physical_qubits[p])
            .collect();
        self.phy2logic.clear();
        for (logical, &physical) in self.logic2phy.iter().enumerate() {
            self.phy2logic.insert(physical, logical);
        }

        Ok(build_output_circuit_from_source(circuit, best_route.ops))
    }

    fn route_items(
        &mut self,
        program: &PreparedProgram,
        items: &[PreparedProgramItem],
        entry_l2p: &[usize],
    ) -> Result<StructuredRoute, CompileError> {
        let mut current = Self::identity_route(entry_l2p);
        let mut idx = 0usize;
        while idx < items.len() {
            match &items[idx] {
                PreparedProgramItem::Segment(segment) => {
                    let next = self.route_segment(
                        segment,
                        &program.logical_qubits,
                        &program.parameters,
                        &current.exit_l2p,
                    )?;
                    self.extend_route(&mut current, next);
                    idx += 1;
                }
                PreparedProgramItem::Passthrough(op) => {
                    current
                        .ops
                        .push(self.map_passthrough(op, &current.exit_l2p));
                    idx += 1;
                }
                PreparedProgramItem::IfElse(node) => {
                    let next =
                        self.route_if_else(program, node, &items[idx + 1..], &current.exit_l2p)?;
                    self.extend_route(&mut current, next);
                    return Ok(current);
                }
                PreparedProgramItem::WhileLoop(node) => {
                    let next =
                        self.route_while_loop(program, node, &items[idx + 1..], &current.exit_l2p)?;
                    self.extend_route(&mut current, next);
                    return Ok(current);
                }
            }
        }
        current.objective = self.routing_objective(current.cost, current.log_fidelity);
        Ok(current)
    }

    fn route_segment(
        &mut self,
        segment: &PreparedSegment,
        logical_qubits: &[Qubit],
        parameters: &[Parameter],
        entry_l2p: &[usize],
    ) -> Result<StructuredRoute, CompileError> {
        if segment.operations.is_empty() {
            return Ok(Self::identity_route(entry_l2p));
        }
        let prepared = segment.to_prepared_circuit(logical_qubits, parameters);
        let info = self.build_circuit_info(&prepared, prepared.logical_qubits.len())?;
        let group = self.execute_routing(
            &info,
            &prepared,
            entry_l2p,
            self.config.swap_iterations.max(1),
        )?;
        Ok(StructuredRoute {
            exit_l2p: group.final_l2p.clone(),
            ops: self.replay_ops(&prepared, &info, &group),
            cost: group.cost,
            log_fidelity: group.log_fidelity,
            objective: group.objective,
        })
    }

    fn route_if_else(
        &mut self,
        program: &PreparedProgram,
        node: &PreparedIfElse,
        continuation: &[PreparedProgramItem],
        entry_l2p: &[usize],
    ) -> Result<StructuredRoute, CompileError> {
        let entry_state = StructuredLayoutState::new(entry_l2p);
        let true_route = self.route_items(
            &node.true_body,
            &node.true_body.items,
            entry_state.as_slice(),
        )?;
        let false_route = if let Some(false_body) = &node.false_body {
            self.route_items(false_body, &false_body.items, entry_state.as_slice())?
        } else {
            Self::identity_route(entry_state.as_slice())
        };

        let mut merge_states = vec![
            entry_state.clone(),
            StructuredLayoutState::new(&true_route.exit_l2p),
            StructuredLayoutState::new(&false_route.exit_l2p),
        ];
        merge_states.extend(self.best_program_layouts(&node.true_body)?);
        if let Some(false_body) = &node.false_body {
            merge_states.extend(self.best_program_layouts(false_body)?);
        }
        let merge_states = self.continuation_entry_states(program, continuation, merge_states)?;

        let mut best: Option<StructuredRoute> = None;
        for merge_state in merge_states {
            let merged = match self.merge_branch_routes(
                &true_route,
                &false_route,
                &merge_state,
                node.false_body.is_some(),
            ) {
                Ok(merged) => merged,
                Err(CompileError::SabreRoutingStuck) => continue,
                Err(err) => return Err(err),
            };

            let if_else_op = build_if_else_operation(
                entry_state.condition_view(
                    &self.topology,
                    node.condition_logical,
                    node.condition.target,
                ),
                merged.true_body,
                merged.false_body,
                node.label.clone(),
            );

            let continuation_route =
                match self.route_continuation(program, continuation, &merge_state) {
                    Ok(route) => route,
                    Err(CompileError::SabreRoutingStuck) => continue,
                    Err(err) => return Err(err),
                };
            let total_cost = merged.cost + continuation_route.cost;
            let total_log_fidelity = merged.log_fidelity + continuation_route.log_fidelity;

            let mut ops = Vec::with_capacity(1 + continuation_route.ops.len());
            ops.push(if_else_op);
            ops.extend(continuation_route.ops.clone());

            let candidate = StructuredRoute {
                exit_l2p: continuation_route.exit_l2p.clone(),
                ops,
                cost: total_cost,
                log_fidelity: total_log_fidelity,
                objective: self.routing_objective(total_cost, total_log_fidelity),
            };
            if best
                .as_ref()
                .map(|current| self.structured_route_better(&candidate, current))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }

        best.ok_or(CompileError::SabreRoutingStuck)
    }

    fn route_while_loop(
        &mut self,
        program: &PreparedProgram,
        node: &PreparedWhileLoop,
        continuation: &[PreparedProgramItem],
        entry_l2p: &[usize],
    ) -> Result<StructuredRoute, CompileError> {
        let entry_state = StructuredLayoutState::new(entry_l2p);
        let mut loop_states = vec![entry_state.clone()];
        loop_states.extend(self.best_program_layouts(&node.body)?);
        let loop_states = self.continuation_entry_states(program, continuation, loop_states)?;

        let mut best: Option<StructuredRoute> = None;
        for loop_state in loop_states {
            let pre_loop =
                match self.reconcile_layout(entry_state.as_slice(), loop_state.as_slice()) {
                    Ok(route) => route,
                    Err(CompileError::SabreRoutingStuck) => continue,
                    Err(err) => return Err(err),
                };
            let body_route =
                match self.route_items(&node.body, &node.body.items, loop_state.as_slice()) {
                    Ok(route) => route,
                    Err(CompileError::SabreRoutingStuck) => continue,
                    Err(err) => return Err(err),
                };
            let closed_loop = match self.close_loop_body(&body_route, &loop_state) {
                Ok(closed) => closed,
                Err(CompileError::SabreRoutingStuck) => continue,
                Err(err) => return Err(err),
            };

            let while_op = build_while_loop_operation(
                loop_state.condition_view(
                    &self.topology,
                    node.condition_logical,
                    node.condition.target,
                ),
                closed_loop.body_ops,
                node.label.clone(),
            );

            let continuation_route =
                match self.route_continuation(program, continuation, &loop_state) {
                    Ok(route) => route,
                    Err(CompileError::SabreRoutingStuck) => continue,
                    Err(err) => return Err(err),
                };
            let total_cost = pre_loop.cost + closed_loop.cost + continuation_route.cost;
            let total_log_fidelity =
                pre_loop.log_fidelity + closed_loop.log_fidelity + continuation_route.log_fidelity;

            let mut ops = pre_loop.ops.clone();
            ops.push(while_op);
            ops.extend(continuation_route.ops.clone());

            let candidate = StructuredRoute {
                exit_l2p: continuation_route.exit_l2p.clone(),
                ops,
                cost: total_cost,
                log_fidelity: total_log_fidelity,
                objective: self.routing_objective(total_cost, total_log_fidelity),
            };
            if best
                .as_ref()
                .map(|current| self.structured_route_better(&candidate, current))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }

        best.ok_or(CompileError::SabreRoutingStuck)
    }

    fn route_continuation(
        &mut self,
        program: &PreparedProgram,
        continuation: &[PreparedProgramItem],
        entry_state: &StructuredLayoutState,
    ) -> Result<StructuredRoute, CompileError> {
        if continuation.is_empty() {
            Ok(Self::identity_route(entry_state.as_slice()))
        } else {
            self.route_items(program, continuation, entry_state.as_slice())
        }
    }

    fn continuation_entry_states(
        &self,
        program: &PreparedProgram,
        continuation: &[PreparedProgramItem],
        seeds: Vec<StructuredLayoutState>,
    ) -> Result<Vec<StructuredLayoutState>, CompileError> {
        let mut states = seeds;
        states.extend(self.best_items_layouts(program, continuation)?);
        Ok(Self::dedup_layout_states(states))
    }

    fn dedup_layout_states(states: Vec<StructuredLayoutState>) -> Vec<StructuredLayoutState> {
        let mut seen = HashSet::new();
        states
            .into_iter()
            .filter(|state| seen.insert(state.clone()))
            .collect()
    }

    fn merge_branch_routes(
        &self,
        true_route: &StructuredRoute,
        false_route: &StructuredRoute,
        merge_state: &StructuredLayoutState,
        preserve_false_body: bool,
    ) -> Result<StructuredBranchMerge, CompileError> {
        let true_tail = self.reconcile_layout(&true_route.exit_l2p, merge_state.as_slice())?;
        let false_tail = self.reconcile_layout(&false_route.exit_l2p, merge_state.as_slice())?;

        let mut true_body = true_route.ops.clone();
        true_body.extend(true_tail.ops.clone());
        let mut false_body_ops = false_route.ops.clone();
        false_body_ops.extend(false_tail.ops.clone());
        let false_body = if false_body_ops.is_empty() && !preserve_false_body {
            None
        } else {
            Some(false_body_ops)
        };

        let true_cost = true_route.cost + true_tail.cost;
        let false_cost = false_route.cost + false_tail.cost;
        let true_log = true_route.log_fidelity + true_tail.log_fidelity;
        let false_log = false_route.log_fidelity + false_tail.log_fidelity;

        Ok(StructuredBranchMerge {
            true_body,
            false_body,
            cost: true_cost.max(false_cost),
            log_fidelity: true_log.min(false_log),
        })
    }

    fn close_loop_body(
        &self,
        body_route: &StructuredRoute,
        loop_state: &StructuredLayoutState,
    ) -> Result<StructuredLoopClosure, CompileError> {
        let body_tail = self.reconcile_layout(&body_route.exit_l2p, loop_state.as_slice())?;
        let mut body_ops = body_route.ops.clone();
        body_ops.extend(body_tail.ops.clone());

        Ok(StructuredLoopClosure {
            body_ops,
            cost: body_route.cost + body_tail.cost,
            log_fidelity: body_route.log_fidelity + body_tail.log_fidelity,
        })
    }

    fn structured_layout_candidate_limit(&self) -> usize {
        if self.config.vf2_policy == Vf2Policy::Disabled {
            0
        } else {
            self.config.vf2_seed_top_k.min(4)
        }
    }

    fn best_program_layouts(
        &self,
        program: &PreparedProgram,
    ) -> Result<Vec<StructuredLayoutState>, CompileError> {
        let prepared = program.flatten_interaction_circuit();
        if prepared.operations.is_empty() {
            return Ok(vec![]);
        }
        self.best_prepared_layouts(&prepared)
    }

    fn best_items_layouts(
        &self,
        program: &PreparedProgram,
        items: &[PreparedProgramItem],
    ) -> Result<Vec<StructuredLayoutState>, CompileError> {
        if items.is_empty() {
            return Ok(vec![]);
        }
        let continuation_program = PreparedProgram {
            logical_qubits: program.logical_qubits.clone(),
            parameters: program.parameters.clone(),
            items: items.to_vec(),
        };
        self.best_program_layouts(&continuation_program)
    }

    fn best_prepared_layouts(
        &self,
        prepared: &PreparedCircuit,
    ) -> Result<Vec<StructuredLayoutState>, CompileError> {
        let top_k = self.structured_layout_candidate_limit();
        if prepared.operations.is_empty() || top_k == 0 {
            return Ok(vec![]);
        }

        let vf2 = Vf2Mapping::from_adapter(self.topology.clone());
        let options = Vf2CandidateOptions {
            top_k,
            weights: self.config.vf2_seed_weights,
            ..Vf2CandidateOptions::default()
        };
        Ok(vf2
            .find_prepared_layout_candidate_indices(prepared, Some(options))?
            .into_iter()
            .map(|l2p| StructuredLayoutState { l2p })
            .collect())
    }

    fn map_passthrough(&self, op: &PreparedPassthroughOp, entry_l2p: &[usize]) -> Operation {
        let mapped_qubits: Vec<Qubit> = op
            .logical_qubits
            .iter()
            .map(|&logical| self.topology.physical_qubits[entry_l2p[logical]])
            .collect();
        map_operation_qubits(&op.op, &mapped_qubits)
    }

    fn extend_route(&self, current: &mut StructuredRoute, next: StructuredRoute) {
        current.ops.extend(next.ops);
        current.exit_l2p = next.exit_l2p;
        current.cost += next.cost;
        current.log_fidelity += next.log_fidelity;
        current.objective = self.routing_objective(current.cost, current.log_fidelity);
    }

    fn identity_route(entry_l2p: &[usize]) -> StructuredRoute {
        StructuredRoute {
            exit_l2p: entry_l2p.to_vec(),
            ops: Vec::new(),
            cost: 0,
            log_fidelity: 0.0,
            objective: 0.0,
        }
    }

    /// Executes SABRE routing using a specific initial mapping provided by the Genetic Algorithm.
    ///
    /// Unlike `execute`, which generates its own initial layouts (via random sampling or VF2),
    /// this function forces the router to start from the provided `initial_mapping`. This is
    /// crucial for the Genetic Algorithm to evaluate the fitness of specific individuals.
    pub fn execute_with_genetic_algorithm(
        &mut self,
        circuit: &Circuit,
        initial_mapping: Vec<usize>,
    ) -> Result<Circuit, CompileError> {
        let prepared = preprocess_circuit(circuit)?;
        let reverse_circuit = circuit.inverse()?;
        let reverse_prepared = preprocess_circuit(&reverse_circuit)?;

        let logical_width = prepared.logical_qubits.len();
        let available_nodes = self.usable_nodes();

        if logical_width > available_nodes.len() {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: available_nodes.len(),
            });
        }

        if initial_mapping.len() != logical_width {
            return Err(CompileError::Internal(format!(
                "GA initial mapping size mismatch: expected {}, got {}",
                logical_width,
                initial_mapping.len()
            )));
        }

        let original_info = self.build_circuit_info(&prepared, logical_width)?;
        let reverse_info = self.build_circuit_info(&reverse_prepared, logical_width)?;

        let repeat_iters = self.config.repeat_iterations;
        let swap_iters = self.config.swap_iterations.max(1);

        let mut best_group: Option<AnsGroup> = None;
        let mut current_mapping = initial_mapping;

        for iter in 0..=repeat_iters {
            let forward_group =
                self.execute_routing(&original_info, &prepared, &current_mapping, swap_iters)?;

            if best_group
                .as_ref()
                .map(|g| self.group_better(&forward_group, g))
                .unwrap_or(true)
            {
                best_group = Some(forward_group.clone());
            }

            if iter == repeat_iters {
                break;
            }

            let best_ref = best_group
                .as_ref()
                .ok_or_else(|| CompileError::Internal("missing best SABRE group".into()))?;

            // 执行后向路由来优化映射
            let reverse_group = self.execute_routing(
                &reverse_info,
                &reverse_prepared,
                &best_ref.final_l2p,
                swap_iters,
            )?;

            // 将后向路由的结果作为下一次前向路由的起点
            current_mapping = reverse_group.final_l2p;
        }

        let best_group = best_group.ok_or(CompileError::SabreRoutingStuck)?;

        // 5. 更新 Mapper 的内部状态 (Logic -> Physical)
        self.logic2phy = best_group
            .final_l2p
            .iter()
            .map(|&p| self.topology.physical_qubits[p])
            .collect();

        self.phy2logic.clear();
        for (logical, &physical) in self.logic2phy.iter().enumerate() {
            self.phy2logic.insert(physical, logical);
        }

        // 6. 重建并返回映射后的物理量子线路
        let mapped_ops = self.replay_ops(&prepared, &original_info, &best_group);
        Ok(build_output_circuit_from_source(circuit, mapped_ops))
    }

    /// Internal helper for replay ops.
    fn replay_ops(
        &self,
        prepared: &PreparedCircuit,
        info: &GateDependencyDag,
        ans_group: &AnsGroup,
    ) -> Vec<Operation> {
        let mut mapped = Vec::new();

        for &(op_idx, logical) in &info.initial_single_qubit_ops {
            let physical = self.topology.physical_qubits[ans_group.initial_l2p[logical]];
            let op = &prepared.operations[op_idx].op;
            mapped.push(map_operation_qubits(op, &[physical]));
        }

        for step in &ans_group.steps {
            match step {
                AnsStep::Op(op) => mapped.push(op.clone()),
                AnsStep::Swap { u, v } => {
                    let qubits = [
                        self.topology.physical_qubits[*u],
                        self.topology.physical_qubits[*v],
                    ];
                    mapped.push(self.standard_op(StandardGate::SWAP, &qubits));
                }
                AnsStep::Bridge { u, v, bridge } => {
                    let u = self.topology.physical_qubits[*u];
                    let v = self.topology.physical_qubits[*v];
                    let b = self.topology.physical_qubits[*bridge];
                    mapped.push(self.standard_op(StandardGate::CX, &[b, v]));
                    mapped.push(self.standard_op(StandardGate::CX, &[u, b]));
                    mapped.push(self.standard_op(StandardGate::CX, &[b, v]));
                    mapped.push(self.standard_op(StandardGate::CX, &[u, b]));
                }
            }
        }

        mapped
    }

    /// Internal helper for standard op.
    fn standard_op(&self, gate: StandardGate, qubits: &[Qubit]) -> Operation {
        Operation {
            instruction: Instruction::Standard(gate),
            qubits: qubits.iter().copied().collect(),
            params: SmallVec::new(),
            label: None,
        }
    }

    /// Internal helper for usable nodes.
    fn usable_nodes(&self) -> Vec<usize> {
        let mut nodes = self.topology.largest_component.clone();
        nodes.sort_by_key(|idx| self.topology.physical_qubits[*idx].id());
        nodes
    }

    /// Internal helper for random initial mapping.
    fn random_initial_mapping(
        &mut self,
        available_nodes: &[usize],
        logical_width: usize,
    ) -> Vec<usize> {
        let mut nodes = available_nodes.to_vec();
        nodes.shuffle(&mut self.rng);
        nodes.truncate(logical_width);
        nodes
    }

    /// Internal helper for build circuit info.
    fn build_circuit_info(
        &self,
        prepared: &PreparedCircuit,
        logical_width: usize,
    ) -> Result<GateDependencyDag, CompileError> {
        let graph = GateGraph::from_prepared(prepared)?;
        let view = graph.dependency_view(logical_width);

        let mut indegree = Vec::with_capacity(view.nodes.len() + 1);
        indegree.push(0);
        for node in &view.nodes {
            indegree.push(node.indegree);
        }

        Ok(GateDependencyDag {
            nodes: view.nodes,
            indegree,
            initial_single_qubit_ops: view.initial_single_qubit_ops,
            front_layer: view.front_layer,
        })
    }

    /// Internal helper for execute routing.
    fn execute_routing(
        &mut self,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        initial_mapping: &[usize],
        swap_iterations: usize,
    ) -> Result<AnsGroup, CompileError> {
        let logical_width = prepared.logical_qubits.len();
        if initial_mapping.len() != logical_width {
            return Err(CompileError::Internal(format!(
                "initial mapping size mismatch: expected {}, got {}",
                logical_width,
                initial_mapping.len()
            )));
        }

        let mut best_group: Option<AnsGroup> = None;

        for _ in 0..swap_iterations {
            let mut phy2logic = vec![None; self.topology.num_qubits()];
            for (logical, &physical) in initial_mapping.iter().enumerate() {
                phy2logic[physical] = Some(logical);
            }

            let mut state = RoutingState {
                logic2phy: initial_mapping.to_vec(),
                phy2logic,
                pre_number: info.indegree.clone(),
                front_layer: info.front_layer.clone(),
                ans_steps: Vec::new(),
                decay: vec![1.0; self.topology.num_qubits()],
                decay_time: 0,
                weight_gates: vec![Vec::new(); self.topology.num_qubits()],
                preprocessing_h: 0.0,
            };

            if !self.execute_once(info, prepared, &mut state)? {
                continue;
            }

            let cost = self.calculate_cost(&state.ans_steps);
            let log_fidelity = self.predict_log_fidelity(&state.ans_steps);
            let objective = self.routing_objective(cost, log_fidelity);
            let candidate = AnsGroup {
                initial_l2p: initial_mapping.to_vec(),
                final_l2p: state.logic2phy.clone(),
                steps: state.ans_steps.clone(),
                cost,
                log_fidelity,
                objective,
            };

            if best_group
                .as_ref()
                .map(|g| self.group_better(&candidate, g))
                .unwrap_or(true)
            {
                best_group = Some(candidate);
            }
        }

        best_group.ok_or(CompileError::SabreRoutingStuck)
    }

    /// Internal helper for calculate cost.
    fn calculate_cost(&self, steps: &[AnsStep]) -> usize {
        steps.iter().map(AnsStep::cost).sum()
    }

    /// Internal helper for numeric weight sanitization.
    fn sanitize_weight(&self, value: f64, default: f64) -> f64 {
        if value.is_finite() && value >= 0.0 {
            value
        } else {
            default
        }
    }

    /// Internal helper for local swap-fidelity weight.
    fn swap_fidelity_weight(&self) -> f64 {
        if self.config.field_mode {
            self.sanitize_weight(self.config.swap_fidelity_weight, 0.25)
        } else {
            0.0
        }
    }

    /// Internal helper for global gate-cost weight.
    fn gate_cost_weight(&self) -> f64 {
        self.sanitize_weight(self.config.gate_cost_weight, 1.0)
    }

    /// Internal helper for global predicted-fidelity weight.
    fn predicted_fidelity_weight(&self) -> f64 {
        self.sanitize_weight(self.config.predicted_fidelity_weight, 0.1)
    }

    /// Internal helper for computing routing objective.
    fn routing_objective(&self, cost: usize, log_fidelity: f64) -> f64 {
        self.gate_cost_weight() * cost as f64 + self.predicted_fidelity_weight() * (-log_fidelity)
    }

    /// Internal helper for operation-edge index lookup.
    fn operation_edge_indices(&self, op: &Operation) -> Option<(usize, usize)> {
        if op.qubits.len() != 2 {
            return None;
        }

        let u = self.physical_index(op.qubits[0])?;
        let v = self.physical_index(op.qubits[1])?;
        Some((u, v))
    }

    /// Internal helper for physical index lookup.
    fn physical_index(&self, q: Qubit) -> Option<usize> {
        self.topology
            .physical_qubits
            .binary_search_by_key(&q.id(), Qubit::id)
            .ok()
    }

    /// Internal helper for edge log-fidelity.
    fn edge_log_fidelity(&self, u: usize, v: usize) -> f64 {
        self.topology
            .edge_fidelity(u, v)
            .max(self.fidelity_eps)
            .ln()
    }

    /// Internal helper for routed-step log-fidelity prediction.
    fn predict_log_fidelity(&self, steps: &[AnsStep]) -> f64 {
        let mut log_fidelity = 0.0;

        for step in steps {
            match step {
                AnsStep::Op(op) => {
                    if let Some((u, v)) = self.operation_edge_indices(op) {
                        log_fidelity += self.edge_log_fidelity(u, v);
                    }
                }
                AnsStep::Swap { u, v } => {
                    log_fidelity += 3.0 * self.edge_log_fidelity(*u, *v);
                }
                AnsStep::Bridge { u, v, bridge } => {
                    log_fidelity += 2.0 * self.edge_log_fidelity(*u, *bridge);
                    log_fidelity += 2.0 * self.edge_log_fidelity(*bridge, *v);
                }
            }
        }

        log_fidelity
    }

    /// Internal helper for epsilon-aware float comparison.
    fn cmp_f64_with_eps(&self, lhs: f64, rhs: f64) -> Ordering {
        if (lhs - rhs).abs() <= self.objective_eps {
            Ordering::Equal
        } else {
            lhs.total_cmp(&rhs)
        }
    }

    /// Internal helper for deterministic ans-step comparison.
    fn compare_steps(&self, lhs: &[AnsStep], rhs: &[AnsStep]) -> Ordering {
        let common = lhs.len().min(rhs.len());
        for i in 0..common {
            let ord = self.compare_step(&lhs[i], &rhs[i]);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        lhs.len().cmp(&rhs.len())
    }

    /// Internal helper for deterministic step comparison.
    fn compare_step(&self, lhs: &AnsStep, rhs: &AnsStep) -> Ordering {
        let rank = |step: &AnsStep| match step {
            AnsStep::Op(_) => 0u8,
            AnsStep::Swap { .. } => 1u8,
            AnsStep::Bridge { .. } => 2u8,
        };

        let rank_ord = rank(lhs).cmp(&rank(rhs));
        if rank_ord != Ordering::Equal {
            return rank_ord;
        }

        match (lhs, rhs) {
            (AnsStep::Op(a), AnsStep::Op(b)) => {
                let inst_ord = format!("{:?}", a.instruction).cmp(&format!("{:?}", b.instruction));
                if inst_ord != Ordering::Equal {
                    return inst_ord;
                }

                let a_qids: Vec<u32> = a.qubits.iter().map(Qubit::id).collect();
                let b_qids: Vec<u32> = b.qubits.iter().map(Qubit::id).collect();
                let q_ord = a_qids.cmp(&b_qids);
                if q_ord != Ordering::Equal {
                    return q_ord;
                }

                format!("{:?}", a.params).cmp(&format!("{:?}", b.params))
            }
            (AnsStep::Swap { u: au, v: av }, AnsStep::Swap { u: bu, v: bv }) => {
                normalize_index_pair(*au, *av).cmp(&normalize_index_pair(*bu, *bv))
            }
            (
                AnsStep::Bridge {
                    u: au,
                    v: av,
                    bridge: ab,
                },
                AnsStep::Bridge {
                    u: bu,
                    v: bv,
                    bridge: bb,
                },
            ) => (*au, *av, *ab).cmp(&(*bu, *bv, *bb)),
            _ => Ordering::Equal,
        }
    }

    /// Internal helper for rated-swap comparison.
    fn compare_rated_swaps(&self, lhs: &RatedSwap, rhs: &RatedSwap) -> Ordering {
        lhs.score
            .total_cmp(&rhs.score)
            .then_with(|| lhs.fidelity_penalty.total_cmp(&rhs.fidelity_penalty))
            .then_with(|| lhs.distance_term.total_cmp(&rhs.distance_term))
            .then_with(|| {
                normalize_index_pair(lhs.u, lhs.v).cmp(&normalize_index_pair(rhs.u, rhs.v))
            })
    }

    /// Internal helper for ans-group comparison.
    fn compare_groups(&self, lhs: &AnsGroup, rhs: &AnsGroup) -> Ordering {
        self.cmp_f64_with_eps(lhs.objective, rhs.objective)
            .then_with(|| lhs.cost.cmp(&rhs.cost))
            .then_with(|| self.cmp_f64_with_eps(rhs.log_fidelity, lhs.log_fidelity))
            .then_with(|| lhs.initial_l2p.cmp(&rhs.initial_l2p))
            .then_with(|| lhs.final_l2p.cmp(&rhs.final_l2p))
            .then_with(|| self.compare_steps(&lhs.steps, &rhs.steps))
    }

    /// Internal helper for ans-group preference.
    fn group_better(&self, candidate: &AnsGroup, best: &AnsGroup) -> bool {
        self.compare_groups(candidate, best) == Ordering::Less
    }

    fn structured_route_better(&self, candidate: &StructuredRoute, best: &StructuredRoute) -> bool {
        self.compare_structured_routes(candidate, best) == Ordering::Less
    }

    fn compare_structured_routes(&self, lhs: &StructuredRoute, rhs: &StructuredRoute) -> Ordering {
        self.cmp_f64_with_eps(lhs.objective, rhs.objective)
            .then_with(|| lhs.cost.cmp(&rhs.cost))
            .then_with(|| self.cmp_f64_with_eps(rhs.log_fidelity, lhs.log_fidelity))
            .then_with(|| lhs.exit_l2p.cmp(&rhs.exit_l2p))
            .then_with(|| lhs.ops.len().cmp(&rhs.ops.len()))
    }

    fn reconcile_layout(
        &self,
        from_l2p: &[usize],
        target_l2p: &[usize],
    ) -> Result<LayoutTransition, CompileError> {
        if from_l2p.len() != target_l2p.len() {
            return Err(CompileError::Internal(format!(
                "layout reconciliation size mismatch: {} vs {}",
                from_l2p.len(),
                target_l2p.len()
            )));
        }

        let mut logic2phy = from_l2p.to_vec();
        let mut phy2logic = vec![None; self.topology.num_qubits()];
        for (logical, &physical) in logic2phy.iter().enumerate() {
            phy2logic[physical] = Some(logical);
        }

        let mut ops = Vec::new();
        let mut cost = 0usize;
        let mut log_fidelity = 0.0;
        let max_steps = self
            .topology
            .num_qubits()
            .saturating_mul(from_l2p.len().max(1))
            .saturating_mul(8);

        let mut steps = 0usize;
        while logic2phy != target_l2p {
            if steps >= max_steps {
                return Err(CompileError::SabreRoutingStuck);
            }
            steps += 1;

            let mut choice: Option<(bool, u32, usize, Vec<usize>)> = None;
            for logical in 0..logic2phy.len() {
                let src = logic2phy[logical];
                let dst = target_l2p[logical];
                if src == dst {
                    continue;
                }
                let Some(path) = self.shortest_path_indices(src, dst) else {
                    continue;
                };
                if path.len() < 2 {
                    continue;
                }
                let target_free = phy2logic[dst].is_none();
                let dist = self.topology.dist[src][dst];
                let candidate = (target_free, dist, logical, path);
                let should_update = match &choice {
                    None => true,
                    Some((best_free, best_dist, best_logical, _)) => {
                        candidate.0 > *best_free
                            || (candidate.0 == *best_free
                                && (candidate.1 < *best_dist
                                    || (candidate.1 == *best_dist && candidate.2 < *best_logical)))
                    }
                };
                if should_update {
                    choice = Some(candidate);
                }
            }

            let Some((_, _, _, path)) = choice else {
                return Err(CompileError::SabreRoutingStuck);
            };
            let u = path[0];
            let v = path[1];

            let logic_u = phy2logic[u];
            let logic_v = phy2logic[v];
            phy2logic[u] = logic_v;
            phy2logic[v] = logic_u;
            if let Some(logical) = logic_u {
                logic2phy[logical] = v;
            }
            if let Some(logical) = logic_v {
                logic2phy[logical] = u;
            }

            ops.push(self.standard_op(
                StandardGate::SWAP,
                &[
                    self.topology.physical_qubits[u],
                    self.topology.physical_qubits[v],
                ],
            ));
            cost += 3;
            log_fidelity += 3.0 * self.edge_log_fidelity(u, v);
        }

        Ok(LayoutTransition {
            ops,
            cost,
            log_fidelity,
        })
    }

    fn shortest_path_indices(&self, src: usize, dst: usize) -> Option<Vec<usize>> {
        if src == dst {
            return Some(vec![src]);
        }

        let n = self.topology.num_qubits();
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[src] = true;
        queue.push_back(src);

        while let Some(node) = queue.pop_front() {
            if node == dst {
                break;
            }
            for &next in &self.topology.neighbors[node] {
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                prev[next] = node;
                queue.push_back(next);
            }
        }

        if !visited[dst] {
            return None;
        }

        let mut path = vec![dst];
        let mut now = dst;
        while now != src {
            now = prev[now];
            if now == usize::MAX {
                return None;
            }
            path.push(now);
        }
        path.reverse();
        Some(path)
    }

    /// Internal helper for execute once.
    fn execute_once(
        &mut self,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) -> Result<bool, CompileError> {
        self.execute_2q_gates(info, prepared, state)?;
        if state.front_layer.is_empty() {
            return Ok(true);
        }

        let mut history_selection: Vec<(usize, usize)> = Vec::new();

        loop {
            let mut rated_swaps = self.obtain_swaps(info, state);
            if rated_swaps.is_empty() {
                return Ok(false);
            }

            rated_swaps.sort_by(|a, b| self.compare_rated_swaps(a, b));
            let best_score = rated_swaps[0].score;

            let top_score_count = rated_swaps
                .iter()
                .take_while(|s| (s.score - best_score).abs() <= self.sample_eps)
                .count();

            let mut selected = if top_score_count > 1 {
                let pick = self.rng.random_range(0..top_score_count);
                rated_swaps[pick].clone()
            } else {
                rated_swaps[0].clone()
            };

            for candidate in &rated_swaps {
                let pair = normalize_index_pair(candidate.u, candidate.v);
                if !history_selection.contains(&pair) {
                    selected = candidate.clone();
                    break;
                }
            }

            self.execute_rated_swap(&selected, info, prepared, state)?;
            self.execute_2q_gates(info, prepared, state)?;

            if state.front_layer.is_empty() {
                return Ok(true);
            }

            if let Some(AnsStep::Swap { u, v }) = state.ans_steps.last() {
                history_selection.push(normalize_index_pair(*u, *v));
            } else {
                history_selection.clear();
            }
        }
    }

    /// Internal helper for execute 2q gates.
    fn execute_2q_gates(
        &mut self,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) -> Result<(), CompileError> {
        loop {
            let mut executable = Vec::new();
            let mut front_layer: Vec<usize> = state.front_layer.iter().copied().collect();
            front_layer.sort_unstable();

            for gateid in front_layer {
                let Some(gate) = info.node(gateid) else {
                    continue;
                };
                if gate.logical_qubits.len() != 2 {
                    continue;
                }

                let can_exec = self.can_execute(gate, state);
                let can_bridge = !can_exec && self.can_bridge(gate, info, prepared, state);

                if can_exec {
                    executable.push(gateid);
                    self.add_ans_2qgate(gate, prepared, state);
                    for &attachid in &gate.attach_ids {
                        if let Some(attach_gate) = info.node(attachid) {
                            self.add_ans_1qgate(attach_gate, prepared, state);
                        }
                    }
                } else if can_bridge {
                    executable.push(gateid);
                    self.add_bridge_gate(gate, state)?;
                    for &attachid in &gate.attach_ids {
                        if let Some(attach_gate) = info.node(attachid) {
                            self.add_ans_1qgate(attach_gate, prepared, state);
                        }
                    }
                }
            }

            if executable.is_empty() {
                break;
            }

            for gateid in executable {
                state.front_layer.remove(&gateid);
                if let Some(gate) = info.node(gateid) {
                    for &succ in &gate.next_ids {
                        if succ < state.pre_number.len() {
                            state.pre_number[succ] = state.pre_number[succ].saturating_sub(1);
                            if state.pre_number[succ] == 0 {
                                state.front_layer.insert(succ);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Internal helper for can execute.
    fn can_execute(&self, gate: &DependencyNode, state: &RoutingState) -> bool {
        let u = state.logic2phy[gate.logical_qubits[0]];
        let v = state.logic2phy[gate.logical_qubits[1]];
        self.topology.is_adjacent(u, v)
    }

    /// Internal helper for can bridge.
    fn can_bridge(
        &self,
        gate: &DependencyNode,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &RoutingState,
    ) -> bool {
        let op = &prepared.operations[gate.op_index].op;
        if !is_cx(op) {
            return false;
        }

        let fixed_u = state.logic2phy[gate.logical_qubits[0]];
        let fixed_v = state.logic2phy[gate.logical_qubits[1]];

        if self.topology.dist[fixed_u][fixed_v] != 2 {
            return false;
        }

        self.calculate_bridge_value(fixed_u, fixed_v, &gate.next_ids, info, state)
    }

    /// Internal helper for calculate bridge value.
    fn calculate_bridge_value(
        &self,
        endpoint_u: usize,
        endpoint_v: usize,
        next_gids: &[usize],
        info: &GateDependencyDag,
        state: &RoutingState,
    ) -> bool {
        let bridge_point = self.topology.neighbors[endpoint_u]
            .iter()
            .find(|&&temp_bp| self.topology.neighbors[temp_bp].contains(&endpoint_v))
            .copied();

        let Some(bridge_point) = bridge_point else {
            return false;
        };

        let endpoints = [endpoint_u, endpoint_v];
        let mut connected_gids = next_gids.to_vec();

        for _ in 0..5 {
            let mut next_next_gids = Vec::new();
            for &gid in &connected_gids {
                let Some(next_gate) = info.node(gid) else {
                    continue;
                };
                if next_gate.logical_qubits.len() != 2 {
                    continue;
                }

                let next_u = state.logic2phy[next_gate.logical_qubits[0]];
                let next_v = state.logic2phy[next_gate.logical_qubits[1]];

                let origin_connected = self.topology.is_adjacent(next_u, next_v);
                let bridge_connected = if endpoints.contains(&next_u) && endpoints.contains(&next_v)
                {
                    self.topology.is_adjacent(next_u, bridge_point)
                        || self.topology.is_adjacent(next_v, bridge_point)
                } else if endpoints.contains(&next_u) {
                    if next_v == bridge_point {
                        return true;
                    }
                    self.topology.is_adjacent(next_u, bridge_point)
                } else if endpoints.contains(&next_v) {
                    if next_u == bridge_point {
                        return true;
                    }
                    self.topology.is_adjacent(next_v, bridge_point)
                } else {
                    continue;
                };

                if origin_connected && !bridge_connected {
                    return true;
                } else if !origin_connected && bridge_connected {
                    return false;
                }

                next_next_gids.extend(next_gate.next_ids.iter().copied());
            }
            connected_gids = next_next_gids;
        }

        false
    }

    /// Internal helper for add ans 2qgate.
    fn add_ans_2qgate(
        &self,
        gate: &DependencyNode,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) {
        let u = self.topology.physical_qubits[state.logic2phy[gate.logical_qubits[0]]];
        let v = self.topology.physical_qubits[state.logic2phy[gate.logical_qubits[1]]];
        let op = &prepared.operations[gate.op_index].op;
        let mapped_op = map_operation_qubits(op, &[u, v]);
        state.ans_steps.push(AnsStep::Op(mapped_op));
    }

    /// Internal helper for add ans 1qgate.
    fn add_ans_1qgate(
        &self,
        gate: &DependencyNode,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) {
        let u = self.topology.physical_qubits[state.logic2phy[gate.logical_qubits[0]]];
        let op = &prepared.operations[gate.op_index].op;
        let mapped_op = map_operation_qubits(op, &[u]);
        state.ans_steps.push(AnsStep::Op(mapped_op));
    }

    /// Internal helper for add bridge gate.
    fn add_bridge_gate(
        &self,
        gate: &DependencyNode,
        state: &mut RoutingState,
    ) -> Result<(), CompileError> {
        let fixed_u = state.logic2phy[gate.logical_qubits[0]];
        let fixed_v = state.logic2phy[gate.logical_qubits[1]];
        let bridge = self.topology.neighbors[fixed_u]
            .iter()
            .find(|&&x| self.topology.neighbors[fixed_v].contains(&x))
            .copied()
            .ok_or_else(|| {
                CompileError::Internal(format!(
                    "failed to find bridge point between {} and {}",
                    fixed_u, fixed_v
                ))
            })?;

        state.ans_steps.push(AnsStep::Bridge {
            u: fixed_u,
            v: fixed_v,
            bridge,
        });
        Ok(())
    }

    /// Internal helper for preprocessing h.
    fn preprocessing_h(&self, info: &GateDependencyDag, state: &mut RoutingState) {
        let mut preprocessing_h = 0.0;
        let mut weight_gates = vec![Vec::new(); self.topology.num_qubits()];
        let mut e_queue = Vec::new();

        let f_count = state.front_layer.len() as f64;
        if f_count > 0.0 {
            let mut front_sorted: Vec<usize> = state.front_layer.iter().copied().collect();
            front_sorted.sort_unstable();
            for &gateid in &front_sorted {
                if let Some(gate) = info.node(gateid) {
                    if gate.logical_qubits.len() != 2 {
                        continue;
                    }
                    let u = state.logic2phy[gate.logical_qubits[0]];
                    let v = state.logic2phy[gate.logical_qubits[1]];
                    preprocessing_h += self.topology.dist[u][v] as f64 / f_count;
                    weight_gates[u].push((gateid, 1.0 / f_count));
                    weight_gates[v].push((gateid, 1.0 / f_count));
                    e_queue.push(gateid);
                }
            }
        }

        let mut e_set = Vec::new();
        let mut dec_queue = Vec::new();

        while e_set.len() < self.config.size_e && !e_queue.is_empty() {
            let gateid = e_queue.pop().unwrap();
            let Some(gate) = info.node(gateid) else {
                continue;
            };

            dec_queue.push(gateid);
            for &succid in &gate.next_ids {
                if succid < state.pre_number.len() {
                    state.pre_number[succid] = state.pre_number[succid].saturating_sub(1);
                    if state.pre_number[succid] == 0 {
                        e_set.push(succid);
                        e_queue.push(succid);
                    }
                }
            }
        }

        if e_set.len() > self.config.size_e {
            e_set.truncate(self.config.size_e);
        }

        let e_count = e_set.len() as f64;
        if e_count > 0.0 {
            for &gateid in &e_set {
                let Some(gate) = info.node(gateid) else {
                    continue;
                };
                if gate.logical_qubits.len() != 2 {
                    continue;
                }
                let u = state.logic2phy[gate.logical_qubits[0]];
                let v = state.logic2phy[gate.logical_qubits[1]];
                preprocessing_h += self.topology.dist[u][v] as f64 / e_count * self.config.w;
                weight_gates[u].push((gateid, self.config.w / e_count));
                weight_gates[v].push((gateid, self.config.w / e_count));
            }
        }

        for &gateid in &dec_queue {
            if let Some(gate) = info.node(gateid) {
                for &succid in &gate.next_ids {
                    if succid < state.pre_number.len() {
                        state.pre_number[succid] = state.pre_number[succid].saturating_add(1);
                    }
                }
            }
        }

        state.weight_gates = weight_gates;
        state.preprocessing_h = preprocessing_h;
    }

    /// Internal helper for rated swap.
    fn rated_swap(
        &self,
        u: usize,
        v: usize,
        info: &GateDependencyDag,
        state: &RoutingState,
    ) -> RatedSwap {
        let mut distance_term = state.preprocessing_h;

        for &(gateid, coff) in &state.weight_gates[u] {
            if let Some(gate) = info.node(gateid) {
                let mut vv = state.logic2phy[gate.logical_qubits[0]];
                if vv == u {
                    vv = state.logic2phy[gate.logical_qubits[1]];
                }
                if vv == v {
                    continue;
                }
                distance_term +=
                    coff * (self.topology.dist[v][vv] as f64 - self.topology.dist[u][vv] as f64);
            }
        }

        for &(gateid, coff) in &state.weight_gates[v] {
            if let Some(gate) = info.node(gateid) {
                let mut uu = state.logic2phy[gate.logical_qubits[0]];
                if uu == v {
                    uu = state.logic2phy[gate.logical_qubits[1]];
                }
                if uu == u {
                    continue;
                }
                distance_term +=
                    coff * (self.topology.dist[u][uu] as f64 - self.topology.dist[v][uu] as f64);
            }
        }

        distance_term *= state.decay[u].max(state.decay[v]);
        let edge_fidelity = self.topology.edge_fidelity(u, v).max(0.0);
        let fidelity_penalty = -(edge_fidelity + self.fidelity_eps).ln();
        let score = distance_term + self.swap_fidelity_weight() * fidelity_penalty;

        RatedSwap {
            u,
            v,
            score,
            distance_term,
            fidelity_penalty,
        }
    }

    /// Internal helper for obtain swaps.
    fn obtain_swaps(&self, info: &GateDependencyDag, state: &mut RoutingState) -> Vec<RatedSwap> {
        self.preprocessing_h(info, state);

        let mut front_bits = HashSet::new();
        for &gateid in &state.front_layer {
            if let Some(gate) = info.node(gateid) {
                if gate.logical_qubits.len() != 2 {
                    continue;
                }
                front_bits.insert(state.logic2phy[gate.logical_qubits[0]]);
                front_bits.insert(state.logic2phy[gate.logical_qubits[1]]);
            }
        }

        let mut visited_pairs = HashSet::new();
        let mut swaps = Vec::new();

        let mut front_bits_sorted: Vec<usize> = front_bits.into_iter().collect();
        front_bits_sorted.sort_unstable();
        for &u in &front_bits_sorted {
            for &v in &self.topology.neighbors[u] {
                let key = normalize_index_pair(u, v);
                if visited_pairs.insert(key) {
                    swaps.push(self.rated_swap(u, v, info, state));
                }
            }
        }

        swaps
    }

    /// Internal helper for execute swap.
    fn execute_swap(&self, u: usize, v: usize, state: &mut RoutingState) {
        let logic_u = state.phy2logic[u];
        let logic_v = state.phy2logic[v];

        state.phy2logic[u] = logic_v;
        state.phy2logic[v] = logic_u;

        if let Some(logic_u) = logic_u {
            state.logic2phy[logic_u] = v;
        }
        if let Some(logic_v) = logic_v {
            state.logic2phy[logic_v] = u;
        }

        state.ans_steps.push(AnsStep::Swap { u, v });
    }

    /// Internal helper for find shortest path.
    fn find_shortest_path(
        &mut self,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) -> Result<(), CompileError> {
        let mut shortest = u32::MAX;
        let mut src = None;
        let mut dst = None;

        let mut front_sorted: Vec<usize> = state.front_layer.iter().copied().collect();
        front_sorted.sort_unstable();
        for &front_id in &front_sorted {
            let Some(gate) = info.node(front_id) else {
                continue;
            };
            if gate.logical_qubits.len() != 2 {
                continue;
            }

            let u = state.logic2phy[gate.logical_qubits[0]];
            let v = state.logic2phy[gate.logical_qubits[1]];
            let d = self.topology.dist[u][v];
            if d < shortest {
                shortest = d;
                src = Some(u);
                dst = Some(v);
            }
        }

        let (src, dst) = match (src, dst) {
            (Some(s), Some(t)) if s != t => (s, t),
            _ => return Ok(()),
        };

        let n = self.topology.num_qubits();
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();

        visited[src] = true;
        queue.push_back(src);

        while let Some(node) = queue.pop_front() {
            if node == dst {
                break;
            }
            for &next in &self.topology.neighbors[node] {
                if !visited[next] {
                    visited[next] = true;
                    prev[next] = node;
                    queue.push_back(next);
                }
            }
        }

        if !visited[dst] {
            return Ok(());
        }

        let mut path = vec![dst];
        let mut now = dst;
        while now != src {
            now = prev[now];
            if now == usize::MAX {
                return Ok(());
            }
            path.push(now);
        }
        path.reverse();

        if path.len() < 2 {
            return Ok(());
        }

        let mut i = 0usize;
        let mut j = path.len() - 1;

        while i + 1 < j {
            self.execute_swap(path[i], path[i + 1], state);
            i += 1;

            if i + 1 >= j {
                break;
            }

            self.execute_swap(path[j], path[j - 1], state);
            j -= 1;
        }

        self.execute_2q_gates(info, prepared, state)
    }

    /// Internal helper for check greedy strategy.
    fn check_greedy_strategy(
        &mut self,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) -> Result<(), CompileError> {
        let Some(AnsStep::Swap { u, v }) = state.ans_steps.last().cloned() else {
            return Ok(());
        };

        let mut remaining = self.config.greedy_strategy.saturating_sub(1);
        if remaining == 0 {
            return Ok(());
        }

        if state.ans_steps.len() < 2 {
            return Ok(());
        }

        let mut idx = state.ans_steps.len() - 2;
        loop {
            match &state.ans_steps[idx] {
                AnsStep::Op(_) => return Ok(()),
                AnsStep::Swap { u: pu, v: pv } => {
                    let p1 = normalize_index_pair(*pu, *pv);
                    let p2 = normalize_index_pair(u, v);
                    if p1 == p2 {
                        self.find_shortest_path(info, prepared, state)?;
                        return Ok(());
                    }
                }
                AnsStep::Bridge { .. } => {}
            }

            remaining -= 1;
            if remaining == 0 || idx == 0 {
                break;
            }
            idx -= 1;
        }

        Ok(())
    }

    /// Internal helper for execute rated swap.
    fn execute_rated_swap(
        &mut self,
        rated_swap: &RatedSwap,
        info: &GateDependencyDag,
        prepared: &PreparedCircuit,
        state: &mut RoutingState,
    ) -> Result<(), CompileError> {
        self.execute_swap(rated_swap.u, rated_swap.v, state);

        state.decay[rated_swap.u] += self.config.decay_coff;
        state.decay[rated_swap.v] += self.config.decay_coff;
        state.decay_time += 1;

        if state.decay_time % self.config.decay_reset_time.max(1) == 0 {
            state.decay.fill(1.0);
        }

        if self.config.greedy_strategy > 0 {
            self.check_greedy_strategy(info, prepared, state)?;
        }

        Ok(())
    }

    /// Internal helper for initial layout candidates.
    fn initial_layout_candidates(
        &mut self,
        prepared: &PreparedCircuit,
        available_nodes: &[usize],
        logical_width: usize,
        iterations: usize,
    ) -> Result<Vec<Vec<usize>>, CompileError> {
        let mut candidates = Vec::with_capacity(iterations);
        let mut seen = HashSet::new();
        let available_set: HashSet<usize> = available_nodes.iter().copied().collect();

        if self.config.vf2_policy != Vf2Policy::Disabled && self.config.vf2_seed_top_k > 0 {
            for seed in self.vf2_seed_layouts(prepared, &available_set)? {
                if seen.insert(seed.clone()) {
                    candidates.push(seed);
                }
                if candidates.len() >= iterations {
                    break;
                }
            }
        }

        let mut duplicate_retries = 0usize;
        while candidates.len() < iterations {
            let random_layout = self.random_initial_mapping(available_nodes, logical_width);
            if seen.insert(random_layout.clone()) {
                candidates.push(random_layout);
                duplicate_retries = 0;
            } else {
                duplicate_retries += 1;
                if duplicate_retries > iterations.saturating_mul(4) {
                    candidates.push(random_layout);
                    duplicate_retries = 0;
                }
            }
        }

        Ok(candidates)
    }

    /// Internal helper for vf2 seed layouts.
    fn vf2_seed_layouts(
        &self,
        prepared: &PreparedCircuit,
        available_nodes: &HashSet<usize>,
    ) -> Result<Vec<Vec<usize>>, CompileError> {
        let vf2 = Vf2Mapping::from_adapter(self.topology.clone());
        let options = Vf2CandidateOptions {
            top_k: self.config.vf2_seed_top_k,
            weights: self.config.vf2_seed_weights,
            ..Vf2CandidateOptions::default()
        };
        let layouts = vf2.find_prepared_layout_candidate_indices(prepared, Some(options))?;

        Ok(layouts
            .into_iter()
            .filter(|layout| layout.iter().all(|phy| available_nodes.contains(phy)))
            .collect())
    }
}

#[cfg(test)]
#[path = "sabre_test.rs"]
mod sabre_test;
