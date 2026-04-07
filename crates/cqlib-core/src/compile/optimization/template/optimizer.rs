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

//! Template matching and execution.
//!
//! The implementation in this module is intentionally scoped to compile-ready
//! circuits validated by `preprocess_circuit`:
//! - no control-flow operations
//! - no directives/delay operations
//! - only 1q/2q operations
//!
//! Matching behavior:
//! - exact instruction compatibility
//! - exact parameter compatibility (numeric or symbolic)
//! - commutation-aware DAG constraints
//!
//! Optimization behavior:
//! - substitutions are built from template identities via inverse gates
//! - candidates are selected by larger cost reduction first, then fidelity gain

use super::library::{CompiledTemplate, TemplateLibrary, compile_template};
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, CircuitParam, Operation, ParameterValue, Qubit};
use crate::compile::error::CompileError;
use crate::compile::graph::{CommutationView, GateGraph, GateNode};
use crate::compile::prepared::{PreparedCircuit, append_operation, preprocess_circuit};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

/// One template match result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TemplateMatch {
    /// `(template_node_id, circuit_node_id)` node mapping pairs.
    pub match_pairs: Vec<(usize, usize)>,
    /// Template logical index -> circuit logical index mapping.
    pub qubit_mapping: Vec<usize>,
}

/// Template matching entry type.
#[derive(Debug, Clone, Default)]
pub struct TemplateMatching;

impl TemplateMatching {
    /// Creates a new matcher.
    pub fn new() -> Self {
        Self
    }

    /// Executes template matching.
    ///
    /// `qubit_fixing_cnt` and `prune_param` are reserved compatibility knobs
    /// and are currently ignored.
    pub fn run(
        circuit: &Circuit,
        template: &Circuit,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Vec<TemplateMatch>, CompileError> {
        let _ = (qubit_fixing_cnt, prune_param);

        let c_prepared = preprocess_circuit(circuit)?;
        let c_graph = GateGraph::from_prepared(&c_prepared)?;
        let c_view = c_graph.commutation_view()?;
        let compiled_template = compile_template(template)?;

        Ok(collect_matches_for_nodes(
            &c_prepared,
            &c_graph,
            &c_view,
            &compiled_template,
            &(0..compiled_template.graph.size()).collect::<Vec<_>>(),
        ))
    }
}

/// Configuration used by template optimization.
#[derive(Debug, Clone, Default)]
pub struct TemplateOptimizationConfig {
    /// Reserved for future matching heuristics. Currently unused.
    pub qubit_fixing_cnt: Option<usize>,
    /// Reserved for future matching heuristics. Currently unused.
    pub prune_param: Option<(usize, usize)>,
}

impl TemplateOptimizationConfig {
    /// Creates a config from reserved compatibility fields.
    pub fn reserved(qubit_fixing_cnt: Option<usize>, prune_param: Option<(usize, usize)>) -> Self {
        Self {
            qubit_fixing_cnt,
            prune_param,
        }
    }
}

/// Template optimization entry type.
#[derive(Debug, Clone)]
pub struct TemplateOptimization {
    library: TemplateLibrary,
    config: TemplateOptimizationConfig,
}

impl TemplateOptimization {
    /// Creates an optimizer from a reusable template library.
    pub fn from_library(library: TemplateLibrary, config: TemplateOptimizationConfig) -> Self {
        Self { library, config }
    }

    /// Compatibility constructor from explicit template circuits.
    ///
    /// Templates must already satisfy compile-layer template constraints.
    pub fn new(
        templates: Vec<Circuit>,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Self {
        let mut library = TemplateLibrary::new();
        library
            .register_rules(templates)
            .expect("invalid template passed to TemplateOptimization::new");
        Self::from_library(
            library,
            TemplateOptimizationConfig::reserved(qubit_fixing_cnt, prune_param),
        )
    }

    /// Creates an optimizer from embedded default templates.
    pub fn with_default_templates(
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Self, CompileError> {
        let library = TemplateLibrary::with_default_rules()?;
        Ok(Self::from_library(
            library,
            TemplateOptimizationConfig::reserved(qubit_fixing_cnt, prune_param),
        ))
    }

    /// Creates an optimizer from a `.json` or `.qcis` template file.
    pub fn from_template_file(
        template_file: &str,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Self, CompileError> {
        let library = TemplateLibrary::from_template_file(template_file)?;
        Ok(Self::from_library(
            library,
            TemplateOptimizationConfig::reserved(qubit_fixing_cnt, prune_param),
        ))
    }

    /// Returns the backing template library.
    pub fn library(&self) -> &TemplateLibrary {
        &self.library
    }

    /// Returns the number of templates held by this optimizer.
    pub fn template_count(&self) -> usize {
        self.library.template_count()
    }

    /// Returns immutable template list.
    pub fn templates(&self) -> &[Circuit] {
        self.library.templates()
    }

    /// Executes one-pass template optimization.
    pub fn execute(&self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let _ = &self.config;

        let mut current = circuit.clone();
        for template_group in self.library.compiled_template_groups() {
            let mut best = current.clone();
            let mut best_cost = estimate_circuit_cost(&best);
            let mut best_penalty = estimate_circuit_penalty(&best);
            let mut best_len = best.operations().len();

            for template in template_group {
                let candidate = apply_template_substitutions(&current, template)?;
                let candidate_cost = estimate_circuit_cost(&candidate);
                let candidate_penalty = estimate_circuit_penalty(&candidate);
                let candidate_len = candidate.operations().len();

                let better_cost = candidate_cost < best_cost;
                let better_penalty =
                    candidate_cost == best_cost && candidate_penalty + 1e-12 < best_penalty;
                let better_len = candidate_cost == best_cost
                    && (candidate_penalty - best_penalty).abs() <= 1e-12
                    && candidate_len < best_len;

                if better_cost || better_penalty || better_len {
                    best = candidate;
                    best_cost = candidate_cost;
                    best_penalty = candidate_penalty;
                    best_len = candidate_len;
                }
            }

            current = best;
        }

        Ok(current)
    }

    /// Executes iterative optimization until no size decrease or max iterations.
    pub fn execute_iterative(
        &self,
        circuit: &Circuit,
        max_iterations: Option<usize>,
    ) -> Result<Circuit, CompileError> {
        let max_iters = max_iterations.unwrap_or(100);
        let mut current = circuit.clone();

        for _ in 0..max_iters {
            let next = self.execute(&current)?;
            if next.operations().len() >= current.operations().len() {
                break;
            }
            current = next;
        }

        Ok(current)
    }
}

/// Optimization candidate selected from one match.
#[derive(Debug, Clone)]
struct SubstitutionCandidate {
    /// Matched circuit node ids.
    matched_nodes: Vec<usize>,
    /// Replacement operation sequence.
    replacement: Vec<Operation>,
    /// Positive or zero gate-cost gain.
    cost_gain: i64,
    /// Fidelity-penalty gain (positive means higher predicted fidelity).
    fidelity_gain: f64,
}

/// Runs matching under one fixed qubit mapping.
fn match_under_mapping(
    circuit_graph: &GateGraph,
    template_graph: &GateGraph,
    circuit_view: &CommutationView,
    template_view: &CommutationView,
    template_node_ids: &[usize],
    qubit_mapping: &[usize],
) -> Vec<TemplateMatch> {
    if template_node_ids.is_empty() {
        return Vec::new();
    }

    let t_size = template_graph.size();
    let c_size = circuit_graph.size();

    let mut candidates = HashMap::<usize, Vec<usize>>::with_capacity(template_node_ids.len());
    for &t_id in template_node_ids {
        if t_id >= t_size {
            return Vec::new();
        }
        let Some(t_node) = template_graph.node(t_id) else {
            return Vec::new();
        };
        debug_assert_eq!(t_node.node_id, t_id);
        let mut t_cands = Vec::<usize>::new();
        for c_id in 0..c_size {
            let Some(c_node) = circuit_graph.node(c_id) else {
                continue;
            };
            debug_assert_eq!(c_node.node_id, c_id);
            if node_compatible(t_node, c_node, qubit_mapping) {
                t_cands.push(c_id);
            }
        }
        if t_cands.is_empty() {
            return Vec::new();
        }
        candidates.insert(t_id, t_cands);
    }

    let mut order: Vec<usize> = template_node_ids.to_vec();
    order.sort_by(|&a, &b| {
        let a_len = candidates.get(&a).map_or(usize::MAX, Vec::len);
        let b_len = candidates.get(&b).map_or(usize::MAX, Vec::len);
        let by_len = a_len.cmp(&b_len);
        if by_len == Ordering::Equal {
            a.cmp(&b)
        } else {
            by_len
        }
    });

    let mut assigned_t = vec![None::<usize>; t_size];
    let mut used_c = vec![false; c_size];
    let mut out = Vec::new();
    backtrack_matches(
        0,
        &order,
        &candidates,
        template_view,
        circuit_view,
        &mut assigned_t,
        &mut used_c,
        template_node_ids,
        qubit_mapping,
        &mut out,
    );
    out
}

/// Backtracking matcher over candidate node assignments.
#[allow(clippy::too_many_arguments)]
fn backtrack_matches(
    depth: usize,
    order: &[usize],
    candidates: &HashMap<usize, Vec<usize>>,
    template_view: &CommutationView,
    circuit_view: &CommutationView,
    assigned_t: &mut [Option<usize>],
    used_c: &mut [bool],
    template_node_ids: &[usize],
    qubit_mapping: &[usize],
    out: &mut Vec<TemplateMatch>,
) {
    if depth == order.len() {
        let mut pairs = Vec::with_capacity(template_node_ids.len());
        for &t_id in template_node_ids {
            let Some(c_opt) = assigned_t.get(t_id) else {
                return;
            };
            if let Some(c_id) = c_opt {
                pairs.push((t_id, *c_id));
            } else {
                return;
            }
        }
        pairs.sort_unstable();
        out.push(TemplateMatch {
            match_pairs: pairs,
            qubit_mapping: qubit_mapping.to_vec(),
        });
        return;
    }

    let t_id = order[depth];
    let Some(candidate_ids) = candidates.get(&t_id) else {
        return;
    };
    for &c_id in candidate_ids {
        if used_c[c_id] {
            continue;
        }
        if !assignment_is_consistent(t_id, c_id, assigned_t, template_view, circuit_view) {
            continue;
        }

        assigned_t[t_id] = Some(c_id);
        used_c[c_id] = true;
        backtrack_matches(
            depth + 1,
            order,
            candidates,
            template_view,
            circuit_view,
            assigned_t,
            used_c,
            template_node_ids,
            qubit_mapping,
            out,
        );
        used_c[c_id] = false;
        assigned_t[t_id] = None;
    }
}

/// Checks whether one assignment satisfies reachability constraints.
fn assignment_is_consistent(
    t_id: usize,
    c_id: usize,
    assigned_t: &[Option<usize>],
    template_view: &CommutationView,
    circuit_view: &CommutationView,
) -> bool {
    for (other_t, other_c_opt) in assigned_t.iter().enumerate() {
        let Some(other_c) = other_c_opt else {
            continue;
        };
        let template_before = template_view.is_reachable(other_t, t_id);
        let template_after = template_view.is_reachable(t_id, other_t);
        let circuit_before = circuit_view.is_reachable(*other_c, c_id);
        let circuit_after = circuit_view.is_reachable(c_id, *other_c);

        if template_before != circuit_before {
            return false;
        }
        if template_after != circuit_after {
            return false;
        }
    }
    true
}

/// Checks exact node compatibility under one qubit mapping.
fn node_compatible(template_node: &GateNode, circuit_node: &GateNode, mapping: &[usize]) -> bool {
    if template_node.logical_qubits.len() != circuit_node.logical_qubits.len() {
        return false;
    }
    if template_node.resolved_params != circuit_node.resolved_params {
        return false;
    }
    if instruction_signature(&template_node.op.instruction)
        != instruction_signature(&circuit_node.op.instruction)
    {
        return false;
    }

    for (&tq, &cq) in template_node
        .logical_qubits
        .iter()
        .zip(circuit_node.logical_qubits.iter())
    {
        let Some(&mapped) = mapping.get(tq) else {
            return false;
        };
        if mapped != cq {
            return false;
        }
    }
    true
}

/// Produces a deterministic instruction signature for matching.
fn instruction_signature(inst: &Instruction) -> String {
    match inst {
        Instruction::Standard(g) => format!("std:{:?}", g),
        Instruction::McGate(g) => format!("mc:{}:{:?}", g.num_ctrl_qubits(), g.base_gate()),
        Instruction::UnitaryGate(g) => format!("unitary:{}:{}", g.label(), g.num_qubits()),
        Instruction::CircuitGate(g) => format!("circuit:{}:{}", g.name(), g.num_qubits()),
        Instruction::Directive(d) => format!("directive:{:?}", d),
        Instruction::ControlFlowGate(g) => format!("control_flow:{:?}", g),
        Instruction::Delay => "delay".to_string(),
    }
}

/// Enumerates all injective template->circuit qubit mappings.
fn enumerate_qubit_mappings(template_width: usize, circuit_width: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(template_width);
    let mut used = vec![false; circuit_width];
    build_qubit_mappings_dfs(
        0,
        template_width,
        circuit_width,
        &mut cur,
        &mut used,
        &mut out,
    );
    out
}

/// DFS helper for qubit mapping enumeration.
fn build_qubit_mappings_dfs(
    depth: usize,
    template_width: usize,
    circuit_width: usize,
    cur: &mut Vec<usize>,
    used: &mut [bool],
    out: &mut Vec<Vec<usize>>,
) {
    if depth == template_width {
        out.push(cur.clone());
        return;
    }
    for q in 0..circuit_width {
        if used[q] {
            continue;
        }
        used[q] = true;
        cur.push(q);
        build_qubit_mappings_dfs(depth + 1, template_width, circuit_width, cur, used, out);
        cur.pop();
        used[q] = false;
    }
}

fn collect_matches_for_nodes(
    circuit_prepared: &PreparedCircuit,
    circuit_graph: &GateGraph,
    circuit_view: &CommutationView,
    compiled_template: &CompiledTemplate,
    template_node_ids: &[usize],
) -> Vec<TemplateMatch> {
    if template_node_ids.is_empty() {
        return Vec::new();
    }
    if compiled_template.prepared.logical_qubits.len() > circuit_prepared.logical_qubits.len() {
        return Vec::new();
    }

    let qubit_mappings = enumerate_qubit_mappings(
        compiled_template.prepared.logical_qubits.len(),
        circuit_prepared.logical_qubits.len(),
    );

    let mut out = HashSet::<TemplateMatch>::new();
    for mapping in qubit_mappings {
        out.extend(match_under_mapping(
            circuit_graph,
            &compiled_template.graph,
            circuit_view,
            &compiled_template.view,
            template_node_ids,
            &mapping,
        ));
    }
    let mut collected: Vec<TemplateMatch> = out.into_iter().collect();
    collected.sort_by(|a, b| a.match_pairs.cmp(&b.match_pairs));
    collected
}

/// Applies template substitutions selected by gate-cost and fidelity priority.
fn apply_template_substitutions(
    circuit: &Circuit,
    compiled_template: &CompiledTemplate,
) -> Result<Circuit, CompileError> {
    let prepared = preprocess_circuit(circuit)?;
    if compiled_template.prepared.operations.is_empty() {
        return Ok(circuit.clone());
    }
    if compiled_template.prepared.logical_qubits.len() > prepared.logical_qubits.len() {
        return Ok(circuit.clone());
    }

    let circuit_graph = GateGraph::from_prepared(&prepared)?;
    let circuit_view = circuit_graph.commutation_view()?;

    let mut candidates = Vec::<SubstitutionCandidate>::new();
    for subset in &compiled_template.node_subsets {
        let matches = collect_matches_for_nodes(
            &prepared,
            &circuit_graph,
            &circuit_view,
            compiled_template,
            subset,
        );
        for match_info in matches {
            let Some(candidate) =
                build_substitution_candidate(&prepared, compiled_template, &match_info)?
            else {
                continue;
            };
            candidates.push(candidate);
        }
    }

    if candidates.is_empty() {
        return Ok(circuit.clone());
    }

    candidates.sort_by(|a, b| {
        b.cost_gain
            .cmp(&a.cost_gain)
            .then_with(|| b.fidelity_gain.total_cmp(&a.fidelity_gain))
            .then_with(|| a.matched_nodes[0].cmp(&b.matched_nodes[0]))
    });

    let mut selected = Vec::new();
    let mut occupied = HashSet::<usize>::new();
    for candidate in candidates {
        if candidate
            .matched_nodes
            .iter()
            .any(|idx| occupied.contains(idx))
        {
            continue;
        }
        occupied.extend(candidate.matched_nodes.iter().copied());
        selected.push(candidate);
    }

    if selected.is_empty() {
        return Ok(circuit.clone());
    }

    let mut insertions: HashMap<usize, Vec<Operation>> = HashMap::new();
    let mut removed = HashSet::<usize>::new();
    for candidate in &selected {
        let first = candidate.matched_nodes[0];
        insertions.insert(first, candidate.replacement.clone());
        removed.extend(candidate.matched_nodes.iter().copied());
    }

    let mut output = Circuit::from_qubits(circuit.qubits())?;
    for op_idx in 0..prepared.operations.len() {
        if let Some(extra_ops) = insertions.remove(&op_idx) {
            for op in extra_ops {
                append_operation(&mut output, &op, prepared.parameters.as_slice())?;
            }
        }
        if removed.contains(&op_idx) {
            continue;
        }
        append_operation(
            &mut output,
            &prepared.operations[op_idx].op,
            prepared.parameters.as_slice(),
        )?;
    }
    for extra_ops in insertions.into_values() {
        for op in extra_ops {
            append_operation(&mut output, &op, prepared.parameters.as_slice())?;
        }
    }

    Ok(output)
}

/// Builds one substitution candidate and computes its ranking metrics.
fn build_substitution_candidate(
    prepared: &PreparedCircuit,
    compiled_template: &CompiledTemplate,
    match_info: &TemplateMatch,
) -> Result<Option<SubstitutionCandidate>, CompileError> {
    let replacement_ops =
        match build_replacement_ops_for_match(match_info, compiled_template, prepared)? {
            Some(ops) => ops,
            None => return Ok(None),
        };

    let mut matched_nodes: Vec<usize> = match_info.match_pairs.iter().map(|(_, c)| *c).collect();
    matched_nodes.sort_unstable();
    matched_nodes.dedup();
    if matched_nodes.is_empty() {
        return Ok(None);
    }

    let old_cost: i64 = matched_nodes
        .iter()
        .map(|&idx| estimate_op_cost(&prepared.operations[idx].op))
        .sum();
    let new_cost: i64 = replacement_ops.iter().map(estimate_op_cost).sum();
    let cost_gain = old_cost - new_cost;

    let old_penalty: f64 = matched_nodes
        .iter()
        .map(|&idx| estimate_fidelity_penalty(&prepared.operations[idx].op))
        .sum();
    let new_penalty: f64 = replacement_ops.iter().map(estimate_fidelity_penalty).sum();
    let fidelity_gain = old_penalty - new_penalty;

    if cost_gain < 0 {
        return Ok(None);
    }
    if cost_gain == 0 && fidelity_gain <= 1e-12 {
        return Ok(None);
    }

    Ok(Some(SubstitutionCandidate {
        matched_nodes,
        replacement: replacement_ops,
        cost_gain,
        fidelity_gain,
    }))
}

/// Builds replacement operations for one match using inverse unmatched template gates.
fn build_replacement_ops_for_match(
    match_info: &TemplateMatch,
    compiled_template: &CompiledTemplate,
    circuit_prepared: &PreparedCircuit,
) -> Result<Option<Vec<Operation>>, CompileError> {
    let matched_template_nodes: HashSet<usize> =
        match_info.match_pairs.iter().map(|(t, _)| *t).collect();
    if matched_template_nodes.is_empty() {
        return Ok(None);
    }

    let predecessors = collect_all_predecessors(&compiled_template.view, &matched_template_nodes);
    let mut successors = HashSet::<usize>::new();
    for node_id in 0..compiled_template.graph.size() {
        if matched_template_nodes.contains(&node_id) || predecessors.contains(&node_id) {
            continue;
        }
        successors.insert(node_id);
    }

    let mut predecessor_ids: Vec<usize> = predecessors.into_iter().collect();
    predecessor_ids.sort_unstable_by(|a, b| b.cmp(a));
    let mut successor_ids: Vec<usize> = successors.into_iter().collect();
    successor_ids.sort_unstable_by(|a, b| b.cmp(a));

    let mut replacement = Vec::<Operation>::new();
    for node_id in predecessor_ids.into_iter().chain(successor_ids.into_iter()) {
        let Some(template_node) = compiled_template.graph.node(node_id) else {
            return Err(CompileError::Internal(format!(
                "template node {} missing during substitution build",
                node_id
            )));
        };
        let Some(inverse_op) = build_inverse_template_op(
            template_node,
            match_info,
            &compiled_template.prepared,
            circuit_prepared,
        )?
        else {
            return Ok(None);
        };
        replacement.push(inverse_op);
    }

    Ok(Some(replacement))
}

/// Collects all transitive predecessors for a set of start nodes.
fn collect_all_predecessors(view: &CommutationView, starts: &HashSet<usize>) -> HashSet<usize> {
    let mut out = HashSet::<usize>::new();
    let mut queue = VecDeque::<usize>::new();
    for &node in starts {
        queue.push_back(node);
    }

    while let Some(node) = queue.pop_front() {
        let Some(preds) = view.predecessors.get(node) else {
            continue;
        };
        for &pred in preds {
            if starts.contains(&pred) {
                continue;
            }
            if out.insert(pred) {
                queue.push_back(pred);
            }
        }
    }

    out
}

/// Builds one inverse replacement operation mapped onto circuit qubits.
fn build_inverse_template_op(
    template_node: &GateNode,
    match_info: &TemplateMatch,
    template_prepared: &PreparedCircuit,
    circuit_prepared: &PreparedCircuit,
) -> Result<Option<Operation>, CompileError> {
    let Some(template_op) = template_prepared
        .operations
        .get(template_node.op_index)
        .map(|prepared| &prepared.op)
    else {
        return Err(CompileError::Internal(format!(
            "template operation {} missing during inverse construction",
            template_node.op_index
        )));
    };

    let mut params = SmallVec::<[crate::circuit::Parameter; 3]>::new();
    for param in &template_op.params {
        match param {
            CircuitParam::Fixed(v) => params.push(crate::circuit::Parameter::from(*v)),
            CircuitParam::Index(index) => {
                let idx = *index as usize;
                let Some(symbolic) = template_prepared.parameters.get(idx) else {
                    return Err(CompileError::Internal(format!(
                        "template operation references missing parameter index {}",
                        idx
                    )));
                };
                params.push(symbolic.clone());
            }
        }
    }

    let Some((inverse_instruction, inverse_params)) = template_op.instruction.inverse(&params)
    else {
        return Ok(None);
    };

    let mut mapped_qubits = SmallVec::<[Qubit; 3]>::new();
    for &template_logical in &template_node.logical_qubits {
        let Some(&circuit_logical) = match_info.qubit_mapping.get(template_logical) else {
            return Err(CompileError::Internal(format!(
                "missing qubit mapping for template logical {}",
                template_logical
            )));
        };
        let Some(&mapped) = circuit_prepared.logical_qubits.get(circuit_logical) else {
            return Err(CompileError::Internal(format!(
                "mapped circuit logical {} is out of range",
                circuit_logical
            )));
        };
        mapped_qubits.push(mapped);
    }

    let mut mapped_params = SmallVec::<[CircuitParam; 1]>::new();
    for param in inverse_params {
        if let Ok(v) = param.evaluate(&None) {
            mapped_params.push(CircuitParam::Fixed(v));
            continue;
        }

        if let Some(index) = find_parameter_index(circuit_prepared.parameters.as_slice(), &param) {
            mapped_params.push(CircuitParam::Index(index));
            continue;
        }

        return Ok(None);
    }

    Ok(Some(Operation {
        instruction: inverse_instruction,
        qubits: mapped_qubits,
        params: mapped_params,
        label: template_op.label.clone(),
    }))
}

/// Finds one parameter index in a parameter pool by exact expression equality.
fn find_parameter_index(
    parameter_pool: &[crate::circuit::Parameter],
    target: &crate::circuit::Parameter,
) -> Option<u32> {
    parameter_pool
        .iter()
        .position(|p| p == target)
        .and_then(|idx| u32::try_from(idx).ok())
}

/// Returns negative-log fidelity penalty for one operation.
fn estimate_fidelity_penalty(op: &Operation) -> f64 {
    let fidelity = estimate_op_fidelity(op).clamp(1e-12, 1.0);
    -fidelity.ln()
}

/// Returns a static fidelity estimate used for replacement ranking.
fn estimate_op_fidelity(op: &Operation) -> f64 {
    match &op.instruction {
        Instruction::Standard(g) => estimate_standard_gate_fidelity(*g),
        Instruction::McGate(g) => {
            if g.num_qubits() <= 2 {
                0.99
            } else {
                0.95
            }
        }
        Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => 0.95,
        Instruction::Directive(_) | Instruction::ControlFlowGate(_) | Instruction::Delay => 1e-6,
    }
}

/// Returns default fidelity estimates for standard gates.
fn estimate_standard_gate_fidelity(gate: StandardGate) -> f64 {
    match gate {
        StandardGate::I | StandardGate::GPhase => 1.0,
        StandardGate::H
        | StandardGate::RX
        | StandardGate::RY
        | StandardGate::RZ
        | StandardGate::S
        | StandardGate::SDG
        | StandardGate::T
        | StandardGate::TDG
        | StandardGate::U
        | StandardGate::X
        | StandardGate::Y
        | StandardGate::Z
        | StandardGate::Phase
        | StandardGate::X2P
        | StandardGate::X2M
        | StandardGate::Y2P
        | StandardGate::Y2M => 0.9995,
        StandardGate::CX | StandardGate::CY | StandardGate::CZ | StandardGate::XY => 0.99,
        StandardGate::SWAP => 0.97,
        StandardGate::RXX
        | StandardGate::RXY
        | StandardGate::RYY
        | StandardGate::RZX
        | StandardGate::RZZ
        | StandardGate::CRX
        | StandardGate::CRY
        | StandardGate::CRZ
        | StandardGate::XY2P
        | StandardGate::XY2M
        | StandardGate::FSIM => 0.985,
        StandardGate::CCX => 0.94,
    }
}

/// Returns a static operation cost estimate used by optimization.
fn estimate_op_cost(op: &Operation) -> i64 {
    match &op.instruction {
        Instruction::Standard(g) => estimate_standard_gate_cost(*g),
        Instruction::McGate(g) => {
            if g.num_qubits() <= 2 {
                2
            } else {
                10
            }
        }
        Instruction::UnitaryGate(_) => 10,
        Instruction::CircuitGate(_) => 10,
        Instruction::Directive(_) | Instruction::ControlFlowGate(_) | Instruction::Delay => 1000,
    }
}

fn estimate_circuit_cost(circuit: &Circuit) -> i64 {
    circuit.operations().iter().map(estimate_op_cost).sum()
}

fn estimate_circuit_penalty(circuit: &Circuit) -> f64 {
    circuit.operations().iter().map(estimate_fidelity_penalty).sum()
}

/// Returns static costs for standard gates.
fn estimate_standard_gate_cost(gate: StandardGate) -> i64 {
    match gate {
        StandardGate::I => 0,
        StandardGate::H
        | StandardGate::RX
        | StandardGate::RY
        | StandardGate::RZ
        | StandardGate::S
        | StandardGate::SDG
        | StandardGate::T
        | StandardGate::TDG
        | StandardGate::U
        | StandardGate::X
        | StandardGate::Y
        | StandardGate::Z
        | StandardGate::Phase
        | StandardGate::X2P
        | StandardGate::X2M
        | StandardGate::Y2P
        | StandardGate::Y2M => 1,
        StandardGate::CX | StandardGate::XY => 2,
        StandardGate::CY | StandardGate::CZ => 4,
        StandardGate::SWAP => 6,
        StandardGate::RXX
        | StandardGate::RXY
        | StandardGate::RYY
        | StandardGate::RZX
        | StandardGate::RZZ
        | StandardGate::CRX
        | StandardGate::CRY
        | StandardGate::CRZ
        | StandardGate::XY2P
        | StandardGate::XY2M
        | StandardGate::FSIM => 5,
        StandardGate::CCX | StandardGate::GPhase => 21,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Qubit};

    /// Builds a simple H-CX-H circuit for matching tests.
    fn simple_circuit() -> Circuit {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.h(Qubit::new(1)).unwrap();
        circuit
    }

    /// Builds an H-CX template.
    fn simple_template() -> Circuit {
        let mut template = Circuit::new(2);
        template.h(Qubit::new(0)).unwrap();
        template.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        template
    }

    /// Verifies basic template matching works.
    #[test]
    fn test_template_matching_basic() {
        let circuit = simple_circuit();
        let template = simple_template();
        let matches = TemplateMatching::run(&circuit, &template, None, None).unwrap();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].match_pairs.len(), 2);
    }

    /// Verifies parameter exactness is enforced.
    #[test]
    fn test_template_matching_parameter_exactness() {
        let mut circuit = Circuit::new(1);
        circuit
            .append(
                Instruction::Standard(StandardGate::RX),
                [Qubit::new(0)],
                [ParameterValue::Fixed(0.1)],
                None,
            )
            .unwrap();

        let mut template = Circuit::new(1);
        template
            .append(
                Instruction::Standard(StandardGate::RX),
                [Qubit::new(0)],
                [ParameterValue::Fixed(0.2)],
                None,
            )
            .unwrap();

        let matches = TemplateMatching::run(&circuit, &template, None, None).unwrap();
        assert!(matches.is_empty());
    }

    /// Verifies cancellation optimization removes matched gates.
    #[test]
    fn test_template_optimization_cancellation() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let mut template = Circuit::new(1);
        template.h(Qubit::new(0)).unwrap();
        template.h(Qubit::new(0)).unwrap();

        let optimizer = TemplateOptimization::new(vec![template], Some(1), Some((3, 1)));
        let optimized = optimizer.execute(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
    }

    /// Verifies replacement can be selected on cost ties when fidelity improves.
    #[test]
    fn test_template_optimization_replacement_prefers_fidelity_on_tie() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.h(Qubit::new(1)).unwrap();

        // Identity template:
        // H(q1) CX(q0,q1) H(q1) CZ(q0,q1) = I
        let mut template = Circuit::new(2);
        template.h(Qubit::new(1)).unwrap();
        template.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        template.h(Qubit::new(1)).unwrap();
        template.cz(Qubit::new(0), Qubit::new(1)).unwrap();

        let optimizer = TemplateOptimization::new(vec![template], None, None);
        let optimized = optimizer.execute(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::CZ)
        ));
    }

    /// Verifies replacement is skipped on cost ties when fidelity gets worse.
    #[test]
    fn test_template_optimization_skips_lower_fidelity_tie() {
        let mut circuit = Circuit::new(2);
        circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();

        // Same identity template as above.
        let mut template = Circuit::new(2);
        template.h(Qubit::new(1)).unwrap();
        template.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        template.h(Qubit::new(1)).unwrap();
        template.cz(Qubit::new(0), Qubit::new(1)).unwrap();

        let optimizer = TemplateOptimization::new(vec![template], None, None);
        let optimized = optimizer.execute(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::CZ)
        ));
    }

    /// Verifies iterative optimization converges.
    #[test]
    fn test_template_optimization_iterative() {
        let mut circuit = Circuit::new(1);
        for _ in 0..3 {
            circuit.h(Qubit::new(0)).unwrap();
            circuit.h(Qubit::new(0)).unwrap();
        }

        let mut template = Circuit::new(1);
        template.h(Qubit::new(0)).unwrap();
        template.h(Qubit::new(0)).unwrap();

        let optimizer = TemplateOptimization::new(vec![template], None, None);
        let optimized = optimizer.execute_iterative(&circuit, Some(8)).unwrap();
        assert!(optimized.operations().is_empty());
    }

    /// Verifies default templates can be loaded.
    #[test]
    fn test_default_template_loading() {
        let optimizer = TemplateOptimization::with_default_templates(None, None).unwrap();
        assert!(optimizer.template_count() >= 1);
    }

    #[test]
    fn test_template_optimizer_reuses_library_across_runs() {
        let mut library = TemplateLibrary::new();
        let mut template = Circuit::new(1);
        template.h(Qubit::new(0)).unwrap();
        template.h(Qubit::new(0)).unwrap();
        library.register_rule(template).unwrap();

        let optimizer =
            TemplateOptimization::from_library(library, TemplateOptimizationConfig::default());

        let mut first = Circuit::new(1);
        first.h(Qubit::new(0)).unwrap();
        first.h(Qubit::new(0)).unwrap();

        let mut second = Circuit::new(2);
        second.h(Qubit::new(0)).unwrap();
        second.h(Qubit::new(0)).unwrap();
        second.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let first_optimized = optimizer.execute(&first).unwrap();
        let second_optimized = optimizer.execute(&second).unwrap();

        assert!(first_optimized.operations().is_empty());
        assert_eq!(second_optimized.operations().len(), 1);
    }
}
