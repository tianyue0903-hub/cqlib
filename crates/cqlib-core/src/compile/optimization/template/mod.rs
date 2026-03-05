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

//! Template matching and optimization.
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

use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::{Circuit, Operation, Qubit};
use crate::compile::error::CompileError;
use crate::compile::graph::{CommutationView, GateGraph, GateNode};
use crate::compile::mapping::{append_operation, preprocess_circuit};
use crate::ir::qcis_loads;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;

/// Embedded default 1q/2q templates.
const DEFAULT_TEMPLATES_JSON: &str = include_str!("default_templates.json");

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

    /// Executes template matching with optional heuristic knobs.
    pub fn run(
        circuit: &Circuit,
        template: &Circuit,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Vec<TemplateMatch>, CompileError> {
        Self::run_internal(circuit, template, qubit_fixing_cnt, prune_param)
    }

    /// Internal matching driver shared by public APIs.
    fn run_internal(
        circuit: &Circuit,
        template: &Circuit,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Vec<TemplateMatch>, CompileError> {
        let _ = (qubit_fixing_cnt, prune_param);

        let c_prepared = preprocess_circuit(circuit)?;
        let t_prepared = preprocess_circuit(template)?;

        if t_prepared.operations.is_empty() {
            return Ok(Vec::new());
        }
        if t_prepared.logical_qubits.len() > c_prepared.logical_qubits.len() {
            return Ok(Vec::new());
        }

        let c_graph = GateGraph::from_prepared(&c_prepared)?;
        let t_graph = GateGraph::from_prepared(&t_prepared)?;
        let c_view = c_graph.commutation_view()?;
        let t_view = t_graph.commutation_view()?;

        let all_mappings =
            enumerate_qubit_mappings(t_prepared.logical_qubits.len(), c_prepared.logical_qubits.len());

        let mut out = HashSet::<TemplateMatch>::new();
        let template_node_ids: Vec<usize> = (0..t_graph.size()).collect();
        for mapping in all_mappings {
            let matches = match_under_mapping(
                &c_graph,
                &t_graph,
                &c_view,
                &t_view,
                &template_node_ids,
                &mapping,
            );
            out.extend(matches);
        }

        let mut collected: Vec<TemplateMatch> = out.into_iter().collect();
        collected.sort_by(|a, b| a.match_pairs.cmp(&b.match_pairs));
        Ok(collected)
    }
}

/// Configuration used by template optimization.
#[derive(Debug, Clone, Default)]
pub struct TemplateOptimizationConfig {
    /// Optional qubit-fixing heuristic depth for matching.
    pub qubit_fixing_cnt: Option<usize>,
    /// Optional `(depth, width)` prune parameters for matching.
    pub prune_param: Option<(usize, usize)>,
}

/// Template optimization entry type.
#[derive(Debug, Clone)]
pub struct TemplateOptimization {
    templates: Vec<Circuit>,
    config: TemplateOptimizationConfig,
}

impl TemplateOptimization {
    /// Creates an optimizer from explicit template circuits.
    pub fn new(
        templates: Vec<Circuit>,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Self {
        Self {
            templates,
            config: TemplateOptimizationConfig {
                qubit_fixing_cnt,
                prune_param,
            },
        }
    }

    /// Creates an optimizer from embedded default templates.
    pub fn with_default_templates(
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Self, CompileError> {
        let templates = load_templates_from_json_string(DEFAULT_TEMPLATES_JSON)?;
        Ok(Self::new(templates, qubit_fixing_cnt, prune_param))
    }

    /// Creates an optimizer from a `.json` or `.qcis` template file.
    pub fn from_template_file(
        template_file: &str,
        qubit_fixing_cnt: Option<usize>,
        prune_param: Option<(usize, usize)>,
    ) -> Result<Self, CompileError> {
        let lower = template_file.to_ascii_lowercase();
        let templates = if lower.ends_with(".json") {
            let content = fs::read_to_string(template_file).map_err(|err| {
                CompileError::Internal(format!(
                    "failed to read template file {}: {}",
                    template_file, err
                ))
            })?;
            load_templates_from_json_string(&content)?
        } else if lower.ends_with(".qcis") {
            load_qcis_templates_from_file(template_file)?
        } else {
            return Err(CompileError::Internal(format!(
                "unsupported template file extension for {}",
                template_file
            )));
        };

        Ok(Self::new(templates, qubit_fixing_cnt, prune_param))
    }

    /// Returns the number of templates held by this optimizer.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Returns immutable template list.
    pub fn templates(&self) -> &[Circuit] {
        &self.templates
    }

    /// Executes one-pass template optimization.
    pub fn execute(&self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let mut current = circuit.clone();

        for template in &self.templates {
            current = apply_template_substitutions(
                &current,
                template,
                self.config.qubit_fixing_cnt,
                self.config.prune_param,
            )?;
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

/// JSON template file wrapper.
#[derive(Debug, Clone, Deserialize)]
struct TemplateFile {
    /// Optional version tag.
    version: Option<u32>,
    /// Template definitions.
    templates: Vec<TemplateDefinition>,
}

/// One JSON template definition.
#[derive(Debug, Clone, Deserialize)]
struct TemplateDefinition {
    /// Optional human-readable template name.
    #[allow(dead_code)]
    name: Option<String>,
    /// Gate sequence.
    gates: Vec<GateDefinition>,
}

/// One gate in JSON template definition.
#[derive(Debug, Clone, Deserialize)]
struct GateDefinition {
    /// Gate mnemonic.
    gate: String,
    /// Gate qubit list.
    qubits: Vec<usize>,
    /// Optional gate parameters.
    params: Option<Vec<f64>>,
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
        if template_view.is_reachable(other_t, t_id) && !circuit_view.is_reachable(*other_c, c_id) {
            return false;
        }
        if template_view.is_reachable(t_id, other_t) && !circuit_view.is_reachable(c_id, *other_c) {
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
    if instruction_signature(&template_node.op.instruction) != instruction_signature(&circuit_node.op.instruction) {
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

/// Applies template substitutions selected by gate-cost and fidelity priority.
fn apply_template_substitutions(
    circuit: &Circuit,
    template: &Circuit,
    qubit_fixing_cnt: Option<usize>,
    prune_param: Option<(usize, usize)>,
) -> Result<Circuit, CompileError> {
    let _ = (qubit_fixing_cnt, prune_param);

    let prepared = preprocess_circuit(circuit)?;
    let template_prepared = preprocess_circuit(template)?;
    if template_prepared.operations.is_empty() {
        return Ok(circuit.clone());
    }
    if template_prepared.logical_qubits.len() > prepared.logical_qubits.len() {
        return Ok(circuit.clone());
    }

    let circuit_graph = GateGraph::from_prepared(&prepared)?;
    let template_graph = GateGraph::from_prepared(&template_prepared)?;
    let circuit_view = circuit_graph.commutation_view()?;
    let template_view = template_graph.commutation_view()?;
    let qubit_mappings =
        enumerate_qubit_mappings(template_prepared.logical_qubits.len(), prepared.logical_qubits.len());

    let template_subsets = enumerate_template_node_subsets(template_graph.size());
    let mut candidates = Vec::<SubstitutionCandidate>::new();
    for subset in template_subsets {
        let matches = collect_matches_for_nodes(
            &circuit_graph,
            &template_graph,
            &circuit_view,
            &template_view,
            &subset,
            &qubit_mappings,
        );
        for m in matches {
            let Some(candidate) = build_substitution_candidate(
                &prepared,
                &template_prepared,
                &template_graph,
                &template_view,
                &m,
            )?
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
    for cand in candidates {
        if cand.matched_nodes.iter().any(|idx| occupied.contains(idx)) {
            continue;
        }
        occupied.extend(cand.matched_nodes.iter().copied());
        selected.push(cand);
    }

    if selected.is_empty() {
        return Ok(circuit.clone());
    }

    let mut insertions: HashMap<usize, Vec<Operation>> = HashMap::new();
    let mut removed = HashSet::<usize>::new();
    for cand in &selected {
        let first = cand.matched_nodes[0];
        insertions.insert(first, cand.replacement.clone());
        removed.extend(cand.matched_nodes.iter().copied());
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

/// Enumerates template node subsets used for substitution matching.
fn enumerate_template_node_subsets(template_size: usize) -> Vec<Vec<usize>> {
    const MAX_EXHAUSTIVE_TEMPLATE_SIZE: usize = 10;

    if template_size == 0 {
        return Vec::new();
    }
    if template_size > MAX_EXHAUSTIVE_TEMPLATE_SIZE {
        return vec![(0..template_size).collect()];
    }

    let mut subsets = Vec::<Vec<usize>>::new();
    let total = 1usize << template_size;
    for mask in 1..total {
        let mut subset = Vec::new();
        for bit in 0..template_size {
            if (mask >> bit) & 1 == 1 {
                subset.push(bit);
            }
        }
        subsets.push(subset);
    }
    subsets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    subsets
}

/// Collects all matches for one selected subset of template nodes.
fn collect_matches_for_nodes(
    circuit_graph: &GateGraph,
    template_graph: &GateGraph,
    circuit_view: &CommutationView,
    template_view: &CommutationView,
    template_node_ids: &[usize],
    qubit_mappings: &[Vec<usize>],
) -> Vec<TemplateMatch> {
    let mut out = HashSet::<TemplateMatch>::new();
    for mapping in qubit_mappings {
        out.extend(match_under_mapping(
            circuit_graph,
            template_graph,
            circuit_view,
            template_view,
            template_node_ids,
            mapping,
        ));
    }
    let mut collected: Vec<TemplateMatch> = out.into_iter().collect();
    collected.sort_by(|a, b| a.match_pairs.cmp(&b.match_pairs));
    collected
}

/// Builds one substitution candidate and computes its ranking metrics.
fn build_substitution_candidate(
    prepared: &crate::compile::mapping::PreparedCircuit,
    template_prepared: &crate::compile::mapping::PreparedCircuit,
    template_graph: &GateGraph,
    template_view: &CommutationView,
    match_info: &TemplateMatch,
) -> Result<Option<SubstitutionCandidate>, CompileError> {
    let replacement_ops = match build_replacement_ops_for_match(
        match_info,
        template_graph,
        template_view,
        template_prepared,
        prepared,
    )? {
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
    template_graph: &GateGraph,
    template_view: &CommutationView,
    template_prepared: &crate::compile::mapping::PreparedCircuit,
    circuit_prepared: &crate::compile::mapping::PreparedCircuit,
) -> Result<Option<Vec<Operation>>, CompileError> {
    let matched_template_nodes: HashSet<usize> =
        match_info.match_pairs.iter().map(|(t, _)| *t).collect();
    if matched_template_nodes.is_empty() {
        return Ok(None);
    }

    let predecessors = collect_all_predecessors(template_view, &matched_template_nodes);
    let mut successors = HashSet::<usize>::new();
    for node_id in 0..template_graph.size() {
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
    for node_id in predecessor_ids
        .into_iter()
        .chain(successor_ids.into_iter())
    {
        let Some(template_node) = template_graph.node(node_id) else {
            return Err(CompileError::Internal(format!(
                "template node {} missing during substitution build",
                node_id
            )));
        };
        let Some(inverse_op) = build_inverse_template_op(
            template_node,
            match_info,
            template_prepared,
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
fn collect_all_predecessors(
    view: &CommutationView,
    starts: &HashSet<usize>,
) -> HashSet<usize> {
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
    template_prepared: &crate::compile::mapping::PreparedCircuit,
    circuit_prepared: &crate::compile::mapping::PreparedCircuit,
) -> Result<Option<Operation>, CompileError> {
    let Some(template_op) = template_prepared
        .operations
        .get(template_node.op_index)
        .map(|p| &p.op)
    else {
        return Err(CompileError::Internal(format!(
            "template operation {} missing during inverse construction",
            template_node.op_index
        )));
    };

    let mut params = smallvec::SmallVec::<[crate::circuit::Parameter; 3]>::new();
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

    let Some((inverse_instruction, inverse_params)) = template_op.instruction.inverse(&params) else {
        return Ok(None);
    };

    let mut mapped_qubits = smallvec::SmallVec::<[Qubit; 3]>::new();
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

    let mut mapped_params = smallvec::SmallVec::<[CircuitParam; 1]>::new();
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

/// Loads templates from one JSON content string.
fn load_templates_from_json_string(content: &str) -> Result<Vec<Circuit>, CompileError> {
    let file: TemplateFile = serde_json::from_str(content)
        .map_err(|err| CompileError::Internal(format!("invalid template JSON: {}", err)))?;
    if let Some(version) = file.version
        && version != 1
    {
        return Err(CompileError::Internal(format!(
            "unsupported template JSON version {}",
            version
        )));
    }

    let mut templates = Vec::with_capacity(file.templates.len());
    for (idx, t) in file.templates.iter().enumerate() {
        if t.gates.is_empty() {
            return Err(CompileError::Internal(format!(
                "template at index {} has no gates",
                idx
            )));
        }
        let circuit = build_template_circuit_from_defs(&t.gates)?;
        // Validate compile constraints up front.
        let _ = preprocess_circuit(&circuit)?;
        templates.push(circuit);
    }
    Ok(templates)
}

/// Loads templates from one QCIS template file.
fn load_qcis_templates_from_file(path: &str) -> Result<Vec<Circuit>, CompileError> {
    let content = fs::read_to_string(path).map_err(|err| {
        CompileError::Internal(format!("failed to read template file {}: {}", path, err))
    })?;

    let mut templates = Vec::new();
    for chunk in split_qcis_templates(&content) {
        let circuit =
            qcis_loads(&chunk).map_err(|err| CompileError::Internal(format!("invalid QCIS template: {}", err)))?;
        let _ = preprocess_circuit(&circuit)?;
        templates.push(circuit);
    }
    Ok(templates)
}

/// Splits a QCIS text into per-template chunks using `---` separators.
fn split_qcis_templates(content: &str) -> Vec<String> {
    let mut chunks = Vec::<String>::new();
    let mut cur = Vec::<String>::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with("---") {
            if !cur.is_empty() {
                chunks.push(cur.join("\n"));
                cur.clear();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        cur.push(line.to_string());
    }
    if !cur.is_empty() {
        chunks.push(cur.join("\n"));
    }
    chunks
}

/// Builds one template circuit from JSON gate definitions.
fn build_template_circuit_from_defs(gates: &[GateDefinition]) -> Result<Circuit, CompileError> {
    let mut max_q = None::<usize>;
    for gate in gates {
        for &q in &gate.qubits {
            max_q = Some(max_q.map_or(q, |m| m.max(q)));
        }
    }
    let width = max_q.map_or(0usize, |v| v + 1);
    let mut circuit = Circuit::new(width);

    for gate in gates {
        let std_gate = parse_standard_gate(&gate.gate)?;
        if std_gate.num_qubits() > 2 {
            return Err(CompileError::Internal(format!(
                "template gate {} exceeds phase-1 2q scope",
                gate.gate
            )));
        }
        if gate.qubits.len() != std_gate.num_qubits() {
            return Err(CompileError::Internal(format!(
                "gate {} expects {} qubits, got {}",
                gate.gate,
                std_gate.num_qubits(),
                gate.qubits.len()
            )));
        }

        let params = gate.params.clone().unwrap_or_default();
        if params.len() != std_gate.num_params() {
            return Err(CompileError::Internal(format!(
                "gate {} expects {} params, got {}",
                gate.gate,
                std_gate.num_params(),
                params.len()
            )));
        }

        let qubits: Vec<Qubit> = gate.qubits.iter().map(|&q| Qubit::new(q as u32)).collect();
        let pvals: Vec<ParameterValue> = params.into_iter().map(ParameterValue::Fixed).collect();
        circuit.append(Instruction::Standard(std_gate), qubits, pvals, None)?;
    }

    Ok(circuit)
}

/// Parses one standard gate name from JSON.
fn parse_standard_gate(name: &str) -> Result<StandardGate, CompileError> {
    let upper = name.trim().to_ascii_uppercase();
    let gate = match upper.as_str() {
        "I" => StandardGate::I,
        "H" => StandardGate::H,
        "RX" => StandardGate::RX,
        "RXX" => StandardGate::RXX,
        "RXY" => StandardGate::RXY,
        "RY" => StandardGate::RY,
        "RYY" => StandardGate::RYY,
        "RZ" => StandardGate::RZ,
        "RZX" => StandardGate::RZX,
        "RZZ" => StandardGate::RZZ,
        "S" => StandardGate::S,
        "SD" | "SDG" => StandardGate::SDG,
        "SWAP" => StandardGate::SWAP,
        "T" => StandardGate::T,
        "TD" | "TDG" => StandardGate::TDG,
        "U" => StandardGate::U,
        "X" => StandardGate::X,
        "XY" => StandardGate::XY,
        "X2P" => StandardGate::X2P,
        "X2M" => StandardGate::X2M,
        "XY2P" => StandardGate::XY2P,
        "XY2M" => StandardGate::XY2M,
        "Y" => StandardGate::Y,
        "Y2P" => StandardGate::Y2P,
        "Y2M" => StandardGate::Y2M,
        "Z" => StandardGate::Z,
        "P" | "PHASE" => StandardGate::Phase,
        "CX" => StandardGate::CX,
        "CCX" => StandardGate::CCX,
        "CY" => StandardGate::CY,
        "CZ" => StandardGate::CZ,
        "CRX" => StandardGate::CRX,
        "CRY" => StandardGate::CRY,
        "CRZ" => StandardGate::CRZ,
        "FSIM" => StandardGate::FSIM,
        _ => {
            return Err(CompileError::Internal(format!(
                "unsupported template gate {}",
                name
            )))
        }
    };
    Ok(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::param::ParameterValue;
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
}
