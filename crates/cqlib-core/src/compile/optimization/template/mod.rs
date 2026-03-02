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
//! - phase-1 applies full-template cancellations only
//! - substitutions are applied only when cost strictly decreases

use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::ParameterValue;
use crate::circuit::{Circuit, Operation, Qubit};
use crate::compile::error::CompileError;
use crate::compile::graph::{CommutationView, GateGraph, GateNode};
use crate::compile::mapping::{append_operation, preprocess_circuit};
use crate::ir::qcis_loads;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
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
        for mapping in all_mappings {
            let matches = match_under_mapping(&c_graph, &t_graph, &c_view, &t_view, &mapping);
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
            let matches = TemplateMatching::run(
                &current,
                template,
                self.config.qubit_fixing_cnt,
                self.config.prune_param,
            )?;
            if matches.is_empty() {
                continue;
            }

            let applied = apply_cancellation_matches(&current, template, &matches)?;
            current = applied;
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
    /// Positive cost gain.
    gain: i64,
}

/// Runs matching under one fixed qubit mapping.
fn match_under_mapping(
    circuit_graph: &GateGraph,
    template_graph: &GateGraph,
    circuit_view: &CommutationView,
    template_view: &CommutationView,
    qubit_mapping: &[usize],
) -> Vec<TemplateMatch> {
    let t_size = template_graph.size();
    let c_size = circuit_graph.size();

    let mut candidates = vec![Vec::<usize>::new(); t_size];
    for (t_id, t_cands) in candidates.iter_mut().enumerate().take(t_size) {
        let Some(t_node) = template_graph.node(t_id) else {
            return Vec::new();
        };
        debug_assert_eq!(t_node.node_id, t_id);
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
    }

    let mut order: Vec<usize> = (0..t_size).collect();
    order.sort_by(|&a, &b| {
        let by_len = candidates[a].len().cmp(&candidates[b].len());
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
    candidates: &[Vec<usize>],
    template_view: &CommutationView,
    circuit_view: &CommutationView,
    assigned_t: &mut [Option<usize>],
    used_c: &mut [bool],
    qubit_mapping: &[usize],
    out: &mut Vec<TemplateMatch>,
) {
    if depth == order.len() {
        let mut pairs = Vec::with_capacity(order.len());
        for (t_id, c_opt) in assigned_t.iter().enumerate() {
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
    for &c_id in &candidates[t_id] {
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

/// Applies cancellation-only substitutions selected from matches.
fn apply_cancellation_matches(
    circuit: &Circuit,
    template: &Circuit,
    matches: &[TemplateMatch],
) -> Result<Circuit, CompileError> {
    let prepared = preprocess_circuit(circuit)?;
    let template_prepared = preprocess_circuit(template)?;
    let template_gate_cnt = template_prepared.operations.len();

    let mut candidates = Vec::<SubstitutionCandidate>::new();
    for m in matches {
        let replacement = build_replacement_ops_for_match(m, template_gate_cnt)?;
        let Some(replacement_ops) = replacement else {
            continue;
        };

        let mut matched_nodes: Vec<usize> = m.match_pairs.iter().map(|(_, c)| *c).collect();
        matched_nodes.sort_unstable();
        matched_nodes.dedup();
        if matched_nodes.is_empty() {
            continue;
        }

        let old_cost: i64 = matched_nodes
            .iter()
            .map(|&idx| estimate_op_cost(&prepared.operations[idx].op))
            .sum();
        let new_cost: i64 = replacement_ops.iter().map(estimate_op_cost).sum();
        let gain = old_cost - new_cost;
        if gain <= 0 {
            continue;
        }

        candidates.push(SubstitutionCandidate {
            matched_nodes,
            replacement: replacement_ops,
            gain,
        });
    }

    if candidates.is_empty() {
        return Ok(circuit.clone());
    }

    candidates.sort_by(|a, b| {
        b.gain
            .cmp(&a.gain)
            .then_with(|| a.matched_nodes[0].cmp(&b.matched_nodes[0]))
    });

    let mut selected = Vec::new();
    let mut occupied = HashSet::<usize>::new();
    for cand in candidates {
        if cand.matched_nodes.iter().any(|idx| occupied.contains(idx)) {
            continue;
        }
        for idx in &cand.matched_nodes {
            occupied.insert(*idx);
        }
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
        for idx in &cand.matched_nodes {
            removed.insert(*idx);
        }
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

/// Builds replacement operations for one match.
///
/// Phase-1 supports only full-template cancellation substitutions, therefore
/// non-full matches are skipped safely.
fn build_replacement_ops_for_match(
    match_info: &TemplateMatch,
    template_gate_cnt: usize,
) -> Result<Option<Vec<Operation>>, CompileError> {
    if match_info.match_pairs.len() != template_gate_cnt {
        return Ok(None);
    }
    Ok(Some(Vec::new()))
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
        StandardGate::CX | StandardGate::SWAP | StandardGate::XY | StandardGate::CZ => 2,
        StandardGate::CY => 4,
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
