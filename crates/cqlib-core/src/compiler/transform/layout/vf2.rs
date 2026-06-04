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

//! VF2++ perfect initial layout.

use super::vf2_engine::{Vf2Graph, Vf2SearchConfig, find_non_induced_mappings};
use super::{
    CircuitLayoutAnalysis, Interaction, LayoutDiagnostics, LayoutObjective, LayoutResult,
    LayoutScore, PhysicalLayoutGraph, analyze_circuit_for_layout, build_physical_layout_graph,
};
use crate::circuit::Circuit;
use crate::compiler::CompilerError;
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rustworkx_core::petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, BTreeSet};

/// Selects which logical interaction edges must be matched by VF2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vf2EdgeRequirement {
    /// Require only interactions with positive weight.
    PositiveInteractions,
    /// Require every interaction stored in the interaction graph.
    AllInteractions,
}

impl Vf2EdgeRequirement {
    fn requires(self, interaction: &Interaction) -> bool {
        match self {
            Self::PositiveInteractions => interaction.weight > 0.0,
            Self::AllInteractions => true,
        }
    }
}

/// Configuration for [`vf2_perfect_layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vf2LayoutConfig {
    /// Maximum number of complete perfect candidates to score.
    pub candidate_limit: usize,
    /// Maximum number of partial mapping extensions attempted by the search.
    pub call_limit: Option<usize>,
    /// Selects which logical interaction edges are hard topology constraints.
    pub edge_requirement: Vf2EdgeRequirement,
}

impl Default for Vf2LayoutConfig {
    fn default() -> Self {
        Self {
            candidate_limit: 10,
            call_limit: None,
            edge_requirement: Vf2EdgeRequirement::PositiveInteractions,
        }
    }
}

/// Searches for a perfect initial layout using non-induced VF2++ matching.
///
/// A perfect layout maps every required logical interaction onto adjacent
/// physical qubits. Physical extra edges are allowed, matching Qiskit's
/// non-induced `VF2Layout` semantics. Coupling direction is not a hard
/// feasibility constraint; direction mismatch is scored by [`LayoutObjective`].
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] if physical capacity is insufficient,
/// `candidate_limit` is zero, no perfect mapping is found, or scoring rejects
/// all complete candidates.
pub fn vf2_perfect_layout(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &Vf2LayoutConfig,
) -> Result<LayoutResult, CompilerError> {
    let analysis = analyze_circuit_for_layout(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    vf2_perfect_layout_prepared(&analysis, &physical, objective, config)
}

/// Searches for a perfect initial layout on an already-built physical graph.
///
/// This lower-level entry point is useful when a workflow has already prepared
/// circuit analysis and physical graph data for one or more layout algorithms.
pub fn vf2_perfect_layout_prepared(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    config: &Vf2LayoutConfig,
) -> Result<LayoutResult, CompilerError> {
    if config.candidate_limit == 0 {
        return Err(CompilerError::InvalidInput(
            "vf2 perfect layout candidate_limit must be greater than zero".to_string(),
        ));
    }
    if analysis.logical_qubits.len() > physical.physical_qubits().len() {
        return Err(CompilerError::InvalidInput(format!(
            "vf2 perfect layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            analysis.logical_qubits.len(),
            physical.physical_qubits().len()
        )));
    }

    let activity = analysis.interactions.logical_activity();
    let required_interactions = analysis
        .interactions
        .interactions()
        .iter()
        .filter(|interaction| config.edge_requirement.requires(interaction))
        .collect::<Vec<_>>();

    if required_interactions.is_empty() {
        let mapping = complete_mapping(BTreeMap::new(), analysis, physical, objective, &activity);
        let layout = layout_from_mapping(analysis, physical, mapping, "vf2 perfect layout")?;
        let score = objective.score_layout(analysis, physical, &layout)?;
        return Ok(LayoutResult {
            layout,
            diagnostics: LayoutDiagnostics {
                is_perfect: true,
                candidates_evaluated: 1,
                used_fidelity: score.used_fidelity,
                notes: Vec::new(),
            },
            score: Some(score),
        });
    }

    let mut logical_index = BTreeMap::new();
    for interaction in &required_interactions {
        if !logical_index.contains_key(&interaction.left) {
            logical_index.insert(interaction.left, logical_index.len());
        }
        if !logical_index.contains_key(&interaction.right) {
            logical_index.insert(interaction.right, logical_index.len());
        }
    }
    let mut logical_nodes = vec![None; logical_index.len()];
    for (logical, node_index) in &logical_index {
        logical_nodes[*node_index] = Some(*logical);
    }
    let logical_nodes = logical_nodes
        .into_iter()
        .map(|logical| logical.expect("logical node indices are contiguous"))
        .collect::<Vec<_>>();

    let mut logical_graph =
        Vf2Graph::with_capacity(logical_index.len(), required_interactions.len());
    for _ in 0..logical_index.len() {
        logical_graph.add_node(());
    }
    for interaction in &required_interactions {
        let left = NodeIndex::new(logical_index[&interaction.left]);
        let right = NodeIndex::new(logical_index[&interaction.right]);
        if left != right && logical_graph.find_edge(left, right).is_none() {
            logical_graph.add_edge(left, right, ());
        }
    }

    let mut physical_graph = Vf2Graph::with_capacity(physical.physical_qubits().len(), 0);
    for _ in physical.physical_qubits() {
        physical_graph.add_node(());
    }
    for left in 0..physical.physical_qubits().len() {
        for right in (left + 1)..physical.physical_qubits().len() {
            if physical.is_adjacent_undirected(
                physical.physical_qubits()[left],
                physical.physical_qubits()[right],
            ) {
                physical_graph.add_edge(NodeIndex::new(left), NodeIndex::new(right), ());
            }
        }
    }

    let (mappings, search_stats) = find_non_induced_mappings(
        &logical_graph,
        &physical_graph,
        Vf2SearchConfig {
            candidate_limit: config.candidate_limit,
            call_limit: config.call_limit,
        },
    );

    let mut best: Option<(Layout, LayoutScore)> = None;
    let mut scored = 0usize;
    for mapping in mappings {
        let mut active_mapping = BTreeMap::new();
        for (logical_index, physical_index) in mapping.into_iter().enumerate() {
            active_mapping.insert(
                logical_nodes[logical_index],
                physical.physical_qubits()[physical_index],
            );
        }
        let mapping = complete_mapping(active_mapping, analysis, physical, objective, &activity);
        let layout = layout_from_mapping(analysis, physical, mapping, "vf2 perfect layout")?;
        let score = objective.score_layout(analysis, physical, &layout)?;
        scored += 1;

        if best.as_ref().is_none_or(|(best_layout, best_score)| {
            let layout_signature = analysis
                .logical_qubits
                .iter()
                .map(|logical| {
                    layout
                        .get_physical(*logical)
                        .expect("layout results map every logical qubit")
                        .id()
                })
                .collect::<Vec<_>>();
            let best_signature = analysis
                .logical_qubits
                .iter()
                .map(|logical| {
                    best_layout
                        .get_physical(*logical)
                        .expect("layout results map every logical qubit")
                        .id()
                })
                .collect::<Vec<_>>();

            score
                .total
                .total_cmp(&best_score.total)
                .then_with(|| layout_signature.cmp(&best_signature))
                .is_lt()
        }) {
            best = Some((layout, score));
        }
    }

    let Some((layout, score)) = best else {
        let reason = if search_stats.stopped_by_call_limit {
            " before the call limit was reached"
        } else {
            ""
        };
        return Err(CompilerError::InvalidInput(format!(
            "vf2 perfect layout could not find a perfect mapping{reason}"
        )));
    };

    let mut notes = Vec::new();
    if search_stats.stopped_by_call_limit {
        notes.push("vf2 search stopped after reaching call_limit".to_string());
    }

    Ok(LayoutResult {
        layout,
        diagnostics: LayoutDiagnostics {
            is_perfect: true,
            candidates_evaluated: scored,
            used_fidelity: score.used_fidelity,
            notes,
        },
        score: Some(score),
    })
}

fn complete_mapping(
    mut mapping: BTreeMap<LogicalQubit, PhysicalQubit>,
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &BTreeMap<LogicalQubit, f64>,
) -> BTreeMap<LogicalQubit, PhysicalQubit> {
    let occupied = mapping.values().copied().collect::<BTreeSet<_>>();
    let mut free_physical = physical
        .physical_qubits()
        .iter()
        .copied()
        .filter(|qubit| !occupied.contains(qubit))
        .collect::<Vec<_>>();
    free_physical.sort_by(|left, right| {
        if objective.readout_error_weight != 0.0 {
            physical
                .readout_error(*left)
                .unwrap_or(f64::INFINITY)
                .total_cmp(&physical.readout_error(*right).unwrap_or(f64::INFINITY))
                .then_with(|| left.cmp(right))
        } else {
            left.cmp(right)
        }
    });

    let mut unmapped_logical = analysis
        .logical_qubits
        .iter()
        .copied()
        .filter(|logical| !mapping.contains_key(logical))
        .collect::<Vec<_>>();
    unmapped_logical.sort_by(|left, right| {
        activity
            .get(right)
            .copied()
            .unwrap_or(1.0)
            .total_cmp(&activity.get(left).copied().unwrap_or(1.0))
            .then_with(|| left.cmp(right))
    });

    for (logical, physical) in unmapped_logical.into_iter().zip(free_physical) {
        mapping.insert(logical, physical);
    }
    mapping
}

fn layout_from_mapping(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    mapping: BTreeMap<LogicalQubit, PhysicalQubit>,
    algorithm: &str,
) -> Result<Layout, CompilerError> {
    Layout::new(
        analysis.logical_qubits.clone(),
        physical.physical_qubits().to_vec(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvariantViolation(format!(
            "{algorithm} failed to construct a valid layout: {error}"
        ))
    })
}
