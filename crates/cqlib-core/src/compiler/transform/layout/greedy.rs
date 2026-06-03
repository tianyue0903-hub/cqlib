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

//! Greedy initial layout.

use super::{
    CircuitLayoutAnalysis, Interaction, LayoutDiagnostics, LayoutObjective, LayoutResult,
    PhysicalLayoutGraph, build_physical_layout_graph,
};
use crate::compiler::CompilerError;
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Builds a deterministic greedy initial layout from weighted interactions.
///
/// The algorithm places higher-weight logical interactions first. For each
/// interaction it chooses the closest currently feasible physical placement,
/// using the shared [`LayoutObjective`] as a local tie-breaker for direction
/// and calibration data. Routing remains a later pass: this function only
/// selects an initial logical-to-physical mapping.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] if there are fewer usable physical
/// qubits than logical qubits, no physical candidate can be selected for a
/// required placement, or if final objective scoring rejects the layout.
pub fn greedy_layout(
    analysis: &CircuitLayoutAnalysis,
    device: &Device,
    objective: &LayoutObjective,
) -> Result<LayoutResult, CompilerError> {
    let physical = build_physical_layout_graph(device)?;
    greedy_layout_with_physical(analysis, &physical, objective)
}

/// Builds a deterministic greedy layout from an already-built physical graph.
///
/// This advanced entry point lets workflow code reuse distance and calibration
/// analysis across several layout algorithms without rebuilding the physical
/// graph.
pub fn greedy_layout_with_physical(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
) -> Result<LayoutResult, CompilerError> {
    if analysis.logical_qubits.len() > physical.physical_qubits().len() {
        return Err(CompilerError::InvalidInput(format!(
            "greedy layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            analysis.logical_qubits.len(),
            physical.physical_qubits().len()
        )));
    }

    let activity = analysis.interactions.logical_activity();
    let mut mapping = BTreeMap::new();
    let mut vacant = physical
        .physical_qubits()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut candidates_evaluated = 0usize;

    let mut interactions = analysis
        .interactions
        .interactions()
        .iter()
        .filter(|interaction| interaction.weight > 0.0)
        .collect::<Vec<_>>();
    interactions.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.first_seen_order.cmp(&b.first_seen_order))
            .then_with(|| a.left.cmp(&b.left))
            .then_with(|| a.right.cmp(&b.right))
    });

    for interaction in interactions {
        match (
            mapping.get(&interaction.left).copied(),
            mapping.get(&interaction.right).copied(),
        ) {
            (Some(_), Some(_)) => {}
            (Some(left_physical), None) => {
                let (right_physical, evaluated) = choose_single_candidate(
                    interaction,
                    left_physical,
                    true,
                    &vacant,
                    physical,
                    objective,
                    &activity,
                )?;
                candidates_evaluated += evaluated;
                mapping.insert(interaction.right, right_physical);
                vacant.remove(&right_physical);
            }
            (None, Some(right_physical)) => {
                let (left_physical, evaluated) = choose_single_candidate(
                    interaction,
                    right_physical,
                    false,
                    &vacant,
                    physical,
                    objective,
                    &activity,
                )?;
                candidates_evaluated += evaluated;
                mapping.insert(interaction.left, left_physical);
                vacant.remove(&left_physical);
            }
            (None, None) => {
                let (left_physical, right_physical, evaluated) =
                    choose_pair_candidate(interaction, &vacant, physical, objective, &activity)?;
                candidates_evaluated += evaluated;
                mapping.insert(interaction.left, left_physical);
                mapping.insert(interaction.right, right_physical);
                vacant.remove(&left_physical);
                vacant.remove(&right_physical);
            }
        }
    }

    for logical in &analysis.logical_qubits {
        if mapping.contains_key(logical) {
            continue;
        }
        let (physical_qubit, evaluated) =
            choose_idle_candidate(*logical, &mapping, &vacant, physical, objective, &activity)?;
        candidates_evaluated += evaluated;
        mapping.insert(*logical, physical_qubit);
        vacant.remove(&physical_qubit);
    }

    let layout = Layout::new(
        analysis.logical_qubits.clone(),
        physical.physical_qubits().to_vec(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvariantViolation(format!(
            "greedy layout failed to construct a valid layout: {error}"
        ))
    })?;

    let score = objective.score_layout(analysis, physical, &layout)?;
    let diagnostics = LayoutDiagnostics {
        is_perfect: is_perfect_layout(analysis, physical, &layout),
        candidates_evaluated,
        used_fidelity: score.used_fidelity,
        notes: Vec::new(),
    };

    Ok(LayoutResult {
        layout,
        score: Some(score),
        diagnostics,
    })
}

fn choose_pair_candidate(
    interaction: &Interaction,
    vacant: &BTreeSet<PhysicalQubit>,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &BTreeMap<LogicalQubit, f64>,
) -> Result<(PhysicalQubit, PhysicalQubit, usize), CompilerError> {
    let mut best: Option<(PhysicalQubit, PhysicalQubit, CandidateCost)> = None;
    let mut evaluated = 0usize;

    for left_physical in vacant {
        for right_physical in vacant {
            if left_physical == right_physical {
                continue;
            }
            evaluated += 1;
            let cost = CandidateCost::for_interaction(
                interaction,
                *left_physical,
                *right_physical,
                physical,
                objective,
                activity,
            );
            let candidate = (*left_physical, *right_physical, cost);
            if best.as_ref().is_none_or(|best| {
                compare_cost(candidate.2, best.2)
                    .then_with(|| candidate.0.cmp(&best.0))
                    .then_with(|| candidate.1.cmp(&best.1))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }
    }

    best.map(|(left, right, _)| (left, right, evaluated))
        .ok_or_else(|| {
            CompilerError::InvalidInput(
                "greedy layout could not find two vacant physical qubits for an interaction"
                    .to_string(),
            )
        })
}

fn choose_single_candidate(
    interaction: &Interaction,
    anchored_physical: PhysicalQubit,
    anchor_is_left: bool,
    vacant: &BTreeSet<PhysicalQubit>,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &BTreeMap<LogicalQubit, f64>,
) -> Result<(PhysicalQubit, usize), CompilerError> {
    let mut best: Option<(PhysicalQubit, CandidateCost)> = None;
    let mut evaluated = 0usize;

    for candidate_physical in vacant {
        evaluated += 1;
        let (left_physical, right_physical) = if anchor_is_left {
            (anchored_physical, *candidate_physical)
        } else {
            (*candidate_physical, anchored_physical)
        };
        let cost = CandidateCost::for_interaction(
            interaction,
            left_physical,
            right_physical,
            physical,
            objective,
            activity,
        );
        let candidate = (*candidate_physical, cost);
        if best.as_ref().is_none_or(|best| {
            compare_cost(candidate.1, best.1)
                .then_with(|| candidate.0.cmp(&best.0))
                .is_lt()
        }) {
            best = Some(candidate);
        }
    }

    best.map(|(physical, _)| (physical, evaluated))
        .ok_or_else(|| {
            CompilerError::InvalidInput(
                "greedy layout could not find a vacant physical qubit for an interaction"
                    .to_string(),
            )
        })
}

fn choose_idle_candidate(
    logical: LogicalQubit,
    mapping: &BTreeMap<LogicalQubit, PhysicalQubit>,
    vacant: &BTreeSet<PhysicalQubit>,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &BTreeMap<LogicalQubit, f64>,
) -> Result<(PhysicalQubit, usize), CompilerError> {
    let active_physical = mapping.values().copied().collect::<Vec<_>>();
    let mut best: Option<(PhysicalQubit, usize, u64, f64)> = None;
    let mut evaluated = 0usize;

    for candidate_physical in vacant {
        evaluated += 1;
        let mut disconnected = 0usize;
        let mut distance_sum = 0u64;
        for mapped_physical in &active_physical {
            match physical.distance(*candidate_physical, *mapped_physical) {
                Some(distance) => distance_sum += u64::from(distance),
                None => disconnected += 1,
            }
        }

        let readout = readout_cost(logical, *candidate_physical, physical, objective, activity);
        let candidate = (*candidate_physical, disconnected, distance_sum, readout);
        if best.as_ref().is_none_or(|best| {
            candidate
                .1
                .cmp(&best.1)
                .then_with(|| candidate.2.cmp(&best.2))
                .then_with(|| candidate.3.total_cmp(&best.3))
                .then_with(|| candidate.0.cmp(&best.0))
                .is_lt()
        }) {
            best = Some(candidate);
        }
    }

    best.map(|(physical, _, _, _)| (physical, evaluated))
        .ok_or_else(|| {
            CompilerError::InvalidInput(
                "greedy layout could not find a vacant physical qubit for an idle logical qubit"
                    .to_string(),
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct CandidateCost {
    disconnected: bool,
    distance: u32,
    objective_total: f64,
}

impl CandidateCost {
    fn for_interaction(
        interaction: &Interaction,
        left_physical: PhysicalQubit,
        right_physical: PhysicalQubit,
        physical: &PhysicalLayoutGraph,
        objective: &LayoutObjective,
        activity: &BTreeMap<LogicalQubit, f64>,
    ) -> Self {
        let Some(distance) = physical.distance(left_physical, right_physical) else {
            return Self {
                disconnected: true,
                distance: u32::MAX,
                objective_total: f64::INFINITY,
            };
        };

        let direction = if distance == 1 {
            direction_cost(interaction, left_physical, right_physical, physical)
        } else {
            0.0
        };
        let two_qubit_error = if distance == 1 && objective.two_qubit_error_weight != 0.0 {
            physical
                .two_qubit_error_undirected(left_physical, right_physical)
                .map(|error| interaction.weight * error)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let readout = readout_cost(
            interaction.left,
            left_physical,
            physical,
            objective,
            activity,
        ) + readout_cost(
            interaction.right,
            right_physical,
            physical,
            objective,
            activity,
        );

        Self {
            disconnected: false,
            distance,
            objective_total: objective.distance_weight * interaction.weight * f64::from(distance)
                + objective.direction_weight * direction
                + objective.two_qubit_error_weight * two_qubit_error
                + objective.readout_error_weight * readout,
        }
    }
}

fn direction_cost(
    interaction: &Interaction,
    left_physical: PhysicalQubit,
    right_physical: PhysicalQubit,
    physical: &PhysicalLayoutGraph,
) -> f64 {
    let mut direction = 0.0;
    if interaction.directed_weight_left_to_right > 0.0
        && !physical.supports_directed_coupling(left_physical, right_physical)
    {
        direction += interaction.directed_weight_left_to_right;
    }
    if interaction.directed_weight_right_to_left > 0.0
        && !physical.supports_directed_coupling(right_physical, left_physical)
    {
        direction += interaction.directed_weight_right_to_left;
    }
    direction
}

fn readout_cost(
    logical: LogicalQubit,
    physical_qubit: PhysicalQubit,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &BTreeMap<LogicalQubit, f64>,
) -> f64 {
    if objective.readout_error_weight == 0.0 {
        return 0.0;
    }

    physical
        .readout_error(physical_qubit)
        .map(|error| activity.get(&logical).copied().unwrap_or(1.0) * error)
        .unwrap_or(0.0)
}

fn compare_cost(a: CandidateCost, b: CandidateCost) -> Ordering {
    a.disconnected
        .cmp(&b.disconnected)
        .then_with(|| a.distance.cmp(&b.distance))
        .then_with(|| a.objective_total.total_cmp(&b.objective_total))
}

fn is_perfect_layout(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    layout: &Layout,
) -> bool {
    analysis
        .interactions
        .interactions()
        .iter()
        .filter(|interaction| interaction.weight > 0.0)
        .all(|interaction| {
            let Some(left) = layout.get_physical(interaction.left) else {
                return false;
            };
            let Some(right) = layout.get_physical(interaction.right) else {
                return false;
            };
            physical.is_adjacent_undirected(left, right)
        })
}
