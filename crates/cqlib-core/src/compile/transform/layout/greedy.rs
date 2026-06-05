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

//! Greedy initial layout.

use super::{
    CircuitLayoutAnalysis, Interaction, LayoutDiagnostics, LayoutObjective, LayoutResult,
    PhysicalLayoutGraph, analyze_circuit_for_layout, build_physical_layout_graph,
};
use crate::circuit::Circuit;
use crate::compile::CompilerError;
use crate::device::{Device, Layout};
use std::cmp::Ordering;
use std::collections::BTreeMap;

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
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::compile::transform::{LayoutObjective, greedy_layout};
/// use cqlib_core::device::{Device, LogicalQubit, PhysicalQubit};
///
/// let device = Device::line("line-3", 3).unwrap();
/// let objective = LayoutObjective::topology_only();
///
/// let mut circuit = Circuit::new(3);
/// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
/// circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
///
/// let result = greedy_layout(&circuit, &device, &objective).unwrap();
/// assert_eq!(
///     result.layout.get_physical(LogicalQubit::new(0)),
///     Some(PhysicalQubit::new(0)),
/// );
/// assert!(result.diagnostics.is_perfect);
/// ```
pub fn greedy_layout(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
) -> Result<LayoutResult, CompilerError> {
    let analysis = analyze_circuit_for_layout(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    greedy_layout_prepared(&analysis, &physical, objective)
}

/// Builds a deterministic greedy layout from an already-built physical graph.
///
/// This lower-level entry point is useful when a workflow has already prepared
/// circuit analysis and physical graph data for one or more layout algorithms.
pub fn greedy_layout_prepared(
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

    let logical_index = analysis
        .logical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(index, logical)| (logical, index))
        .collect::<BTreeMap<_, _>>();
    let mut activity = vec![None; analysis.logical_qubits.len()];
    for interaction in analysis.interactions.interactions() {
        for logical in [interaction.left, interaction.right] {
            let entry = &mut activity[logical_index[&logical]];
            *entry = Some(entry.unwrap_or(0.0) + interaction.weight);
        }
    }
    let pair_candidates = sorted_physical_pair_candidates(physical);
    let mut mapping = vec![None; analysis.logical_qubits.len()];
    let mut vacant = vec![true; physical.physical_qubits().len()];
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
        let left_logical = logical_index[&interaction.left];
        let right_logical = logical_index[&interaction.right];
        match (mapping[left_logical], mapping[right_logical]) {
            (Some(_), Some(_)) => {}
            (Some(left_physical), None) => {
                let (right_physical, evaluated) = choose_single_candidate(
                    interaction,
                    right_logical,
                    left_physical,
                    true,
                    &vacant,
                    physical,
                    objective,
                    &activity,
                )?;
                candidates_evaluated += evaluated;
                mapping[right_logical] = Some(right_physical);
                vacant[right_physical] = false;
            }
            (None, Some(right_physical)) => {
                let (left_physical, evaluated) = choose_single_candidate(
                    interaction,
                    left_logical,
                    right_physical,
                    false,
                    &vacant,
                    physical,
                    objective,
                    &activity,
                )?;
                candidates_evaluated += evaluated;
                mapping[left_logical] = Some(left_physical);
                vacant[left_physical] = false;
            }
            (None, None) => {
                let (left_physical, right_physical, evaluated) = choose_pair_candidate(
                    interaction,
                    left_logical,
                    right_logical,
                    &vacant,
                    physical,
                    objective,
                    &activity,
                    &pair_candidates,
                )?;
                candidates_evaluated += evaluated;
                mapping[left_logical] = Some(left_physical);
                mapping[right_logical] = Some(right_physical);
                vacant[left_physical] = false;
                vacant[right_physical] = false;
            }
        }
    }

    for logical_index in 0..analysis.logical_qubits.len() {
        if mapping[logical_index].is_some() {
            continue;
        }
        let (physical_qubit, evaluated) = choose_idle_candidate(
            logical_index,
            &mapping,
            &vacant,
            physical,
            objective,
            &activity,
        )?;
        candidates_evaluated += evaluated;
        mapping[logical_index] = Some(physical_qubit);
        vacant[physical_qubit] = false;
    }

    let mut layout_mapping = BTreeMap::new();
    for (logical_index, physical_index) in mapping.iter().copied().enumerate() {
        let physical_index = physical_index.ok_or_else(|| {
            CompilerError::InvariantViolation(
                "greedy layout left a logical qubit unmapped".to_string(),
            )
        })?;
        let physical_qubit = physical.physical_at(physical_index).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "greedy layout selected physical index {physical_index} outside target topology"
            ))
        })?;
        layout_mapping.insert(analysis.logical_qubits[logical_index], physical_qubit);
    }
    let layout = Layout::new(
        analysis.logical_qubits.clone(),
        physical.physical_qubits().to_vec(),
        Some(layout_mapping),
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
    left_logical: usize,
    right_logical: usize,
    vacant: &[bool],
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &[Option<f64>],
    pair_candidates: &[(usize, usize)],
) -> Result<(usize, usize, usize), CompilerError> {
    let mut best: Option<(usize, usize, CandidateCost)> = None;
    let mut evaluated = 0usize;

    for (left_physical, right_physical) in pair_candidates.iter().copied() {
        if !vacant[left_physical] || !vacant[right_physical] {
            continue;
        }
        evaluated += 1;
        update_best_pair(
            interaction,
            left_logical,
            right_logical,
            left_physical,
            right_physical,
            physical,
            objective,
            activity,
            &mut best,
        );
    }

    if best.is_none() {
        for left_physical in 0..vacant.len() {
            if !vacant[left_physical] {
                continue;
            }
            for right_physical in 0..vacant.len() {
                if left_physical == right_physical || !vacant[right_physical] {
                    continue;
                }
                evaluated += 1;
                update_best_pair(
                    interaction,
                    left_logical,
                    right_logical,
                    left_physical,
                    right_physical,
                    physical,
                    objective,
                    activity,
                    &mut best,
                );
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
    candidate_logical: usize,
    anchored_physical: usize,
    anchor_is_left: bool,
    vacant: &[bool],
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &[Option<f64>],
) -> Result<(usize, usize), CompilerError> {
    let mut best: Option<(usize, CandidateCost)> = None;
    let mut evaluated = 0usize;

    for candidate_physical in 0..vacant.len() {
        if !vacant[candidate_physical] {
            continue;
        }
        evaluated += 1;
        let (left_physical, right_physical) = if anchor_is_left {
            (anchored_physical, candidate_physical)
        } else {
            (candidate_physical, anchored_physical)
        };
        let (left_logical, right_logical) = if anchor_is_left {
            (usize::MAX, candidate_logical)
        } else {
            (candidate_logical, usize::MAX)
        };
        let cost = CandidateCost::for_interaction(
            interaction,
            left_logical,
            right_logical,
            left_physical,
            right_physical,
            physical,
            objective,
            activity,
        );
        let candidate = (candidate_physical, cost);
        if best.as_ref().is_none_or(|best| {
            compare_cost(candidate.1, best.1)
                .then_with(|| {
                    physical.physical_qubits()[candidate.0].cmp(&physical.physical_qubits()[best.0])
                })
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
    logical: usize,
    mapping: &[Option<usize>],
    vacant: &[bool],
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &[Option<f64>],
) -> Result<(usize, usize), CompilerError> {
    let active_physical = mapping.iter().flatten().copied().collect::<Vec<_>>();
    let mut best: Option<(usize, usize, u64, f64)> = None;
    let mut evaluated = 0usize;

    for candidate_physical in 0..vacant.len() {
        if !vacant[candidate_physical] {
            continue;
        }
        evaluated += 1;
        let mut disconnected = 0usize;
        let mut distance_sum = 0u64;
        for mapped_physical in &active_physical {
            match physical.distance_by_index(candidate_physical, *mapped_physical) {
                Some(distance) => distance_sum += u64::from(distance),
                None => disconnected += 1,
            }
        }

        let readout = readout_cost(logical, candidate_physical, physical, objective, activity);
        let candidate = (candidate_physical, disconnected, distance_sum, readout);
        if best.as_ref().is_none_or(|best| {
            candidate
                .1
                .cmp(&best.1)
                .then_with(|| candidate.2.cmp(&best.2))
                .then_with(|| candidate.3.total_cmp(&best.3))
                .then_with(|| {
                    physical.physical_qubits()[candidate.0].cmp(&physical.physical_qubits()[best.0])
                })
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
        left_logical: usize,
        right_logical: usize,
        left_physical: usize,
        right_physical: usize,
        physical: &PhysicalLayoutGraph,
        objective: &LayoutObjective,
        activity: &[Option<f64>],
    ) -> Self {
        let Some(distance) = physical.distance_by_index(left_physical, right_physical) else {
            return Self {
                disconnected: true,
                distance: u32::MAX,
                objective_total: f64::INFINITY,
            };
        };

        let direction = if distance == 1 {
            let mut direction = 0.0;
            if interaction.directed_weight_left_to_right > 0.0
                && !physical.supports_directed_coupling_by_index(left_physical, right_physical)
            {
                direction += interaction.directed_weight_left_to_right;
            }
            if interaction.directed_weight_right_to_left > 0.0
                && !physical.supports_directed_coupling_by_index(right_physical, left_physical)
            {
                direction += interaction.directed_weight_right_to_left;
            }
            direction
        } else {
            0.0
        };
        let two_qubit_error = if distance == 1 && objective.two_qubit_error_weight != 0.0 {
            physical
                .two_qubit_error_undirected_by_index(left_physical, right_physical)
                .map(|error| interaction.weight * error)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let readout = readout_cost(left_logical, left_physical, physical, objective, activity)
            + readout_cost(right_logical, right_physical, physical, objective, activity);

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

fn readout_cost(
    logical: usize,
    physical_qubit: usize,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &[Option<f64>],
) -> f64 {
    if objective.readout_error_weight == 0.0 {
        return 0.0;
    }

    physical
        .readout_error_by_index(physical_qubit)
        .map(|error| activity.get(logical).copied().flatten().unwrap_or(1.0) * error)
        .unwrap_or(0.0)
}

fn compare_cost(a: CandidateCost, b: CandidateCost) -> Ordering {
    a.disconnected
        .cmp(&b.disconnected)
        .then_with(|| a.distance.cmp(&b.distance))
        .then_with(|| a.objective_total.total_cmp(&b.objective_total))
}

fn update_best_pair(
    interaction: &Interaction,
    left_logical: usize,
    right_logical: usize,
    left_physical: usize,
    right_physical: usize,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    activity: &[Option<f64>],
    best: &mut Option<(usize, usize, CandidateCost)>,
) {
    let cost = CandidateCost::for_interaction(
        interaction,
        left_logical,
        right_logical,
        left_physical,
        right_physical,
        physical,
        objective,
        activity,
    );
    let candidate = (left_physical, right_physical, cost);
    if best.as_ref().is_none_or(|best| {
        compare_cost(candidate.2, best.2)
            .then_with(|| {
                physical.physical_qubits()[candidate.0].cmp(&physical.physical_qubits()[best.0])
            })
            .then_with(|| {
                physical.physical_qubits()[candidate.1].cmp(&physical.physical_qubits()[best.1])
            })
            .is_lt()
    }) {
        *best = Some(candidate);
    }
}

fn sorted_physical_pair_candidates(physical: &PhysicalLayoutGraph) -> Vec<(usize, usize)> {
    let mut candidates = Vec::new();
    for left in 0..physical.physical_qubits().len() {
        for right in 0..physical.physical_qubits().len() {
            if left != right && physical.distance_by_index(left, right).is_some() {
                candidates.push((left, right));
            }
        }
    }
    candidates.sort_by(|a, b| {
        physical
            .distance_by_index(a.0, a.1)
            .cmp(&physical.distance_by_index(b.0, b.1))
            .then_with(|| physical.physical_qubits()[a.0].cmp(&physical.physical_qubits()[b.0]))
            .then_with(|| physical.physical_qubits()[a.1].cmp(&physical.physical_qubits()[b.1]))
    });
    candidates
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
