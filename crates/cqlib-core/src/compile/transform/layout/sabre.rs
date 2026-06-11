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

//! SABRE initial-layout adapter.
//!
//! This module owns layout-candidate generation, objective scoring, and
//! [`LayoutResult`] construction. The SABRE core remains in
//! [`crate::compile::sabre`] and is used here only through crate-internal
//! routing primitives.
//!
//! The dependency direction is intentional: layout may call SABRE to refine and
//! score candidate initial layouts, but the standalone SABRE routing module must
//! not depend on layout algorithms or layout result types.

use super::{
    CircuitLayoutAnalysis, LayoutDiagnostics, LayoutObjective, LayoutResult, LayoutScore,
    PhysicalLayoutGraph, Vf2LayoutConfig, analyze_circuit_for_layout, build_physical_layout_graph,
    greedy_layout_prepared, is_perfect_layout, vf2_perfect_layout_prepared,
};
use crate::circuit::Circuit;
use crate::compile::CompilerError;
use crate::compile::sabre::{
    RoutingTarget, SabreConfig, SabreDag, TrialQuality, compare_trial_quality,
    normalize_initial_layout_for_target, route_trial, route_trial_unchecked, trial_seeds,
    validate_reachable_interactions_for_target,
};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Selects an initial layout with SABRE forward/backward refinement.
///
/// This function only returns the refined initial layout. It does not insert
/// SWAP operations or rebuild a physical circuit; callers that need routing
/// should run the SABRE routing core after selecting a layout.
///
/// Candidate layouts come from deterministic baselines, greedy/VF2 layout when
/// available, and random trials controlled by [`SabreConfig::layout_trials`].
/// Each candidate is refined through SABRE forward/backward passes and ranked
/// by final-route quality, then by [`LayoutObjective`] as a tie-breaker.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE layout
/// configuration, insufficient usable physical qubits, unreachable
/// interactions in the usable topology, or unsupported circuit operations.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::compile::sabre::SabreConfig;
/// use cqlib_core::compile::transform::{LayoutObjective, sabre_layout};
/// use cqlib_core::device::Device;
///
/// let mut circuit = Circuit::new(3);
/// circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
/// let device = Device::line("line-3", 3).unwrap();
///
/// let result = sabre_layout(
///     &circuit,
///     &device,
///     &LayoutObjective::topology_only(),
///     &SabreConfig::default(),
/// )
/// .unwrap();
/// assert!(result.score.is_some());
/// ```
pub fn sabre_layout(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    let analysis = analyze_circuit_for_layout(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    sabre_layout_prepared(circuit, &analysis, &physical, objective, config)
}

/// Selects a SABRE initial layout from already-prepared layout analysis.
///
/// The original circuit is still required because SABRE refinement uses the
/// operation dependency order to run trial routing. `analysis` and `physical`
/// are reused for candidate generation and objective scoring.
pub fn sabre_layout_prepared(
    circuit: &Circuit,
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    validate_layout_config(config)?;
    let target = RoutingTarget::from_physical(physical)?;
    let sabre = SabreDag::from_operations(circuit.operations())?;
    // Refinement only needs interaction order, not the full operation payload.
    let forwards = sabre.only_interactions();
    let backwards = forwards.reverse_interactions();
    let logical_qubits = analysis.logical_qubits.clone();

    if logical_qubits.len() > target.physical_qubits.len() {
        return Err(CompilerError::InvalidInput(format!(
            "sabre layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            logical_qubits.len(),
            target.physical_qubits.len()
        )));
    }

    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
    let candidates = initial_layout_candidates(
        analysis,
        &logical_qubits,
        physical,
        &target,
        objective,
        config.layout_trials,
        &mut rng,
    )?;
    let candidates_evaluated = candidates.len();
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, layout)| {
            let refinement_seeds = (0..config.refinement_iterations)
                .map(|_| (rng.random(), rng.random()))
                .collect();
            CandidateTrial {
                index,
                layout,
                refinement_seeds,
                scoring_seed: rng.random(),
            }
        })
        .collect::<Vec<_>>();

    let evaluations = trials
        .into_par_iter()
        .map(|trial| {
            let mut refined = trial.layout;
            for (forward_seed, backward_seed) in trial.refinement_seeds {
                // One refinement iteration routes forward, keeps the final
                // layout, then routes the reversed interaction DAG. This is
                // the SABRE layout-refinement loop, not final circuit routing.
                refined = match route_trial(
                    &forwards,
                    &target,
                    &refined,
                    &config.heuristic,
                    forward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(CompilerError::InvalidInput(message))
                        if message.contains("disconnected") =>
                    {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };

                refined = match route_trial(
                    &backwards,
                    &target,
                    &refined,
                    &config.heuristic,
                    backward_seed,
                ) {
                    Ok(result) => result.final_layout,
                    Err(CompilerError::InvalidInput(message))
                        if message.contains("disconnected") =>
                    {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            }

            // Rank refined layouts by how well they route the original DAG.
            // Multiple scoring trials reduce seed sensitivity without exposing
            // final SWAP insertion through this layout API.
            let route_quality =
                match best_route_quality(&sabre, &target, &refined, config, trial.scoring_seed) {
                    Ok(quality) => quality,
                    Err(CompilerError::InvalidInput(message))
                        if message.contains("disconnected") =>
                    {
                        return Ok(None);
                    }
                    Err(error) => return Err(error),
                };
            let score = objective.score_layout(analysis, physical, &refined)?;
            Ok(Some(CandidateEvaluation {
                index: trial.index,
                route_quality,
                layout: refined,
                score,
            }))
        })
        .collect::<Result<Vec<_>, CompilerError>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let best = evaluations
        .into_iter()
        .min_by(|left, right| {
            compare_trial_quality(
                config.trial_objective,
                left.route_quality,
                0,
                right.route_quality,
                0,
            )
            .then_with(|| left.score.total.total_cmp(&right.score.total))
            .then_with(|| left.index.cmp(&right.index))
        })
        .ok_or_else(|| {
            CompilerError::InvalidInput(
                "sabre layout found no candidate whose interactions are connected in the usable topology"
                    .to_string(),
            )
        })?;
    let swap_count = best.route_quality.swap_count;
    let layout = best.layout;
    let score = best.score;
    let is_perfect = is_perfect_layout(analysis, physical, &layout);

    Ok(LayoutResult {
        layout,
        score: Some(score.clone()),
        diagnostics: LayoutDiagnostics {
            is_perfect,
            candidates_evaluated,
            used_fidelity: score.used_fidelity,
            notes: vec![format!(
                "selected SABRE refined layout with {swap_count} final-route swaps"
            )],
        },
    })
}

struct CandidateTrial {
    /// Stable candidate order used as the final deterministic tie-breaker.
    index: usize,
    /// Candidate layout before forward/backward refinement.
    layout: Layout,
    /// Seed pairs for forward and backward refinement route trials.
    refinement_seeds: Vec<(u64, u64)>,
    /// Seed used to derive final-route scoring trials.
    scoring_seed: u64,
}

struct CandidateEvaluation {
    /// Original candidate index, retained after parallel evaluation.
    index: usize,
    /// Best final-route quality observed for the refined layout.
    route_quality: TrialQuality,
    /// Refined initial layout.
    layout: Layout,
    /// Objective score for the refined initial layout.
    score: LayoutScore,
}

/// Validates SABRE settings that are specific to layout selection.
///
/// The routing core owns generic SABRE validation. This adapter adds checks for
/// layout-only knobs that must be non-zero before candidate generation or
/// final-route scoring starts.
fn validate_layout_config(config: &SabreConfig) -> Result<(), CompilerError> {
    if config.layout_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_trials must be greater than zero".to_string(),
        ));
    }
    if config.layout_scoring_trials == 0 {
        return Err(CompilerError::InvalidInput(
            "sabre layout_scoring_trials must be greater than zero".to_string(),
        ));
    }
    crate::compile::sabre::validate_config(config)
}

/// Generates the candidate set refined by SABRE layout.
///
/// Candidates include deterministic anchors, opportunistic greedy/VF2 results,
/// and random physical orders. The result is deduplicated in logical-qubit
/// order so duplicate layouts from different sources are evaluated once.
fn initial_layout_candidates(
    analysis: &CircuitLayoutAnalysis,
    logical_qubits: &[LogicalQubit],
    physical: &PhysicalLayoutGraph,
    target: &RoutingTarget,
    objective: &LayoutObjective,
    layout_trials: usize,
    rng: &mut StdRng,
) -> Result<Vec<Layout>, CompilerError> {
    let mut candidates = Vec::new();

    // Always include deterministic extremes so SABRE has reproducible anchors
    // even when random trials are disabled or collapse to duplicates.
    let trivial = layout_from_physical_order(logical_qubits, target, &target.physical_qubits)?;
    candidates.push(trivial);

    let mut reverse_physical = target.physical_qubits.clone();
    reverse_physical.reverse();
    candidates.push(layout_from_physical_order(
        logical_qubits,
        target,
        &reverse_physical,
    )?);

    if let Ok(greedy) = greedy_layout_prepared(analysis, physical, objective) {
        candidates.push(normalize_initial_layout_for_target(
            logical_qubits,
            target,
            &greedy.layout,
        )?);
    }
    if let Ok(vf2) =
        vf2_perfect_layout_prepared(analysis, physical, objective, &Vf2LayoutConfig::default())
    {
        candidates.push(normalize_initial_layout_for_target(
            logical_qubits,
            target,
            &vf2.layout,
        )?);
    }

    for _ in 0..layout_trials {
        let mut physical_order = target.physical_qubits.clone();
        physical_order.shuffle(rng);
        candidates.push(layout_from_physical_order(
            logical_qubits,
            target,
            &physical_order,
        )?);
    }

    deduplicate_layouts(candidates, logical_qubits)
}

/// Builds a layout by assigning logical qubits to a physical order prefix.
///
/// `target.physical_qubits` remains the full physical register. `physical_order`
/// controls only which physical qubits the active logical qubits occupy.
fn layout_from_physical_order(
    logical_qubits: &[LogicalQubit],
    target: &RoutingTarget,
    physical_order: &[PhysicalQubit],
) -> Result<Layout, CompilerError> {
    let mapping = logical_qubits
        .iter()
        .copied()
        .zip(physical_order.iter().copied())
        .collect::<BTreeMap<_, _>>();
    Layout::new(
        logical_qubits.to_vec(),
        target.physical_qubits.clone(),
        Some(mapping),
    )
    .map_err(|error| {
        CompilerError::InvariantViolation(format!(
            "sabre layout failed to construct an initial candidate: {error}"
        ))
    })
}

/// Removes duplicate candidate layouts while preserving first occurrence.
///
/// Equality is based on physical-qubit IDs in logical-qubit order, not on
/// object identity or candidate source.
fn deduplicate_layouts(
    candidates: Vec<Layout>,
    logical_qubits: &[LogicalQubit],
) -> Result<Vec<Layout>, CompilerError> {
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for layout in candidates {
        // Signatures use physical IDs in logical-qubit order so layouts built
        // from different candidate sources deduplicate consistently.
        let signature = logical_qubits
            .iter()
            .map(|logical| {
                layout
                    .get_physical(*logical)
                    .map(PhysicalQubit::id)
                    .ok_or_else(|| {
                        CompilerError::InvariantViolation(format!(
                            "sabre layout candidate does not map logical qubit {logical}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, CompilerError>>()?;
        if seen.insert(signature) {
            unique.push(layout);
        }
    }
    Ok(unique)
}

/// Returns the best observed route quality for one refined initial layout.
///
/// The candidate is first checked for reachable interactions. Then several
/// seeded unchecked route trials are run and ranked with the same SABRE trial
/// objective used by the routing core.
fn best_route_quality(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    config: &SabreConfig,
    seed: u64,
) -> Result<TrialQuality, CompilerError> {
    validate_reachable_interactions_for_target(sabre, target, initial_layout)?;
    let mut best: Option<(usize, TrialQuality)> = None;
    for (index, seed) in trial_seeds(Some(seed), config.layout_scoring_trials)
        .into_iter()
        .enumerate()
    {
        let quality =
            route_trial_unchecked(sabre, target, initial_layout, &config.heuristic, seed)?.quality;
        if best.as_ref().is_none_or(|(best_index, best_quality)| {
            compare_trial_quality(
                config.trial_objective,
                quality,
                index,
                *best_quality,
                *best_index,
            )
            .is_lt()
        }) {
            best = Some((index, quality));
        }
    }
    Ok(best
        .expect("layout_scoring_trials is validated to be non-zero")
        .1)
}
