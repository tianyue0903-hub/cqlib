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

use super::dag::SabreDag;
use super::heuristic::SabreConfig;
use super::routing::{
    RoutingTarget, SabreRoutingResult, TrialQuality, compare_trial_quality,
    normalize_initial_layout, route_trial, route_trial_unchecked, sabre_route, trial_seeds,
    validate_config, validate_reachable_interactions,
};
use crate::circuit::Circuit;
use crate::compiler::CompilerError;
use crate::compiler::transform::layout::{
    CircuitLayoutAnalysis, LayoutDiagnostics, LayoutObjective, LayoutResult, LayoutScore,
    PhysicalLayoutGraph, Vf2LayoutConfig, analyze_circuit_for_layout, build_physical_layout_graph,
    greedy_layout_with_physical, vf2_perfect_layout_with_physical,
};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// Combined SABRE layout-and-routing result.
#[derive(Debug, Clone)]
pub struct SabreCompileResult {
    /// Routed physical circuit and routing metadata.
    pub routing: SabreRoutingResult,
    /// Score of the refined initial layout, when an objective was supplied.
    pub layout_score: Option<crate::compiler::transform::layout::LayoutScore>,
}

/// Refines an initial layout with SABRE forward/backward trial routing.
///
/// If `initial_layout` is supplied, it is treated as a concrete candidate and
/// additional random/heuristic candidates are still evaluated according to
/// `config`.  The returned layout is selected by final-route SWAP count, with
/// `objective` used as a deterministic tie-breaker.
pub fn sabre_refine_layout(
    circuit: &Circuit,
    device: &Device,
    initial_layout: Option<&Layout>,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    validate_config(config)?;
    let analysis = analyze_circuit_for_layout(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    let target = RoutingTarget::from_physical(&physical)?;
    let sabre = SabreDag::from_operations(circuit.operations())?;
    let forwards = sabre.only_interactions();
    let backwards = forwards.reverse_interactions();
    let logical_qubits = circuit
        .qubits()
        .into_iter()
        .map(LogicalQubit::from_qubit)
        .collect::<Vec<_>>();

    if logical_qubits.len() > target.physical_qubits.len() {
        return Err(CompilerError::InvalidInput(format!(
            "sabre layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            logical_qubits.len(),
            target.physical_qubits.len()
        )));
    }

    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random));
    let candidates = initial_layout_candidates(
        CandidateLayoutContext {
            analysis: &analysis,
            logical_qubits: &logical_qubits,
            physical: &physical,
            target: &target,
            objective,
        },
        initial_layout,
        config.layout_trials,
        &mut rng,
    )?;
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, layout)| {
            let refinement_seeds = (0..config.refinement_iterations)
                .map(|_| (rand::Rng::random(&mut rng), rand::Rng::random(&mut rng)))
                .collect();
            CandidateTrial {
                index,
                layout,
                refinement_seeds,
                scoring_seed: rand::Rng::random(&mut rng),
            }
        })
        .collect::<Vec<_>>();
    let candidates_evaluated = trials.len();

    let evaluations = trials
        .into_par_iter()
        .map(|trial| {
            let mut refined = trial.layout;
            for (forward_seed, backward_seed) in trial.refinement_seeds {
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
            let score = objective.score_layout(&analysis, &physical, &refined)?;
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
    Ok(LayoutResult {
        diagnostics: LayoutDiagnostics {
            is_perfect: is_perfect_layout(&analysis, &physical, &layout),
            candidates_evaluated,
            used_fidelity: score.used_fidelity,
            notes: vec![format!(
                "selected SABRE refined layout with {swap_count} final-route swaps"
            )],
        },
        layout,
        score: Some(score),
    })
}

/// Runs SABRE layout refinement and final routing as one operation.
pub fn sabre_layout_and_route(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<SabreCompileResult, CompilerError> {
    let layout_result = sabre_refine_layout(circuit, device, None, objective, config)?;
    let routing = sabre_route(circuit, device, &layout_result.layout, config)?;
    Ok(SabreCompileResult {
        routing,
        layout_score: layout_result.score,
    })
}

struct CandidateTrial {
    index: usize,
    layout: Layout,
    refinement_seeds: Vec<(u64, u64)>,
    scoring_seed: u64,
}

struct CandidateEvaluation {
    index: usize,
    route_quality: TrialQuality,
    layout: Layout,
    score: LayoutScore,
}

struct CandidateLayoutContext<'a> {
    analysis: &'a CircuitLayoutAnalysis,
    logical_qubits: &'a [LogicalQubit],
    physical: &'a PhysicalLayoutGraph,
    target: &'a RoutingTarget,
    objective: &'a LayoutObjective,
}

fn initial_layout_candidates(
    context: CandidateLayoutContext<'_>,
    initial_layout: Option<&Layout>,
    layout_trials: usize,
    rng: &mut StdRng,
) -> Result<Vec<Layout>, CompilerError> {
    let mut candidates = Vec::new();
    if let Some(layout) = initial_layout {
        candidates.push(normalize_initial_layout(
            context.logical_qubits,
            context.target,
            layout,
        )?);
    }

    let trivial = layout_from_physical_order(
        context.logical_qubits,
        context.target,
        &context.target.physical_qubits,
    )?;
    candidates.push(trivial);

    let mut reverse_physical = context.target.physical_qubits.clone();
    reverse_physical.reverse();
    candidates.push(layout_from_physical_order(
        context.logical_qubits,
        context.target,
        &reverse_physical,
    )?);

    if let Ok(greedy) =
        greedy_layout_with_physical(context.analysis, context.physical, context.objective)
    {
        candidates.push(normalize_initial_layout(
            context.logical_qubits,
            context.target,
            &greedy.layout,
        )?);
    }
    if let Ok(vf2) = vf2_perfect_layout_with_physical(
        context.analysis,
        context.physical,
        context.objective,
        &Vf2LayoutConfig::default(),
    ) {
        candidates.push(normalize_initial_layout(
            context.logical_qubits,
            context.target,
            &vf2.layout,
        )?);
    }

    for _ in 0..layout_trials {
        let mut physical_order = context.target.physical_qubits.clone();
        physical_order.shuffle(rng);
        candidates.push(layout_from_physical_order(
            context.logical_qubits,
            context.target,
            &physical_order,
        )?);
    }

    deduplicate_layouts(candidates, context.logical_qubits)
}

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
            "sabre failed to construct an initial layout candidate: {error}"
        ))
    })
}

fn deduplicate_layouts(
    candidates: Vec<Layout>,
    logical_qubits: &[LogicalQubit],
) -> Result<Vec<Layout>, CompilerError> {
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for layout in candidates {
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

fn best_route_quality(
    sabre: &SabreDag,
    target: &RoutingTarget,
    initial_layout: &Layout,
    config: &SabreConfig,
    seed: u64,
) -> Result<TrialQuality, CompilerError> {
    validate_reachable_interactions(sabre, target, initial_layout)?;
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

fn is_perfect_layout(
    analysis: &crate::compiler::transform::layout::CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    layout: &Layout,
) -> bool {
    analysis
        .interactions
        .interactions()
        .iter()
        .all(|interaction| {
            let Some(left) = layout.get_physical(interaction.left) else {
                return false;
            };
            let Some(right) = layout.get_physical(interaction.right) else {
                return false;
            };
            interaction.weight <= 0.0 || physical.is_adjacent_undirected(left, right)
        })
}
