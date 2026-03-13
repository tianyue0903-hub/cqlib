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

//! Structural VF2 mapping pass.
//!
//! This mapper focuses on strict structural embedding first, and then offers a
//! candidate-search fallback for initial-layout selection when strict full
//! matching is not available.
//!
//! The candidate path combines:
//! - partial VF2-style monomorphism seeds
//! - region expansion on the physical topology
//! - weighted ranking by fidelity, topology distance, and gate distribution

use super::{
    FidelityMap, PreparedCircuit, TopologyAdapter, build_output_circuit_from_source,
    map_program_static, normalize_index_pair, preprocess_program,
};
use crate::circuit::{Circuit, Qubit};
use crate::compile::error::CompileError;
use crate::device::Topology;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
/// Weights for candidate scoring components.
///
/// Values are normalized internally. Non-finite or non-positive values are
/// treated as `0.0`; if all weights become zero, defaults are used.
pub struct Vf2ScoreWeights {
    /// Weight for 2q edge fidelity fit.
    pub fidelity: f64,
    /// Weight for topology distance fit.
    pub topology: f64,
    /// Weight for gate-distribution fit between logical and physical qubits.
    pub gate_distribution: f64,
}

impl Default for Vf2ScoreWeights {
    /// Returns baseline scoring weights used by candidate search.
    fn default() -> Self {
        Self {
            fidelity: 0.5,
            topology: 0.3,
            gate_distribution: 0.2,
        }
    }
}

impl Vf2ScoreWeights {
    /// Normalizes positive finite weights so their sum is 1.0.
    ///
    /// Invalid entries (non-finite or non-positive) are treated as zero; if all
    /// values become zero, defaults are restored.
    fn normalized(self) -> Self {
        let mut fidelity = if self.fidelity.is_finite() && self.fidelity > 0.0 {
            self.fidelity
        } else {
            0.0
        };
        let mut topology = if self.topology.is_finite() && self.topology > 0.0 {
            self.topology
        } else {
            0.0
        };
        let mut gate_distribution =
            if self.gate_distribution.is_finite() && self.gate_distribution > 0.0 {
                self.gate_distribution
            } else {
                0.0
            };

        let sum = fidelity + topology + gate_distribution;
        if sum <= 0.0 {
            return Self::default();
        }

        fidelity /= sum;
        topology /= sum;
        gate_distribution /= sum;

        Self {
            fidelity,
            topology,
            gate_distribution,
        }
    }
}

#[derive(Debug, Clone)]
/// Configuration for VF2 initial-layout candidate search.
pub struct Vf2CandidateOptions {
    /// Maximum number of returned candidates.
    pub top_k: usize,
    /// Scoring weights applied to candidate ranking.
    pub weights: Vf2ScoreWeights,
    /// Maximum connected logical subgraphs explored in max-subgraph mode.
    pub max_seed_subgraphs: usize,
    /// Maximum VF2 matches collected per explored logical subgraph.
    pub max_matches_per_subgraph: usize,
    /// Beam width used when expanding physical candidate regions.
    pub region_beam_width: usize,
    /// Oversampling multiplier used before final Top-K filtering.
    pub region_oversample_factor: usize,
}

impl Default for Vf2CandidateOptions {
    /// Returns baseline candidate search options.
    fn default() -> Self {
        Self {
            top_k: 10,
            weights: Vf2ScoreWeights::default(),
            max_seed_subgraphs: 2000,
            max_matches_per_subgraph: 128,
            region_beam_width: 32,
            region_oversample_factor: 3,
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Scoring breakdown for one candidate layout.
pub struct Vf2CandidateScore {
    /// Weighted total score.
    pub total: f64,
    /// 2q fidelity component.
    pub fidelity: f64,
    /// Topology-distance component.
    pub topology_fit: f64,
    /// Gate-distribution component.
    pub gate_distribution: f64,
}

#[derive(Debug, Clone)]
/// One initial-layout candidate and its score.
pub struct Vf2LayoutCandidate {
    /// Physical region selected for the candidate.
    pub region: Vec<Qubit>,
    /// Logical-to-physical mapping for all logical qubits.
    pub logic2phy: Vec<Qubit>,
    /// Scoring details for this candidate.
    pub score: Vf2CandidateScore,
}

#[derive(Debug, Clone)]
/// Internal struct `InteractionGraph` used by compile mapping workflows.
struct InteractionGraph {
    active_nodes: Vec<usize>,
    adjacency: Vec<HashSet<usize>>,
    edges: Vec<(usize, usize)>,
    edge_weights: HashMap<(usize, usize), usize>,
    single_count: Vec<usize>,
    twoq_count: Vec<usize>,
}

#[derive(Debug, Clone)]
/// Internal struct `SubgraphCandidate` used by compile mapping workflows.
struct SubgraphCandidate {
    nodes: Vec<usize>,
    covered_weight: usize,
    edge_count: usize,
}

#[derive(Debug, Clone)]
/// Internal struct `PartialSeed` used by compile mapping workflows.
struct PartialSeed {
    mapping: HashMap<usize, usize>,
    score: i64,
}

#[derive(Debug, Clone)]
/// Internal struct `IndexedCandidate` used by compile mapping workflows.
struct IndexedCandidate {
    region: Vec<usize>,
    logic2phy: Vec<usize>,
    score: Vf2CandidateScore,
}

/// VF2 mapping implementation.
#[derive(Debug, Clone)]
pub struct Vf2Mapping {
    /// Logical -> physical mapping after the latest `execute` call.
    pub logic2phy: Vec<Qubit>,
    /// Physical -> logical mapping after the latest `execute` call.
    pub phy2logic: HashMap<Qubit, usize>,
    topology: TopologyAdapter,
}

impl Vf2Mapping {
    /// Creates a VF2 mapper using topology and optional fidelity map.
    ///
    /// Note: fidelity values are only validated here for interface consistency.
    /// The strict VF2 path is structural; fidelity is used by candidate scoring.
    pub fn new(
        topology: Topology,
        fidelity_map: Option<FidelityMap>,
    ) -> Result<Self, CompileError> {
        let topology = TopologyAdapter::new(&topology, fidelity_map.as_ref())?;
        Ok(Self::from_adapter(topology))
    }

    /// Creates mapper from a precomputed topology adapter.
    pub(crate) fn from_adapter(topology: TopologyAdapter) -> Self {
        Self {
            logic2phy: Vec::new(),
            phy2logic: HashMap::new(),
            topology,
        }
    }

    /// Returns whether a strict subgraph-monomorphic mapping exists.
    pub fn is_subgraph_isomorphic(&self, circuit: &Circuit) -> Result<bool, CompileError> {
        let prepared = preprocess_program(circuit)?.flatten_interaction_circuit();
        let mapping = self.solve_prepared_initial_layout(&prepared)?;
        Ok(mapping.is_some())
    }

    /// Finds a logical->physical initial layout without inserting routing gates.
    ///
    /// The method first tries strict full-graph monomorphism matching. If that fails,
    /// it falls back to candidate generation and returns the top-1 layout.
    pub fn find_initial_layout(
        &self,
        circuit: &Circuit,
    ) -> Result<Option<Vec<Qubit>>, CompileError> {
        let prepared = preprocess_program(circuit)?.flatten_interaction_circuit();
        if let Some(mapping) = self.solve_prepared_initial_layout(&prepared)? {
            let strict_layout = mapping
                .into_iter()
                .map(|idx| self.topology.physical_qubits[idx])
                .collect();
            return Ok(Some(strict_layout));
        }

        let options = Vf2CandidateOptions {
            top_k: 1,
            ..Vf2CandidateOptions::default()
        };
        let fallback = self.find_prepared_layout_candidates(&prepared, &options, false)?;
        Ok(fallback.into_iter().next().map(|c| c.logic2phy))
    }

    /// Finds top-K initial layout candidates with scoring metadata.
    ///
    /// Configuration behavior:
    /// - `top_k`: maximum returned candidates
    /// - `max_seed_subgraphs`: cap for connected logical-subgraph exploration
    /// - `max_matches_per_subgraph`: cap for VF2 matches per subgraph
    /// - `region_beam_width`: beam width for region expansion
    /// - `region_oversample_factor`: pre-filter candidate pool multiplier
    pub fn find_initial_layout_candidates(
        &self,
        circuit: &Circuit,
        options: Option<Vf2CandidateOptions>,
    ) -> Result<Vec<Vf2LayoutCandidate>, CompileError> {
        let prepared = preprocess_program(circuit)?.flatten_interaction_circuit();
        let options = options.unwrap_or_default();
        self.find_prepared_layout_candidates(&prepared, &options, true)
    }

    /// Solves strict prepared-layout mapping without mutating mapper state.
    pub(crate) fn solve_prepared_initial_layout(
        &self,
        prepared: &PreparedCircuit,
    ) -> Result<Option<Vec<usize>>, CompileError> {
        self.find_best_mapping(prepared)
    }

    /// Finds top-K prepared-circuit layout candidates as index layouts.
    ///
    /// Returned layouts are sorted by descending candidate score (best first),
    /// following the same ranking pipeline as `find_initial_layout_candidates`.
    pub(crate) fn find_prepared_layout_candidate_indices(
        &self,
        prepared: &PreparedCircuit,
        options: Option<Vf2CandidateOptions>,
    ) -> Result<Vec<Vec<usize>>, CompileError> {
        let options = options.unwrap_or_default();
        let candidates = self.find_prepared_layout_candidates(prepared, &options, true)?;

        let qubit_to_index: HashMap<Qubit, usize> = self
            .topology
            .physical_qubits
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, q)| (q, idx))
            .collect();

        let mut out = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let mut layout = Vec::with_capacity(candidate.logic2phy.len());
            for q in candidate.logic2phy {
                let Some(&idx) = qubit_to_index.get(&q) else {
                    return Err(CompileError::Internal(format!(
                        "candidate references unknown physical qubit {q}"
                    )));
                };
                layout.push(idx);
            }
            out.push(layout);
        }
        Ok(out)
    }

    /// Executes strict monomorphism-based mapping and returns mapped circuit.
    pub fn execute(&mut self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let program = preprocess_program(circuit)?;
        let prepared = program.flatten_interaction_circuit();
        let mapping_idx = self
            .find_best_mapping(&prepared)?
            .ok_or(CompileError::Vf2NoMapping)?;

        let logic2phy: Vec<Qubit> = mapping_idx
            .iter()
            .map(|&idx| self.topology.physical_qubits[idx])
            .collect();

        let mut phy2logic = HashMap::new();
        for (logical, &physical) in logic2phy.iter().enumerate() {
            phy2logic.insert(physical, logical);
        }

        let mapped_ops = map_program_static(&program, &mapping_idx, &self.topology.physical_qubits);

        self.logic2phy = logic2phy;
        self.phy2logic = phy2logic;

        Ok(build_output_circuit_from_source(circuit, mapped_ops))
    }

    /// Internal helper for find prepared layout candidates.
    fn find_prepared_layout_candidates(
        &self,
        prepared: &PreparedCircuit,
        options: &Vf2CandidateOptions,
        include_strict_seed: bool,
    ) -> Result<Vec<Vf2LayoutCandidate>, CompileError> {
        let logical_width = prepared.logical_qubits.len();
        let physical_width = self.topology.num_qubits();
        if logical_width > physical_width {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: physical_width,
            });
        }

        if options.top_k == 0 || logical_width == 0 {
            return Ok(vec![]);
        }

        let interaction = self.build_interaction_graph(prepared);
        let mut seed_mappings: Vec<HashMap<usize, usize>> = vec![];
        let oversample = options.region_oversample_factor.max(1);
        let target_seed_count = options.top_k.saturating_mul(oversample).max(1);

        if include_strict_seed {
            if interaction.active_nodes.is_empty() {
                let strict_layout = self.assign_isolates(logical_width, &HashMap::new())?;
                let mut strict = HashMap::with_capacity(strict_layout.len());
                for (logical, physical) in strict_layout.into_iter().enumerate() {
                    strict.insert(logical, physical);
                }
                seed_mappings.push(strict);
            } else {
                let strict_partials = self.collect_partial_mappings_for_nodes(
                    &interaction,
                    &interaction.active_nodes,
                    options.max_matches_per_subgraph.max(1),
                )?;
                for partial in strict_partials.into_iter().take(target_seed_count) {
                    seed_mappings.push(partial.mapping);
                }
            }
        }

        if seed_mappings.is_empty() {
            let mut found = self.find_seed_mappings_from_max_subgraph(
                &interaction,
                options,
                target_seed_count,
            )?;
            seed_mappings.append(&mut found);
        }

        if seed_mappings.is_empty() {
            return Ok(vec![]);
        }

        let max_pool = target_seed_count.saturating_mul(4).max(target_seed_count);
        let mut raw_candidates: Vec<IndexedCandidate> = vec![];
        let mut seen_layouts: HashSet<Vec<usize>> = HashSet::new();

        for seed in &seed_mappings {
            let mut regions = self.generate_candidate_regions(seed, logical_width, options);
            if regions.is_empty() {
                continue;
            }

            for region in regions.drain(..) {
                let Some(layout_idx) =
                    self.complete_layout_in_region(logical_width, &interaction, seed, &region)
                else {
                    continue;
                };
                if !seen_layouts.insert(layout_idx.clone()) {
                    continue;
                }

                let score = self.score_layout(&interaction, &layout_idx, &region, options.weights);
                raw_candidates.push(IndexedCandidate {
                    region,
                    logic2phy: layout_idx,
                    score,
                });

                if raw_candidates.len() >= max_pool {
                    break;
                }
            }

            if raw_candidates.len() >= max_pool {
                break;
            }
        }

        if raw_candidates.is_empty() {
            return Ok(vec![]);
        }

        raw_candidates.sort_by(|a, b| {
            b.score
                .total
                .total_cmp(&a.score.total)
                .then_with(|| b.score.fidelity.total_cmp(&a.score.fidelity))
                .then_with(|| b.score.topology_fit.total_cmp(&a.score.topology_fit))
                .then_with(|| {
                    b.score
                        .gate_distribution
                        .total_cmp(&a.score.gate_distribution)
                })
                .then_with(|| vec_lex_cmp(&a.logic2phy, &b.logic2phy))
        });
        raw_candidates.truncate(options.top_k);

        let mut output = Vec::with_capacity(raw_candidates.len());
        for candidate in raw_candidates {
            output.push(Vf2LayoutCandidate {
                region: candidate
                    .region
                    .into_iter()
                    .map(|idx| self.topology.physical_qubits[idx])
                    .collect(),
                logic2phy: candidate
                    .logic2phy
                    .into_iter()
                    .map(|idx| self.topology.physical_qubits[idx])
                    .collect(),
                score: candidate.score,
            });
        }
        Ok(output)
    }

    /// Internal helper for find best mapping.
    fn find_best_mapping(
        &self,
        prepared: &PreparedCircuit,
    ) -> Result<Option<Vec<usize>>, CompileError> {
        let logical_width = prepared.logical_qubits.len();
        let physical_width = self.topology.num_qubits();

        if logical_width > physical_width {
            return Err(CompileError::TopologyTooSmall {
                required: logical_width,
                available: physical_width,
            });
        }

        let interaction = self.build_interaction_graph(prepared);

        // No 2q interactions: assign deterministically by physical degree.
        if interaction.active_nodes.is_empty() {
            return Ok(Some(self.assign_isolates(logical_width, &HashMap::new())?));
        }

        let partials =
            self.collect_partial_monomorphisms(&interaction, &interaction.active_nodes, 1);
        let Some(best_partial) = partials.into_iter().next() else {
            return Ok(None);
        };

        let full_mapping = self.assign_isolates(logical_width, &best_partial.mapping)?;
        Ok(Some(full_mapping))
    }

    /// Internal helper for find seed mappings from max subgraph.
    fn find_seed_mappings_from_max_subgraph(
        &self,
        interaction: &InteractionGraph,
        options: &Vf2CandidateOptions,
        target_seed_count: usize,
    ) -> Result<Vec<HashMap<usize, usize>>, CompileError> {
        let mut subgraphs =
            self.enumerate_connected_subgraphs(interaction, options.max_seed_subgraphs.max(1));
        if subgraphs.is_empty() {
            return Ok(vec![]);
        }

        subgraphs.sort_by(|a, b| {
            b.covered_weight
                .cmp(&a.covered_weight)
                .then_with(|| b.nodes.len().cmp(&a.nodes.len()))
                .then_with(|| b.edge_count.cmp(&a.edge_count))
                .then_with(|| vec_lex_cmp(&a.nodes, &b.nodes))
        });

        let mut seeds = vec![];
        let mut best_weight: Option<usize> = None;
        for subgraph in subgraphs {
            if subgraph.nodes.len() < 2 || subgraph.covered_weight == 0 {
                continue;
            }
            if let Some(weight) = best_weight {
                if subgraph.covered_weight < weight {
                    break;
                }
            }

            let partials = self.collect_partial_mappings_for_nodes(
                interaction,
                &subgraph.nodes,
                options.max_matches_per_subgraph.max(1),
            )?;
            if partials.is_empty() {
                continue;
            }

            if best_weight.is_none() {
                best_weight = Some(subgraph.covered_weight);
            }
            for partial in partials {
                seeds.push(partial.mapping);
                if seeds.len() >= target_seed_count {
                    return Ok(seeds);
                }
            }
        }

        Ok(seeds)
    }

    /// Internal helper for enumerate connected subgraphs.
    fn enumerate_connected_subgraphs(
        &self,
        interaction: &InteractionGraph,
        cap: usize,
    ) -> Vec<SubgraphCandidate> {
        if cap == 0 || interaction.active_nodes.is_empty() {
            return vec![];
        }

        let mut queue = VecDeque::new();
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        for &node in &interaction.active_nodes {
            let singleton = vec![node];
            if seen.insert(singleton.clone()) {
                queue.push_back(singleton);
            }
        }

        let mut out = Vec::with_capacity(cap);
        while let Some(nodes) = queue.pop_front() {
            let mut marked = vec![false; interaction.adjacency.len()];
            for &n in &nodes {
                marked[n] = true;
            }

            let mut covered_weight = 0usize;
            let mut edge_count = 0usize;
            for &(u, v) in &interaction.edges {
                if marked[u] && marked[v] {
                    edge_count += 1;
                    covered_weight += interaction.edge_weights.get(&(u, v)).copied().unwrap_or(0);
                }
            }
            out.push(SubgraphCandidate {
                nodes: nodes.clone(),
                covered_weight,
                edge_count,
            });
            if out.len() >= cap {
                break;
            }

            let mut boundary: Vec<usize> = vec![];
            let mut boundary_seen: HashSet<usize> = HashSet::new();
            for &u in &nodes {
                let mut neighbors: Vec<usize> = interaction.adjacency[u].iter().copied().collect();
                neighbors.sort_unstable();
                for v in neighbors {
                    if marked[v] || !boundary_seen.insert(v) {
                        continue;
                    }
                    boundary.push(v);
                }
            }
            boundary.sort_unstable();

            for v in boundary {
                let mut next = nodes.clone();
                next.push(v);
                next.sort_unstable();
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
                if seen.len() >= cap.saturating_mul(8) {
                    break;
                }
            }
            if seen.len() >= cap.saturating_mul(8) {
                break;
            }
        }

        out
    }

    /// Internal helper for collect partial mappings for nodes.
    fn collect_partial_mappings_for_nodes(
        &self,
        interaction: &InteractionGraph,
        logical_nodes: &[usize],
        max_matches: usize,
    ) -> Result<Vec<PartialSeed>, CompileError> {
        if logical_nodes.is_empty() || max_matches == 0 {
            return Ok(vec![]);
        }
        let partials = self.collect_partial_monomorphisms(interaction, logical_nodes, max_matches);

        Ok(partials)
    }

    /// Internal helper for collect partial monomorphisms.
    fn collect_partial_monomorphisms(
        &self,
        interaction: &InteractionGraph,
        logical_nodes: &[usize],
        max_matches: usize,
    ) -> Vec<PartialSeed> {
        if logical_nodes.is_empty() || max_matches == 0 {
            return vec![];
        }

        let logical_width = interaction.adjacency.len();
        let physical_width = self.topology.num_qubits();
        let mut in_subgraph = vec![false; logical_width];
        for &l in logical_nodes {
            in_subgraph[l] = true;
        }

        let mut sub_neighbors = vec![Vec::<usize>::new(); logical_width];
        let mut sub_degree = vec![0usize; logical_width];
        for &l in logical_nodes {
            let mut ns: Vec<usize> = interaction.adjacency[l]
                .iter()
                .copied()
                .filter(|n| in_subgraph[*n])
                .collect();
            ns.sort_unstable();
            sub_degree[l] = ns.len();
            sub_neighbors[l] = ns;
        }

        let mut ordered_logical = logical_nodes.to_vec();
        ordered_logical.sort_by(|a, b| {
            sub_degree[*b]
                .cmp(&sub_degree[*a])
                .then_with(|| interaction.twoq_count[*b].cmp(&interaction.twoq_count[*a]))
                .then_with(|| interaction.single_count[*b].cmp(&interaction.single_count[*a]))
                .then_with(|| a.cmp(b))
        });

        let mut physical_order: Vec<usize> = (0..physical_width).collect();
        physical_order.sort_by(|a, b| {
            self.topology.neighbors[*b]
                .len()
                .cmp(&self.topology.neighbors[*a].len())
                .then_with(|| {
                    self.topology.physical_qubits[*a]
                        .id()
                        .cmp(&self.topology.physical_qubits[*b].id())
                })
        });

        let mut mapping_by_logical = vec![usize::MAX; logical_width];
        let mut used = vec![false; physical_width];
        let mut mappings: Vec<HashMap<usize, usize>> = vec![];

        self.collect_partial_monomorphisms_dfs(
            0,
            &ordered_logical,
            &sub_neighbors,
            &sub_degree,
            &physical_order,
            &mut mapping_by_logical,
            &mut used,
            max_matches,
            &mut mappings,
        );

        let mut out: Vec<PartialSeed> = mappings
            .into_iter()
            .map(|mapping| PartialSeed {
                score: self.score_partial_mapping(interaction, &mapping),
                mapping,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| self.partial_mapping_lex_cmp(&a.mapping, &b.mapping))
        });
        out
    }

    /// Internal helper for collect partial monomorphisms dfs.
    fn collect_partial_monomorphisms_dfs(
        &self,
        depth: usize,
        ordered_logical: &[usize],
        sub_neighbors: &[Vec<usize>],
        sub_degree: &[usize],
        physical_order: &[usize],
        mapping_by_logical: &mut [usize],
        used: &mut [bool],
        max_matches: usize,
        out: &mut Vec<HashMap<usize, usize>>,
    ) {
        if out.len() >= max_matches {
            return;
        }
        if depth == ordered_logical.len() {
            let mut mapping = HashMap::with_capacity(ordered_logical.len());
            for &l in ordered_logical {
                mapping.insert(l, mapping_by_logical[l]);
            }
            out.push(mapping);
            return;
        }

        let logical = ordered_logical[depth];
        for &physical in physical_order {
            if used[physical] {
                continue;
            }
            if self.topology.neighbors[physical].len() < sub_degree[logical] {
                continue;
            }

            let mut ok = true;
            for &nbr in &sub_neighbors[logical] {
                let mapped = mapping_by_logical[nbr];
                if mapped == usize::MAX {
                    continue;
                }
                if !self.topology.is_adjacent(physical, mapped) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }

            mapping_by_logical[logical] = physical;
            used[physical] = true;

            self.collect_partial_monomorphisms_dfs(
                depth + 1,
                ordered_logical,
                sub_neighbors,
                sub_degree,
                physical_order,
                mapping_by_logical,
                used,
                max_matches,
                out,
            );

            used[physical] = false;
            mapping_by_logical[logical] = usize::MAX;
            if out.len() >= max_matches {
                return;
            }
        }
    }

    /// Internal helper for score partial mapping.
    fn score_partial_mapping(
        &self,
        interaction: &InteractionGraph,
        mapping: &HashMap<usize, usize>,
    ) -> i64 {
        let mut score = 0i64;
        for (&logical, &physical) in mapping {
            let logical_degree = interaction.adjacency[logical].len() as i64;
            let physical_degree = self.topology.neighbors[physical].len() as i64;
            score += logical_degree * physical_degree;
        }
        score
    }

    /// Internal helper for generate candidate regions.
    fn generate_candidate_regions(
        &self,
        seed_mapping: &HashMap<usize, usize>,
        logical_width: usize,
        options: &Vf2CandidateOptions,
    ) -> Vec<Vec<usize>> {
        if logical_width == 0 {
            return vec![];
        }
        let beam_width = options.region_beam_width.max(1);

        let mut seed_region: Vec<usize> = seed_mapping.values().copied().collect();
        seed_region.sort_unstable();
        seed_region.dedup();
        if seed_region.len() > logical_width {
            return vec![];
        }
        if seed_region.len() == logical_width {
            return vec![seed_region];
        }

        let mut current: Vec<Vec<usize>>;
        if seed_region.is_empty() {
            let mut start_nodes: Vec<usize> = (0..self.topology.num_qubits()).collect();
            start_nodes.sort_by(|a, b| {
                self.topology.neighbors[*b]
                    .len()
                    .cmp(&self.topology.neighbors[*a].len())
                    .then_with(|| {
                        self.topology.physical_qubits[*a]
                            .id()
                            .cmp(&self.topology.physical_qubits[*b].id())
                    })
            });

            current = start_nodes
                .into_iter()
                .take(beam_width)
                .map(|n| vec![n])
                .collect();
        } else {
            current = vec![seed_region];
        }

        let mut visited: HashSet<Vec<usize>> = HashSet::new();
        for state in &current {
            visited.insert(state.clone());
        }

        let mut size = current[0].len();
        while size < logical_width {
            let mut next_states: Vec<Vec<usize>> = vec![];
            let mut next_seen: HashSet<Vec<usize>> = HashSet::new();

            for region in &current {
                let mut candidates = self.boundary_nodes(region);
                if candidates.is_empty() {
                    continue;
                }

                candidates.sort_by(|a, b| self.compare_expansion_nodes(region, *a, *b));
                for node in candidates.into_iter().take(beam_width) {
                    let mut expanded = region.clone();
                    expanded.push(node);
                    expanded.sort_unstable();
                    if visited.contains(&expanded) || !next_seen.insert(expanded.clone()) {
                        continue;
                    }
                    next_states.push(expanded);
                }
            }

            if next_states.is_empty() {
                return vec![];
            }

            next_states.sort_by(|a, b| self.compare_regions(a, b));
            next_states.truncate(beam_width);
            for s in &next_states {
                visited.insert(s.clone());
            }

            current = next_states;
            size += 1;
        }

        current.sort_by(|a, b| vec_lex_cmp(a, b));
        current.dedup();
        current
    }

    /// Internal helper for boundary nodes.
    fn boundary_nodes(&self, region: &[usize]) -> Vec<usize> {
        let mut in_region = vec![false; self.topology.num_qubits()];
        for &p in region {
            in_region[p] = true;
        }

        let mut boundary = vec![];
        let mut seen = HashSet::new();
        for &p in region {
            let mut neighbors = self.topology.neighbors[p].clone();
            neighbors.sort_unstable();
            for n in neighbors {
                if in_region[n] || !seen.insert(n) {
                    continue;
                }
                boundary.push(n);
            }
        }
        boundary
    }

    /// Internal helper for compare expansion nodes.
    fn compare_expansion_nodes(&self, region: &[usize], a: usize, b: usize) -> Ordering {
        let score_a = self.expansion_node_score(region, a);
        let score_b = self.expansion_node_score(region, b);

        score_b
            .0
            .cmp(&score_a.0)
            .then_with(|| score_b.1.total_cmp(&score_a.1))
            .then_with(|| score_b.2.cmp(&score_a.2))
            .then_with(|| score_a.3.cmp(&score_b.3))
    }

    /// Internal helper for expansion node score.
    fn expansion_node_score(&self, region: &[usize], node: usize) -> (usize, f64, usize, u32) {
        let mut connections = 0usize;
        let mut fidelity_sum = 0.0;

        for &r in region {
            if self.topology.is_adjacent(node, r) {
                connections += 1;
                fidelity_sum += self.topology.edge_fidelity(node, r);
            }
        }

        let avg_fidelity = if connections > 0 {
            fidelity_sum / connections as f64
        } else {
            0.0
        };

        (
            connections,
            avg_fidelity,
            self.topology.neighbors[node].len(),
            self.topology.physical_qubits[node].id(),
        )
    }

    /// Internal helper for compare regions.
    fn compare_regions(&self, a: &[usize], b: &[usize]) -> Ordering {
        let score_a = self.region_score(a);
        let score_b = self.region_score(b);
        score_b
            .0
            .cmp(&score_a.0)
            .then_with(|| score_b.1.cmp(&score_a.1))
            .then_with(|| vec_lex_cmp(a, b))
    }

    /// Internal helper for region score.
    fn region_score(&self, region: &[usize]) -> (usize, usize) {
        let region_set: HashSet<usize> = region.iter().copied().collect();
        let mut internal_edges = 0usize;
        let mut degree_sum = 0usize;

        for &u in region {
            let mut deg_in_region = 0usize;
            for &v in &self.topology.neighbors[u] {
                if region_set.contains(&v) {
                    deg_in_region += 1;
                    if u < v {
                        internal_edges += 1;
                    }
                }
            }
            degree_sum += deg_in_region;
        }
        (internal_edges, degree_sum)
    }

    /// Internal helper for complete layout in region.
    fn complete_layout_in_region(
        &self,
        logical_width: usize,
        interaction: &InteractionGraph,
        seed_mapping: &HashMap<usize, usize>,
        region: &[usize],
    ) -> Option<Vec<usize>> {
        if region.len() != logical_width {
            return None;
        }

        let region_set: HashSet<usize> = region.iter().copied().collect();
        let mut layout = vec![usize::MAX; logical_width];
        let mut used = HashSet::new();
        for (&logical, &physical) in seed_mapping {
            if logical >= logical_width || !region_set.contains(&physical) || !used.insert(physical)
            {
                return None;
            }
            layout[logical] = physical;
        }

        let mut degree_in_region = vec![0usize; self.topology.num_qubits()];
        let mut max_degree_in_region = 0usize;
        for &p in region {
            let deg = self.degree_in_region(p, &region_set);
            degree_in_region[p] = deg;
            max_degree_in_region = max_degree_in_region.max(deg);
        }
        let max_degree_in_region = max_degree_in_region as f64;

        let mut logical_order: Vec<usize> = (0..logical_width).collect();
        logical_order.sort_by(|a, b| {
            interaction.twoq_count[*b]
                .cmp(&interaction.twoq_count[*a])
                .then_with(|| interaction.single_count[*b].cmp(&interaction.single_count[*a]))
                .then_with(|| a.cmp(b))
        });

        for logical in logical_order {
            if layout[logical] != usize::MAX {
                continue;
            }

            let mut best: Option<(usize, f64, u32)> = None;
            for &physical in region {
                if used.contains(&physical) {
                    continue;
                }
                let score = self.candidate_score_for_assignment(
                    logical,
                    physical,
                    interaction,
                    &layout,
                    &degree_in_region,
                    max_degree_in_region,
                );
                let tie = self.topology.physical_qubits[physical].id();
                let update = match best {
                    None => true,
                    Some((_, best_score, best_tie)) => {
                        score > best_score || (score == best_score && tie < best_tie)
                    }
                };
                if update {
                    best = Some((physical, score, tie));
                }
            }

            let Some((chosen, _, _)) = best else {
                return None;
            };
            used.insert(chosen);
            layout[logical] = chosen;
        }

        if layout.iter().any(|&p| p == usize::MAX) {
            return None;
        }
        Some(layout)
    }

    /// Internal helper for candidate score for assignment.
    fn candidate_score_for_assignment(
        &self,
        logical: usize,
        physical: usize,
        interaction: &InteractionGraph,
        current_layout: &[usize],
        degree_in_region: &[usize],
        max_degree_in_region: f64,
    ) -> f64 {
        let inf = (self.topology.num_qubits() as u32)
            .saturating_mul(2)
            .saturating_add(1);
        let mut neighbor_compat = 0.0;
        for &nbr in &interaction.adjacency[logical] {
            let mapped = current_layout[nbr];
            if mapped == usize::MAX {
                continue;
            }

            let edge = normalize_index_pair(logical, nbr);
            let weight = interaction.edge_weights.get(&edge).copied().unwrap_or(1) as f64;
            let dist = self.topology.dist[physical][mapped];
            if dist == 0 || dist >= inf {
                continue;
            }

            let adjacent = self.topology.is_adjacent(physical, mapped);
            let proximity = if adjacent { 1.0 } else { 1.0 / dist as f64 };
            let fidelity = if adjacent {
                self.topology.edge_fidelity(physical, mapped)
            } else {
                1.0
            };
            neighbor_compat += weight * proximity * fidelity;
        }

        let capacity_fit = if max_degree_in_region > 0.0 {
            degree_in_region[physical] as f64 / max_degree_in_region
        } else {
            0.0
        };
        0.6 * neighbor_compat + 0.4 * capacity_fit
    }

    /// Internal helper for score layout.
    fn score_layout(
        &self,
        interaction: &InteractionGraph,
        layout: &[usize],
        region: &[usize],
        weights: Vf2ScoreWeights,
    ) -> Vf2CandidateScore {
        let inf = (self.topology.num_qubits() as u32)
            .saturating_mul(2)
            .saturating_add(1);
        let normalized = weights.normalized();
        let mut total_weight = 0.0;
        let mut fidelity_sum = 0.0;
        let mut topology_sum = 0.0;

        for (&(u, v), &w) in &interaction.edge_weights {
            let weight = w as f64;
            total_weight += weight;

            let pu = layout[u];
            let pv = layout[v];
            if self.topology.is_adjacent(pu, pv) {
                fidelity_sum += weight * self.topology.edge_fidelity(pu, pv);
            }

            let dist = self.topology.dist[pu][pv];
            if dist > 0 && dist < inf {
                topology_sum += weight * (1.0 / dist as f64);
            }
        }

        let fidelity = if total_weight > 0.0 {
            clamp01(fidelity_sum / total_weight)
        } else {
            1.0
        };
        let topology_fit = if total_weight > 0.0 {
            clamp01(topology_sum / total_weight)
        } else {
            1.0
        };

        let region_set: HashSet<usize> = region.iter().copied().collect();
        let mut degree_in_region = vec![0usize; self.topology.num_qubits()];
        let mut max_degree = 0usize;
        for &p in region {
            let deg = self.degree_in_region(p, &region_set);
            degree_in_region[p] = deg;
            max_degree = max_degree.max(deg);
        }
        let max_degree = max_degree as f64;

        let mut demand_weights_sum = 0.0;
        let mut gate_distribution_sum = 0.0;
        for logical in 0..layout.len() {
            let demand_weight =
                (interaction.single_count[logical] + interaction.twoq_count[logical]) as f64;
            let weight = if demand_weight > 0.0 {
                demand_weight
            } else {
                1.0
            };
            demand_weights_sum += weight;

            let total = interaction.single_count[logical] + interaction.twoq_count[logical];
            let demand = if total > 0 {
                interaction.twoq_count[logical] as f64 / total as f64
            } else {
                0.0
            };
            let supply = if max_degree > 0.0 {
                degree_in_region[layout[logical]] as f64 / max_degree
            } else {
                0.0
            };
            let fit = clamp01(1.0 - (demand - supply).abs());
            gate_distribution_sum += weight * fit;
        }
        let gate_distribution = if demand_weights_sum > 0.0 {
            clamp01(gate_distribution_sum / demand_weights_sum)
        } else {
            1.0
        };

        let total = clamp01(
            normalized.fidelity * fidelity
                + normalized.topology * topology_fit
                + normalized.gate_distribution * gate_distribution,
        );
        Vf2CandidateScore {
            total,
            fidelity,
            topology_fit,
            gate_distribution,
        }
    }

    /// Internal helper for degree in region.
    fn degree_in_region(&self, physical: usize, region_set: &HashSet<usize>) -> usize {
        self.topology.neighbors[physical]
            .iter()
            .filter(|&&n| region_set.contains(&n))
            .count()
    }

    /// Internal helper for assign isolates.
    fn assign_isolates(
        &self,
        logical_width: usize,
        seeded_mapping: &HashMap<usize, usize>,
    ) -> Result<Vec<usize>, CompileError> {
        let mut result = vec![usize::MAX; logical_width];
        let mut used = vec![false; self.topology.num_qubits()];

        for (&logical, &physical) in seeded_mapping {
            if logical < logical_width {
                result[logical] = physical;
                used[physical] = true;
            }
        }

        let mut remaining_physical: Vec<usize> = (0..self.topology.num_qubits())
            .filter(|idx| !used[*idx])
            .collect();

        remaining_physical.sort_by(|a, b| {
            let deg_a = self.topology.neighbors[*a].len();
            let deg_b = self.topology.neighbors[*b].len();
            deg_b.cmp(&deg_a).then_with(|| {
                self.topology.physical_qubits[*a]
                    .id()
                    .cmp(&self.topology.physical_qubits[*b].id())
            })
        });

        let mut cursor = 0usize;
        for logical in 0..logical_width {
            if result[logical] != usize::MAX {
                continue;
            }
            let Some(&physical) = remaining_physical.get(cursor) else {
                return Err(CompileError::TopologyTooSmall {
                    required: logical_width,
                    available: self.topology.num_qubits(),
                });
            };
            cursor += 1;
            result[logical] = physical;
        }

        Ok(result)
    }

    /// Internal helper for partial mapping lex cmp.
    fn partial_mapping_lex_cmp(
        &self,
        left: &HashMap<usize, usize>,
        right: &HashMap<usize, usize>,
    ) -> Ordering {
        let mut left_pairs: Vec<(usize, u32)> = left
            .iter()
            .map(|(&l, &p)| (l, self.topology.physical_qubits[p].id()))
            .collect();
        let mut right_pairs: Vec<(usize, u32)> = right
            .iter()
            .map(|(&l, &p)| (l, self.topology.physical_qubits[p].id()))
            .collect();
        left_pairs.sort_unstable_by_key(|(l, _)| *l);
        right_pairs.sort_unstable_by_key(|(l, _)| *l);

        for ((_, a), (_, b)) in left_pairs.iter().zip(right_pairs.iter()) {
            match a.cmp(b) {
                Ordering::Less => return Ordering::Less,
                Ordering::Greater => return Ordering::Greater,
                Ordering::Equal => {}
            }
        }
        left_pairs.len().cmp(&right_pairs.len())
    }

    /// Internal helper for build interaction graph.
    fn build_interaction_graph(&self, prepared: &PreparedCircuit) -> InteractionGraph {
        let logical_width = prepared.logical_qubits.len();
        let mut adjacency = vec![HashSet::new(); logical_width];
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        let mut edge_weights: HashMap<(usize, usize), usize> = HashMap::new();
        let mut single_count = vec![0usize; logical_width];
        let mut twoq_count = vec![0usize; logical_width];

        for prep_op in &prepared.operations {
            match prep_op.logical_qubits.len() {
                1 => {
                    let q = prep_op.logical_qubits[0];
                    single_count[q] = single_count[q].saturating_add(1);
                }
                2 => {
                    let u = prep_op.logical_qubits[0];
                    let v = prep_op.logical_qubits[1];
                    if u == v {
                        continue;
                    }
                    let edge = normalize_index_pair(u, v);
                    edge_set.insert(edge);
                    adjacency[u].insert(v);
                    adjacency[v].insert(u);
                    *edge_weights.entry(edge).or_insert(0) += 1;
                    twoq_count[u] = twoq_count[u].saturating_add(1);
                    twoq_count[v] = twoq_count[v].saturating_add(1);
                }
                _ => {}
            }
        }

        let mut active_nodes: Vec<usize> = adjacency
            .iter()
            .enumerate()
            .filter_map(|(idx, neigh)| if neigh.is_empty() { None } else { Some(idx) })
            .collect();
        active_nodes.sort_unstable();

        let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        edges.sort_unstable();

        InteractionGraph {
            active_nodes,
            adjacency,
            edges,
            edge_weights,
            single_count,
            twoq_count,
        }
    }
}

/// Internal helper for vec lex cmp.
fn vec_lex_cmp(a: &[usize], b: &[usize]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.cmp(y) {
            Ordering::Equal => {}
            non_eq => return non_eq,
        }
    }
    a.len().cmp(&b.len())
}

/// Internal helper for clamp01.
fn clamp01(v: f64) -> f64 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(0.0, 1.0)
}
