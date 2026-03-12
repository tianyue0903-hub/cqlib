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

//! Genetic Algorithm-based quantum circuit mapping optimizer.
//!
//! This module implements a genetic algorithm approach to find optimal initial
//! qubit mappings for quantum circuit routing. The algorithm evolves a population
//! of mapping candidates through selection, crossover, and mutation operations
//! to minimize the number of SWAP gates required during circuit routing.
//!
//! # Overview
//!
//! The genetic algorithm works as follows:
//! 1. **Initialization**: Generate random initial mappings from valid connected regions
//! 2. **Evaluation**: Use SABRE routing to evaluate each mapping's fitness
//! 3. **Selection**: Select parents using roulette wheel selection based on fitness
//! 4. **Crossover**: Exchange mapping segments in high-SWAP regions
//! 5. **Mutation**: Randomly modify boundary qubits in the mapping
//! 6. **Evolution**: Repeat for multiple generations
//!
//! ```

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::{FidelityMap, TopologyAdapter};
use crate::circuit::{Circuit, Instruction, StandardGate};
use crate::compile::error::CompileError;
use crate::compile::mapping::sabre::{SabreConfig, SabreMapping};
use crate::device::Topology;
use rayon::prelude::*;

/// Configuration for Genetic Algorithm mapping.
#[derive(Debug, Clone)]
pub struct GaConfig {
    /// Number of individuals in the population.
    pub population: usize,
    /// Probability for an individual to be selected.
    pub select_prob: f64,
    /// Probability for crossover operation.
    pub crossover_prob: f64,
    /// Probability for mutation operation.
    pub mutation_prob: f64,
    /// Probability for a forced mutation when no valid mutation is found.
    pub forced_mutation_prob: f64,
    /// Number of qubits involved in a crossover segment.
    pub crossover_qubit_number: usize,
    /// Number of generations/iterations to evolve.
    pub update_iters: usize,
    /// Configuration for the underlying SABRE evaluations.
    pub sabre_config: SabreConfig,
    /// Random seed (`-1` means auto-seeded).
    pub seed: i64,
}

impl Default for GaConfig {
    /// Returns default configuration for GA Mapping.
    fn default() -> Self {
        Self {
            population: 10,
            select_prob: 0.4,
            crossover_prob: 0.4,
            mutation_prob: 0.25,
            forced_mutation_prob: 0.05,
            crossover_qubit_number: 3,
            update_iters: 5,
            sabre_config: SabreConfig::default(),
            seed: -1,
        }
    }
}

/// Genetic Algorithm-based mapping implementation.
#[derive(Debug, Clone)]
pub struct GeneticAlgMapping {
    topology: TopologyAdapter,
    original_topology: Topology,
    fidelity_map: Option<FidelityMap>,
    config: GaConfig,

    qubit_number: usize,
    is_ideal_topology: bool,
    layout_regions: Vec<Vec<usize>>,

    population_space: Vec<Vec<usize>>,
    population_score: Vec<f64>,
    individual_layouts: Vec<Vec<(usize, usize, usize)>>,

    circuit_width: usize,
    circuit_size: usize,

    best_individual: Vec<usize>,
    best_score: f64,
    best_circuit: Option<Circuit>,

    rng: StdRng,
}

impl GeneticAlgMapping {
    /// Creates a GA mapper with an optional fidelity map and explicit config.
    pub fn new(
        topology: Topology,
        config: GaConfig,
        fidelity_map: Option<FidelityMap>,
        invalid_qubits: Option<HashSet<usize>>,
    ) -> Result<Self, CompileError> {
        let topology_adapter = TopologyAdapter::new(&topology, fidelity_map.as_ref())?;

        let qubit_number = topology_adapter.num_qubits();

        let mut is_ideal_topology = true;
        for i in 0..qubit_number {
            for j in 0..qubit_number {
                if i != j
                    && topology_adapter.is_adjacent(i, j)
                    && (topology_adapter.edge_fidelity(i, j) - 1.0).abs() > 1e-9
                {
                    is_ideal_topology = false;
                    break;
                }
            }
        }

        let invalid_qubits_set = invalid_qubits.unwrap_or_default();

        let layout_regions = Self::get_all_connected_nodes(&topology_adapter, &invalid_qubits_set);

        let seed = if config.seed >= 0 {
            config.seed as u64
        } else {
            let mut trng = rand::rng();
            trng.random::<u64>()
        };

        Ok(Self {
            original_topology: topology,
            topology: topology_adapter,
            fidelity_map: fidelity_map,
            config: config.clone(),
            qubit_number: qubit_number,
            is_ideal_topology,
            layout_regions,
            population_space: Vec::with_capacity(config.population),
            population_score: Vec::with_capacity(config.population),
            individual_layouts: Vec::new(),
            circuit_width: 0,
            circuit_size: 0,
            best_individual: Vec::new(),
            best_score: -1000000.0,
            best_circuit: None,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Internal helper to generate a random initial mapping.
    fn generate_random_initial_mapping(&mut self, circuit_width: usize) -> Vec<usize> {
        let mut valid_region_idxes: Vec<usize> = Vec::new();
        for (idx, region) in self.layout_regions.iter().enumerate() {
            if region.len() >= circuit_width {
                valid_region_idxes.push(idx);
            }
        }

        if valid_region_idxes.is_empty() {
            return Vec::new();
        }

        let random_region_idx =
            valid_region_idxes[self.rng.random_range(0..valid_region_idxes.len())];
        let region = &self.layout_regions[random_region_idx];

        let random_start_node_idx = self.rng.random_range(0..region.len());
        let start_q_id = region[random_start_node_idx];

        let mut random_indi: Vec<usize> = Vec::with_capacity(circuit_width);
        let mut period_neigh: Vec<usize> = vec![start_q_id];
        let mut visited: HashSet<usize> = HashSet::new();

        while random_indi.len() < circuit_width && !period_neigh.is_empty() {
            let random_pick_q = self.rng.random_range(0..period_neigh.len());
            let selected_q_id = period_neigh.remove(random_pick_q);

            if visited.contains(&selected_q_id) {
                continue;
            }

            visited.insert(selected_q_id);
            random_indi.push(selected_q_id);

            for &neighbor_id in &self.topology.neighbors[selected_q_id] {
                if !visited.contains(&neighbor_id) && !period_neigh.contains(&neighbor_id) {
                    period_neigh.push(neighbor_id);
                }
            }
        }

        random_indi
    }

    /// Internal helper to initialize the population with random mappings.
    fn initial(&mut self, circuit: &Circuit) -> Result<(), CompileError> {
        if !self.population_space.is_empty() {
            self.population_space.clear();
            self.population_score.clear();
            self.individual_layouts.clear();
        }

        self.circuit_size = circuit.operations().len() as usize;
        self.circuit_width = circuit.width();

        for _ in 0..self.config.population {
            let individual = self.generate_random_initial_mapping(self.circuit_width);
            if individual.len() != self.circuit_width {
                return Err(CompileError::TopologyTooSmall {
                    required: self.circuit_width,
                    available: self
                        .layout_regions
                        .iter()
                        .map(|x| x.len())
                        .max()
                        .unwrap_or(0),
                });
            }
            self.population_space.push(individual);
            self.individual_layouts.push(Vec::new());
            self.population_score.push(0.0);
        }
        
        Ok(())
    }

    /// Internal helper for updating the population via selection, crossover, and mutation.
    fn update(&mut self) {
        let mut next_popu: Vec<Vec<usize>> = Vec::with_capacity(self.config.population);

        for _ in 0..self.config.population {
            let select_id = self.selection();
            let mut selected_individual = self.population_space[select_id].clone();

            let cross_rate: f64 = self.rng.random();
            let muta_rate: f64 = self.rng.random();

            if cross_rate <= self.config.crossover_prob {
                selected_individual = self.crossover(selected_individual, select_id);
            }

            if muta_rate <= self.config.mutation_prob {
                selected_individual = self.mutation(selected_individual);
            }

            next_popu.push(selected_individual);
        }

        self.population_space = next_popu;
    }

    /// Executes the genetic algorithm mapping on the given circuit.
    pub fn execute(&mut self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        self.initial(circuit)?;
        
        for _ in 0..self.config.update_iters {
            // Parallel evaluation of the population
            let evaluated_population: Vec<(usize, Circuit, f64)> = (0..self
                .config
                .population)
                .into_par_iter()
                .map(|idx| {
                    let mut sabre_mapping = SabreMapping::new(
                        self.original_topology.clone(),
                        self.fidelity_map.clone(), // Fidelity map integration typically inherited or passed here
                        self.config.sabre_config.clone(),
                    )
                    .expect("Failed to initialize Sabre within GA");

                    // In a true integration, `execute_with_genetic_algorithm` or a seed layout param is used.
                    let mapped_result = sabre_mapping
                        .execute_with_genetic_algorithm(circuit, self.population_space[idx].clone())
                        .expect("Sabre routing failed during GA evaluation");
                    
                    (idx, mapped_result.0, mapped_result.1)
                })
                .collect();
            
            for (idx, mapped_circuit, mapped_fidelity) in evaluated_population {
                let mapped_info = self.post_mapping_analysis(&mapped_circuit);
                let mapped_score = self.calculate_fitness_function(mapped_info.0, mapped_fidelity);

                self.individual_layouts[idx] = mapped_info.1;
                self.population_score[idx] = mapped_score;

                if mapped_score > self.best_score {
                    self.best_circuit = Some(mapped_circuit.clone());
                    self.best_individual = self.population_space[idx].clone();
                    self.best_score = mapped_score;
                }
            }

            self.update();
        }

        self.get_best_result()
    }

    /// Internal helper for roulette wheel selection.
    fn selection(&mut self) -> usize {
        let sum_score: f64 = self.population_score.iter().sum();

        if sum_score == 0.0 {
            return self.rng.random_range(0..self.config.population);
        }

        let mut select_prob: f64 = self.rng.random();

        for i in 0..self.config.population {
            let regular_score = self.population_score[i] / sum_score;
            if select_prob <= regular_score {
                return i;
            }
            select_prob -= regular_score;
        }

        self.config.population.saturating_sub(1)
    }

    /// Internal helper for crossover operation.
    fn crossover(&mut self, individual: Vec<usize>, related_id: usize) -> Vec<usize> {
        let mapped_info = &self.individual_layouts[related_id];
        if mapped_info.is_empty() {
            return individual;
        }

        let mut connected_qubits: Vec<Vec<usize>> = Vec::new();
        let mut swap_count: Vec<usize> = Vec::new();

        for (u, v, s_n) in mapped_info {
            let mut is_connected = false;
            let mut cq_index = 0;

            for cq in &mut connected_qubits {
                if cq.len() == self.config.crossover_qubit_number {
                    cq_index += 1;
                    continue;
                }

                if cq.contains(u) || cq.contains(v) {
                    swap_count[cq_index] += s_n;
                    is_connected = true;
                    if !cq.contains(u) {
                        cq.push(*u);
                    } else if !cq.contains(v) {
                        cq.push(*v);
                    }
                }
                cq_index += 1;
            }

            if !is_connected {
                connected_qubits.push(vec![*u, *v]);
                swap_count.push(*s_n);
            }
        }

        if swap_count.is_empty() {
            return individual;
        }

        let mut cross_indi = individual.clone();
        let max_index = swap_count
            .into_iter()
            .enumerate()
            .max_by_key(|(_, x)| *x)
            .unwrap();

        let mut cross_q = connected_qubits[max_index.0].clone();
        cross_q.shuffle(&mut self.rng);

        let mut cross_idx = 0;
        for (ind_index, ind) in individual.iter().enumerate() {
            if cross_q.contains(ind) {
                if let Some(&new_q) = cross_q.get(cross_idx) {
                    cross_indi[ind_index] = new_q;
                    cross_idx += 1;
                }
            }
        }

        cross_indi
    }

    /// Internal helper for single-point mutation, ensuring subgraph connectivity is preserved.
    fn mutation(&mut self, individual: Vec<usize>) -> Vec<usize> {
        // Record all valid mutation candidate actions.
        // Tuple format: (index of the qubit to be replaced, new physical qubit ID).
        let mut mutation_candidates: Vec<(usize, usize)> = Vec::new();

        for (idx, &curr_indi) in individual.iter().enumerate() {
            let neighbors = &self.topology.neighbors[curr_indi];

            // Filter out dead-end nodes (degree <= 1).
            if neighbors.len() <= 1 {
                continue;
            }

            // Find the current node's connections within the mapped subgraph.
            let inner_points: Vec<usize> = neighbors
                .iter()
                .filter(|n| individual.contains(n))
                .copied()
                .collect();

            // Allow mutation only if it's a "leaf node" (connected to the subgraph via a single anchor point).
            if inner_points.len() == 1 {
                let inner_node = inner_points[0];
                let muta_neighbors = &self.topology.neighbors[inner_node];

                // Look for free physical qubits around the anchor.
                for &candidate_node in muta_neighbors {
                    if !individual.contains(&candidate_node) {
                        mutation_candidates.push((idx, candidate_node));
                    }
                }
            }
        }

        // If no nodes can be mutated
        if mutation_candidates.is_empty() {
            let force_muta_rate: f64 = self.rng.random();
            if force_muta_rate < self.config.forced_mutation_prob {
                return self.generate_random_initial_mapping(self.circuit_width);
            }
            // Otherwise, return the original individual.
            return individual;
        }

        // Core fix: Randomly pick exactly ONE action from all valid mutation candidates to execute.
        let pick = self.rng.random_range(0..mutation_candidates.len());
        let (mutate_idx, new_node) = mutation_candidates[pick];

        // Clone the old mapping and modify only the selected qubit.
        let mut muta_indi = individual.clone();
        muta_indi[mutate_idx] = new_node;

        muta_indi
    }

    /// Internal helper to extract the best mapping result.
    fn get_best_result(&self) -> Result<Circuit, CompileError> {
        self.best_circuit
            .clone()
            .ok_or_else(|| CompileError::Internal("No valid GA mapping found".into()))
    }

    /// Internal helper to analyze mapping results for routing cost (SWAP count).
    fn post_mapping_analysis(
        &self,
        swaped_circuit: &Circuit,
    ) -> (usize, Vec<(usize, usize, usize)>) {
        let mut swap_counter = 0;
        let mut edge_list_counting: HashMap<(usize, usize), usize> = HashMap::new();

        for op in swaped_circuit.operations() {
            if let Instruction::Standard(StandardGate::SWAP) = op.instruction {
                swap_counter += 1;
                let gate_qubits = &op.qubits;
                if gate_qubits.len() == 2 {
                    let u = self.topology.qubit_to_index[&gate_qubits[0]];
                    let v = self.topology.qubit_to_index[&gate_qubits[1]];

                    let order_q = (u, v);
                    let inv_order_q = (v, u);

                    if let Some(count) = edge_list_counting.get_mut(&order_q) {
                        *count += 1;
                    } else if let Some(count) = edge_list_counting.get_mut(&inv_order_q) {
                        *count += 1;
                    } else {
                        edge_list_counting.insert(order_q, 1);
                    }
                }
            }
        }

        let mut swaped_layout: Vec<(usize, usize, usize)> = edge_list_counting
            .into_iter()
            .map(|((u, v), count)| (u, v, count))
            .collect();
        swaped_layout.sort_by(|a, b| a.2.cmp(&b.2));

        (swap_counter, swaped_layout)
    }

    /// Internal helper for fitness calculation.
    fn calculate_fitness_function(&self, swap_counts: usize, mapped_fidelity: f64) -> f64 {
        if self.is_ideal_topology {
            1.0 - (swap_counts as f64 / (swap_counts + self.circuit_size) as f64)
        } else {
            mapped_fidelity
        }
    }

    /// Finds all connected components in the topology using BFS.
    /// Returns a vector of connected regions, where each region is a vector of qubit IDs
    fn get_all_connected_nodes(
        topology: &TopologyAdapter,
        invalid_qubits: &HashSet<usize>,
    ) -> Vec<Vec<usize>> {
        let mut all_regions: Vec<Vec<usize>> = Vec::new();
        let qubit_number = topology.num_qubits();
        if invalid_qubits.is_empty() {
            all_regions.push((0..qubit_number).collect::<Vec<usize>>());
            return all_regions;
        }

        // Topology Analysis
        let mut valid_nodes: Vec<usize> = Vec::new();
        for idx in 0..qubit_number {
            if !invalid_qubits.contains(&idx) {
                valid_nodes.push(idx);
            }
        }

        while !valid_nodes.is_empty() {
            let mut cnodes: Vec<usize> = Vec::new();
            let start_node: usize = valid_nodes[0];

            // Use a queue for BFS and track visited nodes for this specific region
            let mut queue: Vec<usize> = vec![start_node];
            let mut visited_in_component: HashSet<usize> = HashSet::new();
            visited_in_component.insert(start_node);

            while !queue.is_empty() {
                let mut next_queue: Vec<usize> = Vec::new();

                for node in queue {
                    cnodes.push(node);

                    for &neighbor in &topology.neighbors[node] {
                        // Only process if it's valid AND hasn't been visited yet
                        if !invalid_qubits.contains(&neighbor)
                            && !visited_in_component.contains(&neighbor)
                        {
                            visited_in_component.insert(neighbor);
                            next_queue.push(neighbor);
                        }
                    }
                }
                queue = next_queue;
            }

            // Efficiently remove all found nodes from valid_nodes
            valid_nodes.retain(|x| !visited_in_component.contains(x));
            all_regions.push(cnodes);
        }

        all_regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Qubit};
    use std::collections::HashSet;

    // auxiliary functions：create a line topology
    fn line_topology(ids: &[u32]) -> Topology {
        let qubits: Vec<Qubit> = ids.iter().copied().map(Qubit::new).collect();
        let couplings = ids
            .windows(2)
            .map(|w| (Qubit::new(w[0]), Qubit::new(w[1]), "CX".to_string()))
            .collect();
        Topology::new(qubits, couplings).unwrap()
    }

    // auxiliary functions：create a circuit
    fn simple_circuit() -> Circuit {
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
        circuit
    }

    #[test]
    fn test_ga_initialization_with_empty_invalid_qubits() {
        let topology = line_topology(&[0, 1, 2, 3]);
        let config = GaConfig {
            population: 5,
            ..GaConfig::default()
        };

        let mut ga = GeneticAlgMapping::new(topology, config, None, None).unwrap();
        let circuit = simple_circuit();

        ga.initial(&circuit).unwrap();

        assert_eq!(ga.population_space.len(), 5);
        assert_eq!(ga.population_space[0].len(), 3);
    }

    #[test]
    fn test_connected_components_with_invalid_qubits() {
        let topology = line_topology(&[0, 1, 2, 3, 4]);

        let mut invalid_qubits = HashSet::new();
        invalid_qubits.insert(2);

        let config = GaConfig::default();
        let ga = GeneticAlgMapping::new(topology, config, None, Some(invalid_qubits)).unwrap();

        assert_eq!(ga.layout_regions.len(), 2);

        let has_region_0_1 = ga
            .layout_regions
            .iter()
            .any(|r| r.contains(&0) && r.contains(&1));
        let has_region_3_4 = ga
            .layout_regions
            .iter()
            .any(|r| r.contains(&3) && r.contains(&4));
        let contains_invalid = ga.layout_regions.iter().any(|r| r.contains(&2));

        assert!(has_region_0_1, "Should contain region with 0 and 1");
        assert!(has_region_3_4, "Should contain region with 3 and 4");
        assert!(!contains_invalid, "Should NOT contain the invalid qubit 2");
    }

    #[test]
    fn test_topology_too_small_error() {
        // Topology has only 2 qubits
        let topology = line_topology(&[0, 1]);
        let config = GaConfig::default();
        let mut ga = GeneticAlgMapping::new(topology, config, None, None).unwrap();

        // Circuit needs 3 qubits
        let circuit = simple_circuit();

        // initial() should return TopologyTooSmall error
        let result = ga.initial(&circuit);
        assert!(
            matches!(result, Err(CompileError::TopologyTooSmall { .. })),
            "Expected TopologyTooSmall error, got {:?}",
            result
        );
    }

    #[test]
    fn test_mutation_preserves_length() {
        let topology = line_topology(&[0, 1, 2, 3, 4]);
        let config = GaConfig::default();
        let mut ga = GeneticAlgMapping::new(topology, config, None, None).unwrap();

        ga.circuit_width = 3;
        let individual = vec![0, 1, 2];

        // Run mutation
        let mutated = ga.mutation(individual.clone());

        // The mutated individual should maintain the exact same length (circuit width)
        assert_eq!(mutated.len(), 3);
        // Ensure all elements in the mutated individual are unique (no duplicated qubits)
        let unique_qubits: HashSet<usize> = mutated.into_iter().collect();
        assert_eq!(unique_qubits.len(), 3);
    }

    #[test]
    fn test_post_mapping_analysis_swap_counting() {
        let topology = line_topology(&[0, 1, 2]);
        let ga = GeneticAlgMapping::new(topology, GaConfig::default(), None, None).unwrap();

        let mut circuit = Circuit::new(3);
        // Add some SWAPs manually to simulate a routed circuit
        circuit
            .append(
                Instruction::Standard(StandardGate::SWAP),
                vec![Qubit::new(0), Qubit::new(1)],
                vec![],
                None,
            )
            .unwrap();
        circuit
            .append(
                Instruction::Standard(StandardGate::SWAP),
                vec![Qubit::new(1), Qubit::new(2)],
                vec![],
                None,
            )
            .unwrap();
        circuit
            .append(
                Instruction::Standard(StandardGate::SWAP),
                vec![Qubit::new(0), Qubit::new(1)], // Second SWAP on the (0,1) edge
                vec![],
                None,
            )
            .unwrap();

        let (swap_count, layout) = ga.post_mapping_analysis(&circuit);

        // Total SWAP count should be 3
        assert_eq!(swap_count, 3);

        // Layout should contain two edges, sorted by count ascending: (1, 2) has 1, (0, 1) has 2
        assert_eq!(layout.len(), 2);

        // Ensure they are normalized (u < v) and sorted by usage frequency
        assert_eq!(layout[0], (1, 2, 1));
        assert_eq!(layout[1], (0, 1, 2));
    }

    #[test]
    fn test_execute_end_to_end() {
        // Create a 5-qubit line topology
        let topology = line_topology(&[0, 1, 2, 3, 4]);

        // Create a circuit that intentionally requires routing
        // A CX between 0 and 2 on a line topology requires a SWAP
        let mut circuit = Circuit::new(3);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();


        let mut fidelity = FidelityMap::new();
        fidelity.insert((Qubit::new(0), Qubit::new(1)), 0.5);
        fidelity.insert((Qubit::new(1), Qubit::new(2)), 0.99);
        fidelity.insert((Qubit::new(2), Qubit::new(3)), 0.99);
        fidelity.insert((Qubit::new(3), Qubit::new(4)), 0.5);

        // Configure GA and Sabre for a fast, deterministic test
        let mut sabre_config = SabreConfig::default();
        sabre_config.repeat_iterations = 0; // Pure GA mode (no SABRE backward optimization)
        sabre_config.seed = 42;

        let config = GaConfig {
            population: 10,
            update_iters: 2,
            seed: 42,
            sabre_config,
            ..GaConfig::default()
        };

        let mut ga = GeneticAlgMapping::new(topology, config, Some(fidelity), None).unwrap();

        // Execute the GA mapping
        let mapped_circuit = ga.execute(&circuit).expect("GA mapping failed to execute");
        // The mapped circuit should have at least the same number of gates as the original
        assert!(mapped_circuit.operations().len() >= circuit.operations().len());

        // Verify that all 2Q gates in the resulting circuit are physically adjacent
        for op in mapped_circuit.operations() {
            if op.qubits.len() == 2 {
                let u_id = op.qubits[0].id();
                let v_id = op.qubits[1].id();

                // On a strictly sequential line topology [0, 1, 2, 3, 4],
                // physical adjacency means the difference in IDs is exactly 1.
                let distance = (u_id as i32 - v_id as i32).abs();
                assert_eq!(
                    distance, 1,
                    "Routing failed: found a 2Q gate between unconnected physical qubits {} and {}",
                    u_id, v_id
                );
            }
        }
    }
}
