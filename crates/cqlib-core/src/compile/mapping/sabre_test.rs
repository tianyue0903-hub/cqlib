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

use super::*;
use std::collections::HashSet;

fn line_topology(ids: &[u32]) -> Topology {
    let qubits: Vec<Qubit> = ids.iter().copied().map(Qubit::new).collect();
    let couplings = ids
        .windows(2)
        .map(|w| (Qubit::new(w[0]), Qubit::new(w[1]), "CX".to_string()))
        .collect();
    Topology::new(qubits, couplings).unwrap()
}

fn star_topology() -> Topology {
    Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(0), Qubit::new(2), "CX".to_string()),
        ],
    )
    .unwrap()
}

fn square_topology() -> Topology {
    Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(3), Qubit::new(0), "CX".to_string()),
        ],
    )
    .unwrap()
}

fn triangle_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit
}

fn single_cx_circuit() -> Circuit {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit
}

fn star_fidelity_map() -> FidelityMap {
    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(0), Qubit::new(1)), 0.2);
    fidelity.insert((Qubit::new(0), Qubit::new(2)), 0.9);
    fidelity
}

fn build_star_state(
    mapper: &SabreMapping,
    info: &GateDependencyDag,
    initial_mapping: &[usize],
) -> RoutingState {
    let mut phy2logic = vec![None; mapper.topology.num_qubits()];
    for (logical, &physical) in initial_mapping.iter().enumerate() {
        phy2logic[physical] = Some(logical);
    }

    RoutingState {
        logic2phy: initial_mapping.to_vec(),
        phy2logic,
        pre_number: info.indegree.clone(),
        front_layer: info.front_layer.clone(),
        ans_steps: Vec::new(),
        decay: vec![1.0; mapper.topology.num_qubits()],
        decay_time: 0,
        weight_gates: vec![Vec::new(); mapper.topology.num_qubits()],
        preprocessing_h: 0.0,
    }
}

fn count_swap_ops(ops: &[Operation]) -> usize {
    ops.iter()
        .filter(|op| matches!(&op.instruction, Instruction::Standard(StandardGate::SWAP)))
        .count()
}

#[test]
fn test_initial_layout_candidates_use_ranked_vf2_seed_first() {
    let topology = line_topology(&[0, 1, 2]);
    let circuit = triangle_circuit();
    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        vf2_seed_top_k: 3,
        initial_iterations: 3,
        seed: 7,
        ..SabreConfig::default()
    };

    let mut sabre = SabreMapping::new(topology, None, cfg.clone()).unwrap();
    let prepared = preprocess_circuit(&circuit).unwrap();
    let available_nodes = sabre.usable_nodes();
    let layouts = sabre
        .initial_layout_candidates(
            &prepared,
            &available_nodes,
            prepared.logical_qubits.len(),
            cfg.initial_iterations,
        )
        .unwrap();

    let vf2 = Vf2Mapping::from_adapter(sabre.topology.clone());
    let options = Vf2CandidateOptions {
        top_k: cfg.vf2_seed_top_k,
        weights: cfg.vf2_seed_weights,
        ..Vf2CandidateOptions::default()
    };
    let available_set: HashSet<usize> = available_nodes.iter().copied().collect();
    let mut expected = Vec::new();
    let mut seen = HashSet::new();
    for layout in vf2
        .find_prepared_layout_candidate_indices(&prepared, Some(options))
        .unwrap()
    {
        if layout.iter().all(|phy| available_set.contains(phy)) && seen.insert(layout.clone()) {
            expected.push(layout);
        }
    }

    assert!(!expected.is_empty());
    assert_eq!(layouts[0], expected[0]);
}

#[test]
fn test_best_prepared_layouts_use_bounded_top_k_for_structured_candidates() {
    let topology = square_topology();
    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        vf2_seed_top_k: 4,
        ..SabreConfig::default()
    };

    let mapper = SabreMapping::new(topology, None, cfg).unwrap();
    let prepared = preprocess_circuit(&single_cx_circuit()).unwrap();
    let states = mapper.best_prepared_layouts(&prepared).unwrap();

    assert!(states.len() > 1);
    assert!(states.len() <= 4);
}

#[test]
fn test_continuation_entry_states_preserve_explicit_seed_before_vf2_candidates() {
    let topology = square_topology();
    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        vf2_seed_top_k: 4,
        ..SabreConfig::default()
    };

    let mapper = SabreMapping::new(topology, None, cfg).unwrap();
    let program = preprocess_program(&single_cx_circuit()).unwrap();
    let explicit = StructuredLayoutState::new(&[3, 0]);
    let states = mapper
        .continuation_entry_states(&program, &program.items, vec![explicit.clone()])
        .unwrap();

    assert_eq!(states.first(), Some(&explicit));
    assert!(states.len() > 1);
}

#[test]
fn test_merge_branch_routes_uses_worst_case_cost_and_preserves_empty_false_body() {
    let topology = line_topology(&[0, 1, 2]);
    let mapper = SabreMapping::new(topology, None, SabreConfig::default()).unwrap();
    let true_route = StructuredRoute {
        exit_l2p: vec![0, 2],
        ops: Vec::new(),
        cost: 5,
        log_fidelity: -1.0,
        objective: 0.0,
    };
    let false_route = StructuredRoute {
        exit_l2p: vec![0, 1],
        ops: Vec::new(),
        cost: 2,
        log_fidelity: -0.25,
        objective: 0.0,
    };

    let merged = mapper
        .merge_branch_routes(
            &true_route,
            &false_route,
            &StructuredLayoutState::new(&[0, 1]),
            true,
        )
        .unwrap();

    assert_eq!(count_swap_ops(&merged.true_body), 1);
    assert!(matches!(merged.false_body.as_ref(), Some(body) if body.is_empty()));
    assert_eq!(merged.cost, 8);
    assert_eq!(merged.log_fidelity, -1.0);
}

#[test]
fn test_close_loop_body_reconciles_back_to_loop_layout() {
    let topology = line_topology(&[0, 1, 2]);
    let mapper = SabreMapping::new(topology, None, SabreConfig::default()).unwrap();
    let body_route = StructuredRoute {
        exit_l2p: vec![0, 2],
        ops: Vec::new(),
        cost: 4,
        log_fidelity: -0.5,
        objective: 0.0,
    };

    let closed = mapper
        .close_loop_body(&body_route, &StructuredLayoutState::new(&[0, 1]))
        .unwrap();

    assert_eq!(count_swap_ops(&closed.body_ops), 1);
    assert_eq!(closed.cost, 7);
    assert_eq!(closed.log_fidelity, -0.5);
}

#[test]
fn test_local_swap_scoring_prefers_higher_fidelity_when_weighted() {
    let topology = star_topology();
    let fidelity = star_fidelity_map();
    let circuit = single_cx_circuit();
    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::Disabled,
        swap_fidelity_weight: 1.0,
        predicted_fidelity_weight: 0.0,
        seed: 1,
        ..SabreConfig::default()
    };

    let mapper = SabreMapping::new(topology, Some(fidelity), cfg).unwrap();
    let prepared = preprocess_circuit(&circuit).unwrap();
    let info = mapper
        .build_circuit_info(&prepared, prepared.logical_qubits.len())
        .unwrap();
    let mut state = build_star_state(&mapper, &info, &[1, 2]);

    let swaps = mapper.obtain_swaps(&info, &mut state);
    let mut low_score = None;
    let mut high_score = None;
    for swap in swaps {
        let pair = normalize_index_pair(swap.u, swap.v);
        if pair == (0, 1) {
            low_score = Some(swap.score);
        } else if pair == (0, 2) {
            high_score = Some(swap.score);
        }
    }

    let low_score = low_score.expect("missing (0,1) swap candidate");
    let high_score = high_score.expect("missing (0,2) swap candidate");
    assert!(high_score < low_score);
}

#[test]
fn test_equal_cost_routes_prefer_higher_predicted_fidelity() {
    let topology = star_topology();
    let fidelity = star_fidelity_map();
    let circuit = single_cx_circuit();
    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::Disabled,
        swap_fidelity_weight: 0.0,
        predicted_fidelity_weight: 1.0,
        swap_iterations: 32,
        seed: 11,
        ..SabreConfig::default()
    };

    let mut mapper = SabreMapping::new(topology, Some(fidelity), cfg).unwrap();
    let prepared = preprocess_circuit(&circuit).unwrap();
    let info = mapper
        .build_circuit_info(&prepared, prepared.logical_qubits.len())
        .unwrap();
    let group = mapper
        .execute_routing(&info, &prepared, &[1, 2], 32)
        .unwrap();

    assert_eq!(group.cost, 4);
    match group.steps.first() {
        Some(AnsStep::Swap { u, v }) => {
            assert_eq!(normalize_index_pair(*u, *v), (0, 2));
        }
        _ => panic!("expected first routed step to be a SWAP"),
    }
}

#[test]
fn test_objective_weights_adjust_cost_vs_fidelity_preference() {
    let topology = star_topology();
    let fidelity = star_fidelity_map();

    let low_cost_cfg = SabreConfig {
        gate_cost_weight: 1.0,
        predicted_fidelity_weight: 0.1,
        ..SabreConfig::default()
    };
    let fidelity_cfg = SabreConfig {
        gate_cost_weight: 0.1,
        predicted_fidelity_weight: 1.0,
        ..SabreConfig::default()
    };

    let low_cost_mapper =
        SabreMapping::new(topology.clone(), Some(fidelity.clone()), low_cost_cfg).unwrap();
    let fidelity_mapper = SabreMapping::new(topology, Some(fidelity), fidelity_cfg).unwrap();

    let mut candidate_a = AnsGroup {
        initial_l2p: vec![0, 1],
        final_l2p: vec![0, 1],
        steps: vec![],
        cost: 3,
        log_fidelity: -8.0,
        objective: 0.0,
    };
    let mut candidate_b = AnsGroup {
        initial_l2p: vec![0, 2],
        final_l2p: vec![0, 2],
        steps: vec![],
        cost: 4,
        log_fidelity: -1.0,
        objective: 0.0,
    };

    candidate_a.objective =
        low_cost_mapper.routing_objective(candidate_a.cost, candidate_a.log_fidelity);
    candidate_b.objective =
        low_cost_mapper.routing_objective(candidate_b.cost, candidate_b.log_fidelity);
    assert!(low_cost_mapper.group_better(&candidate_a, &candidate_b));

    candidate_a.objective =
        fidelity_mapper.routing_objective(candidate_a.cost, candidate_a.log_fidelity);
    candidate_b.objective =
        fidelity_mapper.routing_objective(candidate_b.cost, candidate_b.log_fidelity);
    assert!(fidelity_mapper.group_better(&candidate_b, &candidate_a));
}
