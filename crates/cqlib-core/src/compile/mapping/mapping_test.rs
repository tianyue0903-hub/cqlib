use super::*;
use crate::circuit::gate::control_flow::ConditionView;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, Operation, Qubit};
use crate::compile::error::CompileError;
use smallvec::smallvec;
use std::collections::HashSet;

fn connected_undirected(topology: &Topology, a: Qubit, b: Qubit) -> bool {
    topology.is_connected(a, b) || topology.is_connected(b, a)
}

fn assert_mapped_2q_edges(mapped: &Circuit, topology: &Topology) {
    for op in mapped.operations() {
        if op.qubits.len() == 2 {
            assert!(
                connected_undirected(topology, op.qubits[0], op.qubits[1]),
                "2q op is not on a topology edge: {:?}",
                op.qubits
            );
        }
    }
}

fn count_swaps(circuit: &Circuit) -> usize {
    circuit
        .operations()
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Standard(StandardGate::SWAP)))
        .count()
}

fn fingerprint(circuit: &Circuit) -> Vec<String> {
    circuit
        .operations()
        .iter()
        .map(|op| {
            let mut qids: Vec<u32> = op.qubits.iter().map(Qubit::id).collect();
            qids.sort_unstable();
            format!("{:?}:{:?}", op.instruction, qids)
        })
        .collect()
}

#[test]
fn test_module_exports_compile_and_device() {
    let _cfg = crate::compile::SabreConfig::default();
    let _topology = crate::device::Topology::new(vec![Qubit::new(0)], vec![]);
}

#[test]
fn test_reject_control_flow() {
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    circuit.measure(q0).unwrap();

    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    circuit
        .if_else(ConditionView::new(q0, 1), true_body, None)
        .unwrap();

    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let err = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
    assert!(matches!(err, CompileError::UnsupportedControlFlow));
}

#[test]
fn test_reject_directive_and_delay() {
    let mut circuit = Circuit::new(1);
    circuit.barrier(vec![Qubit::new(0)]).unwrap();
    let topology = Topology::line(vec![0.into(), 1.into()]);
    let err = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
    assert!(matches!(
        err,
        CompileError::UnsupportedInstruction {
            instruction: _,
            op_index: _
        }
    ));
}

#[test]
fn test_reject_unsupported_arity() {
    let mut circuit = Circuit::new(3);
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();

    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let err = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
    assert!(matches!(
        err,
        CompileError::UnsupportedArity {
            arity: 3,
            op_index: 0
        }
    ));
}

#[test]
fn test_invalid_fidelity_rejected() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(0), Qubit::new(1)), 1.2);
    let err = Vf2Mapping::new(topology, Some(fidelity)).unwrap_err();
    assert!(matches!(err, CompileError::InvalidFidelity { .. }));
}

#[test]
fn test_missing_fidelity_defaults_to_one() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();

    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(0), Qubit::new(1)), 0.2);

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::Disabled,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, Some(&fidelity), &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_fidelity_pair_normalization() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(2), Qubit::new(1)), 0.9);
    let _ = SabreMapping::new(topology, Some(fidelity), SabreConfig::default()).unwrap();
}

#[test]
fn test_vf2_fast_path_no_overhead() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
    circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert_eq!(mapped.operations().len(), circuit.operations().len());
    assert_eq!(count_swaps(&mapped), 0);
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_vf2_standalone_initial_layout_api() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
    circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let layout = vf2.find_initial_layout(&circuit).unwrap().unwrap();
    assert_eq!(layout.len(), 3);
}

#[test]
fn test_vf2_find_initial_layout_fallback_top1() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

    let layout = vf2.find_initial_layout(&circuit).unwrap();
    assert!(layout.is_some());
    assert_eq!(layout.unwrap().len(), 3);
}

#[test]
fn test_vf2_map_remains_strict_no_fallback() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let mut vf2 = Vf2Mapping::new(topology, None).unwrap();
    let err = vf2.execute(&circuit).unwrap_err();
    assert!(matches!(err, CompileError::Vf2NoMapping));
}

#[test]
fn test_vf2_candidates_topk_and_score_range() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.x(Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 3,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(!candidates.is_empty());
    assert!(candidates.len() <= 3);
    for c in candidates {
        assert_eq!(c.logic2phy.len(), 3);
        assert_eq!(c.region.len(), 3);
        assert!((0.0..=1.0).contains(&c.score.total));
        assert!((0.0..=1.0).contains(&c.score.fidelity));
        assert!((0.0..=1.0).contains(&c.score.topology_fit));
        assert!((0.0..=1.0).contains(&c.score.gate_distribution));
    }
}

#[test]
fn test_vf2_candidates_deterministic_order() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 5,
        ..Vf2CandidateOptions::default()
    };
    let c1 = vf2
        .find_initial_layout_candidates(&circuit, Some(options.clone()))
        .unwrap();
    let c2 = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();

    let l1: Vec<Vec<u32>> = c1
        .iter()
        .map(|c| c.logic2phy.iter().map(Qubit::id).collect())
        .collect();
    let l2: Vec<Vec<u32>> = c2
        .iter()
        .map(|c| c.logic2phy.iter().map(Qubit::id).collect())
        .collect();
    assert_eq!(l1, l2);
}

#[test]
fn test_vf2_candidates_topk_zero() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 0,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.is_empty());
}

#[test]
fn test_vf2_candidates_topk_effective_when_strict_isomorphic() {
    let topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(3), Qubit::new(0), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 4,
        max_matches_per_subgraph: 16,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.len() > 1);
    assert!(candidates.len() <= 4);
}

#[test]
fn test_vf2_candidates_respect_max_matches_per_subgraph() {
    let topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(3), Qubit::new(0), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 8,
        max_matches_per_subgraph: 1,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.len() <= 1);
}

#[test]
fn test_vf2_find_initial_layout_fallback_none_when_no_candidate() {
    let topology = Topology::new(vec![Qubit::new(0), Qubit::new(1)], vec![]).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());
    let layout = vf2.find_initial_layout(&circuit).unwrap();
    assert!(layout.is_none());
}

#[test]
fn test_vf2_isomorphic_on_dense_topology_non_induced_case() {
    let topology = Topology::new(
        vec![
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
        ],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(0), Qubit::new(2), "CX".to_string()),
            (Qubit::new(0), Qubit::new(3), "CX".to_string()),
            (Qubit::new(0), Qubit::new(4), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(1), Qubit::new(3), "CX".to_string()),
            (Qubit::new(1), Qubit::new(4), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(2), Qubit::new(4), "CX".to_string()),
            (Qubit::new(3), Qubit::new(4), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(5);
    circuit.cx(Qubit::new(2), Qubit::new(4)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(4)).unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(4), Qubit::new(3)).unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(3)).unwrap();

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    assert!(vf2.is_subgraph_isomorphic(&circuit).unwrap());
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_policy_initial_only_routes_with_sabre() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        seed: 12345,
        initial_iterations: 2,
        repeat_iterations: 1,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_sabre_fallback_and_state_exposure() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert!(mapped.operations().len() > circuit.operations().len());
    assert_mapped_2q_edges(&mapped, &topology);

    let mut sabre = SabreMapping::new(topology, None, SabreConfig::default()).unwrap();
    let _ = sabre.execute(&circuit).unwrap();
    assert_eq!(sabre.logic2phy.len(), circuit.qubits().len());
}

#[test]
fn test_output_uses_only_physical_qubits_in_use() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into(), 4.into()]);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert_eq!(mapped.qubits().len(), 2);
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_sabre_determinism_with_fixed_seed() {
    let topology = Topology::line(vec![0.into(), 1.into(), 2.into(), 3.into()]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        seed: 12345,
        initial_iterations: 3,
        repeat_iterations: 2,
        swap_iterations: 3,
        ..SabreConfig::default()
    };

    let mut sabre1 = SabreMapping::new(topology.clone(), None, cfg.clone()).unwrap();
    let mut sabre2 = SabreMapping::new(topology, None, cfg).unwrap();

    let out1 = sabre1.execute(&circuit).unwrap();
    let out2 = sabre2.execute(&circuit).unwrap();
    assert_eq!(fingerprint(&out1), fingerprint(&out2));
}

#[test]
fn test_non_contiguous_qubit_ids_supported() {
    let topology = Topology::line(vec![100.into(), 200.into(), 300.into(), 400.into()]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(30), Qubit::new(70)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(30)).unwrap();
    circuit.cx(Qubit::new(30), Qubit::new(70)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();

    let topo_set: HashSet<Qubit> = topology.qubits().collect();
    for q in mapped.qubits() {
        assert!(topo_set.contains(&q));
    }
    assert_mapped_2q_edges(&mapped, &topology);
}
