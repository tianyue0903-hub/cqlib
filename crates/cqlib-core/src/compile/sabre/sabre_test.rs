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

use super::*;
use crate::circuit::{
    Circuit, CircuitParam, ConditionView, ControlFlow, Instruction, Operation, Parameter,
    ParameterValue, Qubit, StandardGate,
};
use crate::compile::CompilerError;
use crate::compile::transform::layout::LayoutObjective;
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit, Topology};
use smallvec::smallvec;
use std::collections::HashSet;

#[test]
fn deterministic_seeded_config_uses_compact_reproducible_settings() {
    let config = SabreConfig::deterministic_seeded(7);

    assert_eq!(config.layout_trials, 2);
    assert_eq!(config.refinement_iterations, 1);
    assert_eq!(config.layout_scoring_trials, 1);
    assert_eq!(config.routing_trials, 1);
    assert_eq!(config.trial_objective, SabreTrialObjective::SwapThenDepth);
    assert_eq!(config.seed, Some(7));
    assert_eq!(config.heuristic.lookahead_weights, vec![0.5]);
    assert_eq!(config.heuristic.attempt_limit, 20);
}

#[test]
fn validate_config_reports_invalid_trial_counts() {
    let config = SabreConfig {
        routing_trials: 0,
        ..SabreConfig::deterministic_seeded(7)
    };

    let error = validate_config(&config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("routing_trials"))
    );
}

#[test]
fn normalize_initial_layout_public_api_uses_device_usable_qubits() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 2), (1, 0)], 3).unwrap();

    let normalized = normalize_initial_layout(
        &[LogicalQubit::new(0), LogicalQubit::new(1)],
        &device,
        &layout,
    )
    .unwrap();

    assert_eq!(normalized.num_logical(), 2);
    assert_eq!(normalized.num_physical(), 3);
    assert_eq!(normalized.num_vacant_physical(), 1);
    assert_eq!(
        normalized.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(2))
    );
}

#[test]
fn validate_reachable_interactions_public_api_reports_disconnected_pairs() {
    let p0 = PhysicalQubit::new(0);
    let p1 = PhysicalQubit::new(1);
    let topology = Topology::new(vec![p0, p1], vec![]).unwrap();
    let device = Device::new("disconnected", HashSet::from([p0, p1]), topology).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = validate_reachable_interactions(&circuit, &device, &layout).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::InvalidInput(message) if message.contains("disconnected")
    ));
}

#[test]
fn route_keeps_adjacent_two_qubit_gate_without_swap() {
    let device = Device::line("line", 2).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.swap_count, 0);
    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert_eq!(
        result.final_layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(0))
    );
}

#[test]
fn route_inserts_swap_on_line_topology() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.swap_count, 1);
    assert_eq!(result.circuit.operations().len(), 2);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::SWAP)
    ));
    let gate_qubits = &result.circuit.operations()[1].qubits;
    assert_eq!(gate_qubits.len(), 2);
    assert!(
        are_adjacent(gate_qubits[0], gate_qubits[1]),
        "routed two-qubit operation must be adjacent"
    );
}

#[test]
fn route_does_not_fold_overlapping_two_qubit_gates() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1), (2, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert!(result.swap_count > 0);
    assert_all_two_qubit_operations_are_adjacent_on_line(&result.circuit);
}

#[test]
fn route_may_fold_consecutive_two_qubit_gates_on_same_pair() {
    let device = Device::line("line", 2).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.swap_count, 0);
    assert_eq!(result.circuit.operations().len(), 2);
    assert_all_two_qubit_operations_are_adjacent_on_line(&result.circuit);
}

#[test]
fn refine_layout_is_reproducible_for_same_seed() {
    let device = Device::line("line", 4).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();

    let first = sabre_refine_layout(&circuit, &device, None, &objective, &config).unwrap();
    let second = sabre_refine_layout(&circuit, &device, None, &objective, &config).unwrap();

    assert_eq!(first.layout.l2p_map(), second.layout.l2p_map());
    assert_eq!(
        first.score.as_ref().map(|score| score.total),
        second.score.as_ref().map(|score| score.total)
    );
}

#[test]
fn route_with_decay_is_reproducible_for_same_seed() {
    let device = Device::line("line", 5).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 4), (2, 2)], 5).unwrap();
    let config = SabreConfig {
        routing_trials: 4,
        seed: Some(23),
        heuristic: SabreHeuristicConfig {
            decay_increment: Some(0.05),
            decay_reset: 2,
            lookahead_weights: vec![0.5, 0.25],
            attempt_limit: 20,
            ..SabreHeuristicConfig::default()
        },
        ..SabreConfig::deterministic_seeded(7)
    };
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let first = sabre_route(&circuit, &device, &layout, &config).unwrap();
    let second = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(first.swap_count, second.swap_count);
    assert_eq!(first.final_layout.l2p_map(), second.final_layout.l2p_map());
    assert_eq!(
        first.diagnostics.selected_trial_index,
        second.diagnostics.selected_trial_index
    );
    assert_eq!(
        first.diagnostics.operation_count,
        second.diagnostics.operation_count
    );
}

#[test]
fn control_flow_body_is_routed_and_restored() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), true_body, None)
        .unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.final_layout.l2p_map(), layout.l2p_map());
    assert_eq!(result.diagnostics.control_flow_blocks_routed, 1);
    match &result.circuit.operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert!(gate.true_body().iter().any(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::SWAP)
            )));
        }
        other => panic!("expected routed if/else operation, got {other:?}"),
    }
}

#[test]
fn if_else_routes_both_branches_and_restores_layout() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let true_body = vec![cx_operation(0, 1)];
    let false_body = vec![cx_operation(1, 0)];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            true_body,
            Some(false_body),
        )
        .unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.final_layout.l2p_map(), layout.l2p_map());
    assert_eq!(result.diagnostics.control_flow_blocks_routed, 2);
    match &result.circuit.operations()[0].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            assert!(gate.true_body().iter().any(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::SWAP)
            )));
            assert!(gate.false_body().unwrap().iter().any(|operation| matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::SWAP)
            )));
        }
        other => panic!("expected routed if/else operation, got {other:?}"),
    }
}

#[test]
fn empty_control_flow_bodies_route_without_layout_drift() {
    let device = Device::line("line", 2).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), vec![], Some(vec![]))
        .unwrap();
    circuit
        .while_loop(ConditionView::new(Qubit::new(1), 1), vec![])
        .unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.swap_count, 0);
    assert_eq!(result.final_layout.l2p_map(), layout.l2p_map());
    assert_eq!(result.diagnostics.control_flow_blocks_routed, 3);
}

#[test]
fn route_keeps_grid_adjacent_gates_without_swap() {
    let device = Device::grid("grid", 2, 2).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1), (2, 2), (3, 3)], 4).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(4);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(2), Qubit::new(3)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.swap_count, 0);
    assert_eq!(result.circuit.operations().len(), 2);
    for operation in result.circuit.operations() {
        assert!(operation_is_adjacent_on_grid(operation, 2));
    }
}

#[test]
fn route_handles_empty_and_single_qubit_circuits_without_swap() {
    let device = Device::line("line", 2).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);

    let empty = Circuit::new(2);
    let empty_result = sabre_route(&empty, &device, &layout, &config).unwrap();
    assert_eq!(empty_result.swap_count, 0);
    assert!(empty_result.circuit.operations().is_empty());

    let mut single_qubit = Circuit::new(2);
    single_qubit.h(Qubit::new(0)).unwrap();
    single_qubit.x(Qubit::new(1)).unwrap();
    let single_result = sabre_route(&single_qubit, &device, &layout, &config).unwrap();
    assert_eq!(single_result.swap_count, 0);
    assert_eq!(single_result.circuit.operations().len(), 2);
    assert_eq!(single_result.final_layout.l2p_map(), layout.l2p_map());
}

#[test]
fn route_rejects_three_qubit_gate_before_routing() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1), (2, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(3);
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();

    let error = sabre_route(&circuit, &device, &layout, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("more than two qubits"))
    );
}

#[test]
fn refine_layout_rejects_more_logical_than_physical_qubits() {
    let device = Device::line("line", 1).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = sabre_refine_layout(&circuit, &device, None, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("2 logical qubits") && message.contains("1 usable physical qubits"))
    );
}

#[test]
fn route_rejects_incomplete_initial_layout() {
    let device = Device::line("line", 2).unwrap();
    let incomplete = Layout::from_pairs(&[(0, 0)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = sabre_route(&circuit, &device, &incomplete, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("does not map logical qubit"))
    );
}

#[test]
fn route_disconnected_topology_errors_without_panic() {
    let qubits = vec![PhysicalQubit::new(0), PhysicalQubit::new(1)];
    let topology = Topology::new(
        qubits.clone(),
        Vec::<(PhysicalQubit, PhysicalQubit, String)>::new(),
    )
    .unwrap();
    let device = Device::new(
        "disconnected",
        qubits.iter().copied().collect::<HashSet<_>>(),
        topology,
    )
    .unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 1)], 2).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = sabre_route(&circuit, &device, &layout, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("disconnected"))
    );
}

#[test]
fn route_preserves_parameterized_gate_parameters() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit
        .rx(Qubit::new(0), ParameterValue::Param(theta.clone()))
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert!(result.circuit.parameters().contains(&theta));
    let rx = result
        .circuit
        .operations()
        .iter()
        .find(|operation| {
            matches!(
                operation.instruction,
                Instruction::Standard(StandardGate::RX)
            )
        })
        .expect("routed circuit preserves RX operation");
    assert!(matches!(rx.params.as_slice(), [CircuitParam::Index(_)]));
}

#[test]
fn route_preserves_multiple_parameters_and_global_phase() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let gamma = Parameter::symbol("gamma");
    let mut circuit = Circuit::new(2);
    circuit.set_global_phase(gamma.clone());
    circuit
        .rx(Qubit::new(0), ParameterValue::Param(theta.clone()))
        .unwrap();
    circuit
        .rz(Qubit::new(1), ParameterValue::Param(phi.clone()))
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.circuit.global_phase(), gamma);
    assert!(result.circuit.parameters().contains(&theta));
    assert!(result.circuit.parameters().contains(&phi));
    assert!(result.circuit.parameters().contains(&gamma));
    assert!(result.circuit.operations().iter().any(|operation| matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::RX)
    ) && matches!(
        operation.params.as_slice(),
        [CircuitParam::Index(_)]
    )));
    assert!(result.circuit.operations().iter().any(|operation| matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::RZ)
    ) && matches!(
        operation.params.as_slice(),
        [CircuitParam::Index(_)]
    )));
}

#[test]
fn nested_control_flow_is_routed_and_restored() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig::deterministic_seeded(7);
    let while_body = vec![cx_operation(0, 1)];
    let while_operation = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::while_loop(
            ConditionView::new(Qubit::new(0), 1),
            while_body,
        )),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![],
        label: None,
    };
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![while_operation],
            None,
        )
        .unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.final_layout.l2p_map(), layout.l2p_map());
    assert_eq!(result.diagnostics.control_flow_blocks_routed, 2);
}

#[test]
fn routing_trials_select_no_worse_than_first_trial() {
    let device = Device::line("line", 4).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 3), (2, 1)], 4).unwrap();
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(2), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let first = sabre_route(
        &circuit,
        &device,
        &layout,
        &SabreConfig {
            routing_trials: 1,
            seed: Some(19),
            ..SabreConfig::deterministic_seeded(7)
        },
    )
    .unwrap();
    let multi = sabre_route(
        &circuit,
        &device,
        &layout,
        &SabreConfig {
            routing_trials: 5,
            seed: Some(19),
            ..SabreConfig::deterministic_seeded(7)
        },
    )
    .unwrap();

    assert_eq!(multi.diagnostics.trials_evaluated, 5);
    assert!(multi.swap_count <= first.swap_count);
}

#[test]
fn fallback_triggers_when_attempt_limit_is_zero() {
    let device = Device::line("line", 4).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 3)], 4).unwrap();
    let config = SabreConfig {
        heuristic: SabreHeuristicConfig {
            attempt_limit: 0,
            ..SabreConfig::deterministic_seeded(7).heuristic
        },
        ..SabreConfig::deterministic_seeded(7)
    };
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert!(result.swap_count > 0);
    assert!(result.diagnostics.fallback_count > 0);
    assert_all_two_qubit_operations_are_adjacent_on_line(&result.circuit);
}

#[test]
fn refinement_with_multiple_iterations_is_no_worse_than_zero_iterations_for_seeded_case() {
    let device = Device::line("line", 5).unwrap();
    let objective = LayoutObjective::topology_only();
    let mut circuit = Circuit::new(4);
    circuit.cx(Qubit::new(0), Qubit::new(3)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(3)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    let no_refinement = SabreConfig {
        refinement_iterations: 0,
        routing_trials: 3,
        seed: Some(31),
        ..SabreConfig::deterministic_seeded(7)
    };
    let refined = SabreConfig {
        refinement_iterations: 2,
        routing_trials: 3,
        seed: Some(31),
        ..SabreConfig::deterministic_seeded(7)
    };

    let base_layout =
        sabre_refine_layout(&circuit, &device, None, &objective, &no_refinement).unwrap();
    let refined_layout =
        sabre_refine_layout(&circuit, &device, None, &objective, &refined).unwrap();
    let base_route = sabre_route(&circuit, &device, &base_layout.layout, &no_refinement).unwrap();
    let refined_route = sabre_route(&circuit, &device, &refined_layout.layout, &refined).unwrap();

    assert!(refined_route.swap_count <= base_route.swap_count);
}

#[test]
fn trial_objective_controls_quality_tie_breaking() {
    let left = super::routing::TrialQuality {
        swap_count: 1,
        two_qubit_depth: 5,
        operation_count: 20,
    };
    let right = super::routing::TrialQuality {
        swap_count: 1,
        two_qubit_depth: 2,
        operation_count: 8,
    };

    assert!(
        super::routing::compare_trial_quality(SabreTrialObjective::SwapCount, left, 0, right, 1)
            .is_lt()
    );
    assert!(
        super::routing::compare_trial_quality(
            SabreTrialObjective::SwapThenDepth,
            left,
            0,
            right,
            1
        )
        .is_gt()
    );
}

#[test]
fn route_diagnostics_report_selected_quality_metrics() {
    let device = Device::line("line", 3).unwrap();
    let layout = Layout::from_pairs(&[(0, 0), (1, 2)], 3).unwrap();
    let config = SabreConfig {
        routing_trials: 3,
        seed: Some(11),
        ..SabreConfig::deterministic_seeded(7)
    };
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = sabre_route(&circuit, &device, &layout, &config).unwrap();

    assert_eq!(result.diagnostics.trials_evaluated, 3);
    assert!(result.diagnostics.selected_trial_index < 3);
    assert!(result.diagnostics.two_qubit_depth > 0);
    assert_eq!(
        result.diagnostics.operation_count,
        result.circuit.operations().len()
    );
}

#[test]
fn layout_scoring_trials_must_be_positive() {
    let device = Device::line("line", 2).unwrap();
    let objective = LayoutObjective::topology_only();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let config = SabreConfig {
        layout_scoring_trials: 0,
        ..SabreConfig::deterministic_seeded(7)
    };

    let error = sabre_refine_layout(&circuit, &device, None, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("layout_scoring_trials"))
    );
}

fn are_adjacent(left: Qubit, right: Qubit) -> bool {
    left.id().abs_diff(right.id()) == 1
}

fn assert_all_two_qubit_operations_are_adjacent_on_line(circuit: &Circuit) {
    for operation in circuit.operations() {
        if operation.qubits.len() == 2 {
            assert!(
                are_adjacent(operation.qubits[0], operation.qubits[1]),
                "operation {operation:?} is not adjacent on line topology"
            );
        }
    }
}

fn operation_is_adjacent_on_grid(operation: &Operation, cols: u32) -> bool {
    if operation.qubits.len() != 2 {
        return true;
    }
    let left = operation.qubits[0].id();
    let right = operation.qubits[1].id();
    left.abs_diff(right) == 1 && left / cols == right / cols || left.abs_diff(right) == cols
}

fn cx_operation(control: u32, target: u32) -> Operation {
    Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![Qubit::new(control), Qubit::new(target)],
        params: smallvec![],
        label: None,
    }
}
