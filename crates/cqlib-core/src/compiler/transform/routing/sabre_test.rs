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
use crate::circuit::{Circuit, Qubit};
use crate::compiler::CompilerError;
use crate::compiler::sabre::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
use crate::device::{Device, PhysicalQubit, Topology};
use std::collections::HashSet;

#[test]
fn sabre_routing_auto_layout_routes_non_embeddable_interactions() {
    let device = line_device(3);
    let objective = LayoutObjective::topology_only();
    let config = deterministic_config();
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let result = route_sabre(&circuit, &device, &objective, &config).unwrap();

    assert!(result.changed);
    assert!(result.swap_count > 0);
    assert_eq!(result.circuit.qubits().len(), 3);
    assert_all_two_qubit_operations_are_adjacent_on_line(&result.circuit);
}

#[test]
fn sabre_routing_keeps_adjacent_two_qubit_circuit_without_swap() {
    let device = line_device(2);
    let objective = LayoutObjective::topology_only();
    let config = deterministic_config();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let result = route_sabre(&circuit, &device, &objective, &config).unwrap();

    assert!(!result.changed);
    assert_eq!(result.swap_count, 0);
    assert_eq!(result.diagnostics.trials_evaluated, config.routing_trials);
    assert_eq!(result.circuit.operations().len(), 1);
}

#[test]
fn sabre_routing_is_reproducible_for_same_seed() {
    let device = line_device(4);
    let objective = LayoutObjective::topology_only();
    let config = deterministic_config();
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let first = route_sabre(&circuit, &device, &objective, &config).unwrap();
    let second = route_sabre(&circuit, &device, &objective, &config).unwrap();

    assert_eq!(
        first.initial_layout.l2p_map(),
        second.initial_layout.l2p_map()
    );
    assert_eq!(first.final_layout.l2p_map(), second.final_layout.l2p_map());
    assert_eq!(first.swap_count, second.swap_count);
    assert_eq!(
        first.diagnostics.operation_count,
        second.diagnostics.operation_count
    );
    assert_eq!(
        first.layout_score.as_ref().map(|score| score.total),
        second.layout_score.as_ref().map(|score| score.total)
    );
}

#[test]
fn sabre_routing_rejects_invalid_config() {
    let device = line_device(2);
    let objective = LayoutObjective::topology_only();
    let mut config = deterministic_config();
    config.routing_trials = 0;
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let error = route_sabre(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("routing_trials"))
    );
}

#[test]
fn sabre_routing_rejects_insufficient_physical_qubits() {
    let p0 = PhysicalQubit::new(0);
    let topology = Topology::new(vec![p0], vec![]).unwrap();
    let device = Device::new("one", HashSet::from_iter([p0]), topology).unwrap();
    let objective = LayoutObjective::topology_only();
    let config = deterministic_config();
    let circuit = Circuit::new(2);

    let error = route_sabre(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("2 logical qubits") && message.contains("1 usable physical qubits"))
    );
}

#[test]
fn sabre_routing_rejects_undecomposed_three_qubit_gate() {
    let device = line_device(3);
    let objective = LayoutObjective::topology_only();
    let config = deterministic_config();
    let mut circuit = Circuit::new(3);
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();

    let error = route_sabre(&circuit, &device, &objective, &config).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("more than two qubits"))
    );
}

fn deterministic_config() -> SabreConfig {
    SabreConfig {
        layout_trials: 2,
        refinement_iterations: 1,
        layout_scoring_trials: 1,
        routing_trials: 1,
        trial_objective: SabreTrialObjective::SwapThenDepth,
        seed: Some(7),
        heuristic: SabreHeuristicConfig {
            lookahead_weights: vec![0.5],
            attempt_limit: 20,
            ..SabreHeuristicConfig::default()
        },
    }
}

fn line_device(count: u32) -> Device {
    let qubits = (0..count).map(PhysicalQubit::new).collect::<Vec<_>>();
    let couplings = qubits
        .windows(2)
        .map(|window| (window[0], window[1], "cx".to_string()))
        .collect::<Vec<_>>();
    let topology = Topology::new(qubits.clone(), couplings).unwrap();
    Device::new(
        "line",
        qubits.iter().copied().collect::<HashSet<_>>(),
        topology,
    )
    .unwrap()
}

fn assert_all_two_qubit_operations_are_adjacent_on_line(circuit: &Circuit) {
    for operation in circuit.operations() {
        if operation.qubits.len() == 2 {
            assert!(
                operation.qubits[0].id().abs_diff(operation.qubits[1].id()) == 1,
                "operation {operation:?} is not adjacent on line topology"
            );
        }
    }
}
