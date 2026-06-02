// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::CompilerWorkflow;
use crate::circuit::{Circuit, Instruction, Qubit, StandardGate};
use crate::compiler::{CompileConfig, CompileMode, CompilerError, compile};
use crate::device::{Device, PhysicalQubit, Topology};
use std::collections::HashSet;

fn standard_ops(circuit: &Circuit) -> Vec<StandardGate> {
    circuit
        .operations()
        .iter()
        .filter_map(|operation| match operation.instruction {
            Instruction::Standard(gate) => Some(gate),
            _ => None,
        })
        .collect()
}

fn compile_config(mode: CompileMode) -> CompileConfig {
    CompileConfig {
        mode,
        target_basis: None,
        device: None,
        seed: None,
    }
}

fn run_workflow(circuit: &Circuit, mode: CompileMode) -> super::CompileResult {
    CompilerWorkflow::new(compile_config(mode))
        .run(circuit)
        .unwrap()
}

fn two_qubit_device(native_gates: Vec<Instruction>) -> Device {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let qubits = HashSet::from([q0, q1]);
    let topology = Topology::new(vec![q0, q1], vec![(q0, q1, "q0-q1".to_string())]).unwrap();
    Device::new("test-device", qubits, topology)
        .unwrap()
        .with_native_gates(native_gates)
}

#[test]
fn normal_workflow_cancels_adjacent_self_inverse_gates() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.h(q0).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(result.changed);
    assert_eq!(result.mode, CompileMode::Normal);
    assert!(result.circuit.operations().is_empty());
}

#[test]
fn normal_workflow_reports_no_change_for_stable_circuit() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();

    let result = run_workflow(&circuit, CompileMode::Normal);

    assert!(!result.changed);
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::H]);
}

#[test]
fn normal_workflow_reports_staged_order() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();

    let result = CompilerWorkflow::new(compile_config(CompileMode::Normal))
        .run(&circuit)
        .unwrap();

    assert_eq!(
        result
            .steps
            .iter()
            .map(|step| step.name)
            .collect::<Vec<_>>(),
        vec![
            "resolve.target",
            "canonicalize.input",
            "optimize.light",
            "translate.target_basis",
            "canonicalize.output",
        ]
    );
    assert!(result.steps[3].skipped);
}

#[test]
fn enhanced_workflow_uses_richer_stage_sequence() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.25).unwrap();
    circuit.rz(Qubit::new(0), 0.5).unwrap();
    circuit.rz(Qubit::new(0), -0.75).unwrap();

    let normal = run_workflow(&circuit, CompileMode::Normal);
    let enhanced = run_workflow(&circuit, CompileMode::Enhanced);

    assert!(enhanced.changed);
    assert!(enhanced.steps.len() > normal.steps.len());
    assert_eq!(
        enhanced
            .steps
            .iter()
            .map(|step| step.name)
            .collect::<Vec<_>>(),
        vec![
            "resolve.target",
            "canonicalize.input",
            "optimize.pre_translation",
            "translate.target_basis",
            "optimize.cleanup",
            "canonicalize.mid",
            "optimize.final",
            "canonicalize.output",
        ]
    );
    assert!(enhanced.steps[3].skipped);
    assert!(enhanced.circuit.operations().is_empty());
}

#[test]
fn explicit_target_basis_runs_lowering() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ]),
        device: None,
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert!(result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::CZ, StandardGate::H]
    );
    assert_eq!(result.circuit.operations()[0].qubits.as_slice(), &[q1]);
    assert_eq!(result.circuit.operations()[1].qubits.as_slice(), &[q0, q1]);
    assert_eq!(result.circuit.operations()[2].qubits.as_slice(), &[q1]);
}

#[test]
fn target_basis_failure_is_reported() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();

    let err = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Normal,
        target_basis: Some(vec![Instruction::Standard(StandardGate::CZ)]),
        device: None,
        seed: None,
    })
    .run(&circuit)
    .unwrap_err();

    assert!(matches!(err, CompilerError::InvalidInput(_)));
}

#[test]
fn device_native_gates_are_used_as_target_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.cx(q0, q1).unwrap();
    let device = two_qubit_device(vec![
        Instruction::Standard(StandardGate::H),
        Instruction::Standard(StandardGate::CZ),
    ]);

    let result = CompilerWorkflow::new(CompileConfig {
        mode: CompileMode::Enhanced,
        target_basis: None,
        device: Some(device),
        seed: None,
    })
    .run(&circuit)
    .unwrap();

    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::H, StandardGate::CZ, StandardGate::H]
    );
}

#[test]
fn compile_matches_built_workflow() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.x(q1).unwrap();
    circuit.h(q0).unwrap();

    let direct = compile(&circuit, compile_config(CompileMode::Normal)).unwrap();
    let built = CompilerWorkflow::new(compile_config(CompileMode::Normal))
        .run(&circuit)
        .unwrap();

    assert_eq!(direct.changed, built.changed);
    assert_eq!(standard_ops(&direct.circuit), standard_ops(&built.circuit));
    assert_eq!(direct.steps, built.steps);
}

#[test]
fn workflow_config_can_build_enhanced_workflow() {
    let workflow = CompilerWorkflow::new(compile_config(CompileMode::Enhanced));

    assert_eq!(workflow.config().mode, CompileMode::Enhanced);
}
