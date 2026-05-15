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

use super::{DecomposeConfig, Decomposer};
use crate::circuit::{
    Circuit, ConditionView, ControlFlow, Instruction, Operation, Qubit, StandardGate,
};
use crate::compiler::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::Transformer;
use crate::device::{Device, Topology};
use smallvec::smallvec;
use std::collections::HashSet;

fn mock_device(native_gates: Vec<Instruction>, qubit_count: usize) -> Device {
    let qubits: Vec<_> = (0..qubit_count)
        .map(|index| Qubit::new(index as u32))
        .collect();
    let topology = Topology::new(qubits.clone(), vec![]).unwrap();
    Device::new("mock-qpu", HashSet::from_iter(qubits), topology)
        .unwrap()
        .with_native_gates(native_gates)
}

#[test]
fn decompose_requires_target_gate_source() {
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let err = Decomposer::new(DecomposeConfig::new())
        .run(&mut ctx)
        .unwrap_err();

    assert!(matches!(err, CompilerError::MissingDevice));
}

#[test]
fn decompose_can_use_explicit_target_gate_basis_without_device() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = Decomposer::new(
        DecomposeConfig::new().with_target_gates(vec![StandardGate::H, StandardGate::CZ]),
    )
    .run(&mut ctx)
    .unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::CZ)
    ));
    assert!(matches!(
        operations[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn decompose_rejects_empty_target_standard_gate_set() {
    let device = mock_device(vec![Instruction::Delay], 1);
    let mut ctx = CompilerContext::with_device(Circuit::new(1), device);

    let err = Decomposer::new(DecomposeConfig::new())
        .run(&mut ctx)
        .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidContextState(message)
            if message.contains("target standard gate set is empty")
    ));
}

#[test]
fn decompose_rejects_empty_explicit_target_gate_basis() {
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let err = Decomposer::new(DecomposeConfig::new().with_target_gates(Vec::new()))
        .run(&mut ctx)
        .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidContextState(message)
            if message.contains("target standard gate set is empty")
    ));
}

#[test]
fn decompose_lowers_cx_to_device_standard_gate_basis() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let device = mock_device(vec![StandardGate::H.into(), StandardGate::CZ.into()], 2);
    let mut ctx = CompilerContext::with_device(circuit, device);

    let outcome = Decomposer::new(DecomposeConfig::new())
        .run(&mut ctx)
        .unwrap();

    assert!(outcome.changed);
    assert_eq!(ctx.revision(), 1);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(1)]);
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::CZ)
    ));
    assert_eq!(
        operations[1].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
    assert!(matches!(
        operations[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(operations[2].qubits.as_slice(), &[Qubit::new(1)]);
}

#[test]
fn decompose_keeps_already_native_standard_gate() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let device = mock_device(vec![StandardGate::H.into()], 1);
    let mut ctx = CompilerContext::with_device(circuit, device);

    let outcome = Decomposer::new(DecomposeConfig::new())
        .run(&mut ctx)
        .unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.revision(), 0);
    assert_eq!(ctx.circuit().operations().len(), 1);
    assert!(matches!(
        ctx.circuit().operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn decompose_recurses_into_control_flow_bodies_by_default() {
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                params: smallvec![],
                label: None,
            }],
            None,
        )
        .unwrap();
    let device = mock_device(vec![StandardGate::H.into(), StandardGate::CZ.into()], 2);
    let mut ctx = CompilerContext::with_device(circuit, device);

    let outcome = Decomposer::new(DecomposeConfig::new())
        .run(&mut ctx)
        .unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) = &operations[0].instruction else {
        panic!("expected rewritten if-else operation");
    };
    let body = gate.true_body();
    assert_eq!(body.len(), 3);
    assert!(matches!(
        body[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        body[1].instruction,
        Instruction::Standard(StandardGate::CZ)
    ));
    assert!(matches!(
        body[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}
