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

use super::decompose_mcx_small;
use crate::circuit::{Instruction, Qubit, StandardGate, operation::ValueOperation};
use crate::compiler::error::CompilerError;

fn assert_single_operation(
    operations: &[ValueOperation],
    expected_gate: StandardGate,
    expected_qubits: &[Qubit],
) {
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(gate) if gate == expected_gate
    ));
    assert_eq!(operations[0].qubits.as_slice(), expected_qubits);
    assert!(operations[0].params.is_empty());
    assert!(operations[0].label.is_none());
}

#[test]
fn trivial_mcx_without_controls_emits_x() {
    let target = Qubit::new(2);

    let operations = decompose_mcx_small(&[], target).unwrap();

    assert_single_operation(&operations, StandardGate::X, &[target]);
}

#[test]
fn trivial_mcx_with_one_control_emits_cx() {
    let control = Qubit::new(1);
    let target = Qubit::new(2);

    let operations = decompose_mcx_small(&[control], target).unwrap();

    assert_single_operation(&operations, StandardGate::CX, &[control, target]);
}

#[test]
fn trivial_mcx_with_two_controls_emits_ccx() {
    let first_control = Qubit::new(1);
    let second_control = Qubit::new(3);
    let target = Qubit::new(2);

    let operations = decompose_mcx_small(&[first_control, second_control], target).unwrap();

    assert_single_operation(
        &operations,
        StandardGate::CCX,
        &[first_control, second_control, target],
    );
}

#[test]
fn trivial_mcx_rejects_three_controls() {
    let error = decompose_mcx_small(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mcx",
            ref reason,
        } if reason == "trivial MCX decomposition supports at most 2 controls, got 3"
    ));
}

#[test]
fn trivial_mcx_rejects_duplicate_controls() {
    let duplicate = Qubit::new(1);
    let error = decompose_mcx_small(&[duplicate, duplicate], Qubit::new(2)).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mcx",
            ref reason,
        } if reason == "MCX controls, target, and ancillas must be distinct; duplicate Q1"
    ));
}

#[test]
fn trivial_mcx_rejects_target_matching_control() {
    let duplicate = Qubit::new(1);
    let error = decompose_mcx_small(&[duplicate], duplicate).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mcx",
            ref reason,
        } if reason == "MCX controls, target, and ancillas must be distinct; duplicate Q1"
    ));
}

#[test]
fn trivial_mcx_validates_qubits_before_rejecting_control_count() {
    let duplicate = Qubit::new(1);
    let error =
        decompose_mcx_small(&[duplicate, duplicate, Qubit::new(2)], Qubit::new(3)).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mcx",
            ref reason,
        } if reason == "MCX controls, target, and ancillas must be distinct; duplicate Q1"
    ));
}
