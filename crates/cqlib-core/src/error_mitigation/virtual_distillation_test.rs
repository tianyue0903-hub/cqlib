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

use super::{VirtualDistillation, VirtualDistillationError};
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::gate::Instruction;
use crate::circuit::gate::standard_gate::StandardGate;
use crate::circuit::Qubit;

#[test]
fn test_vd_new_accepts_valid_input() {
    let circuit = Circuit::new(1);
    let vd = VirtualDistillation::new(circuit, 2);
    assert!(vd.is_ok());
}

#[test]
fn test_vd_new_rejects_invalid_copies() {
    let circuit = Circuit::new(1);
    let err = VirtualDistillation::new(circuit, 1).unwrap_err();
    assert_eq!(err, VirtualDistillationError::InvalidCopies(1));
}

#[test]
fn test_build_copy_swap_circuit_for_two_single_qubit_copies() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x(q0).unwrap();

    let vd = VirtualDistillation::new(circuit, 2).unwrap();
    let copy_swap = vd.build_copy_swap_circuit().unwrap();
    let ops = copy_swap.operations();

    assert_eq!(copy_swap.width(), 2);
    assert_eq!(ops.len(), 3);

    assert!(matches!(
        ops[0].instruction,
        Instruction::Standard(StandardGate::X)
    ));
    assert_eq!(ops[0].qubits.as_slice(), &[Qubit::new(0)]);

    assert!(matches!(
        ops[1].instruction,
        Instruction::Standard(StandardGate::X)
    ));
    assert_eq!(ops[1].qubits.as_slice(), &[Qubit::new(1)]);

    assert!(matches!(
        ops[2].instruction,
        Instruction::Standard(StandardGate::SWAP)
    ));
    assert_eq!(ops[2].qubits.as_slice(), &[Qubit::new(0), Qubit::new(1)]);
}

#[test]
fn test_build_copy_swap_circuit_adds_pairwise_swaps_for_multiple_copies() {
    let vd = VirtualDistillation::new(Circuit::new(1), 3).unwrap();
    let copy_swap = vd.build_copy_swap_circuit().unwrap();
    let ops = copy_swap.operations();

    assert_eq!(copy_swap.width(), 3);
    assert_eq!(ops.len(), 2);

    let swap_count = ops
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Standard(StandardGate::SWAP)))
        .count();
    assert_eq!(swap_count, 2);
}

