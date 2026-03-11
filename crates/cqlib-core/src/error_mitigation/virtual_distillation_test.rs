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
fn test_vd_copies_getter_and_setter() {
    let circuit = Circuit::new(1);
    let mut vd = VirtualDistillation::new(circuit, 2).unwrap();

    assert_eq!(vd.copies(), 2);

    vd.set_copies(3).unwrap();
    assert_eq!(vd.copies(), 3);

    let err = vd.set_copies(1).unwrap_err();
    assert_eq!(err, VirtualDistillationError::InvalidCopies(1));
    assert_eq!(vd.copies(), 3);
}

#[test]
fn test_build_copy_swap_circuit_for_two_single_qubit_copies() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x(q0).unwrap();

    let vd = VirtualDistillation::new(circuit, 2).unwrap();
    let copy_swap = vd.build_copy_swap_circuit(None).unwrap();
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
    let copy_swap = vd.build_copy_swap_circuit(None).unwrap();
    let ops = copy_swap.operations();

    assert_eq!(copy_swap.width(), 3);
    assert_eq!(ops.len(), 2);

    let swap_count = ops
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Standard(StandardGate::SWAP)))
        .count();
    assert_eq!(swap_count, 2);
}

#[test]
fn test_build_copy_swap_circuit_applies_optional_observable_to_first_copy() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x(q0).unwrap();

    let mut observable = Circuit::new(1);
    observable.z(q0).unwrap();

    let vd = VirtualDistillation::new(circuit, 2).unwrap();
    let copy_swap = vd.build_copy_swap_circuit(Some(observable)).unwrap();
    let ops = copy_swap.operations();

    assert_eq!(ops.len(), 4);
    assert!(matches!(
        ops[3].instruction,
        Instruction::Standard(StandardGate::Z)
    ));
    assert_eq!(ops[3].qubits.as_slice(), &[Qubit::new(0)]);
}

#[test]
fn test_build_copy_swap_circuit_rejects_optional_observable_qubit_mismatch() {
    let vd = VirtualDistillation::new(Circuit::new(1), 2).unwrap();
    let observable = Circuit::new(2);

    let err = vd.build_copy_swap_circuit(Some(observable)).unwrap_err();

    assert!(matches!(
        err,
        crate::circuit::CircuitError::QubitCountMismatch {
            expected: 1,
            actual: 2
        }
    ));
}

#[test]
fn test_run_denominator_circuit_runs_copy_swap_circuit() {
    let vd = VirtualDistillation::new(Circuit::new(1), 2).unwrap();

    let observed_values = vd
        .run_denominator_circuit(128, |denominator, _shots| {
            let denominator_ops = denominator.operations();

            assert_eq!(denominator.width(), 2);
            assert_eq!(denominator_ops.len(), 1);
            assert!(matches!(
                denominator_ops[0].instruction,
                Instruction::Standard(StandardGate::SWAP)
            ));

            vec![1.0]
        })
        .unwrap();

    assert_eq!(observed_values, vec![1.0]);
}

#[test]
fn test_run_numerator_circuit_applies_observable_to_first_copy() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.x(q0).unwrap();

    let mut observable = Circuit::new(1);
    observable.z(q0).unwrap();

    let vd = VirtualDistillation::new(circuit, 2).unwrap();
    let observed_values = vd
        .run_numerator_circuit(observable, 128, |numerator, _shots| {
            let numerator_ops = numerator.operations();

            assert_eq!(numerator_ops.len(), 4);

            assert!(matches!(
                numerator_ops[3].instruction,
                Instruction::Standard(StandardGate::Z)
            ));
            assert_eq!(numerator_ops[3].qubits.as_slice(), &[Qubit::new(0)]);

            vec![1.0]
        })
        .unwrap();

    assert_eq!(observed_values, vec![1.0]);
}

#[test]
fn test_run_vd_returns_mu_and_var() {
    let q0 = Qubit::new(0);
    let base_circuit = Circuit::new(1);

    let mut observable_z = Circuit::new(1);
    observable_z.z(q0).unwrap();

    let mut observable_x = Circuit::new(1);
    observable_x.x(q0).unwrap();

    let vd = VirtualDistillation::new(base_circuit, 2).unwrap();
    let (mu_vd, var_vd) = vd
        .run_vd(
            vec![observable_z, observable_x],
            vec![2.0, -0.5],
            3,
            2,
            |circuit, shots| {
                let ops = circuit.operations();

                assert!(!ops.is_empty());
                if matches!(ops.last().unwrap().instruction, Instruction::Standard(StandardGate::Z))
                {
                    assert_eq!(shots, 3);
                    vec![1.0, 2.0, 3.0]
                } else if matches!(
                    ops.last().unwrap().instruction,
                    Instruction::Standard(StandardGate::X)
                ) {
                    assert_eq!(shots, 3);
                    vec![4.0, 5.0, 6.0]
                } else {
                    assert_eq!(shots, 2);
                    assert_eq!(ops.len(), 1);
                    assert!(matches!(
                        ops[0].instruction,
                        Instruction::Standard(StandardGate::SWAP)
                    ));
                    vec![2.0, 2.0]
                }
            },
        )
        .unwrap();

    assert!((mu_vd - 0.75).abs() < 1e-12);
    assert!((var_vd - 0.375).abs() < 1e-12);
}

#[test]
fn test_run_vd_rejects_mismatched_observables_and_coefficients() {
    let vd = VirtualDistillation::new(Circuit::new(1), 2).unwrap();

    let err = vd
        .run_vd(vec![Circuit::new(1)], vec![], 3, 2, |_circuit, _shots| {
            vec![1.0]
        })
        .unwrap_err();

    assert!(matches!(
        err,
        crate::circuit::CircuitError::InvalidOperation(message)
            if message.contains("number of observables and coefficients")
    ));
}

#[test]
fn test_run_vd_rejects_zero_samples() {
    let vd = VirtualDistillation::new(Circuit::new(1), 2).unwrap();

    let err = vd
        .run_vd(vec![], vec![], 0, 2, |_circuit, _shots| vec![])
        .unwrap_err();

    assert!(matches!(
        err,
        crate::circuit::CircuitError::InvalidOperation(message)
            if message.contains("must be greater than 0")
    ));
}

#[test]
fn test_run_vd_rejects_zero_denominator_mean() {
    let q0 = Qubit::new(0);
    let mut observable = Circuit::new(1);
    observable.z(q0).unwrap();

    let vd = VirtualDistillation::new(Circuit::new(1), 2).unwrap();
    let err = vd
        .run_vd(vec![observable], vec![1.0], 2, 2, |circuit, shots| {
            let ops = circuit.operations();

            if matches!(ops.last().unwrap().instruction, Instruction::Standard(StandardGate::Z)) {
                vec![1.0; shots]
            } else {
                vec![0.0; shots]
            }
        })
        .unwrap_err();

    assert!(matches!(
        err,
        crate::circuit::CircuitError::InvalidOperation(message)
            if message.contains("denominator mean is zero")
    ));
}
