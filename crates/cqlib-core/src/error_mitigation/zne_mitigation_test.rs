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

use super::ZNEMitigation;
use crate::circuit::Qubit;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::CircuitError;
use crate::circuit::gate::{Instruction, StandardGate};

#[test]
fn test_zne_new_sets_noise_factors() {
    let circuit = Circuit::new(1);
    let zne = ZNEMitigation::new(circuit, vec![0, 1, 3]);

    assert_eq!(zne.fold_levels(), &[0, 1, 3]);
    assert_eq!(zne.noise_factors(), &[1, 3, 7]);
}

#[test]
fn test_fold_circuits_global_level_one() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.s(q0).unwrap();

    let zne = ZNEMitigation::new(circuit, vec![1]);
    let folded = zne.fold_circuits(None).unwrap();

    assert_eq!(folded.len(), 1);
    let ops = folded[0].operations();
    assert_eq!(ops.len(), 3);
    assert!(matches!(
        ops[0].instruction,
        Instruction::Standard(StandardGate::S)
    ));
    assert!(matches!(
        ops[1].instruction,
        Instruction::Standard(StandardGate::SDG)
    ));
    assert!(matches!(
        ops[2].instruction,
        Instruction::Standard(StandardGate::S)
    ));
}

#[test]
fn test_fold_circuits_selective_gate_set() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.s(q0).unwrap();

    let zne = ZNEMitigation::new(circuit, vec![1]);
    let gate_set = vec![Instruction::Standard(StandardGate::S)];
    let folded = zne.fold_circuits(Some(&gate_set)).unwrap();

    assert_eq!(folded.len(), 1);
    let ops = folded[0].operations();
    assert_eq!(ops.len(), 4);
    assert!(matches!(
        ops[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        ops[1].instruction,
        Instruction::Standard(StandardGate::S)
    ));
    assert!(matches!(
        ops[2].instruction,
        Instruction::Standard(StandardGate::SDG)
    ));
    assert!(matches!(
        ops[3].instruction,
        Instruction::Standard(StandardGate::S)
    ));
}

#[test]
fn test_fold_circuits_negative_level_returns_error() {
    let circuit = Circuit::new(1);
    let zne = ZNEMitigation::new(circuit, vec![-1]);
    let result = zne.fold_circuits(None);

    assert!(matches!(
        result,
        Err(CircuitError::InvalidControlOperation(_))
    ));
}
