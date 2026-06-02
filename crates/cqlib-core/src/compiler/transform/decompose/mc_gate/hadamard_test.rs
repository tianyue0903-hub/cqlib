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

use super::hadamard::{decompose_hadamard_n_clean, decompose_hadamard_no_aux};
use crate::circuit::{
    Circuit, Instruction, MCGate, ParameterValue, Qubit, StandardGate, circuit_to_matrix,
};
use crate::compiler::error::CompilerError;
use crate::util::test_utils::{assert_standard_operation, circuit_from_value_operations};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;

fn mc_hadamard_matrix(num_qubits: usize, controls: &[Qubit], target: Qubit) -> Array2<Complex64> {
    let mut circuit = Circuit::new(num_qubits);
    let mut qubits = controls.to_vec();
    qubits.push(target);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(controls.len() as u8, StandardGate::H))),
            qubits,
            [],
            None,
        )
        .unwrap();
    circuit_to_matrix(&circuit, None).unwrap()
}

fn assert_selected_columns_approx_eq(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
) {
    assert_eq!(actual.shape(), expected.shape());
    for column in columns {
        for row in 0..expected.nrows() {
            assert!(
                (actual[[row, column]] - expected[[row, column]]).norm() < EPSILON,
                "matrix mismatch at row {row}, column {column}: actual={}, expected={}",
                actual[[row, column]],
                expected[[row, column]]
            );
        }
    }
}

fn assert_fixed_parameter(
    operation: &crate::circuit::operation::ValueOperation,
    gate: StandardGate,
    qubits: &[Qubit],
    expected: f64,
) {
    assert!(matches!(
        operation.instruction,
        Instruction::Standard(actual) if actual == gate
    ));
    assert_eq!(operation.qubits.as_slice(), qubits);
    assert!(matches!(
        operation.params.as_slice(),
        [ParameterValue::Fixed(actual)] if actual.to_bits() == expected.to_bits()
    ));
}

#[test]
fn zero_controls_emit_original_standard_h_and_ignore_clean_ancillas() {
    let target = Qubit::new(0);
    let operations = decompose_hadamard_n_clean(&[], target, &[target]).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(&operations[0], StandardGate::H, &[target]);
}

#[test]
fn one_control_emits_conditional_phase_and_zy_rotations() {
    let control = Qubit::new(0);
    let target = Qubit::new(1);
    let operations = decompose_hadamard_no_aux(&[control], target).unwrap();

    assert_eq!(operations.len(), 3);
    assert_fixed_parameter(&operations[0], StandardGate::Phase, &[control], PI / 2.0);
    assert_fixed_parameter(&operations[1], StandardGate::CRZ, &[control, target], PI);
    assert_fixed_parameter(
        &operations[2],
        StandardGate::CRY,
        &[control, target],
        PI / 2.0,
    );
}

#[test]
fn no_aux_decompositions_match_mcgate_semantics_exactly() {
    for num_controls in 1..=4 {
        let controls: Vec<_> = (0..num_controls)
            .map(|index| Qubit::new(index as u32))
            .collect();
        let target = Qubit::new(num_controls as u32);
        let actual = circuit_to_matrix(
            &circuit_from_value_operations(
                num_controls + 1,
                decompose_hadamard_no_aux(&controls, target).unwrap(),
            ),
            None,
        )
        .unwrap();
        let expected = mc_hadamard_matrix(num_controls + 1, &controls, target);

        assert_selected_columns_approx_eq(&actual, &expected, 0..expected.ncols());
    }
}

#[test]
fn clean_decomposition_matches_clean_subspace_and_restores_ancillas() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let clean_ancillas = [Qubit::new(4), Qubit::new(5)];
    let actual = circuit_to_matrix(
        &circuit_from_value_operations(
            6,
            decompose_hadamard_n_clean(&controls, target, &clean_ancillas).unwrap(),
        ),
        None,
    )
    .unwrap();
    let expected = mc_hadamard_matrix(6, &controls, target);
    let clean_mask = clean_ancillas
        .iter()
        .fold(0_usize, |mask, qubit| mask | (1 << qubit.index()));
    let clean_columns = (0..expected.ncols()).filter(|state| state & clean_mask == 0);

    assert_selected_columns_approx_eq(&actual, &expected, clean_columns);
}

#[test]
fn underlying_mc_su2_errors_are_propagated_without_rewriting() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let error = decompose_hadamard_n_clean(&controls, Qubit::new(3), &[]).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mc_su2",
            ref reason,
        } if reason
            == "clean-accumulator MC-SU(2) decomposition with 2 controls requires 1 clean ancillas, got 0"
    ));
}

#[test]
fn duplicate_qubit_errors_are_propagated_without_rewriting() {
    let duplicate = Qubit::new(0);
    let error = decompose_hadamard_no_aux(&[duplicate], duplicate).unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mc_su2",
            ..
        }
    ));
}
