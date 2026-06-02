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

use super::{Su2RotationAxis, decompose_mc_su2_n_clean};
use crate::circuit::{
    Circuit, Instruction, MCGate, Parameter, ParameterValue, Qubit, StandardGate,
    circuit_to_matrix, operation::ValueOperation,
};
use crate::compiler::{
    error::CompilerError, transform::decompose::mc_gate::mcx::decompose_mcx_n_clean,
};
use crate::util::test_utils::{
    assert_value_operations_equal, assert_value_operations_only_use_qubits,
    circuit_from_value_operations,
};
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::smallvec;

const EPSILON: f64 = 1e-9;

fn rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::RX,
        Su2RotationAxis::Y => StandardGate::RY,
        Su2RotationAxis::Z => StandardGate::RZ,
    }
}

fn controlled_rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::CRX,
        Su2RotationAxis::Y => StandardGate::CRY,
        Su2RotationAxis::Z => StandardGate::CRZ,
    }
}

fn mc_rotation_matrix(
    num_qubits: usize,
    controls: &[Qubit],
    target: Qubit,
    axis: Su2RotationAxis,
    theta: f64,
) -> Array2<Complex64> {
    let mut circuit = Circuit::new(num_qubits);
    let mut qubits = controls.to_vec();
    qubits.push(target);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(controls.len() as u8, rotation(axis)))),
            qubits,
            [ParameterValue::Fixed(theta)],
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

fn assert_fixed_parameterized_operation(
    operation: &ValueOperation,
    expected_gate: StandardGate,
    expected_qubits: &[Qubit],
    expected_theta: f64,
) {
    assert!(matches!(
        operation.instruction,
        Instruction::Standard(gate) if gate == expected_gate
    ));
    assert_eq!(operation.qubits.as_slice(), expected_qubits);
    assert!(matches!(
        operation.params.as_slice(),
        [ParameterValue::Fixed(theta)] if theta.to_bits() == expected_theta.to_bits()
    ));
    assert!(operation.label.is_none());
}

fn assert_insufficient_ancilla_error(num_controls: usize, clean_ancillas: &[Qubit]) {
    let controls: Vec<_> = (0..num_controls)
        .map(|index| Qubit::new(index as u32))
        .collect();
    let error = decompose_mc_su2_n_clean(
        Su2RotationAxis::Z,
        &ParameterValue::Fixed(0.731),
        &controls,
        Qubit::new(num_controls as u32),
        clean_ancillas,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mc_su2",
            ref reason,
        } if reason
            == &format!(
                "clean-accumulator MC-SU(2) decomposition with {num_controls} controls requires {} clean ancillas, got {}",
                num_controls - 1,
                clean_ancillas.len()
            )
    ));
}

#[test]
fn zero_and_one_controls_do_not_consume_ancillas() {
    let ignored = Qubit::new(1);
    for (controls, target) in [(vec![], Qubit::new(0)), (vec![Qubit::new(0)], ignored)] {
        for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
            let operations = decompose_mc_su2_n_clean(
                axis,
                &ParameterValue::Fixed(0.731),
                &controls,
                target,
                &[ignored],
            )
            .unwrap();

            assert_eq!(operations.len(), 1);
            let gate = if controls.is_empty() {
                rotation(axis)
            } else {
                controlled_rotation(axis)
            };
            let mut qubits = controls.clone();
            qubits.push(target);
            assert_fixed_parameterized_operation(&operations[0], gate, &qubits, 0.731);
        }
    }
}

#[test]
fn two_controls_require_one_clean_accumulator() {
    assert_insufficient_ancilla_error(2, &[]);

    let controls = [Qubit::new(0), Qubit::new(1)];
    let target = Qubit::new(2);
    let accumulator = Qubit::new(3);
    let operations = decompose_mc_su2_n_clean(
        Su2RotationAxis::Y,
        &ParameterValue::Fixed(0.731),
        &controls,
        target,
        &[accumulator],
    )
    .unwrap();

    assert!(
        operations
            .iter()
            .any(|operation| { operation.qubits.contains(&accumulator) })
    );
}

#[test]
fn larger_inputs_require_controls_len_minus_one_clean_ancillas() {
    for num_controls in 3..=6 {
        let clean_ancillas: Vec<_> = (num_controls + 1..2 * num_controls - 1)
            .map(|index| Qubit::new(index as u32))
            .collect();

        assert_insufficient_ancilla_error(num_controls, &clean_ancillas);
    }
}

#[test]
fn duplicate_consumed_qubits_are_rejected() {
    let duplicate = Qubit::new(1);
    let cases = [
        (vec![Qubit::new(0), duplicate, duplicate], Qubit::new(3)),
        (vec![Qubit::new(0), duplicate], duplicate),
    ];
    for (controls, target) in cases {
        let error = decompose_mc_su2_n_clean(
            Su2RotationAxis::Z,
            &ParameterValue::Fixed(0.731),
            &controls,
            target,
            &[Qubit::new(3), Qubit::new(4)],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CompilerError::TransformFailed {
                name: "decompose.mc_su2",
                ref reason,
            } if reason
                == &format!(
                    "MC-SU(2) controls, target, and ancillas must be distinct; duplicate {duplicate}"
                )
        ));
    }

    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let error = decompose_mc_su2_n_clean(
        Su2RotationAxis::Z,
        &ParameterValue::Fixed(0.731),
        &controls,
        target,
        &[Qubit::new(4), target],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CompilerError::TransformFailed {
            name: "decompose.mc_su2",
            ref reason,
        } if reason
            == &format!(
                "MC-SU(2) controls, target, and ancillas must be distinct; duplicate {target}"
            )
    ));

    let consumed = Qubit::new(4);
    for ancillas in [vec![Qubit::new(0), Qubit::new(5)], vec![consumed, consumed]] {
        let duplicate = ancillas[0];
        let error = decompose_mc_su2_n_clean(
            Su2RotationAxis::Z,
            &ParameterValue::Fixed(0.731),
            &controls,
            target,
            &ancillas,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CompilerError::TransformFailed {
                name: "decompose.mc_su2",
                ref reason,
            } if reason
                == &format!(
                    "MC-SU(2) controls, target, and ancillas must be distinct; duplicate {duplicate}"
                )
        ));
    }
}

#[test]
fn extra_ancillas_are_ignored_without_validation_or_use() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let used_ancillas = [Qubit::new(4), Qubit::new(5)];
    let ignored_ancilla = Qubit::new(6);
    let operations = decompose_mc_su2_n_clean(
        Su2RotationAxis::Z,
        &ParameterValue::Fixed(0.731),
        &controls,
        target,
        &[
            used_ancillas[0],
            used_ancillas[1],
            ignored_ancilla,
            ignored_ancilla,
            target,
        ],
    )
    .unwrap();
    let mut allowed_qubits = controls.to_vec();
    allowed_qubits.push(target);
    allowed_qubits.extend(used_ancillas);

    assert_value_operations_only_use_qubits(&operations, &allowed_qubits);
}

#[test]
fn decomposition_has_mcx_controlled_rotation_mcx_structure() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let clean_ancillas = [Qubit::new(4), Qubit::new(5)];
    let accumulator = clean_ancillas[0];
    let mcx = decompose_mcx_n_clean(&controls, accumulator, &clean_ancillas[1..]).unwrap();

    for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
        let theta = ParameterValue::Fixed(0.731);
        let operations =
            decompose_mc_su2_n_clean(axis, &theta, &controls, target, &clean_ancillas).unwrap();
        let mut expected = mcx.clone();
        expected.push(ValueOperation {
            instruction: Instruction::Standard(controlled_rotation(axis)),
            qubits: smallvec![accumulator, target],
            params: smallvec![theta],
            label: None,
        });
        expected.extend(mcx.clone());

        assert_value_operations_equal(&operations, &expected);
    }
}

#[test]
fn clean_subspace_semantics_match_mcgate_and_restore_ancillas() {
    let theta = 0.731;
    for num_controls in 2..=4 {
        let controls: Vec<_> = (0..num_controls)
            .map(|index| Qubit::new(index as u32))
            .collect();
        let target = Qubit::new(num_controls as u32);
        let clean_ancillas: Vec<_> = (num_controls + 1..=2 * num_controls - 1)
            .map(|index| Qubit::new(index as u32))
            .collect();
        let num_qubits = 2 * num_controls;
        let clean_mask = clean_ancillas
            .iter()
            .fold(0_usize, |mask, qubit| mask | (1 << qubit.index()));
        for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
            let operations = decompose_mc_su2_n_clean(
                axis,
                &ParameterValue::Fixed(theta),
                &controls,
                target,
                &clean_ancillas,
            )
            .unwrap();
            let actual =
                circuit_to_matrix(&circuit_from_value_operations(num_qubits, operations), None)
                    .unwrap();
            let expected = mc_rotation_matrix(num_qubits, &controls, target, axis, theta);
            let clean_columns = (0..expected.ncols()).filter(|state| state & clean_mask == 0);

            assert_selected_columns_approx_eq(&actual, &expected, clean_columns);
        }
    }
}

#[test]
fn symbolic_theta_is_passed_unchanged_to_controlled_rotation() {
    let controls = [Qubit::new(0), Qubit::new(1)];
    let target = Qubit::new(2);
    let accumulator = Qubit::new(3);
    let theta = Parameter::symbol("theta");
    let operations = decompose_mc_su2_n_clean(
        Su2RotationAxis::X,
        &ParameterValue::Param(theta.clone()),
        &controls,
        target,
        &[accumulator],
    )
    .unwrap();

    assert!(operations.iter().any(|operation| {
        matches!(
            (&operation.instruction, operation.params.as_slice()),
            (
                Instruction::Standard(StandardGate::CRX),
                [ParameterValue::Param(parameter)]
            ) if parameter == &theta
        )
    }));
}
