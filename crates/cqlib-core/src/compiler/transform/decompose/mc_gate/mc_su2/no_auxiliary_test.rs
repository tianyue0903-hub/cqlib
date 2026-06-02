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

use super::{Su2RotationAxis, decompose_mc_su2_no_aux};
use crate::circuit::{
    Circuit, Instruction, MCGate, Parameter, ParameterValue, Qubit, StandardGate,
    circuit_to_matrix, operation::ValueOperation,
};
use crate::compiler::error::CompilerError;
use crate::util::test_utils::{
    assert_standard_operation, assert_value_operations_only_use_qubits,
    circuit_from_value_operations,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

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

fn assert_matrix_approx_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), expected_amplitude) in expected.indexed_iter() {
        assert!(
            (actual[[row, column]] - expected_amplitude).norm() < EPSILON,
            "matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
            actual[[row, column]]
        );
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

#[test]
fn zero_controls_emit_standard_rotations() {
    let target = Qubit::new(0);
    for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
        let operations =
            decompose_mc_su2_no_aux(axis, &ParameterValue::Fixed(0.75), &[], target).unwrap();

        assert_eq!(operations.len(), 1);
        assert_fixed_parameterized_operation(&operations[0], rotation(axis), &[target], 0.75);
    }
}

#[test]
fn one_control_emits_standard_controlled_rotations() {
    let control = Qubit::new(0);
    let target = Qubit::new(1);
    for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
        let operations =
            decompose_mc_su2_no_aux(axis, &ParameterValue::Fixed(0.75), &[control], target)
                .unwrap();

        assert_eq!(operations.len(), 1);
        assert_fixed_parameterized_operation(
            &operations[0],
            controlled_rotation(axis),
            &[control, target],
            0.75,
        );
    }
}

#[test]
fn two_control_rz_emits_exact_vale_sequence() {
    let first = Qubit::new(0);
    let second = Qubit::new(1);
    let target = Qubit::new(2);
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::Z,
        &ParameterValue::Fixed(0.8),
        &[first, second],
        target,
    )
    .unwrap();

    assert_eq!(operations.len(), 8);
    assert_standard_operation(&operations[0], StandardGate::CX, &[first, target]);
    assert_fixed_parameterized_operation(&operations[1], StandardGate::RZ, &[target], -0.2);
    assert_standard_operation(&operations[2], StandardGate::CX, &[second, target]);
    assert_fixed_parameterized_operation(&operations[3], StandardGate::RZ, &[target], 0.2);
    assert_standard_operation(&operations[4], StandardGate::CX, &[first, target]);
    assert_fixed_parameterized_operation(&operations[5], StandardGate::RZ, &[target], -0.2);
    assert_standard_operation(&operations[6], StandardGate::CX, &[second, target]);
    assert_fixed_parameterized_operation(&operations[7], StandardGate::RZ, &[target], 0.2);
}

#[test]
fn two_control_rx_uses_h_conjugation_and_internal_rz_rotations() {
    let controls = [Qubit::new(0), Qubit::new(1)];
    let target = Qubit::new(2);
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::X,
        &ParameterValue::Fixed(0.8),
        &controls,
        target,
    )
    .unwrap();

    assert_eq!(operations.len(), 10);
    assert_standard_operation(&operations[0], StandardGate::H, &[target]);
    assert_standard_operation(&operations[1], StandardGate::CX, &[controls[0], target]);
    assert_fixed_parameterized_operation(&operations[2], StandardGate::RZ, &[target], -0.2);
    assert_standard_operation(&operations[3], StandardGate::CX, &[controls[1], target]);
    assert_fixed_parameterized_operation(&operations[4], StandardGate::RZ, &[target], 0.2);
    assert_standard_operation(&operations[5], StandardGate::CX, &[controls[0], target]);
    assert_fixed_parameterized_operation(&operations[6], StandardGate::RZ, &[target], -0.2);
    assert_standard_operation(&operations[7], StandardGate::CX, &[controls[1], target]);
    assert_fixed_parameterized_operation(&operations[8], StandardGate::RZ, &[target], 0.2);
    assert_standard_operation(&operations[9], StandardGate::H, &[target]);
}

#[test]
fn two_control_ry_emits_exact_vale_sequence() {
    let first = Qubit::new(0);
    let second = Qubit::new(1);
    let target = Qubit::new(2);
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::Y,
        &ParameterValue::Fixed(0.8),
        &[first, second],
        target,
    )
    .unwrap();

    assert_eq!(operations.len(), 8);
    assert_standard_operation(&operations[0], StandardGate::CX, &[first, target]);
    assert_fixed_parameterized_operation(&operations[1], StandardGate::RY, &[target], -0.2);
    assert_standard_operation(&operations[2], StandardGate::CX, &[second, target]);
    assert_fixed_parameterized_operation(&operations[3], StandardGate::RY, &[target], 0.2);
    assert_standard_operation(&operations[4], StandardGate::CX, &[first, target]);
    assert_fixed_parameterized_operation(&operations[5], StandardGate::RY, &[target], -0.2);
    assert_standard_operation(&operations[6], StandardGate::CX, &[second, target]);
    assert_fixed_parameterized_operation(&operations[7], StandardGate::RY, &[target], 0.2);
}

#[test]
fn decompositions_match_mcgate_rotation_matrices() {
    let theta = 0.731;
    for num_controls in 2..=5 {
        let controls: Vec<_> = (0..num_controls)
            .map(|index| Qubit::new(index as u32))
            .collect();
        let target = Qubit::new(num_controls as u32);
        for axis in [Su2RotationAxis::X, Su2RotationAxis::Y, Su2RotationAxis::Z] {
            let operations =
                decompose_mc_su2_no_aux(axis, &ParameterValue::Fixed(theta), &controls, target)
                    .unwrap();
            let actual = circuit_to_matrix(
                &circuit_from_value_operations(num_controls + 1, operations),
                None,
            )
            .unwrap();
            let expected = mc_rotation_matrix(num_controls + 1, &controls, target, axis, theta);

            assert_matrix_approx_eq(&actual, &expected);
        }
    }
}

#[test]
fn decomposition_preserves_superposed_control_semantics() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::Y,
        &ParameterValue::Fixed(0.731),
        &controls,
        target,
    )
    .unwrap();
    let actual = circuit_to_matrix(&circuit_from_value_operations(4, operations), None).unwrap();
    let expected = mc_rotation_matrix(4, &controls, target, Su2RotationAxis::Y, 0.731);
    let amplitude = Complex64::new(1.0 / 8.0_f64.sqrt(), 0.0);
    let initial = Array1::from(
        (0..16)
            .map(|state| {
                if state & (1 << target.index()) == 0 {
                    amplitude
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect::<Vec<_>>(),
    );

    let actual_output = actual.dot(&initial);
    let expected_output = expected.dot(&initial);
    for (actual_amplitude, expected_amplitude) in actual_output.iter().zip(expected_output) {
        assert!((*actual_amplitude - expected_amplitude).norm() < EPSILON);
    }
}

#[test]
fn decomposition_uses_only_controls_and_target() {
    let controls = [
        Qubit::new(0),
        Qubit::new(1),
        Qubit::new(2),
        Qubit::new(3),
        Qubit::new(4),
    ];
    let target = Qubit::new(5);
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::Z,
        &ParameterValue::Fixed(0.731),
        &controls,
        target,
    )
    .unwrap();
    let mut allowed_qubits = controls.to_vec();
    allowed_qubits.push(target);

    assert_value_operations_only_use_qubits(&operations, &allowed_qubits);
}

#[test]
fn duplicate_input_qubits_are_rejected() {
    let duplicate = Qubit::new(1);
    for (controls, target) in [
        (vec![Qubit::new(0), duplicate, duplicate], Qubit::new(3)),
        (vec![Qubit::new(0), duplicate], duplicate),
    ] {
        let error = decompose_mc_su2_no_aux(
            Su2RotationAxis::Z,
            &ParameterValue::Fixed(0.731),
            &controls,
            target,
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
fn symbolic_theta_is_preserved_as_quarter_scaled_expressions() {
    let controls = [Qubit::new(0), Qubit::new(1)];
    let target = Qubit::new(2);
    let theta = Parameter::symbol("theta");
    let operations = decompose_mc_su2_no_aux(
        Su2RotationAxis::Z,
        &ParameterValue::Param(theta.clone()),
        &controls,
        target,
    )
    .unwrap();

    let expected = [
        theta.clone() * -0.25,
        theta.clone() * 0.25,
        theta.clone() * -0.25,
        theta * 0.25,
    ];
    for (operation, expected_parameter) in operations.iter().skip(1).step_by(2).zip(expected) {
        assert!(matches!(
            operation.params.as_slice(),
            [ParameterValue::Param(parameter)] if parameter == &expected_parameter
        ));
    }
}

#[test]
fn operation_count_grows_linearly_for_large_inputs() {
    let mut counts = vec![];
    for num_controls in 6..=12 {
        let controls: Vec<_> = (0..num_controls)
            .map(|index| Qubit::new(index as u32))
            .collect();
        let operations = decompose_mc_su2_no_aux(
            Su2RotationAxis::Z,
            &ParameterValue::Fixed(0.731),
            &controls,
            Qubit::new(num_controls as u32),
        )
        .unwrap();

        counts.push(operations.len());
    }

    // Four balanced dirty-V-chain MCX expansions add a constant amount of
    // work for each additional control once both partitions are nontrivial.
    for counts in counts.windows(2) {
        assert_eq!(counts[1] - counts[0], 40);
    }
}
