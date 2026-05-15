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

use super::decompose::{AncillaMode, McGateDecomposeConfig, decompose_mc_gate};
use super::test_utils::{
    ExpectedParameterizedOperation, assert_columns_eq_for_fixed_qubit_inputs, assert_matrix_eq,
    assert_parameterized_operation_sequence, assert_transform_failed_contains,
    circuit_from_operations, original_mc_gate_circuit,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Qubit, StandardGate, circuit_to_matrix};

#[test]
fn zero_and_one_control_rx_ry_boundaries_emit_standard_gates() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (
            0,
            StandardGate::RX,
            vec![0],
            ExpectedParameterizedOperation {
                gate: StandardGate::RX,
                qubits: &[0],
                params: &[0.5],
            },
        ),
        (
            1,
            StandardGate::RX,
            vec![0, 1],
            ExpectedParameterizedOperation {
                gate: StandardGate::CRX,
                qubits: &[0, 1],
                params: &[0.5],
            },
        ),
        (
            0,
            StandardGate::CRX,
            vec![0, 1],
            ExpectedParameterizedOperation {
                gate: StandardGate::CRX,
                qubits: &[0, 1],
                params: &[0.5],
            },
        ),
        (
            0,
            StandardGate::RY,
            vec![0],
            ExpectedParameterizedOperation {
                gate: StandardGate::RY,
                qubits: &[0],
                params: &[0.5],
            },
        ),
        (
            1,
            StandardGate::RY,
            vec![0, 1],
            ExpectedParameterizedOperation {
                gate: StandardGate::CRY,
                qubits: &[0, 1],
                params: &[0.5],
            },
        ),
        (
            0,
            StandardGate::CRY,
            vec![0, 1],
            ExpectedParameterizedOperation {
                gate: StandardGate::CRY,
                qubits: &[0, 1],
                params: &[0.5],
            },
        ),
    ];

    for (added_controls, base_gate, qubits, expected) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let operations =
            decompose_mc_gate(&gate, &qubits, &[CircuitParam::Fixed(0.5)], &config).unwrap();

        assert_parameterized_operation_sequence(&operations, &[expected]);
    }
}

#[test]
fn direct_rx_ry_paths_copy_symbolic_parameter_indices() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (0, StandardGate::RX, vec![0], StandardGate::RX, 3),
        (1, StandardGate::RX, vec![0, 1], StandardGate::CRX, 5),
        (0, StandardGate::RY, vec![0], StandardGate::RY, 7),
        (1, StandardGate::RY, vec![0, 1], StandardGate::CRY, 9),
    ];

    for (added_controls, base_gate, qubits, expected_gate, expected_index) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let operations = decompose_mc_gate(
            &gate,
            &qubits,
            &[CircuitParam::Index(expected_index)],
            &config,
        )
        .unwrap();

        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(actual) if actual == expected_gate
        ));
        assert!(matches!(
            operations[0].params.as_slice(),
            [CircuitParam::Index(actual)] if *actual == expected_index
        ));
    }
}

#[test]
fn multi_control_rx_ry_decompositions_match_original_matrices_strictly() {
    let cases = [
        (0, StandardGate::RX, vec![0], 0.37),
        (1, StandardGate::RX, vec![0, 1], -0.41),
        (2, StandardGate::RX, vec![0, 1, 2], 0.73),
        (0, StandardGate::CRX, vec![0, 1], -0.29),
        (1, StandardGate::CRX, vec![0, 1, 2], 0.61),
        (2, StandardGate::CRX, vec![0, 1, 2, 3], -0.83),
        (0, StandardGate::RY, vec![0], 0.43),
        (1, StandardGate::RY, vec![0, 1], -0.47),
        (2, StandardGate::RY, vec![0, 1, 2], 0.79),
        (0, StandardGate::CRY, vec![0, 1], -0.31),
        (1, StandardGate::CRY, vec![0, 1, 2], 0.67),
        (2, StandardGate::CRY, vec![0, 1, 2, 3], -0.89),
    ];

    for (added_controls, base_gate, qubits, theta) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let params = [CircuitParam::Fixed(theta)];
        let config = McGateDecomposeConfig::default();
        let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();
        assert!(
            operations
                .iter()
                .all(|operation| matches!(operation.instruction, Instruction::Standard(_)))
        );

        let num_qubits = qubits.len();
        let actual =
            circuit_to_matrix(&circuit_from_operations(num_qubits, operations), None).unwrap();
        let expected = circuit_to_matrix(
            &original_mc_gate_circuit(num_qubits, gate, &qubits, &params),
            None,
        )
        .unwrap();
        assert_matrix_eq(&actual, &expected, 1e-9);
    }
}

#[test]
fn multi_control_basis_change_wraps_lifted_rz_path() {
    let config = McGateDecomposeConfig::default();
    let rx_operations = decompose_mc_gate(
        &MCGate::new(1, StandardGate::CRX),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        &[CircuitParam::Fixed(0.5)],
        &config,
    )
    .unwrap();

    assert!(matches!(
        rx_operations
            .first()
            .map(|operation| operation.instruction.clone()),
        Some(Instruction::Standard(StandardGate::H))
    ));
    assert!(matches!(
        rx_operations
            .last()
            .map(|operation| operation.instruction.clone()),
        Some(Instruction::Standard(StandardGate::H))
    ));
    assert_eq!(
        rx_operations.first().unwrap().qubits.as_slice(),
        &[Qubit::new(2)]
    );
    assert_eq!(
        rx_operations.last().unwrap().qubits.as_slice(),
        &[Qubit::new(2)]
    );

    let ry_operations = decompose_mc_gate(
        &MCGate::new(1, StandardGate::CRY),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        &[CircuitParam::Fixed(0.5)],
        &config,
    )
    .unwrap();

    assert_parameterized_operation_sequence(
        &[
            ry_operations.first().unwrap().clone(),
            ry_operations.last().unwrap().clone(),
        ],
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::RX,
                qubits: &[2],
                params: &[std::f64::consts::FRAC_PI_2],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::RX,
                qubits: &[2],
                params: &[-std::f64::consts::FRAC_PI_2],
            },
        ],
    );
}

#[test]
fn clean_mode_rx_ry_matches_original_on_clean_ancilla_zero_subspace() {
    let cases = [(StandardGate::RX, 0.5), (StandardGate::RY, -0.7)];

    for (base_gate, theta) in cases {
        let gate = MCGate::new(2, base_gate);
        let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
        let params = [CircuitParam::Fixed(theta)];
        let config = McGateDecomposeConfig {
            ancilla_mode: AncillaMode::CleanAncilla,
            clean_ancillas: vec![Qubit::new(3)],
            ..McGateDecomposeConfig::default()
        };

        let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();
        let actual = circuit_to_matrix(&circuit_from_operations(4, operations), None).unwrap();
        let expected =
            circuit_to_matrix(&original_mc_gate_circuit(4, gate, &qubits, &params), None).unwrap();

        assert_columns_eq_for_fixed_qubit_inputs(&actual, &expected, &[(Qubit::new(3), 0)], 1e-9);
    }
}

#[test]
fn multi_control_rx_ry_rejects_symbolic_parameter_arithmetic() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (2, StandardGate::RX, vec![0, 1, 2]),
        (2, StandardGate::RY, vec![0, 1, 2]),
        (1, StandardGate::CRX, vec![0, 1, 2]),
        (1, StandardGate::CRY, vec![0, 1, 2]),
    ];

    for (added_controls, base_gate, qubits) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let err =
            decompose_mc_gate(&gate, &qubits, &[CircuitParam::Index(0)], &config).unwrap_err();

        assert_transform_failed_contains(err, "symbolic RZ-family parameters require theta/2");
    }
}

#[test]
fn dirty_mode_rejects_multi_control_rx_ry() {
    let gate = MCGate::new(2, StandardGate::RX);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "dirty-ancilla RX/RY decomposition is not supported");
}

#[test]
fn rx_ry_budget_accounts_for_basis_change_and_lifted_rz() {
    let gate = MCGate::new(2, StandardGate::RY);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 23,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 24 operations");
}
