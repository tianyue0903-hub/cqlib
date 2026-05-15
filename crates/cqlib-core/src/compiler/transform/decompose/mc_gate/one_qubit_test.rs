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
    assert_parameterized_operation_sequence, assert_parameterized_standard_operation,
    assert_transform_failed_contains, circuit_from_operations, original_mc_gate_circuit,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Qubit, StandardGate, circuit_to_matrix};
use std::f64::consts::{FRAC_PI_2, PI};

#[test]
fn zero_control_one_qubit_family_emits_base_standard_gate() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (StandardGate::H, vec![]),
        (
            StandardGate::U,
            vec![
                CircuitParam::Fixed(0.31),
                CircuitParam::Fixed(-0.42),
                CircuitParam::Fixed(0.53),
            ],
        ),
        (
            StandardGate::RXY,
            vec![CircuitParam::Fixed(0.61), CircuitParam::Fixed(-0.27)],
        ),
        (StandardGate::XY, vec![CircuitParam::Fixed(0.44)]),
    ];

    for (base_gate, params) in cases {
        let gate = MCGate::new(0, base_gate);
        let operations = decompose_mc_gate(&gate, &[Qubit::new(0)], &params, &config).unwrap();
        let fixed_params: Vec<_> = params
            .iter()
            .map(|param| match param {
                CircuitParam::Fixed(value) => *value,
                CircuitParam::Index(_) => panic!("test case uses fixed parameters"),
            })
            .collect();

        assert_eq!(operations.len(), 1);
        assert_parameterized_standard_operation(
            &operations[0],
            base_gate,
            &[Qubit::new(0)],
            &fixed_params,
        );
    }
}

#[test]
fn zero_control_symbolic_one_qubit_parameters_are_copied() {
    let gate = MCGate::new(0, StandardGate::U);
    let params = [
        CircuitParam::Index(1),
        CircuitParam::Index(2),
        CircuitParam::Index(3),
    ];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &[Qubit::new(0)], &params, &config).unwrap();

    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::U)
    ));
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0)]);
    assert!(matches!(
        operations[0].params.as_slice(),
        [
            CircuitParam::Index(1),
            CircuitParam::Index(2),
            CircuitParam::Index(3)
        ]
    ));
}

#[test]
fn one_control_square_root_gates_reuse_controlled_rotation_boundaries() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (
            StandardGate::X2P,
            ExpectedParameterizedOperation {
                gate: StandardGate::CRX,
                qubits: &[0, 1],
                params: &[FRAC_PI_2],
            },
        ),
        (
            StandardGate::X2M,
            ExpectedParameterizedOperation {
                gate: StandardGate::CRX,
                qubits: &[0, 1],
                params: &[-FRAC_PI_2],
            },
        ),
        (
            StandardGate::Y2P,
            ExpectedParameterizedOperation {
                gate: StandardGate::CRY,
                qubits: &[0, 1],
                params: &[FRAC_PI_2],
            },
        ),
        (
            StandardGate::Y2M,
            ExpectedParameterizedOperation {
                gate: StandardGate::CRY,
                qubits: &[0, 1],
                params: &[-FRAC_PI_2],
            },
        ),
    ];

    for (base_gate, expected) in cases {
        let gate = MCGate::new(1, base_gate);
        let operations =
            decompose_mc_gate(&gate, &[Qubit::new(0), Qubit::new(1)], &[], &config).unwrap();

        assert_parameterized_operation_sequence(&operations, &[expected]);
    }
}

#[test]
fn controlled_h_emits_conditional_global_phase() {
    let gate = MCGate::new(1, StandardGate::H);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_parameterized_operation_sequence(
        &operations,
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[0],
                params: &[FRAC_PI_2],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CRZ,
                qubits: &[0, 1],
                params: &[PI],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CRY,
                qubits: &[0, 1],
                params: &[FRAC_PI_2],
            },
        ],
    );
}

#[test]
fn one_qubit_family_decompositions_match_original_matrices_strictly() {
    let cases = [
        (1, StandardGate::H, vec![0, 1], vec![]),
        (2, StandardGate::H, vec![0, 1, 2], vec![]),
        (
            1,
            StandardGate::U,
            vec![0, 1],
            vec![
                CircuitParam::Fixed(0.73),
                CircuitParam::Fixed(-0.42),
                CircuitParam::Fixed(1.15),
            ],
        ),
        (
            2,
            StandardGate::U,
            vec![0, 1, 2],
            vec![
                CircuitParam::Fixed(-0.51),
                CircuitParam::Fixed(0.38),
                CircuitParam::Fixed(-1.07),
            ],
        ),
        (
            1,
            StandardGate::RXY,
            vec![0, 1],
            vec![CircuitParam::Fixed(0.61), CircuitParam::Fixed(-0.33)],
        ),
        (
            1,
            StandardGate::XY,
            vec![0, 1],
            vec![CircuitParam::Fixed(0.44)],
        ),
        (
            1,
            StandardGate::XY2P,
            vec![0, 1],
            vec![CircuitParam::Fixed(-0.27)],
        ),
        (
            1,
            StandardGate::XY2M,
            vec![0, 1],
            vec![CircuitParam::Fixed(0.52)],
        ),
        (2, StandardGate::X2P, vec![0, 1, 2], vec![]),
        (2, StandardGate::X2M, vec![0, 1, 2], vec![]),
        (2, StandardGate::Y2P, vec![0, 1, 2], vec![]),
        (2, StandardGate::Y2M, vec![0, 1, 2], vec![]),
    ];

    for (added_controls, base_gate, qubits, params) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
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
fn clean_mode_one_qubit_family_matches_original_on_clean_ancilla_zero_subspace() {
    let gate = MCGate::new(2, StandardGate::H);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(3)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();
    let actual = circuit_to_matrix(&circuit_from_operations(4, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(4, gate, &qubits, &[]), None).unwrap();

    assert_columns_eq_for_fixed_qubit_inputs(&actual, &expected, &[(Qubit::new(3), 0)], 1e-9);
}

#[test]
fn controlled_symbolic_one_qubit_parameters_are_rejected() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (
            StandardGate::U,
            vec![
                CircuitParam::Index(1),
                CircuitParam::Index(2),
                CircuitParam::Index(3),
            ],
        ),
        (
            StandardGate::RXY,
            vec![CircuitParam::Index(4), CircuitParam::Index(5)],
        ),
    ];

    for (base_gate, params) in cases {
        let gate = MCGate::new(1, base_gate);
        let err = decompose_mc_gate(&gate, &[Qubit::new(0), Qubit::new(1)], &params, &config)
            .unwrap_err();

        assert_transform_failed_contains(err, "symbolic OneQubit-family parameters");
    }
}

#[test]
fn dirty_mode_rejects_controlled_one_qubit_family() {
    let gate = MCGate::new(1, StandardGate::H);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "dirty-ancilla OneQubit decomposition is not supported");
}

#[test]
fn one_qubit_family_budget_accounts_for_conditional_phase_and_rotations() {
    let gate = MCGate::new(1, StandardGate::H);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 2,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 3 operations");
}
