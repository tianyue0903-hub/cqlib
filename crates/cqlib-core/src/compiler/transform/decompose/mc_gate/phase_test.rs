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
use crate::circuit::{CircuitParam, MCGate, Qubit, StandardGate, circuit_to_matrix};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

#[test]
fn zero_control_phase_family_emits_base_standard_gate() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (
            StandardGate::S,
            vec![],
            vec![ExpectedParameterizedOperation {
                gate: StandardGate::S,
                qubits: &[0],
                params: &[],
            }],
        ),
        (
            StandardGate::SDG,
            vec![],
            vec![ExpectedParameterizedOperation {
                gate: StandardGate::SDG,
                qubits: &[0],
                params: &[],
            }],
        ),
        (
            StandardGate::T,
            vec![],
            vec![ExpectedParameterizedOperation {
                gate: StandardGate::T,
                qubits: &[0],
                params: &[],
            }],
        ),
        (
            StandardGate::TDG,
            vec![],
            vec![ExpectedParameterizedOperation {
                gate: StandardGate::TDG,
                qubits: &[0],
                params: &[],
            }],
        ),
        (
            StandardGate::Phase,
            vec![CircuitParam::Fixed(0.375)],
            vec![ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[0],
                params: &[0.375],
            }],
        ),
    ];

    for (base_gate, params, expected) in cases {
        let gate = MCGate::new(0, base_gate);
        let operations = decompose_mc_gate(&gate, &[Qubit::new(0)], &params, &config).unwrap();
        assert_parameterized_operation_sequence(&operations, &expected);
    }
}

#[test]
fn zero_control_symbolic_phase_is_copied_without_arithmetic() {
    let gate = MCGate::new(0, StandardGate::Phase);
    let params = [CircuitParam::Index(3)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &[Qubit::new(0)], &params, &config).unwrap();

    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0)]);
    assert!(matches!(
        operations[0].instruction,
        crate::circuit::Instruction::Standard(StandardGate::Phase)
    ));
    assert!(matches!(
        operations[0].params.as_slice(),
        [CircuitParam::Index(3)]
    ));
}

#[test]
fn one_control_phase_uses_exact_controlled_phase_sequence() {
    let gate = MCGate::new(1, StandardGate::Phase);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert_parameterized_operation_sequence(
        &operations,
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[0],
                params: &[0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[1],
                params: &[0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[1],
                params: &[-0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
                params: &[],
            },
        ],
    );
}

#[test]
fn no_ancilla_phase_decompositions_match_original_matrices_strictly() {
    let cases = [
        (0, StandardGate::S, vec![0], vec![]),
        (1, StandardGate::S, vec![0, 1], vec![]),
        (2, StandardGate::S, vec![0, 1, 2], vec![]),
        (0, StandardGate::SDG, vec![0], vec![]),
        (1, StandardGate::SDG, vec![0, 1], vec![]),
        (2, StandardGate::SDG, vec![0, 1, 2], vec![]),
        (0, StandardGate::T, vec![0], vec![]),
        (1, StandardGate::T, vec![0, 1], vec![]),
        (2, StandardGate::T, vec![0, 1, 2], vec![]),
        (0, StandardGate::TDG, vec![0], vec![]),
        (1, StandardGate::TDG, vec![0, 1], vec![]),
        (2, StandardGate::TDG, vec![0, 1, 2], vec![]),
        (
            0,
            StandardGate::Phase,
            vec![0],
            vec![CircuitParam::Fixed(0.37)],
        ),
        (
            1,
            StandardGate::Phase,
            vec![0, 1],
            vec![CircuitParam::Fixed(0.37)],
        ),
        (
            2,
            StandardGate::Phase,
            vec![0, 1, 2],
            vec![CircuitParam::Fixed(0.37)],
        ),
    ];

    for (added_controls, base_gate, qubits, params) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let config = McGateDecomposeConfig::default();
        let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

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
fn clean_mode_uses_clean_flag_for_multi_controlled_phase() {
    let gate = MCGate::new(2, StandardGate::Phase);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(3)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert_parameterized_operation_sequence(
        &operations,
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 3],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[3],
                params: &[0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[2],
                params: &[0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CX,
                qubits: &[3, 2],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::Phase,
                qubits: &[2],
                params: &[-0.25],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CX,
                qubits: &[3, 2],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 3],
                params: &[],
            },
        ],
    );
}

#[test]
fn clean_mode_matches_original_on_clean_ancilla_zero_subspace() {
    let gate = MCGate::new(3, StandardGate::S);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4), Qubit::new(5)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    let actual = circuit_to_matrix(&circuit_from_operations(6, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(6, gate, &qubits, &[]), None).unwrap();
    assert_columns_eq_for_fixed_qubit_inputs(
        &actual,
        &expected,
        &[(Qubit::new(4), 0), (Qubit::new(5), 0)],
        1e-9,
    );
}

#[test]
fn fixed_discrete_phase_angles_are_used_for_controlled_gates() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (StandardGate::S, FRAC_PI_2 / 2.0),
        (StandardGate::SDG, -FRAC_PI_2 / 2.0),
        (StandardGate::T, FRAC_PI_4 / 2.0),
        (StandardGate::TDG, -FRAC_PI_4 / 2.0),
    ];

    for (base_gate, expected_half_angle) in cases {
        let gate = MCGate::new(1, base_gate);
        let qubits = [Qubit::new(0), Qubit::new(1)];
        let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

        assert_parameterized_standard_operation(
            &operations[0],
            StandardGate::Phase,
            &[Qubit::new(0)],
            &[expected_half_angle],
        );
        assert_parameterized_standard_operation(
            &operations[1],
            StandardGate::Phase,
            &[Qubit::new(1)],
            &[expected_half_angle],
        );
        assert_parameterized_standard_operation(
            &operations[3],
            StandardGate::Phase,
            &[Qubit::new(1)],
            &[-expected_half_angle],
        );
    }
}

#[test]
fn controlled_phase_rejects_symbolic_parameter() {
    let gate = MCGate::new(1, StandardGate::Phase);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Index(0)];
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "symbolic Phase-family parameters");
}

#[test]
fn dirty_mode_rejects_controlled_phase_family() {
    let gate = MCGate::new(1, StandardGate::S);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "dirty-ancilla Phase decomposition is not supported");
}

#[test]
fn clean_mode_rejects_insufficient_clean_ancillas_for_phase() {
    let gate = MCGate::new(3, StandardGate::S);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4)],
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "requires 2 clean ancillas");
}

#[test]
fn phase_budget_accounts_for_controlled_phase_expansion() {
    let gate = MCGate::new(1, StandardGate::S);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 4,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 5 operations");
}

#[test]
fn no_ancilla_rejects_exponential_phase_above_cap() {
    let gate = MCGate::new(10, StandardGate::S);
    let qubits: Vec<_> = (0..11).map(Qubit::new).collect();
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would be exponential");
}
