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
use crate::circuit::{
    Circuit, CircuitParam, Instruction, MCGate, Operation, ParameterValue, Qubit, StandardGate,
    circuit_to_matrix,
};
use crate::compiler::error::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;

#[derive(Clone, Copy)]
struct ExpectedOperation {
    gate: StandardGate,
    qubits: &'static [u32],
}

fn assert_standard_operation(operation: &Operation, gate: StandardGate, qubits: &[Qubit]) {
    assert!(matches!(operation.instruction, Instruction::Standard(actual) if actual == gate));
    assert_eq!(operation.qubits.as_slice(), qubits);
    assert!(operation.params.is_empty());
    assert!(operation.label.is_none());
}

fn assert_transform_failed_contains(err: CompilerError, expected: &str) {
    assert!(
        matches!(
            err,
            CompilerError::TransformFailed { ref reason, .. } if reason.contains(expected)
        ),
        "expected TransformFailed reason containing {expected:?}, got {err:?}"
    );
}

fn assert_operation_sequence(operations: &[Operation], expected: &[ExpectedOperation]) {
    assert_eq!(operations.len(), expected.len());
    for (operation, expected) in operations.iter().zip(expected) {
        let qubits: Vec<_> = expected.qubits.iter().copied().map(Qubit::new).collect();
        assert_standard_operation(operation, expected.gate, &qubits);
    }
}

fn circuit_from_operations(num_qubits: usize, operations: Vec<Operation>) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    for operation in operations {
        let Operation {
            instruction,
            qubits,
            params,
            label,
        } = operation;
        let params = params.into_iter().map(|param| match param {
            CircuitParam::Fixed(value) => ParameterValue::Fixed(value),
            CircuitParam::Index(_) => panic!("Pauli decomposition tests expect fixed parameters"),
        });
        circuit
            .append(instruction, qubits, params, label.as_deref())
            .unwrap();
    }
    circuit
}

fn original_mc_gate_circuit(gate: MCGate, qubits: &[Qubit]) -> Circuit {
    let num_qubits = max_qubit_count(qubits);
    let mut circuit = Circuit::new(num_qubits);
    circuit
        .append(
            Instruction::McGate(Box::new(gate)),
            qubits.iter().copied(),
            [],
            None,
        )
        .unwrap();
    circuit
}

fn max_qubit_count(qubits: &[Qubit]) -> usize {
    qubits
        .iter()
        .map(|qubit| qubit.index() as usize + 1)
        .max()
        .unwrap_or(0)
}

fn assert_matrix_eq_up_to_global_phase(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    eps: f64,
) {
    assert_eq!(actual.shape(), expected.shape());

    let mut actual_norm_sq = 0.0;
    let mut expected_norm_sq = 0.0;
    let mut inner = Complex64::new(0.0, 0.0);
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        actual_norm_sq += actual_value.norm_sqr();
        expected_norm_sq += expected_value.norm_sqr();
        inner += expected_value.conj() * actual_value;
    }

    let phase_invariant_frobenius = (actual_norm_sq + expected_norm_sq - 2.0 * inner.norm())
        .max(0.0_f64)
        .sqrt();
    assert!(
        phase_invariant_frobenius < eps,
        "matrices differ beyond global phase: phase-invariant Frobenius residual {phase_invariant_frobenius}"
    );
}

fn assert_pauli_decomposition_matches_original(gate: MCGate, qubits: &[Qubit]) {
    let config = McGateDecomposeConfig::default();
    let operations = decompose_mc_gate(&gate, qubits, &[], &config).unwrap();
    assert!(
        operations
            .iter()
            .all(|operation| matches!(operation.instruction, Instruction::Standard(_)))
    );

    let num_qubits = max_qubit_count(qubits);
    let actual = circuit_to_matrix(&circuit_from_operations(num_qubits, operations), None).unwrap();
    let expected = circuit_to_matrix(&original_mc_gate_circuit(gate, qubits), None).unwrap();

    assert_matrix_eq_up_to_global_phase(&actual, &expected, 1e-9);
}

#[test]
fn zero_added_control_pauli_boundaries_emit_expected_standard_sequences() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        (
            StandardGate::X,
            vec![Qubit::new(0)],
            vec![ExpectedOperation {
                gate: StandardGate::X,
                qubits: &[0],
            }],
        ),
        (
            StandardGate::Y,
            vec![Qubit::new(0)],
            vec![
                ExpectedOperation {
                    gate: StandardGate::SDG,
                    qubits: &[0],
                },
                ExpectedOperation {
                    gate: StandardGate::X,
                    qubits: &[0],
                },
                ExpectedOperation {
                    gate: StandardGate::S,
                    qubits: &[0],
                },
            ],
        ),
        (
            StandardGate::Z,
            vec![Qubit::new(0)],
            vec![
                ExpectedOperation {
                    gate: StandardGate::H,
                    qubits: &[0],
                },
                ExpectedOperation {
                    gate: StandardGate::X,
                    qubits: &[0],
                },
                ExpectedOperation {
                    gate: StandardGate::H,
                    qubits: &[0],
                },
            ],
        ),
        (
            StandardGate::CX,
            vec![Qubit::new(0), Qubit::new(1)],
            vec![ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
            }],
        ),
        (
            StandardGate::CY,
            vec![Qubit::new(0), Qubit::new(1)],
            vec![
                ExpectedOperation {
                    gate: StandardGate::SDG,
                    qubits: &[1],
                },
                ExpectedOperation {
                    gate: StandardGate::CX,
                    qubits: &[0, 1],
                },
                ExpectedOperation {
                    gate: StandardGate::S,
                    qubits: &[1],
                },
            ],
        ),
        (
            StandardGate::CZ,
            vec![Qubit::new(0), Qubit::new(1)],
            vec![
                ExpectedOperation {
                    gate: StandardGate::H,
                    qubits: &[1],
                },
                ExpectedOperation {
                    gate: StandardGate::CX,
                    qubits: &[0, 1],
                },
                ExpectedOperation {
                    gate: StandardGate::H,
                    qubits: &[1],
                },
            ],
        ),
        (
            StandardGate::CCX,
            vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            vec![ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
            }],
        ),
    ];

    for (base_gate, qubits, expected) in cases {
        let gate = MCGate::new(0, base_gate);
        let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();
        assert_operation_sequence(&operations, &expected);
    }
}

#[test]
fn no_ancilla_pauli_decompositions_match_original_matrices() {
    let cases = [
        (0, StandardGate::X, vec![0]),
        (0, StandardGate::Y, vec![0]),
        (0, StandardGate::Z, vec![0]),
        (0, StandardGate::CX, vec![0, 1]),
        (0, StandardGate::CY, vec![0, 1]),
        (0, StandardGate::CZ, vec![0, 1]),
        (0, StandardGate::CCX, vec![0, 1, 2]),
        (1, StandardGate::X, vec![0, 1]),
        (2, StandardGate::X, vec![0, 1, 2]),
        (1, StandardGate::Y, vec![0, 1]),
        (1, StandardGate::Z, vec![0, 1]),
        (1, StandardGate::CX, vec![0, 1, 2]),
        (1, StandardGate::CY, vec![0, 1, 2]),
        (1, StandardGate::CZ, vec![0, 1, 2]),
        (1, StandardGate::CCX, vec![0, 1, 2, 3]),
    ];

    for (added_controls, base_gate, qubits) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        assert_pauli_decomposition_matches_original(gate, &qubits);
    }
}

#[test]
fn x_with_one_added_control_emits_cx() {
    let gate = MCGate::new(1, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[ExpectedOperation {
            gate: StandardGate::CX,
            qubits: &[0, 1],
        }],
    );
}

#[test]
fn base_cx_inherent_control_is_included_in_mcx_controls() {
    let gate = MCGate::new(1, StandardGate::CX);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[ExpectedOperation {
            gate: StandardGate::CCX,
            qubits: &[0, 1, 2],
        }],
    );
}

#[test]
fn base_cy_uses_target_basis_conjugation_with_inherent_control() {
    let gate = MCGate::new(1, StandardGate::CY);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::SDG,
                qubits: &[2],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
            },
            ExpectedOperation {
                gate: StandardGate::S,
                qubits: &[2],
            },
        ],
    );
}

#[test]
fn base_cz_uses_target_basis_conjugation_with_inherent_control() {
    let gate = MCGate::new(1, StandardGate::CZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::H,
                qubits: &[2],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
            },
            ExpectedOperation {
                gate: StandardGate::H,
                qubits: &[2],
            },
        ],
    );
}

#[test]
fn base_ccx_inherent_controls_use_clean_mcx_path() {
    let gate = MCGate::new(1, StandardGate::CCX);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 2, 3],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
        ],
    );
}

#[test]
fn dirty_mode_uses_borrowed_qubit_for_large_pauli_mcx() {
    let gate = MCGate::new(3, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        dirty_ancillas: vec![Qubit::new(4)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 2, 3],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 2, 3],
            },
        ],
    );
}

#[test]
fn pauli_budget_accounts_for_basis_conjugation() {
    let gate = MCGate::new(1, StandardGate::CY);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 2,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 3 operations");
}

#[test]
fn clean_mode_rejects_insufficient_clean_ancillas() {
    let gate = MCGate::new(3, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "requires 1 clean ancillas");
}

#[test]
fn dirty_mode_requires_borrowed_qubit_for_large_pauli_mcx() {
    let gate = MCGate::new(3, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "requires one dirty ancilla");
}

#[test]
fn dirty_mode_respects_recursion_depth_limit() {
    let gate = MCGate::new(4, StandardGate::X);
    let qubits = [
        Qubit::new(0),
        Qubit::new(1),
        Qubit::new(2),
        Qubit::new(3),
        Qubit::new(4),
    ];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        dirty_ancillas: vec![Qubit::new(5)],
        max_recursion_depth: 1,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "recursion depth 2 exceeds");
}

#[test]
fn no_ancilla_rejects_exponential_pauli_mcx_above_primitive_cap() {
    let gate = MCGate::new(10, StandardGate::X);
    let qubits: Vec<_> = (0..11).map(Qubit::new).collect();
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would be exponential");
}
