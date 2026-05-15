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

fn assert_operation_sequence(operations: &[Operation], expected: &[ExpectedOperation]) {
    assert_eq!(operations.len(), expected.len());
    for (operation, expected) in operations.iter().zip(expected) {
        let qubits: Vec<_> = expected.qubits.iter().copied().map(Qubit::new).collect();
        assert_standard_operation(operation, expected.gate, &qubits);
    }
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
            CircuitParam::Index(_) => panic!("SWAP decomposition tests expect fixed parameters"),
        });
        circuit
            .append(instruction, qubits, params, label.as_deref())
            .unwrap();
    }
    circuit
}

fn original_mc_gate_circuit(num_qubits: usize, gate: MCGate, qubits: &[Qubit]) -> Circuit {
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

fn assert_matrix_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>, eps: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), actual_value) in actual.indexed_iter() {
        let expected_value = expected[(row, column)];
        assert!(
            (*actual_value - expected_value).norm() < eps,
            "matrix mismatch at ({row}, {column}): actual={actual_value}, expected={expected_value}"
        );
    }
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

fn no_ancilla_swap_matrices(
    added_controls: u8,
    qubits: &[Qubit],
) -> (Array2<Complex64>, Array2<Complex64>) {
    let gate = MCGate::new(added_controls, StandardGate::SWAP);
    let config = McGateDecomposeConfig::default();
    let operations = decompose_mc_gate(&gate, qubits, &[], &config).unwrap();
    assert!(
        operations
            .iter()
            .all(|operation| matches!(operation.instruction, Instruction::Standard(_)))
    );

    let num_qubits = max_qubit_count(qubits);
    let actual = circuit_to_matrix(&circuit_from_operations(num_qubits, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(num_qubits, gate, qubits), None).unwrap();

    (actual, expected)
}

#[test]
fn zero_added_control_swap_emits_three_cx_operations() {
    let gate = MCGate::new(0, StandardGate::SWAP);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[1, 0],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
            },
        ],
    );
}

#[test]
fn one_added_control_swap_emits_three_toffoli_operations() {
    let gate = MCGate::new(1, StandardGate::SWAP);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 2, 1],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
            },
        ],
    );
}

#[test]
fn no_ancilla_swap_small_control_decompositions_match_original_matrices_strictly() {
    let (actual, expected) = no_ancilla_swap_matrices(0, &[Qubit::new(0), Qubit::new(1)]);
    assert_matrix_eq(&actual, &expected, 1e-9);

    let (actual, expected) =
        no_ancilla_swap_matrices(1, &[Qubit::new(0), Qubit::new(1), Qubit::new(2)]);
    assert_matrix_eq(&actual, &expected, 1e-9);
}

#[test]
fn no_ancilla_swap_phase_polynomial_path_matches_original_up_to_global_phase() {
    let (actual, expected) = no_ancilla_swap_matrices(
        2,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
    );

    assert_matrix_eq_up_to_global_phase(&actual, &expected, 1e-9);
}

#[test]
fn clean_mode_uses_same_clean_ancilla_for_each_controlled_x_block() {
    let gate = MCGate::new(2, StandardGate::SWAP);
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
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 3, 2],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
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
fn dirty_mode_uses_borrowed_qubit_for_each_controlled_x_block() {
    let gate = MCGate::new(2, StandardGate::SWAP);
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
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 3, 2],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 4],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[4, 3, 2],
            },
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
fn swap_budget_accounts_for_all_three_controlled_x_blocks() {
    let gate = MCGate::new(1, StandardGate::SWAP);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 2,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 3 operations");
}

#[test]
fn clean_mode_rejects_insufficient_clean_ancillas_for_swap() {
    let gate = MCGate::new(2, StandardGate::SWAP);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "requires 1 clean ancillas");
}

#[test]
fn dirty_mode_requires_borrowed_qubit_for_large_swap() {
    let gate = MCGate::new(2, StandardGate::SWAP);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "requires one dirty ancilla");
}

#[test]
fn dirty_mode_respects_recursion_depth_limit_for_swap() {
    let gate = MCGate::new(3, StandardGate::SWAP);
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
