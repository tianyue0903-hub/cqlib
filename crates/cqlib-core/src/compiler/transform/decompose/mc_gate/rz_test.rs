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
use super::test_utils::assert_columns_eq_for_fixed_qubit_inputs;
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
    params: &'static [f64],
}

fn assert_standard_operation(
    operation: &Operation,
    gate: StandardGate,
    qubits: &[Qubit],
    params: &[f64],
) {
    assert!(matches!(operation.instruction, Instruction::Standard(actual) if actual == gate));
    assert_eq!(operation.qubits.as_slice(), qubits);
    assert_eq!(operation.params.len(), params.len());
    for (actual, expected) in operation.params.iter().zip(params) {
        let CircuitParam::Fixed(actual) = actual else {
            panic!("RZ decomposition tests expect fixed emitted parameters");
        };
        assert!(
            (*actual - *expected).abs() < 1e-12,
            "parameter mismatch: actual={actual}, expected={expected}"
        );
    }
    assert!(operation.label.is_none());
}

fn assert_operation_sequence(operations: &[Operation], expected: &[ExpectedOperation]) {
    assert_eq!(operations.len(), expected.len());
    for (operation, expected) in operations.iter().zip(expected) {
        let qubits: Vec<_> = expected.qubits.iter().copied().map(Qubit::new).collect();
        assert_standard_operation(operation, expected.gate, &qubits, expected.params);
    }
}

fn assert_index_param(operation: &Operation, expected_index: u32) {
    assert_eq!(operation.params.len(), 1);
    assert!(
        matches!(operation.params[0], CircuitParam::Index(actual) if actual == expected_index),
        "expected copied symbolic parameter index {expected_index}, got {:?}",
        operation.params[0]
    );
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
            CircuitParam::Index(_) => panic!("RZ matrix tests expect fixed parameters"),
        });
        circuit
            .append(instruction, qubits, params, label.as_deref())
            .unwrap();
    }
    circuit
}

fn original_mc_gate_circuit(gate: MCGate, qubits: &[Qubit], params: &[CircuitParam]) -> Circuit {
    let mut circuit = Circuit::new(max_qubit_count(qubits));
    let params = params.iter().map(|param| match param {
        CircuitParam::Fixed(value) => ParameterValue::Fixed(*value),
        CircuitParam::Index(_) => panic!("RZ matrix tests expect fixed parameters"),
    });
    circuit
        .append(
            Instruction::McGate(Box::new(gate)),
            qubits.iter().copied(),
            params,
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

fn assert_rz_decomposition_matches_original(
    added_controls: u8,
    base_gate: StandardGate,
    qubits: &[Qubit],
    theta: f64,
) {
    let gate = MCGate::new(added_controls, base_gate);
    let params = [CircuitParam::Fixed(theta)];
    let config = McGateDecomposeConfig::default();
    let operations = decompose_mc_gate(&gate, qubits, &params, &config).unwrap();
    assert!(
        operations
            .iter()
            .all(|operation| matches!(operation.instruction, Instruction::Standard(_)))
    );

    let num_qubits = max_qubit_count(qubits);
    let actual = circuit_to_matrix(&circuit_from_operations(num_qubits, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(gate, qubits, &params), None).unwrap();

    assert_matrix_eq(&actual, &expected, 1e-9);
}

#[test]
fn zero_and_one_control_rz_boundaries_emit_standard_gates() {
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(
        &MCGate::new(0, StandardGate::RZ),
        &[Qubit::new(0)],
        &[CircuitParam::Fixed(0.5)],
        &config,
    )
    .unwrap();
    assert_operation_sequence(
        &operations,
        &[ExpectedOperation {
            gate: StandardGate::RZ,
            qubits: &[0],
            params: &[0.5],
        }],
    );

    let operations = decompose_mc_gate(
        &MCGate::new(1, StandardGate::RZ),
        &[Qubit::new(0), Qubit::new(1)],
        &[CircuitParam::Fixed(0.5)],
        &config,
    )
    .unwrap();
    assert_operation_sequence(
        &operations,
        &[ExpectedOperation {
            gate: StandardGate::CRZ,
            qubits: &[0, 1],
            params: &[0.5],
        }],
    );

    let operations = decompose_mc_gate(
        &MCGate::new(0, StandardGate::CRZ),
        &[Qubit::new(0), Qubit::new(1)],
        &[CircuitParam::Fixed(0.5)],
        &config,
    )
    .unwrap();
    assert_operation_sequence(
        &operations,
        &[ExpectedOperation {
            gate: StandardGate::CRZ,
            qubits: &[0, 1],
            params: &[0.5],
        }],
    );
}

#[test]
fn direct_rz_and_crz_paths_copy_symbolic_parameter_indices() {
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(
        &MCGate::new(0, StandardGate::RZ),
        &[Qubit::new(0)],
        &[CircuitParam::Index(3)],
        &config,
    )
    .unwrap();
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert_index_param(&operations[0], 3);

    let operations = decompose_mc_gate(
        &MCGate::new(1, StandardGate::RZ),
        &[Qubit::new(0), Qubit::new(1)],
        &[CircuitParam::Index(7)],
        &config,
    )
    .unwrap();
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::CRZ)
    ));
    assert_index_param(&operations[0], 7);
}

#[test]
fn no_ancilla_rz_decompositions_match_original_matrices_strictly() {
    let cases = [
        (0, StandardGate::RZ, vec![0], 0.37),
        (1, StandardGate::RZ, vec![0, 1], -0.41),
        (2, StandardGate::RZ, vec![0, 1, 2], 0.73),
        (0, StandardGate::CRZ, vec![0, 1], -0.29),
        (1, StandardGate::CRZ, vec![0, 1, 2], 0.61),
        (2, StandardGate::CRZ, vec![0, 1, 2, 3], -0.83),
    ];

    for (added_controls, base_gate, qubits, theta) in cases {
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        assert_rz_decomposition_matches_original(added_controls, base_gate, &qubits, theta);
    }
}

#[test]
fn clean_mode_emits_target_phase_and_conditional_phase_compensation() {
    let gate = MCGate::new(2, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(3)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert_operation_sequence(
        &operations,
        &[
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 3],
                params: &[],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[3],
                params: &[0.25],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[2],
                params: &[0.25],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[3, 2],
                params: &[],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[2],
                params: &[-0.25],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[3, 2],
                params: &[],
            },
            ExpectedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 3],
                params: &[],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[0],
                params: &[-0.125],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[1],
                params: &[-0.125],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
                params: &[],
            },
            ExpectedOperation {
                gate: StandardGate::Phase,
                qubits: &[1],
                params: &[0.125],
            },
            ExpectedOperation {
                gate: StandardGate::CX,
                qubits: &[0, 1],
                params: &[],
            },
        ],
    );
}

#[test]
fn clean_mode_rz_matches_original_on_clean_ancilla_subspace() {
    let gate = MCGate::new(2, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(3)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();
    let actual = circuit_to_matrix(&circuit_from_operations(4, operations), None).unwrap();

    let mut expected_circuit = Circuit::new(4);
    expected_circuit
        .append(
            Instruction::McGate(Box::new(gate)),
            qubits,
            [ParameterValue::Fixed(0.5)],
            None,
        )
        .unwrap();
    let expected = circuit_to_matrix(&expected_circuit, None).unwrap();

    assert_columns_eq_for_fixed_qubit_inputs(&actual, &expected, &[(Qubit::new(3), 0)], 1e-9);
}

#[test]
fn multi_control_rz_rejects_symbolic_parameter_arithmetic() {
    let gate = MCGate::new(2, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Index(0)];
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "symbolic RZ-family parameters require theta/2");
}

#[test]
fn dirty_mode_rejects_multi_control_rz() {
    let gate = MCGate::new(2, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "dirty-ancilla RZ decomposition is not supported");
}

#[test]
fn clean_mode_rejects_insufficient_clean_ancillas_for_rz() {
    let gate = MCGate::new(3, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4)],
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "requires 2 clean ancillas");
}

#[test]
fn rz_budget_accounts_for_phase_and_conditional_compensation() {
    let gate = MCGate::new(2, StandardGate::RZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 21,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "would emit 22 operations");
}

#[test]
fn no_ancilla_rejects_exponential_rz_above_cap() {
    let gate = MCGate::new(10, StandardGate::RZ);
    let qubits: Vec<_> = (0..11).map(Qubit::new).collect();
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "would be exponential");
}
