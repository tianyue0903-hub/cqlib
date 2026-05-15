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

use crate::circuit::{
    Circuit, CircuitParam, Instruction, MCGate, Operation, ParameterValue, Qubit, StandardGate,
};
use crate::compiler::error::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;

#[derive(Clone, Copy)]
pub(super) struct ExpectedParameterizedOperation {
    pub(super) gate: StandardGate,
    pub(super) qubits: &'static [u32],
    pub(super) params: &'static [f64],
}

pub(super) fn assert_parameterized_standard_operation(
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
            panic!("decomposition tests expect fixed emitted parameters");
        };
        assert!(
            (*actual - *expected).abs() < 1e-12,
            "parameter mismatch: actual={actual}, expected={expected}"
        );
    }
    assert!(operation.label.is_none());
}

pub(super) fn assert_parameterized_operation_sequence(
    operations: &[Operation],
    expected: &[ExpectedParameterizedOperation],
) {
    assert_eq!(operations.len(), expected.len());
    for (operation, expected) in operations.iter().zip(expected) {
        let qubits: Vec<_> = expected.qubits.iter().copied().map(Qubit::new).collect();
        assert_parameterized_standard_operation(operation, expected.gate, &qubits, expected.params);
    }
}

pub(super) fn assert_transform_failed_contains(err: CompilerError, expected: &str) {
    assert!(
        matches!(
            err,
            CompilerError::TransformFailed { ref reason, .. } if reason.contains(expected)
        ),
        "expected TransformFailed reason containing {expected:?}, got {err:?}"
    );
}

pub(super) fn circuit_from_operations(num_qubits: usize, operations: Vec<Operation>) -> Circuit {
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
            CircuitParam::Index(_) => panic!("decomposition tests expect fixed parameters"),
        });
        circuit
            .append(instruction, qubits, params, label.as_deref())
            .unwrap();
    }
    circuit
}

pub(super) fn original_mc_gate_circuit(
    num_qubits: usize,
    gate: MCGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    let params = params.iter().map(|param| match param {
        CircuitParam::Fixed(value) => ParameterValue::Fixed(*value),
        CircuitParam::Index(_) => panic!("decomposition tests expect fixed parameters"),
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

pub(super) fn assert_matrix_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>, eps: f64) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), actual_value) in actual.indexed_iter() {
        let expected_value = expected[(row, column)];
        assert!(
            (*actual_value - expected_value).norm() < eps,
            "matrix mismatch at ({row}, {column}): actual={actual_value}, expected={expected_value}"
        );
    }
}

pub(super) fn assert_columns_eq_for_fixed_qubit_inputs(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    fixed_inputs: &[(Qubit, u8)],
    eps: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    let (rows, columns) = (actual.nrows(), actual.ncols());
    let mut checked_columns = 0usize;

    for column in 0..columns {
        if fixed_inputs
            .iter()
            .all(|(qubit, bit)| basis_bit(column, *qubit) == *bit)
        {
            checked_columns += 1;
            for row in 0..rows {
                let actual_value = actual[(row, column)];
                let expected_value = expected[(row, column)];
                assert!(
                    (actual_value - expected_value).norm() < eps,
                    "matrix mismatch at ({row}, {column}) with fixed inputs {fixed_inputs:?}: actual={actual_value}, expected={expected_value}"
                );
            }
        }
    }

    assert!(
        checked_columns > 0,
        "fixed input selector {fixed_inputs:?} matched no matrix columns"
    );
}

fn basis_bit(index: usize, qubit: Qubit) -> u8 {
    ((index >> qubit.index() as usize) & 1) as u8
}
