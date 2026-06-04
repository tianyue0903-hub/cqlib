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

//! Shared helpers for crate-local tests.

use crate::circuit::{
    Circuit, Instruction, MCGate, ParameterValue, Qubit, StandardGate, circuit_to_matrix,
    operation::ValueOperation,
};
use crate::compiler::CompileResult;
use crate::compiler::transform::decompose::mc_gate::Su2RotationAxis;
use crate::device::Device;
use crate::qis::Statevector;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashSet;

/// Tolerance for floating-point comparisons in mc_gate decomposition tests.
pub const EPSILON: f64 = 1e-9;

/// Asserts that an operation is a parameter-free, unlabeled standard gate.
pub fn assert_standard_operation(
    operation: &ValueOperation,
    expected_gate: StandardGate,
    expected_qubits: &[Qubit],
) {
    assert!(matches!(
        operation.instruction,
        Instruction::Standard(gate) if gate == expected_gate
    ));
    assert_eq!(operation.qubits.as_slice(), expected_qubits);
    assert!(operation.params.is_empty());
    assert!(operation.label.is_none());
}

/// Asserts equality for value-operation sequences.
pub fn assert_value_operations_equal(actual: &[ValueOperation], expected: &[ValueOperation]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_operation, expected_operation) in actual.iter().zip(expected) {
        match (
            &actual_operation.instruction,
            &expected_operation.instruction,
        ) {
            (Instruction::Standard(actual_gate), Instruction::Standard(expected_gate)) => {
                assert_eq!(actual_gate, expected_gate);
            }
            (actual_instruction, expected_instruction) => {
                panic!(
                    "instruction mismatch: actual={actual_instruction:?}, expected={expected_instruction:?}"
                );
            }
        }
        assert_eq!(actual_operation.qubits, expected_operation.qubits);
        assert_eq!(
            actual_operation.params.len(),
            expected_operation.params.len()
        );
        for (actual_parameter, expected_parameter) in actual_operation
            .params
            .iter()
            .zip(&expected_operation.params)
        {
            match (actual_parameter, expected_parameter) {
                (ParameterValue::Fixed(actual), ParameterValue::Fixed(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                (ParameterValue::Param(actual), ParameterValue::Param(expected)) => {
                    assert_eq!(actual, expected);
                }
                _ => panic!(
                    "parameter mismatch: actual={actual_parameter:?}, expected={expected_parameter:?}"
                ),
            }
        }
        assert_eq!(actual_operation.label, expected_operation.label);
    }
}

/// Asserts that every operation references only qubits from the allowed set.
pub fn assert_value_operations_only_use_qubits(
    operations: &[ValueOperation],
    allowed_qubits: &[Qubit],
) {
    let allowed_qubits: HashSet<_> = allowed_qubits.iter().copied().collect();

    for (operation_index, operation) in operations.iter().enumerate() {
        for qubit in &operation.qubits {
            assert!(
                allowed_qubits.contains(qubit),
                "operation {operation_index} references disallowed qubit {qubit}: {operation:?}"
            );
        }
    }
}

/// Builds a circuit from self-contained value operations.
pub fn circuit_from_value_operations(
    num_qubits: usize,
    operations: Vec<ValueOperation>,
) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    for operation in operations {
        let label = operation.label;
        circuit
            .append(
                operation.instruction,
                operation.qubits,
                operation.params,
                label.as_deref(),
            )
            .unwrap();
    }
    circuit
}

/// Applies value operations to a clone of an initial statevector.
pub fn statevector_after_value_operations(
    initial_state: &Statevector,
    operations: &[ValueOperation],
) -> Statevector {
    let circuit = circuit_from_value_operations(initial_state.num_qubits, operations.to_vec());
    let mut statevector = initial_state.clone();
    statevector.apply_circuit(&circuit).unwrap();
    statevector
}

/// Asserts equality of complete statevectors up to one global phase.
pub fn assert_statevectors_equal_up_to_global_phase(
    actual: &Statevector,
    expected: &Statevector,
    epsilon: f64,
) {
    assert_eq!(actual.num_qubits, expected.num_qubits);
    assert_eq!(actual.data().len(), expected.data().len());

    let reference_index = expected
        .data()
        .iter()
        .position(|amplitude| amplitude.norm() > epsilon)
        .expect("expected statevector must contain a nonzero amplitude");
    let actual_reference = actual.data()[reference_index];
    let expected_reference = expected.data()[reference_index];
    assert!(
        actual_reference.norm() > epsilon,
        "actual statevector has zero amplitude at reference index {reference_index}"
    );

    let global_phase = actual_reference / expected_reference;
    assert!(
        (global_phase.norm() - 1.0).abs() < epsilon,
        "statevectors differ in amplitude magnitude at reference index {reference_index}: actual={actual_reference}, expected={expected_reference}"
    );

    for (index, (actual_amplitude, expected_amplitude)) in
        actual.data().iter().zip(expected.data()).enumerate()
    {
        let phase_adjusted_expected = global_phase * expected_amplitude;
        assert!(
            (*actual_amplitude - phase_adjusted_expected).norm() < epsilon,
            "statevectors differ at index {index}: actual={actual_amplitude}, expected={phase_adjusted_expected}"
        );
    }
}

/// Asserts approximate equality of complete matrices.
pub fn assert_matrix_approx_eq(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), expected_amplitude) in expected.indexed_iter() {
        assert!(
            (actual[[row, column]] - expected_amplitude).norm() < epsilon,
            "matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
            actual[[row, column]]
        );
    }
}

/// Asserts that a matrix is unitary: U† * U = I.
pub fn assert_is_unitary(matrix: &Array2<Complex64>, epsilon: f64) {
    assert_eq!(
        matrix.nrows(),
        matrix.ncols(),
        "unitary matrix must be square, got {}x{}",
        matrix.nrows(),
        matrix.ncols()
    );

    let product = matrix.t().mapv(|value| value.conj()).dot(matrix);
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let diff = (product[[row, column]] - expected).norm();
            assert!(
                diff < epsilon,
                "matrix is not unitary at row {row}, column {column}: actual={}, expected={expected}, diff={diff}",
                product[[row, column]]
            );
        }
    }
}

/// Asserts approximate equality for selected matrix columns.
pub fn assert_selected_matrix_columns_approx_eq(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    for column in columns {
        for row in 0..expected.nrows() {
            assert!(
                (actual[[row, column]] - expected[[row, column]]).norm() < epsilon,
                "matrix mismatch at row {row}, column {column}: actual={}, expected={}",
                actual[[row, column]],
                expected[[row, column]]
            );
        }
    }
}

/// Asserts selected matrix columns are equal up to one global phase.
pub fn assert_selected_matrix_columns_equal_up_to_global_phase(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    let columns: Vec<_> = columns.into_iter().collect();
    let (reference_actual, reference_expected) = columns
        .iter()
        .flat_map(|column| {
            (0..expected.nrows()).map(move |row| (actual[[row, *column]], expected[[row, *column]]))
        })
        .find(|(_, expected)| expected.norm() > epsilon)
        .expect("selected expected columns must contain a nonzero amplitude");
    let global_phase = reference_actual / reference_expected;

    assert!((global_phase.norm() - 1.0).abs() < epsilon);
    for column in columns {
        for row in 0..expected.nrows() {
            let expected_amplitude = global_phase * expected[[row, column]];
            assert!(
                (actual[[row, column]] - expected_amplitude).norm() < epsilon,
                "matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
                actual[[row, column]]
            );
        }
    }
}

/// Asserts that two matrices are equal up to one global phase.
pub fn assert_matrices_equal_up_to_global_phase(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    epsilon: f64,
) {
    assert_eq!(actual.shape(), expected.shape());
    let (reference_actual, reference_expected) = actual
        .iter()
        .zip(expected.iter())
        .find(|(_, expected)| expected.norm() > epsilon)
        .expect("expected matrix must contain a nonzero amplitude");
    assert!(
        reference_actual.norm() > epsilon,
        "actual matrix has zero amplitude where expected matrix is nonzero"
    );

    let global_phase = reference_actual / reference_expected;
    assert!(
        (global_phase.norm() - 1.0).abs() < epsilon,
        "matrices differ in reference amplitude magnitude: actual={reference_actual}, expected={reference_expected}"
    );

    for ((row, column), expected_amplitude) in expected.indexed_iter() {
        let phase_adjusted_expected = global_phase * expected_amplitude;
        assert!(
            (actual[[row, column]] - phase_adjusted_expected).norm() < epsilon,
            "matrix mismatch at row {row}, column {column}: actual={}, expected={phase_adjusted_expected}",
            actual[[row, column]]
        );
    }
}

/// Asserts that two circuits have the same unitary matrix up to global phase.
pub fn assert_circuits_equivalent_up_to_global_phase(
    actual: &Circuit,
    expected: &Circuit,
    epsilon: f64,
) {
    let actual_matrix = circuit_to_matrix(actual, None).unwrap();
    let expected_matrix = circuit_to_matrix(expected, None).unwrap();
    assert_matrices_equal_up_to_global_phase(&actual_matrix, &expected_matrix, epsilon);
}

/// Extracts standard-gate operations from a circuit.
pub fn standard_ops(circuit: &Circuit) -> Vec<StandardGate> {
    circuit
        .operations()
        .iter()
        .filter_map(|operation| match operation.instruction {
            Instruction::Standard(gate) => Some(gate),
            _ => None,
        })
        .collect()
}

/// Builds a two-qubit line device with the given native gates.
pub fn two_qubit_device(native_gates: Vec<Instruction>) -> Device {
    Device::line("test-device", 2)
        .unwrap()
        .with_native_gates(native_gates)
}

/// Returns whether a circuit still contains non-lowered gate-like instructions.
pub fn contains_high_level_gate(circuit: &Circuit) -> bool {
    circuit.operations().iter().any(|operation| {
        matches!(
            operation.instruction,
            Instruction::CircuitGate(_) | Instruction::UnitaryGate(_) | Instruction::McGate(_)
        )
    })
}

/// Returns whether a named compiler workflow step reported a change.
pub fn step_changed(result: &CompileResult, name: &str) -> bool {
    result
        .steps
        .iter()
        .find(|step| step.name == name)
        .is_some_and(|step| step.changed)
}

/// Asserts that an operation is a standard gate with a single fixed parameter.
pub fn assert_fixed_parameter_operation(
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

/// Maps an SU(2) rotation axis to the corresponding single-qubit standard gate.
pub fn rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::RX,
        Su2RotationAxis::Y => StandardGate::RY,
        Su2RotationAxis::Z => StandardGate::RZ,
    }
}

/// Maps an SU(2) rotation axis to the corresponding controlled standard gate.
pub fn controlled_rotation(axis: Su2RotationAxis) -> StandardGate {
    match axis {
        Su2RotationAxis::X => StandardGate::CRX,
        Su2RotationAxis::Y => StandardGate::CRY,
        Su2RotationAxis::Z => StandardGate::CRZ,
    }
}

/// Builds the matrix of a multi-controlled standard gate.
pub fn mc_gate_matrix(
    num_qubits: usize,
    num_controls: u8,
    gate: StandardGate,
    qubits: Vec<Qubit>,
    params: impl IntoIterator<Item = ParameterValue>,
) -> Array2<Complex64> {
    let mut circuit = Circuit::new(num_qubits);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(num_controls, gate))),
            qubits,
            params,
            None,
        )
        .unwrap();
    circuit_to_matrix(&circuit, None).unwrap()
}

/// Returns the unique nonzero matrix output for one computational-basis input.
pub fn single_nonzero_matrix_output(
    matrix: &Array2<Complex64>,
    input_basis_state: usize,
    epsilon: f64,
) -> (usize, Complex64) {
    let outputs: Vec<_> = (0..matrix.nrows())
        .filter_map(|output_basis_state| {
            let amplitude = matrix[[output_basis_state, input_basis_state]];
            (amplitude.norm() > epsilon).then_some((output_basis_state, amplitude))
        })
        .collect();

    assert_eq!(
        outputs.len(),
        1,
        "input basis state {input_basis_state} has outputs {outputs:?}"
    );
    outputs[0]
}
