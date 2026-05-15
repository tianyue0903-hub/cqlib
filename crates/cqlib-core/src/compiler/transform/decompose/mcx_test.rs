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

use super::mcx::{
    MAX_DIRTY_RECURSIVE_CONTROLS, MAX_NO_ANCILLA_PHASE_POLY_QUBITS, decompose_clean_ancilla_mcx,
    decompose_dirty_ancilla_mcx, decompose_no_ancilla_mcx,
};
use crate::circuit::{
    Circuit, CircuitParam, Instruction, MCGate, Operation, ParameterValue, Qubit, StandardGate,
    circuit_to_matrix,
};
use crate::compiler::error::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;
use std::collections::HashSet;

fn assert_standard_operation(operation: &Operation, gate: StandardGate, qubits: &[Qubit]) {
    assert!(matches!(operation.instruction, Instruction::Standard(actual) if actual == gate));
    assert_eq!(operation.qubits.as_slice(), qubits);
    assert!(operation.params.is_empty());
    assert!(operation.label.is_none());
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
            CircuitParam::Index(_) => panic!("decomposition tests expect fixed parameters"),
        });
        circuit
            .append(instruction, qubits, params, label.as_deref())
            .unwrap();
    }
    circuit
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

fn original_mcx_circuit(control_count: usize) -> Circuit {
    let mut circuit = Circuit::new(control_count + 1);
    let controls: Vec<_> = (0..control_count)
        .map(|index| Qubit::new(index as u32))
        .collect();
    let target = Qubit::new(control_count as u32);

    if control_count <= 2 {
        circuit
            .multi_control(StandardGate::X, controls, [target], [])
            .unwrap();
    } else {
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(control_count as u8, StandardGate::X))),
                controls.into_iter().chain(std::iter::once(target)),
                [],
                None,
            )
            .unwrap();
    }

    circuit
}

fn assert_no_ancilla_mcx_equivalent_to_original(control_count: usize) {
    let controls: Vec<_> = (0..control_count)
        .map(|index| Qubit::new(index as u32))
        .collect();
    let target = Qubit::new(control_count as u32);
    let operations = decompose_no_ancilla_mcx(&controls, target).unwrap();

    let actual_circuit = circuit_from_operations(control_count + 1, operations);
    let actual = circuit_to_matrix(&actual_circuit, None).unwrap();
    let expected = circuit_to_matrix(&original_mcx_circuit(control_count), None).unwrap();

    assert_matrix_eq_up_to_global_phase(&actual, &expected, 1e-9);
}

fn assert_dirty_ancilla_mcx_semantics(control_count: usize, borrow_initial: bool) {
    let controls: Vec<_> = (0..control_count)
        .map(|index| Qubit::new(index as u32))
        .collect();
    let target = Qubit::new(control_count as u32);
    let borrow = Qubit::new((control_count + 1) as u32);
    let operations = decompose_dirty_ancilla_mcx(&controls, target, borrow).unwrap();
    let circuit = circuit_from_operations(control_count + 2, operations);
    let matrix = circuit_to_matrix(&circuit, None).unwrap();

    let dim = 1usize << (control_count + 2);
    let target_mask = 1usize << target.index();
    let borrow_mask = 1usize << borrow.index();
    let expected_borrow = if borrow_initial { borrow_mask } else { 0 };

    for input in 0..dim {
        if input & borrow_mask != expected_borrow {
            continue;
        }

        let controls_active = controls
            .iter()
            .all(|control| input & (1usize << control.index()) != 0);
        let expected_output = if controls_active {
            input ^ target_mask
        } else {
            input
        };

        for output in 0..dim {
            let expected = if output == expected_output { 1.0 } else { 0.0 };
            let diff = (matrix[[output, input]] - expected).norm();
            assert!(
                diff < 1e-10,
                "dirty-ancilla matrix column {input} row {output}: expected amplitude {expected}, got {}",
                matrix[[output, input]]
            );
        }
        assert_eq!(expected_output & borrow_mask, expected_borrow);
    }
}

#[test]
fn decomposes_zero_control_mcx_to_x() {
    let operations = decompose_clean_ancilla_mcx(&[], Qubit::new(0), &[]).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(&operations[0], StandardGate::X, &[Qubit::new(0)]);
}

#[test]
fn decomposes_single_control_mcx_to_cx() {
    let operations = decompose_clean_ancilla_mcx(&[Qubit::new(0)], Qubit::new(1), &[]).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CX,
        &[Qubit::new(0), Qubit::new(1)],
    );
}

#[test]
fn decomposes_two_control_mcx_to_ccx() {
    let operations =
        decompose_clean_ancilla_mcx(&[Qubit::new(0), Qubit::new(1)], Qubit::new(2), &[]).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
    );
}

#[test]
fn dirty_ancilla_decomposes_zero_control_mcx_to_x() {
    let operations = decompose_dirty_ancilla_mcx(&[], Qubit::new(0), Qubit::new(0)).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(&operations[0], StandardGate::X, &[Qubit::new(0)]);
}

#[test]
fn dirty_ancilla_decomposes_single_control_mcx_to_cx() {
    let operations =
        decompose_dirty_ancilla_mcx(&[Qubit::new(0)], Qubit::new(1), Qubit::new(0)).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CX,
        &[Qubit::new(0), Qubit::new(1)],
    );
}

#[test]
fn dirty_ancilla_decomposes_two_control_mcx_to_ccx() {
    let operations = decompose_dirty_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1)],
        Qubit::new(2),
        Qubit::new(2),
    )
    .unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
    );
}

#[test]
fn dirty_ancilla_three_control_mcx_restores_borrow_initial_zero() {
    assert_dirty_ancilla_mcx_semantics(3, false);
}

#[test]
fn dirty_ancilla_three_control_mcx_restores_borrow_initial_one() {
    assert_dirty_ancilla_mcx_semantics(3, true);
}

#[test]
fn dirty_ancilla_four_control_mcx_restores_borrow_initial_zero() {
    assert_dirty_ancilla_mcx_semantics(4, false);
}

#[test]
fn dirty_ancilla_four_control_mcx_restores_borrow_initial_one() {
    assert_dirty_ancilla_mcx_semantics(4, true);
}

#[test]
fn dirty_ancilla_rejects_repeated_qubits() {
    let repeated_control = decompose_dirty_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(0)],
        Qubit::new(1),
        Qubit::new(2),
    )
    .unwrap_err();
    let target_reused_as_control = decompose_dirty_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1)],
        Qubit::new(1),
        Qubit::new(2),
    )
    .unwrap_err();
    let borrow_reused_as_control = decompose_dirty_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        Qubit::new(1),
    )
    .unwrap_err();
    let borrow_reused_as_target = decompose_dirty_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        Qubit::new(3),
    )
    .unwrap_err();

    for err in [
        repeated_control,
        target_reused_as_control,
        borrow_reused_as_control,
        borrow_reused_as_target,
    ] {
        assert!(matches!(
            err,
            CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
        ));
    }
}

#[test]
fn dirty_ancilla_output_uses_only_controls_target_and_borrow() {
    let controls = [Qubit::new(5), Qubit::new(1), Qubit::new(7), Qubit::new(3)];
    let target = Qubit::new(0);
    let borrow = Qubit::new(9);
    let operations = decompose_dirty_ancilla_mcx(&controls, target, borrow).unwrap();
    let allowed: HashSet<_> = controls.into_iter().chain([target, borrow]).collect();

    for operation in operations {
        match operation.instruction {
            Instruction::Standard(StandardGate::X)
            | Instruction::Standard(StandardGate::CX)
            | Instruction::Standard(StandardGate::CCX) => {}
            _ => panic!(
                "unexpected gate in dirty-ancilla decomposition: {:?}",
                operation.instruction
            ),
        }

        for qubit in operation.qubits {
            assert!(
                allowed.contains(&qubit),
                "dirty-ancilla decomposition emitted unexpected qubit {qubit}"
            );
        }
    }
}

#[test]
fn dirty_ancilla_rejects_too_large_recursive_mcx() {
    let controls: Vec<_> = (0..=MAX_DIRTY_RECURSIVE_CONTROLS as u32)
        .map(Qubit::new)
        .collect();
    let target = Qubit::new((MAX_DIRTY_RECURSIVE_CONTROLS + 1) as u32);
    let borrow = Qubit::new((MAX_DIRTY_RECURSIVE_CONTROLS + 2) as u32);

    let err = decompose_dirty_ancilla_mcx(&controls, target, borrow).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("dirty-ancilla recursive")
    ));
}

#[test]
fn no_ancilla_decomposes_zero_control_mcx_to_x() {
    let operations = decompose_no_ancilla_mcx(&[], Qubit::new(0)).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(&operations[0], StandardGate::X, &[Qubit::new(0)]);
}

#[test]
fn no_ancilla_decomposes_single_control_mcx_to_cx() {
    let operations = decompose_no_ancilla_mcx(&[Qubit::new(0)], Qubit::new(1)).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CX,
        &[Qubit::new(0), Qubit::new(1)],
    );
}

#[test]
fn no_ancilla_decomposes_two_control_mcx_to_ccx() {
    let operations =
        decompose_no_ancilla_mcx(&[Qubit::new(0), Qubit::new(1)], Qubit::new(2)).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
    );
}

#[test]
fn no_ancilla_three_control_mcx_matches_original_up_to_global_phase() {
    assert_no_ancilla_mcx_equivalent_to_original(3);
}

#[test]
fn no_ancilla_four_control_mcx_matches_original_up_to_global_phase() {
    assert_no_ancilla_mcx_equivalent_to_original(4);
}

#[test]
fn no_ancilla_rejects_too_large_phase_polynomial_mcx() {
    let controls: Vec<_> = (0..MAX_NO_ANCILLA_PHASE_POLY_QUBITS as u32)
        .map(Qubit::new)
        .collect();
    let target = Qubit::new(MAX_NO_ANCILLA_PHASE_POLY_QUBITS as u32);

    let err = decompose_no_ancilla_mcx(&controls, target).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("would be exponential")
    ));
}

#[test]
fn no_ancilla_three_control_uses_only_h_cx_rz() {
    let operations = decompose_no_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
    )
    .unwrap();

    for operation in operations {
        match operation.instruction {
            Instruction::Standard(StandardGate::H)
            | Instruction::Standard(StandardGate::CX)
            | Instruction::Standard(StandardGate::RZ) => {}
            _ => panic!(
                "unexpected gate in no-ancilla decomposition: {:?}",
                operation.instruction
            ),
        }
    }
}

#[test]
fn no_ancilla_rejects_repeated_controls() {
    let err = decompose_no_ancilla_mcx(&[Qubit::new(0), Qubit::new(0)], Qubit::new(1)).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
    ));
}

#[test]
fn no_ancilla_rejects_target_reused_as_control() {
    let err = decompose_no_ancilla_mcx(&[Qubit::new(0), Qubit::new(1)], Qubit::new(1)).unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
    ));
}

#[test]
fn no_ancilla_output_uses_only_controls_and_target() {
    let controls = [Qubit::new(5), Qubit::new(1), Qubit::new(7), Qubit::new(3)];
    let target = Qubit::new(0);
    let operations = decompose_no_ancilla_mcx(&controls, target).unwrap();
    let allowed: HashSet<_> = controls
        .into_iter()
        .chain(std::iter::once(target))
        .collect();

    for operation in operations {
        for qubit in operation.qubits {
            assert!(
                allowed.contains(&qubit),
                "no-ancilla decomposition emitted unexpected qubit {qubit}"
            );
        }
    }
}

#[test]
fn decomposes_three_control_mcx_with_one_clean_ancilla() {
    let operations = decompose_clean_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        &[Qubit::new(4)],
    )
    .unwrap();

    assert_eq!(operations.len(), 3);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(4)],
    );
    assert_standard_operation(
        &operations[1],
        StandardGate::CCX,
        &[Qubit::new(4), Qubit::new(2), Qubit::new(3)],
    );
    assert_standard_operation(
        &operations[2],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(4)],
    );
}

#[test]
fn decomposes_four_control_mcx_with_v_chain_and_reverse_uncompute() {
    let operations = decompose_clean_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        Qubit::new(4),
        &[Qubit::new(5), Qubit::new(6)],
    )
    .unwrap();

    assert_eq!(operations.len(), 5);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(5)],
    );
    assert_standard_operation(
        &operations[1],
        StandardGate::CCX,
        &[Qubit::new(5), Qubit::new(2), Qubit::new(6)],
    );
    assert_standard_operation(
        &operations[2],
        StandardGate::CCX,
        &[Qubit::new(6), Qubit::new(3), Qubit::new(4)],
    );
    assert_standard_operation(
        &operations[3],
        StandardGate::CCX,
        &[Qubit::new(5), Qubit::new(2), Qubit::new(6)],
    );
    assert_standard_operation(
        &operations[4],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(5)],
    );
}

#[test]
fn decomposes_non_contiguous_qubits_without_reordering() {
    let operations = decompose_clean_ancilla_mcx(
        &[Qubit::new(5), Qubit::new(1), Qubit::new(7), Qubit::new(3)],
        Qubit::new(0),
        &[Qubit::new(9), Qubit::new(2)],
    )
    .unwrap();

    assert_eq!(operations.len(), 5);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(5), Qubit::new(1), Qubit::new(9)],
    );
    assert_standard_operation(
        &operations[1],
        StandardGate::CCX,
        &[Qubit::new(9), Qubit::new(7), Qubit::new(2)],
    );
    assert_standard_operation(
        &operations[2],
        StandardGate::CCX,
        &[Qubit::new(2), Qubit::new(3), Qubit::new(0)],
    );
    assert_standard_operation(
        &operations[3],
        StandardGate::CCX,
        &[Qubit::new(9), Qubit::new(7), Qubit::new(2)],
    );
    assert_standard_operation(
        &operations[4],
        StandardGate::CCX,
        &[Qubit::new(5), Qubit::new(1), Qubit::new(9)],
    );
}

#[test]
fn ignores_unused_extra_clean_ancillas_for_base_cases() {
    let operations =
        decompose_clean_ancilla_mcx(&[Qubit::new(0)], Qubit::new(1), &[Qubit::new(0)]).unwrap();

    assert_eq!(operations.len(), 1);
    assert_standard_operation(
        &operations[0],
        StandardGate::CX,
        &[Qubit::new(0), Qubit::new(1)],
    );
}

#[test]
fn ignores_duplicate_unused_extra_clean_ancillas() {
    let operations = decompose_clean_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        &[Qubit::new(4), Qubit::new(4)],
    )
    .unwrap();

    assert_eq!(operations.len(), 3);
    assert_standard_operation(
        &operations[0],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(4)],
    );
    assert_standard_operation(
        &operations[1],
        StandardGate::CCX,
        &[Qubit::new(4), Qubit::new(2), Qubit::new(3)],
    );
    assert_standard_operation(
        &operations[2],
        StandardGate::CCX,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(4)],
    );
}

#[test]
fn rejects_missing_clean_ancillas() {
    let err = decompose_clean_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        &[],
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. }
            if reason.contains("requires 1 clean ancillas")
    ));
}

#[test]
fn rejects_repeated_controls() {
    let err = decompose_clean_ancilla_mcx(&[Qubit::new(0), Qubit::new(0)], Qubit::new(1), &[])
        .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
    ));
}

#[test]
fn rejects_target_reused_as_control() {
    let err = decompose_clean_ancilla_mcx(&[Qubit::new(0), Qubit::new(1)], Qubit::new(1), &[])
        .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
    ));
}

#[test]
fn rejects_repeated_clean_ancillas() {
    let err = decompose_clean_ancilla_mcx(
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        Qubit::new(4),
        &[Qubit::new(5), Qubit::new(5)],
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::TransformFailed { reason, .. } if reason.contains("must be distinct")
    ));
}

#[test]
fn preserves_clean_ancilla_subspace_semantics() {
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let target = Qubit::new(3);
    let ancilla = Qubit::new(4);
    let operations = decompose_clean_ancilla_mcx(&controls, target, &[ancilla]).unwrap();
    let circuit = circuit_from_operations(5, operations);

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    let dim = 1usize << 5;
    let target_mask = 1usize << target.index();
    let ancilla_mask = 1usize << ancilla.index();

    for input in 0..dim {
        if input & ancilla_mask != 0 {
            continue;
        }

        let controls_active = controls
            .iter()
            .all(|control| input & (1usize << control.index()) != 0);
        let expected_output = if controls_active {
            input ^ target_mask
        } else {
            input
        };

        for output in 0..dim {
            let expected = if output == expected_output { 1.0 } else { 0.0 };
            let diff = (matrix[[output, input]] - expected).norm();
            assert!(
                diff < 1e-10,
                "matrix column {input} row {output}: expected amplitude {expected}, got {}",
                matrix[[output, input]]
            );
        }
        assert_eq!(expected_output & ancilla_mask, 0);
    }
}
