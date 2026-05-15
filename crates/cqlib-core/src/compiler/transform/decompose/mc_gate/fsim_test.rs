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

use super::decompose::{AncillaMode, McGateDecomposeConfig, McGateOperandView, decompose_mc_gate};
use super::test_utils::{
    assert_columns_eq_for_fixed_qubit_inputs, assert_matrix_eq,
    assert_parameterized_standard_operation, assert_transform_failed_contains,
    circuit_from_operations, original_mc_gate_circuit,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Qubit, StandardGate, circuit_to_matrix};

#[test]
fn zero_control_fsim_emits_base_standard_gate() {
    let gate = MCGate::new(0, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Fixed(0.37), CircuitParam::Fixed(-0.23)];
    let config = McGateDecomposeConfig::default();
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let operations = super::fsim::decompose_fsim_family(&view, &params, &config).unwrap();

    assert_eq!(operations.len(), 1);
    assert_parameterized_standard_operation(
        &operations[0],
        StandardGate::FSIM,
        &qubits,
        &[0.37, -0.23],
    );
}

#[test]
fn zero_control_symbolic_fsim_parameters_are_copied() {
    let gate = MCGate::new(0, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Index(3), CircuitParam::Index(5)];
    let config = McGateDecomposeConfig::default();
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let operations = super::fsim::decompose_fsim_family(&view, &params, &config).unwrap();

    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::FSIM)
    ));
    assert!(matches!(
        operations[0].params.as_slice(),
        [CircuitParam::Index(3), CircuitParam::Index(5)]
    ));
}

#[test]
fn one_control_fsim_uses_no_gphase_whitelist_body() {
    let gate = MCGate::new(1, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.37), CircuitParam::Fixed(-0.5)];
    let config = McGateDecomposeConfig::default();
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let operations = super::fsim::decompose_fsim_family(&view, &params, &config).unwrap();

    assert!(
        operations
            .iter()
            .all(|operation| !matches!(operation.instruction, Instruction::McGate(_)))
    );
    assert!(operations.iter().all(|operation| !matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::GPhase)
    )));
    assert!(operations.iter().all(|operation| !matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::FSIM)
    )));
}

#[test]
fn one_control_fsim_decomposition_matches_original_matrix_strictly() {
    let gate = MCGate::new(1, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.41), CircuitParam::Fixed(-0.67)];
    let config = McGateDecomposeConfig::default();
    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    let actual = circuit_to_matrix(&circuit_from_operations(3, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(3, gate, &qubits, &params), None).unwrap();

    assert_matrix_eq(&actual, &expected, 1e-9);
}

#[test]
fn clean_mode_fsim_matches_original_on_clean_ancilla_zero_subspace() {
    let gate = MCGate::new(2, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let params = [CircuitParam::Fixed(0.29), CircuitParam::Fixed(0.61)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4), Qubit::new(5)],
        ..McGateDecomposeConfig::default()
    };

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();
    let actual = circuit_to_matrix(&circuit_from_operations(6, operations), None).unwrap();
    let expected =
        circuit_to_matrix(&original_mc_gate_circuit(6, gate, &qubits, &params), None).unwrap();

    assert_columns_eq_for_fixed_qubit_inputs(
        &actual,
        &expected,
        &[(Qubit::new(4), 0), (Qubit::new(5), 0)],
        1e-9,
    );
}

#[test]
fn controlled_symbolic_fsim_rejects_phi_arithmetic() {
    let gate = MCGate::new(1, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Index(7), CircuitParam::Index(9)];
    let config = McGateDecomposeConfig::default();
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let err = super::fsim::decompose_fsim_family(&view, &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "symbolic FSIM-family parameters require phi/2");
}

#[test]
fn no_ancilla_multi_control_fsim_rejects_non_exact_pauli_interaction_path() {
    let gate = MCGate::new(2, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let params = [CircuitParam::Fixed(0.29), CircuitParam::Fixed(0.61)];
    let config = McGateDecomposeConfig::default();

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert!(
        matches!(
            err,
            crate::compiler::error::CompilerError::TransformFailed { ref reason, .. }
                if reason.contains("FSIM-family control-lifting of RXX failed")
                    && reason.contains("non-exact no-ancilla MCX global phase")
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn fsim_budget_is_enforced_across_lifted_base_operations() {
    let gate = MCGate::new(1, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.29), CircuitParam::Fixed(0.61)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 2,
        ..McGateDecomposeConfig::default()
    };

    let err = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap_err();

    assert!(
        matches!(
            err,
            crate::compiler::error::CompilerError::TransformFailed { ref reason, .. }
                if reason.contains("FSIM-family control-lifting of RXX failed")
                    && reason.contains("exceeding max_expansion_ops=")
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn dispatch_decomposes_fsim_family() {
    let gate = MCGate::new(1, StandardGate::FSIM);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.37), CircuitParam::Fixed(-0.5)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert!(
        operations
            .iter()
            .all(|operation| matches!(operation.instruction, Instruction::Standard(_)))
    );
    assert!(operations.iter().all(|operation| !matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::FSIM)
    )));
}
