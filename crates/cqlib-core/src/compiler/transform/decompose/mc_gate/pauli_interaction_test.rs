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
    ExpectedParameterizedOperation, assert_columns_eq_for_fixed_qubit_inputs, assert_matrix_eq,
    assert_parameterized_operation_sequence, assert_parameterized_standard_operation,
    assert_transform_failed_contains, circuit_from_operations, original_mc_gate_circuit,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Qubit, StandardGate, circuit_to_matrix};

#[test]
fn zero_control_pauli_interaction_emits_base_standard_gate() {
    let config = McGateDecomposeConfig::default();
    let cases = [
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::RZX,
    ];

    for base_gate in cases {
        let gate = MCGate::new(0, base_gate);
        let qubits = [Qubit::new(0), Qubit::new(1)];
        let params = [CircuitParam::Fixed(0.37)];
        let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
        let operations =
            super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
                .unwrap();

        assert_eq!(operations.len(), 1);
        assert_parameterized_standard_operation(&operations[0], base_gate, &qubits, &[0.37]);
    }
}

#[test]
fn zero_control_symbolic_pauli_interaction_parameter_is_copied() {
    let gate = MCGate::new(0, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Index(7)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
    let operations =
        super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
            .unwrap();

    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZZ)
    ));
    assert!(matches!(
        operations[0].params.as_slice(),
        [CircuitParam::Index(7)]
    ));
}

#[test]
fn one_control_rzz_lifts_cx_rz_cx_body() {
    let gate = MCGate::new(1, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
    let operations =
        super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
            .unwrap();

    assert_parameterized_operation_sequence(
        &operations,
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CRZ,
                qubits: &[0, 2],
                params: &[0.5],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
                params: &[],
            },
        ],
    );
}

#[test]
fn one_control_symbolic_rzz_uses_symbolic_crz_without_arithmetic() {
    let gate = MCGate::new(1, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Index(11)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
    let operations =
        super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
            .unwrap();

    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::CRZ)
    ));
    assert!(matches!(
        operations[1].params.as_slice(),
        [CircuitParam::Index(11)]
    ));
}

#[test]
fn pauli_interaction_decompositions_match_original_matrices_strictly() {
    let cases = [
        (1, StandardGate::RXX, vec![0, 1, 2], 0.37),
        (1, StandardGate::RYY, vec![0, 1, 2], 0.53),
        (1, StandardGate::RZZ, vec![0, 1, 2], 0.67),
        (1, StandardGate::RZX, vec![0, 1, 2], 0.79),
    ];

    for (added_controls, base_gate, qubits, theta) in cases {
        let gate = MCGate::new(added_controls, base_gate);
        let qubits: Vec<_> = qubits.into_iter().map(Qubit::new).collect();
        let params = [CircuitParam::Fixed(theta)];
        let config = McGateDecomposeConfig::default();
        let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
        let operations =
            super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
                .unwrap();
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
fn clean_mode_pauli_interaction_matches_original_on_clean_ancilla_zero_subspace() {
    let cases = [
        (StandardGate::RXX, 0.41),
        (StandardGate::RYY, -0.59),
        (StandardGate::RZZ, 0.5),
        (StandardGate::RZX, -0.83),
    ];

    for (base_gate, theta) in cases {
        let gate = MCGate::new(2, base_gate);
        let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
        let params = [CircuitParam::Fixed(theta)];
        let config = McGateDecomposeConfig {
            ancilla_mode: AncillaMode::CleanAncilla,
            clean_ancillas: vec![Qubit::new(4)],
            ..McGateDecomposeConfig::default()
        };

        let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();
        let operations =
            super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
                .unwrap();
        let actual = circuit_to_matrix(&circuit_from_operations(5, operations), None).unwrap();
        let expected =
            circuit_to_matrix(&original_mc_gate_circuit(5, gate, &qubits, &params), None).unwrap();

        assert_columns_eq_for_fixed_qubit_inputs(&actual, &expected, &[(Qubit::new(4), 0)], 1e-9);
    }
}

#[test]
fn no_ancilla_multi_control_pauli_interaction_rejects_non_exact_mcx_phase_path() {
    let gate = MCGate::new(2, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig::default();
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let err = super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
        .unwrap_err();

    assert_transform_failed_contains(err, "non-exact no-ancilla MCX global phase");
}

#[test]
fn multi_control_symbolic_pauli_interaction_rejects_rz_parameter_arithmetic() {
    let gate = MCGate::new(2, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let params = [CircuitParam::Index(13)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(4)],
        ..McGateDecomposeConfig::default()
    };
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let err = super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
        .unwrap_err();

    assert_transform_failed_contains(err, "symbolic RZ-family parameters require theta/2");
}

#[test]
fn pauli_interaction_budget_is_enforced_across_lifted_base_operations() {
    let gate = MCGate::new(1, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig {
        max_expansion_ops: 2,
        ..McGateDecomposeConfig::default()
    };
    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    let err = super::pauli_interaction::decompose_pauli_interaction_family(&view, &params, &config)
        .unwrap_err();

    assert!(
        matches!(
            err,
            crate::compiler::error::CompilerError::TransformFailed { ref reason, .. }
                if reason.contains("PauliInteraction-family control-lifting of CX failed")
                    && reason.contains("exceeding max_expansion_ops=0")
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn dispatch_decomposes_pauli_interaction_family() {
    let gate = MCGate::new(1, StandardGate::RZZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.5)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert_parameterized_operation_sequence(
        &operations,
        &[
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
                params: &[],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CRZ,
                qubits: &[0, 2],
                params: &[0.5],
            },
            ExpectedParameterizedOperation {
                gate: StandardGate::CCX,
                qubits: &[0, 1, 2],
                params: &[],
            },
        ],
    );
}
