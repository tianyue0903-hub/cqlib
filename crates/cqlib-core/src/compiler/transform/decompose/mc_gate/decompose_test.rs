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

use super::decompose::{
    AncillaMode, McGateDecomposeConfig, McGateFamily, McGateOperandView, decompose_mc_gate,
    decompose_mc_gate_operation,
};
use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use smallvec::smallvec;

fn assert_transform_failed_contains(err: CompilerError, expected: &str) {
    assert!(
        matches!(
            err,
            CompilerError::TransformFailed { ref reason, .. } if reason.contains(expected)
        ),
        "expected TransformFailed reason containing {expected:?}, got {err:?}"
    );
}

#[test]
fn operand_view_partitions_added_controls_and_target() {
    let gate = MCGate::new(2, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap();

    assert_eq!(view.base_gate(), StandardGate::X);
    assert_eq!(view.added_controls(), &[Qubit::new(0), Qubit::new(1)]);
    assert!(view.inherent_controls().is_empty());
    assert_eq!(view.targets(), &[Qubit::new(2)]);
    assert_eq!(view.all_controls(), &[Qubit::new(0), Qubit::new(1)]);
    assert_eq!(view.total_control_count(), 2);
}

#[test]
fn operand_view_keeps_base_cx_inherent_control_order() {
    let gate = MCGate::new(1, StandardGate::CX);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap();

    assert_eq!(view.added_controls(), &[Qubit::new(0)]);
    assert_eq!(view.inherent_controls(), &[Qubit::new(1)]);
    assert_eq!(view.targets(), &[Qubit::new(2)]);
    assert_eq!(view.all_controls(), &[Qubit::new(0), Qubit::new(1)]);
}

#[test]
fn operand_view_keeps_base_crz_inherent_control_order() {
    let gate = MCGate::new(1, StandardGate::CRZ);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let params = [CircuitParam::Fixed(0.25)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &params, &config).unwrap();

    assert_eq!(view.added_controls(), &[Qubit::new(0)]);
    assert_eq!(view.inherent_controls(), &[Qubit::new(1)]);
    assert_eq!(view.targets(), &[Qubit::new(2)]);
    assert_eq!(view.all_controls(), &[Qubit::new(0), Qubit::new(1)]);
}

#[test]
fn operand_view_keeps_base_ccx_inherent_control_order() {
    let gate = MCGate::new(1, StandardGate::CCX);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig::default();

    let view = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap();

    assert_eq!(view.added_controls(), &[Qubit::new(0)]);
    assert_eq!(view.inherent_controls(), &[Qubit::new(1), Qubit::new(2)]);
    assert_eq!(view.targets(), &[Qubit::new(3)]);
    assert_eq!(
        view.all_controls(),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)]
    );
}

#[test]
fn classifies_all_standard_gates() {
    let cases = [
        (StandardGate::I, McGateFamily::Identity),
        (StandardGate::H, McGateFamily::OneQubit),
        (StandardGate::RX, McGateFamily::Rotation),
        (StandardGate::RXX, McGateFamily::PauliInteraction),
        (StandardGate::RXY, McGateFamily::OneQubit),
        (StandardGate::RY, McGateFamily::Rotation),
        (StandardGate::RYY, McGateFamily::PauliInteraction),
        (StandardGate::RZ, McGateFamily::Rotation),
        (StandardGate::RZX, McGateFamily::PauliInteraction),
        (StandardGate::RZZ, McGateFamily::PauliInteraction),
        (StandardGate::S, McGateFamily::Phase),
        (StandardGate::SDG, McGateFamily::Phase),
        (StandardGate::SWAP, McGateFamily::Swap),
        (StandardGate::T, McGateFamily::Phase),
        (StandardGate::TDG, McGateFamily::Phase),
        (StandardGate::U, McGateFamily::OneQubit),
        (StandardGate::X, McGateFamily::Pauli),
        (StandardGate::XY, McGateFamily::OneQubit),
        (StandardGate::X2P, McGateFamily::OneQubit),
        (StandardGate::X2M, McGateFamily::OneQubit),
        (StandardGate::XY2P, McGateFamily::OneQubit),
        (StandardGate::XY2M, McGateFamily::OneQubit),
        (StandardGate::Y, McGateFamily::Pauli),
        (StandardGate::Y2P, McGateFamily::OneQubit),
        (StandardGate::Y2M, McGateFamily::OneQubit),
        (StandardGate::Z, McGateFamily::Pauli),
        (StandardGate::Phase, McGateFamily::Phase),
        (StandardGate::GPhase, McGateFamily::Unsupported),
        (StandardGate::CX, McGateFamily::Pauli),
        (StandardGate::CCX, McGateFamily::Pauli),
        (StandardGate::CY, McGateFamily::Pauli),
        (StandardGate::CZ, McGateFamily::Pauli),
        (StandardGate::CRX, McGateFamily::Rotation),
        (StandardGate::CRY, McGateFamily::Rotation),
        (StandardGate::CRZ, McGateFamily::Rotation),
        (StandardGate::FSIM, McGateFamily::Fsim),
    ];

    for (gate, expected) in cases {
        assert_eq!(McGateFamily::classify(gate), expected, "{gate:?}");
    }
}

#[test]
fn validation_rejects_qubit_count_mismatch() {
    let gate = MCGate::new(1, StandardGate::X);
    let config = McGateDecomposeConfig::default();

    let err = McGateOperandView::new(&gate, &[Qubit::new(0)], &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "qubit count mismatch");
}

#[test]
fn validation_rejects_parameter_count_mismatch() {
    let gate = MCGate::new(1, StandardGate::RX);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig::default();

    let err = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "parameter count mismatch");
}

#[test]
fn validation_rejects_duplicate_operands() {
    let gate = MCGate::new(2, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig::default();

    let err = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "operands must be distinct");
}

#[test]
fn validation_rejects_duplicate_clean_ancillas() {
    let gate = MCGate::new(1, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::CleanAncilla,
        clean_ancillas: vec![Qubit::new(2), Qubit::new(2)],
        ..McGateDecomposeConfig::default()
    };

    let err = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "clean ancillas must be distinct");
}

#[test]
fn validation_rejects_dirty_ancilla_operand_overlap() {
    let gate = MCGate::new(1, StandardGate::X);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let config = McGateDecomposeConfig {
        ancilla_mode: AncillaMode::DirtyAncilla,
        dirty_ancillas: vec![Qubit::new(1)],
        ..McGateDecomposeConfig::default()
    };

    let err = McGateOperandView::new(&gate, &qubits, &[], &config).unwrap_err();

    assert_transform_failed_contains(err, "dirty ancillas must not overlap");
}

#[test]
fn validation_rejects_controlled_gphase() {
    let gate = MCGate::new(1, StandardGate::GPhase);
    let params = [CircuitParam::Fixed(0.25)];
    let config = McGateDecomposeConfig::default();

    let err = McGateOperandView::new(&gate, &[Qubit::new(0)], &params, &config).unwrap_err();

    assert_transform_failed_contains(err, "controlled GPhase");
}

#[test]
fn identity_decomposition_emits_no_operations() {
    let gate = MCGate::new(3, StandardGate::I);
    let qubits = [Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &[], &config).unwrap();

    assert!(operations.is_empty());
}

#[test]
fn rx_rotation_family_is_implemented() {
    let gate = MCGate::new(1, StandardGate::RX);
    let qubits = [Qubit::new(0), Qubit::new(1)];
    let params = [CircuitParam::Fixed(0.25)];
    let config = McGateDecomposeConfig::default();

    let operations = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();

    assert_eq!(operations.len(), 1);
}

#[test]
fn operation_interface_decomposes_mc_gate() {
    let operation = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![],
        label: None,
    };

    let result =
        decompose_mc_gate_operation(&operation, &McGateDecomposeConfig::default()).unwrap();

    assert!(result.changed);
    assert!(result.notes.is_empty());
    assert_eq!(result.operations.len(), 1);
    assert!(matches!(
        result.operations[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert_eq!(
        result.operations[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
}

#[test]
fn operation_interface_copies_non_mc_gate() {
    let operation = Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: Some("keep".into()),
    };

    let result =
        decompose_mc_gate_operation(&operation, &McGateDecomposeConfig::default()).unwrap();

    assert!(!result.changed);
    assert_eq!(result.operations.len(), 1);
    assert!(matches!(
        result.operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(result.operations[0].label.as_deref(), Some("keep"));
}

#[test]
fn operation_interface_skips_labeled_mc_gate_by_default() {
    let operation = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![],
        label: Some("protected".into()),
    };

    let result =
        decompose_mc_gate_operation(&operation, &McGateDecomposeConfig::default()).unwrap();

    assert!(!result.changed);
    assert_eq!(result.operations.len(), 1);
    assert!(matches!(
        result.operations[0].instruction,
        Instruction::McGate(_)
    ));
    assert_eq!(result.operations[0].label.as_deref(), Some("protected"));
}

#[test]
fn operation_interface_decomposes_labeled_mc_gate_when_enabled() {
    let operation = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![],
        label: Some("lower".into()),
    };
    let config = McGateDecomposeConfig::new().skip_labeled_ops(false);

    let result = decompose_mc_gate_operation(&operation, &config).unwrap();

    assert!(result.changed);
    assert_eq!(result.operations.len(), 1);
    assert!(matches!(
        result.operations[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert_eq!(result.operations[0].label, None);
}

#[test]
fn operation_interface_matches_direct_parameterized_decomposition() {
    let gate = MCGate::new(1, StandardGate::RX);
    let qubits = smallvec![Qubit::new(0), Qubit::new(1)];
    let params = smallvec![CircuitParam::Fixed(0.25)];
    let config = McGateDecomposeConfig::default();
    let operation = Operation {
        instruction: Instruction::McGate(Box::new(gate.clone())),
        qubits: qubits.clone(),
        params: params.clone(),
        label: None,
    };

    let direct = decompose_mc_gate(&gate, &qubits, &params, &config).unwrap();
    let result = decompose_mc_gate_operation(&operation, &config).unwrap();

    assert!(result.changed);
    assert_eq!(result.operations.len(), direct.len());
    for (actual, expected) in result.operations.iter().zip(direct.iter()) {
        match (&actual.instruction, &expected.instruction) {
            (Instruction::Standard(actual_gate), Instruction::Standard(expected_gate)) => {
                assert_eq!(actual_gate, expected_gate);
            }
            other => panic!("expected standard-gate pair, got {other:?}"),
        }
        assert_eq!(actual.qubits, expected.qubits);
        assert_eq!(actual.params.len(), expected.params.len());
        for (actual_param, expected_param) in actual.params.iter().zip(expected.params.iter()) {
            match (actual_param, expected_param) {
                (CircuitParam::Fixed(actual), CircuitParam::Fixed(expected)) => {
                    assert_eq!(actual, expected);
                }
                (CircuitParam::Index(actual), CircuitParam::Index(expected)) => {
                    assert_eq!(actual, expected);
                }
                other => panic!("parameter mismatch: {other:?}"),
            }
        }
        assert_eq!(actual.label, None);
    }
}
