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

//! Tests for the visualization IR builder module.

use super::*;
use crate::circuit::param::ParameterValue;
use crate::circuit::parameter::Parameter;
use crate::circuit::{Circuit, ConditionView, Instruction, Operation, Qubit, StandardGate};
use crate::visualization::circuit::model::{
    VisualChildren, VisualControlFlowKind, VisualOpStyle,
};
use crate::visualization::ParameterDisplayMode;
use smallvec::smallvec;
use std::f64::consts::PI;

fn q(index: usize) -> Qubit {
    let id = u32::try_from(index).expect("qubit index should fit in u32");
    Qubit::new(id)
}

#[test]
fn test_empty_circuit_builds_successfully() {
    let circuit = Circuit::new(0);
    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.num_qubits(), 0);
    assert!(visual.operations.is_empty());
}

#[test]
fn test_single_qubit_gates_build_correctly() {
    let mut circuit = Circuit::new(1);
    circuit.h(q(0)).unwrap();
    circuit.x(q(0)).unwrap();
    circuit.y(q(0)).unwrap();
    circuit.z(q(0)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 4);
    assert_eq!(visual.operations[0].label, "H");
    assert_eq!(visual.operations[1].label, "X");
    assert_eq!(visual.operations[2].label, "Y");
    assert_eq!(visual.operations[3].label, "Z");
}

#[test]
fn test_multi_qubit_gates_reserve_full_span() {
    let mut circuit = Circuit::new(4);
    circuit.cx(q(0), q(3)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].covered_lanes, vec![0, 1, 2, 3]);
}

#[test]
fn test_decompose_circuit_gates_option() {
    let mut sub = Circuit::new(2);
    sub.h(q(0)).unwrap();
    sub.cx(q(0), q(1)).unwrap();
    let sub_gate = sub.to_gate("SUB_BELL").unwrap();

    let mut circuit = Circuit::new(2);
    circuit
        .append(
            sub_gate,
            vec![q(0), q(1)],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();

    let visual_no_decompose =
        build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual_no_decompose.operations.len(), 1);
    assert_eq!(visual_no_decompose.operations[0].label, "SUB_BELL");

    let visual_decompose = build_visual_circuit(
        &circuit,
        &VisualBuildOptions {
            decompose_circuit_gates: true,
            ..VisualBuildOptions::default()
        },
    )
    .unwrap();
    assert_eq!(visual_decompose.operations.len(), 2);
    assert_eq!(visual_decompose.operations[0].label, "H");
    assert_eq!(visual_decompose.operations[1].label, "X");
}

#[test]
fn test_barrier_with_empty_qubits_reserves_all_lanes() {
    let mut circuit = Circuit::new(3);
    circuit.barrier(vec![]).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].covered_lanes, vec![0, 1, 2]);
}

#[test]
fn test_delay_operation() {
    let mut circuit = Circuit::new(1);
    circuit.delay(q(0), ParameterValue::from(40.0)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "D");
    assert!(matches!(visual.operations[0].style, VisualOpStyle::Delay));
}

#[test]
fn test_reset_operation() {
    let mut circuit = Circuit::new(1);
    circuit.reset(q(0)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "R");
    assert!(matches!(visual.operations[0].style, VisualOpStyle::Reset));
}

#[test]
fn test_measure_operation() {
    let mut circuit = Circuit::new(1);
    circuit.measure(q(0)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "M");
    assert!(matches!(visual.operations[0].style, VisualOpStyle::Measure));
}

#[test]
fn test_swap_gate_has_dedicated_style() {
    let mut circuit = Circuit::new(2);
    circuit.swap(q(0), q(1)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert!(matches!(visual.operations[0].style, VisualOpStyle::Swap));
    assert_eq!(visual.operations[0].label, "SWAP");
}

#[test]
fn test_unitary_gate_with_label() {
    let mut circuit = Circuit::new(2);
    let unitary = crate::circuit::UnitaryGate::new("MY_UNITARY", 2);
    circuit.unitary(unitary, vec![q(0), q(1)]).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "MY_UNITARY");
    assert!(visual.operations[0].span_box);
}

#[test]
fn test_unitary_gate_fallback_to_default_label() {
    let mut circuit = Circuit::new(2);
    let unitary = crate::circuit::UnitaryGate::new("", 2);
    circuit.unitary(unitary, vec![q(0), q(1)]).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "Unitary");
}

#[test]
fn test_controlled_gate_label_strips_prefix() {
    let mut circuit = Circuit::new(2);
    circuit.crx(q(0), q(1), PI / 4.0).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "RX");
    assert!(matches!(
        visual.operations[0].style,
        VisualOpStyle::Controlled { num_controls: 1 }
    ));
}

#[test]
fn test_ccx_gate_has_two_controls() {
    let mut circuit = Circuit::new(3);
    circuit.ccx(q(0), q(1), q(2)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "X");
    assert!(matches!(
        visual.operations[0].style,
        VisualOpStyle::Controlled { num_controls: 2 }
    ));
}

#[test]
fn test_fsim_gate() {
    let mut circuit = Circuit::new(2);
    circuit.fsim(q(0), q(1), 0.21, -0.44).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "FSIM");
}

#[test]
fn test_sdg_tdg_gate_labels() {
    let mut circuit = Circuit::new(2);
    circuit.sdg(q(0)).unwrap();
    circuit.tdg(q(1)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 2);
    assert_eq!(visual.operations[0].label, "SD");
    assert_eq!(visual.operations[1].label, "TD");
}

#[test]
fn test_phase_gate_labeled_as_p() {
    let mut circuit = Circuit::new(1);
    circuit.phase(q(0), PI / 4.0).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    assert_eq!(visual.operations[0].label, "P");
}

#[test]
fn test_if_else_control_flow_children() {
    let mut circuit = Circuit::new(2);
    let condition = ConditionView::new(q(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q(1)],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q(1)],
        params: smallvec![],
        label: None,
    }];
    circuit
        .if_else(condition, true_body, Some(false_body))
        .unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    let op = &visual.operations[0];
    assert!(op.label.starts_with("IF q0=1"));
    assert!(matches!(
        op.style,
        VisualOpStyle::ControlFlow {
            kind: VisualControlFlowKind::IfElseBlock {
                has_false_branch: true,
                ..
            },
        }
    ));
    match op.children.as_ref() {
        Some(VisualChildren::IfElse {
            then_circuit,
            else_circuit,
        }) => {
            assert_eq!(then_circuit.operations.len(), 1);
            assert_eq!(then_circuit.operations[0].label, "X");
            assert!(else_circuit.is_some());
            assert_eq!(else_circuit.as_ref().unwrap().operations.len(), 1);
            assert_eq!(else_circuit.as_ref().unwrap().operations[0].label, "Z");
        }
        _ => panic!("expected IfElse children"),
    }
}

#[test]
fn test_if_without_else_control_flow() {
    let mut circuit = Circuit::new(2);
    let condition = ConditionView::new(q(0), 0);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q(1)],
        params: smallvec![],
        label: None,
    }];
    circuit.if_else(condition, true_body, None).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    let op = &visual.operations[0];
    match op.children.as_ref() {
        Some(VisualChildren::IfElse {
            then_circuit,
            else_circuit,
        }) => {
            assert_eq!(then_circuit.operations.len(), 1);
            assert!(else_circuit.is_none());
        }
        _ => panic!("expected IfElse children"),
    }
}

#[test]
fn test_while_loop_control_flow_children() {
    let mut circuit = Circuit::new(2);
    let condition = ConditionView::new(q(0), 0);
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q(0)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![q(0), q(1)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit.while_loop(condition, body).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations.len(), 1);
    let op = &visual.operations[0];
    assert!(op.label.starts_with("WH q0=0"));
    match op.children.as_ref() {
        Some(VisualChildren::While { body_circuit }) => {
            assert_eq!(body_circuit.operations.len(), 2);
            assert_eq!(body_circuit.operations[0].label, "H");
            assert_eq!(body_circuit.operations[1].label, "X");
        }
        _ => panic!("expected While children"),
    }
}

#[test]
fn test_parameter_format_numeric_mode() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), 1.2345).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations[0].params, vec!["1.23".to_string()]);
}

#[test]
fn test_parameter_format_pi_fraction_mode() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), PI / 2.0).unwrap();

    let visual = build_visual_circuit(
        &circuit,
        &VisualBuildOptions {
            parameter_format: crate::visualization::ParameterFormatOptions {
                mode: ParameterDisplayMode::PiFractionPreferred,
                ..crate::visualization::ParameterFormatOptions::default()
            },
            ..VisualBuildOptions::default()
        },
    )
    .unwrap();
    assert_eq!(visual.operations[0].params, vec!["π/2".to_string()]);
}

#[test]
fn test_parameter_format_scientific_notation_for_small_values() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), 0.0004).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations[0].params, vec!["4e-4".to_string()]);
}

#[test]
fn test_parameter_format_scientific_notation_for_large_values() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), 15000.0).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations[0].params, vec!["1.5e4".to_string()]);
}

#[test]
fn test_symbolic_parameter() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit.rx(q(0), theta.clone()).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.operations[0].params, vec!["theta".to_string()]);
}

#[test]
fn test_symbolic_parameter_expression() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit.rx(q(0), theta + 1.0).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert!(visual.operations[0].params[0].contains("theta"));
}

#[test]
fn test_multiple_operations_same_column_when_parallel() {
    let mut circuit = Circuit::new(2);
    circuit.h(q(0)).unwrap();
    circuit.x(q(1)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    let cols: Vec<usize> = visual.operations.iter().map(|op| op.column).collect();
    assert_eq!(cols, vec![0, 0]);
}

#[test]
fn test_operations_sequential_when_sharing_qubit() {
    let mut circuit = Circuit::new(1);
    circuit.h(q(0)).unwrap();
    circuit.x(q(0)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    let cols: Vec<usize> = visual.operations.iter().map(|op| op.column).collect();
    assert_eq!(cols, vec![0, 1]);
}

#[test]
fn test_num_columns_tracks_widest_schedule() {
    let mut circuit = Circuit::new(2);
    circuit.h(q(0)).unwrap();
    circuit.x(q(1)).unwrap();
    circuit.cx(q(0), q(1)).unwrap();

    let visual = build_visual_circuit(&circuit, &VisualBuildOptions::default()).unwrap();
    assert_eq!(visual.num_columns, 2);
}
