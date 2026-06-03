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

use super::{McGateDecomposeConfig, decompose_mc_gates, decompose_mc_gates_for_device};
use crate::circuit::{
    Circuit, CircuitParam, ConditionView, ControlFlow, Instruction, MCGate, Operation, Parameter,
    ParameterValue, Qubit, StandardGate, WhileLoopGate, circuit_to_matrix,
};
use crate::compiler::CompilerError;
use crate::compiler::resource::{ResourceLimits, ResourcePolicy};
use crate::device::{Device, PhysicalQubit, Topology};
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::smallvec;
use std::collections::{HashMap, HashSet};

const EPSILON: f64 = 1e-9;

fn config(max_clean: usize, allow_dirty: bool) -> McGateDecomposeConfig {
    McGateDecomposeConfig {
        resource_policy: ResourcePolicy {
            max_pre_layout_clean_ancillas: max_clean,
            allow_dirty_borrowing: allow_dirty,
        },
        resource_limits: ResourceLimits::default(),
    }
}

fn append_mcx(circuit: &mut Circuit, controls: &[Qubit], target: Qubit, label: Option<&str>) {
    let mut qubits = controls.to_vec();
    qubits.push(target);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(controls.len() as u8, StandardGate::X))),
            qubits,
            [],
            label,
        )
        .unwrap();
}

fn assert_no_mc_gates(operations: &[Operation]) {
    for operation in operations {
        match &operation.instruction {
            Instruction::McGate(gate) => panic!("unexpected residual multi-controlled gate {gate}"),
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                assert_no_mc_gates(gate.true_body());
                if let Some(false_body) = gate.false_body() {
                    assert_no_mc_gates(false_body);
                }
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                assert_no_mc_gates(gate.body());
            }
            _ => {}
        }
    }
}

fn assert_matrix_approx_eq(actual: &Array2<Complex64>, expected: &Array2<Complex64>) {
    assert_eq!(actual.shape(), expected.shape());
    for ((row, column), expected) in expected.indexed_iter() {
        let actual = actual[[row, column]];
        assert!(
            (actual - expected).norm() < EPSILON,
            "matrix mismatch at row {row}, column {column}: actual={actual}, expected={expected}"
        );
    }
}

fn assert_selected_columns_equal_up_to_global_phase(
    context: &str,
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    columns: impl IntoIterator<Item = usize>,
) {
    assert_eq!(actual.shape(), expected.shape(), "{context}");
    let columns = columns.into_iter().collect::<Vec<_>>();
    let (reference_actual, reference_expected) = columns
        .iter()
        .flat_map(|column| {
            (0..expected.nrows()).map(move |row| (actual[[row, *column]], expected[[row, *column]]))
        })
        .find(|(_, expected)| expected.norm() > EPSILON)
        .expect("selected expected columns must contain a nonzero amplitude");
    let global_phase = reference_actual / reference_expected;

    assert!(
        (global_phase.norm() - 1.0).abs() < EPSILON,
        "{context}: invalid global phase {global_phase}"
    );
    for column in columns {
        for row in 0..expected.nrows() {
            let expected_amplitude = global_phase * expected[[row, column]];
            assert!(
                (actual[[row, column]] - expected_amplitude).norm() < EPSILON,
                "{context}: matrix mismatch at row {row}, column {column}: actual={}, expected={expected_amplitude}",
                actual[[row, column]]
            );
        }
    }
}

fn mc_gate_circuit(
    num_qubits: usize,
    additional_controls: u8,
    base_gate: StandardGate,
    params: &[f64],
) -> Circuit {
    let gate = MCGate::new(additional_controls, base_gate);
    assert!(num_qubits >= gate.num_qubits());
    let mut circuit = Circuit::new(num_qubits);
    circuit
        .append(
            Instruction::McGate(Box::new(gate.clone())),
            (0..gate.num_qubits()).map(|index| Qubit::new(index as u32)),
            params.iter().copied().map(ParameterValue::Fixed),
            None,
        )
        .unwrap();
    circuit
}

fn operations_use_qubit(operations: &[Operation], qubit: Qubit) -> bool {
    operations.iter().any(|operation| {
        operation.qubits.contains(&qubit)
            || match &operation.instruction {
                Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                    operations_use_qubit(gate.true_body(), qubit)
                        || gate
                            .false_body()
                            .is_some_and(|body| operations_use_qubit(body, qubit))
                }
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                    operations_use_qubit(gate.body(), qubit)
                }
                _ => false,
            }
    })
}

fn test_device(num_qubits: u32, invalid: &[u32]) -> Device {
    let qubits = (0..num_qubits).map(PhysicalQubit::new).collect::<Vec<_>>();
    let topology = Topology::new(qubits.clone(), vec![]).unwrap();
    Device::new(
        "mc-gate-test",
        qubits.iter().copied().collect::<HashSet<_>>(),
        topology,
    )
    .unwrap()
    .with_invalid_qubits(invalid.iter().copied().map(PhysicalQubit::new).collect())
    .unwrap()
}

#[test]
fn circuit_without_mc_gate_is_preserved() {
    let mut circuit = Circuit::new(1);
    circuit.set_global_phase(Parameter::symbol("phi"));
    circuit
        .append(
            Instruction::Standard(StandardGate::H),
            [Qubit::new(0)],
            [],
            Some("keep"),
        )
        .unwrap();

    let result = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap();

    assert!(!result.changed);
    assert_eq!(result.circuit.num_qubits(), 1);
    assert_eq!(result.circuit.global_phase(), Parameter::symbol("phi"));
    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(
        result.circuit.operations()[0].label.as_deref(),
        Some("keep")
    );
}

#[test]
fn zero_control_wrapper_lowers_to_original_standard_gate() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(0, StandardGate::RXY))),
            [Qubit::new(0)],
            [ParameterValue::Fixed(0.25), ParameterValue::Fixed(-0.5)],
            Some("drop"),
        )
        .unwrap();

    let result = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap();

    assert!(result.changed);
    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::RXY)
    ));
    assert_eq!(result.circuit.operations()[0].label, None);
}

#[test]
fn clean_ancillas_are_selected_first_and_reused_across_gates() {
    let mut circuit = Circuit::new(5);
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    append_mcx(&mut circuit, &controls, Qubit::new(3), None);
    append_mcx(&mut circuit, &controls, Qubit::new(4), None);

    let result = decompose_mc_gates(&circuit, config(2, true)).unwrap();

    assert!(result.changed);
    assert_eq!(result.circuit.num_qubits(), 6);
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(5)
    ));
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn one_clean_ancilla_candidate_is_used_when_two_are_unavailable() {
    let mut circuit = Circuit::new(4);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        None,
    );

    let result = decompose_mc_gates(&circuit, config(1, false)).unwrap();

    assert_eq!(result.circuit.num_qubits(), 5);
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(4)
    ));
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn dirty_v_chain_is_used_when_clean_ancillas_are_forbidden() {
    let mut circuit = Circuit::new(7);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        Qubit::new(4),
        None,
    );

    let result = decompose_mc_gates(&circuit, config(0, true)).unwrap();

    assert_eq!(result.circuit.num_qubits(), 7);
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(5)
    ));
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(6)
    ));
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn ancillary_free_fallback_preserves_mcx_semantics() {
    let mut circuit = Circuit::new(4);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        Some("drop"),
    );
    let expected = circuit_to_matrix(&circuit, None).unwrap();

    let result = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap();
    let actual = circuit_to_matrix(&result.circuit, None).unwrap();

    assert_eq!(result.circuit.num_qubits(), 4);
    assert_no_mc_gates(result.circuit.operations());
    assert!(
        result
            .circuit
            .operations()
            .iter()
            .all(|operation| operation.label.is_none())
    );
    assert_matrix_approx_eq(&actual, &expected);
}

#[test]
fn supported_gate_families_dispatch_and_preserve_no_aux_semantics() {
    struct Case {
        name: &'static str,
        additional_controls: u8,
        base_gate: StandardGate,
        params: &'static [f64],
    }

    let cases = [
        Case {
            name: "X",
            additional_controls: 2,
            base_gate: StandardGate::X,
            params: &[],
        },
        Case {
            name: "CX",
            additional_controls: 2,
            base_gate: StandardGate::CX,
            params: &[],
        },
        Case {
            name: "CCX",
            additional_controls: 1,
            base_gate: StandardGate::CCX,
            params: &[],
        },
        Case {
            name: "Y",
            additional_controls: 2,
            base_gate: StandardGate::Y,
            params: &[],
        },
        Case {
            name: "CY",
            additional_controls: 2,
            base_gate: StandardGate::CY,
            params: &[],
        },
        Case {
            name: "Z",
            additional_controls: 2,
            base_gate: StandardGate::Z,
            params: &[],
        },
        Case {
            name: "CZ",
            additional_controls: 2,
            base_gate: StandardGate::CZ,
            params: &[],
        },
        Case {
            name: "RX",
            additional_controls: 2,
            base_gate: StandardGate::RX,
            params: &[0.31],
        },
        Case {
            name: "RY",
            additional_controls: 2,
            base_gate: StandardGate::RY,
            params: &[-0.27],
        },
        Case {
            name: "RZ",
            additional_controls: 2,
            base_gate: StandardGate::RZ,
            params: &[0.73],
        },
        Case {
            name: "CRX",
            additional_controls: 1,
            base_gate: StandardGate::CRX,
            params: &[0.19],
        },
        Case {
            name: "CRY",
            additional_controls: 1,
            base_gate: StandardGate::CRY,
            params: &[-0.41],
        },
        Case {
            name: "CRZ",
            additional_controls: 1,
            base_gate: StandardGate::CRZ,
            params: &[0.61],
        },
        Case {
            name: "S",
            additional_controls: 2,
            base_gate: StandardGate::S,
            params: &[],
        },
        Case {
            name: "SDG",
            additional_controls: 2,
            base_gate: StandardGate::SDG,
            params: &[],
        },
        Case {
            name: "T",
            additional_controls: 2,
            base_gate: StandardGate::T,
            params: &[],
        },
        Case {
            name: "TDG",
            additional_controls: 2,
            base_gate: StandardGate::TDG,
            params: &[],
        },
        Case {
            name: "Phase",
            additional_controls: 2,
            base_gate: StandardGate::Phase,
            params: &[0.47],
        },
        Case {
            name: "H",
            additional_controls: 2,
            base_gate: StandardGate::H,
            params: &[],
        },
        Case {
            name: "U",
            additional_controls: 2,
            base_gate: StandardGate::U,
            params: &[0.23, -0.37, 0.53],
        },
        Case {
            name: "RXX",
            additional_controls: 2,
            base_gate: StandardGate::RXX,
            params: &[0.29],
        },
        Case {
            name: "RYY",
            additional_controls: 2,
            base_gate: StandardGate::RYY,
            params: &[-0.43],
        },
        Case {
            name: "RZZ",
            additional_controls: 2,
            base_gate: StandardGate::RZZ,
            params: &[0.59],
        },
        Case {
            name: "RZX",
            additional_controls: 2,
            base_gate: StandardGate::RZX,
            params: &[-0.67],
        },
        Case {
            name: "SWAP",
            additional_controls: 2,
            base_gate: StandardGate::SWAP,
            params: &[],
        },
        Case {
            name: "X2P",
            additional_controls: 2,
            base_gate: StandardGate::X2P,
            params: &[],
        },
        Case {
            name: "X2M",
            additional_controls: 2,
            base_gate: StandardGate::X2M,
            params: &[],
        },
        Case {
            name: "Y2P",
            additional_controls: 2,
            base_gate: StandardGate::Y2P,
            params: &[],
        },
        Case {
            name: "Y2M",
            additional_controls: 2,
            base_gate: StandardGate::Y2M,
            params: &[],
        },
        Case {
            name: "XY2P",
            additional_controls: 2,
            base_gate: StandardGate::XY2P,
            params: &[0.17],
        },
        Case {
            name: "XY2M",
            additional_controls: 2,
            base_gate: StandardGate::XY2M,
            params: &[-0.13],
        },
        Case {
            name: "FSIM",
            additional_controls: 2,
            base_gate: StandardGate::FSIM,
            params: &[0.37, -0.21],
        },
    ];

    for case in cases {
        let mc_gate = MCGate::new(case.additional_controls, case.base_gate);
        let circuit = mc_gate_circuit(
            mc_gate.num_qubits(),
            case.additional_controls,
            case.base_gate,
            case.params,
        );
        let expected = circuit_to_matrix(&circuit, None).unwrap();

        let result = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap();
        let actual = circuit_to_matrix(&result.circuit, None).unwrap();

        assert_eq!(
            result.circuit.num_qubits(),
            circuit.num_qubits(),
            "{} should use the ancillary-free fallback",
            case.name
        );
        assert_no_mc_gates(result.circuit.operations());
        assert_selected_columns_equal_up_to_global_phase(
            case.name,
            &actual,
            &expected,
            0..expected.ncols(),
        );
    }
}

#[test]
fn clean_ancilla_paths_preserve_semantics_on_the_clean_subspace() {
    struct Case {
        name: &'static str,
        additional_controls: u8,
        base_gate: StandardGate,
        params: &'static [f64],
        expected_ancillas: usize,
    }

    let cases = [
        Case {
            name: "MCX one-clean fallback",
            additional_controls: 3,
            base_gate: StandardGate::X,
            params: &[],
            expected_ancillas: 1,
        },
        Case {
            name: "multi-controlled RZ clean accumulator",
            additional_controls: 3,
            base_gate: StandardGate::RZ,
            params: &[0.41],
            expected_ancillas: 2,
        },
        Case {
            name: "multi-controlled SWAP clean accumulator",
            additional_controls: 3,
            base_gate: StandardGate::SWAP,
            params: &[],
            expected_ancillas: 2,
        },
        Case {
            name: "multi-controlled FSIM clean accumulator",
            additional_controls: 2,
            base_gate: StandardGate::FSIM,
            params: &[0.31, -0.23],
            expected_ancillas: 2,
        },
    ];

    for case in cases {
        let mc_gate = MCGate::new(case.additional_controls, case.base_gate);
        let source_width = mc_gate.num_qubits();
        let circuit = mc_gate_circuit(
            source_width,
            case.additional_controls,
            case.base_gate,
            case.params,
        );
        let result = decompose_mc_gates(&circuit, config(case.expected_ancillas, false)).unwrap();
        let expected = circuit_to_matrix(
            &mc_gate_circuit(
                result.circuit.num_qubits(),
                case.additional_controls,
                case.base_gate,
                case.params,
            ),
            None,
        )
        .unwrap();
        let actual = circuit_to_matrix(&result.circuit, None).unwrap();
        let clean_mask =
            (source_width..result.circuit.num_qubits()).fold(0, |mask, index| mask | (1 << index));
        let clean_columns = (0..expected.ncols()).filter(|column| column & clean_mask == 0);

        assert_eq!(
            result.circuit.num_qubits(),
            source_width + case.expected_ancillas,
            "{}",
            case.name
        );
        for index in source_width..result.circuit.num_qubits() {
            assert!(
                operations_use_qubit(result.circuit.operations(), Qubit::new(index as u32)),
                "{} did not consume allocated Q{index}",
                case.name
            );
        }
        assert_no_mc_gates(result.circuit.operations());
        assert_selected_columns_equal_up_to_global_phase(
            case.name,
            &actual,
            &expected,
            clean_columns,
        );
    }
}

#[test]
fn dirty_ancilla_path_restores_arbitrary_borrowed_states() {
    let mut circuit = Circuit::new(7);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        Qubit::new(4),
        None,
    );
    let expected = circuit_to_matrix(&circuit, None).unwrap();

    let result = decompose_mc_gates(&circuit, config(0, true)).unwrap();
    let actual = circuit_to_matrix(&result.circuit, None).unwrap();

    assert_eq!(result.circuit.num_qubits(), circuit.num_qubits());
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(5)
    ));
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(6)
    ));
    assert_no_mc_gates(result.circuit.operations());
    assert_selected_columns_equal_up_to_global_phase(
        "dirty borrowed MCX",
        &actual,
        &expected,
        0..expected.ncols(),
    );
}

#[test]
fn two_clean_ancilla_candidate_is_selected_only_when_both_qubits_are_used() {
    let mut circuit = Circuit::new(7);
    append_mcx(
        &mut circuit,
        &[
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
            Qubit::new(5),
        ],
        Qubit::new(6),
        None,
    );

    let result = decompose_mc_gates(&circuit, config(2, false)).unwrap();

    assert_eq!(result.circuit.num_qubits(), 9);
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(7)
    ));
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(8)
    ));
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn symbolic_rotation_parameters_are_reinterned_and_bindable() {
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::RZ))),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            [ParameterValue::Param(Parameter::symbol("theta"))],
            None,
        )
        .unwrap();

    let result = decompose_mc_gates(&circuit, config(1, false)).unwrap();
    let bound = result
        .circuit
        .assign_parameters(&Some(HashMap::from([("theta", 0.375)])))
        .unwrap();

    assert!(result.circuit.symbols().contains("theta"));
    assert_no_mc_gates(result.circuit.operations());
    circuit_to_matrix(&bound, None).unwrap();
}

#[test]
fn if_else_and_while_bodies_are_rebuilt_recursively() {
    let mut circuit = Circuit::new(4);
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let mcx = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
        qubits: smallvec![controls[0], controls[1], controls[2], Qubit::new(3)],
        params: smallvec![],
        label: Some("drop".into()),
    };
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![mcx.clone()],
            Some(vec![mcx.clone()]),
        )
        .unwrap();
    circuit
        .while_loop(ConditionView::new(Qubit::new(1), 1), vec![mcx])
        .unwrap();

    let result = decompose_mc_gates(&circuit, config(2, false)).unwrap();

    assert_eq!(result.circuit.num_qubits(), 5);
    assert_no_mc_gates(result.circuit.operations());
    for operation in result.circuit.operations() {
        assert_eq!(
            operation.qubits.as_slice(),
            &[
                Qubit::new(0),
                Qubit::new(1),
                Qubit::new(2),
                Qubit::new(3),
                Qubit::new(4),
            ]
        );
    }
}

#[test]
fn nested_control_flow_bodies_are_rebuilt_to_arbitrary_depth() {
    let mcx = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        params: smallvec![],
        label: None,
    };
    let nested_while = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
            ConditionView::new(Qubit::new(1), 1),
            vec![mcx],
        ))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        params: smallvec![],
        label: Some("keep-inner".into()),
    };
    let mut circuit = Circuit::new(4);
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![nested_while],
            None,
        )
        .unwrap();

    let result = decompose_mc_gates(&circuit, config(1, false)).unwrap();

    assert_eq!(result.circuit.num_qubits(), 5);
    assert_no_mc_gates(result.circuit.operations());
    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
        ]
    );
    let Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if/else");
    };
    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_loop)) =
        &if_else.true_body()[0].instruction
    else {
        panic!("expected nested while loop");
    };
    assert_eq!(while_loop.num_qubits(), 5);
    assert_eq!(if_else.true_body()[0].label.as_deref(), Some("keep-inner"));
}

#[test]
fn symbolic_parameter_inside_while_body_is_reinterned() {
    let mut circuit = Circuit::new(3);
    let (theta, _) = circuit.add_parameter(Parameter::symbol("theta"));
    let body = vec![Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(2, StandardGate::RZ))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        params: smallvec![CircuitParam::Index(theta as u32)],
        label: None,
    }];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();

    let result = decompose_mc_gates(&circuit, config(1, false)).unwrap();

    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected while loop");
    };
    assert_no_mc_gates(gate.body());
    assert!(result.circuit.symbols().contains("theta"));
    for operation in gate.body() {
        for param in &operation.params {
            if let CircuitParam::Index(index) = param {
                assert!(
                    result
                        .circuit
                        .parameters()
                        .get_index(*index as usize)
                        .is_some()
                );
            }
        }
    }
}

#[test]
fn device_capacity_limits_clean_allocation_but_allows_no_aux_fallback() {
    let mut circuit = Circuit::new(4);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        None,
    );
    let device = test_device(5, &[4]);

    let result = decompose_mc_gates_for_device(
        &circuit,
        &device,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 2,
            allow_dirty_borrowing: false,
        },
    )
    .unwrap();

    assert_eq!(device.num_usable_qubits(), 4);
    assert_eq!(result.circuit.num_qubits(), 4);
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn explicit_resource_limit_forces_no_aux_fallback_without_device() {
    let mut circuit = Circuit::new(4);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        None,
    );
    let mut limited = config(2, false);
    limited.resource_limits = ResourceLimits {
        max_total_qubits: Some(4),
    };

    let result = decompose_mc_gates(&circuit, limited).unwrap();

    assert_eq!(result.circuit.num_qubits(), 4);
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn device_capacity_allows_clean_allocation_when_space_exists() {
    let mut circuit = Circuit::new(4);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        Qubit::new(3),
        None,
    );
    let device = test_device(5, &[]);

    let result = decompose_mc_gates_for_device(
        &circuit,
        &device,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 2,
            allow_dirty_borrowing: false,
        },
    )
    .unwrap();

    assert_eq!(result.circuit.num_qubits(), 5);
    assert!(operations_use_qubit(
        result.circuit.operations(),
        Qubit::new(4)
    ));
    assert_no_mc_gates(result.circuit.operations());
}

#[test]
fn device_capacity_rejects_source_circuit_that_is_already_too_wide() {
    let circuit = Circuit::new(5);
    let device = test_device(4, &[]);

    let error =
        decompose_mc_gates_for_device(&circuit, &device, ResourcePolicy::default()).unwrap_err();

    assert!(matches!(error, CompilerError::InvalidInput(message) if message.contains("capacity")));
}

#[test]
fn decomposition_is_deterministic_across_runs() {
    let mut circuit = Circuit::new(5);
    let controls = [Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    append_mcx(&mut circuit, &controls, Qubit::new(3), None);
    append_mcx(&mut circuit, &controls, Qubit::new(4), None);

    let first = decompose_mc_gates(&circuit, config(2, true)).unwrap();
    let second = decompose_mc_gates(&circuit, config(2, true)).unwrap();

    assert_eq!(first.circuit.qubits(), second.circuit.qubits());
    assert_eq!(
        format!("{:?}", first.circuit.operations()),
        format!("{:?}", second.circuit.operations())
    );
    assert_eq!(first.circuit.parameters(), second.circuit.parameters());
}

#[test]
fn unsupported_controlled_gate_families_are_rejected_explicitly() {
    let cases = [
        ("I", StandardGate::I, &[][..]),
        ("RXY", StandardGate::RXY, &[0.25, -0.5][..]),
        ("XY", StandardGate::XY, &[0.37][..]),
        ("GPhase", StandardGate::GPhase, &[0.19][..]),
    ];

    for (name, gate, params) in cases {
        let mc_gate = MCGate::new(1, gate);
        let circuit = mc_gate_circuit(mc_gate.num_qubits(), 1, gate, params);

        let error = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap_err();

        assert!(matches!(
            error,
            CompilerError::TransformFailed {
                name: "decompose.mc_gates",
                reason
            } if reason.contains(name)
        ));
    }
}

#[test]
fn malformed_mc_gate_arity_is_rejected_before_planning() {
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            [],
            None,
        )
        .unwrap();

    let error = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("expects 4 qubits"))
    );
}

#[test]
fn malformed_mc_gate_parameter_arity_is_rejected_before_planning() {
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RZ))),
            [Qubit::new(0), Qubit::new(1)],
            [],
            None,
        )
        .unwrap();

    let error = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("expects 1 parameters"))
    );
}

#[test]
fn duplicate_mc_gate_qubits_are_rejected_by_synthesis() {
    let mut circuit = Circuit::new(3);
    append_mcx(
        &mut circuit,
        &[Qubit::new(0), Qubit::new(0)],
        Qubit::new(2),
        None,
    );

    let error = decompose_mc_gates(&circuit, config(2, true)).unwrap_err();

    assert!(
        matches!(error, CompilerError::TransformFailed { reason, .. } if reason.contains("duplicate Q0"))
    );
}

#[test]
fn missing_parameter_index_inside_control_flow_is_rejected() {
    let mut circuit = Circuit::new(2);
    let body = vec![Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RZ))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![CircuitParam::Index(999)],
        label: None,
    }];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();

    let error = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("missing parameter index 999"))
    );
}

#[test]
fn non_finite_fixed_parameter_inside_control_flow_is_rejected() {
    let mut circuit = Circuit::new(2);
    let body = vec![Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RZ))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1)],
        params: smallvec![CircuitParam::Fixed(f64::NAN)],
        label: None,
    }];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();

    let error = decompose_mc_gates(&circuit, McGateDecomposeConfig::default()).unwrap_err();

    assert!(
        matches!(error, CompilerError::InvalidInput(message) if message.contains("non-finite"))
    );
}
