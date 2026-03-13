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

use super::*;
use crate::circuit::gate::control_flow::{ConditionView, ControlFlow};
use crate::circuit::gate::{Directive, Instruction, StandardGate};
use crate::circuit::param::ParameterValue;
use crate::circuit::{Circuit, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use smallvec::smallvec;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

fn line_topology(ids: &[u32]) -> Topology {
    let qubits: Vec<Qubit> = ids.iter().copied().map(Qubit::new).collect();
    let couplings = ids
        .windows(2)
        .map(|w| (Qubit::new(w[0]), Qubit::new(w[1]), "CX".to_string()))
        .collect();
    Topology::new(qubits, couplings).unwrap()
}

fn connected_undirected(topology: &Topology, a: Qubit, b: Qubit) -> bool {
    topology.is_connected(a, b) || topology.is_connected(b, a)
}

fn assert_mapped_ops_2q_edges(ops: &[Operation], topology: &Topology) {
    for op in ops {
        if op.qubits.len() == 2 {
            assert!(
                connected_undirected(topology, op.qubits[0], op.qubits[1]),
                "2q op is not on a topology edge: {:?}",
                op.qubits
            );
        }
        if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
            match control_flow {
                ControlFlow::IfElse(gate) => {
                    assert_mapped_ops_2q_edges(gate.true_body(), topology);
                    if let Some(false_body) = gate.false_body() {
                        assert_mapped_ops_2q_edges(false_body, topology);
                    }
                }
                ControlFlow::WhileLoop(gate) => {
                    assert_mapped_ops_2q_edges(gate.body(), topology);
                }
            }
        }
    }
}

fn assert_mapped_2q_edges(mapped: &Circuit, topology: &Topology) {
    assert_mapped_ops_2q_edges(mapped.operations(), topology);
}

fn count_swaps(circuit: &Circuit) -> usize {
    fn count_ops(ops: &[Operation]) -> usize {
        ops.iter()
            .map(|op| {
                let mut total = usize::from(matches!(
                    &op.instruction,
                    Instruction::Standard(StandardGate::SWAP)
                ));
                if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
                    match control_flow {
                        ControlFlow::IfElse(gate) => {
                            total += count_ops(gate.true_body());
                            if let Some(false_body) = gate.false_body() {
                                total += count_ops(false_body);
                            }
                        }
                        ControlFlow::WhileLoop(gate) => {
                            total += count_ops(gate.body());
                        }
                    }
                }
                total
            })
            .sum()
    }

    count_ops(circuit.operations())
}

fn append_test_operation(circuit: &mut Circuit, op: Operation) {
    circuit
        .append(
            op.instruction,
            op.qubits,
            std::iter::empty::<ParameterValue>(),
            op.label.as_deref(),
        )
        .unwrap();
}

fn collect_directives_recursive(ops: &[Operation]) -> Vec<Directive> {
    let mut out = Vec::new();
    for op in ops {
        if let Instruction::Directive(directive) = &op.instruction {
            out.push(*directive);
        }
        if let Instruction::ControlFlowGate(control_flow) = &op.instruction {
            match control_flow {
                ControlFlow::IfElse(gate) => {
                    out.extend(collect_directives_recursive(gate.true_body()));
                    if let Some(false_body) = gate.false_body() {
                        out.extend(collect_directives_recursive(false_body));
                    }
                }
                ControlFlow::WhileLoop(gate) => {
                    out.extend(collect_directives_recursive(gate.body()));
                }
            }
        }
    }
    out
}

fn fingerprint_ops(ops: &[Operation], out: &mut Vec<String>) {
    for op in ops {
        let mut qids: Vec<u32> = op.qubits.iter().map(Qubit::id).collect();
        qids.sort_unstable();
        match &op.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                out.push(format!(
                    "if:{:?}:{:?}:{:?}",
                    gate.condition(),
                    qids,
                    op.label.as_deref()
                ));
                fingerprint_ops(gate.true_body(), out);
                if let Some(false_body) = gate.false_body() {
                    out.push("if:false:some".into());
                    fingerprint_ops(false_body, out);
                } else {
                    out.push("if:false:none".into());
                }
                out.push("if:end".into());
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                out.push(format!(
                    "while:{:?}:{:?}:{:?}",
                    gate.condition(),
                    qids,
                    op.label.as_deref()
                ));
                fingerprint_ops(gate.body(), out);
                out.push("while:end".into());
            }
            _ => out.push(format!(
                "{:?}:{:?}:{:?}",
                &op.instruction,
                qids,
                op.label.as_deref()
            )),
        }
    }
}

fn fingerprint(circuit: &Circuit) -> Vec<String> {
    let mut out = Vec::new();
    fingerprint_ops(circuit.operations(), &mut out);
    out
}

#[test]
fn test_module_exports_compile_and_device() {
    let _cfg = crate::compile::SabreConfig::default();
    let _topology = crate::device::Topology::new(vec![Qubit::new(0)], vec![]).unwrap();
}

#[test]
fn test_vf2_map_while_loop_preserves_structure_and_condition() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    circuit.measure(q0).unwrap();

    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![q1, q2],
        params: smallvec![],
        label: None,
    }];
    circuit.while_loop(ConditionView::new(q0, 1), body).unwrap();

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);

    assert!(matches!(
        &mapped.operations()[1].instruction,
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(_))
    ));
    match &mapped.operations()[1].instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
            assert_eq!(gate.body().len(), 1);
            assert!(matches!(
                &gate.body()[0].instruction,
                Instruction::Standard(StandardGate::CX)
            ));
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_vf2_preserves_control_flow_metadata_and_empty_else() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.set_global_phase(Parameter::from(0.25));
    circuit.measure(Qubit::new(0)).unwrap();

    let labeled_if = build_if_else_operation(
        ConditionView::new(Qubit::new(0), 1),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        }],
        None,
        Some("branch_label".into()),
    );
    append_test_operation(&mut circuit, labeled_if);

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
    assert_eq!(mapped.global_phase(), Parameter::from(0.25));

    match &mapped.operations()[1] {
        Operation {
            instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(gate)),
            label,
            ..
        } => {
            assert_eq!(label.as_deref(), Some("branch_label"));
            assert_eq!(gate.condition().target, 1);
            assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
            assert!(gate.false_body().is_none());
        }
        _ => panic!("expected labeled mapped if_else operation"),
    }
}

#[test]
fn test_preserve_measure_barrier_and_reset() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.reset(Qubit::new(1)).unwrap();

    let topology = line_topology(&[0, 1]);
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    let directives: Vec<Directive> = mapped
        .operations()
        .iter()
        .filter_map(|op| match &op.instruction {
            Instruction::Directive(directive) => Some(*directive),
            _ => None,
        })
        .collect();
    assert_eq!(
        directives,
        vec![Directive::Barrier, Directive::Measure, Directive::Reset]
    );
}

#[test]
fn test_reject_delay() {
    let mut circuit = Circuit::new(1);
    circuit
        .delay(Qubit::new(0), ParameterValue::Fixed(100.0))
        .unwrap();

    let topology = line_topology(&[0, 1]);
    let err = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
    assert!(matches!(
        err,
        CompileError::UnsupportedInstruction {
            instruction,
            op_index: 0
        } if instruction == "Delay"
    ));
}

#[test]
fn test_vf2_map_if_else_preserves_structure_and_condition() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![Qubit::new(1), Qubit::new(2)],
        params: smallvec![],
        label: None,
    }];
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            true_body,
            Some(false_body),
        )
        .unwrap();

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);

    assert!(matches!(
        &mapped.operations()[1].instruction,
        Instruction::ControlFlowGate(ControlFlow::IfElse(_))
    ));
    match &mapped.operations()[1].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            assert_eq!(gate.condition().qubit, mapped.operations()[0].qubits[0]);
            assert_eq!(gate.true_body().len(), 1);
            assert_eq!(gate.false_body().unwrap().len(), 1);
            assert!(matches!(
                &gate.false_body().unwrap()[0].instruction,
                Instruction::Standard(StandardGate::CX)
            ));
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_vf2_isomorphic_on_nested_if_else() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(4);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(1)).unwrap();

    let inner_true = vec![Operation {
        instruction: Instruction::Standard(StandardGate::CX),
        qubits: smallvec![Qubit::new(2), Qubit::new(3)],
        params: smallvec![],
        label: None,
    }];
    let nested_if = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::if_else(
            ConditionView::new(Qubit::new(1), 1),
            inner_true,
            None,
        )),
        qubits: smallvec![Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        params: smallvec![],
        label: None,
    };
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![nested_if],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::X),
                qubits: smallvec![Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
        )
        .unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    assert!(vf2.is_subgraph_isomorphic(&circuit).unwrap());
}

#[test]
fn test_vf2_maps_nested_if_else_inside_while_loop() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(1)).unwrap();

    let nested_if = build_if_else_operation(
        ConditionView::new(Qubit::new(1), 1),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(0), Qubit::new(1)],
            params: smallvec![],
            label: None,
        }],
        Some(vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        }]),
        None,
    );
    circuit
        .while_loop(
            ConditionView::new(Qubit::new(0), 1),
            vec![
                nested_if,
                Operation {
                    instruction: Instruction::Standard(StandardGate::X),
                    qubits: smallvec![Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
            ],
        )
        .unwrap();

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);

    match &mapped.operations()[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert!(matches!(
                &gate.body()[0].instruction,
                Instruction::ControlFlowGate(ControlFlow::IfElse(_))
            ));
        }
        _ => panic!("expected mapped while_loop with nested if_else"),
    }
}

#[test]
fn test_map_with_vf2_sabre_routes_if_else_and_continuation() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                params: smallvec![],
                label: None,
            }],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        repeat_iterations: 0,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();

    assert_mapped_2q_edges(&mapped, &topology);
    assert!(matches!(
        &mapped.operations()[1].instruction,
        Instruction::ControlFlowGate(ControlFlow::IfElse(_))
    ));
    assert!(count_swaps(&mapped) > 0);
    assert!(matches!(
        &mapped.operations().last().unwrap().instruction,
        Instruction::Standard(StandardGate::CX)
    ));
}

#[test]
fn test_map_with_vf2_sabre_routes_while_loop_and_continuation() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .while_loop(
            ConditionView::new(Qubit::new(0), 1),
            vec![
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
            ],
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        repeat_iterations: 0,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();

    assert_mapped_2q_edges(&mapped, &topology);
    match &mapped.operations()[1].instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert!(gate.body().len() > 3);
        }
        _ => panic!("expected mapped while_loop operation"),
    }
    assert!(matches!(
        &mapped.operations().last().unwrap().instruction,
        Instruction::Standard(StandardGate::CX)
    ));
}

#[test]
fn test_sabre_preserves_symbolic_global_phase_and_directives_inside_control_flow() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    let theta = Parameter::try_from("theta").unwrap();
    circuit.set_global_phase(theta.clone());
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .while_loop(
            ConditionView::new(Qubit::new(0), 1),
            vec![
                Operation {
                    instruction: Instruction::Directive(Directive::Measure),
                    qubits: smallvec![Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Directive(Directive::Reset),
                    qubits: smallvec![Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
            ],
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        repeat_iterations: 0,
        seed: 9,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
    assert_eq!(mapped.global_phase(), theta);

    match &mapped.operations()[1].instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert_eq!(
                collect_directives_recursive(gate.body()),
                vec![Directive::Measure, Directive::Reset]
            );
        }
        _ => panic!("expected mapped while_loop operation"),
    }
}

#[test]
fn test_sabre_maps_nested_while_loop_inside_if_else() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(1)).unwrap();

    let nested_while = build_while_loop_operation(
        ConditionView::new(Qubit::new(1), 1),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(0), Qubit::new(2)],
            params: smallvec![],
            label: None,
        }],
        None,
    );
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![nested_while],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        repeat_iterations: 0,
        seed: 17,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);

    match &mapped.operations()[2].instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            assert!(matches!(
                &gate.true_body()[0].instruction,
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(_))
            ));
        }
        _ => panic!("expected mapped if_else with nested while_loop"),
    }
}

#[test]
fn test_reject_unsupported_arity() {
    let mut circuit = Circuit::new(3);
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();

    let topology = line_topology(&[0, 1, 2, 3]);
    let err = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap_err();
    assert!(matches!(
        err,
        CompileError::UnsupportedArity {
            arity: 3,
            op_index: 0
        }
    ));
}

#[test]
fn test_invalid_fidelity_rejected() {
    let topology = line_topology(&[0, 1, 2]);
    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(0), Qubit::new(1)), 1.2);
    let err = Vf2Mapping::new(topology, Some(fidelity)).unwrap_err();
    assert!(matches!(err, CompileError::InvalidFidelity { .. }));
}

#[test]
fn test_missing_fidelity_defaults_to_one() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();

    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(0), Qubit::new(1)), 0.2);

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::Disabled,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, Some(&fidelity), &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_fidelity_pair_normalization() {
    let topology = line_topology(&[0, 1, 2]);
    let mut fidelity = FidelityMap::new();
    fidelity.insert((Qubit::new(2), Qubit::new(1)), 0.9);
    let _ = SabreMapping::new(topology, Some(fidelity), SabreConfig::default()).unwrap();
}

#[test]
fn test_vf2_fast_path_no_overhead() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
    circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert_eq!(mapped.operations().len(), circuit.operations().len());
    assert_eq!(count_swaps(&mapped), 0);
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_vf2_standalone_initial_layout_api() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(20), Qubit::new(30)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(20)).unwrap();
    circuit.cx(Qubit::new(20), Qubit::new(30)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let layout = vf2.find_initial_layout(&circuit).unwrap().unwrap();
    assert_eq!(layout.len(), 3);
}

#[test]
fn test_vf2_find_initial_layout_fallback_top1() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

    let layout = vf2.find_initial_layout(&circuit).unwrap();
    assert!(layout.is_some());
    assert_eq!(layout.unwrap().len(), 3);
}

#[test]
fn test_vf2_map_remains_strict_no_fallback() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let mut vf2 = Vf2Mapping::new(topology, None).unwrap();
    let err = vf2.execute(&circuit).unwrap_err();
    assert!(matches!(err, CompileError::Vf2NoMapping));
}

#[test]
fn test_vf2_candidates_topk_and_score_range() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
    circuit.x(Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 3,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(!candidates.is_empty());
    assert!(candidates.len() <= 3);
    for candidate in candidates {
        assert_eq!(candidate.logic2phy.len(), 3);
        assert_eq!(candidate.region.len(), 3);
        assert!((0.0..=1.0).contains(&candidate.score.total));
        assert!((0.0..=1.0).contains(&candidate.score.fidelity));
        assert!((0.0..=1.0).contains(&candidate.score.topology_fit));
        assert!((0.0..=1.0).contains(&candidate.score.gate_distribution));
    }
}

#[test]
fn test_vf2_candidates_deterministic_order() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 5,
        ..Vf2CandidateOptions::default()
    };
    let c1 = vf2
        .find_initial_layout_candidates(&circuit, Some(options.clone()))
        .unwrap();
    let c2 = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();

    let l1: Vec<Vec<u32>> = c1
        .iter()
        .map(|candidate| candidate.logic2phy.iter().map(Qubit::id).collect())
        .collect();
    let l2: Vec<Vec<u32>> = c2
        .iter()
        .map(|candidate| candidate.logic2phy.iter().map(Qubit::id).collect())
        .collect();
    assert_eq!(l1, l2);
}

#[test]
fn test_vf2_candidates_topk_zero() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 0,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.is_empty());
}

#[test]
fn test_vf2_candidates_topk_effective_when_strict_isomorphic() {
    let topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(3), Qubit::new(0), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 4,
        max_matches_per_subgraph: 16,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.len() > 1);
    assert!(candidates.len() <= 4);
}

#[test]
fn test_vf2_candidates_respect_max_matches_per_subgraph() {
    let topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(3), Qubit::new(0), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    let options = Vf2CandidateOptions {
        top_k: 8,
        max_matches_per_subgraph: 1,
        ..Vf2CandidateOptions::default()
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit, Some(options))
        .unwrap();
    assert!(candidates.len() <= 1);
}

#[test]
fn test_vf2_find_initial_layout_fallback_none_when_no_candidate() {
    let topology = Topology::new(vec![Qubit::new(0), Qubit::new(1)], vec![]).unwrap();
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let vf2 = Vf2Mapping::new(topology, None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());
    let layout = vf2.find_initial_layout(&circuit).unwrap();
    assert!(layout.is_none());
}

#[test]
fn test_vf2_isomorphic_on_dense_topology_non_induced_case() {
    let topology = Topology::new(
        vec![
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
        ],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(0), Qubit::new(2), "CX".to_string()),
            (Qubit::new(0), Qubit::new(3), "CX".to_string()),
            (Qubit::new(0), Qubit::new(4), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
            (Qubit::new(1), Qubit::new(3), "CX".to_string()),
            (Qubit::new(1), Qubit::new(4), "CX".to_string()),
            (Qubit::new(2), Qubit::new(3), "CX".to_string()),
            (Qubit::new(2), Qubit::new(4), "CX".to_string()),
            (Qubit::new(3), Qubit::new(4), "CX".to_string()),
        ],
    )
    .unwrap();
    let mut circuit = Circuit::new(5);
    circuit.cx(Qubit::new(2), Qubit::new(4)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(4)).unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(4), Qubit::new(3)).unwrap();
    circuit.cx(Qubit::new(3), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(3)).unwrap();

    let mut vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    assert!(vf2.is_subgraph_isomorphic(&circuit).unwrap());
    let mapped = vf2.execute(&circuit).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_policy_initial_only_routes_with_sabre() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        seed: 12345,
        initial_iterations: 2,
        repeat_iterations: 1,
        ..SabreConfig::default()
    };
    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &cfg).unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_sabre_fallback_and_state_exposure() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let vf2 = Vf2Mapping::new(topology.clone(), None).unwrap();
    assert!(!vf2.is_subgraph_isomorphic(&circuit).unwrap());

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert!(mapped.operations().len() > circuit.operations().len());
    assert_mapped_2q_edges(&mapped, &topology);

    let mut sabre = SabreMapping::new(topology, None, SabreConfig::default()).unwrap();
    let _ = sabre.execute(&circuit).unwrap();
    assert_eq!(sabre.logic2phy.len(), circuit.qubits().len());
}

#[test]
fn test_output_uses_only_physical_qubits_in_use() {
    let topology = line_topology(&[0, 1, 2, 3, 4]);
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();
    assert_eq!(mapped.qubits().len(), 2);
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_sabre_determinism_with_fixed_seed() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let mut circuit = Circuit::new(3);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        seed: 12345,
        initial_iterations: 3,
        repeat_iterations: 2,
        swap_iterations: 3,
        ..SabreConfig::default()
    };

    let mut sabre1 = SabreMapping::new(topology.clone(), None, cfg.clone()).unwrap();
    let mut sabre2 = SabreMapping::new(topology, None, cfg).unwrap();

    let out1 = sabre1.execute(&circuit).unwrap();
    let out2 = sabre2.execute(&circuit).unwrap();
    assert_eq!(fingerprint(&out1), fingerprint(&out2));
}

#[test]
fn test_sabre_control_flow_determinism_with_fixed_seed() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                params: smallvec![],
                label: None,
            }],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let cfg = SabreConfig {
        vf2_policy: Vf2Policy::InitialOnly,
        seed: 12345,
        initial_iterations: 3,
        repeat_iterations: 0,
        swap_iterations: 3,
        ..SabreConfig::default()
    };

    let mut sabre1 = SabreMapping::new(topology.clone(), None, cfg.clone()).unwrap();
    let mut sabre2 = SabreMapping::new(topology, None, cfg).unwrap();

    let out1 = sabre1.execute(&circuit).unwrap();
    let out2 = sabre2.execute(&circuit).unwrap();
    assert_eq!(fingerprint(&out1), fingerprint(&out2));
}

#[test]
fn test_non_contiguous_qubit_ids_supported() {
    let topology = line_topology(&[100, 200, 300, 400]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(30), Qubit::new(70)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(30)).unwrap();
    circuit.cx(Qubit::new(30), Qubit::new(70)).unwrap();

    let mapped = map_with_vf2_sabre(&circuit, &topology, None, &SabreConfig::default()).unwrap();

    let topo_set: HashSet<Qubit> = topology.qubits().collect();
    for q in mapped.qubits() {
        assert!(topo_set.contains(&q));
    }
    assert_mapped_2q_edges(&mapped, &topology);
}

fn test_circuit(width: usize) -> Circuit {
    let mut circuit = Circuit::new(width);
    if width > 1 {
        circuit
            .cx(Qubit::new(0), Qubit::new((width as u32) - 1))
            .unwrap();
    }
    circuit
}

fn fast_ga_config(seed: i64) -> GaConfig {
    let mut sabre_config = SabreConfig::default();
    sabre_config.repeat_iterations = 0;
    sabre_config.seed = seed;

    GaConfig {
        population: 4,
        update_iters: 2,
        seed,
        sabre_config,
        ..GaConfig::default()
    }
}

#[test]
fn test_map_with_ga_basic_success() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let circuit = test_circuit(3);
    let config = fast_ga_config(42);

    let result = map_with_ga(&circuit, &topology, &config, None, None);

    assert!(result.is_ok(), "GA mapping failed in basic scenario");
    let mapped_circuit = result.unwrap();

    assert!(mapped_circuit.operations().len() >= circuit.operations().len());
}

#[test]
fn test_map_with_ga_routes_if_else_and_continuation() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                params: smallvec![],
                label: None,
            }],
            Some(vec![Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                params: smallvec![],
                label: None,
            }]),
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let mapped = map_with_ga(&circuit, &topology, &fast_ga_config(77), None, None).unwrap();

    assert_mapped_2q_edges(&mapped, &topology);
    assert!(matches!(
        &mapped.operations()[1].instruction,
        Instruction::ControlFlowGate(ControlFlow::IfElse(_))
    ));
    assert!(count_swaps(&mapped) > 0);
    assert!(matches!(
        &mapped.operations().last().unwrap().instruction,
        Instruction::Standard(StandardGate::CX)
    ));
}

#[test]
fn test_map_with_ga_routes_while_loop_and_continuation() {
    let topology = line_topology(&[0, 1, 2]);
    let mut circuit = Circuit::new(3);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit
        .while_loop(
            ConditionView::new(Qubit::new(0), 1),
            vec![
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(1), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
                Operation {
                    instruction: Instruction::Standard(StandardGate::CX),
                    qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                    params: smallvec![],
                    label: None,
                },
            ],
        )
        .unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();

    let mapped = map_with_ga(&circuit, &topology, &fast_ga_config(78), None, None).unwrap();

    assert_mapped_2q_edges(&mapped, &topology);
    match &mapped.operations()[1].instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert!(gate.body().len() > 3);
        }
        _ => panic!("expected mapped while_loop operation"),
    }
}

#[test]
fn test_map_with_ga_invalid_qubits_avoidance() {
    let topology = line_topology(&[0, 1, 2, 3, 4, 5]);
    let circuit = test_circuit(3);

    let mut invalid_qubits = HashSet::new();
    invalid_qubits.insert(2);

    let config = fast_ga_config(42);

    let result = map_with_ga(&circuit, &topology, &config, None, Some(invalid_qubits));
    assert!(
        result.is_ok(),
        "Failed to find mapping in partitioned topology"
    );

    let mapped_circuit = result.unwrap();

    for op in mapped_circuit.operations() {
        for q in &op.qubits {
            let id = q.id();
            assert!(
                id == 3 || id == 4 || id == 5,
                "Algorithm mapped to an invalid or disconnected qubit: {}",
                id
            );
        }
    }
}

#[test]
fn test_map_with_ga_invalid_qubits_causes_too_small() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let circuit = test_circuit(3);

    let mut invalid_qubits = HashSet::new();
    invalid_qubits.insert(1);
    invalid_qubits.insert(2);

    let config = fast_ga_config(42);
    let result = map_with_ga(&circuit, &topology, &config, None, Some(invalid_qubits));

    assert!(
        matches!(result, Err(CompileError::TopologyTooSmall { .. })),
        "Expected TopologyTooSmall error due to fragmentation"
    );
}

#[test]
fn test_map_with_ga_fidelity_map_integration() {
    let topology = line_topology(&[0, 1, 2, 3]);
    let circuit = test_circuit(2);

    let mut fidelity_map = HashMap::new();
    fidelity_map.insert((Qubit::new(0), Qubit::new(1)), 0.5);
    fidelity_map.insert((Qubit::new(1), Qubit::new(2)), 0.99);
    fidelity_map.insert((Qubit::new(2), Qubit::new(3)), 0.99);

    let config = fast_ga_config(1024);

    let result = map_with_ga(&circuit, &topology, &config, Some(&fidelity_map), None);
    assert!(result.is_ok(), "Mapping failed with fidelity map provided");
}

#[test]
fn test_map_with_ga_determinism() {
    let topology = line_topology(&[0, 1, 2, 3, 4]);
    let circuit = test_circuit(4);

    let seed = 999;
    let config = fast_ga_config(seed);

    let result1 = map_with_ga(&circuit, &topology, &config, None, None).unwrap();
    let result2 = map_with_ga(&circuit, &topology, &config, None, None).unwrap();

    let fp1 = fingerprint(&result1);
    let fp2 = fingerprint(&result2);

    assert_eq!(
        fp1, fp2,
        "GA mapping should be deterministic given the same seed. Run 1: {:?}, Run 2: {:?}",
        fp1, fp2
    );
}

#[test]
fn test_map_with_ga_ghz_circuit_on_star_topology() {
    let topology = Topology::new(
        vec![
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
        ],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(0), Qubit::new(2), "CX".to_string()),
            (Qubit::new(0), Qubit::new(3), "CX".to_string()),
            (Qubit::new(0), Qubit::new(4), "CX".to_string()),
        ],
    )
    .unwrap();

    let mut circuit = Circuit::new(5);
    circuit.h(Qubit::new(0)).unwrap();
    for i in 0..4 {
        circuit
            .cx(Qubit::new(i as u32), Qubit::new((i + 1) as u32))
            .unwrap();
    }

    let config = fast_ga_config(100);

    let result = map_with_ga(&circuit, &topology, &config, None, None);
    assert!(
        result.is_ok(),
        "GA failed to map GHZ circuit on star topology"
    );

    let mapped = result.unwrap();

    assert_mapped_2q_edges(&mapped, &topology);
    assert_eq!(mapped.operations().len(), 6);
}

#[test]
fn test_map_with_ga_all_to_all_heavy_routing() {
    let topology = line_topology(&[0, 1, 2, 3, 4]);

    let mut circuit = Circuit::new(5);
    for i in 0..5 {
        for j in (i + 1)..5 {
            circuit
                .cx(Qubit::new(i as u32), Qubit::new(j as u32))
                .unwrap();
        }
    }

    let config = GaConfig {
        population: 10,
        update_iters: 5,
        seed: 2024,
        ..fast_ga_config(2024)
    };

    let result = map_with_ga(&circuit, &topology, &config, None, None);
    assert!(result.is_ok(), "GA failed to map all-to-all circuit");

    let mapped = result.unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}

#[test]
fn test_map_with_ga_non_contiguous_qubit_ids_supported() {
    let topology = line_topology(&[100, 200, 300, 400]);
    let mut circuit =
        Circuit::from_qubits(vec![Qubit::new(10), Qubit::new(30), Qubit::new(70)]).unwrap();
    circuit.cx(Qubit::new(10), Qubit::new(30)).unwrap();
    circuit.cx(Qubit::new(30), Qubit::new(70)).unwrap();
    let config = GaConfig {
        population: 10,
        update_iters: 5,
        seed: 2026,
        ..fast_ga_config(2026)
    };

    let result = map_with_ga(&circuit, &topology, &config, None, None);
    assert!(result.is_ok(), "GA failed to map non-contiguous qubit IDs");
    let mapped = result.unwrap();
    assert_mapped_2q_edges(&mapped, &topology);
}
