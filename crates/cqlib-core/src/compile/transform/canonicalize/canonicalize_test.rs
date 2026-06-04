// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::{CanonicalizeConfig, Canonicalizer, canonicalize_circuit};
use crate::circuit::gate::FrozenCircuit;
use crate::circuit::{
    Circuit, CircuitGate, CircuitParam, ConditionView, ControlFlow, Directive, Instruction, MCGate,
    Operation, Parameter, ParameterValue, Qubit, StandardGate, UnitaryGate, circuit_to_matrix,
};
use crate::compile::CompilerError;
use crate::compile::transform::Transformer;
use indexmap::IndexSet;
use ndarray::array;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};

#[test]
fn parameter_table_is_rebuilt_and_unused_params_are_removed() {
    let mut circuit = Circuit::new(1);
    circuit.add_parameter(Parameter::symbol("unused"));
    let theta = Parameter::symbol("theta");
    circuit.rz(Qubit::new(0), theta.clone() - theta).unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.parameters().is_empty());
    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn top_level_gphase_is_folded_into_circuit_global_phase() {
    let mut circuit = Circuit::new(1);
    circuit.set_global_phase(Parameter::from(0.125));
    circuit
        .append(
            Instruction::Standard(StandardGate::GPhase),
            std::iter::empty::<Qubit>(),
            [ParameterValue::Fixed(0.25)],
            Some("ignored"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!((result.circuit.global_phase().evaluate(&None).unwrap() - 0.375).abs() < 1e-12);
}

#[test]
fn body_gphase_is_merged_into_first_body_operation() {
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::GPhase),
            qubits: smallvec![],
            params: smallvec![CircuitParam::Fixed(0.5)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::GPhase),
            qubits: smallvec![],
            params: smallvec![CircuitParam::Fixed(0.25)],
            label: None,
        },
    ];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if_else");
    };

    assert_eq!(gate.true_body().len(), 2);
    assert!(matches!(
        gate.true_body()[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    match gate.true_body()[0].params.as_slice() {
        [CircuitParam::Fixed(value)] => assert_eq!(*value, 0.75),
        params => panic!("expected one fixed GPhase parameter, got {params:?}"),
    }
    assert!(matches!(
        gate.true_body()[1].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn body_zero_gphase_is_removed() {
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::GPhase),
            qubits: smallvec![],
            params: smallvec![CircuitParam::Fixed(0.0)],
            label: Some("zero-phase".into()),
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if_else");
    };

    assert_eq!(gate.true_body().len(), 1);
    assert!(matches!(
        gate.true_body()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn false_body_and_while_body_keep_independent_local_phase() {
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::GPhase),
        qubits: smallvec![],
        params: smallvec![CircuitParam::Fixed(0.25)],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::GPhase),
        qubits: smallvec![],
        params: smallvec![CircuitParam::Fixed(0.5)],
        label: None,
    }];
    let while_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::GPhase),
            qubits: smallvec![],
            params: smallvec![CircuitParam::Fixed(0.75)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            true_body,
            Some(false_body),
        )
        .unwrap();
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), while_body)
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();
    let Instruction::ControlFlowGate(ControlFlow::IfElse(if_gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if_else");
    };
    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_gate)) =
        &result.circuit.operations()[1].instruction
    else {
        panic!("expected while_loop");
    };

    match if_gate.true_body()[0].params.as_slice() {
        [CircuitParam::Fixed(value)] => assert_eq!(*value, 0.25),
        params => panic!("expected one fixed true-body phase parameter, got {params:?}"),
    }

    let false_body = if_gate.false_body().expect("expected false body");
    match false_body[0].params.as_slice() {
        [CircuitParam::Fixed(value)] => assert_eq!(*value, 0.5),
        params => panic!("expected one fixed false-body phase parameter, got {params:?}"),
    }

    match while_gate.body()[0].params.as_slice() {
        [CircuitParam::Fixed(value)] => assert_eq!(*value, 0.75),
        params => panic!("expected one fixed while-body phase parameter, got {params:?}"),
    }
    assert!(matches!(
        while_gate.body()[1].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn nested_control_flow_is_recursively_canonicalized() {
    let nested_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::I),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
    ];
    let outer_body = vec![Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(
            crate::circuit::WhileLoopGate::new(ConditionView::new(Qubit::new(1), 1), nested_body),
        )),
        qubits: smallvec![Qubit::new(2), Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    let mut circuit = Circuit::new(3);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), outer_body, None)
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();
    let Instruction::ControlFlowGate(ControlFlow::IfElse(if_gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if_else");
    };
    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_gate)) =
        &if_gate.true_body()[0].instruction
    else {
        panic!("expected nested while_loop");
    };

    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)]
    );
    assert_eq!(
        if_gate.true_body()[0].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2)]
    );
    assert_eq!(while_gate.body().len(), 1);
    assert!(matches!(
        while_gate.body()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn mc_gate_collapses_to_standard_gate() {
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            Some("cx-label"),
        )
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert_eq!(
        result.circuit.operations()[0].label.as_deref(),
        Some("cx-label")
    );
}

#[test]
fn noops_are_removed_even_when_labeled() {
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::Standard(StandardGate::I),
            [Qubit::new(0)],
            std::iter::empty(),
            Some("drop-i"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Delay,
            [Qubit::new(0)],
            [ParameterValue::Fixed(0.0)],
            Some("drop-delay"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::RXX),
            [Qubit::new(0), Qubit::new(1)],
            [ParameterValue::Fixed(0.0)],
            Some("drop-rxx"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::U),
            [Qubit::new(0)],
            [
                ParameterValue::Fixed(0.0),
                ParameterValue::Fixed(0.0),
                ParameterValue::Fixed(0.0),
            ],
            Some("drop-u"),
        )
        .unwrap();
    circuit.x(Qubit::new(1)).unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::X)
    ));
}

#[test]
fn config_can_preserve_noops_when_disabled() {
    let mut circuit = Circuit::new(1);
    circuit.i(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().drop_noops(false));
    let result = canonicalizer.run(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 2);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::I)
    ));
}

#[test]
fn canonicalizer_implements_transformer_trait() {
    let mut circuit = Circuit::new(1);
    circuit.i(Qubit::new(0)).unwrap();

    let transformer = &Canonicalizer::production();
    let result = transformer.transform(&circuit).unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
}

#[test]
fn config_can_preserve_top_level_gphase_when_disabled() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::GPhase),
            std::iter::empty::<Qubit>(),
            [ParameterValue::Fixed(0.25)],
            None,
        )
        .unwrap();

    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().fold_gphase(false));
    let result = canonicalizer.run(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    match result.circuit.operations()[0].params.as_slice() {
        [CircuitParam::Fixed(value)] => assert_eq!(*value, 0.25),
        params => panic!("expected one fixed top-level GPhase parameter, got {params:?}"),
    }
}

#[test]
fn config_can_skip_control_flow_recursion() {
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::I),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: Some("preserve-body".into()),
    }];
    let mut circuit = Circuit::new(2);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();

    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().recurse_control_flow(false));
    let result = canonicalizer.run(&circuit).unwrap();
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &result.circuit.operations()[0].instruction
    else {
        panic!("expected if_else");
    };

    assert_eq!(gate.true_body().len(), 1);
    assert!(matches!(
        gate.true_body()[0].instruction,
        Instruction::Standard(StandardGate::I)
    ));
    assert_eq!(gate.true_body()[0].label.as_deref(), Some("preserve-body"));
}

#[test]
fn config_can_preserve_barrier_shape_when_disabled() {
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(2), Qubit::new(0), Qubit::new(2)],
            std::iter::empty(),
            Some("keep-barrier"),
        )
        .unwrap();

    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().canonicalize_barriers(false));
    let result = canonicalizer.run(&circuit).unwrap();
    let op = &result.circuit.operations()[0];

    assert_eq!(
        op.qubits.as_slice(),
        &[Qubit::new(2), Qubit::new(0), Qubit::new(2)]
    );
    assert_eq!(op.label.as_deref(), Some("keep-barrier"));
}

#[test]
fn barrier_scopes_are_canonicalized_and_merged() {
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(2), Qubit::new(1), Qubit::new(2)],
            std::iter::empty(),
            Some("drop-label"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(1), Qubit::new(2), Qubit::new(3)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    let op = &result.circuit.operations()[0];
    assert!(matches!(
        op.instruction,
        Instruction::Directive(Directive::Barrier)
    ));
    assert_eq!(
        op.qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2), Qubit::new(3)]
    );
    assert!(op.label.is_none());
}

#[test]
fn barrier_partial_overlap_and_non_adjacent_barriers_are_not_merged() {
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(1), Qubit::new(2)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 4);
    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
    assert_eq!(
        result.circuit.operations()[1].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2)]
    );
    assert!(matches!(
        result.circuit.operations()[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn empty_barrier_is_removed() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            std::iter::empty::<Qubit>(),
            std::iter::empty(),
            Some("drop-empty"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(result.circuit.operations().len(), 1);
    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn control_flow_qubits_are_recomputed_in_global_order() {
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::I),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    let mut circuit = Circuit::new(3);
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert_eq!(
        result.circuit.operations()[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(2)]
    );
}

#[test]
fn invalid_parameter_reference_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::RX),
            qubits: smallvec![Qubit::new(0)],
            params: smallvec![CircuitParam::Index(999)],
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("missing parameter index"));
}

#[test]
fn invalid_global_phase_reference_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        Vec::new(),
        CircuitParam::Index(3),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("global phase references"));
}

#[test]
fn unknown_qubit_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("unknown qubit"));
}

#[test]
fn duplicate_non_barrier_qubit_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(0), Qubit::new(0)],
            params: smallvec![],
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("duplicate qubit"));
}

#[test]
fn invalid_arity_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![Qubit::new(0)],
            params: smallvec![],
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("qubit count mismatch"));
}

#[test]
fn parameter_count_mismatch_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::RX),
            qubits: smallvec![Qubit::new(0)],
            params: smallvec![],
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("parameter count mismatch"));
}

#[test]
fn non_finite_fixed_parameter_is_rejected() {
    let circuit = Circuit::from_parts(
        IndexSet::from_iter([Qubit::new(0)]),
        IndexSet::new(),
        IndexSet::new(),
        vec![Operation {
            instruction: Instruction::Standard(StandardGate::RX),
            qubits: smallvec![Qubit::new(0)],
            params: SmallVec::from_buf([CircuitParam::Fixed(f64::NAN)]),
            label: None,
        }],
        CircuitParam::Fixed(0.0),
    );

    let err = canonicalize_circuit(&circuit).unwrap_err();
    assert!(matches!(err, CompilerError::InvalidInput(_)));
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn measurement_reset_circuit_gate_and_unitary_gate_are_preserved() {
    let mut inner = Circuit::new(1);
    inner.x(Qubit::new(0)).unwrap();
    let circuit_gate = CircuitGate::new("inner_x", FrozenCircuit::new(inner)).unwrap();
    let unitary = UnitaryGate::new("custom_x", 1, 0)
        .with_matrix(array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .unwrap();
    let mut circuit = Circuit::new(1);
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.reset(Qubit::new(0)).unwrap();
    circuit
        .append(
            Instruction::CircuitGate(Box::new(circuit_gate)),
            [Qubit::new(0)],
            std::iter::empty(),
            Some("composite"),
        )
        .unwrap();
    circuit.unitary(unitary, vec![Qubit::new(0)]).unwrap();

    let result = canonicalize_circuit(&circuit).unwrap();

    assert!(matches!(
        result.circuit.operations()[0].instruction,
        Instruction::Directive(Directive::Measure)
    ));
    assert!(matches!(
        result.circuit.operations()[1].instruction,
        Instruction::Directive(Directive::Reset)
    ));
    assert!(matches!(
        result.circuit.operations()[2].instruction,
        Instruction::CircuitGate(_)
    ));
    assert!(matches!(
        result.circuit.operations()[3].instruction,
        Instruction::UnitaryGate(_)
    ));
    assert_eq!(
        result.circuit.operations()[2].label.as_deref(),
        Some("composite")
    );
}

#[test]
fn matrix_is_strictly_preserved_without_control_flow() {
    let mut circuit = Circuit::new(2);
    circuit.set_global_phase(Parameter::from(0.5));
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.rx(Qubit::new(1), Parameter::from(0.0)).unwrap();
    circuit.barrier(vec![Qubit::new(1), Qubit::new(0)]).unwrap();

    let before = circuit_to_matrix(&circuit, None).unwrap();
    let result = canonicalize_circuit(&circuit).unwrap();
    let after = circuit_to_matrix(&result.circuit, None).unwrap();

    let before = before.as_slice().expect("matrix storage is contiguous");
    let after = after.as_slice().expect("matrix storage is contiguous");
    assert_eq!(before.len(), after.len());
    for (index, (before, after)) in before.iter().zip(after).enumerate() {
        let diff = (*before - *after).norm();
        assert!(
            diff <= 1e-10,
            "matrix entry {index} differs: before={before}, after={after}, diff={diff}"
        );
    }

    assert!(result.circuit.operations().iter().all(|operation| {
        !matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::GPhase)
        )
    }));
}

#[test]
fn canonicalization_is_idempotent() {
    let mut circuit = Circuit::new(2);
    circuit.i(Qubit::new(0)).unwrap();
    circuit.barrier(vec![Qubit::new(1), Qubit::new(0)]).unwrap();
    circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
    circuit.h(Qubit::new(1)).unwrap();

    let first = canonicalize_circuit(&circuit).unwrap();
    let second = canonicalize_circuit(&first.circuit).unwrap();

    assert!(first.changed);
    assert!(!second.changed);
}
