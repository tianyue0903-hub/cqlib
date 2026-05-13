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

//! Integration and unit tests for the canonicalize module.

use crate::circuit::gate::{CircuitGate, FrozenCircuit, UnitaryGate};
use crate::circuit::{
    Circuit, CircuitParam, ConditionView, ControlFlow, Directive, IfElseGate, Instruction, MCGate,
    Operation, Parameter, ParameterValue, Qubit, StandardGate, WhileLoopGate,
};
use crate::compiler::artifact::DiagnosticSeverity;
use crate::compiler::context::{CompilerContext, ContextChangeSet};
use crate::compiler::error::CompilerError;
use crate::compiler::transform::Transformer;
use crate::compiler::transform::canonicalize::canonicalizer::{
    FixpointResult, SingleRoundResult, run_to_fixpoint_with,
};
use crate::compiler::transform::canonicalize::equivalence::instructions_equivalent;
use crate::compiler::transform::canonicalize::parameter_phase::{
    canonicalize_parameter_phase, parameter_phase_changed,
};
use crate::compiler::transform::canonicalize::{
    CanonicalRuleId, CanonicalizeConfig, Canonicalizer,
};
use ndarray::array;
use num_complex::Complex64;
use smallvec::smallvec;

fn param_to_string(circuit: &Circuit, param: CircuitParam) -> String {
    match param {
        CircuitParam::Fixed(value) => Parameter::from(value).to_string(),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(index as usize)
            .unwrap()
            .to_string(),
    }
}

#[test]
fn production_config_enables_all_builtin_behaviors() {
    let config = CanonicalizeConfig::production();

    assert_eq!(config.round_limit(), 8);
    assert!(config.recurses_control_flow());
    assert!(config.normalizes_parameters());
    assert!(config.canonicalizes_instruction_form());
    assert!(config.merges_adjacent_barriers());
    assert!(config.drops_trivial_noops());
}

#[test]
fn config_builder_only_changes_requested_fields() {
    let config = CanonicalizeConfig::new()
        .with_round_limit(3)
        .recurse_control_flow(false)
        .normalize_parameters(false)
        .canonicalize_instruction_form(false)
        .merge_adjacent_barriers(false)
        .drop_trivial_noops(false);

    assert_eq!(config.round_limit(), 3);
    assert!(!config.recurses_control_flow());
    assert!(!config.normalizes_parameters());
    assert!(!config.canonicalizes_instruction_form());
    assert!(!config.merges_adjacent_barriers());
    assert!(!config.drops_trivial_noops());
}

#[test]
fn canonicalize_parameter_phase_simplifies_symbolic_expressions() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit
        .rz(
            Qubit::new(0),
            theta.clone().sin().pow(Parameter::from(2)) + theta.cos().pow(Parameter::from(2)),
        )
        .unwrap();

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    assert_eq!(
        param_to_string(&canonical, canonical.operations()[0].params[0].clone()),
        "1"
    );
    assert!(parameter_phase_changed(&circuit, &canonical));
}

#[test]
fn canonicalize_parameter_phase_keeps_unbound_symbolic_expression_symbolic() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit
        .rz(Qubit::new(0), theta + Parameter::from(1.0))
        .unwrap();

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    assert_eq!(canonical.parameters().len(), 1);
    assert!(matches!(
        canonical.operations()[0].params[0],
        CircuitParam::Index(0)
    ));
    assert_eq!(
        canonical.parameters().get_index(0).unwrap().to_string(),
        "1 + theta"
    );
}

#[test]
fn canonicalize_parameter_phase_folds_symbol_free_expression_to_fixed_param() {
    let mut circuit = Circuit::new(1);
    circuit
        .rz(Qubit::new(0), Parameter::from(2.0) + Parameter::from(3.0))
        .unwrap();

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    assert!(canonical.parameters().is_empty());
    assert!(matches!(
        canonical.operations()[0].params[0],
        CircuitParam::Fixed(5.0)
    ));
}

#[test]
fn canonicalize_parameter_phase_folds_evaluable_global_phase() {
    let mut circuit = Circuit::new(1);
    let phi = Parameter::symbol("phi");
    circuit.set_global_phase(
        phi.clone().sin().pow(Parameter::from(2)) + phi.cos().pow(Parameter::from(2)),
    );

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    assert_eq!(canonical.global_phase().to_string(), "1");
    assert!(parameter_phase_changed(&circuit, &canonical));
}

#[test]
fn parameter_phase_changed_detects_no_difference() {
    let circuit = Circuit::new(1);

    assert!(!parameter_phase_changed(&circuit, &circuit));
}

#[test]
fn canonicalize_parameter_phase_remaps_body_only_symbolic_parameter_in_if_else() {
    let mut circuit = Circuit::new(2);
    circuit.add_parameter(Parameter::from(1.0));
    let theta = Parameter::symbol("theta");
    let (theta_index, _) = circuit.add_parameter(theta.clone() + Parameter::from(0.0));

    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![CircuitParam::Index(theta_index as u32)],
        label: None,
    }];

    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                ConditionView::new(Qubit::new(0), 1),
                true_body,
                None,
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &canonical.operations()[0].instruction
    else {
        panic!("expected if-else gate");
    };
    let CircuitParam::Index(index) = gate.true_body()[0].params[0] else {
        panic!("expected symbolic parameter index");
    };
    assert_eq!(
        canonical
            .parameters()
            .get_index(index as usize)
            .unwrap()
            .to_string(),
        "theta"
    );
}

#[test]
fn canonicalize_parameter_phase_remaps_body_only_symbolic_parameter_in_while_loop() {
    let mut circuit = Circuit::new(2);
    circuit.add_parameter(Parameter::from(1.0));
    let theta = Parameter::symbol("theta");
    let (theta_index, _) = circuit.add_parameter(theta.clone() + Parameter::from(0.0));

    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![CircuitParam::Index(theta_index as u32)],
        label: None,
    }];

    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body,
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();

    let canonical = canonicalize_parameter_phase(&circuit).unwrap();

    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) =
        &canonical.operations()[0].instruction
    else {
        panic!("expected while-loop gate");
    };
    let CircuitParam::Index(index) = gate.body()[0].params[0] else {
        panic!("expected symbolic parameter index");
    };
    assert_eq!(
        canonical
            .parameters()
            .get_index(index as usize)
            .unwrap()
            .to_string(),
        "theta"
    );
}

#[test]
fn canonicalizer_uses_stable_descriptor_contract() {
    let canonicalizer = Canonicalizer::production();
    let descriptor = canonicalizer.descriptor();

    assert_eq!(descriptor.name, "canonicalize.standard");
    assert!(!descriptor.requires_device);
    assert!(!descriptor.requires_layout);
    assert!(descriptor.supports_control_flow);
    assert!(descriptor.supports_symbolic_parameters);
    assert!(descriptor.modifies_circuit);
}

#[test]
fn canonicalizer_noop_when_parameter_normalization_is_disabled() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert!(outcome.notes.is_empty());
}

#[test]
fn canonicalizer_normalizes_parameter_and_marks_context_changed() {
    let canonicalizer = Canonicalizer::production();
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit
        .rz(
            Qubit::new(0),
            theta.clone().sin().pow(Parameter::from(2)) + theta.cos().pow(Parameter::from(2)),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert_eq!(ctx.revision(), 1);
    assert_eq!(
        param_to_string(
            ctx.circuit(),
            ctx.circuit().operations()[0].params[0].clone()
        ),
        "1"
    );
}

#[test]
fn canonicalizer_collapses_top_level_mc_gate_into_standard_form() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert_eq!(ctx.revision(), 1);
    assert!(matches!(
        ctx.circuit().operations()[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
}

#[test]
fn canonicalizer_collapses_mc_gate_without_reordering_qubits_or_params() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RZ))),
            [Qubit::new(1), Qubit::new(0)],
            [ParameterValue::Fixed(0.5)],
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operation = &ctx.circuit().operations()[0];
    assert!(matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::CRZ)
    ));
    assert_eq!(operation.qubits.as_slice(), &[Qubit::new(1), Qubit::new(0)]);
    assert_eq!(operation.params.len(), 1);
    assert!(matches!(operation.params[0], CircuitParam::Fixed(0.5)));
}

#[test]
fn canonicalizer_merges_equal_adjacent_barriers() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(3);
    circuit.barrier(vec![Qubit::new(1), Qubit::new(2)]).unwrap();
    circuit.barrier(vec![Qubit::new(2), Qubit::new(1)]).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 1);
    assert_eq!(
        ctx.circuit().operations()[0].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2)]
    );
}

#[test]
fn canonicalizer_merges_barrier_labels_without_dropping_metadata() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty::<ParameterValue>(),
            Some("lhs"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(1), Qubit::new(0)],
            std::iter::empty::<ParameterValue>(),
            Some("rhs"),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 1);
    assert_eq!(
        ctx.circuit().operations()[0].label.as_deref(),
        Some("lhs | rhs")
    );
}

#[test]
fn canonicalizer_absorbs_adjacent_subset_barrier() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(4);
    circuit
        .barrier(vec![Qubit::new(1), Qubit::new(2), Qubit::new(3)])
        .unwrap();
    circuit.barrier(vec![Qubit::new(2), Qubit::new(1)]).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 1);
    assert_eq!(
        ctx.circuit().operations()[0].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2), Qubit::new(3)]
    );
}

#[test]
fn canonicalizer_absorbs_barrier_labels_into_superset_barrier() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(1), Qubit::new(2), Qubit::new(3)],
            std::iter::empty::<ParameterValue>(),
            Some("outer"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            [Qubit::new(2), Qubit::new(1)],
            std::iter::empty::<ParameterValue>(),
            Some("inner"),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 1);
    assert_eq!(
        ctx.circuit().operations()[0].label.as_deref(),
        Some("outer | inner")
    );
}

#[test]
fn canonicalizer_does_not_merge_non_adjacent_barriers() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(4);
    circuit.barrier(vec![Qubit::new(1), Qubit::new(2)]).unwrap();
    circuit.h(Qubit::new(3)).unwrap();
    circuit.barrier(vec![Qubit::new(2), Qubit::new(1)]).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 3);
}

#[test]
fn canonicalizer_does_not_merge_partially_overlapping_adjacent_barriers() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(3);
    circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
    circuit.barrier(vec![Qubit::new(1), Qubit::new(2)]).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 2);
    assert_eq!(
        ctx.circuit().operations()[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
    assert_eq!(
        ctx.circuit().operations()[1].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2)]
    );
}

#[test]
fn canonicalizer_drops_trivial_noops() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    circuit.i(Qubit::new(0)).unwrap();
    circuit
        .delay(Qubit::new(0), ParameterValue::Fixed(0.0))
        .unwrap();
    circuit.rz(Qubit::new(0), 0.0).unwrap();
    circuit.rzz(Qubit::new(0), Qubit::new(1), 0.0).unwrap();
    circuit.h(Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 1);
    assert!(matches!(
        ctx.circuit().operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn canonicalizer_preserves_labeled_trivial_noops() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::I),
            [Qubit::new(0)],
            std::iter::empty::<ParameterValue>(),
            Some("keep-i"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::RZ),
            [Qubit::new(0)],
            [ParameterValue::Fixed(0.0)],
            Some("keep-rz"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Delay,
            [Qubit::new(0)],
            [ParameterValue::Fixed(0.0)],
            Some("keep-delay"),
        )
        .unwrap();
    circuit
        .append(
            Instruction::Directive(Directive::Barrier),
            std::iter::empty::<Qubit>(),
            std::iter::empty::<ParameterValue>(),
            Some("keep-empty-barrier"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 5);
    assert_eq!(operations[0].label.as_deref(), Some("keep-i"));
    assert_eq!(operations[1].label.as_deref(), Some("keep-rz"));
    assert_eq!(operations[2].label.as_deref(), Some("keep-delay"));
    assert_eq!(operations[3].label.as_deref(), Some("keep-empty-barrier"));
}

#[test]
fn canonicalizer_recurses_into_control_flow_bodies() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(3);
    let true_body = vec![
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(2), Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::RZZ),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![CircuitParam::Fixed(0.0)],
            label: None,
        },
    ];
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                ConditionView::new(Qubit::new(0), 1),
                true_body,
                None,
            ))),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &ctx.circuit().operations()[0].instruction
    else {
        panic!("expected if-else gate");
    };
    assert_eq!(gate.true_body().len(), 1);
    assert!(matches!(
        gate.true_body()[0].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
    assert_eq!(
        gate.true_body()[0].qubits.as_slice(),
        &[Qubit::new(1), Qubit::new(2)]
    );
}

#[test]
fn canonicalizer_preserves_parent_parameter_indices_inside_control_flow_bodies() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    let theta = Parameter::symbol("theta");
    circuit.rz(Qubit::new(1), theta.clone()).unwrap();

    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![circuit.operations()[0].params[0].clone()],
        label: None,
    }];
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body,
                None,
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &ctx.circuit().operations()[1].instruction
    else {
        panic!("expected if-else gate");
    };
    let CircuitParam::Index(index) = gate.true_body()[0].params[0] else {
        panic!("expected symbolic parameter index");
    };
    assert_eq!(
        ctx.circuit()
            .parameters()
            .get_index(index as usize)
            .unwrap()
            .to_string(),
        theta.to_string()
    );
}

#[test]
fn canonicalizer_returns_error_for_invalid_body_parameter_index() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![CircuitParam::Index(99)],
        label: None,
    }];
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body,
                None,
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let error = canonicalizer.run(&mut ctx).unwrap_err();

    assert!(matches!(error, CompilerError::InvalidContextState(_)));
    assert!(
        error
            .to_string()
            .contains("invalid control-flow body parameter index")
    );
}

#[test]
fn canonicalizer_is_idempotent_after_barrier_and_noop_cleanup() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    circuit.barrier(vec![Qubit::new(1), Qubit::new(0)]).unwrap();
    circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
    circuit.rx(Qubit::new(0), 0.0).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let first = canonicalizer.run(&mut ctx).unwrap();
    let second = canonicalizer.run(&mut ctx).unwrap();

    assert!(first.changed);
    assert!(!second.changed);
}

#[test]
fn canonicalizer_is_idempotent_for_stable_control_flow_body() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(2);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(1)],
        params: smallvec![],
        label: None,
    }];
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body,
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let first = canonicalizer.run(&mut ctx).unwrap();
    let second = canonicalizer.run(&mut ctx).unwrap();

    assert!(!first.changed);
    assert!(!second.changed);
    assert_eq!(ctx.revision(), 0);
}

#[test]
fn canonicalizer_rejects_zero_round_limit() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().with_round_limit(0));
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let error = canonicalizer.run(&mut ctx).unwrap_err();

    assert!(matches!(error, CompilerError::InvalidContextState(_)));
    assert!(
        error
            .to_string()
            .contains("canonicalize round_limit must be greater than zero")
    );
}

#[test]
fn fixpoint_helper_stabilizes_after_second_round() {
    let config = CanonicalizeConfig::new().with_round_limit(3);
    let initial = Circuit::new(1);
    let mut first_round = true;

    let result = run_to_fixpoint_with(&initial, &config, |circuit, _| {
        if first_round {
            first_round = false;
            let mut changed = circuit.clone();
            changed.h(Qubit::new(0)).unwrap();
            Ok(SingleRoundResult {
                circuit: changed,
                parameter_phase_changed: false,
                structural_changed: true,
            })
        } else {
            Ok(SingleRoundResult {
                circuit: circuit.clone(),
                parameter_phase_changed: false,
                structural_changed: false,
            })
        }
    })
    .unwrap();

    assert!(result.stabilized);
    assert_eq!(result.rounds_executed, 2);
    assert!(result.any_structural_changed);
}

#[test]
fn fixpoint_helper_reports_round_limit_when_never_stable() {
    let config = CanonicalizeConfig::new()
        .with_round_limit(2)
        .normalize_parameters(false);
    let initial = Circuit::new(1);

    let result = run_to_fixpoint_with(&initial, &config, |circuit, _| {
        let mut changed = circuit.clone();
        changed.h(Qubit::new(0)).unwrap();
        Ok(SingleRoundResult {
            circuit: changed,
            parameter_phase_changed: false,
            structural_changed: true,
        })
    })
    .unwrap();

    assert!(!result.stabilized);
    assert_eq!(result.rounds_executed, 2);
    assert!(result.any_structural_changed);
}

#[test]
fn canonicalizer_emits_warning_when_round_limit_is_reached() {
    let config = CanonicalizeConfig::new()
        .with_round_limit(2)
        .normalize_parameters(false);
    let initial = Circuit::new(1);

    let loop_result = run_to_fixpoint_with(&initial, &config, |circuit, _| {
        let mut changed = circuit.clone();
        changed.h(Qubit::new(0)).unwrap();
        Ok(SingleRoundResult {
            circuit: changed,
            parameter_phase_changed: false,
            structural_changed: true,
        })
    })
    .unwrap();

    let outcome = canonicalizer_outcome_from_fixpoint_result(loop_result);

    assert!(outcome.changed);
    assert!(
        outcome
            .notes
            .iter()
            .any(|note| note.contains("reached round limit after 2 rounds"))
    );
    assert!(outcome.diagnostics.iter().any(|diagnostic| {
        diagnostic.severity == DiagnosticSeverity::Warning
            && diagnostic.code == "compiler.canonicalize.round_limit_reached"
    }));
}

#[test]
fn canonical_rule_ids_are_stable_and_distinct() {
    assert_ne!(
        CanonicalRuleId::NormalizeParameters,
        CanonicalRuleId::CanonicalizeInstructionForm
    );
    assert_ne!(
        CanonicalRuleId::MergeAdjacentBarriers,
        CanonicalRuleId::DropTrivialNoOps
    );
}

#[test]
fn instruction_equivalence_distinguishes_same_named_circuit_gates() {
    let mut lhs_inner = Circuit::new(1);
    lhs_inner.h(Qubit::new(0)).unwrap();
    let mut rhs_inner = Circuit::new(1);
    rhs_inner.x(Qubit::new(0)).unwrap();

    let lhs = Instruction::CircuitGate(Box::new(
        CircuitGate::new("same-name", FrozenCircuit::new(lhs_inner)).unwrap(),
    ));
    let rhs = Instruction::CircuitGate(Box::new(
        CircuitGate::new("same-name", FrozenCircuit::new(rhs_inner)).unwrap(),
    ));

    assert!(!instructions_equivalent(
        &lhs,
        &rhs,
        &Circuit::new(1),
        &Circuit::new(1)
    ));
}

#[test]
fn canonicalizer_is_idempotent_for_stable_circuit_gate() {
    let canonicalizer = Canonicalizer::production();
    let mut inner = Circuit::new(1);
    inner.h(Qubit::new(0)).unwrap();
    let gate = CircuitGate::new("stable", FrozenCircuit::new(inner)).unwrap();

    let mut circuit = Circuit::new(1);
    circuit
        .circuit_gate(
            gate,
            vec![Qubit::new(0)],
            std::iter::empty::<ParameterValue>(),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert!(outcome.diagnostics.is_empty());
    assert_eq!(ctx.revision(), 0);
}

#[test]
fn instruction_equivalence_distinguishes_same_labeled_unitary_gates() {
    let lhs = Instruction::UnitaryGate(Box::new(
        UnitaryGate::new("oracle", 1, 0)
            .with_matrix(array![
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ])
            .unwrap(),
    ));
    let rhs = Instruction::UnitaryGate(Box::new(
        UnitaryGate::new("oracle", 1, 0)
            .with_matrix(array![
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .unwrap(),
    ));

    assert!(!instructions_equivalent(
        &lhs,
        &rhs,
        &Circuit::new(1),
        &Circuit::new(1)
    ));
}

#[test]
fn canonicalizer_recomputes_control_flow_qubits_after_body_cleanup() {
    let canonicalizer = Canonicalizer::new(CanonicalizeConfig::new().normalize_parameters(false));
    let mut circuit = Circuit::new(3);
    let true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::I),
            qubits: smallvec![Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), true_body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operation = &ctx.circuit().operations()[0];
    assert_eq!(operation.qubits.as_slice(), &[Qubit::new(1), Qubit::new(0)]);
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) = &operation.instruction else {
        panic!("expected if-else gate");
    };
    assert_eq!(gate.true_body().len(), 1);
    assert_eq!(gate.true_body()[0].qubits.as_slice(), &[Qubit::new(1)]);
}

#[test]
fn canonicalizer_keeps_label_on_non_gphase_replacement() {
    let canonicalizer = Canonicalizer::production();
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::RX),
            [Qubit::new(0)],
            [ParameterValue::Fixed(std::f64::consts::PI)],
            Some("rx-pi"),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let ops = ctx.circuit().operations();
    assert_eq!(ops.len(), 2);
    assert!(matches!(
        ops[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    assert_eq!(ops[0].label.as_deref(), None);
    assert!(matches!(
        ops[1].instruction,
        Instruction::Standard(StandardGate::X)
    ));
    assert_eq!(ops[1].label.as_deref(), Some("rx-pi"));
}

#[test]
fn canonicalizer_fixpoint_drops_noop_after_parameter_normalization_to_zero() {
    // Parameter normalization turns RZ(theta - theta) into RZ(0), which the
    // structural pass then drops as a trivial no-op. This requires at least two
    // fixpoint rounds to converge.
    let canonicalizer = Canonicalizer::production();
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit
        .rz(Qubit::new(0), theta.clone() - theta.clone())
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let first = canonicalizer.run(&mut ctx).unwrap();

    assert!(first.changed);
    assert_eq!(ctx.circuit().operations().len(), 1);
    assert!(matches!(
        ctx.circuit().operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));

    let second = canonicalizer.run(&mut ctx).unwrap();
    assert!(!second.changed);
}

#[test]
fn canonicalizer_preserves_global_phase_for_two_pi_rotation() {
    let canonicalizer = Canonicalizer::production();
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), std::f64::consts::TAU).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let ops = ctx.circuit().operations();
    assert_eq!(ops.len(), 1);
    assert!(matches!(
        ops[0].instruction,
        Instruction::Standard(StandardGate::GPhase)
    ));
    assert!(ops[0].qubits.is_empty());
    assert!(matches!(
        ops[0].params.as_slice(),
        [CircuitParam::Fixed(value)] if (*value - std::f64::consts::PI).abs() < 1e-12
    ));
}

#[test]
fn canonicalizer_folds_special_angles_to_named_gates() {
    let canonicalizer = Canonicalizer::production();
    let mut circuit = Circuit::new(1);
    circuit
        .phase(Qubit::new(0), std::f64::consts::FRAC_PI_2)
        .unwrap();
    circuit
        .rx(Qubit::new(0), std::f64::consts::FRAC_PI_2)
        .unwrap();
    circuit
        .ry(Qubit::new(0), -std::f64::consts::FRAC_PI_2)
        .unwrap();
    circuit
        .rxy(
            Qubit::new(0),
            std::f64::consts::PI,
            std::f64::consts::TAU + 0.125,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    canonicalizer.run(&mut ctx).unwrap();

    let gates: Vec<_> = ctx
        .circuit()
        .operations()
        .iter()
        .map(|op| match op.instruction {
            Instruction::Standard(gate) => gate,
            _ => panic!("expected standard gate"),
        })
        .collect();
    assert_eq!(
        gates,
        vec![
            StandardGate::S,
            StandardGate::X2P,
            StandardGate::Y2M,
            StandardGate::XY
        ]
    );
    assert!(matches!(
        ctx.circuit().operations()[3].params.as_slice(),
        [CircuitParam::Fixed(value)] if (*value - 0.125).abs() < 1e-12
    ));
}

#[test]
fn canonicalizer_does_not_recurse_into_if_else_when_disabled() {
    let canonicalizer = Canonicalizer::new(
        CanonicalizeConfig::new()
            .normalize_parameters(false)
            .recurse_control_flow(false),
    );
    let mut circuit = Circuit::new(3);
    // Top-level operation that should be canonicalized
    circuit.i(Qubit::new(2)).unwrap();

    let body = vec![
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(2), Qubit::new(1)],
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
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body.clone(),
                None,
            ))),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    // Top-level I gate was dropped
    assert_eq!(ctx.circuit().operations().len(), 1);
    let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
        &ctx.circuit().operations()[0].instruction
    else {
        panic!("expected if-else gate");
    };
    // Body must remain untouched because recurse_control_flow is false
    assert_eq!(gate.true_body().len(), 3);
    assert!(matches!(
        gate.true_body()[0].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
    assert!(matches!(
        gate.true_body()[1].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
    assert!(matches!(
        gate.true_body()[2].instruction,
        Instruction::Standard(StandardGate::I)
    ));
}

#[test]
fn canonicalizer_does_not_recurse_into_while_loop_when_disabled() {
    let canonicalizer = Canonicalizer::new(
        CanonicalizeConfig::new()
            .normalize_parameters(false)
            .recurse_control_flow(false),
    );
    let mut circuit = Circuit::new(2);
    // Top-level operation that should be canonicalized
    circuit.i(Qubit::new(1)).unwrap();

    let body = vec![
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit
        .append(
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                ConditionView::new(Qubit::new(0), 1),
                body.clone(),
            ))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = canonicalizer.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    // Top-level I gate was dropped
    assert_eq!(ctx.circuit().operations().len(), 1);
    let Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) =
        &ctx.circuit().operations()[0].instruction
    else {
        panic!("expected while-loop gate");
    };
    // Body must remain untouched because recurse_control_flow is false
    assert_eq!(gate.body().len(), 2);
    assert!(matches!(
        gate.body()[0].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
    assert!(matches!(
        gate.body()[1].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
}

fn canonicalizer_outcome_from_fixpoint_result(
    loop_result: FixpointResult,
) -> crate::compiler::transform::TransformOutcome {
    let mut outcome = crate::compiler::transform::TransformOutcome::changed().with_changes(
        ContextChangeSet::circuit_changed()
            .with_cfg_structure_changed(loop_result.any_structural_changed)
            .with_parameter_table_changed(loop_result.any_parameter_phase_changed),
    );
    if loop_result.any_parameter_phase_changed {
        outcome =
            outcome.with_note("canonicalize: normalized symbolic parameters and global phase");
    }
    if loop_result.any_structural_changed {
        outcome = outcome.with_note(
            "canonicalize: canonicalized instruction forms, barriers, and trivial no-ops",
        );
    }
    if !loop_result.stabilized {
        outcome = outcome
            .with_note(format!(
                "canonicalize: reached round limit after {} rounds before proving stability",
                loop_result.rounds_executed
            ))
            .with_diagnostic(crate::compiler::artifact::CompileDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: "compiler.canonicalize.round_limit_reached",
                message: format!(
                    "canonicalization stopped after {} rounds before reaching a fixed point",
                    loop_result.rounds_executed
                ),
            });
    }
    outcome
}
