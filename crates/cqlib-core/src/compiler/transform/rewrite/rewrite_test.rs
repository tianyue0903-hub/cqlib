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

use super::{KnowledgeRewriter, RewriteConfig, RewriteMode};
use crate::circuit::symbolic_matrix::{circuit_to_symbolic_matrix, evaluate_symbolic_matrix};
use crate::circuit::{
    Circuit, CircuitParam, ConditionView, Directive, Instruction, Operation, Parameter, Qubit,
    StandardGate, circuit_to_matrix,
};
use crate::compiler::context::CompilerContext;
use crate::compiler::knowledge::library::RuleKind;
use crate::compiler::transform::Transformer;
use indexmap::IndexSet;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

const MATRIX_ASSERT_EPS: f64 = 1e-10;

fn standard_operation(
    gate: StandardGate,
    qubits: &[Qubit],
    params: SmallVec<[CircuitParam; 1]>,
) -> Operation {
    Operation {
        instruction: Instruction::Standard(gate),
        qubits: SmallVec::from_slice(qubits),
        params,
        label: None,
    }
}

fn rewrite_circuit(circuit: Circuit, rewriter: KnowledgeRewriter) -> (Circuit, bool) {
    let mut ctx = CompilerContext::new(circuit);
    let outcome = rewriter.run(&mut ctx).unwrap();
    (ctx.circuit().clone(), outcome.changed)
}

fn assert_matrix_approx_eq(lhs: &Array2<Complex64>, rhs: &Array2<Complex64>) {
    assert_eq!(lhs.shape(), rhs.shape());
    for ((row, col), lhs_value) in lhs.indexed_iter() {
        let rhs_value = rhs[[row, col]];
        let delta = (*lhs_value - rhs_value).norm();
        assert!(
            delta <= MATRIX_ASSERT_EPS,
            "matrix mismatch at ({row}, {col}): lhs={lhs_value:?}, rhs={rhs_value:?}, delta={delta}"
        );
    }
}

fn assert_numeric_matrix_preserved(before: &Circuit, after: &Circuit) {
    let before_matrix = circuit_to_matrix(before, None).unwrap();
    let after_matrix = circuit_to_matrix(after, None).unwrap();
    assert_matrix_approx_eq(&before_matrix, &after_matrix);
}

fn assert_symbolic_matrix_preserved_for_bindings<'a>(
    before: &Circuit,
    after: &Circuit,
    bindings: &[HashMap<&'a str, f64>],
) {
    let before_matrix = circuit_to_symbolic_matrix(before, None).unwrap();
    let after_matrix = circuit_to_symbolic_matrix(after, None).unwrap();
    for binding in bindings {
        let before_evaluated =
            evaluate_symbolic_matrix(&before_matrix, &Some(binding.clone())).unwrap();
        let after_evaluated =
            evaluate_symbolic_matrix(&after_matrix, &Some(binding.clone())).unwrap();
        assert_matrix_approx_eq(&before_evaluated, &after_evaluated);
    }
}

#[test]
fn cancels_adjacent_self_inverse_gates() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
}

#[test]
fn matches_across_disjoint_operations() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(1)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    KnowledgeRewriter::production().run(&mut ctx).unwrap();

    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::X)
    ));
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(1)]);
}

#[test]
fn applies_multiple_non_overlapping_rewrites_in_one_round() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(1)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::new(
        RewriteConfig::new()
            .with_enabled_kinds(vec![RuleKind::Cancel])
            .with_max_rounds(1),
    )
    .run(&mut ctx)
    .unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
}

#[test]
fn does_not_cross_dependent_operations() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter =
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]));
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 3);
}

#[test]
fn max_pattern_len_bounds_rule_search() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::new()
            .with_enabled_kinds(vec![RuleKind::Cancel])
            .with_max_pattern_len(1),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 2);
}

#[test]
fn zero_max_rounds_is_invalid() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let err = KnowledgeRewriter::new(RewriteConfig::new().with_max_rounds(0))
        .run(&mut ctx)
        .unwrap_err();

    assert!(err.to_string().contains("max_rounds"));
}

#[test]
fn skips_labeled_operations_by_default() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            StandardGate::H.into(),
            [Qubit::new(0)],
            std::iter::empty(),
            Some("keep"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 2);
}

#[test]
fn rewrites_labeled_operations_when_configured() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            StandardGate::H.into(),
            [Qubit::new(0)],
            std::iter::empty(),
            Some("rewrite"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::new(
        RewriteConfig::new()
            .skip_labeled_ops(false)
            .with_enabled_kinds(vec![RuleKind::Cancel]),
    )
    .run(&mut ctx)
    .unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
}

#[test]
fn merges_numeric_rotations() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.25).unwrap();
    circuit.rz(Qubit::new(0), 0.5).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    KnowledgeRewriter::production().run(&mut ctx).unwrap();

    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert!(
        matches!(operations[0].params[0], crate::circuit::CircuitParam::Fixed(v) if (v - 0.75).abs() < 1e-12)
    );
}

#[test]
fn production_merges_same_axis_rotations_across_rounds() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.rx(Qubit::new(0), 1.0).unwrap();
    circuit
        .rx(Qubit::new(0), Parameter::from(0.0) - theta)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(
        matches!(operations[0].params[0], crate::circuit::CircuitParam::Fixed(v) if (v - 1.0).abs() < 1e-12)
    );
}

#[test]
fn symbolic_matrix_is_preserved_when_merging_symbolic_rotations() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), theta).unwrap();
    circuit.rz(Qubit::new(0), phi).unwrap();
    let original = circuit.clone();

    let (rewritten, changed) = rewrite_circuit(circuit, KnowledgeRewriter::production());

    assert!(changed);
    assert_eq!(rewritten.operations().len(), 1);
    assert!(matches!(
        rewritten.operations()[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert_symbolic_matrix_preserved_for_bindings(
        &original,
        &rewritten,
        &[
            HashMap::from([("theta", 0.125), ("phi", -0.375)]),
            HashMap::from([("theta", 1.25), ("phi", 0.5)]),
        ],
    );
}

#[test]
fn symbolic_matrix_is_preserved_when_cancelling_symbolic_rotations() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), theta.clone()).unwrap();
    circuit
        .rz(Qubit::new(0), Parameter::from(0.0) - theta)
        .unwrap();
    let original = circuit.clone();

    let (rewritten, changed) = rewrite_circuit(
        circuit,
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel])),
    );

    assert!(changed);
    assert!(rewritten.operations().is_empty());
    assert_symbolic_matrix_preserved_for_bindings(
        &original,
        &rewritten,
        &[
            HashMap::from([("theta", -0.5)]),
            HashMap::from([("theta", 2.0)]),
        ],
    );
}

#[test]
fn folds_top_level_gphase_replacements_into_global_phase() {
    let mut circuit = Circuit::new(1);
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2p(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    KnowledgeRewriter::production().run(&mut ctx).unwrap();

    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::X)
    ));
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!((phase + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
}

#[test]
fn numeric_matrix_is_preserved_when_folding_top_level_gphase_replacement() {
    let mut circuit = Circuit::new(1);
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2p(Qubit::new(0)).unwrap();
    let original = circuit.clone();

    let (rewritten, changed) = rewrite_circuit(circuit, KnowledgeRewriter::production());

    assert!(changed);
    assert_numeric_matrix_preserved(&original, &rewritten);
}

#[test]
fn removes_zero_gphase_operation() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            StandardGate::GPhase.into(),
            std::iter::empty::<Qubit>(),
            [0.0.into()],
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!(phase.abs() < 1e-12);
}

#[test]
fn merges_top_level_gphase_operations_into_global_phase() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            StandardGate::GPhase.into(),
            std::iter::empty::<Qubit>(),
            [0.25.into()],
            None,
        )
        .unwrap();
    circuit
        .append(
            StandardGate::GPhase.into(),
            std::iter::empty::<Qubit>(),
            [0.5.into()],
            None,
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!((phase - 0.75).abs() < 1e-12);
}

#[test]
fn rewrites_control_flow_bodies_without_lifting_phase() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    KnowledgeRewriter::production().run(&mut ctx).unwrap();

    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert!(gate.true_body().is_empty());
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn drops_gphase_replacements_inside_control_flow_body() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::X2P),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::X2P),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!(phase.abs() < 1e-12);

    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            let body = gate.true_body();
            assert_eq!(body.len(), 1);
            assert!(matches!(
                body[0].instruction,
                Instruction::Standard(StandardGate::X)
            ));
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn keeps_original_gphase_inside_control_flow_body() {
    let mut circuit = Circuit::new(2);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::GPhase),
        qubits: smallvec![],
        params: smallvec![CircuitParam::Fixed(0.25)],
        label: None,
    }];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!(phase.abs() < 1e-12);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            let body = gate.true_body();
            assert_eq!(body.len(), 1);
            assert!(matches!(
                body[0].instruction,
                Instruction::Standard(StandardGate::GPhase)
            ));
            assert!(matches!(
                body[0].params.as_slice(),
                [CircuitParam::Fixed(value)] if (*value - 0.25).abs() < 1e-12
            ));
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn control_flow_rewrite_interns_new_symbolic_parameters_in_parent_table() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(2);
    let (theta_index, _) = circuit.add_parameter(theta.clone());
    let (phi_index, _) = circuit.add_parameter(phi.clone());
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![CircuitParam::Index(theta_index as u32)],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![CircuitParam::Index(phi_index as u32)],
            label: None,
        },
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            let body = gate.true_body();
            assert_eq!(body.len(), 1);
            assert!(matches!(
                body[0].instruction,
                Instruction::Standard(StandardGate::RZ)
            ));
            let [CircuitParam::Index(index)] = body[0].params.as_slice() else {
                panic!("expected merged symbolic parameter index");
            };
            let merged = ctx
                .circuit()
                .parameters()
                .get_index(*index as usize)
                .unwrap()
                .clone()
                .simplify()
                .unwrap();
            assert_eq!(
                merged.get_symbols(),
                HashSet::from(["theta".to_string(), "phi".to_string()])
            );
            let bindings = Some(HashMap::from([("theta", 0.25), ("phi", 0.5)]));
            let value = merged.evaluate(&bindings).unwrap();
            assert!((value - 0.75).abs() < 1e-12);
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn rewrites_if_else_false_body() {
    let mut circuit = Circuit::new(2);
    let true_body = vec![standard_operation(
        StandardGate::X,
        &[Qubit::new(1)],
        smallvec![],
    )];
    let false_body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            true_body,
            Some(false_body),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert_eq!(gate.true_body().len(), 1);
            assert!(matches!(
                gate.true_body()[0].instruction,
                Instruction::Standard(StandardGate::X)
            ));
            assert!(gate.false_body().unwrap().is_empty());
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn rewriting_empty_if_else_body_recomputes_operation_qubits() {
    let mut circuit = Circuit::new(2);
    let true_body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), true_body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    match &operations[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert!(gate.true_body().is_empty());
        }
        other => panic!("expected if_else, got {other:?}"),
    }
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0)]);

    let usage = crate::compiler::analysis::QubitUsage::from_circuit(ctx.circuit());
    assert_eq!(usage.total_qubits_touched(), 1);
    assert!(usage.get(Qubit::new(1)).is_none());
}

#[test]
fn rewriting_empty_if_else_false_body_recomputes_operation_qubits() {
    let mut circuit = Circuit::new(2);
    let false_body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            Vec::new(),
            Some(false_body),
        )
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    match &operations[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert!(gate.true_body().is_empty());
            assert!(gate.false_body().unwrap().is_empty());
        }
        other => panic!("expected if_else, got {other:?}"),
    }
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0)]);

    let usage = crate::compiler::analysis::QubitUsage::from_circuit(ctx.circuit());
    assert_eq!(usage.total_qubits_touched(), 1);
    assert!(usage.get(Qubit::new(1)).is_none());
}

#[test]
fn rewrites_while_loop_body() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::WhileLoop(gate)) => {
            assert!(gate.body().is_empty());
        }
        other => panic!("expected while_loop, got {other:?}"),
    }
}

#[test]
fn rewriting_empty_while_loop_body_recomputes_operation_qubits() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    match &operations[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::WhileLoop(gate)) => {
            assert!(gate.body().is_empty());
        }
        other => panic!("expected while_loop, got {other:?}"),
    }
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0)]);

    let usage = crate::compiler::analysis::QubitUsage::from_circuit(ctx.circuit());
    assert_eq!(usage.total_qubits_touched(), 1);
    assert!(usage.get(Qubit::new(1)).is_none());
}

#[test]
fn does_not_rewrite_false_or_while_body_when_recurse_disabled() {
    let mut circuit = Circuit::new(2);
    let false_body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .if_else(
            ConditionView::new(Qubit::new(0), 1),
            Vec::new(),
            Some(false_body),
        )
        .unwrap();
    let while_body = vec![
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::H, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), while_body)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::new(RewriteConfig::new().recurse_control_flow(false))
        .run(&mut ctx)
        .unwrap();

    assert!(!outcome.changed);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert_eq!(gate.false_body().unwrap().len(), 2);
        }
        other => panic!("expected if_else, got {other:?}"),
    }
    match &ctx.circuit().operations()[1].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::WhileLoop(gate)) => {
            assert_eq!(gate.body().len(), 2);
        }
        other => panic!("expected while_loop, got {other:?}"),
    }
}

#[test]
fn invalid_parameter_index_returns_error() {
    let mut qubits = IndexSet::new();
    qubits.insert(Qubit::new(0));
    let circuit = Circuit::from_parts(
        qubits,
        IndexSet::new(),
        IndexSet::new(),
        vec![standard_operation(
            StandardGate::RZ,
            &[Qubit::new(0)],
            smallvec![CircuitParam::Index(0)],
        )],
        CircuitParam::Fixed(0.0),
    );
    let mut ctx = CompilerContext::new(circuit);

    let err = KnowledgeRewriter::production().run(&mut ctx).unwrap_err();

    assert!(
        err.to_string()
            .contains("invalid rewrite parameter index 0")
    );
}

#[test]
fn while_loop_gphase_replacement_is_dropped_by_policy() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        standard_operation(StandardGate::X2P, &[Qubit::new(1)], smallvec![]),
        standard_operation(StandardGate::X2P, &[Qubit::new(1)], smallvec![]),
    ];
    circuit
        .while_loop(ConditionView::new(Qubit::new(0), 1), body)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!(phase.abs() < 1e-12);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::WhileLoop(gate)) => {
            let body = gate.body();
            assert_eq!(body.len(), 1);
            assert!(matches!(
                body[0].instruction,
                Instruction::Standard(StandardGate::X)
            ));
        }
        other => panic!("expected while_loop, got {other:?}"),
    }
}

#[test]
fn does_not_rewrite_across_barrier() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.barrier(vec![Qubit::new(0)]).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[1].instruction,
        Instruction::Directive(Directive::Barrier)
    ));
}

#[test]
fn does_not_rewrite_across_measure() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[1].instruction,
        Instruction::Directive(Directive::Measure)
    ));
}

#[test]
fn does_not_rewrite_across_reset() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.reset(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[1].instruction,
        Instruction::Directive(Directive::Reset)
    ));
}

#[test]
fn does_not_rewrite_across_delay() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.delay(Qubit::new(0), 10.0.into()).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(operations[1].instruction, Instruction::Delay));
}

#[test]
fn recurse_control_flow_can_be_disabled() {
    let mut circuit = Circuit::new(2);
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit
        .if_else(ConditionView::new(Qubit::new(0), 1), body, None)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::new(RewriteConfig::new().recurse_control_flow(false))
        .run(&mut ctx)
        .unwrap();

    assert!(!outcome.changed);
    match &ctx.circuit().operations()[0].instruction {
        Instruction::ControlFlowGate(crate::circuit::ControlFlow::IfElse(gate)) => {
            assert_eq!(gate.true_body().len(), 2);
        }
        other => panic!("expected if_else, got {other:?}"),
    }
}

#[test]
fn applies_numeric_eqmod_condition() {
    let mut circuit = Circuit::new(1);
    circuit
        .rz(Qubit::new(0), 3.0 * std::f64::consts::PI)
        .unwrap();
    circuit
        .rz(Qubit::new(0), 3.0 * std::f64::consts::PI)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
    let phase = ctx.circuit().global_phase().evaluate(&None).unwrap();
    assert!((phase - std::f64::consts::PI).abs() < 1e-12);
}

#[test]
fn applies_symbolic_condition_when_provable() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit
        .rx(Qubit::new(0), Parameter::from(0.0) - theta)
        .unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter =
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]));
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().is_empty());
}

#[test]
fn leaves_unproved_symbolic_conditions_unchanged() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta).unwrap();
    circuit.rx(Qubit::new(0), phi).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter =
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]));
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 2);
}

#[test]
fn commute_rules_are_not_applied_as_ordinary_rewrites() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.25).unwrap();
    circuit.s(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter =
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Commute]));
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::S)
    ));
}

#[test]
fn uses_commute_rules_to_merge_across_commuting_gate() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.25).unwrap();
    circuit.s(Qubit::new(0)).unwrap();
    circuit.rz(Qubit::new(0), 0.5).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 2);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert!(
        matches!(operations[0].params[0], CircuitParam::Fixed(value) if (value - 0.75).abs() < 1e-12)
    );
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::S)
    ));
}

#[test]
fn decompose_rules_require_lowering_mode() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter =
        KnowledgeRewriter::new(RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Decompose]));
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 1);
    assert!(matches!(
        ctx.circuit().operations()[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn lowering_mode_applies_decomposition_rules() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::new()
            .with_mode(RewriteMode::Lowering)
            .with_enabled_kinds(vec![RuleKind::Decompose])
            .with_max_rounds(1),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    assert!(ctx.circuit().operations().iter().all(|operation| !matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::H)
    )));
}

#[test]
fn target_gates_do_not_invent_reverse_rules() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), -0.3).unwrap();
    circuit.rx(Qubit::new(0), 0.7).unwrap();
    circuit.rz(Qubit::new(0), 0.3).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::new()
            .with_target_gates(vec![StandardGate::RXY])
            .with_max_rounds(2),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::RX)
    ));
    assert!(matches!(
        operations[2].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
}

#[test]
fn target_gates_lower_cx_to_h_cz_h_when_cz_is_native() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::lowering()
            .with_target_gates(vec![StandardGate::H, StandardGate::CZ])
            .with_max_rounds(4),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::CZ)
    ));
    assert!(matches!(
        operations[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(1)]);
    assert_eq!(
        operations[1].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
    assert_eq!(operations[2].qubits.as_slice(), &[Qubit::new(1)]);
}

#[test]
fn numeric_matrix_is_preserved_when_lowering_cx_to_cz_basis() {
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let original = circuit.clone();

    let (rewritten, changed) = rewrite_circuit(
        circuit,
        KnowledgeRewriter::new(
            RewriteConfig::lowering()
                .with_target_gates(vec![StandardGate::H, StandardGate::CZ])
                .with_max_rounds(4),
        ),
    );

    assert!(changed);
    assert_numeric_matrix_preserved(&original, &rewritten);
}

#[test]
fn target_gate_mode_allows_more_ops_when_unsupported_ops_decrease() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::lowering()
            .with_target_gates(vec![StandardGate::RZ, StandardGate::RX])
            .with_max_rounds(1),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(operations.iter().all(|operation| matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::RZ | StandardGate::RX)
    )));
}

#[test]
fn target_gate_mode_rejects_more_ops_when_unsupported_ops_do_not_decrease() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::lowering()
            .with_target_gates(vec![StandardGate::H, StandardGate::RZ, StandardGate::RX])
            .with_max_rounds(1),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 1);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn target_gates_require_explicit_rules_for_compression() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(1)).unwrap();
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.h(Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let rewriter = KnowledgeRewriter::new(
        RewriteConfig::new()
            .with_target_gates(vec![StandardGate::H, StandardGate::CZ, StandardGate::CX])
            .with_max_rounds(2),
    );
    let outcome = rewriter.run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    let operations = ctx.circuit().operations();
    assert_eq!(operations.len(), 3);
    assert!(matches!(
        operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert!(matches!(
        operations[1].instruction,
        Instruction::Standard(StandardGate::CZ)
    ));
    assert!(matches!(
        operations[2].instruction,
        Instruction::Standard(StandardGate::H)
    ));
}

#[test]
fn default_production_does_not_compress_h_cz_h_without_explicit_rule() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(1)).unwrap();
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.h(Qubit::new(1)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();

    assert!(!outcome.changed);
    assert_eq!(ctx.circuit().operations().len(), 3);
}

#[test]
fn empty_target_gate_set_is_invalid() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let err = KnowledgeRewriter::new(RewriteConfig::new().with_target_gates(Vec::new()))
        .run(&mut ctx)
        .unwrap_err();

    assert!(err.to_string().contains("target gate set"));
}

#[test]
fn round_limit_reports_diagnostic_when_not_stable() {
    let mut circuit = Circuit::new(1);
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let outcome = KnowledgeRewriter::new(RewriteConfig::new().with_max_rounds(1))
        .run(&mut ctx)
        .unwrap();

    assert!(outcome.changed);
    assert!(
        outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "compiler.rewrite.round_limit_reached")
    );
}
