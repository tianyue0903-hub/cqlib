// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::config::RewriteConfig;
use super::matcher::{
    CompiledRuleSet, RewriteInstructionKey, resolve_operation_param, select_rewrites,
};
use crate::circuit::{
    Circuit, CircuitParam, ConditionView, ControlFlow, Directive, IfElseGate, Instruction, MCGate,
    Operation, Parameter, ParameterValue, Qubit, StandardGate,
};
use crate::compiler::knowledge::library::{RuleKind, RuleLibrary};
use crate::compiler::knowledge::rule::{Condition, Rule, RuleItem};
use smallvec::smallvec;

// ---------------------------------------------------------------------------
// is_rewrite_safe_operation
// ---------------------------------------------------------------------------

#[test]
fn is_rewrite_safe_accepts_standard_gates() {
    let h = Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };
    let x = Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };
    let rz = Operation {
        instruction: Instruction::Standard(StandardGate::RZ),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![CircuitParam::Fixed(0.5)],
        label: None,
    };

    assert!(RewriteInstructionKey::from_instruction(&h.instruction).is_some());
    assert!(RewriteInstructionKey::from_instruction(&x.instruction).is_some());
    assert!(RewriteInstructionKey::from_instruction(&rz.instruction).is_some());
}

#[test]
fn is_rewrite_safe_accepts_mc_gate() {
    let mc_gate = Operation {
        instruction: Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
        qubits: smallvec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        params: smallvec![],
        label: None,
    };

    assert!(RewriteInstructionKey::from_instruction(&mc_gate.instruction).is_some());
}

#[test]
fn is_rewrite_safe_rejects_non_gate_like_operations() {
    let barrier = Operation {
        instruction: Instruction::Directive(Directive::Barrier),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };
    let measure = Operation {
        instruction: Instruction::Directive(Directive::Measure),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };
    let reset = Operation {
        instruction: Instruction::Directive(Directive::Reset),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };
    let delay = Operation {
        instruction: Instruction::Delay,
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![CircuitParam::Fixed(10.0)],
        label: None,
    };
    let if_else = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
            ConditionView::new(Qubit::new(0), 1),
            vec![],
            None,
        ))),
        qubits: smallvec![Qubit::new(0)],
        params: smallvec![],
        label: None,
    };

    assert!(RewriteInstructionKey::from_instruction(&barrier.instruction).is_none());
    assert!(RewriteInstructionKey::from_instruction(&measure.instruction).is_none());
    assert!(RewriteInstructionKey::from_instruction(&reset.instruction).is_none());
    assert!(RewriteInstructionKey::from_instruction(&delay.instruction).is_none());
    assert!(RewriteInstructionKey::from_instruction(&if_else.instruction).is_none());
}

// ---------------------------------------------------------------------------
// CompiledRuleSet::from_library
// ---------------------------------------------------------------------------

#[test]
fn compiled_rule_set_produces_matches_for_known_gates() {
    // Verify the compiled rule set is usable by checking that select_rewrites
    // can find matches for well-known gate patterns.
    let library = crate::compiler::knowledge::library::RuleLibrary::builtin_rules().unwrap();
    let rules = CompiledRuleSet::from_library(library).unwrap();

    // H*H cancel rule exists.
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let config = RewriteConfig::new();
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);

    // RZ+RZ merge rule exists.
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.3).unwrap();
    circuit.rz(Qubit::new(0), 0.7).unwrap();
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);

    // CX*CX cancel rule exists.
    let mut circuit = Circuit::new(2);
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);

    // The builtin library also contains MCGate decomposition rules.
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty::<ParameterValue>(),
            None,
        )
        .unwrap();
    let config = RewriteConfig::lowering().with_enabled_kinds(vec![RuleKind::Decompose]);
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);
}

// ---------------------------------------------------------------------------
// select_rewrites
// ---------------------------------------------------------------------------

fn make_hh_circuit() -> Circuit {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    circuit
}

fn builtin_rules() -> CompiledRuleSet {
    let library = crate::compiler::knowledge::library::RuleLibrary::builtin_rules().unwrap();
    CompiledRuleSet::from_library(library).unwrap()
}

fn compiled_rules_from(rules: Vec<Rule>, kind: RuleKind) -> CompiledRuleSet {
    let library = RuleLibrary::from_rules(rules, kind).unwrap();
    CompiledRuleSet::from_library(&library).unwrap()
}

fn symbolic_param(name: &str) -> ParameterValue {
    ParameterValue::Param(Parameter::symbol(name))
}

#[test]
fn select_rewrites_cancels_self_inverse_pair() {
    let circuit = make_hh_circuit();
    let operations = circuit.operations();
    let rules = builtin_rules();
    let config = RewriteConfig::new();

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();

    assert_eq!(selected.patches.len(), 1);
    let patch = &selected.patches[0];
    assert_eq!(patch.matched_positions, vec![0, 1]);
    // Cancel rule removes both H gates, resulting in zero replacements.
    assert!(patch.replacements.is_empty());
}

#[test]
fn select_rewrites_lowers_mcx1_to_cx() {
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
            [Qubit::new(0), Qubit::new(1)],
            std::iter::empty::<ParameterValue>(),
            None,
        )
        .unwrap();
    let rules = builtin_rules();
    let config = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_max_rounds(1);

    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();

    assert_eq!(selected.patches.len(), 1);
    let patch = &selected.patches[0];
    assert_eq!(patch.matched_positions, vec![0]);
    assert_eq!(patch.replacements.len(), 1);
    assert!(matches!(
        patch.replacements[0].instruction,
        Instruction::Standard(StandardGate::CX)
    ));
    assert_eq!(
        patch.replacements[0].qubits.as_slice(),
        &[Qubit::new(0), Qubit::new(1)]
    );
}

#[test]
fn select_rewrites_lowers_parameterized_mcrz1_to_crz() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(2);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RZ))),
            [Qubit::new(0), Qubit::new(1)],
            [ParameterValue::Param(theta.clone())],
            None,
        )
        .unwrap();
    let rules = builtin_rules();
    let config = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_max_rounds(1);

    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();

    assert_eq!(selected.patches.len(), 1);
    let replacement = &selected.patches[0].replacements[0];
    assert!(matches!(
        replacement.instruction,
        Instruction::Standard(StandardGate::CRZ)
    ));
    assert!(matches!(&replacement.params[0], ParameterValue::Param(param) if param == &theta));
}

#[test]
fn target_filter_allows_mc_gate_decomposition_only_when_rewrite_is_target_native() {
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::Z))),
            [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
            std::iter::empty::<ParameterValue>(),
            None,
        )
        .unwrap();
    let rules = builtin_rules();

    let rejected = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_target_instructions(vec![Instruction::Standard(StandardGate::H)])
        .with_max_rounds(1);
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &rejected).unwrap();
    assert!(selected.is_empty());

    let accepted = RewriteConfig::lowering()
        .with_enabled_kinds(vec![RuleKind::Decompose])
        .with_target_instructions(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CCX),
        ])
        .with_max_rounds(1);
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &accepted).unwrap();
    assert_eq!(selected.patches.len(), 1);
    assert_eq!(selected.patches[0].replacements.len(), 3);
}

#[test]
fn select_rewrites_merges_rotations() {
    let mut circuit = Circuit::new(1);
    circuit.rz(Qubit::new(0), 0.3).unwrap();
    circuit.rz(Qubit::new(0), 0.7).unwrap();

    let operations = circuit.operations();
    let rules = builtin_rules();
    let config = RewriteConfig::new();

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();

    assert_eq!(selected.patches.len(), 1);
    let patch = &selected.patches[0];
    assert_eq!(patch.matched_positions, vec![0, 1]);
    assert_eq!(patch.replacements.len(), 1);
    assert!(matches!(
        patch.replacements[0].instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    // The merged parameter should evaluate to 1.0.
    let merged = &patch.replacements[0].params[0];
    let value = match merged {
        crate::circuit::ParameterValue::Fixed(v) => *v,
        crate::circuit::ParameterValue::Param(p) => p.evaluate(&None).unwrap(),
    };
    assert!((value - 1.0).abs() < 1e-12);
}

#[test]
fn select_rewrites_rejects_overlapping_matches() {
    // Three H gates in a row: H(0) H(0) H(0).
    // The greedy selector must pick only one non-overlapping patch
    // (positions [0,1] or [1,2]), never two.
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let operations = circuit.operations();
    let rules = builtin_rules();
    let config = RewriteConfig::new();

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();

    assert_eq!(selected.patches.len(), 1);
    // The patch should cover the first two H gates (lower cost position wins).
    assert_eq!(selected.patches[0].matched_positions.len(), 2);
}

#[test]
fn select_rewrites_skips_labeled_anchor() {
    let mut circuit = Circuit::new(1);
    circuit
        .append(
            Instruction::Standard(StandardGate::H),
            [Qubit::new(0)],
            std::iter::empty::<crate::circuit::ParameterValue>(),
            Some("keep"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();

    let operations = circuit.operations();
    let rules = builtin_rules();
    let config = RewriteConfig::new(); // skip_labeled_ops = true by default

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_respects_pattern_len_limit() {
    let circuit = make_hh_circuit();
    let operations = circuit.operations();
    let rules = builtin_rules();
    // Cancel rules have match length >= 2, so max_pattern_len=1 blocks them.
    let config = RewriteConfig::new().with_max_pattern_len(1);

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_filters_by_rule_kind() {
    let circuit = make_hh_circuit();
    let operations = circuit.operations();
    let rules = builtin_rules();
    // Only enable Merge rules — H*H cancellation is a Cancel rule, so no match.
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Merge]);

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_no_match_for_independent_qubits() {
    // H(0) X(1): different qubits, no multi-qubit rule spans them.
    // No single-qubit rule matches a single H or single X.
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(1)).unwrap();

    let operations = circuit.operations();
    let rules = builtin_rules();
    let config = RewriteConfig::new();

    let selected = select_rewrites(&circuit, operations, &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn non_adjacent_match_checks_skipped_ops_against_future_matches() {
    let rule = Rule::new(
        "unsafe_non_adjacent_three_op_cancel",
        vec![
            RuleItem::standard(StandardGate::H, &[0], vec![]),
            RuleItem::standard(StandardGate::Z, &[1], vec![]),
            RuleItem::standard(StandardGate::X, &[1], vec![]),
        ],
        vec![],
    );
    let rules = compiled_rules_from(vec![rule], RuleKind::Cancel);
    let config = RewriteConfig::new()
        .with_enabled_kinds(vec![RuleKind::Cancel])
        .with_max_window_ops(4);

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(1)).unwrap();
    circuit.z(Qubit::new(0)).unwrap();
    circuit.z(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(0)).unwrap();

    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();

    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_honors_eq_condition_true_and_false() {
    let mut rule = Rule::new(
        "merge_equal_rz",
        vec![
            RuleItem::standard(StandardGate::RZ, &[0], vec![symbolic_param("a")]),
            RuleItem::standard(StandardGate::RZ, &[0], vec![symbolic_param("b")]),
        ],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("b"),
            )],
        )],
    );
    rule.conditions = Some(smallvec![Condition::Eq(
        Parameter::symbol("a"),
        Parameter::symbol("b"),
    )]);
    let rules = compiled_rules_from(vec![rule], RuleKind::Merge);
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Merge]);

    let mut matching = Circuit::new(1);
    matching.rz(Qubit::new(0), 0.25).unwrap();
    matching.rz(Qubit::new(0), 0.25).unwrap();
    let selected = select_rewrites(&matching, matching.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);

    let mut mismatching = Circuit::new(1);
    mismatching.rz(Qubit::new(0), 0.25).unwrap();
    mismatching.rz(Qubit::new(0), 0.5).unwrap();
    let selected =
        select_rewrites(&mismatching, mismatching.operations(), &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_requires_all_conditions() {
    let mut matching = Circuit::new(1);
    matching.rxy(Qubit::new(0), 0.25, 0.125).unwrap();
    matching.rxy(Qubit::new(0), -0.25, 0.125).unwrap();
    let rules = builtin_rules();
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]);

    let selected = select_rewrites(&matching, matching.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);
    assert!(selected.patches[0].replacements.is_empty());

    let mut mismatching = Circuit::new(1);
    mismatching.rxy(Qubit::new(0), 0.25, 0.125).unwrap();
    mismatching.rxy(Qubit::new(0), -0.25, 0.25).unwrap();
    let selected =
        select_rewrites(&mismatching, mismatching.operations(), &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_rejects_repeated_symbol_mismatch() {
    let rules = builtin_rules();
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Merge]);

    let mut matching = Circuit::new(1);
    matching.rxy(Qubit::new(0), 0.25, 0.125).unwrap();
    matching.rxy(Qubit::new(0), 0.5, 0.125).unwrap();
    let selected = select_rewrites(&matching, matching.operations(), &rules, &config).unwrap();
    assert_eq!(selected.patches.len(), 1);
    assert_eq!(selected.patches[0].replacements.len(), 1);

    let mut mismatching = Circuit::new(1);
    mismatching.rxy(Qubit::new(0), 0.25, 0.125).unwrap();
    mismatching.rxy(Qubit::new(0), 0.5, 0.25).unwrap();
    let selected =
        select_rewrites(&mismatching, mismatching.operations(), &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_respects_max_window_ops_boundary() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(1)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let rules = builtin_rules();

    let narrow = RewriteConfig::new()
        .with_enabled_kinds(vec![RuleKind::Cancel])
        .with_max_window_ops(1);
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &narrow).unwrap();
    assert!(selected.is_empty());

    let exact = RewriteConfig::new()
        .with_enabled_kinds(vec![RuleKind::Cancel])
        .with_max_window_ops(2);
    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &exact).unwrap();
    assert_eq!(selected.patches.len(), 1);
    assert_eq!(selected.patches[0].matched_positions, vec![0, 2]);
}

#[test]
fn select_rewrites_does_not_skip_labeled_intervening_operation() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit
        .append(
            Instruction::Standard(StandardGate::I),
            [Qubit::new(0)],
            std::iter::empty::<crate::circuit::ParameterValue>(),
            Some("keep"),
        )
        .unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let rules = builtin_rules();
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]);

    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn select_rewrites_does_not_backtrack_after_condition_failure() {
    let theta = Parameter::symbol("theta");
    let mut circuit = Circuit::new(1);
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.rx(Qubit::new(0), 1.0).unwrap();
    circuit
        .rx(Qubit::new(0), Parameter::from(0.0) - theta)
        .unwrap();
    let rules = builtin_rules();
    let config = RewriteConfig::new().with_enabled_kinds(vec![RuleKind::Cancel]);

    let selected = select_rewrites(&circuit, circuit.operations(), &rules, &config).unwrap();

    assert!(selected.is_empty());
}

// ---------------------------------------------------------------------------
// resolve_operation_param
// ---------------------------------------------------------------------------

#[test]
fn resolve_operation_param_fixed() {
    let circuit = Circuit::new(1);
    let param = resolve_operation_param(&circuit, &CircuitParam::Fixed(0.5)).unwrap();
    let value = param.evaluate(&None).unwrap();
    assert!((value - 0.5).abs() < 1e-12);
}

#[test]
fn resolve_operation_param_index() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    let (index, _) = circuit.add_parameter(theta.clone());

    let param = resolve_operation_param(&circuit, &CircuitParam::Index(index as u32)).unwrap();
    let symbols = param.get_symbols();
    assert!(symbols.contains("theta"));
}
