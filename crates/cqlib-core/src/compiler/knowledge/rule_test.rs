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
use crate::circuit::{Directive, Instruction, MCGate, Parameter, ParameterValue, StandardGate};
use smallvec::smallvec;
use std::collections::HashSet;

#[test]
fn validate_accepts_cancel_h_rule() {
    let rule = Rule::new(
        "cancel_h",
        vec![
            RuleItem::standard(StandardGate::H, &[0], vec![]),
            RuleItem::standard(StandardGate::H, &[0], vec![]),
        ],
        vec![],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_accepts_symbolic_merge_rule() {
    let rule = Rule::new(
        "merge_rz",
        vec![
            RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("a")]),
            RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("b")]),
        ],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("b"),
            )],
        )],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_accepts_gphase_rule() {
    let rule = Rule::new(
        "merge_gphase",
        vec![
            RuleItem::standard(StandardGate::GPhase, &[], vec![ParameterValue::from("a")]),
            RuleItem::standard(StandardGate::GPhase, &[], vec![ParameterValue::from("b")]),
        ],
        vec![RuleItem::standard(
            StandardGate::GPhase,
            &[],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("b"),
            )],
        )],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn rule_item_standard_uses_none_for_empty_params_and_some_for_params() {
    let h = RuleItem::standard(StandardGate::H, &[0], vec![]);
    assert!(matches!(
        h.instruction,
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(h.qubits.as_slice(), &[0]);
    assert!(h.params.is_none());

    let rz = RuleItem::standard(
        StandardGate::RZ,
        &[0],
        vec![ParameterValue::Param(Parameter::symbol("theta"))],
    );
    assert!(matches!(
        rz.instruction,
        Instruction::Standard(StandardGate::RZ)
    ));
    assert_eq!(rz.qubits.as_slice(), &[0]);
    assert_eq!(rz.params.as_ref().map(|params| params.len()), Some(1));
}

#[test]
fn validate_accepts_multi_controlled_gate_rule_item() {
    let rule = Rule::new(
        "decompose_m3cx",
        vec![RuleItem::mc_gate(
            MCGate::new(3, StandardGate::X),
            &[0, 1, 2, 3],
            vec![],
        )],
        vec![RuleItem::standard(StandardGate::CCX, &[0, 1, 2], vec![])],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn rule_item_mc_gate_uses_none_for_empty_params_and_some_for_params() {
    let mcx = RuleItem::mc_gate(MCGate::new(3, StandardGate::X), &[0, 1, 2, 3], vec![]);
    assert!(matches!(mcx.instruction, Instruction::McGate(_)));
    assert_eq!(mcx.qubits.as_slice(), &[0, 1, 2, 3]);
    assert!(mcx.params.is_none());

    let mcrz = RuleItem::mc_gate(
        MCGate::new(2, StandardGate::RZ),
        &[0, 1, 2],
        vec![ParameterValue::Param(Parameter::symbol("theta"))],
    );
    assert!(matches!(mcrz.instruction, Instruction::McGate(_)));
    assert_eq!(mcrz.qubits.as_slice(), &[0, 1, 2]);
    assert_eq!(mcrz.params.as_ref().map(|params| params.len()), Some(1));
}

#[test]
fn rule_item_equivalent_to_accepts_identical_standard_items() {
    let lhs = RuleItem::standard(
        StandardGate::RZ,
        &[0],
        vec![ParameterValue::Param(Parameter::symbol("theta"))],
    );
    let rhs = RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("theta")]);

    assert!(lhs.equivalent_to(&rhs));
}

#[test]
fn rule_item_equivalent_to_accepts_identical_multi_controlled_items() {
    let lhs = RuleItem::mc_gate(MCGate::new(2, StandardGate::X), &[0, 1, 2], vec![]);
    let rhs = RuleItem::mc_gate(MCGate::new(2, StandardGate::X), &[0, 1, 2], vec![]);

    assert!(lhs.equivalent_to(&rhs));
}

#[test]
fn rule_item_equivalent_to_rejects_different_qubits_and_params() {
    let lhs = RuleItem::standard(
        StandardGate::RZ,
        &[0],
        vec![ParameterValue::Param(Parameter::symbol("theta"))],
    );
    let different_qubits =
        RuleItem::standard(StandardGate::RZ, &[1], vec![ParameterValue::from("theta")]);
    let different_params =
        RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("phi")]);

    assert!(!lhs.equivalent_to(&different_qubits));
    assert!(!lhs.equivalent_to(&different_params));
}

#[test]
fn validate_accepts_rewrite_on_subset_of_match_qubits() {
    let rule = Rule::new(
        "drop_second_qubit",
        vec![RuleItem::standard(StandardGate::CX, &[0, 1], vec![])],
        vec![RuleItem::standard(StandardGate::H, &[0], vec![])],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_accepts_dense_labels_across_multiple_match_items() {
    let rule = Rule::new(
        "dense_across_items",
        vec![
            RuleItem::standard(StandardGate::H, &[1], vec![]),
            RuleItem::standard(StandardGate::H, &[0], vec![]),
        ],
        vec![RuleItem::standard(StandardGate::CX, &[0, 1], vec![])],
    );

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_accepts_conditions_with_symbols_bound_in_match() {
    let mut rule = Rule::new(
        "conditioned_merge",
        vec![
            RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("a")]),
            RuleItem::standard(StandardGate::RZ, &[0], vec![ParameterValue::from("b")]),
        ],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("b"),
            )],
        )],
    );
    rule.conditions = Some(smallvec![
        Condition::Eq(Parameter::symbol("a"), Parameter::symbol("b")),
        Condition::EqMod(
            Parameter::symbol("a") + Parameter::symbol("b"),
            Parameter::from(0.0),
            Parameter::symbol("a")
        ),
    ]);

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_accepts_builtin_constants_in_rewrite_and_conditions() {
    let mut rule = Rule::new(
        "builtins",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a")
                    + Parameter::symbol("π")
                    + Parameter::symbol("pi")
                    + Parameter::symbol("e"),
            )],
        )],
    );
    rule.conditions = Some(smallvec![Condition::EqMod(
        Parameter::symbol("a") + Parameter::symbol("π"),
        Parameter::symbol("pi"),
        Parameter::symbol("e"),
    )]);

    assert_eq!(rule.validate(), Ok(()));
}

#[test]
fn validate_rejects_empty_match() {
    let rule = Rule::new(
        "bad",
        vec![],
        vec![RuleItem::standard(StandardGate::H, &[0], vec![])],
    );

    assert_eq!(rule.validate(), Err(RuleValidationError::EmptyMatch));
}

#[test]
fn validate_rejects_non_standard_instruction() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![0],
            params: None,
        }],
        vec![],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::UnsupportedInstruction { .. })
    ));
}

#[test]
fn validate_rejects_non_standard_rewrite_instruction() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::H, &[0], vec![])],
        vec![RuleItem {
            instruction: Instruction::Directive(Directive::Barrier),
            qubits: smallvec![0],
            params: None,
        }],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::UnsupportedInstruction { .. })
    ));
}

#[test]
fn validate_rejects_wrong_qubit_count() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::CX, &[0], vec![])],
        vec![],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::WrongQubitCount {
            instruction,
            expected: 2,
            got: 1,
        }) if instruction == "CX"
    ));
}

#[test]
fn validate_rejects_wrong_rewrite_qubit_count() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::CX, &[0, 1], vec![])],
        vec![RuleItem::standard(StandardGate::CX, &[0], vec![])],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::WrongQubitCount {
            instruction,
            expected: 2,
            got: 1,
        }) if instruction == "CX"
    ));
}

#[test]
fn validate_rejects_wrong_param_count() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::RZ, &[0], vec![])],
        vec![],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::WrongParamCount {
            instruction,
            expected: 1,
            got: 0,
        }) if instruction == "RZ"
    ));
}

#[test]
fn validate_rejects_wrong_rewrite_param_count() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![RuleItem::standard(StandardGate::RZ, &[0], vec![])],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::WrongParamCount {
            instruction,
            expected: 1,
            got: 0,
        }) if instruction == "RZ"
    ));
}

#[test]
fn validate_rejects_duplicate_gate_qubits() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::CX, &[0, 0], vec![])],
        vec![],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::DuplicateQubit {
            instruction,
            qubit: 0,
        }) if instruction == "CX"
    ));
}

#[test]
fn validate_rejects_duplicate_rewrite_gate_qubits() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::CX, &[0, 1], vec![])],
        vec![RuleItem::standard(StandardGate::CX, &[0, 0], vec![])],
    );

    assert!(matches!(
        rule.validate(),
        Err(RuleValidationError::DuplicateQubit {
            instruction,
            qubit: 0,
        }) if instruction == "CX"
    ));
}

#[test]
fn validate_reports_full_multi_controlled_instruction_name() {
    let wrong_qubits = Rule::new(
        "bad_mcx",
        vec![RuleItem::mc_gate(
            MCGate::new(3, StandardGate::X),
            &[0, 1, 2],
            vec![],
        )],
        vec![],
    );
    assert!(matches!(
        wrong_qubits.validate(),
        Err(RuleValidationError::WrongQubitCount {
            instruction,
            expected: 4,
            got: 3,
        }) if instruction == "MCX[3]"
    ));

    let wrong_params = Rule::new(
        "bad_mcrz",
        vec![RuleItem::mc_gate(
            MCGate::new(2, StandardGate::RZ),
            &[0, 1, 2],
            vec![],
        )],
        vec![],
    );
    assert!(matches!(
        wrong_params.validate(),
        Err(RuleValidationError::WrongParamCount {
            instruction,
            expected: 1,
            got: 0,
        }) if instruction == "MCRZ[2]"
    ));
}

#[test]
fn validate_rejects_unbound_rewrite_qubit() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::H, &[0], vec![])],
        vec![RuleItem::standard(StandardGate::H, &[1], vec![])],
    );

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::UnboundRewriteQubit { qubit: 1 })
    );
}

#[test]
fn validate_rejects_unbound_rewrite_symbol() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("b")],
        )],
    );

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::UnboundRewriteSymbol {
            symbol: "b".to_string()
        })
    );
}

#[test]
fn validate_rejects_unbound_condition_symbol() {
    let mut rule = Rule::new(
        "bad",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![],
    );
    rule.conditions = Some(smallvec![Condition::Eq(
        Parameter::symbol("b"),
        Parameter::from(0.0),
    )]);

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::UnboundConditionSymbol {
            symbol: "b".to_string()
        })
    );
}

#[test]
fn validate_rejects_unbound_eq_mod_condition_symbol() {
    let mut rule = Rule::new(
        "bad",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![],
    );
    rule.conditions = Some(smallvec![Condition::EqMod(
        Parameter::symbol("a"),
        Parameter::from(0.0),
        Parameter::symbol("period"),
    )]);

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::UnboundConditionSymbol {
            symbol: "period".to_string()
        })
    );
}

#[test]
fn validate_rejects_non_dense_qubit_labels() {
    let rule = Rule::new(
        "bad",
        vec![RuleItem::standard(StandardGate::H, &[1], vec![])],
        vec![],
    );

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::NonDenseQubitLabels { labels: vec![1] })
    );
}

#[test]
fn validate_rejects_non_dense_qubit_labels_with_gap() {
    let rule = Rule::new(
        "bad",
        vec![
            RuleItem::standard(StandardGate::H, &[0], vec![]),
            RuleItem::standard(StandardGate::H, &[2], vec![]),
        ],
        vec![],
    );

    assert_eq!(
        rule.validate(),
        Err(RuleValidationError::NonDenseQubitLabels { labels: vec![0, 2] })
    );
}

#[test]
fn num_qubits_counts_union_of_match_and_target_qubits() {
    let rule = Rule::new(
        "two_qubits",
        vec![RuleItem::standard(StandardGate::H, &[0], vec![])],
        vec![RuleItem::standard(StandardGate::H, &[1], vec![])],
    );

    assert_eq!(rule.num_qubits(), 2);
}

#[test]
fn num_qubits_returns_one_for_gphase_only_rule() {
    let rule = Rule::new(
        "gphase",
        vec![RuleItem::standard(
            StandardGate::GPhase,
            &[],
            vec![ParameterValue::from("a")],
        )],
        vec![RuleItem::standard(
            StandardGate::GPhase,
            &[],
            vec![ParameterValue::from("a")],
        )],
    );

    assert_eq!(rule.num_qubits(), 1);
}

#[test]
fn collect_free_symbols_includes_match_rewrite_and_conditions() {
    let mut rule = Rule::new(
        "symbols",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::from("a")],
        )],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("b"),
            )],
        )],
    );
    rule.conditions = Some(smallvec![
        Condition::Eq(Parameter::symbol("c"), Parameter::from(0.0)),
        Condition::EqMod(
            Parameter::symbol("d"),
            Parameter::symbol("g"),
            Parameter::symbol("f")
        ),
    ]);

    let expected = HashSet::from([
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "g".to_string(),
        "f".to_string(),
    ]);
    assert_eq!(rule.collect_free_symbols(), expected);
}

#[test]
fn collect_free_symbols_filters_builtin_constants() {
    let mut rule = Rule::new(
        "builtins",
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("a") + Parameter::symbol("π"),
            )],
        )],
        vec![RuleItem::standard(
            StandardGate::RZ,
            &[0],
            vec![ParameterValue::Param(
                Parameter::symbol("pi") + Parameter::symbol("e"),
            )],
        )],
    );
    rule.conditions = Some(smallvec![Condition::EqMod(
        Parameter::symbol("π"),
        Parameter::symbol("pi"),
        Parameter::symbol("e"),
    )]);

    assert_eq!(
        rule.collect_free_symbols(),
        HashSet::from(["a".to_string()])
    );
}
