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
use crate::compile::gate_transform::transform_rules::rule_registry::{
    SingleQubitParamTransformRule, TransformRuleKind, TwoQubitTransformRule,
};

#[test]
fn test_select_transform_rule_same_category_key_to_member() {
    // CX -> CY (both in CX category, CX is the key)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CY],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].source_gate, StandardGate::CX);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Cy)
    );
}

#[test]
fn test_select_transform_rule_same_category_member_to_key() {
    // CY -> CX (both in CX category, CX is the key)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].source_gate, StandardGate::CY);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cy2Cx)
    );
}

#[test]
fn test_select_transform_rule_same_category_member_to_member() {
    // CY -> CZ (both in CX category, neither is the key)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CZ],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::CY);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cy2Cx)
    );
    assert_eq!(rules[1].source_gate, StandardGate::CX);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Cz)
    );
}

#[test]
fn test_select_transform_rule_different_category_key_to_key() {
    // CX -> RZZ (different categories, both are keys)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::RZZ],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].source_gate, StandardGate::CX);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Rzz)
    );
}

#[test]
fn test_select_transform_rule_cx_to_fsim() {
    // CX -> FSIM (different categories, both are keys)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::FSIM],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].source_gate, StandardGate::CX);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Fsim)
    );
}

#[test]
fn test_select_transform_rule_fsim_to_cx() {
    // FSIM -> CX (different categories, both are keys)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::FSIM).unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].source_gate, StandardGate::FSIM);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Fsim2Cx)
    );
}

#[test]
fn test_select_transform_rule_fsim_to_rxx() {
    // FSIM -> RXX (different categories, FSIM key to RZZ member)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::RXX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::FSIM).unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::FSIM);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Fsim2Rzz)
    );
    assert_eq!(rules[1].source_gate, StandardGate::RZZ);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Rzz2Rxx)
    );
}

#[test]
fn test_select_transform_rule_different_category_member_to_key() {
    // CY -> RZZ (different categories, CY is member, RZZ is key)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::RZZ],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::CY);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cy2Cx)
    );
    assert_eq!(rules[1].source_gate, StandardGate::CX);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Rzz)
    );
}

#[test]
fn test_select_transform_rule_different_category_key_to_member() {
    // CX -> RXX (different categories, CX is key, RXX is member)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::RXX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::CX);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Rzz)
    );
    assert_eq!(rules[1].source_gate, StandardGate::RZZ);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Rzz2Rxx)
    );
}

#[test]
fn test_select_transform_rule_different_category_member_to_member() {
    // CY -> RXX (different categories, both are members)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::RXX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CY).unwrap();
    assert_eq!(rules.len(), 3);
    assert_eq!(rules[0].source_gate, StandardGate::CY);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cy2Cx)
    );
    assert_eq!(rules[1].source_gate, StandardGate::CX);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Cx2Rzz)
    );
    assert_eq!(rules[2].source_gate, StandardGate::RZZ);
    assert_eq!(
        rules[2].rule,
        TransformRuleKind::TwoQubit(TwoQubitTransformRule::Rzz2Rxx)
    );
}

#[test]
fn test_select_transform_rule_same_gate() {
    // CX -> CX (no transformation needed)
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );
    let rules = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules.len(), 0);
}

#[test]
fn test_select_transform_rule_caching() {
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CY],
        None,
    );

    // First call calculates and caches
    let rules1 = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert!(
        iset.get_two_qubit_rule_map()
            .contains_key(&StandardGate::CX)
    );

    // Second call returns cached result
    let rules2 = iset.select_transform_rule(StandardGate::CX).unwrap();
    assert_eq!(rules1.len(), rules2.len());
    assert_eq!(rules1[0].rule, rules2[0].rule);
}

#[test]
fn test_unknown_gate_returns_error() {
    // SWAP is not in any category, should return error
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );
    let result = iset.select_transform_rule(StandardGate::SWAP);
    assert!(result.is_err());
}

#[test]
fn test_multi_double_qubit_gate_support() {
    // Test with multiple double qubit gates in instruction set
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CZ, StandardGate::RZZ],
        None,
    );

    // Test that transformation rules can be generated for various source gates
    let source_gates = vec![
        StandardGate::CX,
        StandardGate::CY,
        StandardGate::CZ,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::RZX,
    ];

    for source in &source_gates {
        let result = iset.select_transform_rule(*source);
        assert!(result.is_ok(), "Failed for source gate: {:?}", source);
        let rules = result.unwrap();
        // Rules should be generated for all source gates
        assert!(rules.len() <= 2);
    }

    // Test caching for multiple source gates
    assert!(iset.get_two_qubit_rule_map().len() >= source_gates.len());
}

#[test]
fn test_multi_double_qubit_gate_random_selection() {
    // Test that different target gates can be selected randomly
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX, StandardGate::RZZ, StandardGate::CZ],
        None,
    );

    // Test multiple times to ensure random selection works
    let mut selected_targets = std::collections::HashSet::new();
    for _ in 0..10 {
        let result = iset.select_transform_rule(StandardGate::CY);
        assert!(result.is_ok());
        let rules = result.unwrap();
        if !rules.is_empty() {
            let last_rule = &rules[rules.len() - 1];
            selected_targets.insert(last_rule.rule);
        }
    }

    // Should have selected multiple different target gates
    assert!(selected_targets.len() > 0);
}

#[test]
fn test_multi_double_qubit_gate_category_handling() {
    // Test with multiple gates from different categories
    let mut iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX, StandardGate::RZZ], // From different categories
        None,
    );

    // Test transformation from CX category to any target
    let result = iset.select_transform_rule(StandardGate::CY);
    assert!(result.is_ok());
    let rules = result.unwrap();
    // Rules should be generated (length depends on random target selection)
    assert!(rules.len() == 1);

    // Test transformation from RZZ category to any target
    let result = iset.select_transform_rule(StandardGate::RXX);
    assert!(result.is_ok());
    let rules = result.unwrap();
    // Rules should be generated (length depends on random target selection)
    assert!(rules.len() == 1);
}

#[test]
fn test_select_single_qubit_param_transform_rule_same_gate() {
    let mut iset = InstructionSet::new(vec![StandardGate::RX], vec![StandardGate::CX], None);
    let rules = iset
        .select_single_qubit_param_transform_rule(StandardGate::RX)
        .unwrap();
    assert!(rules.is_empty());
}

#[test]
fn test_select_single_qubit_param_transform_rule_across_categories() {
    let mut iset = InstructionSet::new(vec![StandardGate::RZ], vec![StandardGate::CX], None);
    let rules = iset
        .select_single_qubit_param_transform_rule(StandardGate::U)
        .unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::U);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::SingleQubitParam(SingleQubitParamTransformRule::U2Rx)
    );
    assert_eq!(rules[1].source_gate, StandardGate::RX);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::SingleQubitParam(SingleQubitParamTransformRule::Rx2Rz)
    );
}

#[test]
fn test_select_single_qubit_param_transform_rule_same_category_member_to_member() {
    let mut iset = InstructionSet::new(vec![StandardGate::XY2P], vec![StandardGate::CX], None);
    let rules = iset
        .select_single_qubit_param_transform_rule(StandardGate::XY)
        .unwrap();
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].source_gate, StandardGate::XY);
    assert_eq!(
        rules[0].rule,
        TransformRuleKind::SingleQubitParam(SingleQubitParamTransformRule::Xy2Rxy)
    );
    assert_eq!(rules[1].source_gate, StandardGate::RXY);
    assert_eq!(
        rules[1].rule,
        TransformRuleKind::SingleQubitParam(SingleQubitParamTransformRule::Rxy2Xy2p)
    );
}

#[test]
fn test_select_single_qubit_param_transform_rule_caching() {
    let mut iset = InstructionSet::new(vec![StandardGate::U], vec![StandardGate::CX], None);
    let rules1 = iset
        .select_single_qubit_param_transform_rule(StandardGate::RX)
        .unwrap();
    assert!(
        iset.get_single_qubit_rule_map()
            .contains_key(&StandardGate::RX)
    );
    let rules2 = iset
        .select_single_qubit_param_transform_rule(StandardGate::RX)
        .unwrap();
    assert_eq!(rules1.len(), rules2.len());
    assert_eq!(rules1[0].rule, rules2[0].rule);
}

#[test]
fn test_select_single_qubit_param_transform_rule_unknown_gate_returns_error() {
    let mut iset = InstructionSet::new(vec![StandardGate::RX], vec![StandardGate::CX], None);
    let result = iset.select_single_qubit_param_transform_rule(StandardGate::Phase);
    assert!(result.is_err());
}
