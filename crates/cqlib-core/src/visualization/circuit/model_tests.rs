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

//! Tests for the visualization data model module.

use super::*;
use crate::circuit::Qubit;
use crate::visualization::VisualCondition;

#[test]
fn test_visual_op_style_clone() {
    let style = VisualOpStyle::Gate;
    let cloned = style.clone();
    assert_eq!(style, cloned);

    let controlled = VisualOpStyle::Controlled { num_controls: 2 };
    let controlled_clone = controlled.clone();
    assert_eq!(controlled, controlled_clone);
}

#[test]
fn test_visual_op_style_equality() {
    assert_eq!(VisualOpStyle::Gate, VisualOpStyle::Gate);
    assert_eq!(
        VisualOpStyle::Controlled { num_controls: 1 },
        VisualOpStyle::Controlled { num_controls: 1 }
    );
    assert_ne!(
        VisualOpStyle::Controlled { num_controls: 1 },
        VisualOpStyle::Controlled { num_controls: 2 }
    );
}

#[test]
fn test_visual_op_style_all_variants() {
    let styles = vec![
        VisualOpStyle::Gate,
        VisualOpStyle::Controlled { num_controls: 1 },
        VisualOpStyle::Cz,
        VisualOpStyle::Swap,
        VisualOpStyle::Barrier,
        VisualOpStyle::Measure,
        VisualOpStyle::Reset,
        VisualOpStyle::Delay,
        VisualOpStyle::ControlFlow {
            kind: VisualControlFlowKind::IfStart,
        },
    ];

    for style in styles {
        let _cloned = style.clone();
        let _debug = format!("{style:?}");
    }
}

#[test]
fn test_visual_control_flow_kind_clone() {
    let if_else = VisualControlFlowKind::IfElseBlock {
        has_false_branch: true,
        condition: VisualCondition {
            qubit_id: 0,
            target: 1,
        },
    };
    let cloned = if_else.clone();
    assert_eq!(if_else, cloned);

    let while_block = VisualControlFlowKind::WhileBlock {
        condition: VisualCondition {
            qubit_id: 1,
            target: 0,
        },
    };
    let while_clone = while_block.clone();
    assert_eq!(while_block, while_clone);
}

#[test]
fn test_visual_control_flow_kind_equality() {
    let cond1 = VisualCondition {
        qubit_id: 0,
        target: 1,
    };
    let cond2 = VisualCondition {
        qubit_id: 0,
        target: 1,
    };
    let cond3 = VisualCondition {
        qubit_id: 0,
        target: 0,
    };

    assert_eq!(
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: true,
            condition: cond1
        },
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: true,
            condition: cond2
        }
    );

    assert_ne!(
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: true,
            condition: cond1
        },
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: false,
            condition: cond2
        }
    );

    assert_ne!(
        VisualControlFlowKind::WhileBlock { condition: cond1 },
        VisualControlFlowKind::WhileBlock { condition: cond3 }
    );
}

#[test]
fn test_visual_control_flow_kind_all_variants() {
    let kinds = vec![
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: true,
            condition: VisualCondition {
                qubit_id: 0,
                target: 1,
            },
        },
        VisualControlFlowKind::IfElseBlock {
            has_false_branch: false,
            condition: VisualCondition {
                qubit_id: 0,
                target: 0,
            },
        },
        VisualControlFlowKind::WhileBlock {
            condition: VisualCondition {
                qubit_id: 1,
                target: 1,
            },
        },
        VisualControlFlowKind::IfStart,
        VisualControlFlowKind::ElseStart,
        VisualControlFlowKind::WhileStart,
        VisualControlFlowKind::End,
    ];

    for kind in kinds {
        let _cloned = kind.clone();
        let _debug = format!("{kind:?}");
    }
}

#[test]
fn test_visual_condition_clone_copy() {
    let cond = VisualCondition {
        qubit_id: 5,
        target: 1,
    };
    let cloned = cond.clone();
    assert_eq!(cond, cloned);

    let copied = cond;
    assert_eq!(copied.qubit_id, 5);
    assert_eq!(copied.target, 1);
}

#[test]
fn test_visual_condition_equality() {
    let cond1 = VisualCondition {
        qubit_id: 0,
        target: 1,
    };
    let cond2 = VisualCondition {
        qubit_id: 0,
        target: 1,
    };
    let cond3 = VisualCondition {
        qubit_id: 0,
        target: 0,
    };

    assert_eq!(cond1, cond2);
    assert_ne!(cond1, cond3);
}

#[test]
fn test_visual_operation_clone() {
    let op = VisualOperation {
        column: 5,
        lanes: vec![0, 2],
        covered_lanes: vec![0, 1, 2],
        label: "CX".to_string(),
        params: vec!["π/2".to_string()],
        style: VisualOpStyle::Controlled { num_controls: 1 },
        span_box: true,
        children: None,
        span_cols: 2,
    };

    let cloned = op.clone();
    assert_eq!(op.column, cloned.column);
    assert_eq!(op.lanes, cloned.lanes);
    assert_eq!(op.covered_lanes, cloned.covered_lanes);
    assert_eq!(op.label, cloned.label);
    assert_eq!(op.params, cloned.params);
    assert_eq!(op.style, cloned.style);
    assert_eq!(op.span_box, cloned.span_box);
    assert_eq!(op.span_cols, cloned.span_cols);
}

#[test]
fn test_visual_operation_with_children() {
    let then_circuit = VisualCircuit {
        qubits: vec![Qubit::new(0)],
        operations: vec![],
        num_columns: 0,
    };

    let op = VisualOperation {
        column: 0,
        lanes: vec![0],
        covered_lanes: vec![0],
        label: "IF".to_string(),
        params: vec![],
        style: VisualOpStyle::ControlFlow {
            kind: VisualControlFlowKind::IfElseBlock {
                has_false_branch: false,
                condition: VisualCondition {
                    qubit_id: 0,
                    target: 1,
                },
            },
        },
        span_box: false,
        children: Some(VisualChildren::IfElse {
            then_circuit: Box::new(then_circuit),
            else_circuit: None,
        }),
        span_cols: 3,
    };

    let cloned = op.clone();
    match cloned.children {
        Some(VisualChildren::IfElse {
            then_circuit,
            else_circuit,
        }) => {
            assert_eq!(then_circuit.num_qubits(), 1);
            assert!(else_circuit.is_none());
        }
        _ => panic!("expected IfElse children"),
    }
}

#[test]
fn test_visual_circuit_num_qubits() {
    let circuit = VisualCircuit {
        qubits: vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        operations: vec![],
        num_columns: 0,
    };
    assert_eq!(circuit.num_qubits(), 3);
}

#[test]
fn test_visual_circuit_empty() {
    let circuit = VisualCircuit {
        qubits: vec![],
        operations: vec![],
        num_columns: 0,
    };
    assert_eq!(circuit.num_qubits(), 0);
    assert!(circuit.operations.is_empty());
}

#[test]
fn test_visual_circuit_with_operations() {
    let ops = vec![
        VisualOperation {
            column: 0,
            lanes: vec![0],
            covered_lanes: vec![0],
            label: "H".to_string(),
            params: vec![],
            style: VisualOpStyle::Gate,
            span_box: false,
            children: None,
            span_cols: 1,
        },
        VisualOperation {
            column: 1,
            lanes: vec![0, 1],
            covered_lanes: vec![0, 1],
            label: "CX".to_string(),
            params: vec![],
            style: VisualOpStyle::Controlled { num_controls: 1 },
            span_box: false,
            children: None,
            span_cols: 1,
        },
    ];

    let circuit = VisualCircuit {
        qubits: vec![Qubit::new(0), Qubit::new(1)],
        operations: ops,
        num_columns: 2,
    };

    assert_eq!(circuit.num_qubits(), 2);
    assert_eq!(circuit.operations.len(), 2);
    assert_eq!(circuit.num_columns, 2);
}

#[test]
fn test_visual_children_clone() {
    let then_circuit = VisualCircuit {
        qubits: vec![Qubit::new(0)],
        operations: vec![],
        num_columns: 0,
    };
    let else_circuit = VisualCircuit {
        qubits: vec![Qubit::new(1)],
        operations: vec![],
        num_columns: 0,
    };

    let if_else = VisualChildren::IfElse {
        then_circuit: Box::new(then_circuit.clone()),
        else_circuit: Some(Box::new(else_circuit.clone())),
    };
    let cloned = if_else.clone();

    match cloned {
        VisualChildren::IfElse {
            then_circuit,
            else_circuit,
        } => {
            assert_eq!(then_circuit.num_qubits(), 1);
            assert!(else_circuit.is_some());
            assert_eq!(else_circuit.unwrap().num_qubits(), 1);
        }
        _ => panic!("expected IfElse"),
    }

    let while_body = VisualCircuit {
        qubits: vec![Qubit::new(0)],
        operations: vec![],
        num_columns: 0,
    };
    let while_children = VisualChildren::While {
        body_circuit: Box::new(while_body),
    };
    let while_cloned = while_children.clone();
    match while_cloned {
        VisualChildren::While { body_circuit } => {
            assert_eq!(body_circuit.num_qubits(), 1);
        }
        _ => panic!("expected While"),
    }
}

#[test]
fn test_visual_children_all_variants() {
    let circuit = VisualCircuit {
        qubits: vec![Qubit::new(0)],
        operations: vec![],
        num_columns: 0,
    };

    let if_else = VisualChildren::IfElse {
        then_circuit: Box::new(circuit.clone()),
        else_circuit: None,
    };
    let _if_else_debug = format!("{if_else:?}");

    let while_children = VisualChildren::While {
        body_circuit: Box::new(circuit),
    };
    let _while_debug = format!("{while_children:?}");
}

#[test]
fn test_visual_circuit_debug_format() {
    let circuit = VisualCircuit {
        qubits: vec![Qubit::new(0), Qubit::new(1)],
        operations: vec![],
        num_columns: 0,
    };
    let debug_str = format!("{circuit:?}");
    assert!(debug_str.contains("VisualCircuit"));
    assert!(debug_str.contains("num_columns: 0"));
}

#[test]
fn test_visual_operation_debug_format() {
    let op = VisualOperation {
        column: 0,
        lanes: vec![0],
        covered_lanes: vec![0],
        label: "H".to_string(),
        params: vec![],
        style: VisualOpStyle::Gate,
        span_box: false,
        children: None,
        span_cols: 1,
    };
    let debug_str = format!("{op:?}");
    assert!(debug_str.contains("VisualOperation"));
    assert!(debug_str.contains("H"));
}

#[test]
fn test_visual_condition_debug_format() {
    let cond = VisualCondition {
        qubit_id: 3,
        target: 1,
    };
    let debug_str = format!("{cond:?}");
    assert!(debug_str.contains("VisualCondition"));
    assert!(debug_str.contains("qubit_id: 3"));
    assert!(debug_str.contains("target: 1"));
}
