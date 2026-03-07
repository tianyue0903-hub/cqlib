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

use crate::circuit::dag::{BasicBlock, CircuitDag, FlowEdge, Terminator};
use crate::circuit::gate::control_flow::{ConditionView, ControlFlow, IfElseGate, WhileLoopGate};
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::operation::Operation;
use crate::circuit::{Circuit, Qubit};
use smallvec::smallvec;

#[test]
fn test_basic_block_creation() {
    let block = BasicBlock::new();
    assert!(block.is_empty());
    assert!(!block.has_terminator());
    assert_eq!(block.len(), 0);
    assert!(block.label().is_none());
}

#[test]
fn test_basic_block_with_label() {
    let block = BasicBlock::new().with_label("test_block");
    assert_eq!(block.label(), Some("test_block"));
}

#[test]
fn test_basic_block_operations() {
    let mut block = BasicBlock::new();
    let q0 = Qubit::new(0);

    let op = Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![q0],
        params: smallvec![],
        label: None,
    };

    block.push_operation(op.clone());
    assert_eq!(block.len(), 1);
    assert!(!block.is_empty());

    block.extend_operations(vec![op.clone(), op.clone()]);
    assert_eq!(block.len(), 3);
}

#[test]
fn test_basic_block_terminator() {
    let mut block = BasicBlock::new();
    let q0 = Qubit::new(0);
    let condition = ConditionView::new(q0, 1);

    block.set_terminator(Terminator::Branch(condition));
    assert!(block.has_terminator());
    assert!(matches!(block.terminator, Some(Terminator::Branch(_))));
}

#[test]
fn test_circuit_dag_empty() {
    let dag = CircuitDag::new(2);
    assert_eq!(dag.num_qubits(), 2);
    assert_eq!(dag.num_blocks(), 0);
    assert!(dag.entry_block().is_none());
    assert_eq!(dag.qubits().len(), 2);
}

#[test]
fn test_circuit_dag_add_block() {
    let mut dag = CircuitDag::new(1);
    let block = BasicBlock::new().with_label("test");
    let idx = dag.add_block(block);

    assert_eq!(dag.num_blocks(), 1);
    assert_eq!(dag.data[idx].label(), Some("test"));
}

#[test]
fn test_empty_circuit_conversion() {
    // Empty circuit should have single entry block with Return
    let circuit = Circuit::new(2);
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    assert_eq!(dag.num_blocks(), 1);
    assert!(dag.entry_block().is_some());

    let entry = dag.entry_block().unwrap();
    assert_eq!(dag.data[entry].len(), 0);
    assert!(matches!(
        dag.data[entry].terminator,
        Some(Terminator::Return)
    ));
}

#[test]
fn test_circuit_to_dag_simple() {
    // Linear circuit: H(0) -> CX(0, 1) -> Measure(0)
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // No control flow = single entry block
    assert_eq!(
        dag.num_blocks(),
        1,
        "Simple circuit should have exactly 1 block"
    );
    assert_eq!(dag.num_qubits(), 2);

    let entry = dag.entry_block().unwrap();
    assert_eq!(
        dag.data[entry].len(),
        3,
        "Entry block should have 3 operations"
    );
    assert!(matches!(
        dag.data[entry].terminator,
        Some(Terminator::Return)
    ));
}

#[test]
fn test_circuit_to_dag_if_else() {
    // if (q[0] == 1): X(q[1]) else: Z(q[1])
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];

    circuit
        .if_else(condition, true_body, Some(false_body))
        .unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // entry, true, false, merge = 4 blocks
    assert!(dag.num_blocks() >= 4, "Expected at least 4 blocks");

    // Verify branch edges exist
    let mut true_branch_count = 0;
    let mut false_branch_count = 0;
    for edge_idx in dag.data.edge_indices() {
        let flow = &dag.data[edge_idx];
        match flow {
            FlowEdge::TrueBranch => true_branch_count += 1,
            FlowEdge::FalseBranch => false_branch_count += 1,
            FlowEdge::Unconditional => {}
        }
    }

    assert!(true_branch_count >= 1, "Should have true branch");
    assert!(false_branch_count >= 1, "Should have false branch");
}

#[test]
fn test_if_without_else() {
    // if (q[0] == 1): X(q[1]) - no else branch
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];

    circuit.if_else(condition, true_body, None).unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // entry, true, false_empty, merge = 4 blocks
    assert_eq!(dag.num_blocks(), 4, "If without else should have 4 blocks");

    let entry = dag.entry_block().unwrap();

    // Find true and false blocks
    let mut true_block = None;
    let mut false_block = None;

    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];

        if source == entry {
            match flow {
                FlowEdge::TrueBranch => true_block = Some(target),
                FlowEdge::FalseBranch => false_block = Some(target),
                _ => {}
            }
        }
    }

    assert!(true_block.is_some(), "Should have true branch");
    assert!(false_block.is_some(), "Should have false branch (empty)");

    // True block contains X gate
    assert_eq!(dag.data[true_block.unwrap()].len(), 1);

    // False block is empty with Jump terminator
    assert_eq!(dag.data[false_block.unwrap()].len(), 0);
    assert!(matches!(
        dag.data[false_block.unwrap()].terminator,
        Some(Terminator::Jump(_))
    ));

    // Both branches merge to same block
    let mut true_to_merge = None;
    let mut false_to_merge = None;

    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];

        if source == true_block.unwrap() && matches!(flow, FlowEdge::Unconditional) {
            true_to_merge = Some(target);
        }
        if source == false_block.unwrap() && matches!(flow, FlowEdge::Unconditional) {
            false_to_merge = Some(target);
        }
    }

    assert_eq!(true_to_merge, false_to_merge, "True and false should merge");
}

#[test]
fn test_circuit_to_dag_while_loop() {
    // while (q[0] == 1): H(q[1])
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    circuit.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];

    circuit.while_loop(condition, body).unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // entry, cond, body, exit = 4 blocks
    assert_eq!(dag.num_blocks(), 4, "Expected 4 blocks");

    let entry = dag.entry_block().unwrap();

    // Entry should jump to cond
    assert!(matches!(
        dag.data[entry].terminator,
        Some(Terminator::Jump(_))
    ));

    // Find cond block
    let mut cond_block = None;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        if source == entry {
            cond_block = Some(target);
            break;
        }
    }
    let cond_block = cond_block.expect("Should have cond block");

    // Cond block should have Branch terminator
    assert!(matches!(
        dag.data[cond_block].terminator,
        Some(Terminator::Branch(_))
    ));

    // Find body and exit blocks
    let mut body_block = None;
    let mut exit_block = None;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];

        if source == cond_block {
            match flow {
                FlowEdge::TrueBranch => body_block = Some(target),
                FlowEdge::FalseBranch => exit_block = Some(target),
                _ => {}
            }
        }
    }

    let body_block = body_block.expect("Should have body block");
    let exit_block = exit_block.expect("Should have exit block");

    // Body contains H gate
    assert_eq!(dag.data[body_block].len(), 1);

    // Body has back edge to cond
    let mut has_back_edge = false;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];

        if source == body_block && target == cond_block && matches!(flow, FlowEdge::Unconditional) {
            has_back_edge = true;
            break;
        }
    }
    assert!(has_back_edge, "Body should have back edge to condition");

    // Exit has Return terminator
    assert!(matches!(
        dag.data[exit_block].terminator,
        Some(Terminator::Return)
    ));
}

#[test]
fn test_circuit_to_dag_nested_control_flow() {
    // if (q[0] == 1): while (q[1] == 1): X(q[2]) else: Z(q[2])
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    circuit.measure(q0).unwrap();
    circuit.measure(q1).unwrap();

    // Outer if-else
    let outer_condition = ConditionView::new(q0, 1);

    // Inner while
    let while_condition = ConditionView::new(q1, 1);
    let while_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q2],
        params: smallvec![],
        label: None,
    }];

    let true_body = vec![Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
            while_condition,
            while_body,
        ))),
        qubits: smallvec![q1, q2],
        params: smallvec![],
        label: None,
    }];

    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q2],
        params: smallvec![],
        label: None,
    }];

    circuit
        .if_else(outer_condition, true_body, Some(false_body))
        .unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // Nested control flow should have multiple blocks
    assert!(dag.num_blocks() >= 5, "Expected at least 5 blocks");
}

#[test]
fn test_to_circuit_simple_linear() {
    // Round-trip: Circuit -> DAG -> Circuit
    let mut original = Circuit::new(2);
    original.h(Qubit::new(0)).unwrap();
    original.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    original.measure(Qubit::new(0)).unwrap();

    let dag = CircuitDag::from_circuit(&original).unwrap();
    let recovered = dag.to_circuit().unwrap();

    // Verify properties
    assert_eq!(recovered.num_qubits(), original.num_qubits());
    assert_eq!(recovered.operations().len(), original.operations().len());

    // Verify operation types match
    for (orig_op, recv_op) in original
        .operations()
        .iter()
        .zip(recovered.operations().iter())
    {
        assert_eq!(
            orig_op.instruction.to_string(),
            recv_op.instruction.to_string()
        );
        assert_eq!(orig_op.qubits.len(), recv_op.qubits.len());
        for (oq, rq) in orig_op.qubits.iter().zip(recv_op.qubits.iter()) {
            assert_eq!(oq.id(), rq.id());
        }
    }
}

#[test]
fn test_to_circuit_if_else() {
    // Round-trip: if-else Circuit -> DAG -> Circuit
    let mut original = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    original.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    original
        .if_else(condition, true_body, Some(false_body))
        .unwrap();

    let dag = CircuitDag::from_circuit(&original).unwrap();
    let recovered = dag.to_circuit().unwrap();

    // Verify structure: Measure + IfElse
    assert_eq!(recovered.operations().len(), 2);

    let if_else_op = &recovered.operations()[1];
    match &if_else_op.instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            assert_eq!(gate.condition().qubit.id(), q0.id());
            assert_eq!(gate.condition().target, 1);

            // Check true body
            let true_ops = gate.true_body();
            assert_eq!(true_ops.len(), 1);
            assert!(matches!(
                true_ops[0].instruction,
                Instruction::Standard(StandardGate::X)
            ));

            // Check false body
            let false_ops = gate.false_body();
            assert!(false_ops.is_some());
            assert_eq!(false_ops.unwrap().len(), 1);
        }
        _ => panic!("Expected IfElse control flow"),
    }
}

#[test]
fn test_to_circuit_while_loop() {
    // Round-trip: while loop Circuit -> DAG -> Circuit
    let mut original = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    original.measure(q0).unwrap();

    let condition = ConditionView::new(q0, 1);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];
    original.while_loop(condition, body).unwrap();

    let dag = CircuitDag::from_circuit(&original).unwrap();
    let recovered = dag.to_circuit().unwrap();

    // Verify structure: Measure + WhileLoop
    assert_eq!(recovered.operations().len(), 2);

    let while_op = &recovered.operations()[1];
    match &while_op.instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            assert_eq!(gate.condition().qubit.id(), q0.id());
            assert_eq!(gate.condition().target, 1);

            // Check loop body
            let body_ops = gate.body();
            assert_eq!(body_ops.len(), 1);
            assert!(matches!(
                body_ops[0].instruction,
                Instruction::Standard(StandardGate::H)
            ));
        }
        _ => panic!("Expected WhileLoop control flow"),
    }
}

#[test]
fn test_to_circuit_nested_if_in_while() {
    // Round-trip: nested control flow Circuit -> DAG -> Circuit
    let mut original = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    original.measure(q0).unwrap();
    original.measure(q1).unwrap();

    // Build inner if-else
    let if_condition = ConditionView::new(q1, 1);
    let if_true = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q2],
        params: smallvec![],
        label: None,
    }];
    let if_false = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Y),
        qubits: smallvec![q2],
        params: smallvec![],
        label: None,
    }];

    let if_op = Operation {
        instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
            if_condition,
            if_true,
            Some(if_false),
        ))),
        qubits: smallvec![q1, q2],
        params: smallvec![],
        label: None,
    };

    // Build outer while
    let while_condition = ConditionView::new(q0, 1);
    let while_body = vec![if_op];
    original.while_loop(while_condition, while_body).unwrap();

    let dag = CircuitDag::from_circuit(&original).unwrap();
    let recovered = dag.to_circuit().unwrap();

    // Verify: Measure, Measure, While(IfElse)
    assert_eq!(recovered.operations().len(), 3);

    let while_op = &recovered.operations()[2];
    match &while_op.instruction {
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_gate)) => {
            assert_eq!(while_gate.body().len(), 1);

            // Check inner IfElse
            match &while_gate.body()[0].instruction {
                Instruction::ControlFlowGate(ControlFlow::IfElse(if_gate)) => {
                    assert_eq!(if_gate.true_body().len(), 1);
                    assert_eq!(if_gate.false_body().unwrap().len(), 1);
                }
                _ => panic!("Expected nested IfElse in While body"),
            }
        }
        _ => panic!("Expected WhileLoop control flow"),
    }
}

use crate::circuit::CircuitError;

#[test]
fn test_invalid_dag_missing_true_branch() {
    // Manually construct an invalid DAG with a Branch terminator but no TrueBranch edge
    let mut dag = CircuitDag::new(1);
    let q0 = Qubit::new(0);
    let condition = ConditionView::new(q0, 1);

    // Create entry block with Branch terminator
    let entry_block = dag.add_block(BasicBlock::new().with_label("entry"));
    dag.set_entry_block(entry_block);
    dag.data[entry_block].set_terminator(Terminator::Branch(condition));

    // Only add FalseBranch edge, intentionally omit TrueBranch
    let false_target = dag.add_block(BasicBlock::new().with_label("false_target"));
    dag.data[false_target].set_terminator(Terminator::Return);
    dag.add_edge(entry_block, false_target, FlowEdge::FalseBranch);

    // to_circuit should return InvalidControlFlow error
    let result = dag.to_circuit();
    match result {
        Err(CircuitError::InvalidControlFlow(msg)) => {
            assert!(
                msg.contains("missing a TrueBranch"),
                "Error message should indicate missing TrueBranch, got: {}",
                msg
            );
        }
        _ => panic!(
            "Expected InvalidControlFlow error for missing TrueBranch edge, got {:?}",
            result
        ),
    }
}

#[test]
fn test_invalid_dag_missing_false_branch() {
    // Manually construct an invalid DAG with a Branch terminator but no FalseBranch edge
    let mut dag = CircuitDag::new(1);
    let q0 = Qubit::new(0);
    let condition = ConditionView::new(q0, 1);

    // Create entry block with Branch terminator
    let entry_block = dag.add_block(BasicBlock::new().with_label("entry"));
    dag.set_entry_block(entry_block);
    dag.data[entry_block].set_terminator(Terminator::Branch(condition));

    // Only add TrueBranch edge, intentionally omit FalseBranch
    let true_target = dag.add_block(BasicBlock::new().with_label("true_target"));
    dag.data[true_target].set_terminator(Terminator::Return);
    dag.add_edge(entry_block, true_target, FlowEdge::TrueBranch);

    // to_circuit should return InvalidControlFlow error
    let result = dag.to_circuit();
    match result {
        Err(CircuitError::InvalidControlFlow(msg)) => {
            assert!(
                msg.contains("missing a FalseBranch"),
                "Error message should indicate missing FalseBranch, got: {}",
                msg
            );
        }
        _ => panic!(
            "Expected InvalidControlFlow error for missing FalseBranch edge, got {:?}",
            result
        ),
    }
}

#[test]
fn test_invalid_dag_error_includes_block_info() {
    // Verify that error messages include the block label and index
    let mut dag = CircuitDag::new(1);
    let q0 = Qubit::new(0);
    let condition = ConditionView::new(q0, 1);

    // Create a labeled entry block with Branch terminator but no edges
    let entry_block = dag.add_block(BasicBlock::new().with_label("my_test_block"));
    dag.set_entry_block(entry_block);
    dag.data[entry_block].set_terminator(Terminator::Branch(condition));

    let result = dag.to_circuit();
    match result {
        Err(CircuitError::InvalidControlFlow(msg)) => {
            assert!(
                msg.contains("my_test_block"),
                "Error message should include block label, got: {}",
                msg
            );
            assert!(
                msg.contains("TrueBranch"),
                "Error message should indicate missing TrueBranch, got: {}",
                msg
            );
        }
        _ => panic!(
            "Expected InvalidControlFlow error with block details, got {:?}",
            result
        ),
    }
}

#[test]
fn test_invalid_dag_unlabeled_block() {
    // Verify error handling for unlabeled blocks
    let mut dag = CircuitDag::new(1);
    let q0 = Qubit::new(0);
    let condition = ConditionView::new(q0, 1);

    // Create unlabeled entry block with Branch terminator
    let entry_block = dag.add_block(BasicBlock::new()); // No label
    dag.set_entry_block(entry_block);
    dag.data[entry_block].set_terminator(Terminator::Branch(condition));

    let result = dag.to_circuit();
    match result {
        Err(CircuitError::InvalidControlFlow(msg)) => {
            assert!(
                msg.contains("<unlabeled>"),
                "Error message should indicate unlabeled block, got: {}",
                msg
            );
        }
        _ => panic!(
            "Expected InvalidControlFlow error for unlabeled block, got {:?}",
            result
        ),
    }
}
