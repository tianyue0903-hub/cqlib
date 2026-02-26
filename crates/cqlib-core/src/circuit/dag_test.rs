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
    let circuit = Circuit::new(2);
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 空线路应该只有一个 entry 块
    assert_eq!(dag.num_blocks(), 1);
    assert!(dag.entry_block().is_some());

    let entry = dag.entry_block().unwrap();
    // entry 块没有操作但有 Return 终结符
    assert_eq!(dag.data[entry].len(), 0);
    assert!(matches!(
        dag.data[entry].terminator,
        Some(Terminator::Return)
    ));
}

#[test]
fn test_circuit_to_dag_simple() {
    // 创建一个简单线路: H(0) -> CX(0, 1) -> Measure(0)
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();

    // 转换为 DAG
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 无控制流时应该只有一个 entry 块
    assert_eq!(
        dag.num_blocks(),
        1,
        "Simple circuit should have exactly 1 block"
    );
    assert_eq!(dag.num_qubits(), 2);

    // 验证 entry 块包含所有操作
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
    // 创建一个带 if-else 的线路
    // if (q[0] == 1): X(q[1])
    // else: Z(q[1])
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // 先测量 q0 得到条件
    circuit.measure(q0).unwrap();

    // 构建 if-else
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

    // 转换为 DAG
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 验证: 应该有 entry, if_true, if_false, merge 等块
    assert!(
        dag.num_blocks() >= 4,
        "Expected at least 4 blocks, got {}",
        dag.num_blocks()
    );

    // 验证入口块
    let entry = dag.entry_block().unwrap();

    // 打印调试信息
    println!("Number of blocks: {}", dag.num_blocks());
    for (idx, block) in dag.blocks() {
        println!(
            "Block {:?}: label={:?}, ops={}, terminator={:?}",
            idx,
            block.label(),
            block.len(),
            block.terminator
        );
    }

    // 验证控制流边
    let mut true_branch_count = 0;
    let mut false_branch_count = 0;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];
        println!("Edge {:?} -> {:?}: {:?}", source, target, flow);
        match flow {
            FlowEdge::TrueBranch => true_branch_count += 1,
            FlowEdge::FalseBranch => false_branch_count += 1,
            FlowEdge::Unconditional => {}
        }
    }

    assert!(
        true_branch_count >= 1,
        "Should have at least one true branch"
    );
    assert!(
        false_branch_count >= 1,
        "Should have at least one false branch"
    );
}

#[test]
fn test_if_without_else() {
    // if (q[0] == 1): X(q[1])
    // 没有 else 分支 - 应该创建一个空的 false 块
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

    // 没有 false_body
    circuit.if_else(condition, true_body, None).unwrap();

    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 块结构：entry, true, false_empty, merge = 4 个块
    assert_eq!(dag.num_blocks(), 4, "If without else should have 4 blocks");

    let entry = dag.entry_block().unwrap();

    // 找到 true 和 false 块
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

    // 验证 true 块包含 X 门
    assert_eq!(dag.data[true_block.unwrap()].len(), 1);

    // 验证 false 块是空的（没有操作）但有 Jump 终结符
    assert_eq!(dag.data[false_block.unwrap()].len(), 0);
    assert!(
        matches!(
            dag.data[false_block.unwrap()].terminator,
            Some(Terminator::Jump(_))
        ),
        "Empty false block should have Jump terminator"
    );

    // 验证 true 和 false 都跳到同一个 merge 块
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

    assert_eq!(
        true_to_merge, false_to_merge,
        "True and false should merge to same block"
    );
}

#[test]
fn test_circuit_to_dag_while_loop() {
    // 创建一个带 while 循环的线路
    // while (q[0] == 1): H(q[1])
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    // 先测量 q0
    circuit.measure(q0).unwrap();

    // 构建 while 循环
    let condition = ConditionView::new(q0, 1);
    let body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::H),
        qubits: smallvec![q1],
        params: smallvec![],
        label: None,
    }];

    circuit.while_loop(condition, body).unwrap();

    // 转换为 DAG
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 块结构：entry(measure), cond, body, exit = 4 个块
    assert_eq!(
        dag.num_blocks(),
        4,
        "Expected 4 blocks: entry, cond, body, exit"
    );

    // 找到各个块
    let entry = dag.entry_block().unwrap();

    // entry 应该 Jump 到 cond
    assert!(matches!(
        dag.data[entry].terminator,
        Some(Terminator::Jump(_))
    ));

    // 找到 cond 块（entry 的跳转目标）
    let mut cond_block = None;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        if source == entry {
            cond_block = Some(target);
            break;
        }
    }
    let cond_block = cond_block.expect("Should have cond block");

    // cond 块应该有 Branch 终结符
    assert!(
        matches!(dag.data[cond_block].terminator, Some(Terminator::Branch(_))),
        "Cond block should have Branch terminator"
    );

    // 找到 body 和 exit 块
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

    // 验证 body 块包含 H 门
    assert_eq!(dag.data[body_block].len(), 1);
    assert!(matches!(
        dag.data[body_block].operations[0].instruction,
        Instruction::Standard(StandardGate::H)
    ));

    // 验证 body 块有回边到 cond 块
    let mut has_back_edge = false;
    for edge_idx in dag.data.edge_indices() {
        let (source, target) = dag.data.edge_endpoints(edge_idx).unwrap();
        let flow = &dag.data[edge_idx];

        if source == body_block && target == cond_block && matches!(flow, FlowEdge::Unconditional) {
            has_back_edge = true;
            break;
        }
    }
    assert!(
        has_back_edge,
        "Body block should have back edge to condition"
    );

    // 验证 exit 块有 Return 终结符
    assert!(
        matches!(dag.data[exit_block].terminator, Some(Terminator::Return)),
        "Exit block should have Return terminator"
    );
}

#[test]
fn test_circuit_to_dag_nested_control_flow() {
    // 创建嵌套控制流
    // if (q[0] == 1):
    //     while (q[1] == 1): X(q[2])
    // else:
    //     Z(q[2])
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    // 测量
    circuit.measure(q0).unwrap();
    circuit.measure(q1).unwrap();

    // 外层 if-else
    let outer_condition = ConditionView::new(q0, 1);

    // 内层 while 循环体
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

    // 转换为 DAG
    let dag = CircuitDag::from_circuit(&circuit).unwrap();

    // 验证: 嵌套控制流应该有更多块
    assert!(
        dag.num_blocks() >= 5,
        "Expected at least 5 blocks, got {}",
        dag.num_blocks()
    );

    println!(
        "Nested control flow - Number of blocks: {}",
        dag.num_blocks()
    );
    for (idx, block) in dag.blocks() {
        println!(
            "Block {:?}: label={:?}, ops={}",
            idx,
            block.label(),
            block.len()
        );
    }
}
