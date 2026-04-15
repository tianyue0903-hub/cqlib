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

use crate::circuit::cfg::{FlowEdge, Terminator};
use crate::circuit::{Circuit, CircuitCFG, CircuitParam, ControlFlow, Operation};
use crate::compiler::error::CompilerError;
use rustworkx_core::petgraph::prelude::NodeIndex;

pub trait CircuitVerifier {
    fn verify(&self, circuit: &Circuit) -> Result<(), CompilerError>;
}

pub trait CfgVerifier {
    fn verify(&self, cfg: &CircuitCFG) -> Result<(), CompilerError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultCircuitVerifier;

#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultCfgVerifier;

impl CircuitVerifier for DefaultCircuitVerifier {
    fn verify(&self, circuit: &Circuit) -> Result<(), CompilerError> {
        verify_operation_list(circuit, circuit.operations(), "root")?;

        if let CircuitParam::Index(index) = circuit.global_phase_param() {
            if circuit.parameters().get_index(*index as usize).is_none() {
                return Err(CompilerError::InvariantViolation(format!(
                    "global phase references missing parameter index {}",
                    index
                )));
            }
        }

        Ok(())
    }
}

impl CfgVerifier for DefaultCfgVerifier {
    fn verify(&self, cfg: &CircuitCFG) -> Result<(), CompilerError> {
        if cfg.num_blocks() > 0 && cfg.entry_block().is_none() {
            return Err(CompilerError::InvariantViolation(
                "cfg with blocks must define an entry block".to_string(),
            ));
        }

        for (block_id, block) in cfg.blocks() {
            let outgoing: Vec<_> = cfg.outgoing_edges(block_id).collect();
            match block.terminator() {
                Some(Terminator::Branch(_)) => {
                    verify_branch_block(block_id, block.label(), &outgoing)?
                }
                Some(Terminator::Jump(target)) => {
                    verify_jump_block(block_id, block.label(), *target, &outgoing)?
                }
                Some(Terminator::Return) => {
                    if !outgoing.is_empty() {
                        return Err(CompilerError::InvariantViolation(format!(
                            "return block '{}' ({:?}) must not have outgoing edges",
                            block.label().unwrap_or("<unlabeled>"),
                            block_id
                        )));
                    }
                }
                None => {}
            }
        }

        Ok(())
    }
}

fn verify_operation_list(
    circuit: &Circuit,
    operations: &[Operation],
    scope: &str,
) -> Result<(), CompilerError> {
    let qubits = circuit.qubits();

    for (op_index, operation) in operations.iter().enumerate() {
        for qubit in &operation.qubits {
            if !qubits.contains(qubit) {
                return Err(CompilerError::InvariantViolation(format!(
                    "operation {} in {} references unknown qubit {}",
                    op_index, scope, qubit
                )));
            }
        }

        for param in &operation.params {
            if let CircuitParam::Index(index) = param {
                if circuit.parameters().get_index(*index as usize).is_none() {
                    return Err(CompilerError::InvariantViolation(format!(
                        "operation {} in {} references missing parameter index {}",
                        op_index, scope, index
                    )));
                }
            }
        }

        if let crate::circuit::Instruction::ControlFlowGate(control_flow) = &operation.instruction {
            match control_flow {
                ControlFlow::IfElse(gate) => {
                    if !qubits.contains(&gate.condition().qubit) {
                        return Err(CompilerError::InvariantViolation(format!(
                            "if_else condition in {} references unknown qubit {}",
                            scope,
                            gate.condition().qubit
                        )));
                    }
                    verify_operation_list(circuit, gate.true_body(), "if_else.true")?;
                    if let Some(false_body) = gate.false_body() {
                        verify_operation_list(circuit, false_body, "if_else.false")?;
                    }
                }
                ControlFlow::WhileLoop(gate) => {
                    if !qubits.contains(&gate.condition().qubit) {
                        return Err(CompilerError::InvariantViolation(format!(
                            "while_loop condition in {} references unknown qubit {}",
                            scope,
                            gate.condition().qubit
                        )));
                    }
                    verify_operation_list(circuit, gate.body(), "while_loop.body")?;
                }
            }
        }
    }

    Ok(())
}

fn verify_branch_block(
    block_id: NodeIndex,
    label: Option<&str>,
    outgoing: &[(NodeIndex, FlowEdge)],
) -> Result<(), CompilerError> {
    let true_edges = outgoing
        .iter()
        .filter(|(_, edge)| matches!(edge, FlowEdge::TrueBranch))
        .count();
    let false_edges = outgoing
        .iter()
        .filter(|(_, edge)| matches!(edge, FlowEdge::FalseBranch))
        .count();
    let unexpected_edges = outgoing
        .iter()
        .filter(|(_, edge)| !matches!(edge, FlowEdge::TrueBranch | FlowEdge::FalseBranch))
        .count();

    if true_edges != 1 || false_edges != 1 || unexpected_edges != 0 || outgoing.len() != 2 {
        return Err(CompilerError::InvariantViolation(format!(
            "branch block '{}' ({:?}) must have exactly one true edge and one false edge",
            label.unwrap_or("<unlabeled>"),
            block_id
        )));
    }

    Ok(())
}

fn verify_jump_block(
    block_id: NodeIndex,
    label: Option<&str>,
    target: NodeIndex,
    outgoing: &[(NodeIndex, FlowEdge)],
) -> Result<(), CompilerError> {
    let unconditional: Vec<_> = outgoing
        .iter()
        .filter(|(_, edge)| matches!(edge, FlowEdge::Unconditional))
        .collect();

    if unconditional.len() != 1 || outgoing.len() != 1 || unconditional[0].0 != target {
        return Err(CompilerError::InvariantViolation(format!(
            "jump block '{}' ({:?}) must have exactly one unconditional edge to {:?}",
            label.unwrap_or("<unlabeled>"),
            block_id,
            target
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CfgVerifier, CircuitVerifier, DefaultCfgVerifier, DefaultCircuitVerifier};
    use crate::circuit::cfg::{BasicBlock, FlowEdge, Terminator};
    use crate::circuit::{
        Circuit, CircuitCFG, CircuitParam, ConditionView, Operation, Qubit, StandardGate,
    };
    use crate::compiler::CompilerError;
    use indexmap::IndexSet;
    use smallvec::smallvec;

    #[test]
    fn circuit_verifier_rejects_invalid_parameter_index() {
        let circuit = Circuit::from_parts(
            IndexSet::from_iter([Qubit::new(0)]),
            IndexSet::default(),
            IndexSet::default(),
            vec![Operation {
                instruction: StandardGate::RX.into(),
                qubits: smallvec![Qubit::new(0)],
                params: smallvec![CircuitParam::Index(99)],
                label: None,
            }],
            CircuitParam::Fixed(0.0),
        );

        let err = DefaultCircuitVerifier.verify(&circuit).unwrap_err();
        assert!(matches!(err, CompilerError::InvariantViolation(_)));
    }

    #[test]
    fn cfg_verifier_rejects_missing_branch_edge() {
        let mut cfg = CircuitCFG::new(1);
        let entry = cfg.add_block(BasicBlock::new().with_label("entry"));
        let false_target = cfg.add_block(BasicBlock::new().with_label("false"));
        cfg.set_entry_block(entry);
        cfg.block_mut(entry)
            .unwrap()
            .set_terminator(Terminator::Branch(ConditionView::new(Qubit::new(0), 1)));
        cfg.add_edge(entry, false_target, FlowEdge::FalseBranch);

        let err = DefaultCfgVerifier.verify(&cfg).unwrap_err();
        assert!(matches!(err, CompilerError::InvariantViolation(_)));
    }
}
