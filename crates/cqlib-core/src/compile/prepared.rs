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

//! Shared flat-circuit preparation helpers for compile passes.

use crate::circuit::dag::Terminator;
use crate::circuit::gate::Instruction;
use crate::circuit::{Circuit, CircuitDag, CircuitParam, Operation, Parameter, ParameterValue, Qubit};
use crate::compile::error::CompileError;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

#[derive(Debug, Clone)]
/// Internal struct `PreparedOperation` used by compile flat-circuit workflows.
pub(crate) struct PreparedOperation {
    /// Original operation from the source circuit.
    pub(crate) op: Operation,
    /// Logical-qubit indices corresponding to `op.qubits`.
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
}

#[derive(Debug, Clone)]
/// Internal struct `PreparedCircuit` used by compile flat-circuit workflows.
pub(crate) struct PreparedCircuit {
    /// Logical qubits in circuit ordering.
    pub(crate) logical_qubits: Vec<Qubit>,
    /// Parameter pool copied from source circuit.
    pub(crate) parameters: Vec<Parameter>,
    /// Validated operations with cached logical indices.
    pub(crate) operations: Vec<PreparedOperation>,
}

/// Appends one operation to an output circuit while resolving parameter references.
pub(crate) fn append_operation(
    output: &mut Circuit,
    op: &Operation,
    parameter_pool: &[Parameter],
) -> Result<(), CompileError> {
    let mut params: SmallVec<[ParameterValue; 3]> = smallvec![];
    for p in &op.params {
        match p {
            CircuitParam::Fixed(v) => params.push(ParameterValue::Fixed(*v)),
            CircuitParam::Index(index) => {
                let idx = *index as usize;
                let Some(param) = parameter_pool.get(idx) else {
                    return Err(CompileError::Internal(format!(
                        "operation references missing parameter index {}",
                        idx
                    )));
                };
                params.push(ParameterValue::Param(param.clone()));
            }
        }
    }

    output.append(
        op.instruction.clone(),
        op.qubits.clone(),
        params,
        op.label.as_deref(),
    )?;
    Ok(())
}

/// Validates and flattens a circuit into compile-friendly internal form.
///
/// The pass currently accepts only single-block, return-terminated DAGs and
/// only 1q/2q operations with no control-flow nodes.
pub(crate) fn preprocess_circuit(circuit: &Circuit) -> Result<PreparedCircuit, CompileError> {
    let dag = CircuitDag::from_circuit(circuit)
        .map_err(|err| CompileError::DagBuildFailed(err.to_string()))?;

    if dag.num_blocks() != 1 {
        return Err(CompileError::UnsupportedControlFlow);
    }

    let entry = dag.entry_block().ok_or(CompileError::MissingEntryBlock)?;
    let block = dag
        .data
        .node_weight(entry)
        .ok_or(CompileError::MissingEntryBlock)?;

    if !matches!(block.terminator, None | Some(Terminator::Return)) {
        return Err(CompileError::UnsupportedControlFlow);
    }

    let logical_qubits = circuit.qubits();
    let parameters = circuit.parameters().iter().cloned().collect();
    let logical_index_map: HashMap<Qubit, usize> = logical_qubits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, q)| (q, idx))
        .collect();

    let mut operations = Vec::with_capacity(block.operations.len());

    for (op_index, op) in block.operations.iter().enumerate() {
        match &op.instruction {
            Instruction::ControlFlowGate(_) => return Err(CompileError::UnsupportedControlFlow),
            Instruction::Directive(d) => {
                return Err(CompileError::UnsupportedInstruction {
                    op_index,
                    instruction: format!("Directive::{d}"),
                });
            }
            Instruction::Delay => {
                return Err(CompileError::UnsupportedInstruction {
                    op_index,
                    instruction: "Delay".to_string(),
                });
            }
            _ => {}
        }

        let arity = op.qubits.len();
        if arity != 1 && arity != 2 {
            return Err(CompileError::UnsupportedArity { op_index, arity });
        }

        let mut logical = SmallVec::<[usize; 2]>::with_capacity(arity);
        for &q in &op.qubits {
            let Some(&logical_idx) = logical_index_map.get(&q) else {
                return Err(CompileError::Internal(format!(
                    "qubit {q} not found in circuit logical ordering"
                )));
            };
            logical.push(logical_idx);
        }

        operations.push(PreparedOperation {
            op: op.clone(),
            logical_qubits: logical,
        });
    }

    Ok(PreparedCircuit {
        logical_qubits,
        parameters,
        operations,
    })
}
