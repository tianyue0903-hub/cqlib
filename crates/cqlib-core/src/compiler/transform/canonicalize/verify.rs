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

//! Input and output validation for the canonicalization contract.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, Directive, Instruction, Operation, Parameter, StandardGate,
};
use crate::compiler::CompilerError;
use std::collections::{BTreeSet, HashSet};

use super::config::CanonicalizeConfig;
use super::ops::{
    BarrierRelation, barrier_relation, canonical_control_flow_qubits_for_operation, is_strict_noop,
};
use super::params::{parameter_is_exact_zero, resolve_parameter};

#[derive(Debug, Clone, Copy)]
pub enum VerifyMode<'a> {
    Input,
    Output { config: &'a CanonicalizeConfig },
}

pub fn verify_circuit(circuit: &Circuit, mode: VerifyMode<'_>) -> Result<(), CompilerError> {
    if let CircuitParam::Index(index) = circuit.global_phase_param() {
        if circuit.parameters().get_index(*index as usize).is_none() {
            return Err(CompilerError::InvalidInput(format!(
                "global phase references missing parameter index {}",
                index
            )));
        }
    }
    verify_parameter_finite(&circuit.global_phase(), "global phase")?;
    verify_operations(circuit, circuit.operations(), "root", mode)?;

    if matches!(mode, VerifyMode::Output { .. }) {
        verify_output_parameter_table(circuit)?;
    }

    Ok(())
}

fn verify_operations(
    circuit: &Circuit,
    operations: &[Operation],
    scope: &str,
    mode: VerifyMode<'_>,
) -> Result<(), CompilerError> {
    let circuit_qubits = circuit.qubits();
    for (index, operation) in operations.iter().enumerate() {
        let op_scope = format!("{scope}[{index}]");

        for qubit in &operation.qubits {
            if !circuit_qubits.contains(qubit) {
                return Err(CompilerError::InvalidInput(format!(
                    "{op_scope} references unknown qubit {qubit}"
                )));
            }
        }

        verify_no_duplicate_qubits(operation, &op_scope)?;
        verify_instruction_arity(operation, &op_scope)?;
        verify_operation_params(circuit, operation, &op_scope)?;

        match &operation.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                if !circuit_qubits.contains(&gate.condition().qubit) {
                    return Err(CompilerError::InvalidInput(format!(
                        "{op_scope} condition references unknown qubit {}",
                        gate.condition().qubit
                    )));
                }
                let body_mode = match mode {
                    VerifyMode::Output { config } if !config.recurses_control_flow() => {
                        VerifyMode::Input
                    }
                    _ => mode,
                };
                verify_operations(circuit, gate.true_body(), "if_else.true", body_mode)?;
                if let Some(false_body) = gate.false_body() {
                    verify_operations(circuit, false_body, "if_else.false", body_mode)?;
                }
                verify_control_flow_qubits(circuit, operation, mode, &op_scope)?;
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                if !circuit_qubits.contains(&gate.condition().qubit) {
                    return Err(CompilerError::InvalidInput(format!(
                        "{op_scope} condition references unknown qubit {}",
                        gate.condition().qubit
                    )));
                }
                let body_mode = match mode {
                    VerifyMode::Output { config } if !config.recurses_control_flow() => {
                        VerifyMode::Input
                    }
                    _ => mode,
                };
                verify_operations(circuit, gate.body(), "while_loop.body", body_mode)?;
                verify_control_flow_qubits(circuit, operation, mode, &op_scope)?;
            }
            Instruction::Standard(StandardGate::GPhase) => match mode {
                VerifyMode::Input => {}
                VerifyMode::Output { config } if config.folds_gphase() => {
                    if scope == "root" {
                        return Err(CompilerError::InvariantViolation(format!(
                            "{op_scope} contains top-level GPhase after canonicalization"
                        )));
                    }
                    if index != 0 {
                        return Err(CompilerError::InvariantViolation(format!(
                            "{op_scope} contains non-leading control-flow body GPhase"
                        )));
                    }
                    if operation.params.len() != 1 {
                        return Err(CompilerError::InvariantViolation(format!(
                            "{op_scope} has malformed GPhase parameters"
                        )));
                    }
                    if parameter_is_exact_zero(&resolve_parameter(circuit, &operation.params[0])?)?
                    {
                        return Err(CompilerError::InvariantViolation(format!(
                            "{op_scope} contains zero GPhase"
                        )));
                    }
                }
                VerifyMode::Output { .. } => {}
            },
            Instruction::Directive(Directive::Barrier) if matches!(mode, VerifyMode::Output { config } if config.canonicalizes_barriers()) =>
            {
                verify_output_barrier(operation, operations.get(index + 1), &op_scope)?;
            }
            _ => {}
        }

        if matches!(mode, VerifyMode::Output { config } if config.drops_noops()) {
            let params = operation
                .params
                .iter()
                .map(|param| resolve_parameter(circuit, param))
                .collect::<Result<Vec<_>, _>>()?;
            if is_strict_noop(&operation.instruction, &params, &operation.qubits)? {
                return Err(CompilerError::InvariantViolation(format!(
                    "{op_scope} contains a removable no-op"
                )));
            }
        }
    }

    Ok(())
}

fn verify_no_duplicate_qubits(operation: &Operation, scope: &str) -> Result<(), CompilerError> {
    if matches!(
        operation.instruction,
        Instruction::Directive(Directive::Barrier)
    ) {
        return Ok(());
    }

    let mut seen = BTreeSet::new();
    for qubit in &operation.qubits {
        if !seen.insert(*qubit) {
            return Err(CompilerError::InvalidInput(format!(
                "{scope} contains duplicate qubit {qubit}"
            )));
        }
    }

    Ok(())
}

fn verify_instruction_arity(operation: &Operation, scope: &str) -> Result<(), CompilerError> {
    match &operation.instruction {
        Instruction::Standard(gate) => {
            verify_fixed_arity(
                gate.num_qubits(),
                operation.qubits.len(),
                gate.num_params(),
                operation.params.len(),
                scope,
            )?;
        }
        Instruction::McGate(gate) => {
            verify_fixed_arity(
                gate.num_qubits(),
                operation.qubits.len(),
                gate.num_params(),
                operation.params.len(),
                scope,
            )?;
        }
        Instruction::UnitaryGate(gate) => {
            verify_fixed_arity(
                gate.num_qubits() as usize,
                operation.qubits.len(),
                gate.num_params() as usize,
                operation.params.len(),
                scope,
            )?;
        }
        Instruction::CircuitGate(gate) => {
            verify_fixed_arity(
                gate.num_qubits(),
                operation.qubits.len(),
                gate.num_params(),
                operation.params.len(),
                scope,
            )?;
        }
        Instruction::Directive(Directive::Barrier) => {
            if !operation.params.is_empty() {
                return Err(CompilerError::InvalidInput(format!(
                    "{scope} barrier expects 0 parameters, got {}",
                    operation.params.len()
                )));
            }
        }
        Instruction::Directive(Directive::Measure | Directive::Reset) => {
            verify_fixed_arity(1, operation.qubits.len(), 0, operation.params.len(), scope)?;
        }
        Instruction::Delay => {
            verify_fixed_arity(1, operation.qubits.len(), 1, operation.params.len(), scope)?;
        }
        Instruction::ControlFlowGate(_) => {
            if !operation.params.is_empty() {
                return Err(CompilerError::InvalidInput(format!(
                    "{scope} control-flow operation expects 0 parameters, got {}",
                    operation.params.len()
                )));
            }
        }
    }
    Ok(())
}

fn verify_fixed_arity(
    expected_qubits: usize,
    actual_qubits: usize,
    expected_params: usize,
    actual_params: usize,
    scope: &str,
) -> Result<(), CompilerError> {
    if expected_qubits != actual_qubits {
        return Err(CompilerError::InvalidInput(format!(
            "{scope} qubit count mismatch: expected {expected_qubits}, got {actual_qubits}"
        )));
    }
    if expected_params != actual_params {
        return Err(CompilerError::InvalidInput(format!(
            "{scope} parameter count mismatch: expected {expected_params}, got {actual_params}"
        )));
    }
    Ok(())
}

fn verify_operation_params(
    circuit: &Circuit,
    operation: &Operation,
    scope: &str,
) -> Result<(), CompilerError> {
    for (param_index, param) in operation.params.iter().enumerate() {
        match param {
            CircuitParam::Fixed(value) => {
                if !value.is_finite() {
                    return Err(CompilerError::InvalidInput(format!(
                        "{scope} parameter {param_index} is non-finite: {value}"
                    )));
                }
            }
            CircuitParam::Index(index) => {
                let Some(param) = circuit.parameters().get_index(*index as usize) else {
                    return Err(CompilerError::InvalidInput(format!(
                        "{scope} references missing parameter index {index}"
                    )));
                };
                verify_parameter_finite(param, &format!("{scope} parameter {param_index}"))?;
            }
        }
    }
    Ok(())
}

fn verify_parameter_finite(param: &Parameter, scope: &str) -> Result<(), CompilerError> {
    if param.get_symbols().is_empty() {
        let value = param.evaluate(&None).map_err(|error| {
            CompilerError::InvalidInput(format!("{scope} cannot be evaluated: {error}"))
        })?;
        if !value.is_finite() {
            return Err(CompilerError::InvalidInput(format!(
                "{scope} evaluates to non-finite value {value}"
            )));
        }
    }
    Ok(())
}

fn verify_control_flow_qubits(
    circuit: &Circuit,
    operation: &Operation,
    mode: VerifyMode<'_>,
    scope: &str,
) -> Result<(), CompilerError> {
    let expected =
        canonical_control_flow_qubits_for_operation(&operation.instruction, &circuit.qubits());
    match mode {
        VerifyMode::Input => {
            for qubit in &expected {
                if !operation.qubits.contains(qubit) {
                    return Err(CompilerError::InvalidInput(format!(
                        "{scope} outer qubit list is missing required qubit {qubit}"
                    )));
                }
            }
        }
        VerifyMode::Output { .. } => {
            if operation.qubits != expected {
                return Err(CompilerError::InvariantViolation(format!(
                    "{scope} control-flow qubits are not canonical: expected {:?}, got {:?}",
                    expected, operation.qubits
                )));
            }
        }
    }
    Ok(())
}

fn verify_output_barrier(
    operation: &Operation,
    next: Option<&Operation>,
    scope: &str,
) -> Result<(), CompilerError> {
    if operation.qubits.is_empty() {
        return Err(CompilerError::InvariantViolation(format!(
            "{scope} contains empty barrier"
        )));
    }
    let mut sorted = operation.qubits.clone();
    sorted.sort_unstable_by_key(|qubit| qubit.id());
    sorted.dedup();
    if operation.qubits != sorted {
        return Err(CompilerError::InvariantViolation(format!(
            "{scope} barrier qubits are not sorted and deduplicated"
        )));
    }
    if operation.label.is_some() {
        return Err(CompilerError::InvariantViolation(format!(
            "{scope} barrier label was not cleared"
        )));
    }
    if let Some(next) = next {
        let relation = barrier_relation(&operation.qubits, &next.qubits);
        if matches!(next.instruction, Instruction::Directive(Directive::Barrier))
            && matches!(
                relation,
                BarrierRelation::Equal
                    | BarrierRelation::LeftSuperset
                    | BarrierRelation::RightSuperset
            )
        {
            return Err(CompilerError::InvariantViolation(format!(
                "{scope} has mergeable adjacent barrier"
            )));
        }
    }
    Ok(())
}

fn verify_output_parameter_table(circuit: &Circuit) -> Result<(), CompilerError> {
    let mut used = BTreeSet::new();
    if let CircuitParam::Index(index) = circuit.global_phase_param() {
        used.insert(*index);
    }
    collect_used_parameters(circuit.operations(), &mut used);

    for index in 0..circuit.parameters().len() {
        if !used.contains(&(index as u32)) {
            return Err(CompilerError::InvariantViolation(format!(
                "parameter table contains unused parameter index {index}"
            )));
        }
    }

    // Canonicalization rebuilds the parameter table from actual operation and
    // phase usage. The symbol cache must exactly reflect that rebuilt table or
    // downstream symbolic-parameter analysis can become stale.
    let mut expected_symbols = HashSet::new();
    for param in circuit.parameters() {
        expected_symbols.extend(param.get_symbols());
    }
    let actual_symbols: HashSet<_> = circuit.symbols().iter().cloned().collect();
    if expected_symbols != actual_symbols {
        return Err(CompilerError::InvariantViolation(
            "symbol table does not match canonical parameter table".to_string(),
        ));
    }

    Ok(())
}

fn collect_used_parameters(operations: &[Operation], used: &mut BTreeSet<u32>) {
    for operation in operations {
        for param in &operation.params {
            if let CircuitParam::Index(index) = param {
                used.insert(*index);
            }
        }
        match &operation.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                collect_used_parameters(gate.true_body(), used);
                if let Some(false_body) = gate.false_body() {
                    collect_used_parameters(false_body, used);
                }
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                collect_used_parameters(gate.body(), used);
            }
            _ => {}
        }
    }
}
