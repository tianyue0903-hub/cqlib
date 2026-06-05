// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Definition-based decomposition.
//!
//! This module expands operations whose implementation is already represented
//! by another circuit. It intentionally does not synthesize matrix-only
//! unitaries or lower standard gates to a target basis.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, Parameter, Qubit,
    StandardGate, WhileLoopGate,
};
use crate::compile::CompilerError;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

const MAX_DEFINITION_DEPTH: usize = 64;

/// Expands all circuit-backed definitions in `circuit`.
///
/// The expansion handles [`Instruction::CircuitGate`] and circuit-backed
/// [`Instruction::UnitaryGate`] operations. Matrix-only unitary gates are left
/// unchanged for a later synthesis stage.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Instruction, Qubit, StandardGate};
/// use cqlib_core::compile::transform::decompose::expand_definitions;
///
/// let mut definition = Circuit::new(1);
/// definition.h(Qubit::new(0)).unwrap();
/// let h_definition = definition.to_gate("h_definition").unwrap();
///
/// let mut circuit = Circuit::new(1);
/// circuit
///     .append(h_definition, [Qubit::new(0)], [], None)
///     .unwrap();
///
/// let expanded = expand_definitions(&circuit).unwrap();
/// assert!(matches!(
///     expanded.operations()[0].instruction,
///     Instruction::Standard(StandardGate::H),
/// ));
/// ```
pub fn expand_definitions(circuit: &Circuit) -> Result<Circuit, CompilerError> {
    DefinitionExpander::new(circuit)?.run()
}

struct DefinitionExpander<'a> {
    source: &'a Circuit,
    target: Circuit,
    top_phase: Parameter,
}

impl<'a> DefinitionExpander<'a> {
    fn new(source: &'a Circuit) -> Result<Self, CompilerError> {
        Ok(Self {
            source,
            target: Circuit::from_qubits(source.qubits())?,
            top_phase: source.global_phase(),
        })
    }

    fn run(mut self) -> Result<Circuit, CompilerError> {
        let qubit_map = QubitMap::Identity;
        let symbol_bindings = HashMap::new();

        for operation in self.source.operations() {
            let mut operations = Vec::new();
            let phase = self.expand_operation(
                operation,
                self.source,
                &qubit_map,
                &symbol_bindings,
                0,
                &mut operations,
            )?;
            self.top_phase = self.top_phase.clone() + phase;
            for operation in operations {
                self.append_top_level(operation)?;
            }
        }

        self.target.set_global_phase(self.top_phase);
        Ok(self.target)
    }

    fn expand_operation(
        &mut self,
        operation: &Operation,
        context: &Circuit,
        qubit_map: &QubitMap<'_>,
        symbol_bindings: &HashMap<String, Parameter>,
        depth: usize,
        operations: &mut Vec<Operation>,
    ) -> Result<Parameter, CompilerError> {
        match &operation.instruction {
            Instruction::CircuitGate(gate) => {
                let definition = gate.circuit();
                self.expand_definition(
                    gate.name(),
                    definition.circuit(),
                    gate.num_qubits(),
                    gate.num_params(),
                    operation,
                    context,
                    qubit_map,
                    symbol_bindings,
                    depth,
                    operations,
                )
            }
            Instruction::UnitaryGate(gate) => {
                if let Some(definition) = gate.circuit().as_ref() {
                    self.expand_definition(
                        gate.label(),
                        definition.circuit(),
                        gate.num_qubits() as usize,
                        gate.num_params() as usize,
                        operation,
                        context,
                        qubit_map,
                        symbol_bindings,
                        depth,
                        operations,
                    )
                } else {
                    self.keep_operation(operation, context, qubit_map, symbol_bindings, operations)
                }
            }
            Instruction::ControlFlowGate(flow) => self.expand_control_flow(
                operation,
                flow,
                context,
                qubit_map,
                symbol_bindings,
                depth,
                operations,
            ),
            _ => self.keep_operation(operation, context, qubit_map, symbol_bindings, operations),
        }
    }

    // Definition expansion carries the recursive source context, qubit map,
    // symbol bindings, depth counter, and output buffer together. Grouping
    // them into a one-off context type would hide the dataflow without reducing
    // the actual contract of this recursive call.
    #[allow(clippy::too_many_arguments)]
    fn expand_definition(
        &mut self,
        name: &str,
        definition: &Circuit,
        expected_qubits: usize,
        expected_params: usize,
        operation: &Operation,
        context: &Circuit,
        qubit_map: &QubitMap<'_>,
        symbol_bindings: &HashMap<String, Parameter>,
        depth: usize,
        operations: &mut Vec<Operation>,
    ) -> Result<Parameter, CompilerError> {
        if depth >= MAX_DEFINITION_DEPTH {
            return Err(CompilerError::InvalidInput(format!(
                "definition expansion exceeded maximum recursion depth {MAX_DEFINITION_DEPTH} while expanding '{name}'"
            )));
        }

        if operation.qubits.len() != expected_qubits {
            return Err(CompilerError::InvalidInput(format!(
                "definition '{name}' expects {expected_qubits} qubits, got {}",
                operation.qubits.len()
            )));
        }
        if operation.params.len() != expected_params {
            return Err(CompilerError::InvalidInput(format!(
                "definition '{name}' expects {expected_params} parameters, got {}",
                operation.params.len()
            )));
        }

        let definition_symbols = definition.symbols();
        if definition_symbols.len() != expected_params {
            return Err(CompilerError::InvariantViolation(format!(
                "definition '{name}' signature has {expected_params} parameters but backing circuit has {} symbols",
                definition_symbols.len()
            )));
        }

        let resolved_params = self.resolve_params(context, &operation.params, symbol_bindings)?;

        // Each definition owns its formal-symbol scope.  The caller arguments
        // have already been resolved through the current scope above, so the
        // nested definition must receive a fresh binding table instead of a
        // merged parent table. Merging would make equal symbol names in nested
        // definitions accidentally capture outer bindings.
        let mut next_symbol_bindings = HashMap::with_capacity(definition_symbols.len());
        for (symbol, value) in definition_symbols.iter().zip(resolved_params) {
            next_symbol_bindings.insert(symbol.clone(), value);
        }

        let definition_qubits = definition.qubits();
        if definition_qubits.len() != expected_qubits {
            return Err(CompilerError::InvariantViolation(format!(
                "definition '{name}' signature has {expected_qubits} qubits but backing circuit has {} qubits",
                definition_qubits.len()
            )));
        }

        let mut next_qubit_map = HashMap::with_capacity(definition_qubits.len());
        for (inner, callsite) in definition_qubits.iter().zip(operation.qubits.iter()) {
            next_qubit_map.insert(*inner, map_qubit(*callsite, qubit_map)?);
        }
        let next_qubit_map = QubitMap::Mapped(&next_qubit_map);

        let mut phase = apply_symbol_bindings(definition.global_phase(), &next_symbol_bindings);
        for inner_operation in definition.operations() {
            let inner_phase = self.expand_operation(
                inner_operation,
                definition,
                &next_qubit_map,
                &next_symbol_bindings,
                depth + 1,
                operations,
            )?;
            phase = phase + inner_phase;
        }

        Ok(phase)
    }

    fn expand_control_flow(
        &mut self,
        operation: &Operation,
        flow: &ControlFlow,
        context: &Circuit,
        qubit_map: &QubitMap<'_>,
        symbol_bindings: &HashMap<String, Parameter>,
        depth: usize,
        operations: &mut Vec<Operation>,
    ) -> Result<Parameter, CompilerError> {
        let instruction = match flow {
            ControlFlow::IfElse(gate) => {
                let true_body =
                    self.expand_body(gate.true_body(), context, qubit_map, symbol_bindings, depth)?;
                let false_body = gate
                    .false_body()
                    .map(|body| self.expand_body(body, context, qubit_map, symbol_bindings, depth))
                    .transpose()?;
                let condition = gate.condition();
                Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                    crate::circuit::ConditionView::new(
                        map_qubit(condition.qubit, qubit_map)?,
                        condition.target,
                    ),
                    true_body,
                    false_body,
                )))
            }
            ControlFlow::WhileLoop(gate) => {
                let body =
                    self.expand_body(gate.body(), context, qubit_map, symbol_bindings, depth)?;
                let condition = gate.condition();
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                    crate::circuit::ConditionView::new(
                        map_qubit(condition.qubit, qubit_map)?,
                        condition.target,
                    ),
                    body,
                )))
            }
        };

        operations.push(Operation {
            instruction,
            qubits: map_qubits(&operation.qubits, qubit_map)?,
            params: smallvec![],
            label: operation.label.clone(),
        });
        Ok(Parameter::from(0.0))
    }

    fn expand_body(
        &mut self,
        body: &[Operation],
        context: &Circuit,
        qubit_map: &QubitMap<'_>,
        symbol_bindings: &HashMap<String, Parameter>,
        depth: usize,
    ) -> Result<Vec<Operation>, CompilerError> {
        let mut operations = Vec::with_capacity(body.len());
        let mut phase = Parameter::from(0.0);

        for operation in body {
            let operation_phase = self.expand_operation(
                operation,
                context,
                qubit_map,
                symbol_bindings,
                depth,
                &mut operations,
            )?;
            phase = phase + operation_phase;
        }

        if !phase.is_zero() {
            let param = self.intern_parameter(phase)?;
            operations.insert(
                0,
                Operation {
                    instruction: Instruction::Standard(StandardGate::GPhase),
                    qubits: smallvec![],
                    params: smallvec![param],
                    label: None,
                },
            );
        }

        Ok(operations)
    }

    fn keep_operation(
        &mut self,
        operation: &Operation,
        context: &Circuit,
        qubit_map: &QubitMap<'_>,
        symbol_bindings: &HashMap<String, Parameter>,
        operations: &mut Vec<Operation>,
    ) -> Result<Parameter, CompilerError> {
        let params = self
            .resolve_params(context, &operation.params, symbol_bindings)?
            .into_iter()
            .map(|param| self.intern_parameter(param))
            .collect::<Result<SmallVec<[CircuitParam; 1]>, _>>()?;

        operations.push(Operation {
            instruction: operation.instruction.clone(),
            qubits: map_qubits(&operation.qubits, qubit_map)?,
            params,
            label: operation.label.clone(),
        });
        Ok(Parameter::from(0.0))
    }

    fn resolve_params(
        &self,
        context: &Circuit,
        params: &[CircuitParam],
        symbol_bindings: &HashMap<String, Parameter>,
    ) -> Result<Vec<Parameter>, CompilerError> {
        params
            .iter()
            .map(|param| {
                let resolved = context.resolve_parameter(param)?;
                Ok(apply_symbol_bindings(resolved, symbol_bindings))
            })
            .collect()
    }

    fn intern_parameter(&mut self, parameter: Parameter) -> Result<CircuitParam, CompilerError> {
        Ok(self.target.map_param(parameter)?)
    }

    fn append_top_level(&mut self, operation: Operation) -> Result<(), CompilerError> {
        let params = operation
            .params
            .iter()
            .map(|param| self.target.parameter_value(param))
            .collect::<Result<Vec<_>, _>>()?;

        self.target.append(
            operation.instruction,
            operation.qubits,
            params,
            operation.label.as_deref(),
        )?;
        Ok(())
    }
}

enum QubitMap<'a> {
    Identity,
    Mapped(&'a HashMap<Qubit, Qubit>),
}

fn map_qubits(
    qubits: &[Qubit],
    qubit_map: &QubitMap<'_>,
) -> Result<SmallVec<[Qubit; 3]>, CompilerError> {
    if matches!(qubit_map, QubitMap::Identity) {
        return Ok(qubits.iter().copied().collect());
    }

    let mut mapped = SmallVec::with_capacity(qubits.len());
    for qubit in qubits {
        mapped.push(map_qubit(*qubit, qubit_map)?);
    }
    Ok(mapped)
}

fn map_qubit(qubit: Qubit, qubit_map: &QubitMap<'_>) -> Result<Qubit, CompilerError> {
    match qubit_map {
        QubitMap::Identity => Ok(qubit),
        QubitMap::Mapped(qubit_map) => qubit_map.get(&qubit).copied().ok_or_else(|| {
            CompilerError::InvalidInput(format!(
                "definition expansion references unmapped qubit {qubit}"
            ))
        }),
    }
}

fn apply_symbol_bindings(mut parameter: Parameter, map: &HashMap<String, Parameter>) -> Parameter {
    if map.is_empty() {
        return parameter;
    }

    if !binding_values_reference_bound_symbols(map) {
        for (symbol, value) in map {
            parameter = parameter.replace(symbol, value.clone());
        }
        return parameter;
    }

    let mut occupied = parameter.get_symbols();
    for (symbol, value) in map {
        occupied.insert(symbol.clone());
        occupied.extend(value.get_symbols());
    }

    let mut temp_map = Vec::with_capacity(map.len());
    for (index, (symbol, value)) in map.iter().enumerate() {
        let temp = fresh_temp_symbol(index, symbol, &mut occupied);
        parameter = parameter.replace(symbol, Parameter::symbol(&temp));
        temp_map.push((temp, value.clone()));
    }

    for (temp, value) in temp_map {
        parameter = parameter.replace(&temp, value);
    }

    parameter
}

fn binding_values_reference_bound_symbols(map: &HashMap<String, Parameter>) -> bool {
    map.values()
        .flat_map(Parameter::get_symbols)
        .any(|symbol| map.contains_key(&symbol))
}

fn fresh_temp_symbol(index: usize, symbol: &str, occupied: &mut HashSet<String>) -> String {
    let mut attempt = 0;
    loop {
        let candidate = format!("__CQLIB_DECOMPOSE_TMP_{index}_{attempt}_{symbol}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ParameterValue;
    use crate::circuit::gate::{FrozenCircuit, UnitaryGate};
    use ndarray::array;
    use num_complex::Complex;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn op_param(circuit: &Circuit, param: &CircuitParam) -> Parameter {
        match param {
            CircuitParam::Fixed(value) => Parameter::from(*value),
            CircuitParam::Index(index) => circuit.parameters()[*index as usize].clone(),
        }
    }

    #[test]
    fn expands_circuit_gate_with_qubit_and_parameter_mapping() {
        let mut inner = Circuit::new(2);
        let theta = Parameter::symbol("theta");
        let beta = Parameter::symbol("beta");
        inner.h(Qubit::new(0)).unwrap();
        inner.rx(Qubit::new(0), theta).unwrap();
        inner.rz(Qubit::new(1), beta + 1.0).unwrap();
        let gate = inner.to_gate("inner").unwrap();

        let mut outer = Circuit::new(2);
        let gamma = Parameter::symbol("gamma");
        let delta = Parameter::symbol("delta");
        outer
            .append(
                gate,
                [Qubit::new(1), Qubit::new(0)],
                [
                    ParameterValue::Param(gamma.clone()),
                    ParameterValue::Param(gamma + delta),
                ],
                None,
            )
            .unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        let operations = expanded.operations();
        assert_eq!(operations.len(), 3);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::H)
        ));
        assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(1)]);
        assert!(matches!(
            operations[1].instruction,
            Instruction::Standard(StandardGate::RX)
        ));
        assert_eq!(operations[1].qubits.as_slice(), &[Qubit::new(1)]);
        assert!(matches!(
            operations[2].instruction,
            Instruction::Standard(StandardGate::RZ)
        ));
        assert_eq!(operations[2].qubits.as_slice(), &[Qubit::new(0)]);

        let rx_param = op_param(&expanded, &operations[1].params[0]);
        let rz_param = op_param(&expanded, &operations[2].params[0]);
        let bindings = HashMap::from([("gamma", 2.0), ("delta", 3.0)]);
        assert_eq!(rx_param.evaluate(&Some(bindings.clone())).unwrap(), 2.0);
        assert_eq!(rz_param.evaluate(&Some(bindings)).unwrap(), 6.0);
    }

    #[test]
    fn uses_simultaneous_parameter_substitution() {
        let mut inner = Circuit::new(1);
        inner.rx(Qubit::new(0), Parameter::symbol("a")).unwrap();
        inner.rz(Qubit::new(0), Parameter::symbol("b")).unwrap();
        let gate = inner.to_gate("swap_params").unwrap();

        let mut outer = Circuit::new(1);
        outer
            .append(
                gate,
                [Qubit::new(0)],
                [
                    ParameterValue::Param(Parameter::symbol("b")),
                    ParameterValue::Param(Parameter::symbol("a")),
                ],
                None,
            )
            .unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        let operations = expanded.operations();
        let rx_param = op_param(&expanded, &operations[0].params[0]);
        let rz_param = op_param(&expanded, &operations[1].params[0]);
        assert_eq!(rx_param.as_symbol().as_deref(), Some("b"));
        assert_eq!(rz_param.as_symbol().as_deref(), Some("a"));
    }

    #[test]
    fn expands_nested_circuit_gates() {
        let mut leaf = Circuit::new(1);
        leaf.h(Qubit::new(0)).unwrap();
        let leaf_gate = leaf.to_gate("leaf").unwrap();

        let mut middle = Circuit::new(1);
        middle
            .rx(Qubit::new(0), Parameter::symbol("theta"))
            .unwrap();
        middle.append(leaf_gate, [Qubit::new(0)], [], None).unwrap();
        let middle_gate = middle.to_gate("middle").unwrap();

        let mut outer = Circuit::new(1);
        outer
            .append(
                middle_gate,
                [Qubit::new(0)],
                [ParameterValue::Param(Parameter::symbol("phi"))],
                None,
            )
            .unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        let operations = expanded.operations();
        assert_eq!(operations.len(), 2);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::RX)
        ));
        assert!(matches!(
            operations[1].instruction,
            Instruction::Standard(StandardGate::H)
        ));
        assert_eq!(
            op_param(&expanded, &operations[0].params[0])
                .as_symbol()
                .as_deref(),
            Some("phi")
        );
    }

    #[test]
    fn resolves_nested_definition_symbol_scopes_without_capture() {
        let mut leaf = Circuit::new(1);
        leaf.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
        let leaf_gate = leaf.to_gate("leaf").unwrap();

        let mut middle = Circuit::new(1);
        middle
            .append(
                leaf_gate,
                [Qubit::new(0)],
                [ParameterValue::Param(Parameter::symbol("theta") + 1.0)],
                None,
            )
            .unwrap();
        let middle_gate = middle.to_gate("middle").unwrap();

        let mut outer = Circuit::new(1);
        outer
            .append(
                middle_gate,
                [Qubit::new(0)],
                [ParameterValue::Param(Parameter::symbol("alpha"))],
                None,
            )
            .unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        let operations = expanded.operations();
        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::RX)
        ));

        let param = op_param(&expanded, &operations[0].params[0]);
        let bindings = HashMap::from([("alpha", 2.0)]);
        assert_eq!(param.evaluate(&Some(bindings)).unwrap(), 3.0);
        assert!(param.get_symbols().contains("alpha"));
        assert!(!param.get_symbols().contains("theta"));
    }

    #[test]
    fn merges_definition_global_phase_at_top_level() {
        let mut inner = Circuit::new(1);
        inner.set_global_phase(Parameter::from(0.5));
        inner.x(Qubit::new(0)).unwrap();
        let gate = inner.to_gate("phase_x").unwrap();

        let mut outer = Circuit::new(1);
        outer.set_global_phase(Parameter::from(0.25));
        outer.append(gate, [Qubit::new(0)], [], None).unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        assert_eq!(expanded.operations().len(), 1);
        assert!(matches!(
            expanded.operations()[0].instruction,
            Instruction::Standard(StandardGate::X)
        ));
        assert!((expanded.global_phase().evaluate(&None).unwrap() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn expands_circuit_backed_unitary_gate() {
        let mut inner = Circuit::new(1);
        inner.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
        let gate = UnitaryGate::new("unitary_rx", 1, 1)
            .with_circuit(Arc::new(FrozenCircuit::new(inner)))
            .unwrap();

        let mut outer = Circuit::new(1);
        outer
            .unitary_with_params(
                gate,
                vec![Qubit::new(0)],
                [ParameterValue::Param(Parameter::symbol("phi"))],
            )
            .unwrap();

        let expanded = expand_definitions(&outer).unwrap();
        let operations = expanded.operations();
        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::RX)
        ));
        assert_eq!(
            op_param(&expanded, &operations[0].params[0])
                .as_symbol()
                .as_deref(),
            Some("phi")
        );
    }

    #[test]
    fn keeps_matrix_only_unitary_gate() {
        let matrix = array![
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]
        ];
        let gate = UnitaryGate::new("matrix_x", 1, 0)
            .with_matrix(matrix)
            .unwrap();

        let mut circuit = Circuit::new(1);
        circuit.unitary(gate, vec![Qubit::new(0)]).unwrap();

        let expanded = expand_definitions(&circuit).unwrap();
        assert_eq!(expanded.operations().len(), 1);
        assert!(matches!(
            expanded.operations()[0].instruction,
            Instruction::UnitaryGate(_)
        ));
    }

    #[test]
    fn expands_definitions_in_control_flow_body_with_body_local_phase() {
        let mut inner = Circuit::new(1);
        inner.set_global_phase(Parameter::from(0.5));
        inner.x(Qubit::new(0)).unwrap();
        let gate = inner.to_gate("phase_x").unwrap();

        let body = vec![Operation {
            instruction: gate,
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        }];
        let mut circuit = Circuit::new(2);
        circuit
            .if_else(
                crate::circuit::ConditionView::new(Qubit::new(0), 1),
                body,
                None,
            )
            .unwrap();

        let expanded = expand_definitions(&circuit).unwrap();
        assert!(expanded.global_phase().is_zero());
        let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
            &expanded.operations()[0].instruction
        else {
            panic!("expected if-else operation");
        };
        let true_body = gate.true_body();
        assert_eq!(true_body.len(), 2);
        assert!(matches!(
            true_body[0].instruction,
            Instruction::Standard(StandardGate::GPhase)
        ));
        assert!(matches!(
            true_body[1].instruction,
            Instruction::Standard(StandardGate::X)
        ));
    }

    #[test]
    fn rejects_excessive_definition_depth() {
        let mut circuit = Circuit::new(1);
        circuit.x(Qubit::new(0)).unwrap();
        let mut gate = circuit.to_gate("level_0").unwrap();

        for index in 1..=MAX_DEFINITION_DEPTH {
            let mut next = Circuit::new(1);
            next.append(gate, [Qubit::new(0)], [], None).unwrap();
            gate = next.to_gate(format!("level_{index}")).unwrap();
        }

        let mut top = Circuit::new(1);
        top.append(gate, [Qubit::new(0)], [], None).unwrap();

        let err = expand_definitions(&top).unwrap_err();
        assert!(
            matches!(err, CompilerError::InvalidInput(message) if message.contains("maximum recursion depth"))
        );
    }
}
