use std::collections::HashMap;

use indexmap::IndexSet;
use ndarray::prelude::*;
use num::complex::Complex;
use smallvec::{SmallVec, smallvec};

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::dag::CircuitDag;
use crate::circuit::gate::StandardGate;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::parameter::Parameter;
use crate::circuit::{Operation, Qubit};
use crate::compile::gate_transform::transform_rules::rule_registry::TransformRuleExecutor;
use crate::compile::gate_transform::transform_rules::single_qubit_rule::SingleQubitRule;
use crate::device::InstructionSet;

pub struct GateTransform {
    pub instruction_set: InstructionSet,
}

impl GateTransform {
    pub fn new(instruction_set: InstructionSet) -> Self {
        Self { instruction_set }
    }

    /// Set the instruction set for gate transformation.
    /// This allows for changing the target gate set dynamically.
    pub fn set_instruction_set(&mut self, instruction_set: InstructionSet) {
        self.instruction_set = instruction_set;
    }

    /// Execute the gate transform on the input circuit.
    /// Returns a new circuit expressed entirely in the instruction set's gates.
    /// Symbolic parameter preservation currently requires mutable parameter interning
    /// during block processing, so blocks are handled sequentially here.
    pub fn execute(&mut self, circuit: &Circuit) -> Circuit {
        let mut dag = CircuitDag::from_circuit(circuit).expect("Failed to create CircuitDag");

        let block_indices: Vec<_> = dag.blocks().map(|(idx, _)| idx).collect();
        for idx in block_indices {
            let operations = dag.data[idx].operations.clone();
            let decomposed =
                self.multi_qubit_decompose(&operations, &mut dag.parameters, &mut dag.symbols);
            let oneq_param_transformed =
                self.oneq_param_transform(&decomposed, &mut dag.parameters, &mut dag.symbols);
            let transformed = self.oneq_transform(&oneq_param_transformed, &dag.parameters);
            dag.data[idx].operations = transformed;
        }

        dag.to_circuit()
            .expect("Failed to convert CircuitDag to Circuit")
    }
}

impl GateTransform {
    /// Convert a stored circuit parameter into a rule-level `Parameter`.
    ///
    /// Fixed values become constant expressions, and indexed values are cloned from
    /// the circuit parameter pool so transform rules can preserve symbolic algebra.
    fn circuit_param_to_parameter(
        param: &CircuitParam,
        parameters: &IndexSet<Parameter>,
    ) -> Parameter {
        match param {
            CircuitParam::Fixed(val) => Parameter::from(*val),
            CircuitParam::Index(idx) => parameters[*idx as usize].clone(),
        }
    }

    /// Convert a rule-level `Parameter` back into compact circuit storage.
    ///
    /// Constant expressions collapse to `CircuitParam::Fixed`; symbolic expressions
    /// are interned into the circuit parameter pool and returned as `Index` values.
    fn parameter_to_circuit_param(
        param: Parameter,
        parameters: &mut IndexSet<Parameter>,
        symbols: &mut IndexSet<String>,
    ) -> CircuitParam {
        match ParameterValue::from(param.clone()) {
            ParameterValue::Fixed(value) => CircuitParam::Fixed(value),
            ParameterValue::Param(param) => {
                let (index, is_new) = parameters.insert_full(param.clone());
                if is_new {
                    for sym in param.get_symbols() {
                        symbols.insert(sym);
                    }
                }
                CircuitParam::Index(index as u32)
            }
        }
    }

    /// Try to resolve a parameter list into concrete numeric values.
    ///
    /// Returns `Some(Vec<f64>)` only when every parameter can be evaluated without
    /// external bindings; otherwise returns `None` so symbolic gates can be preserved.
    fn try_resolve_numeric_params(
        params: &SmallVec<[CircuitParam; 1]>,
        parameters: &IndexSet<Parameter>,
    ) -> Option<Vec<f64>> {
        params
            .iter()
            .map(|p| match p {
                CircuitParam::Fixed(val) => Some(*val),
                CircuitParam::Index(idx) => parameters[*idx as usize].evaluate(&None).ok(),
            })
            .collect()
    }

    /// Decompose a gate into the instruction set's target gates.
    /// Handles CCX, SWAP.
    fn gate_decompose(&self, gate: &StandardGate) -> Vec<(StandardGate, SmallVec<[usize; 2]>)> {
        let mut decomp_gates: Vec<(StandardGate, SmallVec<[usize; 2]>)> = Vec::new();
        match gate {
            StandardGate::CCX => {
                // Decomposition CCX Gate
                decomp_gates.push((StandardGate::H, smallvec![2]));
                decomp_gates.push((StandardGate::CX, smallvec![2, 1]));
                decomp_gates.push((StandardGate::TDG, smallvec![1]));
                decomp_gates.push((StandardGate::CX, smallvec![0, 1]));
                decomp_gates.push((StandardGate::T, smallvec![1]));
                decomp_gates.push((StandardGate::CX, smallvec![2, 1]));
                decomp_gates.push((StandardGate::TDG, smallvec![1]));
                decomp_gates.push((StandardGate::CX, smallvec![0, 1]));
                decomp_gates.push((StandardGate::T, smallvec![1]));
                decomp_gates.push((StandardGate::CX, smallvec![0, 2]));
                decomp_gates.push((StandardGate::TDG, smallvec![2]));
                decomp_gates.push((StandardGate::CX, smallvec![0, 2]));
                decomp_gates.push((StandardGate::T, smallvec![0]));
                decomp_gates.push((StandardGate::T, smallvec![2]));
                decomp_gates.push((StandardGate::H, smallvec![2]));
            }
            StandardGate::SWAP => {
                // Decompose SWAP Gate
                decomp_gates.push((StandardGate::CX, smallvec![0, 1]));
                decomp_gates.push((StandardGate::CX, smallvec![1, 0]));
                decomp_gates.push((StandardGate::CX, smallvec![0, 1]));
            }
            _ => panic!("Invalid single qubit rule name"),
        };

        decomp_gates
    }

    /// Decompose multi-qubit gates into the instruction set's target two-qubit gate.
    /// Also handles CCX, SWAP via build_gate decomposition.
    fn multi_qubit_decompose(
        &mut self,
        ops: &[Operation],
        parameters: &mut IndexSet<Parameter>,
        symbols: &mut IndexSet<String>,
    ) -> Vec<Operation> {
        let mut new_operations: Vec<Operation> = Vec::new();

        for op in ops {
            let instruction = &op.instruction;
            let qubits = &op.qubits;
            let params = &op.params;

            match instruction {
                // Pass through special gates unchanged
                Instruction::Directive(_) | Instruction::McGate(_) => {
                    new_operations.push(op.clone());
                }

                // UNITARY: 1-qubit ok (handle in oneq_transform), >1 qubit unsupported
                // Also supports UnitaryGate with internal circuit
                Instruction::UnitaryGate(ugate) => {
                    if let Some(circuit_ref) = ugate.circuit() {
                        let subcircuit = circuit_ref.circuit();
                        let mut sub_parameters = subcircuit.parameters().clone();
                        let mut sub_symbols = subcircuit.symbols().clone();
                        let transformed_subcircuit = self.multi_qubit_decompose(
                            subcircuit.operations(),
                            &mut sub_parameters,
                            &mut sub_symbols,
                        );
                        // Decompose the transformed subcircuit operations directly
                        // instead of wrapping them in a new UnitaryGate
                        new_operations.extend(transformed_subcircuit);
                    } else if ugate.num_qubits() == 1 {
                        new_operations.push(op.clone());
                    } else {
                        panic!(
                            "Unsupported: UNITARY gate acting on {} qubits. Only 1-qubit UNITARY is supported.",
                            ugate.num_qubits()
                        );
                    }
                }

                Instruction::Standard(sgate) => {
                    if sgate == &StandardGate::CCX || sgate == &StandardGate::SWAP {
                        let decomposed = self.gate_decompose(sgate);

                        // Iterate over the decomposed gate instructions
                        for (gate_inst, qubit_inst) in &decomposed {
                            // Recursively handle the decomposed gate
                            let decomp_qubits = qubit_inst
                                .iter()
                                .map(|&q| qubits[q])
                                .collect::<SmallVec<[Qubit; 3]>>();
                            if qubit_inst.len() == 2 {
                                let transformed_ops = self.append_two_qubit_transformed(
                                    gate_inst,
                                    &smallvec![],
                                    &decomp_qubits,
                                    parameters,
                                    symbols,
                                );
                                if let Some(ops) = transformed_ops {
                                    new_operations.extend(ops);
                                } else {
                                    new_operations.push(Operation {
                                        instruction: Instruction::Standard(*gate_inst),
                                        qubits: decomp_qubits.clone(),
                                        params: smallvec![],
                                        label: None,
                                    });
                                }
                            } else {
                                new_operations.push(Operation {
                                    instruction: Instruction::Standard(*gate_inst),
                                    qubits: decomp_qubits.clone(),
                                    params: smallvec![],
                                    label: None,
                                });
                            }
                        }
                        continue;
                    }

                    if sgate.num_qubits() == 1 {
                        // Single-qubit gates: pass through (handled in oneq_transform)
                        // Convert CircuitParam to ParameterValue for append
                        new_operations.push(op.clone());
                    } else if sgate.num_qubits() == 2 {
                        let param_values: SmallVec<[Parameter; 3]> = params
                            .iter()
                            .map(|p| Self::circuit_param_to_parameter(p, parameters))
                            .collect::<SmallVec<[Parameter; 3]>>();
                        // Two-qubit gates: transform to target gate
                        let transformed_ops = self.append_two_qubit_transformed(
                            sgate,
                            &param_values,
                            &qubits,
                            parameters,
                            symbols,
                        );
                        if let Some(ops) = transformed_ops {
                            new_operations.extend(ops);
                        } else {
                            new_operations.push(Operation {
                                instruction: Instruction::Standard(*sgate),
                                qubits: qubits.clone(),
                                params: params.clone(),
                                label: None,
                            });
                        }
                    }
                }

                Instruction::CircuitGate(cgate) => {
                    let frozen_circuit = cgate.circuit();
                    let subcircuit = frozen_circuit.circuit();
                    let mut sub_parameters = subcircuit.parameters().clone();
                    let mut sub_symbols = subcircuit.symbols().clone();
                    let transformed_subcircuit = self.multi_qubit_decompose(
                        subcircuit.operations(),
                        &mut sub_parameters,
                        &mut sub_symbols,
                    );
                    for sub_cir_op in &transformed_subcircuit {
                        let mapped_qubits = sub_cir_op
                            .qubits
                            .iter()
                            .map(|q| qubits[q.id() as usize])
                            .collect::<SmallVec<[Qubit; 3]>>();
                        new_operations.push(Operation {
                            instruction: sub_cir_op.instruction.clone(),
                            qubits: mapped_qubits,
                            params: sub_cir_op.params.clone(),
                            label: None,
                        });
                    }
                }

                // Anything else with >2 qubits that's not CCX/SWAP
                _ => {
                    panic!(
                        "Unsupported gate type {:?} with {} qubits",
                        instruction,
                        qubits.len()
                    );
                }
            }
        }

        new_operations
    }

    /// Transform a two-qubit gate using the instruction set's rule chain
    /// and append the resulting gates to the circuit.
    fn append_two_qubit_transformed(
        &mut self,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
        qubits: &[Qubit],
        parameters: &mut IndexSet<Parameter>,
        symbols: &mut IndexSet<String>,
    ) -> Option<Vec<Operation>> {
        let mut new_dg_operations: Vec<Operation> = Vec::new();
        let steps = self
            .instruction_set
            .select_transform_rule(gate.clone())
            .unwrap_or_else(|e| panic!("Failed to select transform rule: {}", e));

        if steps.is_empty() {
            // Gate is already the target gate
            return None;
        }

        // Apply the transformation chain
        // We need to track current gates, their parameters, and qubits
        let mut current_gates: Vec<(StandardGate, SmallVec<[Parameter; 3]>, Vec<i32>)> = Vec::new();
        current_gates.push((gate.clone(), params.clone(), Vec::from([0, 1])));

        for step in &steps {
            let mut next_gates: Vec<(StandardGate, SmallVec<[Parameter; 3]>, Vec<i32>)> =
                Vec::new();
            for (g, gate_params, qs) in &current_gates {
                if *g == step.source_gate {
                    // Apply the rule
                    let decomposed = TransformRuleExecutor::apply(step.rule, g, gate_params);

                    // Map decomposed qubits to actual circuit qubits
                    for ((decomp_gate, decomp_params), decomp_qubits) in
                        decomposed.gates.iter().zip(decomposed.qubits.iter())
                    {
                        // Map decomp_qubits to actual circuit qubits
                        let mapped_qubits: Vec<i32> =
                            decomp_qubits.iter().map(|&q| qs[q as usize]).collect();
                        next_gates.push((
                            decomp_gate.clone(),
                            decomp_params.clone(),
                            mapped_qubits,
                        ));
                    }
                } else {
                    // Gate doesn't match this step, keep as-is
                    next_gates.push((g.clone(), gate_params.clone(), qs.clone()));
                }
            }

            current_gates = next_gates;
        }

        // Append all resulting gates to the circuit

        for (g, gate_params, qs) in current_gates {
            let mut circuit_params: SmallVec<[CircuitParam; 1]> = SmallVec::new();
            for param in gate_params {
                circuit_params.push(Self::parameter_to_circuit_param(param, parameters, symbols));
            }
            let mut mapped_qs: SmallVec<[Qubit; 3]> = SmallVec::new();
            for q in qs {
                mapped_qs.push(qubits[q as usize]);
            }
            new_dg_operations.push(Operation {
                instruction: Instruction::Standard(g),
                qubits: mapped_qs,
                params: circuit_params,
                label: None,
            });
        }
        Some(new_dg_operations)
    }

    /// Transform symbolic single-qubit parameterized gates into the instruction set's
    /// target single-qubit parameterized gates. Numeric helper gates are left for
    /// `oneq_transform`, which already handles matrix-based synthesis.
    fn oneq_param_transform(
        &mut self,
        operations: &Vec<Operation>,
        parameters: &mut IndexSet<Parameter>,
        symbols: &mut IndexSet<String>,
    ) -> Vec<Operation> {
        let mut new_operations = Vec::new();

        for op in operations {
            match &op.instruction {
                Instruction::Standard(sgate)
                    if sgate.num_qubits() == 1
                        && !op.params.is_empty()
                        && Self::try_resolve_numeric_params(&op.params, parameters).is_none() =>
                {
                    let param_values: SmallVec<[Parameter; 3]> = op
                        .params
                        .iter()
                        .map(|p| Self::circuit_param_to_parameter(p, parameters))
                        .collect();

                    if let Some(ops) = self.append_single_qubit_param_transformed(
                        sgate,
                        &param_values,
                        &op.qubits,
                        parameters,
                        symbols,
                    ) {
                        new_operations.extend(ops);
                    } else {
                        new_operations.push(op.clone());
                    }
                }
                _ => new_operations.push(op.clone()),
            }
        }

        new_operations
    }

    fn append_single_qubit_param_transformed(
        &mut self,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
        qubits: &[Qubit],
        parameters: &mut IndexSet<Parameter>,
        symbols: &mut IndexSet<String>,
    ) -> Option<Vec<Operation>> {
        let transformed = self.transform_symbolic_single_qubit_gate_recursive(gate, params);
        if transformed.len() == 1 && transformed[0].0 == *gate && transformed[0].1 == *params {
            return None;
        }

        let mut operations = Vec::new();
        for (g, gate_params) in transformed {
            let circuit_params = gate_params
                .into_iter()
                .map(|param| Self::parameter_to_circuit_param(param, parameters, symbols))
                .collect::<SmallVec<[CircuitParam; 1]>>();
            operations.push(Operation {
                instruction: Instruction::Standard(g),
                qubits: qubits.iter().copied().collect(),
                params: circuit_params,
                label: None,
            });
        }
        Some(operations)
    }

    fn transform_symbolic_single_qubit_gate_recursive(
        &mut self,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
    ) -> Vec<(StandardGate, SmallVec<[Parameter; 3]>)> {
        if gate.num_qubits() != 1
            || params.is_empty()
            || !Self::has_symbolic_rule_params(params)
            || self.instruction_set.single_qubit_gates.contains(gate)
        {
            return vec![(gate.clone(), params.clone())];
        }

        let steps = match self
            .instruction_set
            .select_single_qubit_param_transform_rule(gate.clone())
        {
            Ok(steps) if !steps.is_empty() => steps,
            _ => return vec![(gate.clone(), params.clone())],
        };

        let mut current_gates = vec![(gate.clone(), params.clone())];
        for step in &steps {
            let mut next_gates = Vec::new();
            for (g, gate_params) in &current_gates {
                if *g == step.source_gate {
                    let decomposed = TransformRuleExecutor::apply(step.rule, g, gate_params);
                    next_gates.extend(decomposed.gates.iter().cloned());
                } else {
                    next_gates.push((g.clone(), gate_params.clone()));
                }
            }
            current_gates = next_gates;
        }

        let mut final_gates = Vec::new();
        for (g, gate_params) in current_gates {
            if g.num_qubits() == 1
                && !gate_params.is_empty()
                && Self::has_symbolic_rule_params(&gate_params)
                && !self.instruction_set.single_qubit_gates.contains(&g)
            {
                final_gates
                    .extend(self.transform_symbolic_single_qubit_gate_recursive(&g, &gate_params));
            } else {
                final_gates.push((g, gate_params));
            }
        }

        final_gates
    }

    fn has_symbolic_rule_params(params: &SmallVec<[Parameter; 3]>) -> bool {
        params.iter().any(|param| param.evaluate(&None).is_err())
    }

    /// Transform single-qubit gates using front-layer accumulation.
    /// Consecutive single-qubit gates on the same qubit are multiplied together,
    /// then decomposed using the instruction set's single-qubit rule.
    fn oneq_transform(
        &self,
        operations: &Vec<Operation>,
        parameters: &IndexSet<Parameter>,
    ) -> Vec<Operation> {
        let mut new_operations = Vec::new();

        // Front layer: accumulated unitary per qubit
        let mut front_layer: HashMap<u32, Array2<Complex<f64>>> = HashMap::new();
        let single_qubit_rule = SingleQubitRule::new(
            self.instruction_set
                .get_single_qubit_decomposition_rule()
                .to_string(),
        );

        for op in operations {
            let instruction = &op.instruction;
            let qubits = &op.qubits;
            let params = &op.params;

            match instruction {
                Instruction::Standard(sgate) => {
                    if sgate.num_qubits() == 1 {
                        let Some(param_values) = Self::try_resolve_numeric_params(params, parameters)
                        else {
                            let q = qubits[0];
                            let front_layer_operator =
                                self.flush_front_layer(&mut front_layer, q.id(), &single_qubit_rule);
                            if !front_layer_operator.is_empty() {
                                new_operations.extend(front_layer_operator);
                            }
                            new_operations.push(op.clone());
                            continue;
                        };

                        // Single-qubit gate: accumulate into front layer
                        let q = qubits[0];
                        let gate_matrix: Array2<Complex<f64>> =
                            sgate.matrix(&param_values).into_owned();

                        if let Some(existing) = front_layer.get(&q.id()) {
                            front_layer.insert(q.id(), gate_matrix.dot(existing));
                        } else {
                            front_layer.insert(q.id(), gate_matrix);
                        }
                    } else {
                        // Multi-qubit gate: flush affected qubits first
                        for q in qubits {
                            let front_layer_operator = self.flush_front_layer(
                                &mut front_layer,
                                q.id(),
                                &single_qubit_rule,
                            );
                            if !front_layer_operator.is_empty() {
                                new_operations.extend(front_layer_operator);
                            }
                        }
                        new_operations.push(op.clone());
                    }
                }
                _ => {
                    // Other instruction types: just append as-is
                    for q in qubits {
                        new_operations.extend(self.flush_front_layer(
                            &mut front_layer,
                            q.id(),
                            &single_qubit_rule,
                        ));
                    }
                    new_operations.push(op.clone());
                }
            }
        }

        // Flush remaining front layer entries
        let remaining_qubits: Vec<u32> = front_layer.keys().cloned().collect();
        for q in remaining_qubits {
            new_operations.extend(self.flush_front_layer(&mut front_layer, q, &single_qubit_rule));
        }

        new_operations
    }

    /// Flush the front layer for a specific qubit by decomposing its accumulated
    /// unitary and appending the resulting gates to the circuit.
    fn flush_front_layer(
        &self,
        front_layer: &mut HashMap<u32, Array2<Complex<f64>>>,
        qubit: u32,
        rule: &SingleQubitRule,
    ) -> Vec<Operation> {
        let mut new_operations = Vec::new();
        if let Some(mat) = front_layer.remove(&qubit) {
            // Check if it's close to identity
            let identity: Array2<Complex<f64>> = Array2::eye(2);
            let diff = &mat - &identity;
            let norm_sq: f64 = diff.iter().map(|c| c.norm_sqr()).sum();

            if norm_sq < 1e-12 {
                // Matrix is essentially identity, no gates needed
                return new_operations;
            }

            // Decompose the accumulated unitary
            let decomposed_gates = rule.execute(&mat);

            // Append decomposed gates (in temporal order, index 0 first)
            for (g, params) in decomposed_gates {
                // For now, we'll skip this part as we're not using matrix accumulation
                // In a full implementation, we would convert g to an Instruction and append it
                let mut mapped_params: SmallVec<[CircuitParam; 1]> = smallvec![];
                for param in params {
                    mapped_params.push(CircuitParam::Fixed(param));
                }
                let mut mapped_qubits: SmallVec<[Qubit; 3]> = smallvec![];
                mapped_qubits.push(Qubit::new(qubit));

                new_operations.push(Operation {
                    instruction: Instruction::Standard(g),
                    qubits: mapped_qubits,
                    params: mapped_params,
                    label: None,
                });
            }
        }

        new_operations
    }
}

#[cfg(test)]
#[path = "gate_transfer_test.rs"]
mod tests;
