use std::collections::HashMap;

use indexmap::IndexSet;
use ndarray::prelude::*;
use num::complex::Complex;
use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::dag::CircuitDag;
use crate::circuit::gate::StandardGate;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::param::CircuitParam;
use crate::circuit::parameter::Parameter;
use crate::circuit::{Operation, Qubit};
use crate::compile::gate_transform::transform_rules::double_qubit_rule::{
    DecomposedTwoQubitGate, DoubleQubitRule,
};
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
    /// Uses CircuitDag and parallel processing for improved performance.
    pub fn execute(&mut self, circuit: &Circuit) -> Circuit {
        let mut dag = CircuitDag::from_circuit(circuit).expect("Failed to create CircuitDag");

        let block_indices: Vec<_> = dag.blocks().map(|(idx, _)| idx).collect();
        let instruction_set = self.instruction_set.clone();
        let parameters = circuit.parameters().clone();
        let processed_results: Vec<_> = block_indices
            .par_iter()
            .map(|&idx| {
                let operations = dag.data[idx].operations.clone();

                let mut gt = GateTransform::new(instruction_set.clone());
                let decomposed = gt.multi_qubit_decompose(&operations, &parameters);
                let transformed = gt.oneq_transform(&decomposed, &parameters);

                (idx, transformed)
            })
            .collect();

        for (idx, new_operations) in processed_results {
            dag.data[idx].operations = new_operations;
        }

        dag.to_circuit()
            .expect("Failed to convert CircuitDag to Circuit")
    }
}

impl GateTransform {
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
        parameters: &IndexSet<Parameter>,
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
                        let transformed_subcircuit = self.multi_qubit_decompose(
                            subcircuit.operations(),
                            subcircuit.parameters(),
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
                        let param_values: SmallVec<[f64; 3]> = params
                            .iter()
                            .map(|p| match p {
                                CircuitParam::Fixed(val) => *val,
                                CircuitParam::Index(idx) => {
                                    let param = parameters[*idx as usize].clone();
                                    if let Ok(val) = param.evaluate(&None) {
                                        val
                                    } else {
                                        // If evaluation fails, use 0.0 as default
                                        0.0
                                    }
                                }
                            })
                            .collect::<SmallVec<[f64; 3]>>();
                        // Two-qubit gates: transform to target gate
                        let transformed_ops =
                            self.append_two_qubit_transformed(sgate, &param_values, &qubits);
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
                    let transformed_subcircuit = self
                        .multi_qubit_decompose(subcircuit.operations(), subcircuit.parameters());
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
        params: &SmallVec<[f64; 3]>,
        qubits: &[Qubit],
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
        let mut current_gates: Vec<(StandardGate, SmallVec<[f64; 3]>, Vec<i32>)> = Vec::new();
        current_gates.push((gate.clone(), params.clone(), Vec::from([0, 1])));

        for step in &steps {
            let mut next_gates: Vec<(StandardGate, SmallVec<[f64; 3]>, Vec<i32>)> = Vec::new();
            for (g, gate_params, qs) in &current_gates {
                if *g == step.source_gate {
                    // Apply the rule
                    let decomposed = Self::apply_rule(&step.rule_name, g, gate_params);

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
                circuit_params.push(CircuitParam::Fixed(param));
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

    /// Apply a named double-qubit rule to a gate.
    fn apply_rule(
        rule_name: &str,
        gate: &StandardGate,
        params: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        // Convert ParameterValue to SmallVec<[f64; 3]>
        match rule_name {
            // Between categories
            "cx2rzz_rule" => DoubleQubitRule::cx2rzz_rule(gate, params),
            "rzz2cx_rule" => DoubleQubitRule::rzz2cx_rule(gate, params),

            // CX category
            "cx2cy_rule" => DoubleQubitRule::cx2cy_rule(gate, params),
            "cy2cx_rule" => DoubleQubitRule::cy2cx_rule(gate, params),
            "cx2cz_rule" => DoubleQubitRule::cx2cz_rule(gate, params),
            "cz2cx_rule" => DoubleQubitRule::cz2cx_rule(gate, params),

            // RZZ category
            "rzz2rxx_rule" => DoubleQubitRule::rzz2rxx_rule(gate, params),
            "rxx2rzz_rule" => DoubleQubitRule::rxx2rzz_rule(gate, params),
            "rzz2ryy_rule" => DoubleQubitRule::rzz2ryy_rule(gate, params),
            "ryy2rzz_rule" => DoubleQubitRule::ryy2rzz_rule(gate, params),
            "rzz2rzx_rule" => DoubleQubitRule::rzz2rzx_rule(gate, params),
            "rzx2rzz_rule" => DoubleQubitRule::rzx2rzz_rule(gate, params),
            "rzz2crz_rule" => DoubleQubitRule::rzz2crz_rule(gate, params),
            "crz2rzz_rule" => DoubleQubitRule::crz2rzz_rule(gate, params),
            "rzz2crx_rule" => DoubleQubitRule::rzz2crx_rule(gate, params),
            "crx2rzz_rule" => DoubleQubitRule::crx2rzz_rule(gate, params),
            "rzz2cry_rule" => DoubleQubitRule::rzz2cry_rule(gate, params),
            "cry2rzz_rule" => DoubleQubitRule::cry2rzz_rule(gate, params),

            // Fsim category
            "cx2fsim_rule" => DoubleQubitRule::cx2fsim_rule(gate, params),
            "fsim2cx_rule" => DoubleQubitRule::fsim2cx_rule(gate, params),
            "fsim2rzz_rule" => DoubleQubitRule::fsim2rzz_rule(gate, params),
            "rzz2fsim_rule" => DoubleQubitRule::rzz2fsim_rule(gate, params),

            _ => panic!("Unknown double-qubit rule: {}", rule_name),
        }
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
            let param_values: Vec<_> = params
                .iter()
                .map(|p| match p {
                    CircuitParam::Fixed(val) => *val,
                    CircuitParam::Index(idx) => {
                        let param = parameters[*idx as usize].clone();
                        if let Ok(val) = param.evaluate(&None) {
                            val
                        } else {
                            // If evaluation fails, use 0.0 as default
                            0.0
                        }
                    }
                })
                .collect();

            match instruction {
                Instruction::Standard(sgate) => {
                    if sgate.num_qubits() == 1 {
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
mod tests {
    use super::*;
    use crate::circuit::Qubit;
    use crate::circuit::dag::{BasicBlock, FlowEdge, Terminator};
    use crate::circuit::gate::MCGate;
    use crate::circuit::gate::control_flow::{ConditionView, IfElseGate, WhileLoopGate};
    use crate::circuit::gate::{CircuitGate, FrozenCircuit, UnitaryGate};
    use crate::circuit::param::ParameterValue;
    use num::complex::Complex;
    use num::complex::ComplexFloat;
    use std::sync::Arc;

    #[test]
    fn test_gate_transform_basic() {
        // Create a simple circuit with CZ gate
        let mut circuit = Circuit::new(2);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        circuit.h(q0).unwrap();
        circuit.cz(q0, q1).unwrap();

        // Create instruction set targeting CX
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::CX],
            None,
        );

        let mut gt = GateTransform::new(iset);
        let result = gt.execute(&circuit);

        // Verify the result contains CX not CZ
        let ops = result.operations();
        let has_cx = ops.iter().any(|op| {
            if let Instruction::Standard(sgate) = &op.instruction {
                *sgate == StandardGate::CX
            } else {
                false
            }
        });
        let has_cz = ops.iter().any(|op| {
            if let Instruction::Standard(sgate) = &op.instruction {
                *sgate == StandardGate::CZ
            } else {
                false
            }
        });

        assert!(has_cx, "Result should contain CX gate");
        assert!(!has_cz, "Result should not contain CZ gate");
    }

    fn generate_full_circuit() -> Circuit {
        // Create a 3-qubit circuit
        let mut circuit = Circuit::new(3);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        let q2 = Qubit::new(2);

        // --- Single Qubit Gates --- //
        // Pauli gates
        circuit.x(q0).unwrap();
        circuit.y(q0).unwrap();
        circuit.z(q0).unwrap();
        circuit.i(q0).unwrap();

        // Clifford gates
        circuit.h(q0).unwrap();
        circuit.s(q1).unwrap();
        circuit.sdg(q2).unwrap();
        circuit.t(q0).unwrap();
        circuit.tdg(q1).unwrap();

        // // Sqrt gates
        circuit.x2p(q2).unwrap();
        circuit.x2m(q0).unwrap();
        circuit.y2p(q1).unwrap();
        circuit.y2m(q2).unwrap();

        // --- Directives --- //
        circuit.measure(q0).unwrap();
        circuit.measure(q1).unwrap();
        circuit.measure(q2).unwrap();

        // // Parametric gates with parameters
        circuit.rx(q0, 0.5).unwrap();
        circuit.ry(q1, 0.5).unwrap();
        circuit.rz(q2, 0.5).unwrap();
        circuit.phase(q0, 0.5).unwrap();
        circuit.xy(q1, 0.5).unwrap();
        circuit.xy2p(q2, 0.5).unwrap();
        circuit.xy2m(q0, 0.5).unwrap();
        circuit.rxy(q1, 0.5, 0.5).unwrap(); // Two parameters
        circuit.u(q2, 0.5, 0.5, 0.5).unwrap(); // Three parameters

        // --- Two Qubit Gates --- //
        // Non-parametric
        circuit.cx(q0, q1).unwrap();
        circuit.cy(q1, q2).unwrap();
        circuit.cz(q0, q2).unwrap();
        circuit.swap(q0, q1).unwrap();

        // --- Directives --- //
        circuit.barrier(vec![q0, q1, q2]).unwrap();

        // Parametric
        circuit.rxx(q0, q1, 0.5).unwrap();
        circuit.ryy(q1, q2, 0.5).unwrap();
        circuit.rzz(q0, q2, 0.5).unwrap();
        circuit.rzx(q0, q1, 0.5).unwrap();
        circuit.crx(q1, q2, 0.5).unwrap();
        circuit.cry(q0, q1, 0.5).unwrap();
        circuit.crz(q1, q2, 0.5).unwrap();
        // circuit.fsim(q0, q2, 0.5, 0.5).unwrap(); // Two parameters

        // --- Three Qubit Gates --- //
        circuit.ccx(q0, q1, q2).unwrap();
        circuit.ccx(q0, q2, q1).unwrap();

        // --- Directives --- //
        circuit.reset(q0).unwrap();
        circuit.reset(q1).unwrap();
        circuit.reset(q2).unwrap();

        circuit
    }

    /// Check if all gates in the circuit are in the instruction set or are directives
    fn check_all_gates_in_instruction_set(
        circuit: &Circuit,
        instruction_set: &InstructionSet,
    ) -> bool {
        for op in circuit.operations() {
            match &op.instruction {
                Instruction::Standard(sgate) => {
                    // For single-qubit gates, check if they can be decomposed
                    // For multi-qubit gates, check if they can be transformed
                    if sgate.num_qubits() == 1 {
                        // Single-qubit gates should be decomposable
                        // We'll assume the instruction set can handle them
                        if !instruction_set.single_qubit_gates.contains(sgate) {
                            return false;
                        }
                    } else if sgate.num_qubits() == 2 {
                        // Two-qubit gates should be transformable
                        if !instruction_set.double_qubit_gate.contains(sgate) {
                            return false;
                        }
                    } else {
                        // CCX is a special case
                        return false;
                    }
                }
                Instruction::UnitaryGate(_) => {
                    // Unitary gates are not allowed
                    return false;
                }
                _ => {
                    // Other instruction types are allowed
                }
            }
        }
        true
    }

    #[test]
    fn test_gate_transform_identity_elimination() {
        // Create a circuit where single-qubit gates cancel out
        let mut circuit = Circuit::new(1);
        let q0 = Qubit::new(0);
        circuit.h(q0).unwrap();
        circuit.h(q0).unwrap();

        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::CX],
            None,
        );

        let mut gt = GateTransform::new(iset);
        let result = gt.execute(&circuit);

        // H·H = I, so no gates should remain
        // Note: Current implementation doesn't perform identity elimination
        // This test is kept for future implementation
        assert_eq!(
            result.operations().len(),
            0,
            "H·H should cancel to identity"
        );
    }

    fn complex_inner_product(vec1: &[Complex<f64>], vec2: &[Complex<f64>]) -> Complex<f64> {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    fn is_matrix_differ_by_phase(
        matrix1: &Array2<Complex<f64>>,
        matrix2: &Array2<Complex<f64>>,
    ) -> bool {
        let vec1: Vec<Complex<f64>> = matrix1.iter().copied().collect();
        let vec2: Vec<Complex<f64>> = matrix2.iter().copied().collect();
        let inner: Complex<f64> = complex_inner_product(&vec1, &vec2);
        let inner_abs: f64 = inner.abs();
        let vec1_norm: f64 = complex_inner_product(&vec1, &vec1).re.sqrt();
        let vec2_norm: f64 = complex_inner_product(&vec2, &vec2).re.sqrt();

        let cos_vec = inner_abs / (vec1_norm * vec2_norm);
        (cos_vec - 1.0).abs() < 1e-10
    }

    fn gate_transfer_circuit_test(iset: &InstructionSet, circuit: &Circuit) {
        // let circuit = generate_full_circuit();
        let mut gt = GateTransform::new(iset.clone());
        let result = gt.execute(&circuit);
        assert!(
            check_all_gates_in_instruction_set(&result, &mut iset.clone()),
            "Gate transfer contains other Standard Gate not in iset."
        );

        let result_matrix = result.to_matrix(None);
        let circuit_matrix = circuit.to_matrix(None);

        if !is_matrix_differ_by_phase(&result_matrix, &circuit_matrix) {
            eprintln!("Assertion failed! Result circuit operations:");
            for (i, op) in result.operations().iter().enumerate() {
                eprintln!(
                    "  [{}] {:?} on qubits {:?}, with parameter {:?}",
                    i, op.instruction, op.qubits, op.params
                );
            }
            eprintln!("Original circuit operations:");
            for (i, op) in circuit.operations().iter().enumerate() {
                eprintln!("  [{}] {:?} on qubits {:?}", i, op.instruction, op.qubits);
            }
        }

        assert!(
            is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
            "Gate transfer should not change circuit size {}",
            result.operations().len()
        );
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cx() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX],
            vec![StandardGate::CX],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cz() {
        let iset = InstructionSet::new(
            vec![
                StandardGate::X2M,
                StandardGate::X2P,
                StandardGate::Y2P,
                StandardGate::Y2M,
                StandardGate::RZ,
            ],
            vec![StandardGate::CZ, StandardGate::CX],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cy() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::CY],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_dynamic_iset() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::CY],
            None,
        );
        let new_iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::CX],
            None,
        );
        let mut gt = GateTransform::new(iset.clone());
        gt.set_instruction_set(new_iset.clone());

        let circuit = generate_full_circuit();
        let result = gt.execute(&circuit);
        assert!(
            check_all_gates_in_instruction_set(&result, &mut new_iset.clone()),
            "Gate transfer contains other Standard Gate not in iset."
        );

        let result_matrix = result.to_matrix(None);
        let circuit_matrix = circuit.to_matrix(None);

        if !is_matrix_differ_by_phase(&result_matrix, &circuit_matrix) {
            eprintln!("Assertion failed! Result circuit operations:");
            for (i, op) in result.operations().iter().enumerate() {
                eprintln!(
                    "  [{}] {:?} on qubits {:?}, with parameter {:?}",
                    i, op.instruction, op.qubits, op.params
                );
            }
            eprintln!("Original circuit operations:");
            for (i, op) in circuit.operations().iter().enumerate() {
                eprintln!("  [{}] {:?} on qubits {:?}", i, op.instruction, op.qubits);
            }
        }

        assert!(
            is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
            "Gate transfer should not change circuit size {}",
            result.operations().len()
        );
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_rxx() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::RXX],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_ryy() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::RYY],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_rzz() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::RZZ],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_rzx() {
        let iset = InstructionSet::new(
            vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
            vec![StandardGate::RZX],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cx_hrz() {
        let iset = InstructionSet::new(
            vec![StandardGate::H, StandardGate::RZ],
            vec![StandardGate::CX],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_fsim() {
        let iset = InstructionSet::new(
            vec![
                StandardGate::RX,
                StandardGate::RZ,
                StandardGate::RY,
                StandardGate::H,
            ],
            vec![StandardGate::FSIM],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cz_u() {
        let iset = InstructionSet::new(vec![StandardGate::U], vec![StandardGate::CZ], None);
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_cx_rxx() {
        let iset = InstructionSet::new(
            vec![
                StandardGate::RX,
                StandardGate::RZ,
                StandardGate::RY,
                StandardGate::H,
            ],
            vec![StandardGate::CZ, StandardGate::RZZ],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_full_circuit());
    }

    fn generate_circuit_with_special_operation() -> Circuit {
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        let q2 = Qubit::new(2);

        let mut sub_circuit = Circuit::new(2);
        sub_circuit.cx(q0, q1).unwrap();

        let frozen_sub_circuit = FrozenCircuit::new(sub_circuit);
        let mut ugate = UnitaryGate::new("TestUnitary", 2);
        ugate = ugate
            .with_matrix(StandardGate::CX.matrix(&[]).into_owned())
            .unwrap();
        ugate = ugate.with_circuit(Arc::new(frozen_sub_circuit));

        let cgate = CircuitGate::new(
            "TestCircuit",
            FrozenCircuit::new({ generate_full_circuit() }),
        )
        .unwrap();

        let mut circuit = generate_full_circuit();
        circuit
            .append(
                Instruction::UnitaryGate(Box::new(ugate)),
                vec![q0, q1],
                std::iter::empty(),
                None,
            )
            .unwrap();
        circuit
            .append(
                Instruction::CircuitGate(Box::new(cgate)),
                vec![q0, q2, q1],
                std::iter::empty(),
                None,
            )
            .unwrap();
        let mc_gate = MCGate::new(2, StandardGate::X);
        circuit
            .append(
                Instruction::McGate(Box::new(mc_gate)),
                vec![q0, q1, q2],
                std::iter::empty(),
                None,
            )
            .unwrap();

        circuit
    }

    fn generate_circuit_with_control_flow() -> Circuit {
        let mut circuit = generate_full_circuit();
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);
        let q2 = Qubit::new(2);

        circuit.h(q0).unwrap();
        circuit.h(q1).unwrap();

        circuit.measure(q0).unwrap();
        circuit.measure(q1).unwrap();

        let condition1 = ConditionView::new(q0, 1);

        let if1_true_body = vec![
            Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![q1],
                params: smallvec![],
                label: None,
            },
            Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![q1, q2],
                params: smallvec![],
                label: None,
            },
        ];

        let if1_false_body = vec![
            Operation {
                instruction: Instruction::Standard(StandardGate::X),
                qubits: smallvec![q1],
                params: smallvec![],
                label: None,
            },
            Operation {
                instruction: Instruction::Standard(StandardGate::Z),
                qubits: smallvec![q2],
                params: smallvec![],
                label: None,
            },
        ];

        circuit
            .if_else(condition1, if1_true_body, Some(if1_false_body))
            .unwrap();

        let condition2 = ConditionView::new(q0, 1);
        let while_body = vec![
            Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![q1],
                params: smallvec![],
                label: None,
            },
            Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![q2, q1],
                params: smallvec![],
                label: None,
            },
        ];

        circuit.while_loop(condition2, while_body).unwrap();

        circuit
    }

    #[test]
    fn test_gate_transfer_full_circuit_with_unitary_and_circuit_gate() {
        let iset = InstructionSet::new(
            vec![
                StandardGate::RX,
                StandardGate::RZ,
                StandardGate::RY,
                StandardGate::H,
            ],
            vec![StandardGate::CX, StandardGate::RZZ],
            None,
        );
        gate_transfer_circuit_test(&iset, &generate_circuit_with_special_operation());
    }

    #[test]
    fn test_gate_transform_with_control_flow() {
        let iset = InstructionSet::new(
            vec![
                StandardGate::RX,
                StandardGate::RZ,
                StandardGate::RY,
                StandardGate::H,
            ],
            vec![StandardGate::CX, StandardGate::RZZ],
            None,
        );

        let circuit = generate_circuit_with_control_flow();
        let dag = CircuitDag::from_circuit(&circuit).expect("Failed to create CircuitDag");
        assert!(
            dag.num_blocks() > 1,
            "Control flow circuit should have multiple blocks"
        );

        let mut gt = GateTransform::new(iset.clone());
        let result = gt.execute(&circuit);

        let result_matrix = result.to_matrix(None);
        let circuit_matrix = circuit.to_matrix(None);
        assert!(
            is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
            "Gate transfer should not change circuit size {}",
            result.operations().len()
        );

        let result_dag = CircuitDag::from_circuit(&result).expect("Failed to create CircuitDag");
        assert_eq!(
            dag.num_blocks(),
            result_dag.num_blocks(),
            "Gate transfer should not change circuit block number"
        );

        for (_idx, block) in result_dag.blocks() {
            let mut old_subc = Circuit::new(3);
            let mut new_subc = Circuit::new(3);
            let operations = block.operations.clone();
            for op in &operations {
                if let Instruction::Standard(gate) = &op.instruction {
                    if gate.num_qubits() == 1 {
                        assert!(
                            iset.single_qubit_gates.contains(gate),
                            "Single qubit gate {} do not in instruction set",
                            gate
                        );
                    } else if gate.num_qubits() == 2 {
                        assert!(
                            iset.double_qubit_gate.contains(gate),
                            "Double qubit gate {} do not in instruction set",
                            gate
                        );
                    }
                }
                let params = op
                    .params
                    .iter()
                    .map(|x| match x {
                        CircuitParam::Fixed(f) => ParameterValue::Fixed(*f),
                        _ => ParameterValue::Fixed(0.0),
                    })
                    .collect::<Vec<_>>();
                new_subc
                    .append(op.instruction.clone(), op.qubits.clone(), params, None)
                    .unwrap();
            }

            for op in dag.data[_idx].operations.iter() {
                let params = op
                    .params
                    .iter()
                    .map(|x| match x {
                        CircuitParam::Fixed(f) => ParameterValue::Fixed(*f),
                        _ => ParameterValue::Fixed(0.0),
                    })
                    .collect::<Vec<_>>();
                old_subc
                    .append(op.instruction.clone(), op.qubits.clone(), params, None)
                    .unwrap();
            }

            let result_matrix = old_subc.to_matrix(None);
            let circuit_matrix = new_subc.to_matrix(None);
            assert!(
                is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
                "Gate transfer should not change circuit size {}",
                result.operations().len()
            );
        }
    }
}
