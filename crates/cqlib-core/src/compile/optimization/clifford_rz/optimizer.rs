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

use super::canonical::{
    CanonicalGate, CanonicalOp, approx_angle_eq, approx_zero, canonical_sequence_eq,
    exact_rz_rewrite, normalize_4pi, try_canonicalize,
};
use super::dag::SegmentDag;
use super::phase_poly::PhasePolynomial;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, CircuitParam, Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::compile::mapping::{build_if_else_operation, build_while_loop_operation};
use crate::compile::structured::{PreparedProgramItem, PreparedSegment, preprocess_program};
use indexmap::IndexSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordRzLevel {
    Light,
    Heavy,
}

#[derive(Debug, Clone)]
pub struct CliffordRzConfig {
    pub level: CliffordRzLevel,
    pub numeric_tol: f64,
}

impl Default for CliffordRzConfig {
    fn default() -> Self {
        Self {
            level: CliffordRzLevel::Light,
            numeric_tol: 1e-10,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CliffordRzPass {
    config: CliffordRzConfig,
}

#[derive(Debug, Clone, Default)]
pub struct CliffordRzOptimization {
    inner: CliffordRzPass,
}

impl CliffordRzOptimization {
    pub fn new(config: CliffordRzConfig) -> Self {
        Self {
            inner: CliffordRzPass::new(config),
        }
    }

    pub fn execute(&self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        self.inner.run(circuit)
    }
}

impl CliffordRzPass {
    pub(crate) fn new(config: CliffordRzConfig) -> Self {
        Self { config }
    }

    pub(crate) fn run(&self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        let program = preprocess_program(circuit)?;
        let (ops, phase_delta) = self.optimize_program_items(
            &program.logical_qubits,
            &program.parameters,
            &program.items,
            true,
        )?;
        Ok(build_output_circuit_preserving_registers(
            circuit,
            ops,
            phase_delta,
        ))
    }

    fn optimize_program_items(
        &self,
        logical_qubits: &[Qubit],
        parameter_pool: &[Parameter],
        items: &[PreparedProgramItem],
        preserve_phase: bool,
    ) -> Result<(Vec<Operation>, f64), CompileError> {
        let mut ops = Vec::new();
        let mut phase_delta = 0.0;

        for item in items {
            match item {
                PreparedProgramItem::Segment(segment) => {
                    let (segment_ops, segment_phase) =
                        self.optimize_segment(segment, logical_qubits, parameter_pool)?;
                    ops.extend(segment_ops);
                    if preserve_phase {
                        phase_delta += segment_phase;
                    }
                }
                PreparedProgramItem::Passthrough(op) => ops.push(op.op.clone()),
                PreparedProgramItem::IfElse(node) => {
                    let (true_body, _) = self.optimize_program_items(
                        logical_qubits,
                        parameter_pool,
                        &node.true_body.items,
                        false,
                    )?;
                    let false_body = node
                        .false_body
                        .as_ref()
                        .map(|body| {
                            self.optimize_program_items(
                                logical_qubits,
                                parameter_pool,
                                &body.items,
                                false,
                            )
                            .map(|(ops, _)| ops)
                        })
                        .transpose()?;
                    ops.push(build_if_else_operation(
                        node.condition,
                        true_body,
                        false_body,
                        node.label.clone(),
                    ));
                }
                PreparedProgramItem::WhileLoop(node) => {
                    let (body, _) = self.optimize_program_items(
                        logical_qubits,
                        parameter_pool,
                        &node.body.items,
                        false,
                    )?;
                    ops.push(build_while_loop_operation(
                        node.condition,
                        body,
                        node.label.clone(),
                    ));
                }
            }
        }

        Ok((ops, phase_delta))
    }

    fn optimize_segment(
        &self,
        segment: &PreparedSegment,
        logical_qubits: &[Qubit],
        parameter_pool: &[Parameter],
    ) -> Result<(Vec<Operation>, f64), CompileError> {
        let mut out = Vec::new();
        let mut current_chunk = Vec::<CanonicalOp>::new();
        let mut phase_delta = 0.0;

        for prep_op in &segment.operations {
            match try_canonicalize(prep_op, parameter_pool)? {
                Some((canonical, dphi)) => {
                    current_chunk.push(canonical);
                    phase_delta += dphi;
                }
                None => {
                    let (optimized_chunk, chunk_phase) =
                        self.flush_chunk(&mut current_chunk, logical_qubits);
                    out.extend(optimized_chunk);
                    phase_delta += chunk_phase;
                    out.push(prep_op.op.clone());
                }
            }
        }

        let (optimized_chunk, chunk_phase) = self.flush_chunk(&mut current_chunk, logical_qubits);
        out.extend(optimized_chunk);
        phase_delta += chunk_phase;
        Ok((out, phase_delta))
    }

    fn flush_chunk(
        &self,
        chunk: &mut Vec<CanonicalOp>,
        logical_qubits: &[Qubit],
    ) -> (Vec<Operation>, f64) {
        if chunk.is_empty() {
            return (Vec::new(), 0.0);
        }
        let canonical = std::mem::take(chunk);
        self.optimize_supported_chunk(canonical, logical_qubits)
    }

    fn optimize_supported_chunk(
        &self,
        chunk: Vec<CanonicalOp>,
        logical_qubits: &[Qubit],
    ) -> (Vec<Operation>, f64) {
        let max_iters = match self.config.level {
            CliffordRzLevel::Light => 4,
            CliffordRzLevel::Heavy => 32,
        };

        let mut current = chunk;
        for _ in 0..max_iters {
            let next = self.optimize_canonical_chunk(&current);
            if canonical_sequence_eq(&current, &next, self.config.numeric_tol) {
                current = next;
                break;
            }
            current = next;
        }

        let mut out = Vec::new();
        let mut phase_delta = 0.0;

        for op in current {
            if op.is_rz() {
                let theta = normalize_4pi(op.theta_value());
                if let Some((maybe_gate, dphi)) = exact_rz_rewrite(theta, self.config.numeric_tol) {
                    phase_delta += dphi;
                    if let Some(gate) = maybe_gate {
                        out.push(build_standard_operation(
                            gate,
                            &op.logical_qubits,
                            op.label.clone(),
                            logical_qubits,
                        ));
                    }
                    continue;
                }

                if approx_zero(theta, self.config.numeric_tol)
                    || approx_angle_eq(theta, 4.0 * std::f64::consts::PI, self.config.numeric_tol)
                {
                    continue;
                }
                out.push(op.with_theta(theta).to_operation(logical_qubits));
                continue;
            }

            out.push(op.to_operation(logical_qubits));
        }

        (out, phase_delta)
    }

    fn optimize_canonical_chunk(&self, ops: &[CanonicalOp]) -> Vec<CanonicalOp> {
        let mut dag = SegmentDag::from_ops(ops);

        loop {
            let mut changed = false;
            changed |= normalize_rz_nodes(&mut dag, self.config.numeric_tol);
            changed |= merge_adjacent_rz(&mut dag);
            changed |= cancel_self_inverse_pairs(&mut dag);
            if !changed {
                break;
            }
        }

        let simplified = dag.to_ops();
        self.optimize_h_free_components(&simplified)
    }

    fn optimize_h_free_components(&self, ops: &[CanonicalOp]) -> Vec<CanonicalOp> {
        let dag = SegmentDag::from_ops(ops);
        let mut replacements = std::collections::HashMap::<
            usize,
            (std::collections::HashSet<usize>, Vec<CanonicalOp>),
        >::new();

        for component in dag.h_free_components() {
            if let Some(rewritten) =
                PhasePolynomial::optimize_component(&dag, &component, self.config.numeric_tol)
            {
                let Some(&anchor) = component.first() else {
                    continue;
                };
                replacements.insert(anchor, (component.iter().copied().collect(), rewritten));
            }
        }

        if replacements.is_empty() {
            return ops.to_vec();
        }

        let mut skipped = std::collections::HashSet::<usize>::new();
        let mut out = Vec::new();
        for node_id in dag.topological_ids() {
            if skipped.contains(&node_id) {
                continue;
            }
            if let Some((members, rewritten)) = replacements.get(&node_id) {
                skipped.extend(members.iter().copied());
                out.extend(rewritten.iter().cloned());
                continue;
            }
            out.push(dag.node(node_id).op.clone());
        }
        out
    }
}

fn build_output_circuit_preserving_registers(
    source: &Circuit,
    ops: Vec<Operation>,
    phase_delta: f64,
) -> Circuit {
    let qubits: IndexSet<Qubit> = source.qubits().into_iter().collect();
    let mut circuit = Circuit::from_parts(
        qubits,
        source.symbols().clone(),
        source.parameters().clone(),
        ops,
        CircuitParam::Fixed(0.0),
    );
    let phase = if phase_delta.abs() <= 1e-12 {
        source.global_phase()
    } else {
        source.global_phase() + Parameter::from(phase_delta)
    };
    circuit.set_global_phase(phase);
    circuit
}

fn build_standard_operation(
    gate: StandardGate,
    logical_indices: &[usize],
    label: Option<Box<str>>,
    logical_qubits: &[Qubit],
) -> Operation {
    Operation {
        instruction: Instruction::Standard(gate),
        qubits: logical_indices
            .iter()
            .map(|&logical| logical_qubits[logical])
            .collect(),
        params: smallvec::smallvec![],
        label,
    }
}

fn normalize_rz_nodes(dag: &mut SegmentDag, tol: f64) -> bool {
    let mut changed = false;
    let ids = dag.topological_ids();
    for node_id in ids {
        let node = dag.node(node_id).clone();
        if node.op.gate != CanonicalGate::RZ {
            continue;
        }
        let theta = normalize_4pi(node.op.theta_value());
        if approx_zero(theta, tol) || approx_angle_eq(theta, 4.0 * std::f64::consts::PI, tol) {
            dag.erase_node(node_id);
            changed = true;
            continue;
        }
        if !approx_angle_eq(theta, node.op.theta_value(), tol) {
            dag.node_mut(node_id).op.theta = Some(theta);
            changed = true;
        }
    }
    changed
}

fn merge_adjacent_rz(dag: &mut SegmentDag) -> bool {
    let mut changed = false;
    let ids = dag.topological_ids();
    for node_id in ids {
        if dag.node(node_id).erased || dag.node(node_id).op.gate != CanonicalGate::RZ {
            continue;
        }
        let Some(successor) = dag.node(node_id).successors[0] else {
            continue;
        };
        if !dag.is_exposed_pair(node_id, successor.node_id)
            || dag.node(successor.node_id).op.gate != CanonicalGate::RZ
        {
            continue;
        }

        let merged =
            dag.node(node_id).op.theta_value() + dag.node(successor.node_id).op.theta_value();
        dag.node_mut(successor.node_id).op.theta = Some(merged);
        dag.erase_node(node_id);
        changed = true;
    }
    changed
}

fn cancel_self_inverse_pairs(dag: &mut SegmentDag) -> bool {
    let mut changed = false;
    let ids = dag.topological_ids();
    for node_id in ids {
        if dag.node(node_id).erased {
            continue;
        }
        let gate = dag.node(node_id).op.gate;
        if !matches!(
            gate,
            CanonicalGate::H | CanonicalGate::X | CanonicalGate::CX
        ) {
            continue;
        }
        let Some(successor) = dag.node(node_id).successors[0] else {
            continue;
        };
        if !dag.is_exposed_pair(node_id, successor.node_id) {
            continue;
        }
        dag.erase_node(successor.node_id);
        dag.erase_node(node_id);
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ConditionView;
    use crate::circuit::circuit_to_matrix;
    use crate::circuit::{Circuit, Operation, Qubit};
    use num_complex::Complex64;
    use smallvec::smallvec;

    fn matrix_with_global_phase(circuit: &Circuit) -> ndarray::Array2<Complex64> {
        let mut matrix = circuit_to_matrix(circuit, None).unwrap();
        let phase = circuit.global_phase().evaluate(&None).unwrap();
        let factor = Complex64::from_polar(1.0, phase);
        matrix.mapv_inplace(|value| factor * value);
        matrix
    }

    fn assert_matrix_eq(lhs: &Circuit, rhs: &Circuit) {
        let left = matrix_with_global_phase(lhs);
        let right = matrix_with_global_phase(rhs);
        assert_eq!(left.dim(), right.dim());
        for (a, b) in left.iter().zip(right.iter()) {
            assert!(
                (*a - *b).norm() <= 1e-9,
                "matrix mismatch: lhs={:?}, rhs={:?}",
                a,
                b
            );
        }
    }

    fn op_x(logical: u32) -> Operation {
        Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![Qubit::new(logical)],
            params: smallvec![],
            label: None,
        }
    }

    fn op_rz(logical: u32, theta: f64) -> Operation {
        Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(logical)],
            params: smallvec![CircuitParam::Fixed(theta)],
            label: None,
        }
    }

    fn op_cz(q0: u32, q1: u32) -> Operation {
        Operation {
            instruction: Instruction::Standard(StandardGate::CZ),
            qubits: smallvec![Qubit::new(q0), Qubit::new(q1)],
            params: smallvec![],
            label: None,
        }
    }

    fn op_measure(logical: u32) -> Operation {
        Operation {
            instruction: Instruction::Directive(crate::circuit::Directive::Measure),
            qubits: smallvec![Qubit::new(logical)],
            params: smallvec![],
            label: None,
        }
    }

    #[test]
    fn test_linear_rewrite_preserves_matrix_and_qubits() {
        let mut circuit = Circuit::new(1);
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 0);
        assert_eq!(optimized.qubits(), circuit.qubits());
        assert_matrix_eq(&circuit, &optimized);
    }

    #[test]
    fn test_phase_polynomial_rewrite_reduces_cx_network() {
        let mut circuit = Circuit::new(2);
        circuit.rz(Qubit::new(1), 0.2).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();

        let optimized = CliffordRzPass::new(CliffordRzConfig {
            level: CliffordRzLevel::Heavy,
            numeric_tol: 1e-10,
        })
        .run(&circuit)
        .unwrap();

        assert!(optimized.operations().len() < circuit.operations().len());
        assert_matrix_eq(&circuit, &optimized);
    }

    #[test]
    fn test_unsupported_gate_splits_segment() {
        let mut circuit = Circuit::new(2);
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::CZ)
        ));
        assert_matrix_eq(&circuit, &optimized);
    }

    #[test]
    fn test_symbolic_rz_is_hard_boundary() {
        let theta = Parameter::symbol("theta");
        let mut circuit = Circuit::new(1);
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.rz(Qubit::new(0), theta).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::RZ)
        ));
        assert_eq!(optimized.operations()[0].params.len(), 1);
        assert!(matches!(
            optimized.operations()[0].params[0],
            CircuitParam::Index(_)
        ));
    }

    #[test]
    fn test_if_else_body_is_optimized_recursively() {
        let mut circuit = Circuit::new(3);
        circuit.measure(Qubit::new(0)).unwrap();
        let condition = ConditionView::new(Qubit::new(0), 1);
        let true_body = vec![op_x(1), op_rz(1, 0.3), op_x(1), op_rz(1, 0.3), op_cz(1, 2)];
        let false_body = vec![op_x(2), op_rz(2, 0.4), op_x(2), op_rz(2, 0.4)];
        circuit
            .if_else(condition, true_body, Some(false_body))
            .unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        let if_else = &optimized.operations()[1];
        let Instruction::ControlFlowGate(control_flow) = &if_else.instruction else {
            panic!("expected control-flow operation");
        };
        let crate::circuit::ControlFlow::IfElse(gate) = control_flow else {
            panic!("expected if_else control flow");
        };
        assert_eq!(gate.true_body().len(), 1);
        assert!(matches!(
            gate.true_body()[0].instruction,
            Instruction::Standard(StandardGate::CZ)
        ));
        assert_eq!(gate.false_body().unwrap().len(), 0);
    }

    #[test]
    fn test_while_loop_body_optimizes_segments_and_preserves_passthrough() {
        let mut circuit = Circuit::new(2);
        circuit.measure(Qubit::new(0)).unwrap();
        let condition = ConditionView::new(Qubit::new(0), 1);
        let body = vec![
            op_x(1),
            op_rz(1, 0.2),
            op_x(1),
            op_rz(1, 0.2),
            op_measure(1),
        ];
        circuit.while_loop(condition, body).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        let while_loop = &optimized.operations()[1];
        let Instruction::ControlFlowGate(control_flow) = &while_loop.instruction else {
            panic!("expected control-flow operation");
        };
        let crate::circuit::ControlFlow::WhileLoop(gate) = control_flow else {
            panic!("expected while_loop control flow");
        };
        assert_eq!(gate.body().len(), 1);
        assert!(matches!(
            gate.body()[0].instruction,
            Instruction::Directive(crate::circuit::Directive::Measure)
        ));
    }

    #[test]
    fn test_nested_control_flow_structure_is_preserved() {
        let mut circuit = Circuit::new(2);
        circuit.measure(Qubit::new(0)).unwrap();
        let condition = ConditionView::new(Qubit::new(0), 1);
        let nested_true = vec![op_x(1), op_rz(1, 0.25), op_x(1), op_rz(1, 0.25)];
        let nested_if = build_if_else_operation(condition, nested_true, None, None);
        let body = vec![nested_if, op_measure(1)];
        circuit.while_loop(condition, body).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        let while_loop = &optimized.operations()[1];
        let Instruction::ControlFlowGate(control_flow) = &while_loop.instruction else {
            panic!("expected control-flow operation");
        };
        let crate::circuit::ControlFlow::WhileLoop(gate) = control_flow else {
            panic!("expected while_loop control flow");
        };
        assert_eq!(gate.body().len(), 2);
        let Instruction::ControlFlowGate(nested) = &gate.body()[0].instruction else {
            panic!("expected nested control flow");
        };
        let crate::circuit::ControlFlow::IfElse(nested_if) = nested else {
            panic!("expected nested if_else control flow");
        };
        assert_eq!(nested_if.true_body().len(), 0);
        assert!(matches!(
            gate.body()[1].instruction,
            Instruction::Directive(crate::circuit::Directive::Measure)
        ));
    }
}
