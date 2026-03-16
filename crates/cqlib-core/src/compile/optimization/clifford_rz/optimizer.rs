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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CliffordRzStrategy {
    Hadamard,
    SingleQubit,
    TwoQubit,
    PhasePolynomial,
    GlobalRz,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliffordRzFlowKind {
    Preset(CliffordRzLevel),
    Custom,
}

#[derive(Debug, Clone)]
pub(crate) struct CliffordRzPass {
    config: CliffordRzConfig,
    flow_kind: CliffordRzFlowKind,
    strategies: Vec<CliffordRzStrategy>,
}

impl Default for CliffordRzPass {
    fn default() -> Self {
        Self::new(CliffordRzConfig::default())
    }
}

#[derive(Debug, Clone)]
pub struct CliffordRzOptimization {
    inner: CliffordRzPass,
}

impl Default for CliffordRzOptimization {
    fn default() -> Self {
        Self::new(CliffordRzConfig::default())
    }
}

impl CliffordRzOptimization {
    pub fn new(config: CliffordRzConfig) -> Self {
        Self {
            inner: CliffordRzPass::new(config),
        }
    }

    pub fn with_custom_flow(strategies: Vec<CliffordRzStrategy>, numeric_tol: f64) -> Self {
        Self {
            inner: CliffordRzPass::with_custom_flow(strategies, numeric_tol),
        }
    }

    pub fn execute(&self, circuit: &Circuit) -> Result<Circuit, CompileError> {
        self.inner.run(circuit)
    }

    pub fn strategies(&self) -> &[CliffordRzStrategy] {
        self.inner.strategies()
    }
}

impl CliffordRzPass {
    pub(crate) fn new(config: CliffordRzConfig) -> Self {
        let level = config.level;
        Self {
            config,
            flow_kind: CliffordRzFlowKind::Preset(level),
            strategies: built_in_strategies(level),
        }
    }

    pub(crate) fn with_custom_flow(strategies: Vec<CliffordRzStrategy>, numeric_tol: f64) -> Self {
        Self {
            config: CliffordRzConfig {
                level: CliffordRzLevel::Heavy,
                numeric_tol,
            },
            flow_kind: CliffordRzFlowKind::Custom,
            strategies,
        }
    }

    pub(crate) fn strategies(&self) -> &[CliffordRzStrategy] {
        &self.strategies
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
                    current_chunk.extend(canonical);
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
        let mut current = chunk;
        for _ in 0..self.max_iters() {
            let mut next = current.clone();
            for &strategy in &self.strategies {
                next = self.apply_strategy(strategy, &next);
            }
            if canonical_sequence_eq(&current, &next, self.config.numeric_tol) {
                current = next;
                break;
            }
            current = next;
        }

        canonical_to_output_operations(current, logical_qubits, self.config.numeric_tol)
    }

    fn apply_strategy(
        &self,
        strategy: CliffordRzStrategy,
        ops: &[CanonicalOp],
    ) -> Vec<CanonicalOp> {
        match strategy {
            CliffordRzStrategy::Hadamard => optimize_hadamard_ops(ops),
            CliffordRzStrategy::SingleQubit => {
                optimize_single_qubit_ops(ops, self.config.numeric_tol)
            }
            CliffordRzStrategy::TwoQubit => optimize_two_qubit_ops(ops),
            CliffordRzStrategy::PhasePolynomial => rewrite_h_free_components(
                ops,
                self.config.numeric_tol,
                PhasePolynomial::optimize_ops,
            ),
            CliffordRzStrategy::GlobalRz => rewrite_h_free_components(
                ops,
                self.config.numeric_tol,
                PhasePolynomial::relocate_ops,
            ),
        }
    }

    fn max_iters(&self) -> usize {
        match self.flow_kind {
            CliffordRzFlowKind::Preset(CliffordRzLevel::Light) => 4,
            CliffordRzFlowKind::Preset(CliffordRzLevel::Heavy) | CliffordRzFlowKind::Custom => 32,
        }
    }
}

fn built_in_strategies(level: CliffordRzLevel) -> Vec<CliffordRzStrategy> {
    match level {
        CliffordRzLevel::Light => vec![
            CliffordRzStrategy::Hadamard,
            CliffordRzStrategy::SingleQubit,
            CliffordRzStrategy::TwoQubit,
        ],
        CliffordRzLevel::Heavy => vec![
            CliffordRzStrategy::Hadamard,
            CliffordRzStrategy::SingleQubit,
            CliffordRzStrategy::TwoQubit,
            CliffordRzStrategy::PhasePolynomial,
            CliffordRzStrategy::GlobalRz,
            CliffordRzStrategy::SingleQubit,
            CliffordRzStrategy::TwoQubit,
        ],
    }
}

fn canonical_to_output_operations(
    ops: Vec<CanonicalOp>,
    logical_qubits: &[Qubit],
    tol: f64,
) -> (Vec<Operation>, f64) {
    let mut out = Vec::new();
    let mut phase_delta = 0.0;

    for op in ops {
        if op.is_rz() {
            let theta = normalize_4pi(op.theta_value());
            if let Some((maybe_gate, dphi)) = exact_rz_rewrite(theta, tol) {
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

            if approx_zero(theta, tol) || approx_angle_eq(theta, 4.0 * std::f64::consts::PI, tol) {
                continue;
            }
            out.push(op.with_theta(theta).to_operation(logical_qubits));
            continue;
        }

        out.push(op.to_operation(logical_qubits));
    }

    (out, phase_delta)
}

fn optimize_hadamard_ops(ops: &[CanonicalOp]) -> Vec<CanonicalOp> {
    let mut current = cancel_exposed_gate_pairs(ops, |gate| gate == CanonicalGate::H);
    loop {
        let next = rewrite_hadamard_patterns(&current);
        if current == next {
            return current;
        }
        current = cancel_exposed_gate_pairs(&next, |gate| gate == CanonicalGate::H);
    }
}

fn optimize_single_qubit_ops(ops: &[CanonicalOp], tol: f64) -> Vec<CanonicalOp> {
    let mut current = ops.to_vec();
    loop {
        let mut dag = SegmentDag::from_ops(&current);
        let mut changed = false;
        changed |= normalize_rz_nodes(&mut dag, tol);
        changed |= merge_adjacent_rz(&mut dag);
        changed |= cancel_exposed_gate_pairs_in_dag(&mut dag, |gate| gate == CanonicalGate::X);
        let mut next = dag.to_ops();
        let (rewritten, rewrote_x_rz_x) = rewrite_x_rz_x_patterns(&next);
        next = rewritten;
        changed |= rewrote_x_rz_x;
        if !changed {
            return next;
        }
        current = next;
    }
}

fn optimize_two_qubit_ops(ops: &[CanonicalOp]) -> Vec<CanonicalOp> {
    cancel_exposed_gate_pairs(ops, |gate| gate == CanonicalGate::CX)
}

fn rewrite_h_free_components(
    ops: &[CanonicalOp],
    tol: f64,
    rewrite: fn(&[CanonicalOp], f64) -> Option<Vec<CanonicalOp>>,
) -> Vec<CanonicalOp> {
    let dag = SegmentDag::from_ops(ops);
    let mut replacements = std::collections::HashMap::<
        usize,
        (std::collections::HashSet<usize>, Vec<CanonicalOp>),
    >::new();

    for component in dag.h_free_components() {
        let original = component
            .iter()
            .map(|&node_id| dag.node(node_id).op.clone())
            .collect::<Vec<_>>();
        if let Some(rewritten) = rewrite(&original, tol) {
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

fn rewrite_hadamard_patterns(ops: &[CanonicalOp]) -> Vec<CanonicalOp> {
    let mut out = Vec::with_capacity(ops.len());
    let mut index = 0usize;
    while index < ops.len() {
        if let Some((consumed, replacement)) = hadamard_rewrite_at(ops, index) {
            out.extend(replacement);
            index += consumed;
            continue;
        }
        out.push(ops[index].clone());
        index += 1;
    }
    out
}

fn hadamard_rewrite_at(ops: &[CanonicalOp], index: usize) -> Option<(usize, Vec<CanonicalOp>)> {
    if let Some(slice) = ops.get(index..index + 5) {
        let cx = slice.get(2)?;
        if cx.gate == CanonicalGate::CX
            && match_h_pair(&slice[0..2], cx.logical_qubits[0], cx.logical_qubits[1])
            && match_h_pair(&slice[3..5], cx.logical_qubits[0], cx.logical_qubits[1])
        {
            let label = first_label(slice);
            return Some((
                5,
                vec![CanonicalOp::cx(cx.logical_qubits[1], cx.logical_qubits[0]).with_label(label)],
            ));
        }
    }

    if let Some(slice) = ops.get(index..index + 3) {
        if let Some(rewritten) = hadamard_sandwich_rewrite(slice) {
            return Some((3, rewritten));
        }
        if let Some(rewritten) = hadamard_pair_move_right(slice) {
            return Some((3, rewritten));
        }
        if let Some(rewritten) = hadamard_pair_move_left(slice) {
            return Some((3, rewritten));
        }
    }

    None
}

fn hadamard_sandwich_rewrite(slice: &[CanonicalOp]) -> Option<Vec<CanonicalOp>> {
    let [left, middle, right] = slice else {
        return None;
    };
    if left.gate != CanonicalGate::H
        || right.gate != CanonicalGate::H
        || left.logical_qubits != right.logical_qubits
    {
        return None;
    }
    if middle.gate != CanonicalGate::CX {
        return None;
    }

    let qubit = left.logical_qubits[0];
    let control = middle.logical_qubits[0];
    let target = middle.logical_qubits[1];
    if control > target {
        return None;
    }
    let other = if qubit == control {
        target
    } else if qubit == target {
        control
    } else {
        return None;
    };
    let label = first_label(slice);
    Some(vec![
        CanonicalOp::h(other).with_label(label),
        CanonicalOp::cx(target, control),
        CanonicalOp::h(other),
    ])
}

fn hadamard_pair_move_right(slice: &[CanonicalOp]) -> Option<Vec<CanonicalOp>> {
    let [first, second, third] = slice else {
        return None;
    };
    if third.gate != CanonicalGate::CX {
        return None;
    }
    let control = third.logical_qubits[0];
    let target = third.logical_qubits[1];
    if control > target {
        return None;
    }
    if !match_h_pair(&[first.clone(), second.clone()], control, target) {
        return None;
    }
    let label = first_label(slice);
    Some(vec![
        CanonicalOp::cx(target, control).with_label(label),
        CanonicalOp::h(control),
        CanonicalOp::h(target),
    ])
}

fn hadamard_pair_move_left(slice: &[CanonicalOp]) -> Option<Vec<CanonicalOp>> {
    let [first, second, third] = slice else {
        return None;
    };
    if first.gate != CanonicalGate::CX {
        return None;
    }
    let control = first.logical_qubits[0];
    let target = first.logical_qubits[1];
    if control > target {
        return None;
    }
    if !match_h_pair(&[second.clone(), third.clone()], control, target) {
        return None;
    }
    let label = first_label(slice);
    Some(vec![
        CanonicalOp::h(control).with_label(label),
        CanonicalOp::h(target),
        CanonicalOp::cx(target, control),
    ])
}

fn match_h_pair(slice: &[CanonicalOp], q0: usize, q1: usize) -> bool {
    if slice.len() != 2 {
        return false;
    }
    let a = &slice[0];
    let b = &slice[1];
    if a.gate != CanonicalGate::H || b.gate != CanonicalGate::H {
        return false;
    }
    let qa = a.logical_qubits[0];
    let qb = b.logical_qubits[0];
    (qa == q0 && qb == q1) || (qa == q1 && qb == q0)
}

fn first_label(slice: &[CanonicalOp]) -> Option<Box<str>> {
    slice.iter().find_map(|op| op.label.clone())
}

fn rewrite_x_rz_x_patterns(ops: &[CanonicalOp]) -> (Vec<CanonicalOp>, bool) {
    let mut out = Vec::with_capacity(ops.len());
    let mut index = 0usize;
    let mut changed = false;
    while index < ops.len() {
        if let Some(slice) = ops.get(index..index + 3) {
            let [first, middle, third] = slice else {
                unreachable!()
            };
            if first.gate == CanonicalGate::X
                && middle.gate == CanonicalGate::RZ
                && third.gate == CanonicalGate::X
                && first.logical_qubits == middle.logical_qubits
                && middle.logical_qubits == third.logical_qubits
            {
                let label = first_label(slice);
                out.push(
                    CanonicalOp::rz(first.logical_qubits[0], -middle.theta_value())
                        .with_label(label),
                );
                index += 3;
                changed = true;
                continue;
            }
        }
        out.push(ops[index].clone());
        index += 1;
    }
    (out, changed)
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

fn cancel_exposed_gate_pairs(
    ops: &[CanonicalOp],
    predicate: impl Fn(CanonicalGate) -> bool,
) -> Vec<CanonicalOp> {
    let mut dag = SegmentDag::from_ops(ops);
    while cancel_exposed_gate_pairs_in_dag(&mut dag, &predicate) {}
    dag.to_ops()
}

fn cancel_exposed_gate_pairs_in_dag(
    dag: &mut SegmentDag,
    predicate: impl Fn(CanonicalGate) -> bool,
) -> bool {
    let mut changed = false;
    let ids = dag.topological_ids();
    for node_id in ids {
        if dag.node(node_id).erased {
            continue;
        }
        let gate = dag.node(node_id).op.gate;
        if !predicate(gate) {
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
        let mut circuit = Circuit::new(1);
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();
        circuit.rx(Qubit::new(0), 0.2).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();
        circuit.x(Qubit::new(0)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::RX)
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
    fn test_symbolic_phase_is_hard_boundary() {
        let theta = Parameter::symbol("theta");
        let mut circuit = Circuit::new(1);
        circuit.x(Qubit::new(0)).unwrap();
        circuit.phase(Qubit::new(0), theta).unwrap();
        circuit.x(Qubit::new(0)).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 3);
        assert!(matches!(
            optimized.operations()[1].instruction,
            Instruction::Standard(StandardGate::Phase)
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
        assert_eq!(gate.true_body().len(), 3);
        assert!(matches!(
            gate.true_body()[0].instruction,
            Instruction::Standard(StandardGate::H)
        ));
        assert!(matches!(
            gate.true_body()[1].instruction,
            Instruction::Standard(StandardGate::CX)
        ));
        assert!(matches!(
            gate.true_body()[2].instruction,
            Instruction::Standard(StandardGate::H)
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

    #[test]
    fn test_rz_merge_happens_in_light_flow() {
        let mut circuit = Circuit::new(1);
        circuit.rz(Qubit::new(0), 0.2).unwrap();
        circuit.rz(Qubit::new(0), 0.3).unwrap();

        let optimized = CliffordRzOptimization::new(CliffordRzConfig {
            level: CliffordRzLevel::Light,
            numeric_tol: 1e-10,
        })
        .execute(&circuit)
        .unwrap();

        assert_eq!(optimized.operations().len(), 1);
        assert!(matches!(
            optimized.operations()[0].instruction,
            Instruction::Standard(StandardGate::RZ)
        ));
    }

    #[test]
    fn test_phase_gate_is_optimized_as_rz_alias() {
        let mut circuit = Circuit::new(1);
        circuit.phase(Qubit::new(0), 0.2).unwrap();
        circuit.phase(Qubit::new(0), 0.3).unwrap();

        let optimized = CliffordRzPass::default().run(&circuit).unwrap();
        assert_eq!(optimized.operations().len(), 1);
        assert_matrix_eq(&circuit, &optimized);
    }

    #[test]
    fn test_cx_pair_cancels_in_two_qubit_strategy() {
        let ops = vec![
            CanonicalOp::cx(0, 1),
            CanonicalOp::cx(0, 1),
            CanonicalOp::x(0),
        ];
        let optimized = optimize_two_qubit_ops(&ops);
        assert_eq!(optimized, vec![CanonicalOp::x(0)]);
    }

    #[test]
    fn test_hadamard_strategy_flips_cx_direction() {
        let ops = vec![
            CanonicalOp::h(0),
            CanonicalOp::h(1),
            CanonicalOp::cx(0, 1),
            CanonicalOp::h(0),
            CanonicalOp::h(1),
        ];
        let optimized = optimize_hadamard_ops(&ops);
        assert_eq!(optimized, vec![CanonicalOp::cx(1, 0)]);
    }

    #[test]
    fn test_light_flow_does_not_run_phase_polynomial_global_rewrite() {
        let mut circuit = Circuit::new(2);
        circuit.rz(Qubit::new(1), 0.2).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.rz(Qubit::new(0), 0.4).unwrap();

        let light = CliffordRzOptimization::new(CliffordRzConfig {
            level: CliffordRzLevel::Light,
            numeric_tol: 1e-10,
        })
        .execute(&circuit)
        .unwrap();
        let heavy = CliffordRzOptimization::new(CliffordRzConfig {
            level: CliffordRzLevel::Heavy,
            numeric_tol: 1e-10,
        })
        .execute(&circuit)
        .unwrap();

        assert!(heavy.operations().len() < light.operations().len());
        assert_matrix_eq(&circuit, &light);
        assert_matrix_eq(&circuit, &heavy);
    }

    #[test]
    fn test_global_rz_relocation_reduces_heavy_flow() {
        let mut circuit = Circuit::new(2);
        circuit.rz(Qubit::new(0), 0.2).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.rz(Qubit::new(1), 0.2).unwrap();

        let custom =
            CliffordRzOptimization::with_custom_flow(vec![CliffordRzStrategy::GlobalRz], 1e-10);
        let optimized = custom.execute(&circuit).unwrap();

        assert!(optimized.operations().len() < circuit.operations().len());
        assert_matrix_eq(&circuit, &optimized);
    }

    #[test]
    fn test_custom_flow_order_matters() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.h(Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.h(Qubit::new(0)).unwrap();
        circuit.h(Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();

        let no_hadamard =
            CliffordRzOptimization::with_custom_flow(vec![CliffordRzStrategy::TwoQubit], 1e-10)
                .execute(&circuit)
                .unwrap();
        let hadamard_then_two = CliffordRzOptimization::with_custom_flow(
            vec![CliffordRzStrategy::Hadamard, CliffordRzStrategy::TwoQubit],
            1e-10,
        )
        .execute(&circuit)
        .unwrap();

        assert!(hadamard_then_two.operations().len() < no_hadamard.operations().len());
        assert_matrix_eq(&circuit, &hadamard_then_two);
    }

    #[test]
    fn test_custom_flow_duplicate_strategies_are_executed_literally() {
        let optimizer = CliffordRzOptimization::with_custom_flow(
            vec![
                CliffordRzStrategy::SingleQubit,
                CliffordRzStrategy::SingleQubit,
                CliffordRzStrategy::TwoQubit,
            ],
            1e-10,
        );
        assert_eq!(
            optimizer.strategies(),
            &[
                CliffordRzStrategy::SingleQubit,
                CliffordRzStrategy::SingleQubit,
                CliffordRzStrategy::TwoQubit
            ]
        );
    }

    #[test]
    fn test_supported_clifford_alias_gates_preserve_matrix() {
        let mut circuit = Circuit::new(2);
        circuit.y(Qubit::new(0)).unwrap();
        circuit.x2p(Qubit::new(0)).unwrap();
        circuit.x2m(Qubit::new(0)).unwrap();
        circuit.y2p(Qubit::new(1)).unwrap();
        circuit.y2m(Qubit::new(1)).unwrap();
        circuit.cy(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.swap(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.phase(Qubit::new(0), 0.125).unwrap();

        let optimized = CliffordRzPass::new(CliffordRzConfig {
            level: CliffordRzLevel::Heavy,
            numeric_tol: 1e-10,
        })
        .run(&circuit)
        .unwrap();

        assert_matrix_eq(&circuit, &optimized);
    }
}
