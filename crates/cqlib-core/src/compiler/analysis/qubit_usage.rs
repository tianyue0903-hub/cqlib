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

//! Per-qubit usage summaries for the current circuit.
//!
//! This module answers "how each qubit is used" over the full circuit timeline.
//! It records first/last touch indices, operation category participation, and
//! control-flow condition involvement for each touched qubit.
//!
//! The analysis is designed for:
//! - layout/routing heuristics
//! - live-range aware optimization decisions
//! - user-facing diagnostics on qubit activity

use crate::circuit::{Circuit, ControlFlow, Directive, Instruction, Qubit};
use std::collections::BTreeMap;

/// Per-qubit participation summary derived from the current circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QubitUsage {
    per_qubit: BTreeMap<Qubit, QubitUsageSummary>,
}

impl QubitUsage {
    /// Builds qubit usage statistics by scanning the circuit once.
    ///
    /// Each qubit is counted at most once per operation, even if the underlying
    /// operation stores duplicate qubit handles (for example, control-flow
    /// operations assembled from bodies that reference the same qubit multiple
    /// times).
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut usage = Self::default();

        for (op_index, operation) in circuit.operations().iter().enumerate() {
            let mut unique_qubits = Vec::with_capacity(operation.qubits.len());
            for &qubit in &operation.qubits {
                if !unique_qubits.contains(&qubit) {
                    unique_qubits.push(qubit);
                }
            }

            for qubit in unique_qubits {
                let summary = usage
                    .per_qubit
                    .entry(qubit)
                    .or_insert_with(|| QubitUsageSummary::new(op_index));
                summary.record_operation(
                    op_index,
                    operation.qubits.len(),
                    !operation.params.is_empty(),
                );

                match &operation.instruction {
                    Instruction::Directive(directive) => match directive {
                        Directive::Barrier => summary.barrier_ops += 1,
                        Directive::Measure => {
                            summary.measure_ops += 1;
                            summary.is_measured = true;
                        }
                        Directive::Reset => {
                            summary.reset_ops += 1;
                            summary.is_reset = true;
                        }
                    },
                    Instruction::Delay => summary.delay_ops += 1,
                    Instruction::ControlFlowGate(_) => summary.control_flow_ops += 1,
                    Instruction::Standard(_)
                    | Instruction::McGate(_)
                    | Instruction::UnitaryGate(_)
                    | Instruction::CircuitGate(_) => {}
                }
            }

            if let Instruction::ControlFlowGate(control_flow) = &operation.instruction {
                let condition_qubit = match control_flow {
                    ControlFlow::IfElse(gate) => gate.condition().qubit,
                    ControlFlow::WhileLoop(gate) => gate.condition().qubit,
                };

                let summary = usage
                    .per_qubit
                    .entry(condition_qubit)
                    .or_insert_with(|| QubitUsageSummary::new(op_index));
                summary.appears_in_control_flow_condition = true;
            }
        }

        usage
    }

    /// Returns usage summary for a qubit, if it appears in the circuit.
    pub fn get(&self, qubit: Qubit) -> Option<&QubitUsageSummary> {
        self.per_qubit.get(&qubit)
    }

    /// Returns all touched qubits and their summaries in stable qubit order.
    pub fn entries(&self) -> impl Iterator<Item = (Qubit, &QubitUsageSummary)> {
        self.per_qubit
            .iter()
            .map(|(&qubit, summary)| (qubit, summary))
    }

    /// Returns the number of qubits that appear in at least one operation.
    pub fn total_qubits_touched(&self) -> usize {
        self.per_qubit.len()
    }
}

/// Summary of how a single qubit participates in the current circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QubitUsageSummary {
    pub first_op_index: usize,
    pub last_op_index: usize,
    pub total_ops: usize,
    pub single_qubit_ops: usize,
    pub two_qubit_ops: usize,
    pub multi_qubit_ops: usize,
    pub measure_ops: usize,
    pub reset_ops: usize,
    pub barrier_ops: usize,
    pub delay_ops: usize,
    pub control_flow_ops: usize,
    pub parameterized_ops: usize,
    pub is_measured: bool,
    pub is_reset: bool,
    pub appears_in_control_flow_condition: bool,
}

impl QubitUsageSummary {
    fn new(first_op_index: usize) -> Self {
        Self {
            first_op_index,
            last_op_index: first_op_index,
            total_ops: 0,
            single_qubit_ops: 0,
            two_qubit_ops: 0,
            multi_qubit_ops: 0,
            measure_ops: 0,
            reset_ops: 0,
            barrier_ops: 0,
            delay_ops: 0,
            control_flow_ops: 0,
            parameterized_ops: 0,
            is_measured: false,
            is_reset: false,
            appears_in_control_flow_condition: false,
        }
    }

    fn record_operation(&mut self, op_index: usize, arity: usize, is_parameterized: bool) {
        self.last_op_index = op_index;
        self.total_ops += 1;

        match arity {
            0 => {}
            1 => self.single_qubit_ops += 1,
            2 => self.two_qubit_ops += 1,
            _ => self.multi_qubit_ops += 1,
        }

        if is_parameterized {
            self.parameterized_ops += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QubitUsage;
    use crate::circuit::{Circuit, ConditionView, Operation, Parameter, Qubit, StandardGate};
    use smallvec::smallvec;

    #[test]
    fn qubit_usage_tracks_linear_ranges_and_categories() {
        let mut circuit = Circuit::new(3);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.rx(Qubit::new(1), Parameter::from(0.5)).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();
        circuit.reset(Qubit::new(2)).unwrap();

        let usage = QubitUsage::from_circuit(&circuit);

        assert_eq!(usage.total_qubits_touched(), 3);

        let q0 = usage.get(Qubit::new(0)).unwrap();
        assert_eq!(q0.first_op_index, 0);
        assert_eq!(q0.last_op_index, 1);
        assert_eq!(q0.total_ops, 2);
        assert_eq!(q0.single_qubit_ops, 1);
        assert_eq!(q0.two_qubit_ops, 1);

        let q1 = usage.get(Qubit::new(1)).unwrap();
        assert_eq!(q1.first_op_index, 1);
        assert_eq!(q1.last_op_index, 3);
        assert_eq!(q1.total_ops, 3);
        assert_eq!(q1.two_qubit_ops, 1);
        assert_eq!(q1.single_qubit_ops, 2);
        assert_eq!(q1.measure_ops, 1);
        assert_eq!(q1.parameterized_ops, 1);
        assert!(q1.is_measured);

        let q2 = usage.get(Qubit::new(2)).unwrap();
        assert_eq!(q2.total_ops, 1);
        assert_eq!(q2.reset_ops, 1);
        assert!(q2.is_reset);
    }

    #[test]
    fn qubit_usage_deduplicates_qubits_within_control_flow_operations() {
        let mut circuit = Circuit::new(2);
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![
                    Operation {
                        instruction: StandardGate::H.into(),
                        qubits: smallvec![Qubit::new(1)],
                        params: smallvec![],
                        label: None,
                    },
                    Operation {
                        instruction: StandardGate::X.into(),
                        qubits: smallvec![Qubit::new(1)],
                        params: smallvec![],
                        label: None,
                    },
                ],
                None,
            )
            .unwrap();

        let usage = QubitUsage::from_circuit(&circuit);

        let q0 = usage.get(Qubit::new(0)).unwrap();
        assert_eq!(q0.total_ops, 1);
        assert_eq!(q0.control_flow_ops, 1);
        assert!(q0.appears_in_control_flow_condition);

        let q1 = usage.get(Qubit::new(1)).unwrap();
        assert_eq!(q1.total_ops, 1);
        assert_eq!(q1.control_flow_ops, 1);
        assert_eq!(q1.multi_qubit_ops, 1);
        assert!(!q1.appears_in_control_flow_condition);
    }

    #[test]
    fn qubit_usage_entries_are_stable_by_qubit_order() {
        let mut circuit = Circuit::new(3);
        circuit.x(Qubit::new(2)).unwrap();
        circuit.x(Qubit::new(0)).unwrap();

        let usage = QubitUsage::from_circuit(&circuit);
        let touched: Vec<_> = usage.entries().map(|(qubit, _)| qubit).collect();

        assert_eq!(touched, vec![Qubit::new(0), Qubit::new(2)]);
    }
}
