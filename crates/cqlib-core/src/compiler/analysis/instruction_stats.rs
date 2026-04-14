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

//! Circuit-wide instruction category statistics.
//!
//! This module provides a fast, single-pass summary of operation categories in a
//! [`crate::circuit::Circuit`]. The result is intended for:
//! - transform prechecks
//! - workflow before/after reporting
//! - coarse optimization telemetry
//!
//! Counters are category-based and may overlap. For example, a measurement is
//! counted as both a single-qubit operation and a `measure_ops` directive.

use crate::circuit::{Circuit, Directive, Instruction};

/// Lightweight circuit-wide instruction statistics derived from the current IR.
///
/// These counters are intended for fast transformer prechecks, workflow reporting,
/// and before/after optimization summaries. Category counters may overlap; for
/// example, a measurement contributes to both `single_qubit_ops` and
/// `measure_ops`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstructionStats {
    pub total_ops: usize,
    pub single_qubit_ops: usize,
    pub two_qubit_ops: usize,
    pub multi_qubit_ops: usize,
    pub barrier_ops: usize,
    pub measure_ops: usize,
    pub reset_ops: usize,
    pub delay_ops: usize,
    pub control_flow_ops: usize,
    pub parameterized_ops: usize,
}

impl InstructionStats {
    /// Builds instruction statistics by scanning the circuit once.
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut stats = Self::default();

        for operation in circuit.operations() {
            stats.total_ops += 1;

            match operation.qubits.len() {
                0 => {}
                1 => stats.single_qubit_ops += 1,
                2 => stats.two_qubit_ops += 1,
                _ => stats.multi_qubit_ops += 1,
            }

            if !operation.params.is_empty() {
                stats.parameterized_ops += 1;
            }

            match &operation.instruction {
                Instruction::Directive(directive) => match directive {
                    Directive::Barrier => stats.barrier_ops += 1,
                    Directive::Measure => stats.measure_ops += 1,
                    Directive::Reset => stats.reset_ops += 1,
                },
                Instruction::Delay => stats.delay_ops += 1,
                Instruction::ControlFlowGate(_) => stats.control_flow_ops += 1,
                Instruction::Standard(_)
                | Instruction::McGate(_)
                | Instruction::UnitaryGate(_)
                | Instruction::CircuitGate(_) => {}
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::InstructionStats;
    use crate::circuit::{
        Circuit, ConditionView, Directive, Instruction, Operation, Parameter, Qubit, StandardGate,
    };
    use smallvec::smallvec;

    #[test]
    fn instruction_stats_count_linear_circuit_categories() {
        let mut circuit = Circuit::new(3);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.rx(Qubit::new(2), Parameter::from(0.25)).unwrap();
        circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();
        circuit.reset(Qubit::new(2)).unwrap();

        let stats = InstructionStats::from_circuit(&circuit);

        assert_eq!(stats.total_ops, 6);
        assert_eq!(stats.single_qubit_ops, 4);
        assert_eq!(stats.two_qubit_ops, 2);
        assert_eq!(stats.multi_qubit_ops, 0);
        assert_eq!(stats.barrier_ops, 1);
        assert_eq!(stats.measure_ops, 1);
        assert_eq!(stats.reset_ops, 1);
        assert_eq!(stats.delay_ops, 0);
        assert_eq!(stats.control_flow_ops, 0);
        assert_eq!(stats.parameterized_ops, 1);
    }

    #[test]
    fn instruction_stats_track_delay_and_control_flow() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(Instruction::Delay, [Qubit::new(0)], [], None)
            .unwrap();
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: StandardGate::X.into(),
                    qubits: smallvec![Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }],
                Some(vec![Operation {
                    instruction: Instruction::Directive(Directive::Barrier),
                    qubits: smallvec![Qubit::new(0), Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }]),
            )
            .unwrap();

        let stats = InstructionStats::from_circuit(&circuit);

        assert_eq!(stats.total_ops, 2);
        assert_eq!(stats.single_qubit_ops, 1);
        assert_eq!(stats.two_qubit_ops, 0);
        assert_eq!(stats.multi_qubit_ops, 1);
        assert_eq!(stats.delay_ops, 1);
        assert_eq!(stats.control_flow_ops, 1);
        assert_eq!(stats.barrier_ops, 0);
    }
}
