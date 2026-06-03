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

//! Circuit interaction analysis used by layout methods.
//!
//! Extracts weighted two-qubit interactions from the main circuit body.
//! Control-flow operations (if/else, while loops) are skipped — only
//! top-level operations in the flat instruction list are analyzed.

use crate::circuit::{Circuit, Qubit};
use crate::compiler::CompilerError;
use crate::device::LogicalQubit;
use std::collections::BTreeMap;

/// Layout-relevant summary of a circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct CircuitLayoutAnalysis {
    /// Logical qubits in circuit order.
    pub logical_qubits: Vec<LogicalQubit>,
    /// Weighted two-qubit interaction graph extracted from operations.
    pub interactions: InteractionGraph,
}

/// One weighted logical interaction.
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    /// Lower-sorted logical qubit endpoint.
    pub left: LogicalQubit,
    /// Higher-sorted logical qubit endpoint.
    pub right: LogicalQubit,
    /// Total interaction weight accumulated for this unordered pair.
    pub weight: f64,
    /// Weight observed in operation order `left -> right`.
    pub directed_weight_left_to_right: f64,
    /// Weight observed in operation order `right -> left`.
    pub directed_weight_right_to_left: f64,
    /// First operation order at which this interaction was seen.
    pub first_seen_order: usize,
}

/// Weighted logical interaction graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InteractionGraph {
    interactions: Vec<Interaction>,
}

impl InteractionGraph {
    /// Creates an empty interaction graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all interactions in deterministic endpoint order.
    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    /// Returns whether no two-qubit interactions were observed.
    pub fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    /// Returns the number of logical interaction edges.
    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    /// Returns per-logical-qubit activity weight derived from incident
    /// interactions.
    pub fn logical_activity(&self) -> BTreeMap<LogicalQubit, f64> {
        let mut activity = BTreeMap::new();
        for interaction in &self.interactions {
            *activity.entry(interaction.left).or_insert(0.0) += interaction.weight;
            *activity.entry(interaction.right).or_insert(0.0) += interaction.weight;
        }
        activity
    }

    fn add_operation_interaction(
        &mut self,
        first: LogicalQubit,
        second: LogicalQubit,
        weight: f64,
        order: usize,
    ) {
        let (left, right, left_to_right) = if first <= second {
            (first, second, true)
        } else {
            (second, first, false)
        };

        match self
            .interactions
            .binary_search_by_key(&(left, right), |i| (i.left, i.right))
        {
            Ok(index) => {
                let interaction = &mut self.interactions[index];
                interaction.weight += weight;
                if left_to_right {
                    interaction.directed_weight_left_to_right += weight;
                } else {
                    interaction.directed_weight_right_to_left += weight;
                }
                interaction.first_seen_order = interaction.first_seen_order.min(order);
            }
            Err(index) => {
                self.interactions.insert(
                    index,
                    Interaction {
                        left,
                        right,
                        weight,
                        directed_weight_left_to_right: if left_to_right { weight } else { 0.0 },
                        directed_weight_right_to_left: if left_to_right { 0.0 } else { weight },
                        first_seen_order: order,
                    },
                );
            }
        }
    }
}

/// Extracts layout-relevant interaction data from a circuit.
///
/// Scans the flat operation list only. Control-flow gates (if/else, while)
/// are skipped — their bodies are not recursively analyzed.
pub fn analyze_circuit_for_layout(
    circuit: &Circuit,
) -> Result<CircuitLayoutAnalysis, CompilerError> {
    let mut analyzer = InteractionAnalyzer::new(circuit.qubits());
    for operation in circuit.operations() {
        analyzer.scan_operation(operation)?;
    }
    Ok(analyzer.finish())
}

struct InteractionAnalyzer {
    logical_qubits: Vec<LogicalQubit>,
    interactions: InteractionGraph,
    operation_order: usize,
}

impl InteractionAnalyzer {
    fn new(qubits: Vec<Qubit>) -> Self {
        Self {
            logical_qubits: qubits.into_iter().map(LogicalQubit::from_qubit).collect(),
            interactions: InteractionGraph::new(),
            operation_order: 0,
        }
    }

    fn scan_operation(
        &mut self,
        operation: &crate::circuit::Operation,
    ) -> Result<(), CompilerError> {
        use crate::circuit::Instruction;

        match &operation.instruction {
            // Skip control-flow gates — layout does not analyze loop/conditional bodies.
            Instruction::ControlFlowGate(_) => Ok(()),
            // Skip directives and delays.
            Instruction::Directive(_) | Instruction::Delay => Ok(()),
            _ => {
                match operation.qubits.len() {
                    0 | 1 => {}
                    2 => {
                        let first = LogicalQubit::from_qubit(operation.qubits[0]);
                        let second = LogicalQubit::from_qubit(operation.qubits[1]);
                        self.interactions.add_operation_interaction(
                            first,
                            second,
                            1.0,
                            self.operation_order,
                        );
                    }
                    arity => {
                        return Err(CompilerError::InvalidInput(format!(
                            "layout requires unitary operations with more than two qubits to be decomposed before layout; found {}-qubit operation {}",
                            arity, operation.instruction
                        )));
                    }
                }
                self.operation_order += 1;
                Ok(())
            }
        }
    }

    fn finish(self) -> CircuitLayoutAnalysis {
        CircuitLayoutAnalysis {
            logical_qubits: self.logical_qubits,
            interactions: self.interactions,
        }
    }
}
