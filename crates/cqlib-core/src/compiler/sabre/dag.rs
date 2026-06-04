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

//! Dependency DAG used by the compiler SABRE implementation.
//!
//! The DAG is built from circuit operation order and logical-qubit overlap.
//! Nodes are intentionally coarser than single operations: consecutive
//! operations that share the same dependency boundary can be folded together so
//! routing sees the smallest set of scheduling barriers needed for progress.
//!
//! [`SabreNodeKind::TwoQ`] represents a two-logical-qubit interaction that
//! must be adjacent before it can be emitted. Dependencies are derived from a
//! per-wire frontier: each new operation depends on the latest node touching
//! any of its logical qubits, and then becomes the frontier for those qubits.
//! This is enough for SABRE because layout routing only reasons about
//! two-qubit interaction readiness.
//!
//! [`SabreNodeKind::Synchronize`] is used for zero- and one-qubit operations,
//! delays, and directives. These operations do not create a routed two-qubit
//! interaction, but they still preserve sequencing at the current dependency
//! boundary. Initial synchronize operations that touch no mapped frontier stay
//! in [`SabreDag::initial`].
//!
//! Control-flow operations become recursive DAG nodes. The outer node preserves
//! the control-flow operation as a scheduling boundary, while each body is
//! decomposed into its own [`SabreDag`] so routing can restore layouts at block
//! boundaries.

use crate::circuit::{ControlFlow, Instruction, Operation};
use crate::compiler::CompilerError;
use crate::device::LogicalQubit;
use rustworkx_core::petgraph::Direction;
use rustworkx_core::petgraph::graph::DiGraph;
use rustworkx_core::petgraph::prelude::NodeIndex;
use rustworkx_core::petgraph::visit::EdgeRef;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) enum SabreNodeKind {
    Synchronize,
    TwoQ([LogicalQubit; 2]),
    ControlFlow(SabreControlFlow),
}

#[derive(Debug, Clone)]
pub(crate) enum SabreControlFlow {
    IfElse {
        condition: crate::circuit::ConditionView,
        true_body: SabreDag,
        false_body: Option<SabreDag>,
    },
    WhileLoop {
        condition: crate::circuit::ConditionView,
        body: SabreDag,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct SabreNode {
    pub(crate) operations: Vec<Operation>,
    pub(crate) kind: SabreNodeKind,
}

#[derive(Debug, Clone)]
pub(crate) struct SabreDag {
    pub(crate) initial: Vec<Operation>,
    pub(crate) graph: DiGraph<SabreNode, ()>,
    pub(crate) first_layer: Vec<NodeIndex>,
}

impl SabreDag {
    pub(crate) fn from_operations(operations: &[Operation]) -> Result<Self, CompilerError> {
        let mut initial = Vec::new();
        let mut graph = DiGraph::new();
        let mut wire_pos: BTreeMap<LogicalQubit, NodeIndex> = BTreeMap::new();
        let mut first_layer = Vec::new();

        for operation in operations {
            let kind = kind_from_operation(operation)?;
            let qubits = operation
                .qubits
                .iter()
                .copied()
                .map(LogicalQubit::from_qubit)
                .collect::<Vec<_>>();

            let predecessors = predecessors_for(&qubits, &wire_pos);
            match predecessors {
                Predecessors::AllUnmapped => match kind {
                    SabreNodeKind::Synchronize => initial.push(operation.clone()),
                    kind => {
                        let node = graph.add_node(SabreNode {
                            operations: vec![operation.clone()],
                            kind,
                        });
                        first_layer.push(node);
                        for logical in qubits {
                            wire_pos.insert(logical, node);
                        }
                    }
                },
                Predecessors::Single(previous) => {
                    // Synchronize operations share the same dependency boundary,
                    // and consecutive two-qubit operations on the same active
                    // wires remain routable once the first one is routed.
                    let fold_into_previous = can_fold_into_previous(&graph[previous].kind, &kind);
                    if fold_into_previous {
                        graph[previous].operations.push(operation.clone());
                        for logical in qubits {
                            wire_pos.insert(logical, previous);
                        }
                    } else {
                        let node = graph.add_node(SabreNode {
                            operations: vec![operation.clone()],
                            kind,
                        });
                        graph.add_edge(previous, node, ());
                        for logical in qubits {
                            wire_pos.insert(logical, node);
                        }
                    }
                }
                Predecessors::Multiple(parents) => {
                    let node = graph.add_node(SabreNode {
                        operations: vec![operation.clone()],
                        kind,
                    });
                    for parent in parents {
                        if graph.find_edge(parent, node).is_none() {
                            graph.add_edge(parent, node, ());
                        }
                    }
                    for logical in qubits {
                        wire_pos.insert(logical, node);
                    }
                }
            }
        }

        Ok(Self {
            initial,
            graph,
            first_layer,
        })
    }

    pub(crate) fn only_interactions(&self) -> Self {
        let mut graph = DiGraph::with_capacity(self.graph.node_count(), self.graph.edge_count());
        for node in self.graph.node_weights() {
            let kind = match &node.kind {
                SabreNodeKind::TwoQ(pair) => SabreNodeKind::TwoQ(*pair),
                SabreNodeKind::Synchronize | SabreNodeKind::ControlFlow(_) => {
                    SabreNodeKind::Synchronize
                }
            };
            graph.add_node(SabreNode {
                operations: Vec::new(),
                kind,
            });
        }
        for edge in self.graph.edge_references() {
            graph.add_edge(edge.source(), edge.target(), ());
        }
        Self {
            initial: Vec::new(),
            graph,
            first_layer: self.first_layer.clone(),
        }
    }

    pub(crate) fn reverse_interactions(&self) -> Self {
        let mut graph = self.graph.clone();
        graph.reverse();
        let first_layer = graph.externals(Direction::Incoming).collect();
        Self {
            initial: Vec::new(),
            graph,
            first_layer,
        }
    }
}

fn can_fold_into_previous(previous: &SabreNodeKind, current: &SabreNodeKind) -> bool {
    match (previous, current) {
        (_, SabreNodeKind::Synchronize) => true,
        (SabreNodeKind::TwoQ(previous), SabreNodeKind::TwoQ(current)) => {
            same_logical_pair(*previous, *current)
        }
        _ => false,
    }
}

fn same_logical_pair(left: [LogicalQubit; 2], right: [LogicalQubit; 2]) -> bool {
    left == right || [left[1], left[0]] == right
}

enum Predecessors {
    AllUnmapped,
    Single(NodeIndex),
    Multiple(Vec<NodeIndex>),
}

fn predecessors_for(
    qubits: &[LogicalQubit],
    wire_pos: &BTreeMap<LogicalQubit, NodeIndex>,
) -> Predecessors {
    let mut parents = Vec::new();
    for logical in qubits {
        if let Some(parent) = wire_pos.get(logical).copied() {
            if !parents.contains(&parent) {
                parents.push(parent);
            }
        }
    }

    match parents.as_slice() {
        [] => Predecessors::AllUnmapped,
        [parent] => Predecessors::Single(*parent),
        _ => Predecessors::Multiple(parents),
    }
}

fn kind_from_operation(operation: &Operation) -> Result<SabreNodeKind, CompilerError> {
    match &operation.instruction {
        Instruction::ControlFlowGate(flow) => match flow {
            ControlFlow::IfElse(gate) => Ok(SabreNodeKind::ControlFlow(SabreControlFlow::IfElse {
                condition: gate.condition(),
                true_body: SabreDag::from_operations(gate.true_body())?,
                false_body: gate
                    .false_body()
                    .map(SabreDag::from_operations)
                    .transpose()?,
            })),
            ControlFlow::WhileLoop(gate) => {
                Ok(SabreNodeKind::ControlFlow(SabreControlFlow::WhileLoop {
                    condition: gate.condition(),
                    body: SabreDag::from_operations(gate.body())?,
                }))
            }
        },
        Instruction::Directive(_) | Instruction::Delay => Ok(SabreNodeKind::Synchronize),
        _ => match operation.qubits.len() {
            0 | 1 => Ok(SabreNodeKind::Synchronize),
            2 => Ok(SabreNodeKind::TwoQ([
                LogicalQubit::from_qubit(operation.qubits[0]),
                LogicalQubit::from_qubit(operation.qubits[1]),
            ])),
            arity => Err(CompilerError::InvalidInput(format!(
                "sabre requires operations with more than two qubits to be decomposed before routing; found {arity}-qubit operation {}",
                operation.instruction
            ))),
        },
    }
}
