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

//! # Circuit DAG Module
//!
//! This module provides a directed acyclic graph (DAG) representation of quantum circuits,
//! enabling advanced circuit analysis and transformations.
//!
//! ## Overview
//!
//! A [`CircuitDag`] represents a quantum circuit as a DAG where:
//! - **Nodes** represent quantum operations (gates, measurements, etc.)
//! - **Edges** represent data flow dependencies between operations on the same qubit
//!
//! This representation is particularly useful for:
//! - **Circuit depth analysis**: Find the critical path length
//! - **Dependency analysis**: Understand operation ordering constraints
//! - **Parallelization**: Identify operations that can be executed concurrently
//! - **Circuit transformations**: Perform optimizations that preserve quantum equivalence
//!
//! ## DAG vs Linear Representation
//!
//! The standard [`Circuit`] representation stores operations in a linear sequence.
//! The DAG representation captures explicit dependencies:
//!
//! ```text
//! Circuit (Linear):  H(0) -> CX(0,1) -> H(1) -> CX(1,2)
//!
//! CircuitDag (DAG):
//!     H(0) -----> CX(0,1)
//!                   |
//!                   v
//!     H(1) -----> CX(1,2)
//! ```
//!
//! Each edge indicates that the source operation must complete before the target operation
//! can begin on the connected qubit.

use crate::circuit::operation::Operation;
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::{Circuit, Parameter, Qubit};
use indexmap::IndexSet;
use rustworkx_core::petgraph::prelude::{NodeIndex, StableDiGraph};
use rustworkx_core::petgraph::visit::Topo;
use smallvec::SmallVec;
use std::collections::HashMap;

/// A directed acyclic graph (DAG) representation of a quantum circuit.
///
/// This structure represents a quantum circuit as a DAG, where nodes are quantum
/// operations and edges represent data flow dependencies on qubits.
///
/// # Type Parameters
///
/// - `N`: Node weight type (always [`Operation`] for this implementation)
/// - `E`: Edge weight type (always `()` as edge weights are not needed)
///
/// # Fields
///
/// - `qubits`: The set of qubits used in the circuit, maintaining insertion order
/// - `symbols`: The set of symbolic variables used in the circuit
/// - `parameters`: The parameter pool for parameterized gates
/// - `global_phase`: The global phase of the circuit
/// - `data`: The underlying graph structure storing operations and dependencies
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::Circuit;
/// use cqlib_core::circuit::dag::CircuitDag;
/// use cqlib_core::circuit::Qubit;
///
/// // Create a simple quantum circuit
/// let mut circuit = Circuit::new(2);
/// circuit.h(Qubit::new(0)).unwrap();
/// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
/// circuit.h(Qubit::new(1)).unwrap();
///
/// // Convert to DAG representation
/// let dag = CircuitDag::from_circuit(&circuit);
///
/// // Convert back to linear circuit
/// let recovered = dag.to_circuit();
///
/// // Verify the operations are preserved
/// assert_eq!(circuit.operations().len(), recovered.operations().len());
/// ```
pub struct CircuitDag {
    /// The set of qubits used in the circuit, maintaining deterministic insertion order.
    pub(crate) qubits: IndexSet<Qubit>,

    /// The set of symbolic variables (e.g., "theta", "phi") used in parameters.
    pub(crate) symbols: IndexSet<String>,

    /// The parameter pool for deduplicated parameter storage.
    pub(crate) parameters: IndexSet<Parameter>,

    /// The global phase of the circuit ($e^{i\theta}$).
    pub(crate) global_phase: CircuitParam,

    /// The underlying DAG structure storing operations as nodes and qubit dependencies as edges.
    pub(crate) data: StableDiGraph<Operation, ()>,
}

impl CircuitDag {
    /// Creates a [`CircuitDag`] from a linear [`Circuit`] representation.
    ///
    /// This function converts a quantum circuit into its DAG representation by:
    /// 1. Preserving all qubits, symbols, and parameters
    /// 2. Creating a node for each operation
    /// 3. Adding edges between operations that operate on the same qubit
    ///
    /// # Arguments
    ///
    /// * `circuit` - The source circuit to convert
    ///
    /// # Returns
    ///
    /// A new [`CircuitDag`] representing the circuit's data flow dependencies
    ///
    /// # Example
    ///
    /// ```rust
    /// use cqlib_core::circuit::{Circuit, CircuitDag, Qubit};
    ///
    /// // Create a circuit with 3 qubits
    /// let mut circuit = Circuit::new(3);
    /// circuit.h(Qubit::new(0)).unwrap();
    /// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    /// circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    ///
    /// // Convert to DAG
    /// let dag = CircuitDag::from_circuit(&circuit);
    /// ```
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let global_phase = circuit.global_phase();
        let mut parameters = circuit.parameters().clone();
        let gp = if let Ok(v) = global_phase.evaluate(&None) {
            CircuitParam::Fixed(v)
        } else {
            let index = parameters.insert_full(global_phase).0;
            CircuitParam::Index(index as u32)
        };
        let mut dag = StableDiGraph::<Operation, ()>::new();
        let mut qubit_last_nodes: HashMap<Qubit, NodeIndex> = HashMap::new();

        for op in circuit.operations() {
            let node_idx = dag.add_node(op.clone());
            for qubit in &op.qubits {
                if let Some(&last_node) = qubit_last_nodes.get(qubit) {
                    // Add edge from the last node to current node
                    dag.add_edge(last_node, node_idx, ());
                }
                // Update the last node for this qubit
                qubit_last_nodes.insert(*qubit, node_idx);
            }
        }

        Self {
            qubits: circuit.qubits().into_iter().collect(),
            symbols: circuit.symbols().clone(),
            parameters,
            global_phase: gp,
            data: dag,
        }
    }

    /// Converts the [`CircuitDag`] back to a linear [`Circuit`] representation.
    ///
    /// This function reconstructs a circuit from its DAG representation by:
    /// 1. Creating a new circuit with the same qubits
    /// 2. Performing a topological sort to determine operation order
    /// 3. Appending operations in topologically sorted order
    /// 4. Restoring the global phase
    ///
    /// # Returns
    ///
    /// A new [`Circuit`] reconstructed from the DAG
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::CircuitDag;
    ///
    /// // Create and convert to DAG
    /// let circuit = Circuit::new(2);
    /// let dag = CircuitDag::from_circuit(&circuit);
    ///
    /// // Convert back to circuit
    /// let recovered = dag.to_circuit();
    ///
    /// // Both should have the same number of qubits
    /// assert_eq!(circuit.num_qubits(), recovered.num_qubits());
    /// ```
    ///
    /// # Note
    ///
    /// The topological sort produces a valid execution order, but may differ
    /// from the original circuit's operation order if the original circuit
    /// had no dependencies between certain operations.
    pub fn to_circuit(&self) -> Circuit {
        let mut circuit = Circuit::from_qubits(self.qubits.clone().into_iter().collect()).unwrap();

        // Use topological sort to get the correct order of operations
        let mut topo = Topo::new(&self.data);
        while let Some(node_idx) = topo.next(&self.data) {
            let op = &self.data[node_idx];

            // Convert CircuitParams to ParameterValues
            let params: SmallVec<[ParameterValue; 1]> = op
                .params
                .iter()
                .map(|p| match p {
                    CircuitParam::Fixed(v) => ParameterValue::Fixed(*v),
                    CircuitParam::Index(idx) => {
                        // Get the parameter from the DAG's parameters pool
                        let param = &self.parameters[*idx as usize];
                        ParameterValue::Param(param.clone())
                    }
                })
                .collect();

            // Append the operation to the circuit
            let _ = circuit.append(
                op.instruction.clone(),
                op.qubits.iter().copied(),
                params.iter().cloned(),
                op.label.as_deref(),
            );
        }

        // Set global phase
        let phase = match &self.global_phase {
            CircuitParam::Fixed(v) => Parameter::from(*v),
            CircuitParam::Index(idx) => self.parameters[*idx as usize].clone(),
        };
        circuit.set_global_phase(phase);

        circuit
    }
}

#[cfg(test)]
#[path = "./dag_test.rs"]
mod dag_test;
