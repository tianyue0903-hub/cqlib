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

//! Control Flow Graph (CFG) representation for quantum circuits.
//!
//! This module provides a CFG-based intermediate representation (IR) for quantum circuits
//! with classical control flow. Unlike traditional quantum circuit representations that
//! assume a linear execution order, this module supports:
//!
//! - **Basic blocks**: Linear sequences of quantum operations with single entry and exit points
//! - **Conditional branching**: If-else constructs based on classical measurement outcomes
//! - **Loops**: While loops with back edges for iterative quantum algorithms
//!
//! # Architecture Overview
//!
//! ```text
//! +-------------------------------------------------------------+
//! |                      CircuitDag                             |
//! |  +-----------------------------------------------------+    |
//! |  |                StableDiGraph                         |    |
//! |  |                                                     |    |
//! |  |   +-----------+         +-----------+              |    |
//! |  |   |  Entry    |-------->|  Block A  |              |    |
//! |  |   |  Block    |         |  [H, CX]  |              |    |
//! |  |   +-----------+         +-----------+              |    |
//! |  |                                   |                 |    |
//! |  |                          [Branch] |                 |    |
//! |  |                                   v                 |    |
//! |  |                        +----------+------+         |    |
//! |  |              +-------->|  Block B      |          |    |
//! |  |              |         |  [X gate]     |          |    |
//! |  |    [True]    |         +---------------+          |    |
//! |  |              |                   |                 |    |
//! |  |              +-------------------+-----------------+    |
//! |  |                                  |  [Jump]              |
//! |  |                                  v                     |
//! |  |                           +----------+                 |
//! |  |                           |  Merge   |                 |
//! |  |                           |  Block   |                 |
//! |  |                           +----------+                 |
//! |  +-----------------------------------------------------+    |
//! |                           |                                |
//! |                    [entry_block]                          |
//! +---------------------------|---------------------------------+
//!                             v
//!                      Circuit Execution
//! ```
//!
//! # Basic Block Structure
//!
//! Each basic block follows the single-entry, single-exit principle:
//!
//! ```text
//!     +-----------------------------+
//!     |        Basic Block          |
//!     |  +-----------------------+  |
//!     |  | Operation 1: H(q0)    |  |
//!     |  | Operation 2: CX(q0,q1)|  |
//!     |  | Operation 3: Measure  |  |
//!     |  +-----------------------+  |
//!     |                             |
//!     |  [Terminator]               |
//!     |  Branch / Jump / Return     |
//!     +-----------------------------+
//! ```
//!
//! # Control Flow Patterns
//!
//! ## If-Else Statement
//!
//! ```text
//!                            +-----------+
//!                            |   Entry   |
//!                            |  [Measure]|
//!                            +-----+-----+
//!                                  |
//!                          [Branch on c]
//!                                  |
//!                    +-------------+-------------+
//!                    |                           |
//!             [TrueBranch]                [FalseBranch]
//!                    |                           |
//!                    v                           v
//!            +-------------+             +-------------+
//!            | True Block  |             | False Block |
//!            |   [X(q1)]   |             |   [Z(q1)]   |
//!            +------+------+             +------+------+
//!                   |                           |
//!                   |      [Unconditional]      |
//!                   +-------------+-------------+
//!                                 |
//!                                 v
//!                          +-------------+
//!                          |Merge Block  |
//!                          |[Return]     |
//!                          +-------------+
//! ```
//!
//! ## While Loop
//!
//! ```text
//!            +-----------+
//!            |   Entry   |
//!            |[Pre-loop] |
//!            +-----+-----+
//!                  |
//!                  | [Unconditional]
//!                  v
//!         +-----------------+
//!    +--->|  Condition      |
//!    |    |  [Branch on c]  |
//!    |    +--------+--------+
//!    |             |
//!    |      [True] | [False]
//!    |             |        \
//!    |             v         v
//!    |    +-------------+  +-------------+
//!    |    | Body Block  |  | Exit Block  |
//!    |    |  [H(q1)]    |  |  [Return]   |
//!    |    +------+------+  +-------------+
//!    |           |
//!    |           | [Unconditional]
//!    +-----------+
//!         (back edge)
//! ```
//!
//! # Example
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, CircuitCFG, Qubit};
//!
//! // Create a simple circuit
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! // Convert to CFG representation
//! let dag = CircuitCFG::from_circuit(&circuit).unwrap();
//! assert_eq!(dag.num_blocks(), 1); // Linear circuit has one basic block
//! ```

use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::gate::control_flow::ControlFlow;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::{Circuit, CircuitError, IfElseGate, WhileLoopGate};
use crate::circuit::{ConditionView, Operation, Parameter, Qubit};
use indexmap::IndexSet;
use rustworkx_core::petgraph::prelude::{EdgeIndex, NodeIndex, StableDiGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use smallvec::smallvec;
use std::collections::{HashSet, VecDeque};

/// Edge weights in the control flow graph representing different types of transitions.
///
/// In a CFG, edges determine how control flows between basic blocks. The edge type
/// is used by graph algorithms and code generation to understand the semantics
/// of the control transfer.
///
/// # Edge Types
///
/// ```text
///     +--------+                           +--------+
///     | Block A|----[TrueBranch]---------> | Block B|
///     |        |                           |        |
///     |        |----[FalseBranch]------->  | Block C|
///     +--------+                           +--------+
///         |
///         | [Unconditional]
///         v
///     +--------+
///     | Block D|
///     +--------+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowEdge {
    /// Edge taken when a conditional branch evaluates to true.
    TrueBranch,
    /// Edge taken when a conditional branch evaluates to false.
    FalseBranch,
    /// Edge representing an unconditional jump or sequential execution.
    Unconditional,
}

/// Terminator instruction that determines control flow out of a basic block.
///
/// Every basic block must end with exactly one terminator, which determines
/// where execution continues. The terminator type determines the outgoing
/// edges required from the block.
///
/// # Terminator Types and Required Edges
///
/// ```text
/// 1. Branch (conditional):
///    +-----------+
///    |   Block   |
///    |[Branch(c)]|
///    +-----+-----+
///       [True]|[False]
///          |  |
///          v  v
///
/// 2. Jump (unconditional):
///    +-----------+              +-----------+
///    |   Block   |--[Jump]----> |   Target  |
///    |[Jump(idx)]|              |   Block   |
///    +-----------+              +-----------+
///
/// 3. Return (termination):
///    +-----------+
///    |   Block   |
///    | [Return]  |--(no outgoing edges)
///    +-----------+
/// ```
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Conditional branch based on a classical measurement outcome.
    ///
    /// The CFG structure must contain two outgoing edges from this block:
    /// one labeled `FlowEdge::TrueBranch` and one labeled `FlowEdge::FalseBranch`.
    Branch(ConditionView),
    /// Unconditional jump to a target basic block.
    ///
    /// The CFG structure must contain one outgoing edge labeled
    /// `FlowEdge::Unconditional` pointing to the target block.
    Jump(NodeIndex),
    /// Termination of the circuit execution.
    ///
    /// This terminator has no outgoing edges and represents the end
    /// of the quantum program.
    Return,
}

/// A basic block in the control flow graph.
///
/// A basic block is a linear sequence of operations with the following properties:
/// - **Single entry**: Control can only enter at the beginning of the block
/// - **Single exit**: Control leaves only through the terminator at the end
/// - **Linear execution**: Operations within the block execute sequentially without branching
///
/// # Structure
///
/// ```text
///     +-----------------------------+
///     |        Basic Block          |
///     |  label: "if_true_0"         |
///     |                             |
///     |  Operations:                |
///     |  +-----------------------+  |
///     |  | 1. H(q0)              |  |
///     |  | 2. CX(q0, q1)         |  |
///     |  +-----------------------+  |
///     |                             |
///     |  Terminator:                |
///     |  [Jump to merge_block]      |
///     +-----------------------------+
/// ```
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::cfg::BasicBlock;
///
/// // Create a new empty basic block
/// let mut block = BasicBlock::new();
/// assert!(block.is_empty());
///
/// // Add a label for debugging
/// let block = BasicBlock::new().with_label("entry");
/// assert_eq!(block.label(), Some("entry"));
/// ```
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Linear sequence of quantum operations within this block.
    ///
    /// Operations are executed in order without any internal branching.
    pub(crate) operations: Vec<Operation>,
    /// Terminator instruction defining control flow out of this block.
    ///
    /// `None` indicates the block is still being constructed and hasn't been
    /// terminated yet. Once set, this should not be changed.
    pub(crate) terminator: Option<Terminator>,
    /// Optional human-readable label for debugging and visualization.
    pub(crate) label: Option<String>,
}

impl BasicBlock {
    /// Creates a new empty basic block.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::cfg::BasicBlock;
    ///
    /// let block = BasicBlock::new();
    /// assert!(block.is_empty());
    /// assert!(!block.has_terminator());
    /// ```
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            terminator: None,
            label: None,
        }
    }

    /// Sets a human-readable label for this basic block.
    ///
    /// This is useful for debugging and generating human-readable output.
    ///
    /// # Arguments
    ///
    /// * `label` - A string identifier for this block
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::cfg::BasicBlock;
    ///
    /// let block = BasicBlock::new().with_label("if_true_branch");
    /// assert_eq!(block.label(), Some("if_true_branch"));
    /// ```
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Appends a single operation to this basic block.
    ///
    /// # Arguments
    ///
    /// * `op` - The quantum operation to append
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::cfg::BasicBlock;
    /// use cqlib_core::circuit::{Operation, Qubit};
    ///
    /// let mut block = BasicBlock::new();
    /// // block.push_operation(operation);
    /// assert_eq!(block.len(), 0);
    /// ```
    pub fn push_operation(&mut self, op: Operation) {
        self.operations.push(op);
    }

    /// Extends this basic block with multiple operations.
    ///
    /// # Arguments
    ///
    /// * `ops` - An iterator of quantum operations to append
    pub fn extend_operations(&mut self, ops: impl IntoIterator<Item = Operation>) {
        self.operations.extend(ops);
    }

    /// Sets the terminator instruction for this basic block.
    ///
    /// # Arguments
    ///
    /// * `terminator` - The terminator defining control flow out of this block
    ///
    /// # Panics
    ///
    /// While this function doesn't panic, setting a terminator when one already
    /// exists may indicate a logic error in the CFG construction.
    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.terminator = Some(terminator);
    }

    /// Returns `true` if this block has no operations and no terminator.
    ///
    /// An empty block typically represents a control flow join point or
    /// an unused code path.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty() && self.terminator.is_none()
    }

    /// Returns `true` if this block has been terminated.
    ///
    /// A terminated block has a defined control flow out of it and is
    /// considered complete.
    pub fn has_terminator(&self) -> bool {
        self.terminator.is_some()
    }

    /// Returns the number of operations in this basic block.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns the label of this basic block, if any.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Control Flow Graph (CFG) representation of a quantum circuit.
///
/// `CircuitDag` represents a quantum circuit as a directed graph of basic blocks,
/// enabling efficient analysis and transformation of circuits with classical
/// control flow (conditionals and loops).
///
/// # Structure
///
/// ```text
/// +-------------------------------------------------+
/// |                  CircuitDag                     |
/// |                                                 |
/// |  Fields:                                        |
/// |  - qubits: IndexSet<Qubit>                      |
/// |  - symbols: IndexSet<String>                    |
/// |  - parameters: IndexSet<Parameter>              |
/// |  - global_phase: CircuitParam                   |
/// |  - entry_block: Option<NodeIndex>               |
/// |                                                 |
/// |  +-------------------------------------------+  |
/// |  |         StableDiGraph                     |  |
/// |  |  Nodes: BasicBlock                        |  |
/// |  |  Edges: FlowEdge                          |  |
/// |  +-------------------------------------------+  |
/// |                                                 |
/// +-------------------------------------------------+
/// ```
///
/// # Control Flow Construction
///
/// The CFG is constructed from a linear `Circuit` representation by:
///
/// 1. Creating a single entry basic block
/// 2. Linearly processing operations, appending to the current block
/// 3. When encountering control flow operations, creating appropriate structures:
///
/// ## If-Else Construction
///
/// ```text
///     Before:                    After:
///     +--------+                +-----------+
///     | Current|                |  Current  |
///     |  Block |                |[Branch(c)]|
///     +--------+                +-----+-----+
///                                      |
///                    +-----------------+------------------+
///                    |                                    |
///             [TrueBranch]                          [FalseBranch]
///                    |                                    |
///                    v                                    v
///           +-------------+                      +-------------+
///           | true_entry  |                      | false_entry |
///           | [ops...]    |                      | [ops...]    |
///           +------+------+                      +------+------+
///                  |                                    |
///                  +------------------+-----------------+
///                                     |
///                                     v
///                              +-------------+
///                              |merge_block  |
///                              | (continue)  |
///                              +-------------+
/// ```
///
/// ## While Loop Construction
///
/// ```text
///     Before:                    After:
///     +--------+                +-----------+
///     | Current|                |  Current  |
///     |  Block |                |[Jump]     |
///     +--------+                +-----+-----+
///                                      |
///                                      v
///                               +-------------+
///                          +--->|  cond_block |
///                          |    |[Branch(c)]  |
///                          |    +------+------+
///                          |           |
///                    [True]|           |[False]
///                          |           |
///                          |    +------+------+
///                          |    |  exit_block |
///                          |    |  (continue) |
///                          |    +-------------+
///                          |
///                   +------+------+
///                   |  body_block |
///                   |  [ops...]   |
///                   +------+------+
///                          |
///                          | [Jump back]
///                          +-------------+
/// ```
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, CircuitCFG, Qubit};
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(Qubit::new(0)).unwrap();
/// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
///
/// let dag = CircuitCFG::from_circuit(&circuit).unwrap();
/// assert_eq!(dag.num_qubits(), 2);
/// assert_eq!(dag.num_blocks(), 1); // Linear circuit
/// ```
pub struct CircuitCFG {
    /// The set of qubits used in the circuit, maintaining deterministic insertion order.
    pub(crate) qubits: IndexSet<Qubit>,

    /// The set of symbolic variables (e.g., "theta", "phi") used in parameters.
    #[allow(dead_code)]
    pub(crate) symbols: IndexSet<String>,

    /// The parameter pool for deduplicated parameter storage.
    pub(crate) parameters: IndexSet<Parameter>,

    /// The global phase of the circuit ($e^{i\theta}$).
    pub(crate) global_phase: CircuitParam,

    /// The underlying directed graph storing basic blocks and control flow edges.
    ///
    /// Uses `StableDiGraph` to ensure node indices remain stable across modifications.
    pub(crate) data: StableDiGraph<BasicBlock, FlowEdge>,

    /// The entry basic block of the CFG.
    ///
    /// This is where circuit execution begins. `None` indicates an uninitialized CFG.
    pub(crate) entry_block: Option<NodeIndex>,
}

impl CircuitCFG {
    /// Creates a new empty `CircuitDag` with the specified number of qubits.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the circuit
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::CircuitCFG;
    ///
    /// let dag = CircuitCFG::new(3);
    /// assert_eq!(dag.num_qubits(), 3);
    /// assert_eq!(dag.num_blocks(), 0);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        let qubits = (0..num_qubits).map(|i| Qubit::new(i as u32)).collect();

        Self {
            qubits,
            symbols: IndexSet::new(),
            parameters: IndexSet::new(),
            global_phase: CircuitParam::Fixed(0.0),
            data: StableDiGraph::new(),
            entry_block: None,
        }
    }

    /// Creates a `CircuitDag` from an existing vector of qubits.
    ///
    /// This is useful when you need to preserve specific qubit identities
    /// rather than creating sequential qubits.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of qubit handles to use in the circuit
    pub fn from_qubits(qubits: Vec<Qubit>) -> Self {
        Self {
            qubits: qubits.into_iter().collect(),
            symbols: IndexSet::new(),
            parameters: IndexSet::new(),
            global_phase: CircuitParam::Fixed(0.0),
            data: StableDiGraph::new(),
            entry_block: None,
        }
    }

    /// Adds a basic block to the CFG and returns its node index.
    ///
    /// # Arguments
    ///
    /// * `block` - The basic block to add
    ///
    /// # Returns
    ///
    /// The `NodeIndex` that can be used to reference this block in the graph.
    pub fn add_block(&mut self, block: BasicBlock) -> NodeIndex {
        self.data.add_node(block)
    }

    /// Adds a control flow edge between two basic blocks.
    ///
    /// # Arguments
    ///
    /// * `source` - The source basic block
    /// * `target` - The target basic block
    /// * `flow` - The type of control flow (true/false branch or unconditional)
    ///
    /// # Returns
    ///
    /// The `EdgeIndex` of the newly created edge, or `None` if the edge couldn't be created.
    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        flow: FlowEdge,
    ) -> Option<EdgeIndex> {
        Some(self.data.add_edge(source, target, flow))
    }

    /// Returns the entry block of the CFG, if set.
    pub fn entry_block(&self) -> Option<NodeIndex> {
        self.entry_block
    }

    /// Sets the entry block of the CFG.
    ///
    /// # Arguments
    ///
    /// * `index` - The node index of the entry basic block
    pub fn set_entry_block(&mut self, index: NodeIndex) {
        self.entry_block = Some(index);
    }

    /// Returns an iterator over all basic blocks in the CFG.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(NodeIndex, &BasicBlock)` tuples.
    pub fn blocks(&self) -> impl Iterator<Item = (NodeIndex, &BasicBlock)> {
        self.data.node_indices().map(|i| (i, &self.data[i]))
    }

    /// Returns the number of basic blocks in the CFG.
    pub fn num_blocks(&self) -> usize {
        self.data.node_indices().count()
    }

    /// Returns the number of qubits in the circuit.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns all qubits used in the circuit.
    ///
    /// # Returns
    ///
    /// A vector of qubit handles in deterministic order.
    pub fn qubits(&self) -> Vec<Qubit> {
        self.qubits.iter().cloned().collect()
    }

    /// Converts a linear `Circuit` into a CFG representation.
    ///
    /// This function transforms a sequential circuit into a control flow graph,
    /// handling:
    /// - Linear sequences of operations (single basic block)
    /// - If-else constructs (entry → true/false branches → merge)
    /// - While loops (condition → body → back edge → exit)
    ///
    /// # Arguments
    ///
    /// * `circuit` - The circuit to convert
    ///
    /// # Returns
    ///
    /// `Ok(CircuitDag)` on success, or a `CircuitError` if conversion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::{Circuit, CircuitCFG, Qubit};
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.h(Qubit::new(0)).unwrap();
    ///
    /// let dag = CircuitCFG::from_circuit(&circuit).unwrap();
    /// assert_eq!(dag.num_blocks(), 1);
    /// ```
    pub fn from_circuit(circuit: &Circuit) -> Result<Self, CircuitError> {
        let mut dag = Self::from_qubits(circuit.qubits());
        dag.symbols = circuit.symbols().clone();
        dag.parameters = circuit.parameters().clone();

        // Convert global phase to CircuitParam
        let phase_param = circuit.global_phase();
        dag.global_phase = if let Ok(val) = phase_param.evaluate(&None) {
            CircuitParam::Fixed(val)
        } else {
            let (index, is_new) = dag.parameters.insert_full(phase_param.clone());
            if is_new {
                for sym in phase_param.get_symbols() {
                    dag.symbols.insert(sym);
                }
            }
            CircuitParam::Index(index as u32)
        };

        // 1. Create entry block (even for empty circuits)
        let entry_idx = dag.add_block(BasicBlock::new().with_label("entry"));
        dag.set_entry_block(entry_idx);

        // For empty circuits, just set Return terminator and return
        if circuit.operations().is_empty() {
            dag.data[entry_idx].set_terminator(Terminator::Return);
            return Ok(dag);
        }

        // 2. Process all operations linearly, tracking the final block
        let final_block = process_operations(circuit.operations(), &mut dag, entry_idx)?;

        // 3. Set Return terminator on the final block if not already terminated
        if !dag.data[final_block].has_terminator() {
            dag.data[final_block].set_terminator(Terminator::Return);
        }

        Ok(dag)
    }

    /// Converts a `CircuitDag` back to a nested `Circuit` representation.
    ///
    /// This function performs the inverse of `from_circuit()`: it traverses
    /// the CFG and reconstructs the nested AST structure by matching
    /// control flow patterns (If-Else convergence and While Loop back-edges).
    pub fn to_circuit(&self) -> Result<Circuit, CircuitError> {
        let mut ops = Vec::new();

        if let Some(entry) = self.entry_block {
            ops = self.parse_subgraph(entry, None)?;
        }

        Ok(Circuit::from_parts(
            self.qubits.clone(),
            self.symbols.clone(),
            self.parameters.clone(),
            ops,
            self.global_phase.clone(),
        ))
    }

    fn parse_subgraph(
        &self,
        start_node: NodeIndex,
        stop_node: Option<NodeIndex>,
    ) -> Result<Vec<Operation>, CircuitError> {
        let mut ops = Vec::new();
        let mut current = Some(start_node);

        while let Some(node) = current {
            if Some(node) == stop_node {
                break;
            }

            // Append regular quantum operations from the basic block
            let block = &self.data[node];
            ops.extend(block.operations.clone());

            match &block.terminator {
                Some(Terminator::Return) | None => {
                    current = None;
                }
                Some(Terminator::Jump(target)) => {
                    current = Some(*target);
                }
                Some(Terminator::Branch(condition)) => {
                    let mut true_target = None;
                    let mut false_target = None;

                    for edge in self.data.edges(node) {
                        match edge.weight() {
                            FlowEdge::TrueBranch => true_target = Some(edge.target()),
                            FlowEdge::FalseBranch => false_target = Some(edge.target()),
                            _ => {}
                        }
                    }

                    let true_target = true_target.ok_or_else(|| {
                        let block_label = block.label().unwrap_or("<unlabeled>");
                        CircuitError::InvalidControlFlow(format!(
                            "Block '{}' (index {:?}) has a Branch terminator but is missing a TrueBranch outgoing edge. \
                             Expected a FlowEdge::TrueBranch edge from this block to the true branch target.",
                            block_label, node
                        ))
                    })?;
                    let false_target = false_target.ok_or_else(|| {
                        let block_label = block.label().unwrap_or("<unlabeled>");
                        CircuitError::InvalidControlFlow(format!(
                            "Block '{}' (index {:?}) has a Branch terminator but is missing a FalseBranch outgoing edge. \
                             Expected a FlowEdge::FalseBranch edge from this block to the false branch target.",
                            block_label, node
                        ))
                    })?;

                    // Determine structure type by checking block label
                    // While loop: cond block label starts with "while_cond_"
                    // If-Else: cond block label starts with "if_" or other
                    let is_while = block.label().is_some_and(|l| l.starts_with("while_cond_"));

                    if is_while {
                        // While Loop: true branch is the loop body with back edge
                        let body_ops = self.parse_subgraph(true_target, Some(node))?;

                        ops.push(Operation {
                            instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                                WhileLoopGate::new(*condition, body_ops),
                            )),
                            qubits: smallvec![],
                            params: smallvec![],
                            label: None,
                        });

                        // Continue with false branch (exit path)
                        current = Some(false_target);
                    } else {
                        // If-Else: find merge point where both branches converge
                        let merge_node = self.find_merge_node(true_target, false_target);

                        let true_ops = if let Some(merge) = merge_node {
                            self.parse_subgraph(true_target, Some(merge))?
                        } else {
                            self.parse_subgraph(true_target, None)?
                        };

                        let false_ops = if let Some(merge) = merge_node {
                            self.parse_subgraph(false_target, Some(merge))?
                        } else {
                            self.parse_subgraph(false_target, None)?
                        };

                        let false_body = if false_ops.is_empty() {
                            None
                        } else {
                            Some(false_ops)
                        };

                        ops.push(Operation {
                            instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(
                                IfElseGate::new(*condition, true_ops, false_body),
                            )),
                            qubits: smallvec![],
                            params: smallvec![],
                            label: None,
                        });

                        current = merge_node;
                    }
                }
            }
        }

        Ok(ops)
    }

    /// Finds the merge node of an if-else structure by label pattern.
    ///
    /// If-Else blocks are labeled with patterns like "if_true_0", "if_false_0", "if_merge_0".
    /// This method finds the merge node by looking for the "if_merge_" label
    /// that corresponds to the true/false branch labels.
    fn find_merge_node(
        &self,
        true_branch: NodeIndex,
        false_branch: NodeIndex,
    ) -> Option<NodeIndex> {
        // Extract the index from the true branch label
        let merge_label_prefix = self.data[true_branch].label().and_then(|label| {
            // Extract index from "if_true_X" pattern
            label
                .strip_prefix("if_true_")
                .map(|idx| format!("if_merge_{}", idx))
        });

        if let Some(expected_label) = merge_label_prefix {
            // Search for the merge block with matching label
            for node_idx in self.data.node_indices() {
                if let Some(label) = self.data[node_idx].label() {
                    if label == expected_label {
                        return Some(node_idx);
                    }
                }
            }
        }

        // Fallback: use graph traversal to find common descendant
        let mut descendants1 = HashSet::new();
        let mut stack = vec![true_branch];

        while let Some(n) = stack.pop() {
            if descendants1.insert(n) {
                for edge in self.data.edges(n) {
                    let target = edge.target();
                    stack.push(target);
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut visited2 = HashSet::new();
        queue.push_back(false_branch);

        while let Some(n) = queue.pop_front() {
            if descendants1.contains(&n) {
                return Some(n);
            }
            if visited2.insert(n) {
                for edge in self.data.edges(n) {
                    queue.push_back(edge.target());
                }
            }
        }

        None
    }
}

/// Recursively processes a sequence of operations, building the CFG.
///
/// This is the core function for CFG construction. It linearly processes
/// operations, creating new basic blocks as needed for control flow constructs.
///
/// # Arguments
///
/// * `circuit` - The source circuit (for context)
/// * `operations` - The slice of operations to process
/// * `dag` - The CFG being constructed
/// * `current_block` - The current basic block to append operations to
///
/// # Returns
///
/// The `NodeIndex` of the last block processed, for use in connecting
/// control flow edges.
///
/// # Control Flow Handling
///
/// ## If-Else
///
/// ```text
/// Input: IfElse { condition, true_body, false_body }
///
/// Output CFG:
///
///     current_block (terminated with Branch)
///           |
///      +----+----+
///      |         |
/// [TrueBranch] [FalseBranch]
///      |         |
///      v         v
/// true_entry  false_entry
///      |         |
///      | [process_operations recursively]
///      |         |
///      v         v
/// true_exit   false_exit
///      |         |
///      +----+----+
///           |
///           v
///     merge_block (becomes new current_block)
/// ```
///
/// ## While Loop
///
/// ```text
/// Input: WhileLoop { condition, body }
///
/// Output CFG:
///
///     current_block (terminated with Jump)
///           |
///           v
///     cond_block (terminated with Branch)
///           |
///      +----+----+
///      |         |
/// [TrueBranch] [FalseBranch]
///      |         |
///      v         v
/// body_entry  exit_block
///      |         |
///      |     (becomes new
///      |      current_block)
///      v
/// [process body operations]
///      |
///      v
/// body_exit (terminated with Jump)
///      |
///      +----+----+
///           |
///           v
///     cond_block (back edge - loop!)
/// ```
fn process_operations(
    operations: &[Operation],
    dag: &mut CircuitCFG,
    mut current_block: NodeIndex,
) -> Result<NodeIndex, CircuitError> {
    for (idx, op) in operations.iter().enumerate() {
        match &op.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(if_else)) => {
                // If-Else structure: cond_block -> [true_body, false_body] -> merge_block
                // 1. Terminate current block with a conditional branch
                dag.data[current_block].set_terminator(Terminator::Branch(if_else.condition()));

                // 2. Create and process True branch
                let true_entry =
                    dag.add_block(BasicBlock::new().with_label(format!("if_true_{}", idx)));
                dag.add_edge(current_block, true_entry, FlowEdge::TrueBranch);
                let true_exit = process_operations(if_else.true_body(), dag, true_entry)?;

                // 3. Create and process False branch
                let false_entry =
                    dag.add_block(BasicBlock::new().with_label(format!("if_false_{}", idx)));
                dag.add_edge(current_block, false_entry, FlowEdge::FalseBranch);
                let false_exit = if let Some(false_ops) = if_else.false_body() {
                    process_operations(false_ops, dag, false_entry)?
                } else {
                    false_entry // No else: empty block falls through
                };

                // 4. Create Merge block (convergence point)
                let merge_block =
                    dag.add_block(BasicBlock::new().with_label(format!("if_merge_{}", idx)));

                // 5. Connect True/False exits to Merge block with unconditional jumps
                dag.data[true_exit].set_terminator(Terminator::Jump(merge_block));
                dag.add_edge(true_exit, merge_block, FlowEdge::Unconditional);

                dag.data[false_exit].set_terminator(Terminator::Jump(merge_block));
                dag.add_edge(false_exit, merge_block, FlowEdge::Unconditional);

                // 6. Set current block to Merge block for subsequent operations
                current_block = merge_block;
            }

            Instruction::ControlFlowGate(ControlFlow::WhileLoop(while_gate)) => {
                // While structure: cond_block -> [body_block -> Jump to cond_block, exit_block]
                // 1. Create a dedicated Condition block (prevents including preceding operations in the loop)
                let cond_block =
                    dag.add_block(BasicBlock::new().with_label(format!("while_cond_{}", idx)));
                dag.data[current_block].set_terminator(Terminator::Jump(cond_block));
                dag.add_edge(current_block, cond_block, FlowEdge::Unconditional);

                // 2. Set Condition block's branching logic
                dag.data[cond_block].set_terminator(Terminator::Branch(while_gate.condition()));

                // 3. Create and process Body branch
                let body_entry =
                    dag.add_block(BasicBlock::new().with_label(format!("while_body_{}", idx)));
                dag.add_edge(cond_block, body_entry, FlowEdge::TrueBranch);
                let body_exit = process_operations(while_gate.body(), dag, body_entry)?;

                // 4. Jump back to Condition block (forms the back edge)
                dag.data[body_exit].set_terminator(Terminator::Jump(cond_block));
                dag.add_edge(body_exit, cond_block, FlowEdge::Unconditional);

                // 5. Create Exit block (path after loop exits)
                let exit_block =
                    dag.add_block(BasicBlock::new().with_label(format!("while_exit_{}", idx)));
                dag.add_edge(cond_block, exit_block, FlowEdge::FalseBranch);

                // 6. Set current block to Exit block for operations after the loop
                current_block = exit_block;
            }

            _ => {
                // Regular operations (quantum gates, measurements) - append to current block
                dag.data[current_block].push_operation(op.clone());
            }
        }
    }

    // Return the index of the last block we were processing
    Ok(current_block)
}

#[cfg(test)]
#[path = "./cfg_test.rs"]
mod cfg_test;
