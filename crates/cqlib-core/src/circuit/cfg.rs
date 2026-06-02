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
//! |                      CircuitCFG                             |
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
//! let cfg = CircuitCFG::from_circuit(&circuit).unwrap();
//! assert_eq!(cfg.num_blocks(), 1); // Linear circuit has one basic block
//! ```

use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::gate::control_flow::ControlFlow;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::{Circuit, CircuitError, IfElseGate, WhileLoopGate};
use crate::circuit::{ConditionView, Operation, Parameter, Qubit};
use indexmap::IndexSet;
use rustworkx_core::petgraph::prelude::{EdgeIndex, NodeIndex, StableDiGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

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
    /// It must also contain a matching [`ControlFlowRegion`] owned by this block.
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

/// Explicit structured-control-flow information owned by a branch block.
///
/// The region identifies how a conditional branch must be reconstructed and
/// stores the outer [`Operation`] fields that are not represented by the
/// contained basic blocks. Basic-block labels are intentionally excluded from
/// this structure: labels are diagnostic text only.
#[derive(Debug, Clone)]
pub enum ControlFlowRegion {
    /// A structured `if`/`else` region whose branches converge at `merge_block`.
    IfElse {
        /// Entry block for the true branch.
        true_entry: NodeIndex,
        /// Entry block for the false branch.
        false_entry: NodeIndex,
        /// Continuation block reached after either branch.
        merge_block: NodeIndex,
        /// Whether the source operation explicitly contained an `else` body.
        has_else: bool,
        /// Outer operation qubit list, preserved exactly for round-trip conversion.
        qubits: SmallVec<[Qubit; 3]>,
        /// Outer operation parameters, preserved exactly for round-trip conversion.
        params: SmallVec<[CircuitParam; 1]>,
        /// Outer operation label, preserved exactly for round-trip conversion.
        label: Option<Box<str>>,
    },
    /// A structured `while` region whose true branch is the loop body.
    WhileLoop {
        /// Entry block for the loop body.
        body_entry: NodeIndex,
        /// Continuation block reached when the loop condition is false.
        exit_block: NodeIndex,
        /// Outer operation qubit list, preserved exactly for round-trip conversion.
        qubits: SmallVec<[Qubit; 3]>,
        /// Outer operation parameters, preserved exactly for round-trip conversion.
        params: SmallVec<[CircuitParam; 1]>,
        /// Outer operation label, preserved exactly for round-trip conversion.
        label: Option<Box<str>>,
    },
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

    pub fn terminator(&self) -> Option<&Terminator> {
        self.terminator.as_ref()
    }

    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
}

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Control Flow Graph (CFG) representation of a quantum circuit.
///
/// `CircuitCFG` represents a quantum circuit as a directed graph of basic blocks,
/// enabling efficient analysis and transformation of circuits with classical
/// control flow (conditionals and loops).
///
/// # Structure
///
/// ```text
/// +-------------------------------------------------+
/// |                  CircuitCFG                     |
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
/// let cfg = CircuitCFG::from_circuit(&circuit).unwrap();
/// assert_eq!(cfg.num_qubits(), 2);
/// assert_eq!(cfg.num_blocks(), 1); // Linear circuit
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

    /// Structured regions keyed by the block whose terminator branches on the condition.
    ///
    /// A `Branch` terminator is complete only when it has an entry here.
    pub(crate) control_flow_regions: HashMap<NodeIndex, ControlFlowRegion>,
}

impl CircuitCFG {
    /// Creates a new incomplete `CircuitCFG` with the specified number of qubits.
    ///
    /// Blocks, an entry block, and complete terminators must be supplied before
    /// this value can be converted to a [`Circuit`].
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
    /// let cfg = CircuitCFG::new(3);
    /// assert_eq!(cfg.num_qubits(), 3);
    /// assert_eq!(cfg.num_blocks(), 0);
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
            control_flow_regions: HashMap::new(),
        }
    }

    /// Creates an incomplete `CircuitCFG` from an existing vector of qubits.
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
            control_flow_regions: HashMap::new(),
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
        if self.data.node_weight(source).is_none() || self.data.node_weight(target).is_none() {
            return None;
        }
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

    /// Defines the structured construct represented by a branch block.
    ///
    /// A branch block without a region remains an incomplete CFG and is
    /// rejected by [`Self::validate`] and [`Self::to_circuit`].
    pub fn set_control_flow_region(&mut self, branch_block: NodeIndex, region: ControlFlowRegion) {
        self.control_flow_regions.insert(branch_block, region);
    }

    /// Returns the structured construct represented by a branch block, if any.
    pub fn control_flow_region(&self, branch_block: NodeIndex) -> Option<&ControlFlowRegion> {
        self.control_flow_regions.get(&branch_block)
    }

    /// Returns whether `block` is the header of a structured while-loop region.
    pub fn is_loop_header(&self, block: NodeIndex) -> bool {
        matches!(
            self.control_flow_region(block),
            Some(ControlFlowRegion::WhileLoop { .. })
        )
    }

    /// Returns an iterator over all basic blocks in the CFG.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(NodeIndex, &BasicBlock)` tuples.
    pub fn blocks(&self) -> impl Iterator<Item = (NodeIndex, &BasicBlock)> {
        self.data.node_indices().map(|i| (i, &self.data[i]))
    }

    pub fn block_mut(&mut self, index: NodeIndex) -> Option<&mut BasicBlock> {
        self.data.node_weight_mut(index)
    }

    pub fn outgoing_edges(
        &self,
        source: NodeIndex,
    ) -> impl Iterator<Item = (NodeIndex, FlowEdge)> + '_ {
        self.data
            .edges(source)
            .map(|edge| (edge.target(), edge.weight().clone()))
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
    /// `Ok(CircuitCFG)` on success, or a `CircuitError` if conversion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::{Circuit, CircuitCFG, Qubit};
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.h(Qubit::new(0)).unwrap();
    ///
    /// let cfg = CircuitCFG::from_circuit(&circuit).unwrap();
    /// assert_eq!(cfg.num_blocks(), 1);
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
            dag.validate()?;
            return Ok(dag);
        }

        // 2. Process all operations linearly, tracking the final block
        let final_block = process_operations(circuit.operations(), &mut dag, entry_idx)?;

        // 3. Set Return terminator on the final block if not already terminated
        if !dag.data[final_block].has_terminator() {
            dag.data[final_block].set_terminator(Terminator::Return);
        }

        dag.validate()?;
        Ok(dag)
    }

    /// Validates that this CFG represents a complete structured circuit.
    ///
    /// Validation rejects incomplete terminators, inconsistent edges, branches
    /// without explicit structured-region information, malformed region
    /// boundaries, invalid qubit or parameter references, unreachable blocks,
    /// and non-structured cycles.
    pub fn validate(&self) -> Result<(), CircuitError> {
        let entry = self.entry_block.ok_or_else(|| {
            CircuitError::InvalidControlFlow("CFG does not define an entry block".to_string())
        })?;
        self.require_block(entry, "entry block")?;

        self.validate_param(&self.global_phase, "global phase")?;

        for (node, block) in self.blocks() {
            for (index, operation) in block.operations.iter().enumerate() {
                self.validate_operation(
                    operation,
                    &format!("block {:?} operation {}", node, index),
                )?;
                if matches!(operation.instruction, Instruction::ControlFlowGate(_)) {
                    return Err(CircuitError::InvalidControlFlow(format!(
                        "block {:?} contains an unexpanded control-flow operation",
                        node
                    )));
                }
            }

            let terminator = block.terminator.as_ref().ok_or_else(|| {
                CircuitError::InvalidControlFlow(format!(
                    "Block '{}' (index {:?}) is missing a terminator",
                    block.label().unwrap_or("<unlabeled>"),
                    node
                ))
            })?;
            let outgoing: Vec<_> = self.data.edges(node).collect();

            match terminator {
                Terminator::Return => {
                    if !outgoing.is_empty() {
                        return Err(CircuitError::InvalidControlFlow(format!(
                            "Block '{}' (index {:?}) has a Return terminator but outgoing edges exist",
                            block.label().unwrap_or("<unlabeled>"),
                            node
                        )));
                    }
                }
                Terminator::Jump(target) => {
                    self.require_block(*target, "jump target")?;
                    if outgoing.len() != 1
                        || !matches!(outgoing[0].weight(), FlowEdge::Unconditional)
                        || outgoing[0].target() != *target
                    {
                        return Err(CircuitError::InvalidControlFlow(format!(
                            "Block '{}' (index {:?}) has an invalid Jump edge; expected one Unconditional edge to {:?}",
                            block.label().unwrap_or("<unlabeled>"),
                            node,
                            target
                        )));
                    }
                }
                Terminator::Branch(condition) => {
                    if !self.qubits.contains(&condition.qubit) {
                        return Err(CircuitError::InvalidControlFlow(format!(
                            "Branch condition in block {:?} references unknown qubit {}",
                            node,
                            condition.qubit.id()
                        )));
                    }
                    let (true_target, false_target) = self.branch_targets(node, block)?;
                    let region = self.control_flow_region(node).ok_or_else(|| {
                        CircuitError::InvalidControlFlow(format!(
                            "Block '{}' (index {:?}) has a Branch terminator but no structured control-flow region",
                            block.label().unwrap_or("<unlabeled>"),
                            node
                        ))
                    })?;

                    match region {
                        ControlFlowRegion::IfElse {
                            true_entry,
                            false_entry,
                            merge_block,
                            qubits,
                            params,
                            ..
                        } => {
                            if *true_entry != true_target || *false_entry != false_target {
                                return Err(CircuitError::InvalidControlFlow(format!(
                                    "IfElse region at block {:?} does not match its branch edges",
                                    node
                                )));
                            }
                            self.require_block(*merge_block, "if_else merge block")?;
                            self.validate_outer_fields(qubits, params, node)?;
                        }
                        ControlFlowRegion::WhileLoop {
                            body_entry,
                            exit_block,
                            qubits,
                            params,
                            ..
                        } => {
                            if *body_entry != true_target || *exit_block != false_target {
                                return Err(CircuitError::InvalidControlFlow(format!(
                                    "WhileLoop region at block {:?} does not match its branch edges",
                                    node
                                )));
                            }
                            self.validate_outer_fields(qubits, params, node)?;
                        }
                    }
                }
            }
        }

        for region_node in self.control_flow_regions.keys() {
            self.require_block(*region_node, "structured region owner")?;
            if !matches!(
                self.data[*region_node].terminator,
                Some(Terminator::Branch(_))
            ) {
                return Err(CircuitError::InvalidControlFlow(format!(
                    "structured control-flow region owner {:?} is not a Branch block",
                    region_node
                )));
            }
        }

        let mut visited = HashSet::new();
        self.parse_subgraph(entry, None, &mut visited)?;
        if visited.len() != self.num_blocks() {
            return Err(CircuitError::InvalidControlFlow(format!(
                "CFG contains {} unreachable or unconsumed block(s)",
                self.num_blocks() - visited.len()
            )));
        }

        Ok(())
    }

    /// Converts this CFG back to a nested `Circuit` representation.
    ///
    /// The conversion preserves operation metadata and is defined only for a
    /// complete structured CFG. Labels on basic blocks are never used to infer
    /// control-flow semantics.
    pub fn to_circuit(&self) -> Result<Circuit, CircuitError> {
        self.validate()?;
        let entry = self
            .entry_block
            .expect("validated CFG must have an entry block");
        let mut visited = HashSet::new();
        let ops = self.parse_subgraph(entry, None, &mut visited)?;

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
        visited: &mut HashSet<NodeIndex>,
    ) -> Result<Vec<Operation>, CircuitError> {
        let mut ops = Vec::new();
        let mut current = Some(start_node);

        while let Some(node) = current {
            if Some(node) == stop_node {
                return Ok(ops);
            }

            if !visited.insert(node) {
                return Err(CircuitError::InvalidControlFlow(format!(
                    "control flow visits block {:?} more than once outside a structured loop boundary",
                    node
                )));
            }

            let block = &self.data[node];
            ops.extend(block.operations.clone());

            match &block.terminator {
                Some(Terminator::Return) => {
                    if stop_node.is_some() {
                        return Err(CircuitError::InvalidControlFlow(format!(
                            "structured region terminates at block {:?} before reaching its boundary",
                            node
                        )));
                    }
                    return Ok(ops);
                }
                Some(Terminator::Jump(target)) => {
                    current = Some(*target);
                }
                Some(Terminator::Branch(condition)) => {
                    match self.control_flow_regions.get(&node).ok_or_else(|| {
                        CircuitError::InvalidControlFlow(format!(
                            "Branch block {:?} is missing structured-region information",
                            node
                        ))
                    })? {
                        ControlFlowRegion::WhileLoop {
                            body_entry,
                            exit_block,
                            qubits,
                            params,
                            label,
                        } => {
                            let body_ops = self.parse_subgraph(*body_entry, Some(node), visited)?;
                            ops.push(Operation {
                                instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                                    WhileLoopGate::new(*condition, body_ops),
                                )),
                                qubits: qubits.clone(),
                                params: params.clone(),
                                label: label.clone(),
                            });
                            current = Some(*exit_block);
                        }
                        ControlFlowRegion::IfElse {
                            true_entry,
                            false_entry,
                            merge_block,
                            has_else,
                            qubits,
                            params,
                            label,
                        } => {
                            let true_ops =
                                self.parse_subgraph(*true_entry, Some(*merge_block), visited)?;
                            let false_ops =
                                self.parse_subgraph(*false_entry, Some(*merge_block), visited)?;
                            ops.push(Operation {
                                instruction: Instruction::ControlFlowGate(ControlFlow::IfElse(
                                    IfElseGate::new(
                                        *condition,
                                        true_ops,
                                        has_else.then_some(false_ops),
                                    ),
                                )),
                                qubits: qubits.clone(),
                                params: params.clone(),
                                label: label.clone(),
                            });
                            current = Some(*merge_block);
                        }
                    }
                }
                None => unreachable!("validation rejects unterminated blocks"),
            }
        }

        Err(CircuitError::InvalidControlFlow(
            "structured control-flow traversal ended without a Return terminator".to_string(),
        ))
    }

    fn validate_operation(&self, operation: &Operation, context: &str) -> Result<(), CircuitError> {
        for qubit in &operation.qubits {
            if !self.qubits.contains(qubit) {
                return Err(CircuitError::InvalidControlFlow(format!(
                    "{} references unknown qubit {}",
                    context,
                    qubit.id()
                )));
            }
        }
        for parameter in &operation.params {
            self.validate_param(parameter, context)?;
        }
        Ok(())
    }

    fn validate_outer_fields(
        &self,
        qubits: &[Qubit],
        params: &[CircuitParam],
        node: NodeIndex,
    ) -> Result<(), CircuitError> {
        for qubit in qubits {
            if !self.qubits.contains(qubit) {
                return Err(CircuitError::InvalidControlFlow(format!(
                    "control-flow operation at block {:?} references unknown qubit {}",
                    node,
                    qubit.id()
                )));
            }
        }
        for param in params {
            self.validate_param(
                param,
                &format!("control-flow operation at block {:?}", node),
            )?;
        }
        Ok(())
    }

    fn validate_param(&self, parameter: &CircuitParam, context: &str) -> Result<(), CircuitError> {
        if let CircuitParam::Index(index) = parameter {
            if self.parameters.get_index(*index as usize).is_none() {
                return Err(CircuitError::InvalidControlFlow(format!(
                    "{} references missing parameter index {}",
                    context, index
                )));
            }
        }
        Ok(())
    }

    fn require_block(&self, node: NodeIndex, context: &str) -> Result<(), CircuitError> {
        if self.data.node_weight(node).is_none() {
            return Err(CircuitError::InvalidControlFlow(format!(
                "{} {:?} does not exist in the CFG",
                context, node
            )));
        }
        Ok(())
    }

    fn branch_targets(
        &self,
        node: NodeIndex,
        block: &BasicBlock,
    ) -> Result<(NodeIndex, NodeIndex), CircuitError> {
        let outgoing: Vec<_> = self.data.edges(node).collect();
        let true_targets: Vec<_> = outgoing
            .iter()
            .filter(|edge| matches!(edge.weight(), FlowEdge::TrueBranch))
            .map(|edge| edge.target())
            .collect();
        let false_targets: Vec<_> = outgoing
            .iter()
            .filter(|edge| matches!(edge.weight(), FlowEdge::FalseBranch))
            .map(|edge| edge.target())
            .collect();
        let label = block.label().unwrap_or("<unlabeled>");

        if true_targets.is_empty() {
            return Err(CircuitError::InvalidControlFlow(format!(
                "Block '{}' (index {:?}) has a Branch terminator but is missing a TrueBranch outgoing edge",
                label, node
            )));
        }
        if false_targets.is_empty() {
            return Err(CircuitError::InvalidControlFlow(format!(
                "Block '{}' (index {:?}) has a Branch terminator but is missing a FalseBranch outgoing edge",
                label, node
            )));
        }
        if true_targets.len() != 1 || false_targets.len() != 1 || outgoing.len() != 2 {
            return Err(CircuitError::InvalidControlFlow(format!(
                "Block '{}' (index {:?}) must have exactly one TrueBranch edge and one FalseBranch edge",
                label, node
            )));
        }

        Ok((true_targets[0], false_targets[0]))
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

                dag.set_control_flow_region(
                    current_block,
                    ControlFlowRegion::IfElse {
                        true_entry,
                        false_entry,
                        merge_block,
                        has_else: if_else.false_body().is_some(),
                        qubits: op.qubits.clone(),
                        params: op.params.clone(),
                        label: op.label.clone(),
                    },
                );

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

                dag.set_control_flow_region(
                    cond_block,
                    ControlFlowRegion::WhileLoop {
                        body_entry,
                        exit_block,
                        qubits: op.qubits.clone(),
                        params: op.params.clone(),
                        label: op.label.clone(),
                    },
                );

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
