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

//! # Visualization IR Data Model
//!
//! This module defines backend-agnostic intermediate representation (IR) types used by
//! visualization backends such as text and figure drawers.
//!
//! ## Design Goals
//!
//! - Decouple circuit semantics from rendering implementation.
//! - Preserve lane/column layout decisions for reuse across backends.
//! - Carry enough metadata (style/children/span) for control-flow-aware drawing.

use crate::circuit::Qubit;

/// Draw style used by visualization backends.
///
/// Each variant determines how one [`VisualOperation`] should be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualOpStyle {
    /// Generic gate-like box.
    Gate,
    /// Controlled operation where the first `num_controls` operands are controls.
    Controlled { num_controls: usize },
    /// Controlled-Z marker rendered as two dots connected vertically.
    Cz,
    /// Swap marker across two qubits.
    Swap,
    /// Barrier marker.
    Barrier,
    /// Measurement marker.
    Measure,
    /// Reset marker.
    Reset,
    /// Delay marker.
    Delay,
    /// Control-flow marker (if/while).
    ControlFlow { kind: VisualControlFlowKind },
}

/// Control-flow operation family used by visualization backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualControlFlowKind {
    /// Source IfElse block before flattening.
    IfElseBlock {
        has_false_branch: bool,
        condition: VisualCondition,
    },
    /// Source While-loop block before flattening.
    WhileBlock { condition: VisualCondition },
    /// Flattened marker: `If ...`.
    IfStart,
    /// Flattened marker: `Else-...`.
    ElseStart,
    /// Flattened marker: `While ...`.
    WhileStart,
    /// Flattened marker: `End-...`.
    End,
}

/// Structured condition metadata used by control-flow visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisualCondition {
    /// Condition qubit id.
    pub qubit_id: usize,
    /// Target bit value (0/1).
    pub target: u8,
}

/// Backend-agnostic operation prepared for rendering.
///
/// This is the atomic IR node consumed by drawers.
#[derive(Debug, Clone)]
pub struct VisualOperation {
    /// Time column after layering.
    pub column: usize,
    /// Operand lanes (in original operand order).
    pub lanes: Vec<usize>,
    /// Lanes that reserve this column to avoid overlap.
    pub covered_lanes: Vec<usize>,
    /// Primary display label.
    pub label: String,
    /// Parameter labels, already formatted.
    pub params: Vec<String>,
    /// Rendering style.
    pub style: VisualOpStyle,
    /// If true, render this operation as a span box on multi-qubit lanes.
    pub span_box: bool,
    /// Optional child circuits for control-flow operations.
    pub children: Option<VisualChildren>,
    /// Number of logical columns reserved by this operation.
    pub span_cols: usize,
}

/// Backend-agnostic circuit after layout.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::Qubit;
/// use cqlib_core::visualization::{VisualCircuit, VisualOpStyle, VisualOperation};
///
/// let visual = VisualCircuit {
///     qubits: vec![Qubit::new(0)],
///     operations: vec![VisualOperation {
///         column: 0,
///         lanes: vec![0],
///         covered_lanes: vec![0],
///         label: "H".to_string(),
///         params: vec![],
///         style: VisualOpStyle::Gate,
///         span_box: false,
///         children: None,
///         span_cols: 1,
///     }],
///     num_columns: 1,
/// };
/// assert_eq!(visual.num_qubits(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct VisualCircuit {
    /// Logical qubits in lane order.
    pub qubits: Vec<Qubit>,
    /// Layered operations.
    pub operations: Vec<VisualOperation>,
    /// Number of occupied columns.
    pub num_columns: usize,
}

impl VisualCircuit {
    /// Number of qubit lanes.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }
}

/// Optional child circuits attached to control-flow operations.
///
/// These child circuits keep lane alignment with the parent visualization context.
#[derive(Debug, Clone)]
pub enum VisualChildren {
    /// IfElse branch children.
    IfElse {
        then_circuit: Box<VisualCircuit>,
        else_circuit: Option<Box<VisualCircuit>>,
    },
    /// While-loop body child.
    While { body_circuit: Box<VisualCircuit> },
}