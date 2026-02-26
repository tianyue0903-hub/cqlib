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

//! # Control Flow Operations
//!
//! This module provides control flow operations for quantum circuits, enabling
//! conditional and iterative quantum computation based on classical conditions.
//!
//! ## Overview
//!
//! - [`IfElseGate`]: Conditional execution (if-else) based on classical measurement outcomes
//! - [`WhileLoopGate`]: Iterative execution (while loop) based on classical conditions
//! - [`ConditionView`]: Representation of classical conditions linking qubits to values
//!
//! ## Usage
//!
//! Control flow operations allow quantum circuits to make decisions based on
//! measurement results, enabling:
//! - Quantum error correction
//! - Quantum feedback loops
//! - Adaptive quantum algorithms
//!
//! ## Important Notes
//!
//! These operations are **not unitary** and cannot be represented as a single unitary matrix.
//! They require special handling during circuit execution and simulation:
//! - Cannot be converted to a matrix representation
//! - Require runtime interpretation/execution
//! - May not be supported by all backends

use crate::circuit::operation::Operation;
use crate::circuit::{Parameter, Qubit};
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

/// Classical condition representing a measurement outcome condition.
///
/// This struct defines a condition based on the most recent measurement result
/// of a specific qubit. It is used in control flow operations to determine
/// which branch to execute.
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::gate::control_flow::ConditionView;
/// use cqlib_core::circuit::Qubit;
///
/// // Condition: when measurement result of qubit 0 equals 1
/// let condition = ConditionView::new(Qubit::new(0), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConditionView {
    /// The qubit ID to check
    pub qubit: Qubit,
    /// The target classical value (typically 0 or 1)
    pub target: u8,
}

impl ConditionView {
    /// Creates a new condition view.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The qubit whose measurement result to check
    /// * `target` - The target value to compare against (typically 0 or 1)
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::gate::control_flow::ConditionView;
    /// use cqlib_core::circuit::Qubit;
    ///
    /// // Trigger when qubit 0 measurement result equals 1
    /// let condition = ConditionView::new(Qubit::new(0), 1);
    /// ```
    pub fn new(qubit: Qubit, target: u8) -> Self {
        Self { qubit, target }
    }
}

/// If-Else gate for conditional quantum operation execution.
///
/// This gate executes different quantum operations based on a classical condition.
/// It contains a true branch that always executes when the condition is met,
/// and optionally a false branch for when the condition is not met.
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::gate::control_flow::{ConditionView, IfElseGate};
/// use cqlib_core::circuit::operation::Operation;
/// use cqlib_core::circuit::gate::{Instruction, StandardGate};
/// use cqlib_core::circuit::Qubit;
/// use smallvec::smallvec;
///
/// let condition = ConditionView::new(Qubit::new(0), 1);
/// let true_body = vec![Operation {
///     instruction: Instruction::Standard(StandardGate::X),
///     qubits: smallvec![Qubit::new(1)],
///     params: smallvec![],
///     label: None,
/// }];
/// let gate = IfElseGate::new(condition, true_body, None);
/// ```
#[derive(Debug, Clone)]
pub struct IfElseGate {
    condition: ConditionView,
    true_body: Arc<Vec<Operation>>,
    false_body: Option<Arc<Vec<Operation>>>,
}

impl IfElseGate {
    /// Creates a new if-else gate.
    ///
    /// # Arguments
    ///
    /// * `condition` - The classical condition to evaluate
    /// * `true_body` - Operations to execute when condition is true
    /// * `false_body` - Optional operations to execute when condition is false
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::gate::control_flow::{ConditionView, IfElseGate};
    /// use cqlib_core::circuit::operation::Operation;
    /// use cqlib_core::circuit::gate::{Instruction, StandardGate};
    /// use cqlib_core::circuit::Qubit;
    /// use smallvec::smallvec;
    ///
    /// let condition = ConditionView::new(Qubit::new(0), 1);
    /// let true_body = vec![Operation {
    ///     instruction: Instruction::Standard(StandardGate::X),
    ///     qubits: smallvec![Qubit::new(1)],
    ///     params: smallvec![],
    ///     label: None,
    /// }];
    /// let false_body = vec![Operation {
    ///     instruction: Instruction::Standard(StandardGate::Z),
    ///     qubits: smallvec![Qubit::new(1)],
    ///     params: smallvec![],
    ///     label: None,
    /// }];
    /// let gate = IfElseGate::new(condition, true_body, Some(false_body));
    /// ```
    pub fn new(
        condition: ConditionView,
        true_body: Vec<Operation>,
        false_body: Option<Vec<Operation>>,
    ) -> Self {
        Self {
            condition,
            true_body: Arc::new(true_body),
            false_body: false_body.map(Arc::new),
        }
    }

    /// Returns the condition for this gate.
    pub fn condition(&self) -> ConditionView {
        self.condition
    }

    /// Returns the operations to execute when condition is true.
    pub fn true_body(&self) -> &[Operation] {
        &self.true_body
    }

    /// Returns the operations to execute when condition is false, if any.
    pub fn false_body(&self) -> Option<&[Operation]> {
        self.false_body.as_ref().map(|v| v.as_slice())
    }

    /// Returns the number of qubits used in this gate.
    ///
    /// This counts all unique qubits referenced in both true and false branches,
    /// plus the condition qubit.
    pub fn num_qubits(&self) -> usize {
        let mut qubits: HashSet<Qubit> = HashSet::new();
        // Include the condition qubit
        qubits.insert(self.condition.qubit);
        for op in self.true_body.iter() {
            qubits.extend(op.qubits.iter());
        }
        if let Some(false_body) = &self.false_body {
            for op in false_body.iter() {
                qubits.extend(op.qubits.iter());
            }
        }
        qubits.len()
    }

    /// Returns the number of parameters.
    ///
    /// Control flow gates themselves have no parameters;
    /// parameters are determined by the operations in their bodies.
    pub fn num_params(&self) -> usize {
        0
    }

    /// Returns the inverse of this gate.
    ///
    /// # Returns
    ///
    /// `None` - Control flow inverse cannot be computed statically at compile time.
    /// It must be computed at runtime based on the execution state.
    pub fn inverse(&self) -> Option<(Self, SmallVec<[Parameter; 3]>)> {
        None
    }
}

/// While-loop gate for iterative quantum operation execution.
///
/// This gate repeatedly executes quantum operations while a classical condition
/// remains true.
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::gate::control_flow::{ConditionView, WhileLoopGate};
/// use cqlib_core::circuit::operation::Operation;
/// use cqlib_core::circuit::gate::{Instruction, StandardGate};
/// use cqlib_core::circuit::Qubit;
/// use smallvec::smallvec;
///
/// let condition = ConditionView::new(Qubit::new(0), 1);
/// let body = vec![Operation {
///     instruction: Instruction::Standard(StandardGate::H),
///     qubits: smallvec![Qubit::new(1)],
///     params: smallvec![],
///     label: None,
/// }];
/// let gate = WhileLoopGate::new(condition, body);
/// ```
#[derive(Debug, Clone)]
pub struct WhileLoopGate {
    condition: ConditionView,
    body: Arc<Vec<Operation>>,
}

impl WhileLoopGate {
    /// Creates a new while-loop gate.
    ///
    /// # Arguments
    ///
    /// * `condition` - The classical condition to evaluate before each iteration
    /// * `body` - The operations to execute in each iteration
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::gate::control_flow::{ConditionView, WhileLoopGate};
    /// use cqlib_core::circuit::operation::Operation;
    /// use cqlib_core::circuit::gate::{Instruction, StandardGate};
    /// use cqlib_core::circuit::Qubit;
    /// use smallvec::smallvec;
    ///
    /// let condition = ConditionView::new(Qubit::new(0), 1);
    /// let body = vec![Operation {
    ///     instruction: Instruction::Standard(StandardGate::H),
    ///     qubits: smallvec![Qubit::new(1)],
    ///     params: smallvec![],
    ///     label: None,
    /// }];
    /// let gate = WhileLoopGate::new(condition, body);
    /// ```
    pub fn new(condition: ConditionView, body: Vec<Operation>) -> Self {
        Self {
            condition,
            body: Arc::new(body),
        }
    }

    /// Returns the condition for this loop.
    pub fn condition(&self) -> ConditionView {
        self.condition
    }

    /// Returns the loop body operations.
    pub fn body(&self) -> &[Operation] {
        &self.body
    }

    /// Returns the number of qubits used in this gate.
    ///
    /// This counts all unique qubits referenced in the loop body,
    /// plus the condition qubit.
    pub fn num_qubits(&self) -> usize {
        let mut qubits: HashSet<Qubit> = HashSet::new();
        // Include the condition qubit
        qubits.insert(self.condition.qubit);
        for op in self.body.iter() {
            qubits.extend(op.qubits.iter());
        }
        qubits.len()
    }

    /// Returns the number of parameters.
    ///
    /// Control flow gates themselves have no parameters;
    /// parameters are determined by the operations in their bodies.
    pub fn num_params(&self) -> usize {
        0
    }

    /// Returns the inverse of this gate.
    ///
    /// # Returns
    ///
    /// `None` - Control flow inverse cannot be computed statically at compile time.
    /// It must be computed at runtime based on the execution state.
    pub fn inverse(&self) -> Option<(Self, SmallVec<[Parameter; 3]>)> {
        None
    }
}

/// Control flow operations for quantum circuits.
///
/// This enum wraps different types of control flow operations including
/// conditional execution (if-else) and iterative execution (while loops).
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::gate::control_flow::{ConditionView, ControlFlow};
/// use cqlib_core::circuit::operation::Operation;
/// use cqlib_core::circuit::gate::{Instruction, StandardGate};
/// use cqlib_core::circuit::Qubit;
/// use smallvec::smallvec;
///
/// let condition = ConditionView::new(Qubit::new(0), 1);
/// let true_body = vec![Operation {
///     instruction: Instruction::Standard(StandardGate::X),
///     qubits: smallvec![Qubit::new(1)],
///     params: smallvec![],
///     label: None,
/// }];
///
/// // Create an if-else control flow
/// let flow = ControlFlow::if_else(condition, true_body, None);
/// ```
#[derive(Debug, Clone)]
pub enum ControlFlow {
    /// Conditional execution (if-else)
    IfElse(IfElseGate),
    /// Iterative execution (while loop)
    WhileLoop(WhileLoopGate),
}

impl ControlFlow {
    /// Creates a new if-else control flow.
    ///
    /// # Arguments
    ///
    /// * `condition` - The classical condition to evaluate
    /// * `true_body` - Operations to execute when condition is true
    /// * `false_body` - Optional operations to execute when condition is false
    pub fn if_else(
        condition: ConditionView,
        true_body: Vec<Operation>,
        false_body: Option<Vec<Operation>>,
    ) -> Self {
        Self::IfElse(IfElseGate::new(condition, true_body, false_body))
    }

    /// Creates a new while-loop control flow.
    ///
    /// # Arguments
    ///
    /// * `condition` - The classical condition to evaluate before each iteration
    /// * `body` - The operations to execute in each iteration
    pub fn while_loop(condition: ConditionView, body: Vec<Operation>) -> Self {
        Self::WhileLoop(WhileLoopGate::new(condition, body))
    }

    /// Returns the number of qubits used in this control flow.
    pub fn num_qubits(&self) -> usize {
        match self {
            Self::IfElse(gate) => gate.num_qubits(),
            Self::WhileLoop(gate) => gate.num_qubits(),
        }
    }

    /// Returns the number of parameters.
    pub fn num_params(&self) -> usize {
        match self {
            Self::IfElse(gate) => gate.num_params(),
            Self::WhileLoop(gate) => gate.num_params(),
        }
    }

    /// Returns the unitary matrix representation.
    ///
    /// # Returns
    ///
    /// `None` - Control flow operations are not unitary and cannot be
    /// represented as a matrix. They require runtime interpretation.
    pub fn matrix(&self) -> Option<Cow<'_, Array2<Complex<f64>>>> {
        None
    }

    /// Returns the inverse of this control flow.
    ///
    /// # Returns
    ///
    /// `None` - Control flow inverse cannot be computed statically.
    /// It must be computed at runtime based on execution state.
    pub fn inverse(&self) -> Option<(Self, SmallVec<[Parameter; 3]>)> {
        match self {
            Self::IfElse(gate) => gate.inverse().map(|(g, params)| (Self::IfElse(g), params)),
            Self::WhileLoop(gate) => gate
                .inverse()
                .map(|(g, params)| (Self::WhileLoop(g), params)),
        }
    }
}

impl fmt::Display for ControlFlow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IfElse(_) => write!(f, "if_else"),
            Self::WhileLoop(_) => write!(f, "while_loop"),
        }
    }
}
