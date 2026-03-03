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

//! Python Bindings for Control Flow Operations
//!
//! This module provides Python bindings for control flow operations from cqlib-core.
//! It enables conditional and iterative quantum computation based on classical conditions.
//!
//! # Key Components
//!
//! - [`PyConditionView`]: Classical condition linking qubits to values
//! - [`PyIfElseGate`]: Conditional execution based on measurement outcomes
//! - [`PyWhileLoopGate`]: Iterative execution based on classical conditions
//! - [`PyControlFlow`]: Wrapper enum for if-else and while-loop operations
//!
//! # Important Notes
//!
//! These operations are **not unitary** and cannot be represented as a single unitary matrix.

use crate::circuit::bit::PyQubit;
use crate::circuit::operation::PyOperation;
use cqlib_core::circuit::gate::control_flow::{
    ConditionView, ControlFlow, IfElseGate, WhileLoopGate,
};
use cqlib_core::circuit::operation::Operation;
use pyo3::prelude::*;

/// Python wrapper for `ConditionView`.
///
/// Represents a classical condition based on a measurement outcome.
/// Used in control flow operations to determine which branch to execute.
#[pyclass(name = "ConditionView", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone, Copy)]
pub struct PyConditionView {
    pub(crate) inner: ConditionView,
}

impl From<ConditionView> for PyConditionView {
    fn from(inner: ConditionView) -> Self {
        Self { inner }
    }
}

impl From<PyConditionView> for ConditionView {
    fn from(py: PyConditionView) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyConditionView {
    /// Creates a new condition view.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The qubit whose measurement result to check.
    /// * `target` - The target value to compare against (typically 0 or 1).
    ///
    /// # Example
    ///
    /// ```rust
    /// use cqlib::circuit::{ConditionView, Qubit};
    ///
    /// // Trigger when qubit 0 measurement result equals 1
    /// let condition = ConditionView::new(Qubit::new(0), 1);
    /// ```
    #[new]
    fn new(qubit: PyQubit, target: u8) -> Self {
        PyConditionView {
            inner: ConditionView::new(qubit.inner, target),
        }
    }

    #[getter]
    fn qubit(&self) -> PyQubit {
        PyQubit::from(self.inner.qubit)
    }

    #[getter]
    fn target(&self) -> u8 {
        self.inner.target
    }

    fn __repr__(&self) -> String {
        format!(
            "ConditionView(qubit={}, target={})",
            self.inner.qubit.index(),
            self.inner.target
        )
    }
}

/// Python wrapper for `IfElseGate`.
///
/// Represents conditional quantum operation execution based on a classical condition.
/// Contains a true branch that executes when condition is met, and optionally a false branch.
#[pyclass(name = "IfElseGate", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone)]
pub struct PyIfElseGate {
    pub(crate) inner: IfElseGate,
}

impl From<IfElseGate> for PyIfElseGate {
    fn from(inner: IfElseGate) -> Self {
        Self { inner }
    }
}

impl From<PyIfElseGate> for IfElseGate {
    fn from(py: PyIfElseGate) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyIfElseGate {
    /// Creates a new if-else gate.
    ///
    /// # Arguments
    ///
    /// * `condition` - The condition to evaluate.
    /// * `true_body` - Operations to execute when condition is true.
    /// * `false_body` - Optional operations to execute when condition is false.
    #[new]
    fn new(
        condition: PyConditionView,
        true_body: Vec<PyOperation>,
        false_body: Option<Vec<PyOperation>>,
    ) -> Self {
        let true_ops: Vec<Operation> = true_body.into_iter().map(|op| op.operation).collect();
        let false_ops: Option<Vec<Operation>> =
            false_body.map(|ops| ops.into_iter().map(|op| op.operation).collect());

        PyIfElseGate {
            inner: IfElseGate::new(condition.inner, true_ops, false_ops),
        }
    }

    /// Returns the condition for this gate.
    #[getter]
    fn condition(&self) -> PyConditionView {
        PyConditionView::from(self.inner.condition())
    }

    /// Returns the number of qubits used in this gate.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn __repr__(&self) -> String {
        format!(
            "IfElseGate(condition={:?})",
            PyConditionView::from(self.inner.condition())
        )
    }
}

/// Python wrapper for `WhileLoopGate`.
///
/// Represents iterative quantum operation execution while a classical condition remains true.
#[pyclass(name = "WhileLoopGate", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone)]
pub struct PyWhileLoopGate {
    pub(crate) inner: WhileLoopGate,
}

impl From<WhileLoopGate> for PyWhileLoopGate {
    fn from(inner: WhileLoopGate) -> Self {
        Self { inner }
    }
}

impl From<PyWhileLoopGate> for WhileLoopGate {
    fn from(py: PyWhileLoopGate) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyWhileLoopGate {
    /// Creates a new while-loop gate.
    ///
    /// # Arguments
    ///
    /// * `condition` - The condition to evaluate before each iteration.
    /// * `body` - The operations to execute in each iteration.
    #[new]
    fn new(condition: PyConditionView, body: Vec<PyOperation>) -> Self {
        let body_ops: Vec<Operation> = body.into_iter().map(|op| op.operation).collect();

        PyWhileLoopGate {
            inner: WhileLoopGate::new(condition.inner, body_ops),
        }
    }

    /// Returns the condition for this loop.
    #[getter]
    fn condition(&self) -> PyConditionView {
        PyConditionView::from(self.inner.condition())
    }

    /// Returns the number of qubits used in this gate.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn __repr__(&self) -> String {
        format!(
            "WhileLoopGate(condition={:?})",
            PyConditionView::from(self.inner.condition())
        )
    }
}

/// Python wrapper for `ControlFlow` enum.
///
/// Wraps different types of control flow operations:
/// - If-else conditional execution
/// - While-loop iterative execution
#[pyclass(name = "ControlFlow", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone)]
pub struct PyControlFlow {
    pub(crate) inner: ControlFlow,
}

impl From<ControlFlow> for PyControlFlow {
    fn from(inner: ControlFlow) -> Self {
        Self { inner }
    }
}

impl From<PyControlFlow> for ControlFlow {
    fn from(py: PyControlFlow) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyControlFlow {
    /// Creates a new if-else control flow.
    ///
    /// # Arguments
    ///
    /// * `condition` - The condition to evaluate.
    /// * `true_body` - Operations to execute when condition is true.
    /// * `false_body` - Optional operations to execute when condition is false.
    #[staticmethod]
    fn if_else(
        condition: PyConditionView,
        true_body: Vec<PyOperation>,
        false_body: Option<Vec<PyOperation>>,
    ) -> Self {
        let true_ops: Vec<Operation> = true_body.into_iter().map(|op| op.operation).collect();
        let false_ops: Option<Vec<Operation>> =
            false_body.map(|ops| ops.into_iter().map(|op| op.operation).collect());

        PyControlFlow {
            inner: ControlFlow::if_else(condition.inner, true_ops, false_ops),
        }
    }

    /// Creates a new while-loop control flow.
    ///
    /// # Arguments
    ///
    /// * `condition` - The condition to evaluate before each iteration.
    /// * `body` - The operations to execute in each iteration.
    #[staticmethod]
    fn while_loop(condition: PyConditionView, body: Vec<PyOperation>) -> Self {
        let body_ops: Vec<Operation> = body.into_iter().map(|op| op.operation).collect();

        PyControlFlow {
            inner: ControlFlow::while_loop(condition.inner, body_ops),
        }
    }

    /// Returns the number of qubits used in this control flow.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Returns true if this is an if-else control flow.
    #[getter]
    fn is_if_else(&self) -> bool {
        matches!(self.inner, ControlFlow::IfElse(_))
    }

    /// Returns true if this is a while-loop control flow.
    #[getter]
    fn is_while_loop(&self) -> bool {
        matches!(self.inner, ControlFlow::WhileLoop(_))
    }

    /// Returns the IfElseGate if this is an if-else, None otherwise.
    fn as_if_else(&self) -> Option<PyIfElseGate> {
        match &self.inner {
            ControlFlow::IfElse(gate) => Some(PyIfElseGate::from(gate.clone())),
            _ => None,
        }
    }

    /// Returns the WhileLoopGate if this is a while-loop, None otherwise.
    fn as_while_loop(&self) -> Option<PyWhileLoopGate> {
        match &self.inner {
            ControlFlow::WhileLoop(gate) => Some(PyWhileLoopGate::from(gate.clone())),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ControlFlow::IfElse(_) => "ControlFlow(if_else)".to_string(),
            ControlFlow::WhileLoop(_) => "ControlFlow(while_loop)".to_string(),
        }
    }
}
