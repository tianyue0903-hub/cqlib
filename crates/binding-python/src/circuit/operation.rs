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

//! Python Bindings for Quantum Circuit Operations
//!
//! This module provides Python bindings for [`Operation`] from cqlib-core.
//! It represents a single, concrete execution step within a quantum circuit.
//!
//! # Key Components
//!
//! - [`PyOperation`]: A fully resolved operation binding a gate to specific qubits and parameters.
//! - [`PyOperationIter`]: An iterator over operations in a circuit.
//!
//! # Operation vs Instruction
//!
//! While [`Instruction`](cqlib_core::circuit::gate::instruction::Instruction) defines *what* to do
//! (e.g., "apply a Hadamard gate"), [`PyOperation`] defines *where* and *how* to do it.
//! An operation binds an instruction to specific qubits and concrete parameter values.

use crate::circuit::bit::PyQubit;
use crate::circuit::instruction::PyInstruction;
use cqlib_core::circuit::operation::Operation;
use cqlib_core::circuit::param::CircuitParam;
use numpy::ToPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use std::sync::Arc;

/// Python wrapper for `Operation`.
///
/// Represents a fully resolved operation in a quantum circuit.
/// Each operation binds a gate (instruction) to specific qubits and parameters.
///
/// # Examples
///
/// ```python
/// from cqlib import Circuit
///
/// circuit = Circuit(2)
/// circuit.h(0)
/// circuit.cx(0, 1)
/// circuit.rx(0, 0.5)
///
/// for op in circuit.operations():
///     print(f"Gate: {op.name}, Qubits: {op.num_qubits}")
/// ```
#[pyclass(name = "Operation", module = "cqlib.circuit")]
#[derive(Debug, Clone)]
pub struct PyOperation {
    pub(crate) operation: Operation,
}

impl From<Operation> for PyOperation {
    fn from(operation: Operation) -> Self {
        Self { operation }
    }
}

impl From<PyOperation> for Operation {
    fn from(py: PyOperation) -> Self {
        py.operation
    }
}

#[pymethods]
impl PyOperation {
    /// Returns the instruction (gate type) of this operation.
    ///
    /// The instruction defines what type of gate or operation to apply.
    #[getter]
    fn instruction(&self) -> PyInstruction {
        PyInstruction::from(self.operation.instruction.clone())
    }

    /// Returns the qubits this operation acts on.
    ///
    /// For controlled gates, control qubits usually come first, followed by target qubits.
    #[getter]
    fn qubits(&self) -> Vec<PyQubit> {
        self.operation
            .qubits
            .iter()
            .map(|&q| PyQubit::from(q))
            .collect()
    }

    /// Returns the number of qubits this operation acts on.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.operation.qubits.len()
    }

    /// Returns the parameters of this operation.
    ///
    /// Parameters can be either:
    /// - Fixed float values (e.g., `0.5` for rotation angle)
    /// - Symbolic parameter indices (returned as tuple `("param", index)`)
    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut result = Vec::with_capacity(self.operation.params.len());
        for param in &self.operation.params {
            match param {
                CircuitParam::Fixed(val) => {
                    result.push(val.into_pyobject(py)?.into_any().unbind());
                }
                CircuitParam::Index(idx) => {
                    // For now, return the index as a tuple ("param", idx)
                    // In a full implementation, we'd need access to the circuit's parameter table
                    let tuple = ("param", *idx).into_pyobject(py)?;
                    result.push(tuple.into_any().unbind());
                }
            }
        }
        Ok(result)
    }

    /// Returns the unitary matrix representation of this operation.
    ///
    /// # Returns
    ///
    /// A 2D numpy array (dtype=complex128) representing the unitary matrix.
    ///
    /// # Raises
    ///
    /// RuntimeError if the operation is non-unitary (e.g., Measure, Barrier).
    fn matrix(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let matrix_cow = self
            .operation
            .matrix()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;
        matrix_cow.to_pyarray(py).into_py_any(py)
    }

    /// Returns the number of parameters.
    #[getter]
    fn num_params(&self) -> usize {
        self.operation.params.len()
    }

    /// Returns the label of this operation, if any.
    #[getter]
    fn label(&self) -> Option<String> {
        self.operation.label.as_ref().map(|s| s.to_string())
    }

    /// Returns the name of the instruction.
    ///
    /// Examples: "h", "cx", "rx", "measure"
    #[getter]
    fn name(&self) -> String {
        format!("{}", self.operation.instruction)
    }

    fn __repr__(&self) -> String {
        format!(
            "Operation({}, qubits={:?}, params={})",
            self.name(),
            self.operation
                .qubits
                .iter()
                .map(|q| q.index())
                .collect::<Vec<_>>(),
            self.operation.params.len()
        )
    }

    fn __str__(&self) -> String {
        format!("{}", self.operation.instruction)
    }
}

/// Iterator over operations in a circuit.
#[pyclass]
pub struct PyOperationIter {
    ops: Arc<Vec<Operation>>,
    index: usize,
}

#[pymethods]
impl PyOperationIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<Self> {
        slf.into()
    }

    fn __next__(&mut self) -> Option<PyOperation> {
        self.ops.get(self.index).map(|op| {
            self.index += 1;
            PyOperation::from(op.clone())
        })
    }

    fn __len__(&self) -> usize {
        self.ops.len().saturating_sub(self.index)
    }
}

impl PyOperationIter {
    /// Creates a new operation iterator.
    pub fn new(ops: Vec<Operation>, index: usize) -> Self {
        Self {
            ops: Arc::new(ops),
            index,
        }
    }
}
