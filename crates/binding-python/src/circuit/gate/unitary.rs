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

//! Python Bindings for Custom Unitary Gates
//!
//! This module provides Python bindings for the [`UnitaryGate`] from cqlib-core.
//! It allows users to define custom quantum gates via their unitary matrix
//! representation or circuit decomposition.
//!
//! # Key Components
//!
//! - [`PyUnitaryGate`]: The main class for creating and manipulating custom gates.

use crate::circuit::PyCircuit;
use cqlib_core::circuit::gate::UnitaryGate;
use cqlib_core::circuit::gate::circuit_gate::FrozenCircuit;
use num_complex::Complex64;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::{PyResult, Python, pyclass, pymethods};
use std::sync::Arc;

/// Python wrapper for `UnitaryGate`.
///
/// Represents a custom quantum gate defined by its unitary matrix or circuit.
/// Each gate has a unique identifier for equality and hashing.
#[pyclass(name = "UnitaryGate", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone)]
pub struct PyUnitaryGate {
    inner: UnitaryGate,
}

#[pymethods]
impl PyUnitaryGate {
    /// Creates a new unitary gate definition without a matrix.
    ///
    /// # Arguments
    ///
    /// * `label` - A descriptive name for the gate (e.g., "QFT", "Oracle").
    /// * `num_qubits` - The number of qubits the gate operates on.
    ///
    /// # Returns
    ///
    /// A new `UnitaryGate` with no matrix attached.
    #[new]
    pub fn new(label: String, num_qubits: u16) -> PyResult<Self> {
        Ok(Self {
            inner: UnitaryGate::new(label.as_ref(), num_qubits),
        })
    }

    /// Attaches a unitary matrix to the gate.
    ///
    /// The matrix must be a 2D array of shape (2^n, 2^n) where n is num_qubits.
    /// Accepts numpy arrays, lists, or any array-like input.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2D square matrix (numpy array or list of lists).
    ///
    /// # Returns
    ///
    /// A new gate with the matrix attached.
    #[pyo3(signature = (matrix))]
    pub fn with_matrix<'py>(&self, py: Python<'py>, matrix: Bound<'py, PyAny>) -> PyResult<Self> {
        let np = py.import("numpy")?;
        // Allow flexible input (list, int array, float array) by casting to complex128 via numpy
        let array_obj = np.call_method1("array", (matrix, "complex128"))?;

        let array: Bound<'py, PyArray2<Complex64>> = array_obj.cast_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Input could not be converted to a 2D complex numpy array.",
            )
        })?;

        let array = array.to_owned();
        let new_inner = self
            .inner
            .clone()
            .with_matrix(array.to_owned_array())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: new_inner })
    }

    /// Attaches a circuit representation to the gate.
    ///
    /// Allows the gate to be defined by its circuit decomposition,
    /// useful for inverse operations and optimization.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The circuit representing this gate.
    ///
    /// # Returns
    ///
    /// A new gate with the circuit attached.
    #[pyo3(signature = (circuit))]
    pub fn with_circuit(&self, circuit: PyCircuit) -> PyResult<Self> {
        // Convert PyCircuit to FrozenCircuit
        let frozen = FrozenCircuit::new(circuit.inner);
        // Create new UnitaryGate with the circuit attached
        let new_inner = self.inner.clone().with_circuit(Arc::new(frozen));
        Ok(Self { inner: new_inner })
    }

    /// Returns the label of the gate.
    #[getter]
    pub fn label(&self) -> String {
        self.inner.label().to_string()
    }

    /// Returns the number of qubits this gate acts on.
    #[getter]
    pub fn num_qubits(&self) -> u16 {
        self.inner.num_qubits()
    }

    /// Returns the unitary matrix as a NumPy array.
    ///
    /// # Returns
    ///
    /// A 2D numpy array (dtype=complex128).
    ///
    /// # Raises
    ///
    /// ValueError if no matrix was attached to the gate.
    pub fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        match self.inner.matrix() {
            Some(mat) => Ok(PyArray2::from_array(py, mat)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "No matrix defined for this unitary gate",
            )),
        }
    }

    /// Implements the numpy array protocol for numpy 2.0+ compatibility.
    ///
    /// Allows direct conversion to numpy array: `np.array(gate)` or `gate.matrix`.
    /// Supports dtype and copy keyword arguments as required by NumPy 2.0.
    #[pyo3(signature = (dtype=None, copy=None))]
    pub fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let mat = match self.inner.matrix() {
            Some(m) => m,
            None => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "No matrix defined for this unitary gate",
                ));
            }
        };

        // Create the array
        let array = PyArray2::from_array(py, mat);

        // Handle dtype conversion if specified
        if let Some(dtype) = dtype {
            // Convert to the requested dtype using numpy's astype
            let astype_result = array.call_method("astype", (dtype,), None)?;
            return Ok(astype_result.extract()?);
        }

        // Handle copy parameter - if copy=True, return a copy
        // The array from from_array is already a copy of the matrix data
        // so we only need to handle explicit copy requests
        if copy == Some(true) {
            let copy_result = array.call_method("copy", (), None)?;
            return Ok(copy_result.extract()?);
        }

        Ok(array)
    }
}

impl From<UnitaryGate> for PyUnitaryGate {
    fn from(inner: UnitaryGate) -> Self {
        Self { inner }
    }
}

impl From<PyUnitaryGate> for UnitaryGate {
    fn from(py: PyUnitaryGate) -> Self {
        py.inner
    }
}
