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

//! Python Bindings for Multi-Controlled Gates
//!
//! This module provides Python bindings for the [`MCGate`] from cqlib-core.
//! It represents gates with multiple control qubits applied to a base gate.
//!
//! # Key Components
//!
//! - [`PyMcGate`]: The main class for multi-controlled quantum gates.

use crate::circuit::PyStandardGate;
use crate::circuit::parameter::PyParameter;
use cqlib_core::circuit::Parameter;
use cqlib_core::circuit::gate::MCGate;
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::{PyResult, pyclass, pymethods};
use std::fmt;

/// Python wrapper for `MCGate`.
///
/// Represents a multi-controlled quantum gate.
/// The gate applies the base operation only when all control qubits are in the |1⟩ state.
#[pyclass(name = "McGate", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone)]
pub struct PyMcGate {
    pub inner: MCGate,
}

#[pymethods]
impl PyMcGate {
    /// Creates a new multi-controlled gate.
    ///
    /// # Arguments
    ///
    /// * `num_controls` - The number of control qubits to add.
    /// * `gate` - The base `StandardGate` to control.
    ///
    /// # Examples
    ///
    /// ```python
    /// # Create a Toffoli-like gate (CCX) with 2 controls
    /// ccx = McGate(2, StandardGate.X)
    ///
    /// # Create a multi-controlled Hadamard
    /// mch = McGate(3, StandardGate.H)
    /// ```
    #[new]
    pub fn new(num_controls: u8, gate: PyStandardGate) -> Self {
        Self {
            inner: MCGate::new(num_controls, gate.inner),
        }
    }

    /// Returns the unitary matrix representation as a NumPy array.
    ///
    /// # Arguments
    ///
    /// * `params` - Optional parameters for the base gate (if parametric).
    ///
    /// # Returns
    ///
    /// A 2D numpy array (dtype=complex128).
    #[pyo3(signature = (params=None))]
    pub fn matrix<'py>(
        &self,
        py: Python<'py>,
        params: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let params = params.unwrap_or_default();
        let mat = self.inner.matrix(&params);
        Ok(mat.to_pyarray(py))
    }

    /// Computes the inverse (Hermitian conjugate) of the gate.
    ///
    /// The inverse of a controlled gate C(U) is C(U†).
    ///
    /// # Arguments
    ///
    /// * `params` - Optional parameters for the base gate.
    ///
    /// # Returns
    ///
    /// A tuple of (inverse gate, inverse parameters), or None if not invertible.
    #[pyo3(signature = (params=None))]
    pub fn inverse(
        &self,
        params: Option<Vec<PyParameter>>,
    ) -> PyResult<Option<(Self, Vec<PyParameter>)>> {
        let params_core: Vec<Parameter> = params
            .unwrap_or_default()
            .into_iter()
            .map(|p| p.into_inner())
            .collect();

        match self.inner.inverse(&params_core) {
            Some((inv_gate, inv_params)) => {
                let py_params: Vec<PyParameter> = inv_params
                    .into_iter()
                    .map(|p| PyParameter { inner: p })
                    .collect();
                Ok(Some((PyMcGate { inner: inv_gate }, py_params)))
            }
            None => Ok(None),
        }
    }

    /// Returns the number of control qubits.
    #[getter]
    pub fn num_ctrl_qubits(&self) -> usize {
        self.inner.num_ctrl_qubits()
    }

    /// Returns the total number of qubits (controls + targets).
    #[getter]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of parameters required by the gate.
    #[getter]
    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Returns the base gate (without controls).
    #[getter]
    pub fn base_gate(&self) -> PyStandardGate {
        PyStandardGate::from(*self.inner.base_gate(), vec![])
    }
}

impl fmt::Display for PyMcGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}
