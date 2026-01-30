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

use crate::circuit::PyStandardGate;
use crate::circuit::parameter::PyParameter;
use cqlib_core::circuit::Parameter;
use cqlib_core::circuit::gate::MCGate;
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::{PyResult, pyclass, pymethods};
use std::fmt;

#[pyclass(name = "McGate", module = "cqlib.circuit.gates")]
#[derive(Debug, Clone)]
pub struct PyMcGate {
    inner: MCGate,
}

#[pymethods]
impl PyMcGate {
    #[new]
    pub fn new(num_controls: u8, gate: PyStandardGate) -> Self {
        Self {
            inner: MCGate::new(num_controls, gate.inner).into(),
        }
    }

    /// Returns the unitary matrix representation of the gate.
    ///
    /// Args:
    ///     params: A list of floating-point parameters for parametric gates.
    ///
    /// Returns:
    ///     The unitary matrix as a numpy array.
    pub fn matrix<'py>(
        &self,
        py: Python<'py>,
        params: Vec<f64>,
    ) -> Bound<'py, PyArray2<Complex64>> {
        let mat = self.inner.matrix(&params);
        mat.to_pyarray(py)
    }

    /// Computes the inverse (Hermitian conjugate) of the gate.
    ///
    /// Args:
    ///     params: A list of Parameter objects for parametric gates.
    ///
    /// Returns:
    ///     A tuple of (inverse McGate, inverse parameters) or None if not invertible.
    #[pyo3(signature = (params=None))]
    pub fn inverse(
        &self,
        params: Option<Vec<PyParameter>>,
    ) -> PyResult<Option<(Self, Vec<PyParameter>)>> {
        let params_core: Vec<Parameter> = params
            .unwrap_or_default()
            .into_iter()
            .map(|p| p.inner)
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
        PyStandardGate {
            inner: *self.inner.base_gate(),
            params: vec![],
        }
    }
}

impl fmt::Display for PyMcGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}
