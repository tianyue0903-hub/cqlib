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

use cqlib_core::circuit::gate::UnitaryDef;
use num_complex::Complex64;

use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::{PyResult, Python, pyclass, pymethods};

#[pyclass(name = "UnitaryGate", module = "cqlib.circuit.gates")]
#[derive(Debug, Clone)]
pub struct PyUnitaryGate {
    inner: UnitaryDef,
}

#[pymethods]
impl PyUnitaryGate {
    #[new]
    pub fn new(label: String, num_qubits: u16) -> PyResult<Self> {
        Ok(Self {
            inner: UnitaryDef::new(label.as_ref(), num_qubits),
        })
    }

    pub fn with_matrix<'py>(&self, py: Python<'py>, mat: Bound<'py, PyAny>) -> PyResult<Self> {
        let np = py.import("numpy")?;
        // Allow flexible input (list, int array, float array) by casting to complex128 via numpy
        let array_obj = np.call_method1("array", (mat, "complex128"))?;

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

    #[getter]
    pub fn label(&self) -> String {
        self.inner.label().to_string()
    }

    #[getter]
    pub fn num_qubits(&self) -> u16 {
        self.inner.num_qubits()
    }

    #[getter]
    pub fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        match self.inner.matrix() {
            Some(mat) => Ok(PyArray2::from_array(py, mat)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "No matrix defined for this unitary gate",
            )),
        }
    }

    pub fn __array__<'py>(
        &self,
        py: Python<'py>,
        _dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        match self.inner.matrix() {
            Some(mat) => Ok(PyArray2::from_array(py, mat)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "No matrix defined for this unitary gate",
            )),
        }
    }
}

impl From<UnitaryDef> for PyUnitaryGate {
    fn from(inner: UnitaryDef) -> Self {
        Self { inner }
    }
}

impl From<PyUnitaryGate> for UnitaryDef {
    fn from(py: PyUnitaryGate) -> Self {
        py.inner
    }
}
