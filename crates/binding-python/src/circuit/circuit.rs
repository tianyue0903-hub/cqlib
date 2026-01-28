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

use super::parameter::PyParameter;
use cqlib_core::circuit::{Circuit, Qubit};
use pyo3::prelude::*;

/// 这是一个辅助枚举，用于接收 Python 传入的参数（可能是浮点数，也可能是 Parameter 对象）
#[derive(FromPyObject)]
pub enum PyParamLike {
    Float(f64),
    Param(PyParameter),
}

impl Into<cqlib_core::circuit::param::ParameterValue> for PyParamLike {
    fn into(self) -> cqlib_core::circuit::param::ParameterValue {
        match self {
            PyParamLike::Float(f) => cqlib_core::circuit::param::ParameterValue::Fixed(f),
            PyParamLike::Param(p) => cqlib_core::circuit::param::ParameterValue::Param(p.inner),
        }
    }
}

#[pyclass(name = "Circuit", module = "cqlib.circuit")]
pub struct PyCircuit {
    pub inner: Circuit,
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn new(num_qubits: usize) -> Self {
        PyCircuit {
            inner: Circuit::new(num_qubits),
        }
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    // --- Gates ---

    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .h(Qubit::new(qubit as u32))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner
            .cx(Qubit::new(control as u32), Qubit::new(target as u32))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn rx(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rx(Qubit::new(qubit as u32), theta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn ry(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .ry(Qubit::new(qubit as u32), theta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn rz(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rz(Qubit::new(qubit as u32), theta)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn measure(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .measure(Qubit::new(qubit as u32))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // --- Advanced ---

    fn inverse(&self) -> PyResult<Self> {
        let new_inner = self
            .inner
            .inverse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyCircuit { inner: new_inner })
    }
}
