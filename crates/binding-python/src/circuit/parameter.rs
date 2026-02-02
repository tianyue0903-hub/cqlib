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

use cqlib_core::circuit::Parameter;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "Parameter", module = "cqlib.circuit")]
#[derive(Clone)]
pub struct PyParameter {
    pub(crate) inner: Parameter,
}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> Self {
        PyParameter {
            inner: Parameter::symbol(name),
        }
    }

    #[staticmethod]
    fn from_float(val: f64) -> Self {
        PyParameter {
            inner: Parameter::from(val),
        }
    }

    #[staticmethod]
    fn pi() -> Self {
        PyParameter {
            inner: Parameter::pi(),
        }
    }

    #[staticmethod]
    fn e() -> Self {
        PyParameter {
            inner: Parameter::e(),
        }
    }

    #[pyo3(signature = (bindings=None))]
    fn evaluate(&self, bindings: Option<HashMap<String, f64>>) -> PyResult<f64> {
        self.inner
            .evaluate(&bindings)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (max_iterations=None))]
    fn simplify(&self, max_iterations: Option<i32>) -> Self {
        PyParameter {
            inner: self.inner.simplify(max_iterations),
        }
    }

    fn derivative(&self, var: String) -> Self {
        PyParameter {
            inner: self.inner.derivative(&var),
        }
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.get_symbols()
    }

    fn abs(&self) -> Self {
        PyParameter {
            inner: self.inner.abs(),
        }
    }

    fn sqrt(&self) -> Self {
        PyParameter {
            inner: self.inner.sqrt(),
        }
    }

    fn exp(&self) -> Self {
        PyParameter {
            inner: self.inner.exp(),
        }
    }

    fn sin(&self) -> Self {
        PyParameter {
            inner: self.inner.sin(),
        }
    }

    fn cos(&self) -> Self {
        PyParameter {
            inner: self.inner.cos(),
        }
    }

    fn tan(&self) -> Self {
        PyParameter {
            inner: self.inner.tan(),
        }
    }

    fn asin(&self) -> Self {
        PyParameter {
            inner: self.inner.asin(),
        }
    }

    fn acos(&self) -> Self {
        PyParameter {
            inner: self.inner.acos(),
        }
    }

    fn atan(&self) -> Self {
        PyParameter {
            inner: self.inner.atan(),
        }
    }

    fn ln(&self) -> Self {
        PyParameter {
            inner: self.inner.ln(),
        }
    }

    fn log(&self, base: Option<PyParameter>) -> Self {
        let base_inner = base.map(|p| p.inner);
        PyParameter {
            inner: self.inner.log(base_inner),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Parameter({})", self.inner)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.clone() + param.inner,
            })
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: self.inner.clone() + val,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.clone() - param.inner,
            })
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: self.inner.clone() - val,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: val - self.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.clone() * param.inner,
            })
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: self.inner.clone() * val,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.clone() / param.inner,
            })
        } else if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: self.inner.clone() / val,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f64>() {
            Ok(PyParameter {
                inner: val / self.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }

    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.pow(&param.inner),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let exp_param = Parameter::from(val);
            Ok(PyParameter {
                inner: self.inner.pow(&exp_param),
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for **"))
        }
    }

    fn __neg__(&self) -> Self {
        PyParameter {
            inner: Parameter::from(0.0) - self.inner.clone(),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(param) = other.extract::<PyParameter>() {
            self.inner == param.inner
        } else {
            false
        }
    }
}

impl PyParameter {
    pub fn inner(&self) -> &Parameter {
        &self.inner
    }

    pub fn into_inner(self) -> Parameter {
        self.inner.clone()
    }
}

// 辅助转换：允许 CoreParameter 转换为 PyParameter
impl From<Parameter> for PyParameter {
    fn from(inner: Parameter) -> Self {
        PyParameter { inner }
    }
}
