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

//! Python Bindings for Symbolic Parameters
//!
//! This module provides Python bindings for the [`Parameter`] from cqlib-core.
//! It supports parameterized quantum circuits (PQC) and variational quantum algorithms (VQA).
//!
//! # Key Features
//!
//! - Symbolic parameter creation and manipulation
//! - Arithmetic operations via operator overloading
//! - Mathematical functions (trigonometric, exponential, etc.)
//! - Symbolic differentiation and simplification

use cqlib_core::circuit::Parameter;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Python wrapper for `Parameter`.
///
/// Represents a symbolic parameter for parameterized quantum circuits.
/// Supports arithmetic operations, mathematical functions, differentiation, and simplification.
#[pyclass(name = "Parameter", module = "cqlib.circuit")]
#[derive(Clone)]
pub struct PyParameter {
    pub(crate) inner: Parameter,
}

#[pymethods]
impl PyParameter {
    /// Creates a new symbolic parameter.
    ///
    /// # Arguments
    ///
    /// * `name` - The symbol name (e.g., "theta", "x").
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> Self {
        PyParameter {
            inner: Parameter::symbol(name),
        }
    }

    /// Creates a parameter from a float value.
    ///
    /// # Arguments
    ///
    /// * `val` - A floating-point number.
    #[staticmethod]
    fn from_float(val: f64) -> Self {
        PyParameter {
            inner: Parameter::from(val),
        }
    }

    /// Parses a mathematical expression string into a Parameter.
    ///
    /// # Arguments
    ///
    /// * `expr` - The expression string to parse (e.g., "theta + 1", "pi/2", "sin(x)").
    ///
    /// # Returns
    ///
    /// A new Parameter representing the parsed expression.
    ///
    /// # Supported Syntax
    ///
    /// - Numbers: `1`, `3.14`, `-2.5`
    /// - Constants: `pi`, `e`
    /// - Variables: `theta`, `x`, `y`
    /// - Operators: `+`, `-`, `*`, `/`
    /// - Functions: `sin`, `cos`, `tan`, `exp`, `sqrt`, `ln`, etc.
    /// - Parentheses: `(`, `)`
    #[staticmethod]
    fn from_expression(expr: String) -> PyResult<Self> {
        use cqlib_core::circuit::parameter::parse_parameter;
        parse_parameter(expr.as_str())
            .map(|param| PyParameter { inner: param })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns the mathematical constant Pi.
    #[staticmethod]
    fn pi() -> Self {
        PyParameter {
            inner: Parameter::pi(),
        }
    }

    /// Returns the mathematical constant e (Euler's number).
    #[staticmethod]
    fn e() -> Self {
        PyParameter {
            inner: Parameter::e(),
        }
    }

    /// Evaluates the parameter with concrete values.
    ///
    /// # Arguments
    ///
    /// * `bindings` - Optional mapping from symbol names to values.
    ///
    /// # Returns
    ///
    /// The computed floating-point result.
    #[pyo3(signature = (bindings=None))]
    fn evaluate(&self, bindings: Option<HashMap<String, f64>>) -> PyResult<f64> {
        self.inner
            .evaluate(&bindings)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Simplifies the parameter expression.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum simplification passes (default: 100).
    ///
    /// # Returns
    ///
    /// A new simplified parameter.
    #[pyo3(signature = (max_iterations=None))]
    fn simplify(&self, max_iterations: Option<i32>) -> Self {
        PyParameter {
            inner: self.inner.simplify(max_iterations),
        }
    }

    /// Computes the symbolic derivative with respect to a variable.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable name to differentiate by.
    ///
    /// # Returns
    ///
    /// A new parameter representing the derivative.
    fn derivative(&self, var: String) -> Self {
        PyParameter {
            inner: self.inner.derivative(&var),
        }
    }

    /// Returns all unique symbols in this parameter.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.get_symbols()
    }

    /// Returns the absolute value |x|.
    fn abs(&self) -> Self {
        PyParameter {
            inner: self.inner.abs(),
        }
    }

    /// Returns the square root sqrt(x).
    fn sqrt(&self) -> Self {
        PyParameter {
            inner: self.inner.sqrt(),
        }
    }

    /// Returns the exponential e^x.
    fn exp(&self) -> Self {
        PyParameter {
            inner: self.inner.exp(),
        }
    }

    /// Returns the sine sin(x).
    fn sin(&self) -> Self {
        PyParameter {
            inner: self.inner.sin(),
        }
    }

    /// Returns the cosine cos(x).
    fn cos(&self) -> Self {
        PyParameter {
            inner: self.inner.cos(),
        }
    }

    /// Returns the tangent tan(x).
    fn tan(&self) -> Self {
        PyParameter {
            inner: self.inner.tan(),
        }
    }

    /// Returns the inverse sine asin(x).
    fn asin(&self) -> Self {
        PyParameter {
            inner: self.inner.asin(),
        }
    }

    /// Returns the inverse cosine acos(x).
    fn acos(&self) -> Self {
        PyParameter {
            inner: self.inner.acos(),
        }
    }

    /// Returns the inverse tangent atan(x).
    fn atan(&self) -> Self {
        PyParameter {
            inner: self.inner.atan(),
        }
    }

    /// Returns the natural logarithm ln(x).
    fn ln(&self) -> Self {
        PyParameter {
            inner: self.inner.ln(),
        }
    }

    /// Returns the logarithm. If base is None, returns natural log.
    ///
    /// # Arguments
    ///
    /// * `base` - Optional base for the logarithm.
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

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.clone() % param.inner,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for %"))
        }
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = other.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: param.inner % self.inner.clone(),
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for %"))
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

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    /// Replaces all occurrences of a symbol with another parameter expression.
    ///
    /// This method performs symbolic substitution, replacing every instance of the
    /// specified symbol with the given parameter's expression tree.
    ///
    /// # Arguments
    ///
    /// * `symbol` - The name of the symbol to replace.
    /// * `param` - The parameter expression to substitute.
    ///
    /// # Returns
    ///
    /// A new `Parameter` with the substitution applied. The original parameter is unchanged.
    ///
    /// # Examples
    ///
    /// ```python
    /// from cqlib import Parameter
    ///
    /// # Create expression: x + 2
    /// x = Parameter("x")
    /// expr = x + Parameter.from_float(2.0)
    ///
    /// # Replace x with y * 3
    /// y = Parameter("y")
    /// replacement = y * Parameter.from_float(3.0)
    /// new_expr = expr.replace("x", replacement)
    /// # Result: (y * 3) + 2
    /// ```
    fn replace(&self, symbol: String, param: PyParameter) -> Self {
        // Need to clone self to get mutable reference, but we use replace which returns new
        let mut inner_clone = self.inner.clone();
        PyParameter {
            inner: inner_clone.replace(&symbol, &param.inner),
        }
    }
}

impl PyParameter {
    /// Returns a reference to the inner core Parameter.
    pub fn inner(&self) -> &Parameter {
        &self.inner
    }

    /// Consumes the wrapper and returns the inner core Parameter.
    pub fn into_inner(self) -> Parameter {
        self.inner
    }
}

// Helper conversion: allows CoreParameter to be converted to PyParameter
impl From<Parameter> for PyParameter {
    fn from(inner: Parameter) -> Self {
        PyParameter { inner }
    }
}
