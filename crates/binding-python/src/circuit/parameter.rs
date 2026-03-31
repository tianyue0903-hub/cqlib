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
    /// Creates a new parameter.
    ///
    /// This method intelligently detects the input type:
    /// - If a number is passed, creates a numeric parameter (e.g., `Parameter(3.14)` creates 3.14)
    /// - If a string that looks like a pure number is passed, creates a numeric parameter
    /// - Otherwise, creates a symbolic parameter (e.g., `Parameter("theta")`)
    ///
    /// # Arguments
    ///
    /// * `value` - The value (number or string symbol name).
    ///
    /// # Examples
    ///
    /// ```python
    /// # Numeric parameter
    /// p1 = Parameter(3.14)  # Creates 3.14
    /// p2 = Parameter("3.14") # Also creates 3.14
    ///
    /// # Symbolic parameter
    /// p3 = Parameter("theta") # Creates symbol 'theta'
    /// p4 = Parameter("x + 1") # Creates expression
    /// ```
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // First, try to extract as a number (int or float)
        if let Ok(val) = value.extract::<f64>() {
            return Ok(PyParameter {
                inner: Parameter::from(val),
            });
        }
        if let Ok(val) = value.extract::<i64>() {
            return Ok(PyParameter {
                inner: Parameter::from(val as f64),
            });
        }

        // If not a number, try as string
        if let Ok(name) = value.extract::<String>() {
            // Try to parse as a number first
            if let Ok(num) = name.parse::<f64>() {
                return Ok(PyParameter {
                    inner: Parameter::from(num),
                });
            }
            // Try to parse as expression (might contain numbers like "3.14+2")
            if let Ok(param) = Parameter::try_from(name.as_str()) {
                return Ok(PyParameter { inner: param });
            }
            // Otherwise, treat as a symbol name
            return Ok(PyParameter {
                inner: Parameter::symbol(&name),
            });
        }

        Err(PyTypeError::new_err(
            "Parameter value must be a number or string",
        ))
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
        Parameter::try_from(expr.as_str())
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
        let bindings_ref = bindings.as_ref().map(|map| {
            map.iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect::<HashMap<&str, f64>>()
        });
        self.inner
            .evaluate(&bindings_ref)
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
    #[pyo3(signature = ())]
    fn simplify(&self) -> PyResult<Self> {
        // Note: max_iterations is ignored in the current implementation
        let inner = self
            .inner
            .simplify()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyParameter { inner })
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
    fn derivative(&self, var: String) -> PyResult<Self> {
        let inner = self
            .inner
            .derivative(&var)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyParameter { inner })
    }

    /// Returns the power of this parameter raised to the given exponent.
    ///
    /// # Arguments
    ///
    /// * `val` - The exponent (can be a float or Parameter).
    ///
    /// # Returns
    ///
    /// A new parameter representing `self^val`.
    ///
    /// # Example
    ///
    /// ```python
    /// x = Parameter("x")
    /// y = Parameter("y")
    /// result = x.pow(y)  # x^y
    /// result = x.pow(2)  # x^2
    /// ```
    fn pow(&self, val: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(param) = val.extract::<PyParameter>() {
            Ok(PyParameter {
                inner: self.inner.pow(param.inner),
            })
        } else if let Ok(val_f64) = val.extract::<f64>() {
            let exp_param = Parameter::from(val_f64);
            Ok(PyParameter {
                inner: self.inner.pow(exp_param),
            })
        } else {
            Err(PyTypeError::new_err(
                "pow argument must be a number or Parameter",
            ))
        }
    }

    /// Returns all unique symbols in this parameter.
    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.get_symbols().into_iter().collect()
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
    #[pyo3(signature = (base=None))]
    fn log(&self, base: Option<PyParameter>) -> Self {
        let base_inner = base.map(|p| p.inner).unwrap_or(Parameter::e());
        PyParameter {
            inner: self.inner.log(base_inner),
        }
    }

    /// Returns the hyperbolic sine sinh(x).
    fn sinh(&self) -> Self {
        PyParameter {
            inner: self.inner.sinh(),
        }
    }

    /// Returns the hyperbolic cosine cosh(x).
    fn cosh(&self) -> Self {
        PyParameter {
            inner: self.inner.cosh(),
        }
    }

    /// Returns the hyperbolic tangent tanh(x).
    fn tanh(&self) -> Self {
        PyParameter {
            inner: self.inner.tanh(),
        }
    }

    /// Returns the floor of the expression.
    fn floor(&self) -> Self {
        PyParameter {
            inner: self.inner.floor(),
        }
    }

    /// Returns the ceiling of the expression.
    fn ceil(&self) -> Self {
        PyParameter {
            inner: self.inner.ceil(),
        }
    }

    /// Returns the rounded value of the expression.
    fn round(&self) -> Self {
        PyParameter {
            inner: self.inner.round(),
        }
    }

    /// Returns True if this parameter is a constant (has no free variables).
    ///
    /// # Examples
    ///
    /// ```python
    /// >>> Parameter(3.14).is_constant()
    /// True
    /// >>> Parameter("x").is_constant()
    /// False
    /// ```
    fn is_constant(&self) -> bool {
        self.inner.is_constant()
    }

    /// Returns True if this parameter evaluates to zero.
    ///
    /// Returns False if the parameter cannot be evaluated (contains unbound symbols).
    ///
    /// # Examples
    ///
    /// ```python
    /// >>> Parameter(0.0).is_zero()
    /// True
    /// >>> Parameter(1.0).is_zero()
    /// False
    /// ```
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns True if this parameter evaluates to one.
    ///
    /// Returns False if the parameter cannot be evaluated (contains unbound symbols).
    ///
    /// # Examples
    ///
    /// ```python
    /// >>> Parameter(1.0).is_one()
    /// True
    /// >>> Parameter(2.0).is_one()
    /// False
    /// ```
    fn is_one(&self) -> bool {
        self.inner.is_one()
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
                inner: self.inner.pow(param.inner),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let exp_param = Parameter::from(val);
            Ok(PyParameter {
                inner: self.inner.pow(exp_param),
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
        PyParameter {
            inner: self.inner.replace(&symbol, param.inner),
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
