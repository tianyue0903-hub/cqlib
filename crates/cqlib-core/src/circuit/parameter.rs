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

//! Symbolic and numeric parameter type for quantum circuit gate angles.
//!
//! [`Parameter`] wraps a [`symb_anafis::Expr`] expression tree, allowing gate parameters
//! to be either concrete floating-point numbers or symbolic expressions containing
//! free variables (e.g. `θ`, `φ`).  Symbolic parameters can be composed with
//! arithmetic operators, differentiated, simplified, and finally evaluated by
//! supplying variable bindings.
//!
//! # Quick start
//!
//! ```rust
//! use cqlib_core::circuit::Parameter;
//!
//! // Numeric parameter
//! let angle = Parameter::from(std::f64::consts::PI / 2.0);
//!
//! // Symbolic parameter
//! let theta = Parameter::symbol("θ");
//! let expr  = theta.clone() * Parameter::from(2.0) + Parameter::pi();
//!
//! // Evaluate with a concrete binding
//! let val = expr.evaluate(&Some([("θ", 0.5)].iter().cloned().collect())).unwrap();
//! ```

use crate::circuit::error::ParameterError;
use core::fmt::Debug;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::str::FromStr;
use symb_anafis::{Expr, Simplify, parse, symb};

/// A symbolic or numeric parameter used to represent gate angles and other
/// continuous values in a quantum circuit.
///
/// `Parameter` is a thin, cheaply-cloneable wrapper around [`symb_anafis::Expr`].
/// It supports the full suite of arithmetic operators (`+`, `-`, `*`, `/`)
/// between `Parameter` values as well as between `Parameter` and primitive numeric types
/// (`f64`, `f32`, `i32`, `u32`).
///
/// The mathematical constants `π` and `e` are always available as free
/// symbols and are automatically resolved during [`Parameter::evaluate`].
#[derive(Clone, Debug)]
pub struct Parameter {
    /// The root node of the mathematical expression tree.
    expr: Expr,
}

impl From<Expr> for Parameter {
    fn from(expr: Expr) -> Self {
        Parameter { expr }
    }
}

impl From<Parameter> for Expr {
    fn from(p: Parameter) -> Self {
        p.expr
    }
}

impl fmt::Display for Parameter {
    /// Formats the parameter as a human-readable mathematical string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expr, f)
    }
}

impl Default for Parameter {
    /// Creates a default parameter representing the integer `0`.
    fn default() -> Self {
        Self {
            expr: Expr::from(0),
        }
    }
}

impl std::hash::Hash for Parameter {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expr.hash(state);
    }
}

impl PartialEq for Parameter {
    /// Checks equality based on the underlying expression structure.
    ///
    /// Note: This performs **structural** equality of the AST, not semantic
    /// mathematical equality.  For example, `x + y` and `y + x` are
    /// structurally different and will compare as unequal.
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Eq for Parameter {}

impl TryFrom<&str> for Parameter {
    type Error = ParameterError;

    /// Parses a symbolic expression from a string slice.
    ///
    /// Returns [`ParameterError::ParseError`] if the string cannot be parsed
    /// as a valid expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let p = Parameter::try_from("θ * 2 + π").unwrap();
    /// let bad = Parameter::try_from("@@@");
    /// assert!(bad.is_err());
    /// ```
    fn try_from(expr: &str) -> Result<Self, Self::Error> {
        if let Ok(expr) = parse(expr, &HashSet::new(), &HashSet::new(), None) {
            Ok(Self { expr })
        } else {
            Err(ParameterError::ParseError(expr.to_string()))
        }
    }
}

impl FromStr for Parameter {
    type Err = ParameterError;

    /// Parses a symbolic expression from a string.
    ///
    /// Returns [`ParameterError::ParseError`] if the string cannot be parsed
    /// as a valid expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    /// use std::str::FromStr;
    ///
    /// let p = Parameter::from_str("θ * 2 + π").unwrap();
    /// let bad = Parameter::from_str("@@@");
    /// assert!(bad.is_err());
    /// ```
    fn from_str(expr: &str) -> Result<Self, Self::Err> {
        Self::try_from(expr)
    }
}

impl Parameter {
    /// Creates a new `Parameter` directly from a [`symb_anafis::Expr`] node.
    ///
    /// Prefer [`Parameter::symbol`] for named variables or the `From<f64>` / `From<i32>`
    /// impls for numeric constants.
    pub fn new(expr: Expr) -> Self {
        Self { expr }
    }

    /// Returns a reference to the underlying expression.
    pub fn as_expr(&self) -> &Expr {
        &self.expr
    }

    /// Consumes the parameter and returns the underlying expression.
    pub fn into_expr(self) -> Expr {
        self.expr
    }

    /// Creates a symbolic parameter representing a free variable with the
    /// given `name`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let theta = Parameter::symbol("θ");
    /// assert_eq!(theta.get_symbols().contains("θ"), true);
    /// ```
    pub fn symbol(name: &str) -> Self {
        Self::new(Expr::from(symb(name)))
    }

    /// Evaluates the expression to a concrete `f64` value.
    ///
    /// The mathematical constants `π` and `e` are injected automatically, so
    /// callers do not need to supply them.  Additional variable bindings are
    /// provided through `bindings`.
    ///
    /// Returns [`ParameterError::NaN`] if the expression still contains
    /// unbound free variables after applying the supplied bindings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    /// use std::collections::HashMap;
    ///
    /// // Numeric constant — no bindings needed.
    /// let p = Parameter::from(std::f64::consts::PI);
    /// assert!((p.evaluate(&None).unwrap() - std::f64::consts::PI).abs() < 1e-10);
    ///
    /// // Symbolic expression with a user-supplied binding.
    /// let theta = Parameter::symbol("θ");
    /// let expr  = theta * Parameter::from(2.0);
    /// let mut bindings = HashMap::new();
    /// bindings.insert("θ", 1.0_f64);
    /// assert!((expr.evaluate(&Some(bindings)).unwrap() - 2.0).abs() < 1e-10);
    /// ```
    pub fn evaluate(&self, bindings: &Option<HashMap<&str, f64>>) -> Result<f64, ParameterError> {
        // Build the effective binding table used for evaluation.
        let mut actual_bindings = HashMap::new();

        // Automatically inject the fundamental math constants so callers
        // never have to supply them manually.
        actual_bindings.insert("π", std::f64::consts::PI);
        actual_bindings.insert("e", std::f64::consts::E);
        if let Some(user_bindings) = bindings {
            actual_bindings.extend(user_bindings);
        }

        if let Some(n) = self
            .expr
            .evaluate(&actual_bindings, &HashMap::new())
            .as_number()
        {
            // Check for NaN
            if n.is_nan() {
                return Err(ParameterError::DomainError(format!(
                    "Evaluation of '{}' resulted in NaN",
                    self.expr
                )));
            }
            // Check for infinity (positive or negative)
            if n.is_infinite() {
                return Err(ParameterError::DomainError(format!(
                    "Evaluation of '{}' resulted in infinity",
                    self.expr
                )));
            }
            Ok(n)
        } else {
            Err(ParameterError::NaN(self.expr.to_string()))
        }
    }

    /// Returns an algebraically simplified form of this parameter.
    ///
    /// Simplification is performed in a domain-safe mode to avoid invalid
    /// transformations (e.g. taking the log of a potentially negative number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let theta = Parameter::symbol("θ");
    /// let expr  = theta.clone() * Parameter::from(1.0);   // θ * 1
    /// let simplified = expr.simplify().unwrap();
    /// assert_eq!(simplified.to_string(), "θ");
    /// ```
    pub fn simplify(&self) -> Result<Self, ParameterError> {
        let expr = Simplify::new().domain_safe(true).simplify(&self.expr)?;
        Ok(Self { expr })
    }

    /// Computes the symbolic partial derivative of this expression with
    /// respect to the variable `var`.
    ///
    /// Returns [`ParameterError`] if differentiation is not supported for
    /// the expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let theta = Parameter::symbol("θ");
    /// let expr  = theta.clone() * theta.clone();  // θ²
    /// let deriv = expr.derivative("θ").unwrap();  // 2θ
    /// let val   = deriv.evaluate(&Some([("θ", 3.0)].iter().cloned().collect())).unwrap();
    /// assert!((val - 6.0).abs() < 1e-10);
    /// ```
    pub fn derivative(&self, var: &str) -> Result<Self, ParameterError> {
        let expr = self.expr.diff(var)?;
        Ok(Self { expr })
    }

    /// Returns the set of free variable names present in this expression.
    ///
    /// The built-in constants `π` and `e` may appear here if the expression
    /// was constructed symbolically (e.g. via [`Parameter::pi`]), but they are
    /// resolved automatically during [`Parameter::evaluate`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let expr = Parameter::symbol("θ") + Parameter::symbol("φ");
    /// let syms = expr.get_symbols();
    /// assert!(syms.contains("θ"));
    /// assert!(syms.contains("φ"));
    /// ```
    pub fn get_symbols(&self) -> HashSet<String> {
        self.expr.variables()
    }

    /// Returns a new parameter representing the absolute value `|x|`.
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.clone().abs(),
        }
    }

    /// Returns a new parameter representing the square root `√x`.
    pub fn sqrt(&self) -> Self {
        Self {
            expr: self.expr.clone().sqrt(),
        }
    }

    /// Returns a new parameter representing the exponential `eˣ`.
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.clone().exp(),
        }
    }

    /// Returns a new parameter representing the sine `sin(x)`.
    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.clone().sin(),
        }
    }

    /// Returns a new parameter representing the inverse sine `asin(x)`.
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.clone().asin(),
        }
    }

    /// Returns a new parameter representing the cosine `cos(x)`.
    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.clone().cos(),
        }
    }

    /// Returns a new parameter representing the inverse cosine `acos(x)`.
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.clone().acos(),
        }
    }

    /// Returns a new parameter representing the tangent `tan(x)`.
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.clone().tan(),
        }
    }

    /// Returns a new parameter representing the inverse tangent `atan(x)`.
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.clone().atan(),
        }
    }

    /// Returns a new parameter representing the natural logarithm `ln(x)`.
    pub fn ln(&self) -> Self {
        Self {
            expr: self.expr.clone().ln(),
        }
    }

    /// Returns a new parameter representing the logarithm `log_base(x)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// // log base 2 of x
    /// let x    = Parameter::symbol("x");
    /// let log2 = x.log(Parameter::from(2.0));
    /// let val  = log2.evaluate(&Some([("x", 8.0)].iter().cloned().collect())).unwrap();
    /// assert!((val - 3.0).abs() < 1e-10);
    /// ```
    pub fn log(&self, base: impl Into<Self>) -> Self {
        Self {
            expr: self.expr.clone().log(base.into()),
        }
    }

    /// Returns the mathematical constant π as a symbolic parameter.
    ///
    /// The symbol `"π"` is resolved to [`std::f64::consts::PI`] automatically
    /// inside [`Parameter::evaluate`].
    pub fn pi() -> Self {
        Self::symbol("π")
    }

    /// Returns the mathematical constant *e* (Euler's number) as a symbolic
    /// parameter.
    ///
    /// The symbol `"e"` is resolved to [`std::f64::consts::E`] automatically
    /// inside [`Parameter::evaluate`].
    pub fn e() -> Self {
        Self::symbol("e")
    }

    /// Returns a new parameter representing `self` raised to the power `exp`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let x   = Parameter::symbol("x");
    /// let x3  = x.pow(Parameter::from(3.0));   // x³
    /// let val = x3.evaluate(&Some([("x", 2.0)].iter().cloned().collect())).unwrap();
    /// assert!((val - 8.0).abs() < 1e-10);
    /// ```
    pub fn pow(&self, exp: impl Into<Self>) -> Self {
        Self {
            expr: self.expr.clone().pow(exp.into()),
        }
    }

    /// Substitutes every occurrence of `symbol` in this expression with
    /// `param`, returning the resulting expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let expr     = Parameter::symbol("θ") * Parameter::from(2.0);  // θ * 2
    /// let replaced = expr.replace("θ", Parameter::pi());      // π * 2
    /// let val      = replaced.evaluate(&None).unwrap();
    /// assert!((val - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    /// ```
    pub fn replace(&self, symbol: &str, param: impl Into<Self>) -> Self {
        let p: Self = param.into();
        Self {
            expr: self.expr.substitute(symbol, &p.expr),
        }
    }

    /// Returns `true` if this parameter is a constant (has no free variables).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let constant = Parameter::from(3.14);
    /// assert!(constant.is_constant());
    ///
    /// let symbolic = Parameter::symbol("x");
    /// assert!(!symbolic.is_constant());
    /// ```
    pub fn is_constant(&self) -> bool {
        self.get_symbols().is_empty()
    }

    /// Returns `true` if this parameter evaluates to zero.
    ///
    /// Returns `false` if the parameter cannot be evaluated (contains unbound symbols).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let zero = Parameter::from(0.0);
    /// assert!(zero.is_zero());
    ///
    /// let non_zero = Parameter::from(1.0);
    /// assert!(!non_zero.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.evaluate(&None)
            .map(|v| v.abs() < f64::EPSILON)
            .unwrap_or(false)
    }

    /// Returns `true` if this parameter evaluates to one.
    ///
    /// Returns `false` if the parameter cannot be evaluated (contains unbound symbols).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Parameter;
    ///
    /// let one = Parameter::from(1.0);
    /// assert!(one.is_one());
    ///
    /// let non_one = Parameter::from(2.0);
    /// assert!(!non_one.is_one());
    /// ```
    pub fn is_one(&self) -> bool {
        self.evaluate(&None)
            .map(|v| (v - 1.0).abs() < f64::EPSILON)
            .unwrap_or(false)
    }

    /// Returns the hyperbolic sine `sinh(x)`.
    pub fn sinh(&self) -> Self {
        Self {
            expr: self.expr.clone().sinh(),
        }
    }

    /// Returns the hyperbolic cosine `cosh(x)`.
    pub fn cosh(&self) -> Self {
        Self {
            expr: self.expr.clone().cosh(),
        }
    }

    /// Returns the hyperbolic tangent `tanh(x)`.
    pub fn tanh(&self) -> Self {
        Self {
            expr: self.expr.clone().tanh(),
        }
    }

    /// Returns the floor of the expression.
    pub fn floor(&self) -> Self {
        Self {
            expr: self.expr.clone().floor(),
        }
    }

    /// Returns the ceiling of the expression.
    pub fn ceil(&self) -> Self {
        Self {
            expr: self.expr.clone().ceil(),
        }
    }

    /// Returns the rounded value of the expression.
    pub fn round(&self) -> Self {
        Self {
            expr: self.expr.clone().round(),
        }
    }
}

/// Implements `From<$src_type>` for `Parameter` by casting `$src_type` to `$target_type`
/// and delegating to the corresponding `Expr::from` implementation.
///
/// Pattern: `src_type => target_type`, e.g. `f32 => f64`.
macro_rules! impl_from_num_for_p {
    ($($src_type:ty => $target_type:ty),* $(,)?) => {
        $(
            impl From<$src_type> for Parameter {
                fn from(val: $src_type) -> Self {
                    Self {
                        // Cast to the target numeric type, then use the
                        // built-in `Expr::from` conversion provided by
                        // `symb_anafis`.
                        expr: Expr::from(val as $target_type),
                    }
                }
            }
        )*
    };
}

impl_from_num_for_p! {
    f64 => f64,
    f32 => f64,
    u32 => f64,
    i32 => i32,
}

/// Implements the four arithmetic operators between `Parameter` and a primitive numeric
/// type `$t` in both directions (e.g. `Parameter + f64` and `f64 + Parameter`).
///
/// The implementation converts `$t` to `Parameter` via `Parameter::from`, extracts the
/// underlying `Expr`, and delegates to the corresponding `Expr` operator.
macro_rules! impl_ops_for_type {
    ($t:ty) => {
        // Parameter + T
        impl Add<$t> for Parameter {
            type Output = Parameter;
            fn add(self, rhs: $t) -> Self::Output {
                Self {
                    // 1. `Parameter::from(rhs)` converts the primitive to `Parameter` using
                    //    the `impl_from_num_for_p!` impls above.
                    // 2. `.expr` unwraps the underlying `Expr` node.
                    // 3. `self.expr + ...` invokes `symb_anafis`'s `Add` impl.
                    expr: self.expr + Parameter::from(rhs).expr,
                }
            }
        }

        // T + Parameter
        impl Add<Parameter> for $t {
            type Output = Parameter;
            fn add(self, rhs: Parameter) -> Self::Output {
                Parameter {
                    expr: Parameter::from(self).expr + rhs.expr,
                }
            }
        }

        // Parameter - T
        impl Sub<$t> for Parameter {
            type Output = Parameter;
            fn sub(self, rhs: $t) -> Self::Output {
                Self {
                    expr: self.expr - Parameter::from(rhs).expr,
                }
            }
        }

        // T - Parameter
        impl Sub<Parameter> for $t {
            type Output = Parameter;
            fn sub(self, rhs: Parameter) -> Self::Output {
                Parameter {
                    expr: Parameter::from(self).expr - rhs.expr,
                }
            }
        }

        // Parameter * T
        impl Mul<$t> for Parameter {
            type Output = Parameter;
            fn mul(self, rhs: $t) -> Self::Output {
                Self {
                    expr: self.expr * Parameter::from(rhs).expr,
                }
            }
        }

        // T * Parameter
        impl Mul<Parameter> for $t {
            type Output = Parameter;
            fn mul(self, rhs: Parameter) -> Self::Output {
                Parameter {
                    expr: Parameter::from(self).expr * rhs.expr,
                }
            }
        }

        // Parameter / T
        impl Div<$t> for Parameter {
            type Output = Parameter;
            fn div(self, rhs: $t) -> Self::Output {
                Self {
                    expr: self.expr / Parameter::from(rhs).expr,
                }
            }
        }

        // T / Parameter
        impl Div<Parameter> for $t {
            type Output = Parameter;
            fn div(self, rhs: Parameter) -> Self::Output {
                Parameter {
                    expr: Parameter::from(self).expr / rhs.expr,
                }
            }
        }
    };
}

// Apply the operator impls to all supported primitive types.
impl_ops_for_type!(f64);
impl_ops_for_type!(f32);
impl_ops_for_type!(i32);
impl_ops_for_type!(u32);

/// Implements all four combinations of owned/borrowed operands for a binary
/// operator between two `Parameter` values.
///
/// For each `(Trait, method)` pair this macro generates:
///
/// 1. `Parameter  op Parameter`  — both operands consumed.
/// 2. `&Parameter op &Parameter` — neither operand consumed (most common in loop contexts).
/// 3. `Parameter  op &Parameter` — left consumed, right borrowed.
/// 4. `&Parameter op Parameter`  — left borrowed, right consumed.
///
/// Cloning `Expr` is inexpensive because `symb_anafis` uses reference-counted
/// node sharing internally.
macro_rules! impl_binary_op_ref {
    ($($trait:ident, $method:ident),* $(,)?) => {
        $(
            // 1. Parameter op Parameter — both sides consumed.
            impl $trait<Parameter> for Parameter {
                type Output = Parameter;
                fn $method(self, rhs: Parameter) -> Self::Output {
                    Parameter {
                        // Delegate directly to the underlying `Expr` operator.
                        expr: self.expr.$method(rhs.expr),
                    }
                }
            }

            // 2. &Parameter op &Parameter — no ownership transferred; clone the inner `Expr`.
            impl<'a, 'b> $trait<&'b Parameter> for &'a Parameter {
                type Output = Parameter;
                fn $method(self, rhs: &'b Parameter) -> Self::Output {
                    Parameter {
                        // `Expr::clone` is cheap — the tree uses Arc-based
                        // node sharing inside `symb_anafis`.
                        expr: self.expr.clone().$method(rhs.expr.clone()),
                    }
                }
            }

            // 3. Parameter op &Parameter — left consumed, right cloned.
            impl<'a> $trait<&'a Parameter> for Parameter {
                type Output = Parameter;
                fn $method(self, rhs: &'a Parameter) -> Self::Output {
                    Parameter {
                        expr: self.expr.$method(rhs.expr.clone()),
                    }
                }
            }

            // 4. &Parameter op Parameter — left cloned, right consumed.
            impl<'a> $trait<Parameter> for &'a Parameter {
                type Output = Parameter;
                fn $method(self, rhs: Parameter) -> Self::Output {
                    Parameter {
                        expr: self.expr.clone().$method(rhs.expr),
                    }
                }
            }
        )*
    };
}

impl_binary_op_ref! {
    Add, add,
    Sub, sub,
    Mul, mul,
    Div, div,
}

// Implement Neg for owned Parameter
impl Neg for Parameter {
    type Output = Parameter;
    fn neg(self) -> Self::Output {
        Parameter {
            expr: self.expr.neg(),
        }
    }
}

// Implement Neg for borrowed Parameter
impl Neg for &Parameter {
    type Output = Parameter;
    fn neg(self) -> Self::Output {
        Parameter {
            expr: self.expr.clone().neg(),
        }
    }
}

#[cfg(test)]
#[path = "parameter_test.rs"]
mod parameter_test;
