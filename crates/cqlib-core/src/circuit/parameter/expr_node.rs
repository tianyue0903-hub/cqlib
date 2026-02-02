// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::error::EvalError;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

/// Represents a node in the mathematical expression syntax tree.
///
/// `ExprNode` supports basic arithmetic, trigonometric functions, logarithmic functions,
/// and symbolic variables. It is designed to be thread-safe using `Arc` for recursive structures.
///
/// # Examples
///
/// Creating a simple expression `x + 1`:
///
/// ```rust
/// use std::sync::Arc;
/// use cqlib_core::circuit::parameter::expr_node::ExprNode;
///
/// let x = ExprNode::Symbol("x".to_string());
/// let one = ExprNode::Integer(1);
/// let expr = ExprNode::Add(Arc::new(x), Arc::new(one));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ExprNode {
    /// A signed 64-bit integer literal.
    Integer(i64),
    /// A 64-bit floating point literal.
    Float(f64),
    /// A symbolic parameter that must be bound to a value at evaluation time.
    Symbol(String),

    /// Sign function: returns 1.0 if x > 0, -1.0 if x < 0, and 0.0 if x = 0.
    Sign(Arc<ExprNode>),

    /// Mathematical constant π (approx. 3.14159).
    Pi,
    /// Mathematical constant e (Euler's number, approx. 2.71828).
    E,

    /// Absolute value function `|x|`.
    Abs(Arc<ExprNode>),
    /// Square root function `√x`.
    Sqrt(Arc<ExprNode>),
    /// Exponential function `e^x`.
    Exp(Arc<ExprNode>),
    /// Negation operator `-x`.
    Neg(Arc<ExprNode>),

    /// Logarithm with an arbitrary base: `log(value, base)`.
    ///
    /// First argument is the value, second argument is the base.
    Log(Arc<ExprNode>, Arc<ExprNode>),
    /// Natural logarithm `ln(x)` (base e).
    Ln(Arc<ExprNode>),

    /// Sine function `sin(x)` (radians).
    Sin(Arc<ExprNode>),
    /// Inverse sine function `asin(x)`.
    ASin(Arc<ExprNode>),
    /// Cosine function `cos(x)` (radians).
    Cos(Arc<ExprNode>),
    /// Inverse cosine function `acos(x)`.
    ACos(Arc<ExprNode>),
    /// Tangent function `tan(x)` (radians).
    Tan(Arc<ExprNode>),
    /// Inverse tangent function `atan(x)`.
    ATan(Arc<ExprNode>),

    /// Addition operation `lhs + rhs`.
    Add(Arc<ExprNode>, Arc<ExprNode>),
    /// Subtraction operation `lhs - rhs`.
    Sub(Arc<ExprNode>, Arc<ExprNode>),
    /// Multiplication operation `lhs * rhs`.
    Mul(Arc<ExprNode>, Arc<ExprNode>),
    /// Division operation `lhs / rhs`.
    Div(Arc<ExprNode>, Arc<ExprNode>),
    /// Modulo operation `lhs % rhs`.
    Mod(Arc<ExprNode>, Arc<ExprNode>),
    /// Power operation `base ^ exponent`.
    Pow(Arc<ExprNode>, Arc<ExprNode>),
}

impl ExprNode {
    /// Extracts all unique symbolic variables from the expression tree.
    ///
    /// This method performs a recursive traversal of the AST (Abstract Syntax Tree) to find
    /// every `ExprNode::Symbol` node. The results are collected into a `HashSet` to ensure
    /// uniqueness (e.g., if "theta" appears twice, it is returned once).
    ///
    /// # Returns
    ///
    /// A `HashSet<String>` containing the names of all undefined symbols in the expression.
    /// If the expression contains only constants (e.g., `1 + 2`), an empty set is returned.
    ///
    /// # Examples
    ///
    /// Collecting symbols from a mathematical expression like `3 * x + sin(y)`:
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use std::collections::HashSet;
    /// use cqlib_core::circuit::parameter::expr_node::ExprNode;
    ///
    /// // Construct expression: 3 * x + sin(y)
    /// let x = Arc::new(ExprNode::Symbol("x".to_string()));
    /// let y = Arc::new(ExprNode::Symbol("y".to_string()));
    /// let three = Arc::new(ExprNode::Integer(3));
    ///
    /// // term1 = 3 * x
    /// let term1 = Arc::new(ExprNode::Mul(three, x));
    /// // term2 = sin(y)
    /// let term2 = Arc::new(ExprNode::Sin(y));
    /// // expr = term1 + term2
    /// let expr = ExprNode::Add(term1, term2);
    ///
    /// let symbols = expr.symbols();
    ///
    /// assert_eq!(symbols.len(), 2);
    /// assert!(symbols.contains("x"));
    /// assert!(symbols.contains("y"));
    /// assert!(!symbols.contains("z"));
    /// ```
    pub fn symbols(&self) -> HashSet<String> {
        let mut all_symbols = HashSet::new();
        match self {
            // Basic types
            ExprNode::Symbol(name) => {
                all_symbols.insert(name.clone());
            }
            // Constants don't need to be collected
            ExprNode::Integer(_) | ExprNode::Float(_) | ExprNode::Pi | ExprNode::E => {}

            // Unary functions - recursively process inner nodes
            ExprNode::Abs(inner)
            | ExprNode::Sqrt(inner)
            | ExprNode::Exp(inner)
            | ExprNode::Neg(inner)
            | ExprNode::Ln(inner)
            | ExprNode::Sin(inner)
            | ExprNode::ASin(inner)
            | ExprNode::Cos(inner)
            | ExprNode::ACos(inner)
            | ExprNode::Tan(inner)
            | ExprNode::ATan(inner)
            | ExprNode::Sign(inner) => {
                // collect_symbols(inner, symbols);
                all_symbols.extend(inner.symbols());
            }

            // Binary operations - recursively process left and right subtrees
            ExprNode::Add(lhs, rhs)
            | ExprNode::Sub(lhs, rhs)
            | ExprNode::Mul(lhs, rhs)
            | ExprNode::Div(lhs, rhs)
            | ExprNode::Mod(lhs, rhs)
            | ExprNode::Pow(lhs, rhs)
            | ExprNode::Log(lhs, rhs) => {
                all_symbols.extend(lhs.symbols());
                all_symbols.extend(rhs.symbols());
            }
        }

        all_symbols
    }

    /// Evaluates the expression tree numerically using the provided variable bindings.
    ///
    /// This method traverses the AST recursively and computes the final floating-point result.
    ///
    /// # Arguments
    ///
    /// * `bindings` - A map containing values for any `Symbol` nodes present in the tree.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The computed value of the expression.
    /// * `Err(EvalError)` - If an error occurs during evaluation (e.g., missing symbol, division by zero).
    ///
    /// # Errors
    ///
    /// This function will return an error in the following situations:
    ///
    /// * `EvalError::UndefinedSymbol`: If the expression contains a `Symbol` not present in `bindings`.
    /// * `EvalError::DivisionByZero`: If a division or modulo by zero occurs.
    /// * `EvalError::DomainError`: If a mathematical function is called with an invalid argument
    ///   (e.g., `sqrt(-1.0)`, `ln(-5.0)`, `asin(1.5)`).
    /// * `EvalError::NaN`: If intermediate calculation results in `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    /// use cqlib_core::circuit::parameter::expr_node::ExprNode;
    ///
    /// // Expression: x + 2.0
    /// let x_node = Arc::new(ExprNode::Symbol("x".to_string()));
    /// let two_node = Arc::new(ExprNode::Float(2.0));
    /// let expr = ExprNode::Add(x_node, two_node);
    ///
    /// let mut bindings = HashMap::new();
    /// bindings.insert("x".to_string(), 3.0);
    ///
    /// let result = expr.evaluate(&bindings).unwrap();
    /// assert_eq!(result, 5.0);
    /// ```
    pub fn evaluate(&self, bindings: &HashMap<String, f64>) -> Result<f64, EvalError> {
        use std::f64::consts::{E, PI};

        let result = match self {
            ExprNode::Integer(i) => *i as f64,
            ExprNode::Float(f) => *f,
            ExprNode::Symbol(name) => bindings
                .get(name)
                .copied()
                .ok_or_else(|| EvalError::UndefinedSymbol(name.clone()))?,
            ExprNode::Sign(inner) => {
                let value = inner.evaluate(bindings)?;
                if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0 // sign(0) = 0
                }
            }
            ExprNode::Pi => PI,
            ExprNode::E => E,
            // Single-variable function
            ExprNode::Abs(inner) => inner.evaluate(bindings)?.abs(),
            ExprNode::Sqrt(inner) => {
                let value = inner.evaluate(bindings)?;
                if value < 0.0 {
                    return Err(EvalError::DomainError(format!("sqrt({})", value)));
                }
                value.sqrt()
            }
            ExprNode::Exp(inner) => inner.evaluate(bindings)?.exp(),
            ExprNode::Neg(inner) => -inner.evaluate(bindings)?,
            ExprNode::Ln(inner) => {
                let value = inner.evaluate(bindings)?;
                if value <= 0.0 {
                    return Err(EvalError::DomainError(format!(
                        "ln({}) - argument must be positive",
                        value
                    )));
                }
                value.ln()
            }
            ExprNode::Log(arg, base) => {
                let arg_val = arg.evaluate(bindings)?;
                let base_val = base.evaluate(bindings)?;

                if arg_val <= 0.0 {
                    return Err(EvalError::DomainError(format!(
                        "log({}, {}) - argument must be positive",
                        arg_val, base_val
                    )));
                }
                if base_val <= 0.0 || (base_val - 1.0).abs() < 1e-10 {
                    return Err(EvalError::DomainError(format!(
                        "log({}, {}) - base must be positive and not equal to 1",
                        arg_val, base_val
                    )));
                }
                arg_val.log(base_val)
            }

            ExprNode::Sin(inner) => inner.evaluate(bindings)?.sin(),
            ExprNode::ASin(inner) => {
                let value = inner.evaluate(bindings)?;
                if !(-1.0..=1.0).contains(&value) {
                    return Err(EvalError::DomainError(format!("asin({})", value)));
                }
                value.asin()
            }
            ExprNode::Cos(inner) => inner.evaluate(bindings)?.cos(),
            ExprNode::ACos(inner) => {
                let value = inner.evaluate(bindings)?;
                if !(-1.0..=1.0).contains(&value) {
                    return Err(EvalError::DomainError(format!("acos({})", value)));
                }
                value.acos()
            }
            ExprNode::Tan(inner) => inner.evaluate(bindings)?.tan(),
            ExprNode::ATan(inner) => inner.evaluate(bindings)?.atan(),

            // Binary operation
            ExprNode::Add(lhs, rhs) => lhs.evaluate(bindings)? + rhs.evaluate(bindings)?,
            ExprNode::Sub(lhs, rhs) => lhs.evaluate(bindings)? - rhs.evaluate(bindings)?,
            ExprNode::Mul(lhs, rhs) => lhs.evaluate(bindings)? * rhs.evaluate(bindings)?,
            ExprNode::Div(lhs, rhs) => {
                let denominator = rhs.evaluate(bindings)?;
                if denominator.abs() < f64::EPSILON {
                    return Err(EvalError::DivisionByZero);
                }
                lhs.evaluate(bindings)? / denominator
            }
            ExprNode::Mod(lhs, rhs) => {
                let divisor = rhs.evaluate(bindings)?;
                if divisor.abs() < f64::EPSILON {
                    return Err(EvalError::DivisionByZero);
                }
                lhs.evaluate(bindings)? % divisor
            }
            ExprNode::Pow(lhs, rhs) => lhs.evaluate(bindings)?.powf(rhs.evaluate(bindings)?),
        };

        if result.is_nan() {
            return Err(EvalError::NaN(format!("{} evaluated to NaN", self)));
        } else if result.is_infinite() {
            return Err(EvalError::DomainError(format!(
                "{} evaluated to infinity",
                self
            )));
        }

        Ok(result)
    }

    /// Partially evaluates the expression tree using the provided variable bindings.
    ///
    /// This method simplifies the expression by substituting known symbols with their values
    /// and performing constant folding and algebraic simplifications (e.g., `x + 0 -> x`, `x * 0 -> 0`).
    /// Symbols that are not present in `bindings` are left as-is, allowing for partial application
    /// of parameters.
    ///
    /// # Arguments
    ///
    /// * `bindings` - A map containing values for the subset of symbols to be substituted.
    ///
    /// # Returns
    ///
    /// * `Ok(ExprNode)` - The simplified expression node.
    /// * `Err(EvalError)` - If an error occurs during evaluation of sub-expressions (e.g., division by zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    /// use cqlib_core::circuit::parameter::expr_node::ExprNode;
    ///
    /// // Expression: a + b
    /// let a = Arc::new(ExprNode::Symbol("a".to_string()));
    /// let b = Arc::new(ExprNode::Symbol("b".to_string()));
    /// let expr = ExprNode::Add(a, b);
    ///
    /// // Bind a = 1.0, leave b unbound
    /// let mut bindings = HashMap::new();
    /// bindings.insert("a".to_string(), 1.0);
    ///
    /// let result = expr.evaluate_partial(&bindings).unwrap();
    /// // Result should be structurally equivalent to: 1.0 + b
    /// assert_eq!(result.to_string(), "1 + b");
    /// ```
    pub fn evaluate_partial(&self, bindings: &HashMap<String, f64>) -> Result<Self, EvalError> {
        use std::f64::consts::{E, FRAC_PI_2, PI};

        match self {
            ExprNode::Integer(i) => Ok(ExprNode::Integer(*i)),
            ExprNode::Float(f) => Ok(ExprNode::Float(*f)),
            ExprNode::Pi => Ok(ExprNode::Pi),
            ExprNode::E => Ok(ExprNode::E),
            ExprNode::Symbol(name) => {
                if let Some(&value) = bindings.get(name) {
                    Ok(ExprNode::Float(value))
                } else {
                    Ok(ExprNode::Symbol(name.clone()))
                }
            }
            ExprNode::Sign(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => {
                    let sign = if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    Ok(ExprNode::Float(sign))
                }
                ExprNode::Integer(i) => {
                    let sign = if i > 0 {
                        1
                    } else if i < 0 {
                        -1
                    } else {
                        0
                    };
                    Ok(ExprNode::Integer(sign))
                }
                ExprNode::E => Ok(ExprNode::Integer(1)),
                ExprNode::Pi => Ok(ExprNode::Integer(1)),
                evaluated => Ok(ExprNode::Sign(Arc::new(evaluated))),
            },

            ExprNode::Neg(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(-v)),
                ExprNode::Integer(i) => Ok(ExprNode::Integer(-i)),
                evaluated => Ok(ExprNode::Neg(Arc::new(evaluated))),
            },
            ExprNode::Abs(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.abs())),
                ExprNode::Integer(i) => Ok(ExprNode::Integer(i.abs())),
                ExprNode::Pi => Ok(ExprNode::Float(PI)),
                ExprNode::E => Ok(ExprNode::Float(E)),
                evaluated => Ok(ExprNode::Abs(Arc::new(evaluated))),
            },
            ExprNode::Sqrt(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => {
                    if v < 0.0 {
                        return Err(EvalError::DomainError(format!("sqrt({})", v)));
                    }
                    Ok(ExprNode::Float(v.sqrt()))
                }
                ExprNode::Integer(i) if i >= 0 => {
                    let f = (i as f64).sqrt();
                    if f.fract() == 0.0 {
                        Ok(ExprNode::Integer(f as i64))
                    } else {
                        Ok(ExprNode::Float(f))
                    }
                }
                ExprNode::Pi => Ok(ExprNode::Float(PI.sqrt())),
                ExprNode::E => Ok(ExprNode::Float(E.sqrt())),
                evaluated => Ok(ExprNode::Sqrt(Arc::new(evaluated))),
            },
            ExprNode::Exp(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.exp())),
                ExprNode::Integer(i) => Ok(ExprNode::Float((i as f64).exp())),
                ExprNode::Pi => Ok(ExprNode::Float(PI.exp())),
                ExprNode::E => Ok(ExprNode::Float(E.exp())),
                evaluated => Ok(ExprNode::Exp(Arc::new(evaluated))),
            },
            ExprNode::Ln(inner) => {
                match inner.evaluate_partial(bindings)? {
                    ExprNode::Float(1.0) => Ok(ExprNode::Integer(0)),
                    ExprNode::Float(v) if v > 0.0 => Ok(ExprNode::Float(v.ln())),
                    ExprNode::Float(v) => Err(EvalError::DomainError(format!(
                        "ln({}) - argument must be positive",
                        v
                    ))),

                    ExprNode::Integer(1) => Ok(ExprNode::Integer(0)),
                    ExprNode::Integer(i) if i > 1 => Ok(ExprNode::Float((i as f64).ln())),
                    ExprNode::Integer(i) => Err(EvalError::DomainError(format!(
                        "ln({}) - argument must be positive",
                        i
                    ))),

                    ExprNode::E => Ok(ExprNode::Integer(1)),
                    ExprNode::Exp(x) => Ok((*x).clone()),
                    ExprNode::Pow(base, exp) if matches!(base.as_ref(), ExprNode::E) => {
                        // ln(e^y) = y
                        Ok((*exp).clone())
                    }
                    evaluated => Ok(ExprNode::Ln(Arc::new(evaluated))),
                }
            }
            ExprNode::Log(arg, base) => {
                match (
                    arg.evaluate_partial(bindings)?,
                    base.evaluate_partial(bindings)?,
                ) {
                    (ExprNode::Float(a), ExprNode::Float(b)) => {
                        if a <= 0.0 {
                            return Err(EvalError::DomainError(format!(
                                "log({}, {}) - argument must be positive",
                                a, b
                            )));
                        }
                        if b <= 0.0 || (b - 1.0).abs() < f64::EPSILON {
                            return Err(EvalError::DomainError(format!(
                                "log({}, {}) - base must be positive and not 1",
                                a, b
                            )));
                        }
                        Ok(ExprNode::Float(a.log(b)))
                    }
                    (ExprNode::Integer(1), _) | (ExprNode::Float(1.0), _) => {
                        Ok(ExprNode::Integer(0))
                    }
                    (a, b) => Ok(ExprNode::Log(Arc::new(a), Arc::new(b))),
                }
            }
            ExprNode::Sin(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.sin())),
                ExprNode::Integer(v) => Ok(ExprNode::Float((v as f64).sin())),
                ExprNode::Pi => Ok(ExprNode::Float(0.0)),
                ExprNode::E => Ok(ExprNode::Float(E.sin())),
                evaluated => Ok(ExprNode::Sin(Arc::new(evaluated))),
            },
            ExprNode::ASin(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => {
                    if !(-1.0..=1.0).contains(&v) {
                        return Err(EvalError::DomainError(format!("asin({})", v)));
                    }
                    Ok(ExprNode::Float(v.asin()))
                }
                ExprNode::Integer(0) => Ok(ExprNode::Integer(0)),
                ExprNode::Integer(1) => Ok(ExprNode::Float(FRAC_PI_2)),
                ExprNode::Integer(-1) => Ok(ExprNode::Float(-FRAC_PI_2)),
                ExprNode::Integer(i) => Err(EvalError::DomainError(format!("asin({})", i))),
                ExprNode::Pi => Err(EvalError::DomainError("asin(π)".to_string())),
                ExprNode::E => Err(EvalError::DomainError("asin(e)".to_string())),
                evaluated => Ok(ExprNode::ASin(Arc::new(evaluated))),
            },
            ExprNode::Cos(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.cos())),
                ExprNode::Integer(0) => Ok(ExprNode::Integer(1)),
                ExprNode::Integer(v) => Ok(ExprNode::Float((v as f64).cos())),
                ExprNode::Pi => Ok(ExprNode::Float(-1.0)),
                ExprNode::E => Ok(ExprNode::Float(E.cos())),
                evaluated => Ok(ExprNode::Cos(Arc::new(evaluated))),
            },
            ExprNode::ACos(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => {
                    if !(-1.0..=1.0).contains(&v) {
                        return Err(EvalError::DomainError(format!("acos({})", v)));
                    }
                    Ok(ExprNode::Float(v.acos()))
                }
                ExprNode::Integer(1) => Ok(ExprNode::Integer(0)),
                ExprNode::Integer(-1) => Ok(ExprNode::Pi),
                ExprNode::Integer(0) => Ok(ExprNode::Float(std::f64::consts::FRAC_PI_2)),
                ExprNode::Integer(i) => Err(EvalError::DomainError(format!("acos({})", i))),
                ExprNode::Pi => Err(EvalError::DomainError("acos(π)".to_string())),
                ExprNode::E => Err(EvalError::DomainError("acos(e)".to_string())),
                evalued => Ok(ExprNode::ACos(Arc::new(evalued))),
            },
            ExprNode::Tan(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.tan())),
                ExprNode::Integer(0) => Ok(ExprNode::Integer(0)),
                ExprNode::Integer(v) => Ok(ExprNode::Float((v as f64).tan())),
                ExprNode::Pi => Ok(ExprNode::Float(0.0)),
                ExprNode::E => Ok(ExprNode::Float(E.tan())),
                evalued => Ok(ExprNode::Tan(Arc::new(evalued))),
            },
            ExprNode::ATan(inner) => match inner.evaluate_partial(bindings)? {
                ExprNode::Float(v) => Ok(ExprNode::Float(v.atan())),
                ExprNode::Integer(0) => Ok(ExprNode::Integer(0)),
                ExprNode::Integer(v) => Ok(ExprNode::Float((v as f64).atan())),
                ExprNode::Pi => Ok(ExprNode::Float(PI.atan())),
                ExprNode::E => Ok(ExprNode::Float(E.atan())),
                evalued => Ok(ExprNode::ATan(Arc::new(evalued))),
            },

            ExprNode::Add(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => Ok(ExprNode::Integer(a + b)),
                    (ExprNode::Float(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a + b)),
                    (ExprNode::Integer(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a as f64 + b)),
                    (ExprNode::Float(a), ExprNode::Integer(b)) => Ok(ExprNode::Float(a + b as f64)),
                    // Identity: x + 0 = x
                    (x, ExprNode::Integer(0)) | (x, ExprNode::Float(0.0)) => Ok(x),
                    (ExprNode::Integer(0), x) | (ExprNode::Float(0.0), x) => Ok(x),
                    (l, r) => Ok(ExprNode::Add(Arc::new(l), Arc::new(r))),
                }
            }
            ExprNode::Sub(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => Ok(ExprNode::Integer(a - b)),
                    (ExprNode::Float(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a - b)),
                    (ExprNode::Integer(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a as f64 - b)),
                    (ExprNode::Float(a), ExprNode::Integer(b)) => Ok(ExprNode::Float(a - b as f64)),
                    // Identity: x - 0 = x
                    (x, ExprNode::Integer(0)) | (x, ExprNode::Float(0.0)) => Ok(x),
                    // 0 - x = -x
                    (ExprNode::Integer(0), x) => Ok(ExprNode::Neg(Arc::new(x))),
                    (ExprNode::Float(0.0), x) => Ok(ExprNode::Neg(Arc::new(x))),
                    (l, r) => Ok(ExprNode::Sub(Arc::new(l), Arc::new(r))),
                }
            }
            ExprNode::Mul(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => Ok(ExprNode::Integer(a * b)),
                    (ExprNode::Float(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a * b)),
                    (ExprNode::Integer(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a as f64 * b)),
                    (ExprNode::Float(a), ExprNode::Integer(b)) => Ok(ExprNode::Float(a * b as f64)),
                    // Identity: x * 1 = x
                    (x, ExprNode::Integer(1)) | (x, ExprNode::Float(1.0)) => Ok(x),
                    (ExprNode::Integer(1), x) | (ExprNode::Float(1.0), x) => Ok(x),
                    // Zero property: x * 0 = 0
                    (_, ExprNode::Integer(0)) => Ok(ExprNode::Integer(0)),
                    (ExprNode::Integer(0), _) => Ok(ExprNode::Integer(0)),
                    (_, ExprNode::Float(0.0)) => Ok(ExprNode::Float(0.0)),
                    (ExprNode::Float(0.0), _) => Ok(ExprNode::Float(0.0)),

                    (l, r) => Ok(ExprNode::Mul(Arc::new(l), Arc::new(r))),
                }
            }
            ExprNode::Div(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => {
                        if b == 0 {
                            return Err(EvalError::DivisionByZero);
                        }
                        if a % b == 0 {
                            Ok(ExprNode::Integer(a / b))
                        } else {
                            Ok(ExprNode::Float(a as f64 / b as f64))
                        }
                    }
                    (ExprNode::Float(a), ExprNode::Float(b)) => {
                        if b.abs() < f64::EPSILON {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a / b))
                    }
                    (ExprNode::Integer(a), ExprNode::Float(b)) => {
                        if b.abs() < f64::EPSILON {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a as f64 / b))
                    }
                    (ExprNode::Float(a), ExprNode::Integer(b)) => {
                        if b == 0 {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a / b as f64))
                    }
                    // Identity: x / 1 = x
                    (x, ExprNode::Integer(1)) | (x, ExprNode::Float(1.0)) => Ok(x),
                    // 0 / x = 0 (if x != 0) - hard to check x!=0 partially, but assuming safe
                    (ExprNode::Integer(0), _) => Ok(ExprNode::Integer(0)),
                    (ExprNode::Float(0.0), _) => Ok(ExprNode::Float(0.0)),

                    (l, r) => Ok(ExprNode::Div(Arc::new(l), Arc::new(r))),
                }
            }
            ExprNode::Mod(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => {
                        if b == 0 {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Integer(a % b))
                    }
                    (ExprNode::Float(a), ExprNode::Float(b)) => {
                        if b.abs() < f64::EPSILON {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a % b))
                    }
                    (ExprNode::Integer(a), ExprNode::Float(b)) => {
                        if b.abs() < f64::EPSILON {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a as f64 % b))
                    }
                    (ExprNode::Float(a), ExprNode::Integer(b)) => {
                        if b == 0 {
                            return Err(EvalError::DivisionByZero);
                        }
                        Ok(ExprNode::Float(a % b as f64))
                    }
                    (l, r) => Ok(ExprNode::Mod(Arc::new(l), Arc::new(r))),
                }
            }
            ExprNode::Pow(lhs, rhs) => {
                let l = lhs.evaluate_partial(bindings)?;
                let r = rhs.evaluate_partial(bindings)?;
                match (l, r) {
                    (ExprNode::Integer(a), ExprNode::Integer(b)) => {
                        if b >= 0 && b < 10 {
                            // Optimization for small powers
                            Ok(ExprNode::Integer(a.pow(b as u32)))
                        } else {
                            Ok(ExprNode::Float((a as f64).powf(b as f64)))
                        }
                    }
                    (ExprNode::Float(a), ExprNode::Float(b)) => Ok(ExprNode::Float(a.powf(b))),
                    (ExprNode::Integer(a), ExprNode::Float(b)) => {
                        Ok(ExprNode::Float((a as f64).powf(b)))
                    }
                    (ExprNode::Float(a), ExprNode::Integer(b)) => {
                        Ok(ExprNode::Float(a.powf(b as f64)))
                    }
                    // x ^ 0 = 1
                    (_, ExprNode::Integer(0)) | (_, ExprNode::Float(0.0)) => {
                        Ok(ExprNode::Integer(1))
                    }
                    // x ^ 1 = x
                    (x, ExprNode::Integer(1)) | (x, ExprNode::Float(1.0)) => Ok(x),
                    // 0 ^ x = 0 (if x > 0)
                    (ExprNode::Integer(0), _) => Ok(ExprNode::Integer(0)),
                    (ExprNode::Float(0.0), _) => Ok(ExprNode::Float(0.0)),
                    (l, r) => Ok(ExprNode::Pow(Arc::new(l), Arc::new(r))),
                }
            }
        }
    }
}

impl fmt::Display for ExprNode {
    /// Formats the expression tree into a human-readable mathematical string.
    ///
    /// This implementation delegates to `format_with_parent` to handle
    /// operator precedence and parentheses generation recursively.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.format_with_parent(f, 0, false)
    }
}

impl Hash for ExprNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ExprNode::Integer(i) => i.hash(state),
            ExprNode::Float(f) => f.to_bits().hash(state),
            ExprNode::Symbol(s) => s.hash(state),
            ExprNode::Sign(inner) => inner.hash(state),
            ExprNode::Pi => {} // Discriminant already hashed
            ExprNode::E => {}  // Discriminant already hashed
            ExprNode::Abs(inner) => inner.hash(state),
            ExprNode::Sqrt(inner) => inner.hash(state),
            ExprNode::Exp(inner) => inner.hash(state),
            ExprNode::Neg(inner) => inner.hash(state),
            ExprNode::Log(arg, base) => {
                arg.hash(state);
                base.hash(state);
            }
            ExprNode::Ln(inner) => inner.hash(state),
            ExprNode::Sin(inner) => inner.hash(state),
            ExprNode::ASin(inner) => inner.hash(state),
            ExprNode::Cos(inner) => inner.hash(state),
            ExprNode::ACos(inner) => inner.hash(state),
            ExprNode::Tan(inner) => inner.hash(state),
            ExprNode::ATan(inner) => inner.hash(state),
            ExprNode::Add(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
            ExprNode::Sub(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
            ExprNode::Mul(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
            ExprNode::Div(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
            ExprNode::Mod(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
            ExprNode::Pow(lhs, rhs) => {
                lhs.hash(state);
                rhs.hash(state);
            }
        }
    }
}

impl ExprNode {
    /// Formats the expression tree recursively, adding parentheses where necessary.
    ///
    /// This function implements the standard "precedence climbing" logic for pretty-printing.
    /// It decides whether to wrap the current node in `()` based on the parent's precedence
    /// and the node's position (left or right child) to respect associativity.
    ///
    /// # Arguments
    ///
    /// * `parent_prec` - The precedence level of the parent node.
    /// * `is_right_child` - Whether this node is the right operand of a binary operator.
    ///   This is crucial for left-associative operators like `-` and `/`.
    ///   For example, in `(a - b) - c`, `b` is right child of first `-`, but `(a - b)` is left child.
    pub fn format_with_parent(
        &self,
        f: &mut fmt::Formatter<'_>,
        parent_prec: u8,
        is_right_child: bool,
    ) -> fmt::Result {
        let my_prec = self.precedence();
        let has_parent = parent_prec != u8::MAX;

        let need_parens = has_parent
            && (my_prec < parent_prec
                || (my_prec == parent_prec
                    && matches!(
                        (self, is_right_child),
                        (ExprNode::Sub(_, _), true)
                            | (ExprNode::Div(_, _), true)
                            | (ExprNode::Pow(_, _), true)
                            | (ExprNode::Mod(_, _), true)
                    )));

        if need_parens {
            write!(f, "(")?;
        }

        match self {
            ExprNode::Integer(i) => write!(f, "{}", i),
            ExprNode::Float(x) => {
                if x.is_nan() || x.is_infinite() {
                    return Err(fmt::Error);
                }
                if x.is_sign_negative() && x.abs() < f64::EPSILON {
                    return write!(f, "{:?}", x);
                }
                if (x.fract() - 0.0).abs() < f64::EPSILON {
                    write!(f, "{:.0}", x)
                } else {
                    write!(f, "{}", x)
                }
            }
            ExprNode::Symbol(s) => write!(f, "{}", s),
            ExprNode::Sign(inner) => {
                write!(f, "sign(")?;
                inner.format_with_parent(f, u8::MAX, false)?;
                write!(f, ")")
            }
            ExprNode::Pi => write!(f, "π"),
            ExprNode::E => write!(f, "e"),
            ExprNode::Ln(inner) => format_call("ln", inner, f),
            ExprNode::Log(arg, base) => {
                write!(f, "log(")?;
                arg.format_with_parent(f, u8::MAX, false)?;
                write!(f, ", ")?;
                base.format_with_parent(f, u8::MAX, false)?;
                write!(f, ")")
            }
            ExprNode::Neg(inner) => {
                // Optimization: Handle double negation `--x` -> `x`
                if let ExprNode::Neg(inner_inner) = inner.as_ref() {
                    inner_inner.format_with_parent(f, parent_prec, is_right_child)
                } else {
                    write!(f, "-")?;
                    // If the inner expression is an additive operation, it needs parens
                    // to avoid ambiguity (e.g. `-(a + b)` vs `-a + b`).
                    let inner_needs_parens =
                        matches!(inner.as_ref(), ExprNode::Add(_, _) | ExprNode::Sub(_, _));

                    if inner_needs_parens {
                        write!(f, "(")?;
                        inner.format_with_parent(f, u8::MAX, false)?;
                        write!(f, ")")
                    } else {
                        inner.format_with_parent(f, u8::MAX, false) // 使用高优先级确保正确
                    }
                }
            }

            // Optimization: `x + (-y)` should be formatted as `x - y` for readability.
            ExprNode::Add(lhs, rhs) => {
                if let ExprNode::Neg(rhs_inner) = rhs.as_ref() {
                    lhs.format_with_parent(f, my_prec, false)?;
                    write!(f, " - ")?;
                    rhs_inner.format_with_parent(f, my_prec, true)
                } else {
                    format_infix(lhs, "+", rhs, my_prec, f)
                }
            }

            // Optimization: `x - (-y)` should be formatted as `x + y` for readability.
            ExprNode::Sub(lhs, rhs) => {
                if let ExprNode::Neg(rhs_inner) = rhs.as_ref() {
                    lhs.format_with_parent(f, my_prec, false)?;
                    write!(f, " + ")?;
                    rhs_inner.format_with_parent(f, my_prec, true)
                } else {
                    lhs.format_with_parent(f, my_prec, false)?;
                    write!(f, " - ")?;
                    if rhs.is_negative() {
                        write!(f, "(")?;
                        rhs.format_with_parent(f, u8::MAX, false)?;
                        write!(f, ")")
                    } else {
                        rhs.format_with_parent(f, my_prec, true)
                    }
                }
            }

            // Division needs special care if the denominator is negative or complex.
            ExprNode::Div(lhs, rhs) => {
                lhs.format_with_parent(f, my_prec, false)?;
                write!(f, " / ")?;
                // Force parentheses for negative denominators: `1 / (-2)` instead of `1 / -2`
                if rhs.is_negative() || rhs.precedence() < my_prec {
                    write!(f, "(")?;
                    rhs.format_with_parent(f, u8::MAX, false)?;
                    write!(f, ")")
                } else {
                    rhs.format_with_parent(f, my_prec + 1, true)
                }
            }

            ExprNode::Mod(lhs, rhs) => {
                lhs.format_with_parent(f, my_prec, false)?;
                write!(f, " % ")?;

                if rhs.is_negative() || rhs.precedence() < my_prec {
                    write!(f, "(")?;
                    rhs.format_with_parent(f, u8::MAX, false)?;
                    write!(f, ")")
                } else {
                    rhs.format_with_parent(f, my_prec + 1, true)
                }
            }

            ExprNode::Mul(lhs, rhs) => {
                if let ExprNode::Integer(-1) = lhs.as_ref() {
                    write!(f, "-")?;
                    let rhs_needs_parens =
                        matches!(rhs.as_ref(), ExprNode::Add(_, _) | ExprNode::Sub(_, _));
                    if rhs_needs_parens {
                        write!(f, "(")?;
                        rhs.format_with_parent(f, u8::MAX, false)?;
                        write!(f, ")")
                    } else {
                        rhs.format_with_parent(f, 6, false)
                    }
                } else {
                    format_infix(lhs, "*", rhs, my_prec, f)
                }
            }

            ExprNode::Pow(lhs, rhs) => {
                // Base needs parens if it's negative or has lower precedence.
                // E.g., `(-2)^x` vs `-2^x` (which is `-(2^x)`).
                if lhs.is_negative() || lhs.precedence() < my_prec {
                    write!(f, "(")?;
                    lhs.format_with_parent(f, u8::MAX, false)?;
                    write!(f, ")")?;
                } else {
                    lhs.format_with_parent(f, my_prec, false)?;
                }

                write!(f, "^")?;

                // Exponent usually needs protection if it's complex.
                if rhs.is_negative() || rhs.precedence() <= my_prec {
                    write!(f, "(")?;
                    rhs.format_with_parent(f, u8::MAX, false)?;
                    write!(f, ")")
                } else {
                    rhs.format_with_parent(f, u8::MAX, false)
                }
            }

            // Standard function calls
            ExprNode::Abs(inner) => format_call("abs", inner, f),
            ExprNode::Sqrt(inner) => format_call("sqrt", inner, f),
            ExprNode::Exp(inner) => format_call("exp", inner, f),
            ExprNode::Sin(inner) => format_call("sin", inner, f),
            ExprNode::ASin(inner) => format_call("asin", inner, f),
            ExprNode::Cos(inner) => format_call("cos", inner, f),
            ExprNode::ACos(inner) => format_call("acos", inner, f),
            ExprNode::Tan(inner) => format_call("tan", inner, f),
            ExprNode::ATan(inner) => format_call("atan", inner, f),
        }?;

        if need_parens {
            write!(f, ")")?;
        }
        Ok(())
    }

    /// Returns the operator precedence level for formatting decisions.
    ///
    /// Higher values indicate tighter binding.
    ///
    /// # Precedence Table
    /// * 7: Atoms (Int, Float, Symbol)
    /// * 6: Unary ops (Neg, Functions like sin, log)
    /// * 5: Power (^)
    /// * 4: Multiplicative (*, /, %)
    /// * 3: Additive (+, -)
    fn precedence(&self) -> u8 {
        match self {
            ExprNode::Abs(_)
            | ExprNode::Sqrt(_)
            | ExprNode::Exp(_)
            | ExprNode::Neg(_)
            | ExprNode::Log(_, _)
            | ExprNode::Ln(_)
            | ExprNode::Sin(_)
            | ExprNode::ASin(_)
            | ExprNode::Cos(_)
            | ExprNode::ACos(_)
            | ExprNode::Tan(_)
            | ExprNode::ATan(_)
            | ExprNode::Sign(_) => 6,
            ExprNode::Pow(_, _) => 5,
            ExprNode::Mul(_, _) | ExprNode::Div(_, _) | ExprNode::Mod(_, _) => 4,
            ExprNode::Add(_, _) | ExprNode::Sub(_, _) => 3,
            ExprNode::Integer(_)
            | ExprNode::Float(_)
            | ExprNode::Symbol(_)
            | ExprNode::Pi
            | ExprNode::E => 7,
        }
    }

    /// Checks if the node represents a semantically negative value.
    ///
    /// This is used during formatting to decide if parentheses are needed
    /// around negative numbers in contexts like `x + (-5)`.
    fn is_negative(&self) -> bool {
        match self {
            ExprNode::Neg(_) => true,
            ExprNode::Integer(i) => *i < 0,
            ExprNode::Float(f) => *f < 0.0,
            ExprNode::Mul(lhs, _) | ExprNode::Div(lhs, _) => lhs.is_negative(),
            _ => false,
        }
    }
}

/// Helper to format standard binary infix operations (e.g., `a + b`).
///
/// Automatically handles the recursive formatting of left and right operands.
fn format_infix(
    lhs: &Arc<ExprNode>,
    op: &str,
    rhs: &Arc<ExprNode>,
    my_prec: u8,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    lhs.format_with_parent(f, my_prec, false)?;
    write!(f, " {} ", op)?;
    rhs.format_with_parent(f, my_prec, true)
}

/// Helper to format function call style nodes (e.g., `sin(x)`).
fn format_call(name: &str, arg: &Arc<ExprNode>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}(", name)?;
    arg.format_with_parent(f, u8::MAX, false)?;
    write!(f, ")")
}

#[cfg(test)]
#[path = "./expr_node_test.rs"]
mod expr_node_test;
