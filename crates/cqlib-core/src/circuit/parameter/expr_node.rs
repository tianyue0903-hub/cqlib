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

use std::collections::{HashMap, HashSet};
use std::fmt;
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

/// Errors that occur during the evaluation of an expression.
#[derive(Debug)]
pub enum EvalError {
    /// A symbol was encountered in the expression but was not found in the bindings map.
    UndefinedSymbol(String),
    /// Attempted to divide by zero or modulo by zero.
    DivisionByZero,
    /// The input value is outside the definition domain of the function.
    ///
    /// Examples include `sqrt(-1)`, `ln(0)`, or `asin(2)`.
    DomainError(String),
    /// The calculation resulted in a NaN (Not a Number).
    NaN(String),
}

impl std::error::Error for EvalError {}
impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UndefinedSymbol(name) => write!(f, "Symbol '{}' value not provided", name),
            EvalError::DivisionByZero => write!(f, "Division by zero occurred"),
            EvalError::DomainError(msg) => write!(f, "Function domain error: {}", msg),
            EvalError::NaN(msg) => write!(f, "Calculation result is not a real number: {}", msg),
        }
    }
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
                    format_infix(lhs, "-", rhs, my_prec, f)
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
#[path = "./test_expr_node.rs"]
mod test_expr_node;
