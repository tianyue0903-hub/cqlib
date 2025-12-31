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
use crate::circuit::parameter::expr_node::{EvalError, ExprNode};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::{Arc, RwLock};

/// A symbolic parameter used in parameterized quantum circuits (PQC).
///
/// `Parameter` serves as the fundamental building block for variational quantum algorithms.
/// It wraps an abstract syntax tree ([`ExprNode`]) representing a mathematical expression
/// and provides thread-safe caching mechanisms for symbol resolution.
///
/// # Key Features
///
/// * **Symbolic Expression**: Can represent variables (e.g., "θ"), constants, or complex arithmetic expressions.
/// * **Thread-Safe**: Uses `Arc` for shared ownership and `RwLock` for internal caching, making it safe to share across threads.
/// * **Lazy Evaluation**: The expression is only evaluated when concrete values are provided via [`Parameter::evaluate`].
/// * **Rich Arithmetic**: Supports standard operators (`+`, `-`, `*`, `/`) via operator overloading.
///
/// # Examples
///
/// Creating a parameter and performing arithmetic:
///
/// ```rust
/// use cqlib_core::circuit::parameter::parameter::Parameter;
///
/// let theta = Parameter::from("theta");
/// let phi = Parameter::from("phi");
///
/// // Create a new expression: θ + 2 * φ
/// let expr = theta + 2.0 * phi;
/// ```
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Thread-safe cache for storing the set of unique symbols found in the expression.
    /// Used to avoid traversing the AST repeatedly.
    pub symbols_cache: Arc<RwLock<Option<Vec<String>>>>,
    /// The root node of the mathematical expression tree.
    pub node: Arc<ExprNode>,
}

impl fmt::Display for Parameter {
    /// Formats the parameter as a mathematical string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.format_with_parent(f, 0, false)
    }
}

impl Default for Parameter {
    /// Creates a default parameter representing the integer `0`.
    fn default() -> Self {
        Self {
            symbols_cache: Arc::new(RwLock::new(None)),
            node: Arc::new(ExprNode::Integer(0)),
        }
    }
}

impl PartialEq for Parameter {
    /// Checks equality based on the underlying expression structure.
    ///
    /// Note: This performs structural equality checking of the AST, not semantic mathematical equality.
    /// For example, `x + y` is not equal to `y + x` structurally.
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl From<ExprNode> for Parameter {
    /// Wraps a raw `ExprNode` into a `Parameter`.
    fn from(node: ExprNode) -> Self {
        Parameter::new(node) // 复用上面的 new
    }
}

impl From<String> for Parameter {
    /// Creates a symbolic parameter from a `String`.
    fn from(val: String) -> Self {
        Parameter::new(ExprNode::Symbol(val))
    }
}

impl From<&str> for Parameter {
    /// Creates a symbolic parameter from a string slice.
    fn from(val: &str) -> Self {
        Parameter::new(ExprNode::Symbol(val.to_string()))
    }
}

macro_rules! impl_numeric_from {
    // 匹配模式：源类型 => Enum变体(强转的目标类型)
    ($($src_type:ty => $variant:ident($target_type:ty)),* $(,)?) => {
        $(
            impl From<$src_type> for Parameter {
                fn from(val: $src_type) -> Self {
                    // val as $target_type 处理了类型转换 (如 u32 -> i64, u64 -> f64)
                    Parameter::new(ExprNode::$variant(val as $target_type))
                }
            }
        )*
    };
}

// Implement From<T> for numeric types to allow easy parameter creation.
// e.g., Parameter::from(1.0)
impl_numeric_from! {
    f64 => Float(f64),
    f32 => Float(f64),

    i64 => Integer(i64),
    i32 => Integer(i64),
    u32 => Integer(i64),
}

impl Parameter {
    /// Constructs a new `Parameter` from an expression node.
    ///
    /// Initializes the symbol cache to `None`.
    pub fn new(node: ExprNode) -> Self {
        Parameter {
            symbols_cache: Arc::new(RwLock::new(None)),
            node: Arc::new(node),
        }
    }

    /// Evaluates the parameter expression given a set of variable bindings.
    ///
    /// # Arguments
    ///
    /// * `bindings` - An optional map where keys are symbol names and values are their numerical substitutions.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The computed floating-point result.
    /// * `Err(EvalError)` - If a symbol is missing from bindings or a math error occurs (e.g., division by zero).
    pub fn evaluate(&self, bindings: &Option<HashMap<String, f64>>) -> Result<f64, EvalError> {
        match bindings {
            Some(map) => self.node.evaluate(map),
            None => self.node.evaluate(&HashMap::new()),
        }
    }

    /// Applies various algebraic and trigonometric simplification rules to this parameter's
    /// underlying expression tree.
    ///
    /// This method is a convenience wrapper around [`ExprNode::simplify`], applying the
    /// simplification logic to the internal `ExprNode` and returning a new `Parameter`
    /// with the simplified expression.
    ///
    /// The simplification process is iterative, meaning it applies rules repeatedly
    /// until the expression no longer changes or a maximum number of iterations is reached.
    /// This allows for multi-step simplifications.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - An `Option<i32>` specifying the maximum number of simplification
    ///   passes to attempt. If `None`, a default of `100` iterations is used. A higher number
    ///   allows for more complex simplifications but increases computation time.
    ///
    /// # Returns
    ///
    /// A new `Parameter` instance containing the simplified expression.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use cqlib_core::circuit::parameter::parameter::Parameter;
    ///
    /// let x = Parameter::from("x");
    ///
    /// // Example 1: Basic algebraic simplification
    /// let expr1 = x.clone() + Parameter::from(0); // x + 0
    /// let simplified1 = expr1.simplify(None);
    /// assert_eq!(simplified1, x);
    ///
    /// // Example 2: Constant folding and term combination
    /// let two_x = Parameter::from(2.0) * x.clone(); // 2*x
    /// let three_x = Parameter::from(3.0) * x.clone(); // 3*x
    /// let expr2 = two_x + three_x; // 2*x + 3*x
    /// let expected2 = Parameter::from(5.0) * x.clone(); // 5*x
    /// let simplified2 = expr2.simplify(None);
    /// assert_eq!(simplified2, expected2);
    ///
    /// // Example 3: Trigonometric identity (e.g., tan(arctan(y)))
    /// let y = Parameter::from("y");
    /// let expr3 = y.atan().tan(); // tan(arctan(y))
    /// let simplified3 = expr3.simplify(None);
    /// assert_eq!(simplified3, y);
    ///
    /// ```
    pub fn simplify(&self, max_iterations: Option<i32>) -> Self {
        let max_iterations = max_iterations.unwrap_or(100);
        Self::new(self.node.simplify(max_iterations))
    }

    ///  Calculate the derivative of the expression with respect to the specified variable (symbolic differentiation)
    ///
    /// # Arguments
    /// * `var` - Which variable to differentiate with respect to
    ///
    /// # Returns
    /// New Parameter object representing the derivative expression
    pub fn derivative(&self, var: &str) -> Self {
        Self::new(self.node.derivative(var))
    }

    /// Retrieves all unique symbols (variables) used in this parameter expression.
    ///
    /// # Caching Strategy
    ///
    /// This method uses a **Double-Checked Locking** pattern to ensure performance:
    /// 1. **Fast Path**: Acquires a read lock to check if symbols are already cached.
    /// 2. **Slow Path**: If not cached, traverses the AST to collect symbols, sorts them, and acquires a write lock to update the cache.
    ///
    /// # Returns
    ///
    /// A sorted `Vec<String>` containing unique symbol names.
    pub fn get_symbols(&self) -> Vec<String> {
        // Fast path: read lock to check cache
        {
            let cache = self.symbols_cache.read().unwrap();
            if let Some(ref symbols) = *cache {
                return symbols.clone();
            }
        }

        // Slow path: calculate symbol set
        let symbols_set = self.node.symbols();
        let mut symbols: Vec<String> = symbols_set.into_iter().collect();
        symbols.sort();

        // Write to cache
        {
            let mut cache = self.symbols_cache.write().unwrap();
            *cache = Some(symbols.clone());
        }

        symbols
    }

    /// Returns a new parameter representing the absolute value `|x|`.
    pub fn abs(&self) -> Self {
        Self::new(ExprNode::Abs(self.node.clone()))
    }

    /// Returns a new parameter representing the square root `√x`.
    pub fn sqrt(&self) -> Self {
        Self::new(ExprNode::Sqrt(self.node.clone()))
    }

    /// Returns a new parameter representing the exponential `e^x`.
    pub fn exp(&self) -> Self {
        Self::new(ExprNode::Exp(self.node.clone()))
    }

    /// Returns a new parameter representing the sine `sin(x)`.
    pub fn sin(&self) -> Self {
        Self::new(ExprNode::Sin(self.node.clone()))
    }

    /// Returns a new parameter representing the inverse sine `asin(x)`.
    pub fn asin(&self) -> Self {
        Self::new(ExprNode::ASin(self.node.clone()))
    }

    /// Returns a new parameter representing the cosine `cos(x)`.
    pub fn cos(&self) -> Self {
        Self::new(ExprNode::Cos(self.node.clone()))
    }

    /// Returns a new parameter representing the inverse cosine `acos(x)`.
    pub fn acos(&self) -> Self {
        Self::new(ExprNode::ACos(self.node.clone()))
    }

    /// Returns a new parameter representing the tangent `tan(x)`.
    pub fn tan(&self) -> Self {
        Self::new(ExprNode::Tan(self.node.clone()))
    }

    /// Returns a new parameter representing the inverse tangent `atan(x)`.
    pub fn atan(&self) -> Self {
        Self::new(ExprNode::ATan(self.node.clone()))
    }

    /// Returns a new parameter representing the natural logarithm `ln(x)`.
    pub fn ln(&self) -> Self {
        Self::new(ExprNode::Ln(self.node.clone()))
    }

    /// Returns a new parameter representing the logarithm with an arbitrary base `log(x, base)`.
    ///
    /// If `base` is `None`, this acts as natural logarithm.
    pub fn log(&self, base: Option<Self>) -> Self {
        match base {
            None => Self::new(ExprNode::Ln(self.node.clone())),
            Some(base_obj) => Self::new(ExprNode::Log(self.node.clone(), base_obj.node.clone())),
        }
    }

    /// Returns a new parameter representing the mathematical constant Pi (π).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::f64::consts;
    /// use cqlib_core::circuit::parameter::parameter::Parameter;
    /// let pi_param = Parameter::pi();
    /// assert_eq!(pi_param.evaluate(&None).unwrap(), consts::PI)
    /// // Now `pi_param` represents the constant π in an expression tree.
    /// ```
    pub fn pi() -> Self {
        Self::new(ExprNode::Pi)
    }

    /// Returns a new parameter representing the mathematical constant Euler's number (e).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::f64::consts;
    /// use cqlib_core::circuit::parameter::parameter::Parameter;
    /// let e_param = Parameter::e();
    /// assert_eq!(e_param.evaluate(&None).unwrap(), consts::E)
    /// // Now `e_param` represents the constant e in an expression tree.
    /// ```
    pub fn e() -> Self {
        Self::new(ExprNode::E)
    }

    /// Returns a new parameter representing the power `self ^ val`.
    ///
    /// # Arguments
    ///
    /// * `val` - The exponent parameter.
    pub fn pow(&self, val: &Self) -> Self {
        Self::new(ExprNode::Pow(self.node.clone(), val.node.clone()))
    }
}

/// A helper trait for converting primitive numeric types into `Arc<ExprNode>`.
///
/// This trait is primarily used by the operator overloading macros (e.g., `impl_ops_for_type`)
/// to provide a unified interface for transforming raw numbers (like `f64`, `i32`)
/// into the internal expression tree representation.
trait IntoArcExprNode {
    fn into_arc_expr(self) -> Arc<ExprNode>;
}

// 定义宏
macro_rules! impl_into_arc_expr {
    // Matches a comma-separated list of type mappings.
    // Pattern: `SourceType => VariantName(CastTargetType)`
    // Example: `f32 => Float(f64)` maps `f32` input to `ExprNode::Float` holding an `f64`.
    ($($src_type:ty => $variant:ident($target_type:ty)),* $(,)?) => {
        $(
            impl IntoArcExprNode for $src_type {
                fn into_arc_expr(self) -> Arc<ExprNode> {
                    // Perform the numeric cast using `as`.
                    // Note: `as` is a safe, lossless no-op when types are identical (e.g., f64 as f64).
                    // For widening conversions (e.g., u32 as i64), it is also safe.
                    Arc::new(ExprNode::$variant(self as $target_type))
                }
            }
        )*
    };
}

impl_into_arc_expr! {
    f64 => Float(f64),
    f32 => Float(f64),

    i64 => Integer(i64),
    i32 => Integer(i64),
    u32 => Integer(i64),
}

/// Helper macro to implement binary operations between `Parameter` and primitive types.
/// e.g., `Parameter + f64` or `f64 - Parameter`.
macro_rules! impl_ops_for_type {
    ($t:ty) => {
        // Parameter + T
        impl Add<$t> for Parameter {
            type Output = Parameter;
            fn add(self, rhs: $t) -> Self::Output {
                Parameter::new(ExprNode::Add(self.node, rhs.into_arc_expr()))
            }
        }

        // T + Parameter
        impl Add<Parameter> for $t {
            type Output = Parameter;
            fn add(self, rhs: Parameter) -> Self::Output {
                Parameter::new(ExprNode::Add(self.into_arc_expr(), rhs.node))
            }
        }

        // Parameter - T
        impl Sub<$t> for Parameter {
            type Output = Parameter;
            fn sub(self, rhs: $t) -> Self::Output {
                Parameter::new(ExprNode::Sub(self.node, rhs.into_arc_expr()))
            }
        }

        // T - Parameter
        impl Sub<Parameter> for $t {
            type Output = Parameter;
            fn sub(self, rhs: Parameter) -> Self::Output {
                Parameter::new(ExprNode::Sub(self.into_arc_expr(), rhs.node))
            }
        }

        // Parameter * Primitive
        impl Mul<$t> for Parameter {
            type Output = Parameter;
            fn mul(self, rhs: $t) -> Self::Output {
                Parameter::new(ExprNode::Mul(self.node, rhs.into_arc_expr()))
            }
        }

        // Primitive * Parameter
        impl Mul<Parameter> for $t {
            type Output = Parameter;
            fn mul(self, rhs: Parameter) -> Self::Output {
                Parameter::new(ExprNode::Mul(self.into_arc_expr(), rhs.node))
            }
        }

        // Parameter / Primitive
        impl Div<$t> for Parameter {
            type Output = Parameter;
            fn div(self, rhs: $t) -> Self::Output {
                Parameter::new(ExprNode::Div(self.node, rhs.into_arc_expr()))
            }
        }

        // Primitive / Parameter
        impl Div<Parameter> for $t {
            type Output = Parameter;
            fn div(self, rhs: Parameter) -> Self::Output {
                Parameter::new(ExprNode::Div(self.into_arc_expr(), rhs.node))
            }
        }
    };
}

impl_ops_for_type!(f64);
impl_ops_for_type!(i64);
impl_ops_for_type!(f32);
impl_ops_for_type!(i32);
impl_ops_for_type!(u32);

macro_rules! impl_binary_op_ref {
    ($($trait:ident, $method:ident, $variant:ident),* $(,)?) => {
        $(
            // 1. Parameter + Parameter (消耗所有权)
            impl $trait<Parameter> for Parameter {
                type Output = Parameter;
                fn $method(self, rhs: Parameter) -> Self::Output {
                    Parameter::new(ExprNode::$variant(self.node, rhs.node))
                }
            }

            // 2. &Parameter + &Parameter (不消耗所有权，最常用)
            impl<'a, 'b> $trait<&'b Parameter> for &'a Parameter {
                type Output = Parameter;
                fn $method(self, rhs: &'b Parameter) -> Self::Output {
                    Parameter::new(ExprNode::$variant(self.node.clone(), rhs.node.clone()))
                }
            }

            impl<'a> $trait<&'a Parameter> for Parameter {
                type Output = Parameter;
                fn $method(self, rhs: &'a Parameter) -> Self::Output {
                    Parameter::new(ExprNode::$variant(self.node, rhs.node.clone()))
                }
            }

            impl<'a> $trait<Parameter> for &'a Parameter {
                type Output = Parameter;
                fn $method(self, rhs: Parameter) -> Self::Output {
                    Parameter::new(ExprNode::$variant(self.node.clone(), rhs.node))
                }
            }
        )*
    };
}

impl_binary_op_ref! {
    Add, add, Add,
    Sub, sub, Sub,
    Mul, mul, Mul,
    Div, div, Div,
    Rem, rem, Mod,
}

#[cfg(test)]
#[path = "./parameter_simplify_test.rs"]
mod parameter_simplify_test;

#[cfg(test)]
#[path = "./parameter_derivative_test.rs"]
mod parameter_derivative_test;
