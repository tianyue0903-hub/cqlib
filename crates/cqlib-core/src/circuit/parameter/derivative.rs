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

//! Symbolic differentiation engine for mathematical expressions.
//!
//! This module implements the recursive logic required to compute the symbolic partial
//! derivative of an expression tree ([`ExprNode`]). It applies standard calculus rules,
//! including the Sum, Product, Quotient, and Chain rules, to generate a new AST representing
//! the derivative.
//!
//! # Mathematical Basis
//!
//! The implementation relies on the recursive definition of derivatives:
//! - $\frac{\partial}{\partial x} (f(x) + g(x)) = f'(x) + g'(x)$
//! - $\frac{\partial}{\partial x} (f(x) \cdot g(x)) = f'(x)g(x) + f(x)g'(x)$
//! - $\frac{\partial}{\partial x} (f(g(x))) = f'(g(x)) \cdot g'(x)$

use crate::circuit::parameter::expr_node::ExprNode;
use std::sync::Arc;

impl ExprNode {
    /// Computes the symbolic partial derivative of the expression with respect to a specific variable.
    ///
    /// This method traverses the Abstract Syntax Tree (AST) recursively, applying standard
    /// differentiation rules to generate a new expression tree representing $\frac{\partial f}{\partial var}$.
    ///
    /// # Arguments
    /// * `var` - The variable name to differentiate with respect to
    ///
    /// # Returns
    /// A new `ExprNode` representing the derivative of this expression
    ///
    /// # Panics
    /// when attempting to differentiate non-differentiable operations
    /// like modulo when the divisor contains the variable
    /// # Examples
    /// ```rust
    /// use cqlib_core::circuit::parameter::expr_node::ExprNode;
    /// use std::sync::Arc;
    ///
    /// let x = ExprNode::Symbol("x".to_string());
    /// let expr = ExprNode::Pow(Arc::new(x), Arc::new(ExprNode::Integer(2)));
    ///
    /// let deriv = expr.derivative("x").simplify(100);
    /// assert_eq!(deriv.to_string(), "2 * x")
    /// ```
    pub fn derivative(&self, var: &str) -> ExprNode {
        match self {
            // Rule 1: d(c)/dx = 0 (derivative of constant is zero)
            ExprNode::Integer(_) | ExprNode::Float(_) | ExprNode::Pi | ExprNode::E => {
                ExprNode::Integer(0)
            }
            // Rule 2: d(x)/dx = 1, d(y)/dx = 0
            ExprNode::Symbol(name) => {
                if name == var {
                    ExprNode::Integer(1) // Derivative of variable with respect to itself is 1
                } else {
                    ExprNode::Integer(0) // Derivative of other variables is 0
                }
            }
            // Rule 3: d(-f)/dx = -f'
            ExprNode::Neg(inner) => ExprNode::Neg(Arc::new(inner.derivative(var))),

            // Rule 4: d(|f|)/dx = sign(f) * f'
            ExprNode::Abs(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Mul(
                    Arc::new(ExprNode::Sign(inner.clone())),
                    Arc::new(inner_deriv),
                )
            }

            // Rule 5: d(√f)/dx = 1/(2√f) * f'
            ExprNode::Sqrt(inner) => {
                let inner_deriv = inner.derivative(var);
                // (√f)' = f' / (2√f)
                ExprNode::Div(
                    Arc::new(inner_deriv),
                    Arc::new(ExprNode::Mul(
                        Arc::new(ExprNode::Integer(2)),
                        Arc::new(ExprNode::Sqrt(inner.clone())),
                    )),
                )
            }

            // Rule 6: d(e^f)/dx = e^f * f'
            ExprNode::Exp(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Mul(
                    Arc::new(ExprNode::Exp(inner.clone())),
                    Arc::new(inner_deriv),
                )
            }

            // Rule 7: d(ln(f))/dx = f'/f
            ExprNode::Ln(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Div(Arc::new(inner_deriv), inner.clone())
            }

            // Rule 8: d(log_b(f))/dx
            ExprNode::Log(arg, base) => {
                let f = arg;
                let b = base;
                let f_prime = f.derivative(var);

                // Optimization: Constant expression check: d/dx(log(C, b)) = 0
                if !f.symbols().contains(var) && !b.symbols().contains(var) {
                    return ExprNode::Integer(0);
                }
                // Optimization: Natural logarithm case (base e): d/dx(ln(f)) = f'/f
                // This step is critical to prevent generating redundant 'ln(e)' nodes.
                if let ExprNode::E = b.as_ref() {
                    return ExprNode::Div(Arc::new(f_prime), f.clone());
                }

                let ln_b = Arc::new(ExprNode::Ln(b.clone()));

                // Check if base is constant w.r.t. var
                if !b.symbols().contains(var) {
                    // Case 1: Base 'b' is constant.
                    // d(log_b(f)) = f' / (f * ln(b))
                    ExprNode::Div(Arc::new(f_prime), Arc::new(ExprNode::Mul(f.clone(), ln_b)))
                } else {
                    // Case 2: Variable Base.
                    // Use expanded form instead of the standard quotient rule to reduce AST nesting depth.
                    // Formula: f'/(f * ln b) - (b' * ln f) / (b * (ln b)^2)
                    let b_prime = b.derivative(var);

                    // Term 1: f' / (f * ln b)  (Same as the result in Case 1)
                    let term1 = ExprNode::Div(
                        Arc::new(f_prime),
                        Arc::new(ExprNode::Mul(f.clone(), ln_b.clone())),
                    );

                    // Term 2: (b' * ln f) / (b * (ln b)^2)
                    // This term is more complex, but the logic remains clear
                    let ln_b_sq = Arc::new(ExprNode::Pow(ln_b, Arc::new(ExprNode::Integer(2))));

                    let numerator_t2 =
                        ExprNode::Mul(Arc::new(b_prime), Arc::new(ExprNode::Ln(f.clone())));

                    let denominator_t2 = ExprNode::Mul(b.clone(), ln_b_sq);

                    let term2 = ExprNode::Div(Arc::new(numerator_t2), Arc::new(denominator_t2));

                    // Result: Term1 - Term2
                    ExprNode::Sub(Arc::new(term1), Arc::new(term2))
                }
            }

            // Rule 9: d(sin(f))/dx = cos(f) * f'
            ExprNode::Sin(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Mul(
                    Arc::new(ExprNode::Cos(inner.clone())),
                    Arc::new(inner_deriv),
                )
            }

            // Rule 10: d(cos(f))/dx = -sin(f) * f'
            ExprNode::Cos(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Mul(
                    Arc::new(ExprNode::Neg(Arc::new(ExprNode::Sin(inner.clone())))),
                    Arc::new(inner_deriv),
                )
            }

            // Rule 11: d(tan(f))/dx = sec²(f) * f' = f'/cos²(f)
            ExprNode::Tan(inner) => {
                let inner_deriv = inner.derivative(var);
                // tan'(f) = 1/cos²(f) * f'
                ExprNode::Div(
                    Arc::new(inner_deriv),
                    Arc::new(ExprNode::Pow(
                        Arc::new(ExprNode::Cos(inner.clone())),
                        Arc::new(ExprNode::Integer(2)),
                    )),
                )
            }

            // Rule 12: d(asin(f))/dx = f' / √(1-f²)
            ExprNode::ASin(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Div(
                    Arc::new(inner_deriv),
                    Arc::new(ExprNode::Sqrt(Arc::new(ExprNode::Sub(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Pow(inner.clone(), Arc::new(ExprNode::Integer(2)))),
                    )))),
                )
            }

            // Rule 13: d(acos(f))/dx = -f' / √(1-f²)
            ExprNode::ACos(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Neg(Arc::new(ExprNode::Div(
                    Arc::new(inner_deriv),
                    Arc::new(ExprNode::Sqrt(Arc::new(ExprNode::Sub(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Pow(inner.clone(), Arc::new(ExprNode::Integer(2)))),
                    )))),
                )))
            }

            // Rule 14: d(atan(f))/dx = f' / (1+f²)
            ExprNode::ATan(inner) => {
                let inner_deriv = inner.derivative(var);
                ExprNode::Div(
                    Arc::new(inner_deriv),
                    Arc::new(ExprNode::Add(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Pow(inner.clone(), Arc::new(ExprNode::Integer(2)))),
                    )),
                )
            }

            // Rule 15: d(f + g)/dx = f' + g'
            ExprNode::Add(lhs, rhs) => {
                ExprNode::Add(Arc::new(lhs.derivative(var)), Arc::new(rhs.derivative(var)))
            }

            // Rule 16: d(f - g)/dx = f' - g'
            ExprNode::Sub(lhs, rhs) => {
                ExprNode::Sub(Arc::new(lhs.derivative(var)), Arc::new(rhs.derivative(var)))
            }

            // Rule 17: d(f * g)/dx = f' * g + f * g'
            ExprNode::Mul(lhs, rhs) => ExprNode::Add(
                Arc::new(ExprNode::Mul(Arc::new(lhs.derivative(var)), rhs.clone())),
                Arc::new(ExprNode::Mul(lhs.clone(), Arc::new(rhs.derivative(var)))),
            ),

            // Rule 18: d(f / g)/dx = (f'g - fg') / g²
            ExprNode::Div(lhs, rhs) => ExprNode::Div(
                Arc::new(ExprNode::Sub(
                    Arc::new(ExprNode::Mul(Arc::new(lhs.derivative(var)), rhs.clone())),
                    Arc::new(ExprNode::Mul(lhs.clone(), Arc::new(rhs.derivative(var)))),
                )),
                Arc::new(ExprNode::Pow(rhs.clone(), Arc::new(ExprNode::Integer(2)))),
            ),

            // Rule 19: d(f % g)/dx
            ExprNode::Mod(lhs, rhs) => {
                let lhs_sym = lhs.symbols();
                let rhs_sym = rhs.symbols();

                if !lhs_sym.contains(var) && !rhs_sym.contains(var) {
                    return ExprNode::Integer(0);
                }
                panic!(
                    "Cannot differentiate modulo operation 'f % g' with respect to '{}' \
                     when the divisor 'g' contains the variable. \
                     The modulo operation is not differentiable in this case.
\
                     Expression: {} % {}",
                    var, lhs, rhs
                );
            }

            // Rule 20: d(f^g)/dx
            ExprNode::Pow(base, exp) => {
                let base_sym = base.symbols();
                let exp_sym = exp.symbols();
                if !base_sym.contains(var) && !exp_sym.contains(var) {
                    return ExprNode::Integer(0);
                }

                // Case 1: g is constant → d(f^c)/dx = c * f^(c-1) * f'
                if !exp_sym.contains(var) {
                    return ExprNode::Mul(
                        Arc::new(ExprNode::Mul(
                            exp.clone(),
                            Arc::new(ExprNode::Pow(
                                base.clone(),
                                Arc::new(ExprNode::Sub(
                                    exp.clone(),
                                    Arc::new(ExprNode::Integer(1)),
                                )),
                            )),
                        )),
                        Arc::new(base.derivative(var)),
                    );
                }

                // Case 2: f is constant → d(c^g)/dx = c^g * ln(c) * g'
                if !base_sym.contains(var) {
                    return ExprNode::Mul(
                        Arc::new(ExprNode::Mul(
                            Arc::new(ExprNode::Pow(base.clone(), exp.clone())),
                            Arc::new(ExprNode::Ln(base.clone())),
                        )),
                        Arc::new(exp.derivative(var)),
                    );
                }

                // Case 3: Both f and g contain variables → Use logarithmic differentiation
                // d(f^g)/dx = f^g * d(g*ln(f))/dx
                //           = f^g * [g' * ln(f) + g * f'/f]
                let base_deriv = base.derivative(var);
                let exp_deriv = exp.derivative(var);

                ExprNode::Mul(
                    Arc::new(ExprNode::Pow(base.clone(), exp.clone())),
                    Arc::new(ExprNode::Add(
                        Arc::new(ExprNode::Mul(
                            Arc::new(exp_deriv),
                            Arc::new(ExprNode::Ln(base.clone())),
                        )),
                        Arc::new(ExprNode::Mul(
                            exp.clone(),
                            Arc::new(ExprNode::Div(Arc::new(base_deriv), base.clone())),
                        )),
                    )),
                )
            }

            // Rule 21: d(sign(f))/dx = 0
            // The sign function is not differentiable at f=0, and its derivative is 0 elsewhere
            ExprNode::Sign(_inner) => ExprNode::Integer(0),
        }
    }
}
