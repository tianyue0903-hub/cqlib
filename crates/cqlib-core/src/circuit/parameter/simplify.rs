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

use crate::circuit::parameter::expr_node::ExprNode;
use std::sync::Arc;

impl ExprNode {
    /// Applies basic algebraic simplifications to the expression tree.
    ///
    /// This method performs a recursive bottom-up simplification, focusing on:
    /// - **Constant Folding**: Computing operations on constants (e.g., `2 + 3 -> 5`).
    /// - **Identity Elements**: Removing identity operations (e.g., `x + 0 -> x`, `x * 1 -> x`).
    /// - **Zero Annihilation**: Handling zero multiplication (e.g., `x * 0 -> 0`).
    /// - **Basic Linear Algebra**: Combining like terms (e.g., `2x + 3x -> 5x`).
    ///
    /// # Complexity
    ///
    /// This operation is generally O(N) where N is the number of nodes in the tree,
    /// but nested recursive calls can increase the practical cost.
    pub fn simplify_basic(&self) -> ExprNode {
        match self {
            // x + 0 = x
            ExprNode::Add(lhs, rhs) if rhs.is_zero() => lhs.simplify(2),
            // 0 + x = x
            ExprNode::Add(lhs, rhs) if lhs.is_zero() => rhs.simplify(2),
            // x + x = 2x
            ExprNode::Add(lhs, rhs) if lhs == rhs => {
                ExprNode::Mul(Arc::new(ExprNode::Integer(2)), Arc::new(lhs.simplify(2)))
            }

            ExprNode::Add(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                // 2 + 3 = 5
                if let (Some(a), Some(b)) = (as_constant(&l), as_constant(&r)) {
                    return ExprNode::Float(a + b);
                }
                // 2x + 3x = 5x
                if let (ExprNode::Mul(lhs_coef, lhs_term), ExprNode::Mul(rhs_coef, rhs_term)) =
                    (&l, &r)
                {
                    if lhs_term == rhs_term {
                        if let (Some(a), Some(b)) = (as_constant(lhs_coef), as_constant(rhs_coef)) {
                            return ExprNode::Mul(
                                Arc::new(ExprNode::Float(a + b)),
                                lhs_term.clone(),
                            );
                        }
                    }
                }

                // x + 3x = 4x
                if let (ExprNode::Symbol(x), ExprNode::Mul(rhs_coef, rhs_term)) = (&l, &r) {
                    if let ExprNode::Symbol(rhs_x) = rhs_term.as_ref() {
                        if let Some(rhs_c) = as_constant(rhs_coef) {
                            if x == rhs_x {
                                return ExprNode::Mul(
                                    Arc::new(ExprNode::Float(rhs_c + 1.0)),
                                    rhs_term.clone(),
                                );
                            }
                        }
                    }
                }

                // 3x + x = 4x
                if let (ExprNode::Mul(lhs_coef, lhs_term), ExprNode::Symbol(x)) = (&l, &r) {
                    if let ExprNode::Symbol(lhs_x) = lhs_term.as_ref() {
                        if let Some(lhs_c) = as_constant(lhs_coef) {
                            if x == lhs_x {
                                return ExprNode::Mul(
                                    Arc::new(ExprNode::Float(lhs_c + 1.0)),
                                    lhs_term.clone(),
                                );
                            }
                        }
                    }
                }
                ExprNode::Add(Arc::new(l), Arc::new(r))
            }

            // x - 0 = x
            ExprNode::Sub(lhs, rhs) if rhs.is_zero() => lhs.simplify(2),
            // x - x = 0
            ExprNode::Sub(lhs, rhs) if lhs == rhs => ExprNode::Integer(0),
            // 0 - x = -x
            ExprNode::Sub(lhs, rhs) if lhs.is_zero() => ExprNode::Neg(Arc::new(rhs.simplify(2))),

            ExprNode::Sub(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                if let (Some(a), Some(b)) = (as_constant(&l), as_constant(&r)) {
                    return ExprNode::Float(a - b);
                }

                // 3x - 2x = x
                if let (ExprNode::Mul(lhs_coef, lhs_term), ExprNode::Mul(rhs_coef, rhs_term)) =
                    (&l, &r)
                {
                    if lhs_term == rhs_term {
                        if let (Some(a), Some(b)) = (as_constant(lhs_coef), as_constant(rhs_coef)) {
                            return ExprNode::Mul(
                                Arc::new(ExprNode::Float(a - b)),
                                lhs_term.clone(),
                            );
                        }
                    }
                }

                // 3x - x = 2x
                if let (ExprNode::Mul(lhs_coef, lhs_term), ExprNode::Symbol(x)) = (&l, &r) {
                    if let ExprNode::Symbol(lhs_x) = lhs_term.as_ref() {
                        if let Some(lhs_c) = as_constant(lhs_coef) {
                            if x == lhs_x {
                                return ExprNode::Mul(
                                    Arc::new(ExprNode::Float(lhs_c - 1.0)),
                                    lhs_term.clone(),
                                );
                            }
                        }
                    }
                }
                // x - 3x = -2x
                if let (ExprNode::Symbol(x), ExprNode::Mul(rhs_coef, rhs_term)) = (&l, &r) {
                    if let ExprNode::Symbol(rhs_x) = rhs_term.as_ref() {
                        if let Some(rhs_c) = as_constant(rhs_coef) {
                            if x == rhs_x {
                                return ExprNode::Mul(
                                    Arc::new(ExprNode::Float(1.0 - rhs_c)),
                                    rhs_term.clone(),
                                );
                            }
                        }
                    }
                }
                ExprNode::Sub(Arc::new(l), Arc::new(r))
            }

            // x * 0 = 0
            ExprNode::Mul(_, rhs) if rhs.is_zero() => ExprNode::Integer(0),
            ExprNode::Mul(lhs, _) if lhs.is_zero() => ExprNode::Integer(0),

            // x * 1 = x
            ExprNode::Mul(lhs, rhs) if rhs.is_one() => lhs.simplify(2),
            ExprNode::Mul(lhs, rhs) if lhs.is_one() => rhs.simplify(2),

            // x * x = x²
            ExprNode::Mul(lhs, rhs) if lhs == rhs => {
                ExprNode::Pow(Arc::new(lhs.simplify(2)), Arc::new(ExprNode::Integer(2)))
            }

            //   x * 2 = 2 * x
            ExprNode::Mul(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                // 2 * 3 = 6
                if let (Some(a), Some(b)) = (as_constant(&l), as_constant(&r)) {
                    return ExprNode::Float(a * b);
                }
                // rhs is Float
                if let Some(_rc) = as_constant(&r) {
                    return ExprNode::Mul(Arc::new(r), Arc::new(l));
                }

                ExprNode::Mul(Arc::new(l), Arc::new(r))
            }

            // x / 1 = x
            ExprNode::Div(lhs, rhs) if rhs.is_one() => lhs.simplify(2),
            // x / x = 1
            ExprNode::Div(lhs, rhs) if lhs == rhs => ExprNode::Integer(1),

            ExprNode::Div(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                // 3 / 2 = 1.5
                if let (Some(a), Some(b)) = (as_constant(&l), as_constant(&r)) {
                    if b.abs() > f64::EPSILON {
                        return ExprNode::Float(a / b);
                    }
                }

                ExprNode::Div(Arc::new(l), Arc::new(r))
            }

            // x^0 = 1
            ExprNode::Pow(_, exp) if exp.is_zero() => ExprNode::Integer(1),
            // x^1 = x
            ExprNode::Pow(base, exp) if exp.is_one() => base.simplify(2),
            // 1^x = 1
            ExprNode::Pow(base, _) if base.is_one() => ExprNode::Integer(1),

            ExprNode::Pow(base, exp) => {
                let b = base.simplify(2);
                let e = exp.simplify(2);

                // 2^3 = 8
                if let (Some(base_val), Some(exp_val)) = (as_constant(&b), as_constant(&e)) {
                    return ExprNode::Float(base_val.powf(exp_val));
                }

                ExprNode::Pow(Arc::new(b), Arc::new(e))
            }

            ExprNode::Neg(inner) => {
                let simplified = inner.simplify(2);
                // -(-x) = x
                if let ExprNode::Neg(inner_inner) = simplified {
                    return (*inner_inner).clone();
                }
                // -(2) = -2
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(-val);
                }
                ExprNode::Neg(Arc::new(simplified))
            }

            // abs(x) = abs(x)
            ExprNode::Abs(inner) => ExprNode::Abs(Arc::new(inner.simplify(2))),
            // sqrt(4) = 2
            ExprNode::Sqrt(inner) => {
                let simplified = inner.simplify(2);
                if let Some(val) = as_constant(&simplified) {
                    if val >= 0.0 {
                        return ExprNode::Float(val.sqrt());
                    }
                }
                ExprNode::Sqrt(Arc::new(simplified))
            }
            ExprNode::Exp(inner) => {
                let simplified = inner.simplify(2);

                // e^0 = 1
                if simplified.is_zero() {
                    return ExprNode::Integer(1);
                }
                // e^2 = e^2
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.exp());
                }
                ExprNode::Exp(Arc::new(simplified))
            }

            ExprNode::Ln(inner) => {
                let simplified = inner.simplify(2);

                // ln(1) = 0
                if simplified.is_one() {
                    return ExprNode::Integer(0);
                }

                // ln(12) = ln(12)
                if let Some(val) = as_constant(&simplified) {
                    if val > 0.0 {
                        return ExprNode::Float(val.ln());
                    }
                }
                ExprNode::Ln(Arc::new(simplified))
            }
            ExprNode::Sin(inner) => {
                let simplified = inner.simplify(2);

                // sin(0) = 0
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.sin());
                }
                ExprNode::Sin(Arc::new(simplified))
            }
            ExprNode::ASin(inner) => {
                let simplified = inner.simplify(2);

                // asin(0)
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.asin());
                }
                ExprNode::ASin(Arc::new(simplified))
            }
            ExprNode::Cos(inner) => {
                let simplified = inner.simplify(2);

                // cos(0) = 1
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.cos());
                }
                ExprNode::Cos(Arc::new(simplified))
            }
            ExprNode::ACos(inner) => {
                let simplified = inner.simplify(2);

                // acos(n)
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.acos());
                }
                ExprNode::ACos(Arc::new(simplified))
            }
            ExprNode::Tan(inner) => {
                let simplified = inner.simplify(2);

                // tan(0) = 0
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.tan());
                }
                ExprNode::Tan(Arc::new(simplified))
            }
            ExprNode::ATan(inner) => {
                let simplified = inner.simplify(2);

                // tan(0) = 0
                if let Some(val) = as_constant(&simplified) {
                    return ExprNode::Float(val.atan());
                }
                ExprNode::ATan(Arc::new(simplified))
            }

            // 其他节点保持不变
            _ => self.clone(),
        }
    }

    /// Applies trigonometric simplifications based on identities.
    ///
    /// This method handles structural simplifications between trigonometric functions
    /// and their inverses.
    ///
    /// # Safety and Domain Constraints
    ///
    /// Simplifications are only applied when they are mathematically sound over the
    /// entire domain or when the structure implies a safe transformation.
    /// - **Safe**: `tan(atan(x)) = x` (valid for all real x).
    /// - **Unsafe (Skipped)**: `asin(sin(x))` is NOT simplified to `x` because it is periodic.
    ///   Simplifying it would erase domain constraints and potentially hide bugs.
    pub fn simplify_trig(&self) -> ExprNode {
        match self {
            ExprNode::Sin(inner) => {
                let simplified = inner.simplify(2);

                // sin(arccos(x)) = sqrt(1 - x²)
                if let ExprNode::ACos(arg) = &simplified {
                    return ExprNode::Sqrt(Arc::new(ExprNode::Sub(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Pow(arg.clone(), Arc::new(ExprNode::Integer(2)))),
                    )));
                }

                // sin(atan(x)) = x/sqrt(1 + x²)
                if let ExprNode::ATan(arg) = &simplified {
                    return ExprNode::Div(
                        arg.clone(),
                        Arc::new(ExprNode::Sqrt(Arc::new(ExprNode::Add(
                            Arc::new(ExprNode::Integer(1)),
                            Arc::new(ExprNode::Pow(arg.clone(), Arc::new(ExprNode::Integer(2)))),
                        )))),
                    );
                }

                ExprNode::Sin(Arc::new(simplified))
            }

            ExprNode::Cos(inner) => {
                let simplified = inner.simplify(2);

                // cos(arcsin(x)) = sqrt(1 - x²)
                if let ExprNode::ASin(arg) = &simplified {
                    return ExprNode::Sqrt(Arc::new(ExprNode::Sub(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Pow(arg.clone(), Arc::new(ExprNode::Integer(2)))),
                    )));
                }

                // cos(atan(x)) = 1/sqrt(1 + x²)
                if let ExprNode::ATan(arg) = &simplified {
                    return ExprNode::Div(
                        Arc::new(ExprNode::Integer(1)),
                        Arc::new(ExprNode::Sqrt(Arc::new(ExprNode::Add(
                            Arc::new(ExprNode::Integer(1)),
                            Arc::new(ExprNode::Pow(arg.clone(), Arc::new(ExprNode::Integer(2)))),
                        )))),
                    );
                }

                ExprNode::Cos(Arc::new(simplified))
            }

            ExprNode::Tan(inner) => {
                let simplified = inner.simplify(2);
                // tan(arctan(x)) = x
                if let ExprNode::ATan(arg) = &simplified {
                    return arg.as_ref().clone();
                }

                ExprNode::Tan(Arc::new(simplified))
            }

            ExprNode::ATan(inner) => {
                let simplified = inner.simplify(2);

                // arctan(tan(x)) = x
                if let ExprNode::Tan(arg) = &simplified {
                    return arg.as_ref().clone();
                }

                ExprNode::ATan(Arc::new(simplified))
            }

            ExprNode::Div(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                // sin(x)/cos(x) = tan(x)
                if let (ExprNode::Sin(sin_arg), ExprNode::Cos(cos_arg)) = (&l, &r) {
                    if sin_arg == cos_arg {
                        return ExprNode::Tan(sin_arg.clone());
                    }
                }

                ExprNode::Div(Arc::new(l), Arc::new(r))
            }

            ExprNode::Mul(lhs, rhs) => {
                let l = lhs.simplify(2);
                let r = rhs.simplify(2);

                // tan(x) * cos(x) = sin(x)
                if let (ExprNode::Tan(tan_arg), ExprNode::Cos(cos_arg)) = (&l, &r) {
                    if tan_arg == cos_arg {
                        return ExprNode::Sin(tan_arg.clone());
                    }
                }
                // cos(x) * tan(x) = sin(x)
                if let (ExprNode::Cos(cos_arg), ExprNode::Tan(tan_arg)) = (&l, &r) {
                    if cos_arg == tan_arg {
                        return ExprNode::Sin(cos_arg.clone());
                    }
                }

                ExprNode::Mul(Arc::new(l), Arc::new(r))
            }
            _ => self.clone(),
        }
    }

    /// Recursively simplifies the expression tree using multiple strategies.
    ///
    /// This serves as the main entry point for expression simplification. It iterates
    /// through basic algebraic rules and trigonometric identities until the expression
    /// stabilizes or the maximum iteration count is reached.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of simplification passes to perform.
    ///   This prevents infinite loops in case of cyclic simplification rules.
    pub fn simplify(&self, max_iterations: i32) -> ExprNode {
        let mut current = self.clone();

        for _ in 0..max_iterations {
            let previous = current.clone();

            // 应用各种简化策略
            current = current.simplify_basic();
            current = current.simplify_trig();

            // 再次应用基础简化（清理中间结果）
            current = current.simplify_basic();

            // 如果不再变化，提前退出
            if current == previous {
                break;
            }
        }

        current
    }

    /// Checks if an expression node represents the integer 0 or a float close to 0.0.
    pub fn is_zero(&self) -> bool {
        matches!(self, ExprNode::Integer(0))
            || matches!(self, ExprNode::Float(f) if f.abs() < f64::EPSILON)
    }

    /// Checks if an expression node represents the integer 1 or a float close to 1.0.
    pub fn is_one(&self) -> bool {
        matches!(self, ExprNode::Integer(1))
            || matches!(self, ExprNode::Float(f) if (f - 1.0).abs() < f64::EPSILON)
    }
}

/// Attempts to extract a constant numerical value from an expression node.
///
/// Returns `Some(f64)` if the node is a literal constant (Integer, Float, Pi, E).
/// Returns `None` if the node involves symbols or operations.
fn as_constant(node: &ExprNode) -> Option<f64> {
    match node {
        ExprNode::Integer(i) => Some(*i as f64),
        ExprNode::Float(f) => Some(*f),
        ExprNode::Pi => Some(std::f64::consts::PI),
        ExprNode::E => Some(std::f64::consts::E),
        _ => None,
    }
}
