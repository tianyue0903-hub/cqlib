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

//! Symbolic matrix representations for quantum gates and circuits.
//!
//! This module mirrors the numerical [`super::circuit_to_matrix`] path, but
//! keeps unresolved gate parameters as symbolic [`Parameter`] expressions
//! instead of evaluating them to concrete `f64` values. This enables
//! deferred evaluation: you can build a symbolic unitary once and then bind
//! different parameter values at a later point without recomputing the
//! full matrix.
//!
//! # Core types
//!
//! - [`SymbolicComplex`] — a complex number whose real and imaginary parts are
//!   independent [`Parameter`] expressions.
//! - [`SymbolicMatrix`] — a dense `Array2<SymbolicComplex>` in the same
//!   row-major layout used by the numerical circuit-matrix API.
//!
//! # Usage
//!
//! ```
//! use cqlib_core::circuit::{Circuit, Parameter, Qubit};
//! use cqlib_core::circuit::symbolic_matrix::{
//!     circuit_to_symbolic_matrix, evaluate_symbolic_matrix,
//! };
//! use std::collections::HashMap;
//!
//! // Build a parametric circuit: RX(theta) on qubit 0.
//! let theta = Parameter::symbol("theta");
//! let mut circuit = Circuit::new(1);
//! circuit.rx(Qubit::new(0), theta).unwrap();
//!
//! // Compute the symbolic unitary — parameters stay symbolic.
//! let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
//!
//! // Bind a concrete value and evaluate to a numerical matrix.
//! let mut bindings = HashMap::new();
//! bindings.insert("theta", std::f64::consts::FRAC_PI_2);
//! let numeric = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
//!
//! // The result is a 2×2 complex matrix, same as circuit_to_matrix.
//! assert_eq!(numeric.shape(), &[2, 2]);
//! ```
//!
//! # Parallelism
//!
//! Gate application is parallelised with **rayon** when the matrix element
//! count exceeds [`PARALLEL_THRESHOLD_OPS`] (2²⁰). Small circuits run on a
//! single thread to avoid scheduling overhead.

use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::error::{CircuitError, ParameterError};
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::{Circuit, Parameter};
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use num_complex::Complex64;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Neg, Sub};

/// Minimum number of matrix elements that triggers parallel gate application
/// via rayon. Below this threshold the work is done on the calling thread to
/// avoid the overhead of thread-pool scheduling.
///
/// Kept in sync with the numerical path in [`super::circuit_to_matrix`].
const PARALLEL_THRESHOLD_OPS: usize = 1 << 20;

/// A complex-valued symbolic expression whose real and imaginary parts are
/// independent [`Parameter`] trees.
///
/// Arithmetic operators (`+`, `-`, `*`, unary `-`) are implemented for all
/// combinations of owned and referenced operands, mirroring the ergonomics
/// of `num_complex::Complex64`.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::symbolic_matrix::SymbolicComplex;
/// use cqlib_core::circuit::Parameter;
///
/// // Construct cos(π/4) + i·sin(π/4)
/// let z = SymbolicComplex::exp_i(Parameter::from(std::f64::consts::FRAC_PI_4));
/// let evaluated = z.evaluate(&None).unwrap();
/// assert!((evaluated.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicComplex {
    /// Real part of the symbolic complex number.
    pub re: Parameter,
    /// Imaginary part of the symbolic complex number.
    pub im: Parameter,
}

/// A dense symbolic matrix in the same row-major ndarray shape used by the
/// numerical circuit matrix API.
pub type SymbolicMatrix = Array2<SymbolicComplex>;

impl Default for SymbolicComplex {
    /// Returns the additive identity `0 + 0i`.
    fn default() -> Self {
        Self::zero()
    }
}

impl SymbolicComplex {
    /// Creates a new `SymbolicComplex` from separate real and imaginary parts.
    ///
    /// Both arguments accept anything that implements [`Into<Parameter>`],
    /// so you can pass `f64`, [`Parameter`], or symbolic expressions directly.
    pub fn new(re: impl Into<Parameter>, im: impl Into<Parameter>) -> Self {
        Self {
            re: re.into(),
            im: im.into(),
        }
    }

    /// Returns the additive identity `0 + 0i`.
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Returns the multiplicative identity `1 + 0i`.
    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Returns the imaginary unit `0 + i`.
    pub fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Creates a purely real symbolic complex number with zero imaginary part.
    pub fn from_real(value: impl Into<Parameter>) -> Self {
        Self::new(value, 0.0)
    }

    /// Creates a `SymbolicComplex` from a concrete [`Complex64`] value.
    pub fn from_complex(value: Complex64) -> Self {
        Self::new(value.re, value.im)
    }

    /// Builds `exp(i·θ)` as `cos(θ) + i·sin(θ)`.
    ///
    /// This is the fundamental building block for rotation-gate matrices
    /// (e.g. `RZ`, `Phase`, `GPhase`) and phase factors.
    pub fn exp_i(theta: impl Into<Parameter>) -> Self {
        let theta = theta.into();
        Self::new(theta.cos(), theta.sin())
    }

    /// Evaluates both the real and imaginary parts under the given parameter
    /// bindings and returns a concrete [`Complex64`].
    ///
    /// Returns [`ParameterError`] if any symbol required by either part is
    /// missing from `bindings`.
    pub fn evaluate(
        &self,
        bindings: &Option<HashMap<&str, f64>>,
    ) -> Result<Complex64, ParameterError> {
        Ok(Complex64::new(
            self.re.evaluate(bindings)?,
            self.im.evaluate(bindings)?,
        ))
    }

    /// Applies algebraic simplification to both the real and imaginary parts.
    ///
    /// Delegates to [`Parameter::simplify`] with domain-safe rules enabled.
    pub fn simplify(&self) -> Result<Self, ParameterError> {
        Ok(Self::new(self.re.simplify()?, self.im.simplify()?))
    }

    /// Replaces every occurrence of `symbol` in both the real and imaginary
    /// parts with the given `value`.
    ///
    /// This is used by [`substitute_symbolic_matrix`] to bind
    /// [`CircuitGate`] parameters.
    pub fn replace(&self, symbol: &str, value: impl Into<Parameter>) -> Self {
        let value = value.into();
        Self::new(
            self.re.replace(symbol, value.clone()),
            self.im.replace(symbol, value),
        )
    }
}

// Arithmetic operator implementations for `SymbolicComplex`.
//
// Each binary operator is implemented for all four ownership combinations
// (owned/owned, owned/ref, ref/owned, ref/ref) to avoid unnecessary clones
// when one or both operands are consumed. Because `Parameter::clone` is O(1)
// (Arc reference-counted), the cost difference is small, but the explicit
// implementations allow the compiler to elide Arc increments when the
// owned value's fields can be moved instead of cloned.

impl Add for SymbolicComplex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Add<&SymbolicComplex> for SymbolicComplex {
    type Output = SymbolicComplex;

    fn add(self, rhs: &SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(self.re + rhs.re.clone(), self.im + rhs.im.clone())
    }
}

impl Add<SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn add(self, rhs: SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(self.re.clone() + rhs.re, self.im.clone() + rhs.im)
    }
}

impl Add<&SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn add(self, rhs: &SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(
            self.re.clone() + rhs.re.clone(),
            self.im.clone() + rhs.im.clone(),
        )
    }
}

/// Subtraction — same four ownership combinations as addition.
impl Sub for SymbolicComplex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Sub<&SymbolicComplex> for SymbolicComplex {
    type Output = SymbolicComplex;

    fn sub(self, rhs: &SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(self.re - rhs.re.clone(), self.im - rhs.im.clone())
    }
}

impl Sub<SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn sub(self, rhs: SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(self.re.clone() - rhs.re, self.im.clone() - rhs.im)
    }
}

impl Sub<&SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn sub(self, rhs: &SymbolicComplex) -> Self::Output {
        SymbolicComplex::new(
            self.re.clone() - rhs.re.clone(),
            self.im.clone() - rhs.im.clone(),
        )
    }
}

/// Complex multiplication: `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`.
///
/// The four ownership variants let the compiler move (rather than clone)
/// `Parameter` fields from owned operands where possible.
impl Mul for SymbolicComplex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re * rhs.im + self.im * rhs.re;
        Self::new(re, im)
    }
}

impl Mul<&SymbolicComplex> for SymbolicComplex {
    type Output = SymbolicComplex;

    fn mul(self, rhs: &SymbolicComplex) -> Self::Output {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re * rhs.im.clone() + self.im * rhs.re.clone();
        SymbolicComplex::new(re, im)
    }
}

impl Mul<SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn mul(self, rhs: SymbolicComplex) -> Self::Output {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re.clone() * rhs.im + self.im.clone() * rhs.re;
        SymbolicComplex::new(re, im)
    }
}

impl Mul<&SymbolicComplex> for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn mul(self, rhs: &SymbolicComplex) -> Self::Output {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.re.clone() * rhs.im.clone() + self.im.clone() * rhs.re.clone();
        SymbolicComplex::new(re, im)
    }
}

/// Unary negation — owned and reference variants.
impl Neg for SymbolicComplex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl Neg for &SymbolicComplex {
    type Output = SymbolicComplex;

    fn neg(self) -> Self::Output {
        SymbolicComplex::new(-self.re.clone(), -self.im.clone())
    }
}

/// Formats the symbolic complex number in a human-readable form.
///
/// - Pure real: `3.14`
/// - Pure imaginary: `2.0i`
/// - Mixed: `1.0 + 2.0i` or `1.0 - 2.0i` (negative imaginary part)
impl fmt::Display for SymbolicComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.is_zero() {
            write!(f, "{}", self.re)
        } else if self.re.is_zero() {
            write!(f, "{}i", self.im)
        } else {
            let im_str = self.im.to_string();
            if im_str.starts_with('-') {
                write!(f, "{} - {}i", self.re, im_str.trim_start_matches('-'))
            } else {
                write!(f, "{} + {}i", self.re, im_str)
            }
        }
    }
}

/// Converts a numerical complex matrix into a [`SymbolicMatrix`] by wrapping
/// each element with [`SymbolicComplex::from_complex`].
fn symbolic_matrix_from_numeric(matrix: &Array2<Complex64>) -> SymbolicMatrix {
    matrix.mapv(SymbolicComplex::from_complex)
}

/// Returns a `dim × dim` symbolic identity matrix.
fn symbolic_eye(dim: usize) -> SymbolicMatrix {
    let mut matrix = Array2::from_elem((dim, dim), SymbolicComplex::zero());
    for i in 0..dim {
        matrix[[i, i]] = SymbolicComplex::one();
    }
    matrix
}

/// Returns `θ / 2` as a new [`Parameter`] expression.
fn half(theta: &Parameter) -> Parameter {
    theta.clone() / 2.0
}

/// Returns `cos(θ/2)` as a purely real [`SymbolicComplex`].
///
/// This is the diagonal-element pattern shared by `RX`, `RY`, `RXX`, etc.
fn cos_half(theta: &Parameter) -> SymbolicComplex {
    SymbolicComplex::from_real(half(theta).cos())
}

/// Returns `sin(θ/2)` as a [`Parameter`].
///
/// The caller decides whether to wrap this as a real or imaginary component.
fn sin_half(theta: &Parameter) -> Parameter {
    half(theta).sin()
}

/// Returns `−i · value`, i.e. a purely imaginary [`SymbolicComplex`] with
/// negative imaginary coefficient.
///
/// Used for the off-diagonal elements of `RX`, `RXX`, `CRX`, etc.
fn neg_i_times(value: Parameter) -> SymbolicComplex {
    SymbolicComplex::new(0.0, -1.0 * value)
}

/// Returns `i · value`, i.e. a purely imaginary [`SymbolicComplex`] with
/// positive imaginary coefficient.
///
/// Used for the off-diagonal elements of `RYY`, `RZX`, etc.
fn i_times(value: Parameter) -> SymbolicComplex {
    SymbolicComplex::new(0.0, value)
}

/// Returns `exp(−i·θ)` as `cos(θ) − i·sin(θ)`.
///
/// Used for the diagonal phase factors of `RZ`, `RZZ`, `CRZ`, etc.
fn exp_neg_i(theta: Parameter) -> SymbolicComplex {
    SymbolicComplex::exp_i(-theta)
}

/// Validates that the number of supplied `params` matches the gate's
/// declared [`StandardGate::num_params`].
///
/// Returns [`CircuitError::ParameterCountMismatch`] on failure.
fn validate_params(gate: StandardGate, params: &[Parameter]) -> Result<(), CircuitError> {
    let expected = gate.num_params();
    if params.len() != expected {
        return Err(CircuitError::ParameterCountMismatch {
            expected,
            actual: params.len(),
        });
    }
    Ok(())
}

/// Returns the symbolic unitary matrix for a [`StandardGate`].
///
/// Non-parametric gates (H, X, SWAP, CCX, …) delegate to the numerical
/// [`StandardGate::matrix`] and convert the result via
/// [`symbolic_matrix_from_numeric`]. Parametric gates (RX, RY, RZ, U, …)
/// build their matrices symbolically so that parameters remain as
/// [`Parameter`] expressions for deferred evaluation.
///
/// # Errors
///
/// Returns [`CircuitError::ParameterCountMismatch`] if the number of
/// parameters does not match the gate's declared count.
pub fn standard_gate_symbolic_matrix(
    gate: StandardGate,
    params: &[Parameter],
) -> Result<SymbolicMatrix, CircuitError> {
    validate_params(gate, params)?;
    let z = SymbolicComplex::zero();
    let o = SymbolicComplex::one();
    let i = SymbolicComplex::i();
    let neg_i = -i.clone();
    let h = SymbolicComplex::from_real(1.0 / std::f64::consts::SQRT_2);

    Ok(match gate {
        StandardGate::H
        | StandardGate::I
        | StandardGate::S
        | StandardGate::SDG
        | StandardGate::T
        | StandardGate::TDG
        | StandardGate::X
        | StandardGate::Y
        | StandardGate::Z
        | StandardGate::X2P
        | StandardGate::X2M
        | StandardGate::Y2P
        | StandardGate::Y2M
        | StandardGate::SWAP
        | StandardGate::CX
        | StandardGate::CY
        | StandardGate::CZ
        | StandardGate::CCX => {
            let numeric = gate
                .matrix(&[])
                .map_err(|_| CircuitError::NoMatrixRepresentation)?;
            symbolic_matrix_from_numeric(numeric.as_ref())
        }
        StandardGate::RX => {
            let c = cos_half(&params[0]);
            let s = neg_i_times(sin_half(&params[0]));
            ndarray::array![[c.clone(), s.clone()], [s, c]]
        }
        StandardGate::RY => {
            let c = cos_half(&params[0]);
            let s = SymbolicComplex::from_real(sin_half(&params[0]));
            ndarray::array![[c.clone(), -s.clone()], [s, c]]
        }
        StandardGate::RZ => {
            let h = half(&params[0]);
            ndarray::array![
                [exp_neg_i(h.clone()), z.clone()],
                [z, SymbolicComplex::exp_i(h)]
            ]
        }
        StandardGate::Phase => ndarray::array![
            [o.clone(), z.clone()],
            [z, SymbolicComplex::exp_i(params[0].clone())]
        ],
        StandardGate::GPhase => {
            let phase = SymbolicComplex::exp_i(params[0].clone());
            ndarray::array![[phase.clone(), z.clone()], [z, phase]]
        }
        StandardGate::RXX => {
            let c = cos_half(&params[0]);
            let s = neg_i_times(sin_half(&params[0]));
            ndarray::array![
                [c.clone(), z.clone(), z.clone(), s.clone()],
                [z.clone(), c.clone(), s.clone(), z.clone()],
                [z.clone(), s.clone(), c.clone(), z.clone()],
                [s, z.clone(), z, c]
            ]
        }
        StandardGate::RYY => {
            let c = cos_half(&params[0]);
            let s = i_times(sin_half(&params[0]));
            let ns = -s.clone();
            ndarray::array![
                [c.clone(), z.clone(), z.clone(), s.clone()],
                [z.clone(), c.clone(), ns.clone(), z.clone()],
                [z.clone(), ns, c.clone(), z.clone()],
                [s, z.clone(), z, c]
            ]
        }
        StandardGate::RZZ => {
            let h = half(&params[0]);
            let exp_neg = exp_neg_i(h.clone());
            let exp_pos = SymbolicComplex::exp_i(h);
            ndarray::array![
                [exp_neg.clone(), z.clone(), z.clone(), z.clone()],
                [z.clone(), exp_pos.clone(), z.clone(), z.clone()],
                [z.clone(), z.clone(), exp_pos, z.clone()],
                [z.clone(), z.clone(), z, exp_neg]
            ]
        }
        StandardGate::RZX => {
            let c = cos_half(&params[0]);
            let s = i_times(sin_half(&params[0]));
            let ns = -s.clone();
            ndarray::array![
                [c.clone(), ns.clone(), z.clone(), z.clone()],
                [ns, c.clone(), z.clone(), z.clone()],
                [z.clone(), z.clone(), c.clone(), s.clone()],
                [z.clone(), z.clone(), s, c]
            ]
        }
        StandardGate::CRX => {
            let c = cos_half(&params[0]);
            let s = neg_i_times(sin_half(&params[0]));
            ndarray::array![
                [o.clone(), z.clone(), z.clone(), z.clone()],
                [z.clone(), o.clone(), z.clone(), z.clone()],
                [z.clone(), z.clone(), c.clone(), s.clone()],
                [z.clone(), z.clone(), s, c]
            ]
        }
        StandardGate::CRY => {
            let c = cos_half(&params[0]);
            let s = SymbolicComplex::from_real(sin_half(&params[0]));
            ndarray::array![
                [o.clone(), z.clone(), z.clone(), z.clone()],
                [z.clone(), o.clone(), z.clone(), z.clone()],
                [z.clone(), z.clone(), c.clone(), -s.clone()],
                [z.clone(), z.clone(), s, c]
            ]
        }
        StandardGate::CRZ => {
            let h = half(&params[0]);
            ndarray::array![
                [o.clone(), z.clone(), z.clone(), z.clone()],
                [z.clone(), o.clone(), z.clone(), z.clone()],
                [z.clone(), z.clone(), exp_neg_i(h.clone()), z.clone()],
                [z.clone(), z.clone(), z, SymbolicComplex::exp_i(h)]
            ]
        }
        StandardGate::RXY => {
            let c = cos_half(&params[0]);
            let s = SymbolicComplex::from_real(sin_half(&params[0]));
            let upper = neg_i.clone() * exp_neg_i(params[1].clone()) * s.clone();
            let lower = neg_i * SymbolicComplex::exp_i(params[1].clone()) * s;
            ndarray::array![[c.clone(), upper], [lower, c]]
        }
        StandardGate::U => {
            let c = cos_half(&params[0]);
            let s = SymbolicComplex::from_real(sin_half(&params[0]));
            let exp_phi = SymbolicComplex::exp_i(params[1].clone());
            let exp_lambda = SymbolicComplex::exp_i(params[2].clone());
            let exp_phi_lambda = SymbolicComplex::exp_i(params[1].clone() + params[2].clone());
            ndarray::array![
                [c.clone(), -(exp_lambda * s.clone())],
                [exp_phi * s, exp_phi_lambda * c]
            ]
        }
        StandardGate::XY => {
            let upper = neg_i.clone() * exp_neg_i(params[0].clone());
            let lower = neg_i * SymbolicComplex::exp_i(params[0].clone());
            ndarray::array![[z, upper], [lower, SymbolicComplex::zero()]]
        }
        StandardGate::XY2P => {
            let upper = neg_i.clone() * exp_neg_i(params[0].clone()) * h.clone();
            let lower = neg_i * SymbolicComplex::exp_i(params[0].clone()) * h.clone();
            ndarray::array![[h.clone(), upper], [lower, h]]
        }
        StandardGate::XY2M => {
            let upper = i.clone() * exp_neg_i(params[0].clone()) * h.clone();
            let lower = i * SymbolicComplex::exp_i(params[0].clone()) * h.clone();
            ndarray::array![[h.clone(), upper], [lower, h]]
        }
        StandardGate::FSIM => {
            let c = SymbolicComplex::from_real(params[0].cos());
            let s = neg_i_times(params[0].sin());
            let phase = exp_neg_i(params[1].clone());
            ndarray::array![
                [o.clone(), z.clone(), z.clone(), z.clone()],
                [z.clone(), c.clone(), s.clone(), z.clone()],
                [z.clone(), s, c, z.clone()],
                [z.clone(), z.clone(), z, phase]
            ]
        }
    })
}

/// Constructs a controlled-symbolic matrix by embedding `base` into the
/// bottom-right block of a larger identity matrix.
///
/// For `num_ctrls` control qubits the resulting dimension is
/// `base_dim × 2^num_ctrls`. All control-basis states map to identity
/// rows; only when every control qubit is `|1⟩` does the base gate act.
fn control_matrix(base: &SymbolicMatrix, num_ctrls: usize) -> SymbolicMatrix {
    if num_ctrls == 0 {
        return base.clone();
    }

    let base_dim = base.nrows();
    let total_dim = base_dim << num_ctrls;
    let mut matrix = symbolic_eye(total_dim);
    let start = total_dim - base_dim;

    for row in 0..base_dim {
        for col in 0..base_dim {
            matrix[[start + row, start + col]] = base[[row, col]].clone();
        }
    }

    matrix
}

/// Internal prefix for temporary symbol names used during two-phase
/// substitution (see [`substitute_symbolic_matrix`]).
///
/// User-defined symbol names containing this prefix are rejected to prevent
/// collisions with the temporary names generated by the substitution algorithm.
const INTERNAL_SUB_PREFIX: &str = "__cqlib_internal_sub";

/// Simultaneously substitutes multiple symbols in a [`SymbolicMatrix`].
///
/// When replacement values reference symbols that are also keys in the
/// `replacements` map (e.g. swapping `a` and `b`), a two-phase approach is
/// used: symbols are first renamed to temporary names prefixed with
/// [`INTERNAL_SUB_PREFIX`], then the temporary names are replaced with the
/// actual values. This avoids the non-deterministic ordering artefacts that
/// would arise from sequential substitution.
///
/// # Errors
///
/// - [`CircuitError::InvalidOperation`] if any key or replacement value
///   contains [`INTERNAL_SUB_PREFIX`], which would collide with the
///   algorithm's temporary symbol names.
fn substitute_symbolic_matrix(
    matrix: SymbolicMatrix,
    replacements: &HashMap<String, Parameter>,
) -> Result<SymbolicMatrix, CircuitError> {
    if replacements.is_empty() {
        return Ok(matrix);
    }

    // Guard against symbol names that collide with our internal temp prefix.
    for key in replacements.keys() {
        if key.contains(INTERNAL_SUB_PREFIX) {
            return Err(CircuitError::InvalidOperation(format!(
                "symbol name '{}' collides with internal substitution prefix '{}'",
                key, INTERNAL_SUB_PREFIX
            )));
        }
    }
    for param in replacements.values() {
        for sym in param.get_symbols() {
            if sym.contains(INTERNAL_SUB_PREFIX) {
                return Err(CircuitError::InvalidOperation(format!(
                    "symbol name '{}' in replacement value collides with internal substitution prefix '{}'",
                    sym, INTERNAL_SUB_PREFIX
                )));
            }
        }
    }

    let dim = matrix.raw_dim();
    let (mut raw, _) = matrix.into_raw_vec_and_offset();

    // Fast path: single replacement never needs temp symbols.
    if replacements.len() == 1 {
        let (symbol, replacement) = replacements.iter().next().unwrap();
        raw.par_iter_mut().for_each(|v| {
            *v = v.replace(symbol, replacement.clone());
        });
        return Ok(Array2::from_shape_vec(dim, raw).expect("valid shape for owned matrix"));
    }

    // Check whether any replacement value contains a symbol that is also a key.
    // If there is no overlap, we can skip the expensive two-phase temp-symbol dance.
    let keys: HashSet<&str> = replacements.keys().map(|s| s.as_str()).collect();
    let has_overlap = replacements.values().any(|param| {
        param
            .get_symbols()
            .iter()
            .any(|sym| keys.contains(sym.as_str()))
    });

    if has_overlap {
        let temp_replacements: Vec<(String, String, Parameter)> = replacements
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    format!("{}{}", INTERNAL_SUB_PREFIX, k),
                    v.clone(),
                )
            })
            .collect();

        raw.par_iter_mut().for_each(|v| {
            for (symbol, temp_key, _) in &temp_replacements {
                *v = v.replace(symbol, Parameter::symbol(temp_key));
            }
            for (_, temp_key, replacement) in &temp_replacements {
                *v = v.replace(temp_key, replacement.clone());
            }
        });
    } else {
        let reps: Vec<(String, Parameter)> = replacements
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        raw.par_iter_mut().for_each(|v| {
            for (symbol, replacement) in &reps {
                *v = v.replace(symbol, replacement.clone());
            }
        });
    }

    Ok(Array2::from_shape_vec(dim, raw).expect("valid shape for owned matrix"))
}

/// Resolves a slice of [`CircuitParam`] values into concrete [`Parameter`]
/// expressions.
///
/// - [`CircuitParam::Fixed`] is converted directly from its `f64` value.
/// - [`CircuitParam::Index`] is looked up in the circuit's parameter table.
///
/// # Errors
///
/// Returns [`CircuitError::InvalidParameterIndex`] if an index is out of
/// bounds for the circuit's parameter set.
fn resolve_params(
    circuit: &Circuit,
    params: &[CircuitParam],
) -> Result<Vec<Parameter>, CircuitError> {
    params
        .iter()
        .map(|param| match param {
            CircuitParam::Fixed(value) => Ok(Parameter::from(*value)),
            CircuitParam::Index(idx) => circuit
                .parameters()
                .get_index(*idx as usize)
                .cloned()
                .ok_or(CircuitError::InvalidParameterIndex(*idx)),
        })
        .collect()
}

/// Applies a gate matrix to the target qubit positions of a state matrix.
///
/// Dispatches to an optimised code path based on the number of target
/// qubits:
///
/// | `bits.len()` | Code path                    |
/// |-------------|------------------------------|
/// | 1           | [`apply_single_qubit_gate`]  |
/// | 2           | [`apply_two_qubit_gate`]     |
/// | 3+          | [`apply_general_gate`]       |
///
/// The `bits` parameter uses the system's **Little-Endian** convention:
/// qubit 0 is the least-significant bit. Callers that receive gate-local
/// Big-Endian bit order (e.g. from [`StandardGate`] matrices) must reverse
/// the bits before calling this function.
pub fn apply_gate_to_matrix(matrix: &mut SymbolicMatrix, gate: &SymbolicMatrix, bits: &[usize]) {
    match bits.len() {
        1 => apply_single_qubit_gate(matrix, gate, bits[0]),
        2 => apply_two_qubit_gate(matrix, gate, bits[0], bits[1]),
        _ => apply_general_gate(matrix, gate, bits),
    }
}

/// Helper struct for raw pointer access to split mutable borrow across threads.
///
/// # Safety
/// This struct is `Send` + `Sync` only because the caller guarantees that
/// concurrent accesses target disjoint row indices. Each worker thread must
/// process a distinct subset of row indices so that no two threads ever write
/// to the same memory location.
struct UnsafeSymbolicSlice<'a> {
    ptr: *mut SymbolicComplex,
    _marker: PhantomData<&'a mut [SymbolicComplex]>,
}

unsafe impl<'a> Sync for UnsafeSymbolicSlice<'a> {}
unsafe impl<'a> Send for UnsafeSymbolicSlice<'a> {}

impl<'a> UnsafeSymbolicSlice<'a> {
    fn new(slice: &'a mut [SymbolicComplex]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    /// Caller must ensure that the returned pointer is only used while the
    /// underlying slice is alive, and that concurrent accesses to the same
    /// row index do not occur.
    unsafe fn row_ptr(&self, row_idx: usize, cols: usize) -> *mut SymbolicComplex {
        unsafe { self.ptr.add(row_idx * cols) }
    }
}

/// Applies a single-qubit gate to the given bit position of `matrix`.
///
/// For each pair of rows that differ only in the target bit, the gate is
/// multiplied as a 2×2 matrix transformation:
///
/// ```text
/// [v0']   [u00 u01] [v0]
/// [v1'] = [u10 u11] [v1]
/// ```
///
/// Uses [`UnsafeSymbolicSlice`] and rayon parallelism when the matrix is
/// large enough (see [`PARALLEL_THRESHOLD_OPS`]).
fn apply_single_qubit_gate(matrix: &mut SymbolicMatrix, gate: &SymbolicMatrix, bit: usize) {
    let dim = matrix.nrows();
    let cols = matrix.ncols();
    let step = 1usize << bit;
    let total_ops = dim.saturating_mul(cols);
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let u00 = &gate[[0, 0]];
    let u01 = &gate[[0, 1]];
    let u10 = &gate[[1, 0]];
    let u11 = &gate[[1, 1]];

    let slice = matrix
        .as_slice_mut()
        .expect("Symbolic matrix must be contiguous");
    let unsafe_slice = UnsafeSymbolicSlice::new(slice);

    let process_block = |i: usize| {
        // SAFETY: Each worker processes a unique block starting at `i` and
        // touches rows `i + j` and `i + j + step` where `j < step`. Because
        // the outer iterator steps by `step * 2`, no two workers can ever
        // access the same row index, so there is no aliasing.
        unsafe {
            for j in 0..step {
                let r0_idx = i + j;
                let r1_idx = r0_idx + step;

                let r0_ptr = unsafe_slice.row_ptr(r0_idx, cols);
                let r1_ptr = unsafe_slice.row_ptr(r1_idx, cols);

                for col in 0..cols {
                    let v0 = (*r0_ptr.add(col)).clone();
                    let v1 = (*r1_ptr.add(col)).clone();

                    *r0_ptr.add(col) = u00 * &v0 + u01 * &v1;
                    *r1_ptr.add(col) = u10 * &v0 + u11 * &v1;
                }
            }
        }
    };

    if parallel {
        // SAFETY: `into_par_iter().step_by(step * 2)` partitions the row
        // index space into disjoint blocks, each processed by a single
        // worker. The raw pointers derived from `UnsafeSymbolicSlice` are
        // never aliased across workers, satisfying Rust's safety rules.
        (0..dim)
            .into_par_iter()
            .step_by(step * 2)
            .for_each(process_block);
    } else {
        (0..dim).step_by(step * 2).for_each(process_block);
    }
}

/// Applies a two-qubit gate to the given bit positions of `matrix`.
///
/// For each group of four rows that differ only in the two target bits,
/// the gate is multiplied as a 4×4 matrix transformation. The index
/// mapping uses a bit-insertion scheme to compute the base row index
/// from the compact iteration variable `i`.
///
/// Uses [`UnsafeSymbolicSlice`] and rayon parallelism when the matrix is
/// large enough (see [`PARALLEL_THRESHOLD_OPS`]).
fn apply_two_qubit_gate(matrix: &mut SymbolicMatrix, gate: &SymbolicMatrix, b0: usize, b1: usize) {
    let dim = matrix.nrows();
    let cols = matrix.ncols();
    let (low, high) = if b0 < b1 { (b0, b1) } else { (b1, b0) };
    let mask_low = (1usize << low) - 1;
    let mask_high = (1usize << high) - 1;
    let off0 = 1usize << b0;
    let off1 = 1usize << b1;
    let loop_limit = dim >> 2;
    let total_ops = dim.saturating_mul(cols);
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let g00 = &gate[[0, 0]];
    let g01 = &gate[[0, 1]];
    let g02 = &gate[[0, 2]];
    let g03 = &gate[[0, 3]];
    let g10 = &gate[[1, 0]];
    let g11 = &gate[[1, 1]];
    let g12 = &gate[[1, 2]];
    let g13 = &gate[[1, 3]];
    let g20 = &gate[[2, 0]];
    let g21 = &gate[[2, 1]];
    let g22 = &gate[[2, 2]];
    let g23 = &gate[[2, 3]];
    let g30 = &gate[[3, 0]];
    let g31 = &gate[[3, 1]];
    let g32 = &gate[[3, 2]];
    let g33 = &gate[[3, 3]];

    let slice = matrix
        .as_slice_mut()
        .expect("Symbolic matrix must be contiguous");
    let unsafe_slice = UnsafeSymbolicSlice::new(slice);

    let process_idx = |i: usize| {
        let left_part = (i & !mask_low) << 1;
        let right_part = i & mask_low;
        let tmp = left_part | right_part;

        let left_final = (tmp & !mask_high) << 1;
        let right_final = tmp & mask_high;
        let base = left_final | right_final;

        let r0_idx = base;
        let r1_idx = base | off0;
        let r2_idx = base | off1;
        let r3_idx = base | off0 | off1;

        // SAFETY: Each `i` in the outer iterator maps to a unique `base`,
        // and the four derived row indices (`base`, `base|off0`, `base|off1`,
        // `base|off0|off1`) never overlap with those of any other `i`. Thus
        // no two workers touch the same row index, so there is no aliasing.
        unsafe {
            let p0 = unsafe_slice.row_ptr(r0_idx, cols);
            let p1 = unsafe_slice.row_ptr(r1_idx, cols);
            let p2 = unsafe_slice.row_ptr(r2_idx, cols);
            let p3 = unsafe_slice.row_ptr(r3_idx, cols);

            for col in 0..cols {
                let v0 = (*p0.add(col)).clone();
                let v1 = (*p1.add(col)).clone();
                let v2 = (*p2.add(col)).clone();
                let v3 = (*p3.add(col)).clone();

                *p0.add(col) = g00 * &v0 + g01 * &v1 + g02 * &v2 + g03 * &v3;
                *p1.add(col) = g10 * &v0 + g11 * &v1 + g12 * &v2 + g13 * &v3;
                *p2.add(col) = g20 * &v0 + g21 * &v1 + g22 * &v2 + g23 * &v3;
                *p3.add(col) = g30 * &v0 + g31 * &v1 + g32 * &v2 + g33 * &v3;
            }
        }
    };

    if parallel {
        // SAFETY: `into_par_iter()` distributes distinct `i` values across
        // workers. Because the mapping from `i` to the four touched rows is
        // injective, different workers never access the same row index. The
        // raw pointers derived from `UnsafeSymbolicSlice` are therefore
        // non-aliased across workers.
        (0..loop_limit).into_par_iter().for_each(process_idx);
    } else {
        (0..loop_limit).for_each(process_idx);
    }
}

/// Applies an n-qubit gate to the given bit positions of `matrix`.
///
/// This is the general (unoptimised) code path used when the gate acts on
/// three or more qubits. It iterates over all `2^n` row groups and
/// performs a full matrix-vector multiply for each column.
///
/// # Parallelism
///
/// When the matrix element count exceeds [`PARALLEL_THRESHOLD_OPS`],
/// each worker thread receives its own scratch buffers (`row_ptrs` and
/// `input`) via [`rayon::iter::ParallelIterator::for_each_init`].
fn apply_general_gate(matrix: &mut SymbolicMatrix, gate: &SymbolicMatrix, bits: &[usize]) {
    let dim = matrix.nrows();
    let cols = matrix.ncols();
    let num_targets = bits.len();
    let gate_dim = 1usize << num_targets;

    let mut sorted_bits: SmallVec<[usize; 8]> = bits.iter().copied().collect();
    sorted_bits.sort();

    let mut gate_offsets = vec![0usize; gate_dim];
    for (k, offset_ref) in gate_offsets.iter_mut().enumerate() {
        let mut offset = 0usize;
        for (j, &physical_bit) in bits.iter().enumerate() {
            if (k >> j) & 1 == 1 {
                offset |= 1usize << physical_bit;
            }
        }
        *offset_ref = offset;
    }

    let loop_limit = dim >> num_targets;
    let total_ops = dim.saturating_mul(cols);
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let slice = matrix
        .as_slice_mut()
        .expect("Symbolic matrix must be contiguous");
    let unsafe_slice = UnsafeSymbolicSlice::new(slice);

    let process_idx =
        |i: usize, row_ptrs: &mut Vec<*mut SymbolicComplex>, input: &mut Vec<SymbolicComplex>| {
            let mut base = i;
            for &q in &sorted_bits {
                let mask = (1usize << q) - 1;
                let left = (base & !mask) << 1;
                let right = base & mask;
                base = left | right;
            }

            // SAFETY: Each `i` maps to a unique `base`, and `gate_offsets`
            // are fixed non-overlapping bit patterns. Therefore the set of
            // row indices `{base | offset}` for a given `i` is disjoint from
            // the set for any other `i`. No two workers ever write to the
            // same row, so aliasing cannot occur.
            unsafe {
                row_ptrs.clear();
                for offset in gate_offsets.iter().take(gate_dim) {
                    row_ptrs.push(unsafe_slice.row_ptr(base | offset, cols));
                }

                for col in 0..cols {
                    for g in 0..gate_dim {
                        input[g] = (*row_ptrs[g].add(col)).clone();
                    }

                    for row in 0..gate_dim {
                        let mut sum = SymbolicComplex::zero();
                        for col_gate in 0..gate_dim {
                            sum = sum + &gate[[row, col_gate]] * &input[col_gate];
                        }
                        *row_ptrs[row].add(col) = sum;
                    }
                }
            }
        };

    if parallel {
        // SAFETY: `into_par_iter()` partitions the range `0..loop_limit`
        // across workers. Because each `i` produces a disjoint set of row
        // indices, the per-worker `row_ptrs` never alias. `for_each_init`
        // further guarantees that each worker owns its own scratch buffers.
        (0..loop_limit).into_par_iter().for_each_init(
            || {
                (
                    Vec::<*mut SymbolicComplex>::with_capacity(gate_dim),
                    vec![SymbolicComplex::zero(); gate_dim],
                )
            },
            |(row_ptrs, input), i| process_idx(i, row_ptrs, input),
        );
    } else {
        let mut row_ptrs = Vec::<*mut SymbolicComplex>::with_capacity(gate_dim);
        let mut input = vec![SymbolicComplex::zero(); gate_dim];
        for i in 0..loop_limit {
            process_idx(i, &mut row_ptrs, &mut input);
        }
    }
}

/// Computes the symbolic unitary matrix representation of a quantum circuit.
///
/// This is the symbolic counterpart of [`super::circuit_to_matrix`]: instead
/// of evaluating gate parameters to `f64` values immediately, it preserves
/// them as [`Parameter`] expressions so that the resulting matrix can be
/// evaluated later with different bindings via [`evaluate_symbolic_matrix`].
///
/// # Qubit ordering
///
/// The `qubits_order` parameter controls which qubit maps to which bit
/// position in the matrix:
///
/// - `None` — qubits are sorted by index in ascending order (qubit 0 → bit 0).
/// - `Some(order)` — the provided slice defines the bit assignment from
///   most-significant to least-significant.
///
/// The order must contain exactly the same set of qubit indices as the
/// circuit, with no duplicates.
///
/// # Errors
///
/// - [`CircuitError::InvalidOperation`] if `qubits_order` does not match
///   the circuit's qubit set, or if the circuit contains control-flow
///   operations.
/// - [`CircuitError::NoMatrixRepresentation`] for non-unitary operations
///   (measure, reset) or gates without a matrix definition.
/// - [`CircuitError::ParameterCountMismatch`] if a gate or circuit gate
///   receives the wrong number of parameters.
/// - [`CircuitError::QubitNotFound`] if an operation references a qubit
///   not present in the circuit.
pub fn circuit_to_symbolic_matrix(
    circuit: &Circuit,
    qubits_order: Option<&[usize]>,
) -> Result<SymbolicMatrix, CircuitError> {
    let circuit_qubits: Vec<usize> = circuit.qubits().iter().map(|q| q.index()).collect();
    let num_qubits = circuit_qubits.len();
    let dim = 1usize.checked_shl(num_qubits as u32).ok_or_else(|| {
        CircuitError::InvalidOperation(format!(
            "cannot build matrix for {num_qubits} qubits: dimension overflows usize"
        ))
    })?;
    dim.checked_mul(dim).ok_or_else(|| {
        CircuitError::InvalidOperation(format!(
            "cannot build matrix for {num_qubits} qubits: matrix element count overflows usize"
        ))
    })?;

    let target_order: Vec<usize> = match qubits_order {
        Some(order) => {
            let circuit_set: HashSet<usize> = circuit_qubits.iter().copied().collect();
            let order_set: HashSet<usize> = order.iter().copied().collect();
            if circuit_set != order_set || circuit_set.len() != order.len() {
                return Err(CircuitError::InvalidOperation(format!(
                    "qubits_order mismatch! Circuit has {:?}, but order provided is {:?}",
                    circuit_qubits, order
                )));
            }
            order.to_vec()
        }
        None => {
            let mut sorted = circuit_qubits.clone();
            sorted.sort();
            sorted
        }
    };

    let qubit_bit_map: HashMap<usize, usize> = target_order
        .iter()
        .enumerate()
        .map(|(i, &q_id)| (q_id, i))
        .collect();

    let mut matrix = symbolic_eye(dim);

    for op in circuit.operations() {
        let bits: SmallVec<[usize; 3]> = op
            .qubits
            .iter()
            .map(|q| {
                qubit_bit_map
                    .get(&q.index())
                    .copied()
                    .ok_or(CircuitError::QubitNotFound(q.id()))
            })
            .collect::<Result<_, _>>()?;
        let params = resolve_params(circuit, &op.params)?;

        match &op.instruction {
            Instruction::Standard(gate) => {
                let gate_matrix = standard_gate_symbolic_matrix(*gate, &params)?;
                let reversed_bits: SmallVec<[usize; 3]> = bits.iter().copied().rev().collect();
                apply_gate_to_matrix(&mut matrix, &gate_matrix, &reversed_bits);
            }
            Instruction::McGate(mc_gate) => {
                let base = standard_gate_symbolic_matrix(*mc_gate.base_gate(), &params)?;
                let gate_matrix = control_matrix(&base, mc_gate.num_ctrl_qubits());
                let reversed_bits: SmallVec<[usize; 3]> = bits.iter().copied().rev().collect();
                apply_gate_to_matrix(&mut matrix, &gate_matrix, &reversed_bits);
            }
            Instruction::UnitaryGate(u_gate) => {
                if let Some(gate_matrix) = u_gate.matrix() {
                    // UnitaryGate matrices follow the standard gate-local Big-Endian convention,
                    // so we reverse bits to align with the system's Little-Endian layout.
                    let sub_matrix = symbolic_matrix_from_numeric(gate_matrix);
                    let reversed_bits: SmallVec<[usize; 3]> = bits.iter().copied().rev().collect();
                    apply_gate_to_matrix(&mut matrix, &sub_matrix, &reversed_bits);
                } else if let Some(sub_circuit) = u_gate.circuit().as_ref() {
                    // Sub-circuit matrices are already built in the system's Little-Endian basis,
                    // so we apply them directly without reversing bits.
                    let sub_matrix = circuit_to_symbolic_matrix(sub_circuit.circuit(), None)?;
                    apply_gate_to_matrix(&mut matrix, &sub_matrix, &bits);
                } else {
                    return Err(CircuitError::NoMatrixRepresentation);
                }
            }
            Instruction::CircuitGate(circuit_gate) => {
                let symbols = circuit_gate.symbols();
                let expected = symbols.len();
                let actual = params.len();
                if actual != expected {
                    return Err(CircuitError::ParameterCountMismatch { expected, actual });
                }
                let replacements: HashMap<String, Parameter> = symbols
                    .iter()
                    .cloned()
                    .zip(params.iter().cloned())
                    .collect();
                let sub_matrix =
                    circuit_to_symbolic_matrix(circuit_gate.circuit().circuit(), None)?;
                let sub_matrix = substitute_symbolic_matrix(sub_matrix, &replacements)?;
                apply_gate_to_matrix(&mut matrix, &sub_matrix, &bits);
            }
            Instruction::ControlFlowGate(_) => {
                return Err(CircuitError::InvalidOperation(
                    "control-flow operations do not have an unconditional matrix representation"
                        .to_string(),
                ));
            }
            Instruction::Directive(directive) => match directive {
                crate::circuit::Directive::Barrier => continue,
                crate::circuit::Directive::Measure | crate::circuit::Directive::Reset => {
                    return Err(CircuitError::NoMatrixRepresentation);
                }
            },
            Instruction::Delay => continue,
        }
    }

    let global_phase = circuit.global_phase();
    if !global_phase.is_zero() {
        let phase = SymbolicComplex::exp_i(global_phase);
        matrix
            .as_slice_mut()
            .expect("symbolic matrix must be contiguous")
            .par_iter_mut()
            .for_each(|value| {
                let old = std::mem::take(value);
                *value = &phase * old;
            });
    }

    Ok(matrix)
}

/// Evaluates a [`SymbolicMatrix`] to a numerical `Array2<Complex64>` by
/// binding concrete values to every free symbol.
///
/// # Parallelism
///
/// When the matrix storage is contiguous (the common case), evaluation is
/// parallelised with rayon. Otherwise it falls back to a sequential scan.
///
/// # Errors
///
/// Returns [`ParameterError`] if any symbol required by the matrix elements
/// is missing from `bindings`.
pub fn evaluate_symbolic_matrix(
    matrix: &SymbolicMatrix,
    bindings: &Option<HashMap<&str, f64>>,
) -> Result<Array2<Complex64>, ParameterError> {
    let mut out = Array2::from_elem(matrix.raw_dim(), Complex64::new(0.0, 0.0));

    if let (Some(m), Some(o)) = (matrix.as_slice(), out.as_slice_mut()) {
        m.par_iter().zip(o.par_iter_mut()).try_for_each(
            |(v, o)| -> Result<(), ParameterError> {
                *o = v.evaluate(bindings)?;
                Ok(())
            },
        )?;
    } else {
        for ((row, col), value) in matrix.indexed_iter() {
            out[[row, col]] = value.evaluate(bindings)?;
        }
    }

    Ok(out)
}

#[cfg(test)]
#[path = "./symbolic_matrix_test.rs"]
mod symbolic_matrix_test;
