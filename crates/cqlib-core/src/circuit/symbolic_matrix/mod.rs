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
//! This module provides a symbolic counterpart of [`super::circuit_to_matrix`].
//! Instead of immediately evaluating gate parameters to concrete `f64` values,
//! it keeps unresolved parameters as [`Parameter`] expression trees inside a
//! dense symbolic matrix.  The resulting matrix can later be evaluated under
//! different parameter bindings without rebuilding the circuit-level unitary.
//!
//! # Purpose
//!
//! The symbolic matrix layer is primarily intended for **small quantum circuits**
//! and compiler-internal analysis tasks, especially:
//!
//! - preserving symbolic parameters in parametric gates such as `RX(theta)`,
//!   `RZ(phi)`, `U(theta, phi, lambda)`, and circuit-defined custom gates;
//! - comparing small rewrite patterns and replacement circuits;
//! - validating decomposition rules and peephole optimizations;
//! - debugging matrix conventions, qubit ordering, and global phase behavior;
//! - building reusable symbolic matrices for [`CircuitGate`] and circuit-backed
//!   [`UnitaryGate`](crate::circuit::gate::UnitaryGate) definitions.
//!
//! This module is **not** a large-scale symbolic simulator.  A full dense
//! unitary matrix for `n` qubits contains `4^n` symbolic entries, and each entry
//! may itself contain a growing symbolic expression tree.  In practice, this
//! backend should be used for small subcircuits, rule validation, and custom
//! gate definitions rather than full deep circuits.
//!
//! # Core types
//!
//! - [`SymbolicComplex`] stores one complex symbolic value as two independent
//!   [`Parameter`] expressions: one for the real part and one for the imaginary
//!   part.
//! - [`SymbolicMatrix`] is a dense `Array2<SymbolicComplex>` with the same
//!   storage layout and matrix convention as the numerical circuit-matrix API.
//!
//! # Matrix construction
//!
//! [`circuit_to_symbolic_matrix`] walks through the circuit operations in order
//! and applies each instruction to an initially symbolic identity matrix.
//!
//! The implementation mirrors the numerical matrix path:
//!
//! - fixed standard gates may use the numeric fast path and are lifted into
//!   symbolic values only at the matrix-element boundary;
//! - parametric standard gates are constructed directly as symbolic matrices;
//! - [`CircuitGate`](crate::circuit::gate::CircuitGate) and circuit-backed
//!   [`UnitaryGate`](crate::circuit::gate::UnitaryGate) definitions preserve
//!   symbolic parameters through simultaneous substitution;
//! - non-unitary operations such as measurement and reset return
//!   [`CircuitError::NoMatrixRepresentation`];
//! - control-flow operations return [`CircuitError::InvalidOperation`] because
//!   they do not have a single unconditional unitary matrix.
//!
//! # Qubit ordering
//!
//! Internally, matrix application uses the library's little-endian bit layout:
//! qubit position `0` is the least-significant bit.  The public
//! [`circuit_to_symbolic_matrix`] API accepts an optional `qubits_order` argument
//! to control the visible matrix order:
//!
//! - `None` sorts circuit qubits by qubit index;
//! - `Some(order)` uses the supplied qubit order and validates that it contains
//!   exactly the same qubit set as the circuit.
//!
//! Gate-local matrices, such as standard gate matrices and numeric custom
//! unitary matrices, follow the usual gate-local ordering convention and are
//! converted before application where necessary.
//!
//! # Deferred evaluation
//!
//! Use [`evaluate_symbolic_matrix`] to bind concrete values to all free symbols
//! in a symbolic matrix.
//!
//! ```
//! use cqlib_core::circuit::{Circuit, Parameter, Qubit};
//! use cqlib_core::circuit::symbolic_matrix::{
//!     circuit_to_symbolic_matrix,
//!     evaluate_symbolic_matrix,
//! };
//! use std::collections::HashMap;
//!
//! let theta = Parameter::symbol("theta");
//! let mut circuit = Circuit::new(1);
//! circuit.rx(Qubit::new(0), theta).unwrap();
//!
//! let symbolic = circuit_to_symbolic_matrix(&circuit, None).unwrap();
//!
//! let mut bindings = HashMap::new();
//! bindings.insert("theta", std::f64::consts::FRAC_PI_2);
//!
//! let numeric = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
//! assert_eq!(numeric.shape(), &[2, 2]);
//! ```
//!
//! # Circuit-gate parameter substitution
//!
//! Circuit-defined gates are expanded by first obtaining the symbolic matrix of
//! the inner frozen circuit and then simultaneously substituting the outer call
//! arguments for the inner symbols.  Simultaneous substitution is important for
//! cases such as `a -> b` and `b -> a`, where sequential replacement would be
//! order-dependent.
//!
//! ```
//! use cqlib_core::circuit::{Circuit, Parameter, ParameterValue, Qubit};
//! use cqlib_core::circuit::symbolic_matrix::{
//!     circuit_to_symbolic_matrix,
//!     evaluate_symbolic_matrix,
//! };
//! use std::collections::HashMap;
//!
//! let mut inner = Circuit::new(1);
//! inner.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
//! let gate = inner.to_gate("InnerRx").unwrap();
//!
//! let mut outer = Circuit::new(1);
//! outer.append(
//!     gate,
//!     [Qubit::new(0)],
//!     [ParameterValue::from(Parameter::symbol("x") * 2.0)],
//!     None,
//! ).unwrap();
//!
//! let symbolic = circuit_to_symbolic_matrix(&outer, None).unwrap();
//!
//! let mut bindings = HashMap::new();
//! bindings.insert("x", 0.25);
//! let numeric = evaluate_symbolic_matrix(&symbolic, &Some(bindings)).unwrap();
//!
//! assert_eq!(numeric.shape(), &[2, 2]);
//! ```
//!
//! # Global-phase equivalence
//!
//! Quantum rewrite rules often need equivalence up to a global phase rather
//! than exact matrix equality.  [`symbolic_matrices_equivalent`]
//! and [`circuits_equivalent`] provide a conservative symbolic
//! check for this purpose.
//!
//! The checker first verifies that both matrices have the same zero structure.
//! It then chooses a nonzero pivot entry and checks that all corresponding
//! entries have the same symbolic ratio by cross-multiplication.  This avoids
//! symbolic division and works well for many small rewrite-rule matrices.
//!
//! ```
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::circuit::symbolic_matrix::circuits_equivalent;
//!
//! let mut lhs = Circuit::new(1);
//! lhs.x(Qubit::new(0)).unwrap();
//! lhs.set_global_phase(std::f64::consts::PI.into());
//!
//! let mut rhs = Circuit::new(1);
//! rhs.x(Qubit::new(0)).unwrap();
//!
//! // X and -X represent the same quantum operation up to global phase.
//! assert!(circuits_equivalent(&lhs, &rhs, None).unwrap());
//! ```
//!
//! This check is intentionally conservative.  It depends on the simplification
//! power of [`Parameter::simplify`], so failure to prove equivalence does not
//! necessarily mean the two circuits are mathematically inequivalent.
//!
//! # Gate-application fast paths
//!
//! Symbolic expression growth is the main performance risk in this module.
//! To reduce unnecessary expression construction, gate matrices are classified
//! before application:
//!
//! - **Diagonal gates** only scale affected rows.  This avoids building
//!   expressions such as `0 * x + phase * y`.
//! - **Permutation gates** move or scale rows according to a monomial matrix.
//!   This is efficient for gates such as `X`, `CX`, `SWAP`, `CCX`, and other
//!   permutation-like unitaries.
//! - **Dense gates** fall back to the general matrix-vector update path.
//!
//! The same classification exists for both symbolic gate matrices and numeric
//! gate matrices.  Constant-parameter gates therefore benefit from the numeric
//! fast path, while symbolic gates still avoid unnecessary zero multiplications
//! whenever their structure is exactly diagonal or permutation-like.
//!
//! # Parallelism and safety
//!
//! Matrix application and matrix evaluation use **rayon** once the matrix element
//! count exceeds [`PARALLEL_THRESHOLD_OPS`].  Below that threshold, work stays on
//! the current thread to avoid scheduling overhead.
//!
//! Some hot paths use raw pointers internally to split mutable access across
//! rows.  The safety invariant is that each parallel worker receives a disjoint
//! set of row indices.  No two workers may write to the same matrix row.
//!
//! # Complexity and performance notes
//!
//! A dense unitary matrix for `n` qubits has dimension `2^n × 2^n`, so both
//! memory use and matrix-element work scale as `O(4^n)`.  Symbolic entries are
//! significantly more expensive than `Complex64` entries because each arithmetic
//! operation can create new [`Parameter`] expression trees.
//!
//! Recommended usage:
//!
//! - prefer this module for small circuits and local rewrite patterns;
//! - reuse cached symbolic matrices for frozen subcircuits and circuit gates;
//! - prefer circuit-backed custom gates when symbolic parameters must be
//!   preserved;
//! - avoid constructing full symbolic matrices for large application-level
//!   circuits.
//!
//! # Limitations
//!
//! - Numeric parameterized [`UnitaryGate`](crate::circuit::gate::UnitaryGate)
//!   factories take `&[f64]` and therefore cannot preserve unbound symbolic
//!   parameters.  Use a circuit-backed custom gate when symbolic preservation is
//!   required.
//! - Equivalence checking is conservative and is not a complete theorem prover.
//!   It does not replace SMT-based or proof-assistant-based verification for
//!   all symbolic identities.
//! - Gate classification uses exact structural zero/one checks.  Expressions
//!   that are mathematically zero but not simplified to exact zero may fall back
//!   to the dense path.
//! - The representation is dense; sparse symbolic matrices and expression-DAG
//!   common-subexpression sharing are future optimization directions.
//!
//! # Parallelism
//!
//! Gate application is parallelised with **rayon** when the matrix element
//! count exceeds [`PARALLEL_THRESHOLD_OPS`] (2²⁰). Small circuits run on a
//! single thread to avoid scheduling overhead.

/// Minimum number of matrix elements that triggers parallel gate application
/// via rayon. Below this threshold the work is done on the calling thread to
/// avoid the overhead of thread-pool scheduling.
///
/// Kept in sync with the numerical path in [`super::circuit_to_matrix`].
pub(crate) const PARALLEL_THRESHOLD_OPS: usize = 1 << 20;

pub mod equivalence;
pub mod gate;
pub mod matrix;

#[cfg(test)]
pub mod test_utils;

pub use matrix::{
    SymbolicComplex, SymbolicMatrix, apply_numeric_diagonal_gate, apply_numeric_permutation_gate,
    apply_symbolic_diagonal_gate, apply_symbolic_permutation_gate, evaluate_symbolic_matrix,
    substitute_symbolic_matrix, symbolic_eye,
};

pub use gate::{
    apply_gate_to_matrix, apply_gate_to_matrix_num, apply_general_gate, apply_general_gate_num,
    apply_single_qubit_gate, apply_single_qubit_gate_num, apply_standard_gate_to_matrix,
    apply_two_qubit_gate, apply_two_qubit_gate_num, circuit_to_symbolic_matrix,
    standard_gate_symbolic_matrix,
};

pub use equivalence::{circuits_equivalent, symbolic_matrices_equivalent};
