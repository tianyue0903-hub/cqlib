// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! # Gate Commutation
//!
//! This module proves when two concrete instruction applications can be
//! exchanged by compiler passes without changing circuit semantics.  A
//! successful query returns a [`Commutation`] proof, while `None` means only
//! that the available oracles could not prove commutation.
//!
//! The checker combines several conservative proof sources:
//!
//! - fast structural facts such as identity, global phase, disjoint support,
//!   and identical applications;
//! - algebraic rules for diagonal gates, Pauli-axis strings, controlled-axis
//!   gates, and selected symmetric two-qubit families;
//! - explicit `A; B -> B; A` rules from the compiler knowledge library;
//! - an optional small local-matrix fallback for concrete parameters.
//!
//! # Examples
//!
//! ```rust
//! use cqlib_core::circuit::{Instruction, Parameter, Qubit, StandardGate};
//! use cqlib_core::compile::commutation::{check_commutation, Commutation};
//!
//! let result = check_commutation(
//!     &Instruction::Standard(StandardGate::RZ),
//!     &[Qubit::new(0)],
//!     &[Parameter::symbol("a")],
//!     &Instruction::Standard(StandardGate::RZ),
//!     &[Qubit::new(0)],
//!     &[Parameter::symbol("b")],
//! );
//!
//! assert_eq!(result, Some(Commutation::Exact));
//! ```

mod algebra;
pub mod checker;
mod matrix;
mod rules;

pub use checker::{
    Commutation, CommutationChecker, CommutationConfig, CommutationResult, check_commutation,
};

pub use algebra::algebraic_commutation;

#[cfg(test)]
#[path = "./commutation_test.rs"]
mod commutation_test;
