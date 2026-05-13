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

//! Production canonicalization support.
//!
//! This module provides the compiler transform that rewrites a circuit into the
//! stable representation expected by later compiler passes. Canonicalization is
//! intentionally conservative: it changes representation details that are known
//! to be semantics-preserving, and it reports whether the compiler context was
//! actually changed.
//!
//! The transform is split into two phases that run in this order each round:
//!
//! 1. **Parameter phase** (`parameter_phase`) rebuilds the circuit parameter
//!    table, simplifies symbolic expressions, folds parameters that evaluate to
//!    fixed values, remaps operation parameter references, and normalizes the
//!    circuit global phase.
//! 2. **Structural phase** (`linear`, `ops`, and `equivalence`) scans top-level
//!    operation sequences and, when enabled, control-flow bodies. It collapses
//!    supported multi-controlled gates into standard forms, sorts and deduplicates
//!    barrier qubits, merges adjacent barriers with equal or superset scopes,
//!    drops trivial no-ops, and rebuilds the circuit through the public
//!    [`Circuit`](crate::circuit::Circuit) construction APIs.
//!
//! # Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `config` | Stable user-facing configuration (`CanonicalizeConfig`) |
//! | `canonicalizer` | [`Transformer`](crate::compiler::transform::Transformer) entry point, descriptor contract, and fixpoint loop |
//! | `parameter_phase` | Symbolic parameter remapping, parameter-table rebuilds, and global-phase normalization |
//! | `linear` | Linear operation scanning and rebuilding for top-level operations and control-flow bodies |
//! | `ops` | Per-operation helpers for parameter resolution, no-op detection, barrier qubit canonicalization, and barrier label merging |
//! | `standard_gate_normalize` | Numeric standard-gate canonical forms, special-angle folding, and explicit global-phase compensation |
//! | `equivalence` | Conservative representation-equivalence checks used to decide whether a round changed anything |
//!
//! # Example
//!
//! The canonicalizer can be run directly as a [`Transformer`] without building a
//! full [`CompilerWorkflow`](crate::compiler::CompilerWorkflow). This example
//! creates a circuit with two trivial no-ops and two adjacent barriers with the
//! same qubit scope. Canonicalization removes the no-ops, sorts the barrier
//! qubits, and merges the barriers into one operation.
//!
//! [`Transformer`]: crate::compiler::transform::Transformer
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Directive, Instruction, Qubit};
//! use cqlib_core::compiler::{CanonicalizeConfig, Canonicalizer, CompilerContext};
//! use cqlib_core::compiler::transform::Transformer;
//!
//!
//! let mut circuit = Circuit::new(2);
//! circuit.i(Qubit::new(0)).unwrap();
//! circuit.rz(Qubit::new(0), 0.0).unwrap();
//! circuit.barrier(vec![Qubit::new(1), Qubit::new(0)]).unwrap();
//! circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
//!
//! let mut ctx = CompilerContext::new(circuit);
//! let canonicalizer = Canonicalizer::new(
//!     CanonicalizeConfig::new().normalize_parameters(false),
//! );
//!
//! let outcome = canonicalizer.run(&mut ctx).unwrap();
//! assert!(outcome.changed);
//!
//! let operations = ctx.circuit().operations();
//! assert_eq!(operations.len(), 1);
//! assert!(matches!(
//!     operations[0].instruction,
//!     Instruction::Directive(Directive::Barrier)
//! ));
//! assert_eq!(operations[0].qubits.as_slice(), &[Qubit::new(0), Qubit::new(1)]);
//!
//! ```
//!
//! Individual rules can be enabled or disabled through [`CanonicalizeConfig`]:
//!
//! ```rust
//! use cqlib_core::compiler::CanonicalizeConfig;
//!
//! let config = CanonicalizeConfig::new()
//!     .normalize_parameters(true)
//!     .canonicalize_instruction_form(true)
//!     .merge_adjacent_barriers(true)
//!     .drop_trivial_noops(true)
//!     .recurse_control_flow(true)
//!     .with_round_limit(8);
//! ```
//!
//! # Integrating into the compiler workflow
//!
//! [`Canonicalizer`] implements [`Transformer`], so it can be chained into a
//! [`CompilerWorkflow`](crate::compiler::CompilerWorkflow) just like any other
//! pass.
//!
//! # Configuration flags
//!
//! - `normalize_parameters` – simplifies symbolic expressions and folds constants
//!   in the circuit parameter table and global phase.
//! - `canonicalize_instruction_form` – collapses `McGate` forms into their
//!   `StandardGate` equivalents (e.g. 1-controlled `X` → `CX`).
//! - `merge_adjacent_barriers` – merges consecutive barrier directives when their
//!   qubit scopes are equal or one scope is a superset of the other.
//! - `drop_trivial_noops` – removes unlabeled operations that are semantically
//!   identity (for example `I`, `RZ(0)`, `Delay(0)`, and empty barriers). Labeled
//!   no-ops are retained because labels are user-visible operation metadata.
//! - `recurse_control_flow` – applies all enabled structural rules inside
//!   `IfElse` and `WhileLoop` bodies.
//! - `round_limit` – maximum number of fixpoint iterations (default: 8).
//!
//! # Fixpoint mechanism
//!
//! A single canonicalization round may enable further simplifications (for
//! example, parameter normalization can turn a symbolic angle into `0.0`, which
//! in turn makes an `RZ` gate a trivial no-op that structural cleanup can drop
//! in the next round). The `Canonicalizer` therefore runs rounds in a loop until
//! either:
//!
//! - both the parameter phase and the structure report *no change*, indicating
//!   the circuit has reached a stable fixed point, or
//! - the number of executed rounds hits `round_limit`. In the latter case a
//!   warning diagnostic is emitted but the partially canonicalized circuit is
//!   still returned.
//!
//! This module intentionally keeps `mod.rs` free of implementation logic.

mod canonicalizer;
mod config;
mod equivalence;
mod linear;
mod ops;
mod parameter_phase;
mod standard_gate_normalize;

pub use canonicalizer::{CanonicalRuleId, Canonicalizer};
pub use config::CanonicalizeConfig;

#[cfg(test)]
mod canonicalize_test;
