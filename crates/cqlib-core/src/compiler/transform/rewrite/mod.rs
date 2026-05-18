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

//! Knowledge-base driven local circuit rewrite.
//!
//! This module provides the compiler transform that applies peephole rewrites
//! to quantum circuits using a rule-based knowledge base. Each rewrite rule
//! describes an equivalence between a local gate pattern (the *match*) and a
//! replacement (the *rewrite*), optionally guarded by parameter conditions such
//! as `EqMod(theta, 0, 2π)`. The rewriter scans contiguous blocks of
//! rewrite-safe gate-like operations, selects a non-overlapping set of beneficial
//! rewrites, and rebuilds the circuit.
//!
//! The default [`production`](KnowledgeRewriter::production) profile is a
//! conservative logical optimizer: it only accepts rewrites that strictly reduce
//! the local cost. When [`RewriteConfig::with_target_gates`] or
//! [`RewriteConfig::with_target_instructions`] is configured, the same matcher
//! becomes target-basis aware. In that mode rules are filtered so replacement
//! instructions stay inside the requested target basis, and local cost first
//! minimizes the number of non-target operations before comparing normal logical
//! cost terms.
//!
//! Target-gate rewrite is intentionally an explicit configuration knob in this
//! module. It does not infer native gates from a device or attach itself to a
//! target workflow; callers that want target lowering pass the desired standard
//! gate set or gate-like instruction set through [`RewriteConfig`].
//!
//! # Processing pipeline
//!
//! On each fixpoint round the rewriter:
//!
//! 1. **Splits** the top-level operation list (and, when enabled, control-flow
//!    bodies) into contiguous blocks of rewrite-safe operations. Opaque or
//!    non-unitary operations (barriers, measurements, delays, control-flow gates)
//!    act as block boundaries and are emitted unchanged.
//! 2. **Compiles** knowledge-base rewrite rules into a first-instruction index and
//!    extracts `A; B -> B; A` commute rules into a read-only commutation oracle.
//!    Reverse rewrites are never generated automatically; if both directions
//!    are wanted, both directions must be present in the knowledge files.
//! 3. **Filters and matches** candidate rules. In target-basis mode, a rule's
//!    match instructions must appear in the current block and its replacement
//!    instructions must be standard gates or multi-controlled gate wrappers in
//!    the configured target basis. Replacement `GPhase` is allowed implicitly
//!    because top-level replacements are folded into circuit global phase and
//!    control-flow-body replacements are discarded.
//!    [`Commute`](crate::compiler::knowledge::library::RuleKind::Commute) rules
//!    do not emit standalone swap patches; they only let the matcher cross an
//!    intervening operation when the oracle can prove the skipped operation
//!    commutes with the matched and replacement operations.
//! 4. **Selects** a non-overlapping set of rewrite patches sorted by local cost
//!    improvement. In [`Optimize`](RewriteMode::Optimize) mode every accepted
//!    rewrite must strictly reduce the
//!    [`LocalRewriteCost`](config::LocalRewriteCost); in
//!    [`Lowering`](RewriteMode::Lowering) mode decomposition rules are also
//!    allowed, but target-basis mode still rejects rewrites that increase the
//!    number of non-target operations.
//! 5. **Rebuilds** the circuit from the selected patches, interleaving replaced
//!    and unchanged operations. Replacement `GPhase` gates at the top level are
//!    folded into the circuit's global phase rather than emitted as explicit
//!    operations. Replacement `GPhase` gates produced inside control-flow bodies
//!    are dropped because the current circuit IR has no branch-local global
//!    phase field; explicit source `GPhase` operations already present in those
//!    bodies are preserved unchanged.
//! 6. **Iterates** until the circuit stabilizes or the configured
//!    [`max_rounds`](RewriteConfig::max_rounds) limit is reached.
//!
//! # Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `config` | Stable user-facing configuration ([`RewriteConfig`]), rewrite mode, target gate set, symbolic policy, and the local cost model |
//! | `matcher` | Rule compilation, target-aware filtering, commute-oracle construction, dependency-aware sequence matching, and greedy non-overlapping patch selection |
//! | `rewriter` | [`Transformer`](crate::compiler::transform::Transformer) entry point, fixpoint loop, circuit rebuild, and GPhase folding |
//!
//! # Examples
//!
//! ## Production optimization
//!
//! The [`KnowledgeRewriter`] can be run directly as a [`Transformer`] without
//! building a full [`CompilerWorkflow`](crate::compiler::CompilerWorkflow).
//! This example creates a circuit with two adjacent Hadamard gates, which the
//! Cancel rule eliminates:
//!
//! [`Transformer`]: crate::compiler::transform::Transformer
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::compiler::CompilerContext;
//! use cqlib_core::compiler::transform::rewrite::KnowledgeRewriter;
//! use cqlib_core::compiler::transform::Transformer;
//!
//! let mut circuit = Circuit::new(1);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.h(Qubit::new(0)).unwrap();
//!
//! let mut ctx = CompilerContext::new(circuit);
//! let outcome = KnowledgeRewriter::production().run(&mut ctx).unwrap();
//!
//! assert!(outcome.changed);
//! assert!(ctx.circuit().operations().is_empty());
//! ```
//!
//! ## Lowering mode
//!
//! In [`Lowering`](RewriteMode::Lowering) mode the rewriter is allowed to apply
//! decomposition rules. Supplying a target gate set keeps decomposition directed
//! toward that basis:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Instruction, Qubit, StandardGate};
//! use cqlib_core::compiler::CompilerContext;
//! use cqlib_core::compiler::transform::rewrite::{KnowledgeRewriter, RewriteConfig};
//! use cqlib_core::compiler::transform::Transformer;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let config = RewriteConfig::lowering()
//!     .with_target_gates(vec![StandardGate::H, StandardGate::CZ])
//!     .with_max_rounds(4);
//!
//! let mut ctx = CompilerContext::new(circuit);
//! let outcome = KnowledgeRewriter::new(config).run(&mut ctx).unwrap();
//!
//! assert!(outcome.changed);
//! let operations = ctx.circuit().operations();
//! assert_eq!(operations.len(), 3);
//! assert!(operations.iter().all(|op| matches!(
//!     op.instruction,
//!     Instruction::Standard(StandardGate::H | StandardGate::CZ)
//! )));
//! ```
//!
//! ## Target-gate compression
//!
//! Target-gate mode can also pick a shorter supported representation when both
//! the source and target gates are available. For example, a CZ expression of a
//! CNOT is compressed back to `CX` when `CX` is in the target gate set:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Instruction, Qubit, StandardGate};
//! use cqlib_core::compiler::CompilerContext;
//! use cqlib_core::compiler::transform::rewrite::{KnowledgeRewriter, RewriteConfig};
//! use cqlib_core::compiler::transform::Transformer;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(1)).unwrap();
//! circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
//! circuit.h(Qubit::new(1)).unwrap();
//!
//! let config = RewriteConfig::new()
//!     .with_target_gates(vec![StandardGate::H, StandardGate::CX]);
//!
//! let mut ctx = CompilerContext::new(circuit);
//! let outcome = KnowledgeRewriter::new(config).run(&mut ctx).unwrap();
//!
//! assert!(outcome.changed);
//! let operations = ctx.circuit().operations();
//! assert_eq!(operations.len(), 1);
//! assert!(matches!(
//!     operations[0].instruction,
//!     Instruction::Standard(StandardGate::CX)
//! ));
//! ```
//!
//! ## Custom configuration
//!
//! [`RewriteConfig`] uses a builder pattern. This example disables label
//! skipping, restricts the rule set to Cancel only, and limits the fixpoint
//! to a single round:
//!
//! ```rust
//! use cqlib_core::compiler::transform::rewrite::RewriteConfig;
//! use cqlib_core::compiler::knowledge::library::RuleKind;
//!
//! let config = RewriteConfig::new()
//!     .skip_labeled_ops(false)
//!     .with_enabled_kinds(vec![RuleKind::Cancel])
//!     .with_max_rounds(1)
//!     .with_max_pattern_len(6)
//!     .with_max_window_ops(12)
//!     .recurse_control_flow(true);
//! ```

mod config;
mod matcher;
mod rewriter;

pub use config::{RewriteConfig, RewriteMode, RewriteSymbolicPolicy};
pub use rewriter::{KnowledgeRewriteStats, KnowledgeRewriter};

#[cfg(test)]
#[path = "./rewrite_test.rs"]
mod rewrite_test;

#[cfg(test)]
#[path = "./matcher_test.rs"]
mod matcher_test;
