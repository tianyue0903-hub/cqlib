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

//! Compiler knowledge-base rules.
//!
//! This module contains the data model and runtime helpers used by the active
//! compiler rule system. A knowledge rule describes an ordered match pattern,
//! optional symbolic-parameter conditions, and a rewrite target.
//!
//! The main layers are:
//! - [`rule`]: runtime rule structures and structural validation.
//! - [`rule_dsl`]: parser, loader, lowering, and dumper for `.rule` files.
//! - [`library`]: validated rule storage, stable rule ids, metadata, and gate
//!   indexes for candidate selection.
//! - [`matcher`]: adjacent-operation structural matching, binding, condition
//!   checks, and target instantiation.
//! - [`rule_equivalence`]: symbolic and sampling-based equivalence validation.
//!
//! Transform passes should use [`RuleLibrary`] to select candidate rules, then
//! use matcher helpers to bind a specific operation window. Search policy,
//! commutation-aware scans, cost decisions, and circuit patching live outside
//! this module.

pub mod library;
pub mod matcher;
pub mod rule;
pub mod rule_dsl;
pub mod rule_equivalence;

pub use library::{RuleId, RuleKind, RuleLibrary, RuleLibraryError, RuleMetadata};
pub use matcher::{
    ConcreteOperationView, KnowledgeInstructionKey, MatchBindings, MatchError, MatchedReplacement,
    conditions_hold, instantiate_target, match_rule_item, rule_matches_operations,
};
