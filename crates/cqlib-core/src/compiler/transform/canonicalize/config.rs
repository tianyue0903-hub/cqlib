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

//! Stable configuration for production canonicalization.
//!
//! The configuration surface is intentionally small:
//! - it exposes only durable policy toggles
//! - it keeps the built-in canonicalization rule set internally owned by the
//!   compiler
//! - it avoids pass-manager style tuning knobs that would leak implementation
//!   detail into the public API too early
//!
//! These flags cover the currently implemented parameter and structural
//! canonicalization stages while keeping the public shape narrow.
//!
//! # Defaults
//!
//! `CanonicalizeConfig::production()` enables all safe canonicalization rules
//! with a round limit of 8:
//!
//! | Flag | Production default |
//! |------|-------------------|
//! | `normalize_parameters` | `true` |
//! | `canonicalize_instruction_form` | `true` |
//! | `merge_adjacent_barriers` | `true` |
//! | `drop_trivial_noops` | `true` |
//! | `recurse_control_flow` | `true` |
//! | `round_limit` | `8` |

/// Configuration for the production canonicalizer.
///
/// The surface is intentionally small: only durable policy toggles are
/// exposed, while the built-in rule set remains compiler-internal. This
/// avoids leaking pass-manager details into the public API too early.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeConfig {
    /// Maximum number of fixpoint rounds.
    round_limit: u8,
    /// Whether to recurse into `IfElse` and `WhileLoop` bodies.
    recurse_control_flow: bool,
    /// Whether to simplify symbolic parameters and fold constants.
    normalize_parameters: bool,
    /// Whether to collapse multi-controlled gates into standard forms.
    canonicalize_instruction_form: bool,
    /// Whether to merge adjacent barriers when their scopes allow it.
    merge_adjacent_barriers: bool,
    /// Whether to drop unlabeled trivial no-ops such as `I` or `RZ(0)`.
    drop_trivial_noops: bool,
}

impl Default for CanonicalizeConfig {
    fn default() -> Self {
        Self::production()
    }
}

impl CanonicalizeConfig {
    /// Returns the production default canonicalization configuration.
    ///
    /// These defaults enable the currently implemented parameter and structural
    /// canonicalization rules.
    pub const fn production() -> Self {
        Self {
            round_limit: 8,
            recurse_control_flow: true,
            normalize_parameters: true,
            canonicalize_instruction_form: true,
            merge_adjacent_barriers: true,
            drop_trivial_noops: true,
        }
    }

    /// Returns a new canonicalization configuration using production defaults.
    pub const fn new() -> Self {
        Self::production()
    }

    /// Sets the maximum number of canonicalization rounds.
    ///
    /// This bounds the entire canonicalization process, not the retry count of
    /// any individual rule. The value must be greater than zero; `0` is
    /// treated as an invalid runtime configuration by the canonicalizer.
    pub const fn with_round_limit(mut self, round_limit: u8) -> Self {
        self.round_limit = round_limit;
        self
    }

    /// Controls whether canonicalization should recurse into control-flow bodies.
    pub const fn recurse_control_flow(mut self, enabled: bool) -> Self {
        self.recurse_control_flow = enabled;
        self
    }

    /// Controls whether symbolic parameters and global phase are normalized.
    pub const fn normalize_parameters(mut self, enabled: bool) -> Self {
        self.normalize_parameters = enabled;
        self
    }

    /// Controls whether instruction forms are canonicalized.
    pub const fn canonicalize_instruction_form(mut self, enabled: bool) -> Self {
        self.canonicalize_instruction_form = enabled;
        self
    }

    /// Controls whether adjacent barriers are merged.
    pub const fn merge_adjacent_barriers(mut self, enabled: bool) -> Self {
        self.merge_adjacent_barriers = enabled;
        self
    }

    /// Controls whether unlabeled trivial no-op operations are dropped.
    pub const fn drop_trivial_noops(mut self, enabled: bool) -> Self {
        self.drop_trivial_noops = enabled;
        self
    }

    /// Returns the configured round limit for the whole canonicalization run.
    pub const fn round_limit(&self) -> u8 {
        self.round_limit
    }

    /// Returns whether control-flow recursion is enabled.
    pub const fn recurses_control_flow(&self) -> bool {
        self.recurse_control_flow
    }

    /// Returns whether parameter normalization is enabled.
    pub const fn normalizes_parameters(&self) -> bool {
        self.normalize_parameters
    }

    /// Returns whether instruction-form canonicalization is enabled.
    pub const fn canonicalizes_instruction_form(&self) -> bool {
        self.canonicalize_instruction_form
    }

    /// Returns whether adjacent barrier merging is enabled.
    pub const fn merges_adjacent_barriers(&self) -> bool {
        self.merge_adjacent_barriers
    }

    /// Returns whether trivial no-op dropping is enabled.
    pub const fn drops_trivial_noops(&self) -> bool {
        self.drop_trivial_noops
    }
}
