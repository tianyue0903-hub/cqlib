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

//! User-facing configuration for canonicalization.

/// Configuration for circuit canonicalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeConfig {
    round_limit: u8,
    recurse_control_flow: bool,
    fold_gphase: bool,
    canonicalize_instruction_form: bool,
    drop_noops: bool,
    canonicalize_barriers: bool,
}

impl Default for CanonicalizeConfig {
    fn default() -> Self {
        Self::production()
    }
}

impl CanonicalizeConfig {
    /// Returns production defaults for the first canonicalization stage.
    pub const fn production() -> Self {
        Self {
            round_limit: 8,
            recurse_control_flow: true,
            fold_gphase: true,
            canonicalize_instruction_form: true,
            drop_noops: true,
            canonicalize_barriers: true,
        }
    }

    /// Returns a new configuration using production defaults.
    pub const fn new() -> Self {
        Self::production()
    }

    /// Sets the maximum number of canonicalization rounds.
    pub const fn with_round_limit(mut self, round_limit: u8) -> Self {
        self.round_limit = round_limit;
        self
    }

    /// Controls whether control-flow bodies are recursively canonicalized.
    pub const fn recurse_control_flow(mut self, enabled: bool) -> Self {
        self.recurse_control_flow = enabled;
        self
    }

    /// Controls whether `GPhase` operations are folded into scope-local phase.
    pub const fn fold_gphase(mut self, enabled: bool) -> Self {
        self.fold_gphase = enabled;
        self
    }

    /// Controls whether `McGate` forms are collapsed into existing standard gates.
    pub const fn canonicalize_instruction_form(mut self, enabled: bool) -> Self {
        self.canonicalize_instruction_form = enabled;
        self
    }

    /// Controls strict no-op removal.
    pub const fn drop_noops(mut self, enabled: bool) -> Self {
        self.drop_noops = enabled;
        self
    }

    /// Controls barrier scope canonicalization and adjacent barrier merging.
    pub const fn canonicalize_barriers(mut self, enabled: bool) -> Self {
        self.canonicalize_barriers = enabled;
        self
    }

    /// Returns the configured round limit.
    pub const fn round_limit(&self) -> u8 {
        self.round_limit
    }

    /// Returns whether control-flow recursion is enabled.
    pub const fn recurses_control_flow(&self) -> bool {
        self.recurse_control_flow
    }

    /// Returns whether `GPhase` operations are folded into scope-local phase.
    pub const fn folds_gphase(&self) -> bool {
        self.fold_gphase
    }

    /// Returns whether instruction forms are canonicalized.
    pub const fn canonicalizes_instruction_form(&self) -> bool {
        self.canonicalize_instruction_form
    }

    /// Returns whether strict no-ops are removed.
    pub const fn drops_noops(&self) -> bool {
        self.drop_noops
    }

    /// Returns whether barrier scopes are canonicalized and merged.
    pub const fn canonicalizes_barriers(&self) -> bool {
        self.canonicalize_barriers
    }
}
