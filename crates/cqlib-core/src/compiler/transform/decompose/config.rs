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

//! Configuration for compiler-level decomposition.

use crate::circuit::StandardGate;

/// Stable configuration for target-basis decomposition.
///
/// When `target_gates` is unset, the decomposer falls back to the active
/// compiler-context device's native standard gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposeConfig {
    /// Explicit standard-gate target basis.
    ///
    /// `None` means the transformer should derive the basis from the active
    /// compiler-context device.
    target_gates: Option<Vec<StandardGate>>,
    /// Maximum number of fixpoint rounds delegated to the lowering rewriter.
    max_rounds: u8,
    /// Whether nested control-flow bodies should be decomposed recursively.
    recurse_control_flow: bool,
    /// Whether labeled operations should be protected from local rewrites.
    skip_labeled_ops: bool,
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            target_gates: None,
            max_rounds: 8,
            recurse_control_flow: true,
            skip_labeled_ops: true,
        }
    }
}

impl DecomposeConfig {
    /// Creates decomposition config with production defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses an explicit target standard-gate basis.
    ///
    /// When this is set, decomposition does not require a target device in the
    /// compiler context.
    pub fn with_target_gates(mut self, target_gates: Vec<StandardGate>) -> Self {
        self.target_gates = Some(dedup_gates(target_gates));
        self
    }

    /// Sets the maximum number of lowering fixpoint rounds.
    ///
    /// A value of zero is rejected by the transformer entry point.
    pub fn with_max_rounds(mut self, max_rounds: u8) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Controls whether control-flow bodies are recursively decomposed.
    pub fn recurse_control_flow(mut self, enabled: bool) -> Self {
        self.recurse_control_flow = enabled;
        self
    }

    /// Controls whether labeled operations are protected from local rewrites.
    pub fn skip_labeled_ops(mut self, enabled: bool) -> Self {
        self.skip_labeled_ops = enabled;
        self
    }

    /// Returns the explicit target standard-gate basis, if configured.
    pub fn target_gates(&self) -> Option<&[StandardGate]> {
        self.target_gates.as_deref()
    }

    /// Returns the maximum number of lowering fixpoint rounds.
    pub const fn max_rounds(&self) -> u8 {
        self.max_rounds
    }

    /// Returns whether control-flow bodies are recursively decomposed.
    pub const fn recurses_control_flow(&self) -> bool {
        self.recurse_control_flow
    }

    /// Returns whether labeled operations are protected from local rewrites.
    pub const fn skips_labeled_ops(&self) -> bool {
        self.skip_labeled_ops
    }
}

/// Removes duplicate gates while preserving the caller-supplied basis order.
fn dedup_gates(gates: Vec<StandardGate>) -> Vec<StandardGate> {
    let mut deduped = Vec::with_capacity(gates.len());
    for gate in gates {
        if !deduped.contains(&gate) {
            deduped.push(gate);
        }
    }
    deduped
}
