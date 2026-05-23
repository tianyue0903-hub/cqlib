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

//! Configuration and local cost model for knowledge-based rewrite.
//!
//! This module owns the public policy surface for the rewrite transformer.  It
//! separates semantic rule eligibility (which rule kinds may run) from local
//! cost acceptance (whether an already-matched rewrite should be applied).  The
//! cost model is intentionally small and lexicographic so optimization mode has
//! a stable termination argument: accepted logical rewrites must strictly reduce
//! the local cost tuple.

use crate::circuit::{Instruction, MCGate, StandardGate};
use crate::compiler::knowledge::library::RuleKind;

/// High-level rule application mode.
///
/// The mode controls whether rewrite is used as a conservative optimizer or as
/// an explicit lowering pass.  It does not change rule semantics: all accepted
/// rewrites still come from the validated knowledge base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteMode {
    /// Conservative optimization mode. Every accepted rewrite must strictly
    /// improve the local logical cost.
    Optimize,
    /// Explicit lowering mode. Decomposition and hardware-native rules may be
    /// applied even when they do not improve the logical cost.
    Lowering,
}

/// Stable configuration for the knowledge-based rewrite transformer.
///
/// `RewriteConfig` is deliberately builder-style and immutable once installed
/// on a [`KnowledgeRewriter`](super::KnowledgeRewriter).  Runtime code reads it
/// frequently while matching, so the fields are private and exposed through
/// small accessors to keep policy checks centralized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewriteConfig {
    /// Maximum number of fixpoint rounds before the transformer stops.
    max_rounds: u8,
    /// Maximum number of operations scanned while looking for the next pattern
    /// item after the current cursor.
    max_window_ops: usize,
    /// Maximum number of rule operations allowed in a match pattern.
    max_pattern_len: usize,
    /// Whether nested control-flow bodies are recursively rewritten.
    recurse_control_flow: bool,
    /// Whether labeled operations act as protected operations.
    skip_labeled_ops: bool,
    /// Rule kinds considered during candidate generation.
    enabled_kinds: Vec<RuleKind>,
    /// High-level optimization or lowering mode.
    mode: RewriteMode,
    /// Optional target gate-like instruction basis used to steer rewrite selection.
    target_instructions: Option<Vec<TargetInstruction>>,
}

/// Gate-like instruction accepted by rewrite target-basis configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TargetInstruction {
    /// A standard gate target.
    Standard(StandardGate),
    /// A multi-controlled gate target.
    McGate(MCGate),
    /// A non-gate-like instruction supplied through the public API.
    Unsupported(String),
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self::production()
    }
}

impl RewriteConfig {
    /// Production defaults for conservative logical rewrite.
    ///
    /// The production profile intentionally excludes decomposition rules so it
    /// cannot expand a circuit unless that expansion also reduces the local
    /// logical cost through a non-decomposition rule.
    pub fn production() -> Self {
        Self {
            max_rounds: 8,
            max_window_ops: 16,
            max_pattern_len: 8,
            recurse_control_flow: true,
            skip_labeled_ops: true,
            enabled_kinds: vec![
                RuleKind::Simplify,
                RuleKind::Cancel,
                RuleKind::Merge,
                RuleKind::Canonicalize,
            ],
            mode: RewriteMode::Optimize,
            target_instructions: None,
        }
    }

    /// Defaults for explicit knowledge-based lowering.
    ///
    /// Lowering enables decomposition and hardware-native rule kinds and relaxes
    /// the cost check for those kinds when no target instruction basis is configured.
    pub fn lowering() -> Self {
        Self {
            enabled_kinds: vec![
                RuleKind::Simplify,
                RuleKind::Cancel,
                RuleKind::Merge,
                RuleKind::Canonicalize,
                RuleKind::Decompose,
                RuleKind::HardwareNative,
            ],
            mode: RewriteMode::Lowering,
            ..Self::production()
        }
    }

    /// Returns a new production configuration.
    pub fn new() -> Self {
        Self::production()
    }

    /// Sets the maximum number of fixpoint rounds.
    ///
    /// A value of zero is rejected by the transformer entry point because no
    /// stability proof can be attempted without at least one round.
    pub fn with_max_rounds(mut self, max_rounds: u8) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Sets the per-pattern lookahead window.
    ///
    /// The window bounds dependency-aware non-adjacent matching.  Larger values
    /// may find more opportunities but increase per-anchor scan cost.
    pub fn with_max_window_ops(mut self, max_window_ops: usize) -> Self {
        self.max_window_ops = max_window_ops;
        self
    }

    /// Sets the maximum rule match length considered by the optimizer.
    ///
    /// This is a coarse guard against expensive or accidentally broad knowledge
    /// rules.  Rules longer than this value are skipped before matching.
    pub fn with_max_pattern_len(mut self, max_pattern_len: usize) -> Self {
        self.max_pattern_len = max_pattern_len;
        self
    }

    /// Controls whether rewrite recurses into control-flow bodies.
    ///
    /// When disabled, control-flow operations are copied as opaque operations and
    /// their bodies are not inspected.
    pub fn recurse_control_flow(mut self, enabled: bool) -> Self {
        self.recurse_control_flow = enabled;
        self
    }

    /// Controls whether labeled operations are kept intact.
    ///
    /// When enabled, labeled operations cannot be used as anchors or skipped
    /// across during dependency-aware matching.
    pub fn skip_labeled_ops(mut self, enabled: bool) -> Self {
        self.skip_labeled_ops = enabled;
        self
    }

    /// Replaces the enabled rule kinds.
    ///
    /// This does not by itself allow decomposition or hardware-native rules in
    /// optimization mode; [`allows_kind`](Self::allows_kind) applies that mode
    /// guard after checking this list.
    pub fn with_enabled_kinds(mut self, kinds: Vec<RuleKind>) -> Self {
        self.enabled_kinds = kinds;
        self
    }

    /// Sets the high-level rule application mode.
    pub fn with_mode(mut self, mode: RewriteMode) -> Self {
        self.mode = mode;
        self
    }

    /// Restricts rewrite selection toward a target gate-like instruction basis.
    ///
    /// Only [`Instruction::Standard`] and [`Instruction::McGate`] are valid
    /// target instructions. Other instruction variants are stored so builder
    /// usage remains infallible, then rejected when the transformer runs.
    pub fn with_target_instructions(mut self, target_instructions: Vec<Instruction>) -> Self {
        let mut deduped = Vec::with_capacity(target_instructions.len());

        for instruction in target_instructions {
            let instruction = match instruction {
                Instruction::Standard(gate) => TargetInstruction::Standard(gate),
                Instruction::McGate(gate) => TargetInstruction::McGate(*gate),
                other => TargetInstruction::Unsupported(format!("{other:?}")),
            };
            if deduped.contains(&instruction) {
                continue;
            }

            deduped.push(instruction);
        }

        self.target_instructions = Some(deduped);
        self
    }

    /// Returns the maximum number of fixpoint rounds.
    pub const fn max_rounds(&self) -> u8 {
        self.max_rounds
    }

    /// Returns the per-pattern lookahead window.
    pub const fn max_window_ops(&self) -> usize {
        self.max_window_ops
    }

    /// Returns the maximum rule match length considered by the optimizer.
    pub const fn max_pattern_len(&self) -> usize {
        self.max_pattern_len
    }

    /// Returns whether control-flow bodies are rewritten.
    pub const fn recurses_control_flow(&self) -> bool {
        self.recurse_control_flow
    }

    /// Returns whether labeled operations are skipped.
    pub const fn skips_labeled_ops(&self) -> bool {
        self.skip_labeled_ops
    }

    /// Returns the high-level rule application mode.
    pub const fn mode(&self) -> RewriteMode {
        self.mode
    }

    /// Returns the configured target gate-like instruction basis, if any.
    pub(crate) fn target_instructions(&self) -> Option<&[TargetInstruction]> {
        self.target_instructions.as_deref()
    }

    /// Returns whether the user configuration names a rule kind.
    pub(crate) fn enables_kind(&self, kind: RuleKind) -> bool {
        self.enabled_kinds.contains(&kind)
    }

    /// Returns whether a rule kind may participate in matching.
    ///
    /// Decomposition and hardware-native rules are intentionally locked behind
    /// lowering mode because they may expand logical cost and are not suitable as
    /// default optimization rewrites.
    pub(crate) fn allows_kind(&self, kind: RuleKind) -> bool {
        if !self.enables_kind(kind) {
            return false;
        }

        !matches!(kind, RuleKind::Decompose | RuleKind::HardwareNative)
            || self.mode == RewriteMode::Lowering
    }

    /// Returns whether a concrete local rewrite passes the configured cost policy.
    ///
    /// The cost tuple is compared lexicographically.  Target-basis mode first
    /// rejects any rewrite that increases unsupported operations, then requires a
    /// strict total cost improvement.  Without target instructions, lowering mode
    /// also permits decomposition and hardware-native rewrites that are
    /// cost-neutral or cost-increasing in logical terms.
    pub(crate) fn allows_rewrite(
        &self,
        kind: RuleKind,
        before: LocalRewriteCost,
        after: LocalRewriteCost,
    ) -> bool {
        if after.unsupported_ops > before.unsupported_ops {
            return false;
        }
        if after < before {
            return true;
        }
        if self.target_instructions.is_some() {
            return false;
        }

        self.mode == RewriteMode::Lowering
            && matches!(kind, RuleKind::Decompose | RuleKind::HardwareNative)
    }
}

/// Local rewrite cost used to enforce termination in logical rewrite mode.
///
/// The field order is the comparison order because this type derives `Ord`.
/// Earlier fields dominate later fields:
///
/// 1. target-basis unsupported operations,
/// 2. distance from the physical target basis through lowerable intermediates,
/// 3. two-qubit operations,
/// 4. multi-qubit operations,
/// 5. local depth estimate,
/// 6. total operations,
/// 7. parameterized operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LocalRewriteCost {
    /// Operations outside the requested physical target instruction basis.
    pub(crate) unsupported_ops: usize,
    /// Sum of non-physical intermediate distances to the requested target set.
    pub(crate) lowering_distance: usize,
    /// Two-qubit operations counted by local logical cost.
    pub(crate) two_qubit_ops: usize,
    /// Operations acting on more than two qubits.
    pub(crate) multi_qubit_ops: usize,
    /// Greedy depth estimate over the rewritten local region.
    pub(crate) depth_estimate: usize,
    /// Total counted gate-like operations.
    pub(crate) total_ops: usize,
    /// Counted operations carrying at least one parameter.
    pub(crate) parameterized_ops: usize,
}

/// Policy for counting `GPhase` operations in local rewrite cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPhaseCost {
    /// Count an explicit GPhase operation in the source circuit as removable
    /// representation cost.
    ExplicitOperation,
    /// Treat a replacement GPhase as implicit phase metadata. The rewriter
    /// folds it into circuit global phase at the top level and discards it
    /// inside control-flow bodies.
    ImplicitReplacement,
}

impl LocalRewriteCost {
    /// Adds one gate-like operation to the local cost tuple.
    ///
    /// Returns `true` when the operation contributes to the depth estimate.  A
    /// replacement `GPhase` returns `false` because replacement phase is handled
    /// by the sequence emission context instead of emitted as a normal operation.
    ///
    /// `target_supported` is precomputed by the matcher because target-basis
    /// filtering can include both standard and multi-controlled gates.
    pub(crate) fn add_gate_like(
        &mut self,
        standard_gate: Option<StandardGate>,
        target_supported: bool,
        qubit_count: usize,
        param_count: usize,
        gphase_cost: GPhaseCost,
    ) -> bool {
        if standard_gate == Some(StandardGate::GPhase)
            && gphase_cost == GPhaseCost::ImplicitReplacement
        {
            return false;
        }

        self.total_ops += 1;
        if !target_supported {
            self.unsupported_ops += 1;
        }
        match qubit_count {
            0 | 1 => {}
            2 => self.two_qubit_ops += 1,
            _ => self.multi_qubit_ops += 1,
        }
        if param_count > 0 {
            self.parameterized_ops += 1;
        }
        true
    }
}
