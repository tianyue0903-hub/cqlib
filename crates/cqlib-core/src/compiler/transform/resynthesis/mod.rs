//! Block and subcircuit resynthesis transformers.
//!
//! Resynthesis is reserved for transforms that extract a bounded region of a
//! circuit and rebuild an equivalent implementation under an explicit objective,
//! such as reducing two-qubit count or targeting a native basis. It is more
//! expensive and more general than deterministic local rewrite rules, and should
//! therefore remain a narrow, opt-in stage in a workflow.

use crate::compiler::transform::{TransformDescriptor, Transformer};

/// The circuit region shape a resynthesis pass is allowed to replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResynthesisScope {
    /// Contiguous runs of single-qubit gates acting on one logical wire.
    SingleQubitRuns,
    /// Small two-qubit interaction regions, typically after routing or lowering.
    TwoQubitBlocks,
    /// An implementation-defined region class with a stable external label.
    Custom(&'static str),
}

/// Resource budget that bounds a resynthesis search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResynthesisBudget {
    pub max_qubits: usize,
    pub max_operations: usize,
}

impl ResynthesisBudget {
    /// Conservative budget for local, production-safe resynthesis.
    pub const fn local(max_qubits: usize, max_operations: usize) -> Self {
        Self {
            max_qubits,
            max_operations,
        }
    }
}

/// Primary optimization goal for a resynthesis pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResynthesisObjective {
    MinGateCount,
    MinTwoQubitCount,
    MinDepth,
    MinDuration,
    MinError,
}

/// Stable profile describing where and why a resynthesis pass is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResynthesisProfile {
    pub scope: ResynthesisScope,
    pub budget: ResynthesisBudget,
    pub objective: ResynthesisObjective,
    pub target_aware: bool,
}

impl ResynthesisProfile {
    /// Profile for a cheap logical-level cleanup stage.
    pub const fn logical_local(
        scope: ResynthesisScope,
        budget: ResynthesisBudget,
        objective: ResynthesisObjective,
    ) -> Self {
        Self {
            scope,
            budget,
            objective,
            target_aware: false,
        }
    }

    /// Profile for a target-aware resynthesis stage that uses device knowledge.
    pub const fn target_local(
        scope: ResynthesisScope,
        budget: ResynthesisBudget,
        objective: ResynthesisObjective,
    ) -> Self {
        Self {
            scope,
            budget,
            objective,
            target_aware: true,
        }
    }
}

/// Specialized transformer for bounded region resynthesis.
///
/// Implementors should:
/// - declare a narrow replacement scope
/// - operate under an explicit budget
/// - only accept a replacement when the chosen objective improves
pub trait Resynthesizer: Transformer {
    /// Returns the stable operational profile of this resynthesizer.
    fn profile(&self) -> ResynthesisProfile;

    /// Returns the transform descriptor used by the workflow layer.
    fn descriptor(&self) -> &'static TransformDescriptor;
}

#[cfg(test)]
mod tests {
    use super::{ResynthesisBudget, ResynthesisObjective, ResynthesisProfile, ResynthesisScope};

    #[test]
    fn logical_profile_is_not_target_aware() {
        let profile = ResynthesisProfile::logical_local(
            ResynthesisScope::SingleQubitRuns,
            ResynthesisBudget::local(1, 32),
            ResynthesisObjective::MinGateCount,
        );

        assert_eq!(profile.scope, ResynthesisScope::SingleQubitRuns);
        assert_eq!(profile.budget, ResynthesisBudget::local(1, 32));
        assert_eq!(profile.objective, ResynthesisObjective::MinGateCount);
        assert!(!profile.target_aware);
    }

    #[test]
    fn target_profile_marks_device_aware_resynthesis() {
        let profile = ResynthesisProfile::target_local(
            ResynthesisScope::TwoQubitBlocks,
            ResynthesisBudget::local(2, 24),
            ResynthesisObjective::MinTwoQubitCount,
        );

        assert_eq!(profile.scope, ResynthesisScope::TwoQubitBlocks);
        assert_eq!(profile.budget.max_qubits, 2);
        assert_eq!(profile.budget.max_operations, 24);
        assert!(profile.target_aware);
    }
}
