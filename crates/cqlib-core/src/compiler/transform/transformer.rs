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

use crate::compiler::analysis::{InstructionStats, LogicalCost};
use crate::compiler::artifact::CompileDiagnostic;
use crate::compiler::context::{CompilerContext, ContextChangeSet};
use crate::compiler::error::CompilerError;
use crate::compiler::transform::descriptor::TransformDescriptor;

/// Snapshot of the pass-visible logical statistics surface.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformStatsSnapshot {
    pub instruction_stats: InstructionStats,
    pub logical_cost: LogicalCost,
}

impl TransformStatsSnapshot {
    pub fn new(instruction_stats: InstructionStats, logical_cost: LogicalCost) -> Self {
        Self {
            instruction_stats,
            logical_cost,
        }
    }
}

/// Before/after logical statistics for one transform execution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformStatsChange {
    pub before: TransformStatsSnapshot,
    pub after: TransformStatsSnapshot,
}

impl TransformStatsChange {
    pub fn new(before: TransformStatsSnapshot, after: TransformStatsSnapshot) -> Self {
        Self { before, after }
    }

    pub fn from_parts(
        before_stats: InstructionStats,
        after_stats: InstructionStats,
        before_cost: LogicalCost,
        after_cost: LogicalCost,
    ) -> Self {
        Self {
            before: TransformStatsSnapshot::new(before_stats, before_cost),
            after: TransformStatsSnapshot::new(after_stats, after_cost),
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.after = other.after.clone();
    }
}

fn sample_stats(ctx: &mut CompilerContext) -> Result<TransformStatsSnapshot, CompilerError> {
    Ok(TransformStatsSnapshot::new(
        ctx.instruction_stats()?.clone(),
        ctx.cost_analysis()?.logical.clone(),
    ))
}

/// Lightweight execution result returned by a transformer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransformOutcome {
    pub changed: bool,
    pub changes: ContextChangeSet,
    pub notes: Vec<String>,
    pub diagnostics: Vec<CompileDiagnostic>,
    pub stats_change: Option<TransformStatsChange>,
}

impl TransformOutcome {
    /// Creates an outcome that reports no state change.
    pub fn unchanged() -> Self {
        Self::default()
    }

    /// Creates an outcome that reports a state change.
    pub fn changed() -> Self {
        Self {
            changed: true,
            changes: ContextChangeSet::none(),
            notes: Vec::new(),
            diagnostics: Vec::new(),
            stats_change: None,
        }
    }

    /// Attaches a compiler-context change set to the outcome.
    pub fn with_changes(mut self, changes: ContextChangeSet) -> Self {
        self.changes.extend(changes);
        self.changed |= self.changes.has_effects();
        self
    }

    /// Appends a diagnostic note to the outcome.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Appends a structured diagnostic to the outcome.
    pub fn with_diagnostic(mut self, diagnostic: CompileDiagnostic) -> Self {
        self.diagnostics.push(diagnostic);
        self
    }

    /// Attaches a before/after statistics snapshot to the outcome.
    pub fn with_stats_change(mut self, stats_change: TransformStatsChange) -> Self {
        self.stats_change = Some(stats_change);
        self
    }

    /// Merges another outcome into this one.
    pub fn extend(&mut self, other: Self) {
        self.changed |= other.changed;
        self.changes.extend(other.changes);
        self.notes.extend(other.notes);
        self.diagnostics.extend(other.diagnostics);

        if let Some(other_change) = other.stats_change {
            match self.stats_change.as_mut() {
                Some(current) => current.merge(&other_change),
                None => self.stats_change = Some(other_change),
            }
        }
    }
}

/// Smallest executable unit in the compiler transform layer.
pub trait Transformer {
    fn descriptor(&self) -> &'static TransformDescriptor;

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError>;

    fn run(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        self.descriptor().validate(ctx)?;
        let before = sample_stats(ctx)?;
        let mut outcome = self.transform(ctx)?;
        ctx.apply_changes(outcome.changes.clone());
        outcome.changed |= outcome.changes.has_effects();

        if outcome.changed {
            if ctx
                .verification_config()
                .verify_after_each_changed_transform
            {
                ctx.verify()?;
            }

            let after = sample_stats(ctx)?;
            outcome.stats_change = Some(TransformStatsChange::new(before, after));
        } else {
            outcome.stats_change = None;
        }

        Ok(outcome)
    }
}

/// Sequential composition of multiple transformers.
pub struct CompositeTransformer {
    descriptor: &'static TransformDescriptor,
    transformers: Vec<Box<dyn Transformer>>,
}

impl CompositeTransformer {
    pub fn new(
        descriptor: &'static TransformDescriptor,
        transformers: Vec<Box<dyn Transformer>>,
    ) -> Self {
        Self {
            descriptor,
            transformers,
        }
    }
}

impl Transformer for CompositeTransformer {
    fn descriptor(&self) -> &'static TransformDescriptor {
        self.descriptor
    }

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        let mut aggregate = TransformOutcome::unchanged();

        for transformer in &self.transformers {
            let outcome = transformer.run(ctx)?;
            aggregate.extend(outcome);
        }

        Ok(aggregate)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompositeTransformer, TransformOutcome, TransformStatsChange, Transformer};
    use crate::circuit::{Circuit, CircuitParam, Operation, Qubit, StandardGate};
    use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
    use crate::compiler::context::{CompilerContext, ContextChangeSet, VerificationConfig};
    use crate::compiler::error::CompilerError;
    use crate::compiler::transform::descriptor::TransformDescriptor;
    use indexmap::IndexSet;
    use smallvec::smallvec;

    struct AddGate;

    static ADD_GATE_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.add_gate", "Adds one gate").modifies_circuit();

    impl Transformer for AddGate {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &ADD_GATE_DESCRIPTOR
        }

        fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            ctx.circuit_mut().h(Qubit::new(0)).unwrap();
            Ok(TransformOutcome::changed()
                .with_changes(ContextChangeSet::circuit_changed())
                .with_note("gate added")
                .with_diagnostic(CompileDiagnostic {
                    severity: DiagnosticSeverity::Info,
                    code: "test.add_gate.added",
                    message: "gate added".to_string(),
                }))
        }
    }

    struct ManualStatsOverride;

    static MANUAL_OVERRIDE_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.override", "Returns incorrect manual stats")
            .modifies_circuit();

    impl Transformer for ManualStatsOverride {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &MANUAL_OVERRIDE_DESCRIPTOR
        }

        fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            ctx.circuit_mut()
                .append(StandardGate::X.into(), [Qubit::new(0)], [], None)?;

            Ok(TransformOutcome::changed()
                .with_changes(ContextChangeSet::circuit_changed())
                .with_stats_change(TransformStatsChange::from_parts(
                    crate::compiler::analysis::InstructionStats {
                        total_ops: 100,
                        ..crate::compiler::analysis::InstructionStats::default()
                    },
                    crate::compiler::analysis::InstructionStats::default(),
                    crate::compiler::analysis::LogicalCost {
                        total_ops: 100,
                        depth_estimate: 100,
                        ..crate::compiler::analysis::LogicalCost::default()
                    },
                    crate::compiler::analysis::LogicalCost::default(),
                )))
        }
    }

    struct NoOp;

    static NOOP_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.noop", "No-op transformer");

    impl Transformer for NoOp {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &NOOP_DESCRIPTOR
        }

        fn transform(&self, _ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            Ok(TransformOutcome::unchanged())
        }
    }

    struct DeviceOnly;

    static DEVICE_ONLY_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.device", "Needs a device").requires_device();

    impl Transformer for DeviceOnly {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &DEVICE_ONLY_DESCRIPTOR
        }

        fn transform(&self, _ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            Ok(TransformOutcome::changed())
        }
    }

    struct InvalidCircuit;

    static INVALID_CIRCUIT_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.invalid", "Produces invalid circuit").modifies_circuit();

    impl Transformer for InvalidCircuit {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &INVALID_CIRCUIT_DESCRIPTOR
        }

        fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            *ctx.circuit_mut() = Circuit::from_parts(
                IndexSet::from_iter([Qubit::new(0)]),
                IndexSet::default(),
                IndexSet::default(),
                vec![Operation {
                    instruction: StandardGate::RX.into(),
                    qubits: smallvec![Qubit::new(0)],
                    params: smallvec![CircuitParam::Index(77)],
                    label: None,
                }],
                CircuitParam::Fixed(0.0),
            );
            Ok(TransformOutcome::changed().with_changes(ContextChangeSet::circuit_changed()))
        }
    }

    #[test]
    fn unchanged_outcome_is_default() {
        let outcome = TransformOutcome::unchanged();

        assert!(!outcome.changed);
        assert!(outcome.notes.is_empty());
    }

    #[test]
    fn changed_outcome_records_note() {
        let outcome = TransformOutcome::changed().with_note("rewritten");

        assert!(outcome.changed);
        assert_eq!(outcome.notes, vec!["rewritten"]);
        assert!(outcome.diagnostics.is_empty());
        assert!(outcome.stats_change.is_none());
    }

    #[test]
    fn outcome_collects_diagnostics_and_stats_change() {
        let change = TransformStatsChange::from_parts(
            crate::compiler::analysis::InstructionStats::default(),
            crate::compiler::analysis::InstructionStats {
                total_ops: 1,
                ..crate::compiler::analysis::InstructionStats::default()
            },
            crate::compiler::analysis::LogicalCost::default(),
            crate::compiler::analysis::LogicalCost {
                total_ops: 1,
                depth_estimate: 1,
                ..crate::compiler::analysis::LogicalCost::default()
            },
        );
        let outcome = TransformOutcome::changed()
            .with_diagnostic(CompileDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: "test.outcome.warning",
                message: "something changed".to_string(),
            })
            .with_stats_change(change.clone());

        assert_eq!(outcome.diagnostics.len(), 1);
        assert_eq!(outcome.stats_change, Some(change));
    }

    #[test]
    fn composite_transformer_aggregates_changes_and_notes() {
        static COMPOSITE_DESCRIPTOR: TransformDescriptor =
            TransformDescriptor::new("test.composite", "Composite transformer");

        let composite = CompositeTransformer::new(
            &COMPOSITE_DESCRIPTOR,
            vec![Box::new(NoOp), Box::new(AddGate)],
        );
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let outcome = composite.run(&mut ctx).unwrap();

        assert!(outcome.changed);
        assert_eq!(outcome.notes, vec!["gate added"]);
        assert_eq!(outcome.diagnostics.len(), 1);
        assert_eq!(ctx.circuit().operations().len(), 1);
        let stats_change = outcome
            .stats_change
            .expect("stats change should be sampled");
        assert_eq!(stats_change.before.instruction_stats.total_ops, 0);
        assert_eq!(stats_change.after.instruction_stats.total_ops, 1);
        assert_eq!(stats_change.after.logical_cost.depth_estimate, 1);
    }

    #[test]
    fn run_overrides_manual_stats_change_with_sampled_values() {
        let transformer = ManualStatsOverride;
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let outcome = transformer.run(&mut ctx).unwrap();
        let stats_change = outcome
            .stats_change
            .expect("stats change should be sampled");

        assert_eq!(stats_change.before.instruction_stats.total_ops, 0);
        assert_eq!(stats_change.after.instruction_stats.total_ops, 1);
        assert_eq!(stats_change.after.logical_cost.total_ops, 1);
        assert_eq!(stats_change.after.logical_cost.depth_estimate, 1);
    }

    #[test]
    fn run_validates_descriptor_requirements_before_transform() {
        let transformer = DeviceOnly;
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let err = transformer.run(&mut ctx).unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));
    }

    #[test]
    fn run_verifies_changed_transform_when_enabled() {
        let transformer = InvalidCircuit;
        let mut ctx = CompilerContext::new(Circuit::new(1));
        ctx.set_verification_config(
            VerificationConfig::default().after_each_changed_transform(true),
        );

        let err = transformer.run(&mut ctx).unwrap_err();
        assert!(matches!(err, CompilerError::InvariantViolation(_)));
    }
}
