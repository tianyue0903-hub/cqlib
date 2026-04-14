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

//! Workflow-level orchestration for compiler transforms.
//!
//! The workflow layer is the user-facing compilation path selector. It turns a
//! high-level preset into a stable sequence of named transform steps, executes
//! them in order, and returns a structured report describing what ran.

use crate::compiler::api::{CompileOptions, CompilePreset};
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::Transformer;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// Structured execution report for one workflow step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkflowStepReport {
    /// Stable workflow-local step name.
    pub name: String,
    /// Name of the underlying transformer descriptor.
    pub transform_name: String,
    /// Whether the step reported a semantic change.
    pub changed: bool,
    /// Notes emitted by the step outcome.
    pub notes: Vec<String>,
}

/// Aggregate execution report for a compiler workflow.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkflowReport {
    /// Stable workflow name.
    pub name: String,
    /// Whether any step reported a semantic change.
    pub changed: bool,
    /// Number of steps that executed.
    pub executed_steps: usize,
    /// Per-step execution reports in run order.
    pub steps: Vec<WorkflowStepReport>,
    /// Flattened notes from all executed steps.
    pub notes: Vec<String>,
}

/// One named step in a compiler workflow.
pub struct WorkflowStep {
    name: &'static str,
    transform: Box<dyn Transformer>,
}

impl WorkflowStep {
    /// Creates a named workflow step around a transformer.
    pub fn new(name: &'static str, transform: Box<dyn Transformer>) -> Self {
        Self { name, transform }
    }

    /// Returns the stable workflow-local step name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the underlying transformer.
    pub fn transform(&self) -> &dyn Transformer {
        self.transform.as_ref()
    }
}

/// Sequential orchestration of compiler transforms.
///
/// A workflow is the user-facing unit of compilation policy. It defines a stable
/// transform order and produces a single report, while each transform remains
/// responsible for its own local validation and state mutation.
pub struct CompilerWorkflow {
    name: &'static str,
    steps: Vec<WorkflowStep>,
}

impl CompilerWorkflow {
    /// Creates an empty workflow with a stable external name.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            steps: Vec::new(),
        }
    }

    /// Adds a named step to the workflow.
    pub fn with_step(mut self, name: &'static str, transform: Box<dyn Transformer>) -> Self {
        self.steps.push(WorkflowStep::new(name, transform));
        self
    }

    /// Adds a transform step whose workflow step name matches the transformer descriptor.
    pub fn with_transform(mut self, transform: Box<dyn Transformer>) -> Self {
        let name = transform.descriptor().name;
        self.steps.push(WorkflowStep::new(name, transform));
        self
    }

    /// Returns the workflow name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the number of configured transform steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether the workflow has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the configured steps in order.
    pub fn steps(&self) -> &[WorkflowStep] {
        &self.steps
    }

    /// Executes all workflow steps in order.
    pub fn run(&self, ctx: &mut CompilerContext) -> Result<WorkflowReport, CompilerError> {
        ctx.metadata_mut().workflow_name = Some(self.name.to_string());

        let mut report = WorkflowReport {
            name: self.name.to_string(),
            changed: false,
            executed_steps: 0,
            steps: Vec::new(),
            notes: Vec::new(),
        };

        for step in &self.steps {
            let outcome = step.transform.run(ctx)?;
            report.changed |= outcome.changed;
            report.executed_steps += 1;
            report.notes.extend(outcome.notes.iter().cloned());
            report.steps.push(WorkflowStepReport {
                name: step.name.to_string(),
                transform_name: step.transform.descriptor().name.to_string(),
                changed: outcome.changed,
                notes: outcome.notes,
            });
        }

        Ok(report)
    }
}

/// Builds the stable workflow for a compile preset and option set.
///
/// The current implementation keeps the preset-to-workflow mapping explicit and
/// intentionally simple. Optional stages are inserted only at build time.
pub fn build_workflow(preset: CompilePreset, _options: &CompileOptions) -> CompilerWorkflow {
    match preset {
        CompilePreset::LogicalOptimize => CompilerWorkflow::new("logical.optimize"),
        CompilePreset::TargetLowering => CompilerWorkflow::new("target.lowering"),
        CompilePreset::ExecutionReady => CompilerWorkflow::new("execution.ready"),
    }
}

#[cfg(test)]
mod tests {
    use super::{CompilerWorkflow, WorkflowReport, WorkflowStepReport, build_workflow};
    use crate::circuit::Circuit;
    use crate::compiler::api::{CompileOptions, CompilePreset};
    use crate::compiler::context::CompilerContext;
    use crate::compiler::error::CompilerError;
    use crate::compiler::transform::{TransformDescriptor, TransformOutcome, Transformer};

    struct TagStep;

    static TAG_STEP_DESCRIPTOR: TransformDescriptor =
        TransformDescriptor::new("test.tag", "Tags workflow metadata").modifies_circuit();

    impl Transformer for TagStep {
        fn descriptor(&self) -> &'static TransformDescriptor {
            &TAG_STEP_DESCRIPTOR
        }

        fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
            ctx.metadata_mut().tags.push("visited".to_string());
            Ok(TransformOutcome::changed().with_note("tagged"))
        }
    }

    #[test]
    fn empty_workflow_reports_no_changes() {
        let workflow = CompilerWorkflow::new("empty");
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let report = workflow.run(&mut ctx).unwrap();

        assert_eq!(
            report,
            WorkflowReport {
                name: "empty".to_string(),
                changed: false,
                executed_steps: 0,
                steps: vec![],
                notes: vec![],
            }
        );
        assert_eq!(ctx.metadata().workflow_name.as_deref(), Some("empty"));
    }

    #[test]
    fn workflow_runs_steps_in_order_and_aggregates_notes() {
        let workflow = CompilerWorkflow::new("logical.optimize").with_transform(Box::new(TagStep));
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let report = workflow.run(&mut ctx).unwrap();

        assert!(report.changed);
        assert_eq!(report.executed_steps, 1);
        assert_eq!(report.notes, vec!["tagged"]);
        assert_eq!(
            report.steps,
            vec![WorkflowStepReport {
                name: "test.tag".to_string(),
                transform_name: "test.tag".to_string(),
                changed: true,
                notes: vec!["tagged".to_string()],
            }]
        );
        assert_eq!(ctx.metadata().tags, vec!["visited"]);
    }

    #[test]
    fn workflow_preserves_explicit_step_name() {
        let workflow =
            CompilerWorkflow::new("logical.optimize").with_step("canonicalize", Box::new(TagStep));
        let mut ctx = CompilerContext::new(Circuit::new(1));

        let report = workflow.run(&mut ctx).unwrap();

        assert_eq!(report.executed_steps, 1);
        assert_eq!(report.steps[0].name, "canonicalize");
        assert_eq!(report.steps[0].transform_name, "test.tag");
    }

    #[test]
    fn build_workflow_maps_presets_to_stable_names() {
        let options = CompileOptions::new();

        assert_eq!(
            build_workflow(CompilePreset::LogicalOptimize, &options).name(),
            "logical.optimize"
        );
        assert_eq!(
            build_workflow(CompilePreset::TargetLowering, &options).name(),
            "target.lowering"
        );
        assert_eq!(
            build_workflow(CompilePreset::ExecutionReady, &options).name(),
            "execution.ready"
        );
    }
}
