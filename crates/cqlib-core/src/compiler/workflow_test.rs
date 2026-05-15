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

use super::{CompilerWorkflow, WorkflowReport, WorkflowStepReport, build_workflow};
use crate::circuit::{Circuit, Qubit};
use crate::compiler::api::{CompileOptions, CompilePreset};
use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
use crate::compiler::context::{CompilerContext, ContextChangeSet};
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
        ctx.circuit_mut().h(Qubit::new(0)).unwrap();
        Ok(TransformOutcome::changed()
            .with_changes(ContextChangeSet::circuit_changed())
            .with_note("tagged")
            .with_diagnostic(CompileDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: "test.tag.visited",
                message: "tagged".to_string(),
            }))
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
            diagnostics: vec![],
            stats_change: None,
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
        report.diagnostics,
        vec![CompileDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: "test.tag.visited",
            message: "tagged".to_string(),
        }]
    );
    assert_eq!(ctx.metadata().tags, vec!["visited"]);
    assert_eq!(ctx.circuit().operations().len(), 1);
}

#[test]
fn workflow_condition_can_skip_branch() {
    let workflow = CompilerWorkflow::new("logical.optimize").with_condition(
        "cond",
        |_| Ok(false),
        CompilerWorkflow::new("true").with_transform(Box::new(TagStep)),
        Some(CompilerWorkflow::new("false")),
    );
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let report = workflow.run(&mut ctx).unwrap();

    assert_eq!(report.executed_steps, 1);
    assert!(!report.changed);
    assert_eq!(ctx.circuit().operations().len(), 0);
}

#[test]
fn workflow_repeat_until_stops_when_predicate_fails() {
    let workflow = CompilerWorkflow::new("logical.optimize").with_repeat_until(
        "repeat",
        Box::new(TagStep),
        4,
        |ctx, _, _| Ok(ctx.circuit().operations().len() < 2),
    );
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let report = workflow.run(&mut ctx).unwrap();

    assert_eq!(report.executed_steps, 2);
    assert_eq!(ctx.circuit().operations().len(), 2);
}

#[test]
fn workflow_select_best_commits_best_branch() {
    let branch_a = CompilerWorkflow::new("a").with_transform(Box::new(TagStep));
    let branch_b = CompilerWorkflow::new("b");
    let workflow = CompilerWorkflow::new("logical.optimize").with_select_best(
        "pick",
        vec![branch_a, branch_b],
        |ctx| Ok(ctx.circuit().operations().len() as i64),
    );
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let report = workflow.run(&mut ctx).unwrap();

    assert_eq!(ctx.circuit().operations().len(), 0);
    assert_eq!(report.steps[0].transform_name, "workflow.select_best");
}

#[test]
fn build_workflow_returns_named_presets() {
    let options = CompileOptions::default();

    let logical = build_workflow(CompilePreset::LogicalOptimize, &options);
    let lowering = build_workflow(CompilePreset::TargetLowering, &options);
    let ready = build_workflow(CompilePreset::ExecutionReady, &options);

    assert_eq!(logical.name(), "logical.optimize");
    assert_eq!(logical.len(), 1);
    assert_eq!(logical.steps()[0].name(), "rewrite.knowledge");
    assert_eq!(
        logical.steps()[0].transform().descriptor().name,
        "rewrite.knowledge"
    );
    assert_eq!(lowering.name(), "target.lowering");
    assert!(lowering.is_empty());
    assert_eq!(ready.name(), "execution.ready");
    assert!(ready.is_empty());
}

#[test]
fn logical_optimize_workflow_runs_knowledge_rewrite() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(0)).unwrap();
    let mut ctx = CompilerContext::new(circuit);

    let workflow = build_workflow(CompilePreset::LogicalOptimize, &CompileOptions::default());
    let report = workflow.run(&mut ctx).unwrap();

    assert!(report.changed);
    assert_eq!(report.executed_steps, 1);
    assert_eq!(report.steps[0].transform_name, "rewrite.knowledge");
    assert!(ctx.circuit().operations().is_empty());
}

#[test]
fn target_presets_do_not_include_logical_rewrite_yet() {
    let options = CompileOptions::default();

    let lowering = build_workflow(CompilePreset::TargetLowering, &options);
    let ready = build_workflow(CompilePreset::ExecutionReady, &options);

    assert_eq!(lowering.name(), "target.lowering");
    assert!(lowering.is_empty());
    assert_eq!(ready.name(), "execution.ready");
    assert!(ready.is_empty());
}

#[test]
fn workflow_step_report_keeps_transform_name() {
    let workflow =
        CompilerWorkflow::new("logical.optimize").with_step("canonicalize", Box::new(TagStep));
    let mut ctx = CompilerContext::new(Circuit::new(1));

    let report = workflow.run(&mut ctx).unwrap();

    assert_eq!(
        report.steps,
        vec![WorkflowStepReport {
            name: "canonicalize".to_string(),
            transform_name: "test.tag".to_string(),
            changed: true,
            notes: vec!["tagged".to_string()],
            diagnostics: vec![CompileDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: "test.tag.visited",
                message: "tagged".to_string(),
            }],
            stats_change: report.steps[0].stats_change.clone(),
            iteration: None,
            branch: None,
        }]
    );
}
