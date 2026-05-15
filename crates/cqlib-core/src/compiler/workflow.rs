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

use crate::compiler::api::{CompileOptions, CompilePreset};
use crate::compiler::artifact::CompileDiagnostic;
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::Transformer;
use crate::compiler::transform::rewrite::KnowledgeRewriter;
use crate::compiler::transform::transformer::TransformStatsChange;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

type WorkflowPredicate =
    dyn Fn(&mut CompilerContext) -> Result<bool, CompilerError> + Send + Sync + 'static;
type WorkflowRepeatPredicate =
    dyn Fn(&CompilerContext, usize, bool) -> Result<bool, CompilerError> + Send + Sync + 'static;
type WorkflowScore = dyn Fn(&CompilerContext) -> Result<i64, CompilerError> + Send + Sync + 'static;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkflowStepReport {
    pub name: String,
    pub transform_name: String,
    pub changed: bool,
    pub notes: Vec<String>,
    pub diagnostics: Vec<CompileDiagnostic>,
    pub stats_change: Option<TransformStatsChange>,
    pub iteration: Option<usize>,
    pub branch: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkflowReport {
    pub name: String,
    pub changed: bool,
    pub executed_steps: usize,
    pub steps: Vec<WorkflowStepReport>,
    pub notes: Vec<String>,
    pub diagnostics: Vec<CompileDiagnostic>,
    pub stats_change: Option<TransformStatsChange>,
}

pub struct WorkflowStep {
    name: &'static str,
    transform: Box<dyn Transformer>,
}

impl WorkflowStep {
    pub fn new(name: &'static str, transform: Box<dyn Transformer>) -> Self {
        Self { name, transform }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn transform(&self) -> &dyn Transformer {
        self.transform.as_ref()
    }
}

enum WorkflowNode {
    Transform {
        step: WorkflowStep,
        next: Option<usize>,
    },
    Condition {
        name: &'static str,
        predicate: Box<WorkflowPredicate>,
        on_true: Option<usize>,
        on_false: Option<usize>,
    },
    RepeatUntil {
        name: &'static str,
        transform: Box<dyn Transformer>,
        max_iters: usize,
        should_continue: Box<WorkflowRepeatPredicate>,
        next: Option<usize>,
    },
    SelectBest {
        name: &'static str,
        branches: Vec<CompilerWorkflow>,
        scorer: Box<WorkflowScore>,
        next: Option<usize>,
    },
}

pub struct CompilerWorkflow {
    name: &'static str,
    nodes: Vec<WorkflowNode>,
    entry: Option<usize>,
    tail: Option<usize>,
}

impl CompilerWorkflow {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            entry: None,
            tail: None,
        }
    }

    pub fn with_step(mut self, name: &'static str, transform: Box<dyn Transformer>) -> Self {
        let node = self.push_transform_node(name, transform);
        self.link_from_tail(node);
        self.tail = Some(node);
        self
    }

    pub fn with_transform(mut self, transform: Box<dyn Transformer>) -> Self {
        let name = transform.descriptor().name;
        let node = self.push_transform_node(name, transform);
        self.link_from_tail(node);
        self.tail = Some(node);
        self
    }

    pub fn with_condition(
        mut self,
        name: &'static str,
        predicate: impl Fn(&mut CompilerContext) -> Result<bool, CompilerError> + Send + Sync + 'static,
        on_true: CompilerWorkflow,
        on_false: Option<CompilerWorkflow>,
    ) -> Self {
        let true_entry = self.embed_workflow(on_true);
        let false_entry = on_false.and_then(|workflow| self.embed_workflow(workflow));
        let node = self.nodes.len();
        self.nodes.push(WorkflowNode::Condition {
            name,
            predicate: Box::new(predicate),
            on_true: true_entry,
            on_false: false_entry,
        });
        self.link_from_tail(node);
        self.tail = Some(node);
        self
    }

    pub fn with_repeat_until(
        mut self,
        name: &'static str,
        transform: Box<dyn Transformer>,
        max_iters: usize,
        should_continue: impl Fn(&CompilerContext, usize, bool) -> Result<bool, CompilerError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let node = self.nodes.len();
        self.nodes.push(WorkflowNode::RepeatUntil {
            name,
            transform,
            max_iters,
            should_continue: Box::new(should_continue),
            next: None,
        });
        self.link_from_tail(node);
        self.tail = Some(node);
        self
    }

    pub fn with_select_best(
        mut self,
        name: &'static str,
        branches: Vec<CompilerWorkflow>,
        scorer: impl Fn(&CompilerContext) -> Result<i64, CompilerError> + Send + Sync + 'static,
    ) -> Self {
        let node = self.nodes.len();
        self.nodes.push(WorkflowNode::SelectBest {
            name,
            branches,
            scorer: Box::new(scorer),
            next: None,
        });
        self.link_from_tail(node);
        self.tail = Some(node);
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn steps(&self) -> Vec<&WorkflowStep> {
        self.nodes
            .iter()
            .filter_map(|node| match node {
                WorkflowNode::Transform { step, .. } => Some(step),
                _ => None,
            })
            .collect()
    }

    pub fn run(&self, ctx: &mut CompilerContext) -> Result<WorkflowReport, CompilerError> {
        ctx.metadata_mut().workflow_name = Some(self.name.to_string());

        let mut report = WorkflowReport {
            name: self.name.to_string(),
            changed: false,
            executed_steps: 0,
            steps: Vec::new(),
            notes: Vec::new(),
            diagnostics: Vec::new(),
            stats_change: None,
        };

        let mut cursor = self.entry;
        let mut fuel = self.nodes.len().saturating_mul(16).max(1);

        while let Some(node_index) = cursor {
            if fuel == 0 {
                return Err(CompilerError::WorkflowFailed {
                    name: self.name,
                    reason: "workflow graph exceeded execution fuel; check for unintended cycles"
                        .to_string(),
                });
            }
            fuel -= 1;

            cursor = match &self.nodes[node_index] {
                WorkflowNode::Transform { step, next } => {
                    let outcome = step.transform.run(ctx)?;
                    append_step_report(
                        &mut report,
                        WorkflowStepReport {
                            name: step.name.to_string(),
                            transform_name: step.transform.descriptor().name.to_string(),
                            changed: outcome.changed,
                            notes: outcome.notes,
                            diagnostics: outcome.diagnostics,
                            stats_change: outcome.stats_change,
                            iteration: None,
                            branch: None,
                        },
                    );
                    *next
                }
                WorkflowNode::Condition {
                    name,
                    predicate,
                    on_true,
                    on_false,
                } => {
                    let take_true = predicate(ctx)?;
                    report.executed_steps += 1;
                    report.steps.push(WorkflowStepReport {
                        name: (*name).to_string(),
                        transform_name: "workflow.condition".to_string(),
                        changed: false,
                        notes: vec![if take_true {
                            "condition evaluated to true".to_string()
                        } else {
                            "condition evaluated to false".to_string()
                        }],
                        diagnostics: Vec::new(),
                        stats_change: None,
                        iteration: None,
                        branch: Some(if take_true { "true" } else { "false" }.to_string()),
                    });
                    if take_true { *on_true } else { *on_false }
                }
                WorkflowNode::RepeatUntil {
                    name,
                    transform,
                    max_iters,
                    should_continue,
                    next,
                } => {
                    for iteration in 0..*max_iters {
                        let outcome = transform.run(ctx)?;
                        let changed = outcome.changed;
                        append_step_report(
                            &mut report,
                            WorkflowStepReport {
                                name: (*name).to_string(),
                                transform_name: transform.descriptor().name.to_string(),
                                changed: outcome.changed,
                                notes: outcome.notes,
                                diagnostics: outcome.diagnostics,
                                stats_change: outcome.stats_change,
                                iteration: Some(iteration),
                                branch: None,
                            },
                        );
                        if !should_continue(ctx, iteration + 1, changed)? {
                            break;
                        }
                    }
                    *next
                }
                WorkflowNode::SelectBest {
                    name,
                    branches,
                    scorer,
                    next,
                } => {
                    let mut best: Option<(i64, usize, CompilerContext, WorkflowReport)> = None;

                    for (index, branch) in branches.iter().enumerate() {
                        let mut branch_ctx = ctx.fork();
                        let branch_report = branch.run(&mut branch_ctx)?;
                        let score = scorer(&branch_ctx)?;

                        let replace = best
                            .as_ref()
                            .is_none_or(|(best_score, _, _, _)| score < *best_score);
                        if replace {
                            best = Some((score, index, branch_ctx, branch_report));
                        }
                    }

                    let Some((score, index, best_ctx, branch_report)) = best else {
                        return Err(CompilerError::WorkflowFailed {
                            name: self.name,
                            reason: format!("select_best node '{}' has no branches", name),
                        });
                    };

                    ctx.replace_with(best_ctx);
                    report.executed_steps += 1;
                    report.steps.push(WorkflowStepReport {
                        name: (*name).to_string(),
                        transform_name: "workflow.select_best".to_string(),
                        changed: branch_report.changed,
                        notes: vec![format!("selected branch {} with score {}", index, score)],
                        diagnostics: Vec::new(),
                        stats_change: branch_report.stats_change.clone(),
                        iteration: None,
                        branch: Some(index.to_string()),
                    });
                    report.changed |= branch_report.changed;
                    report.notes.extend(branch_report.notes);
                    report.diagnostics.extend(branch_report.diagnostics);
                    report.steps.extend(branch_report.steps);
                    if let Some(change) = branch_report.stats_change {
                        merge_stats_change(&mut report, change);
                    }
                    *next
                }
            };
        }

        Ok(report)
    }

    fn push_transform_node(
        &mut self,
        name: &'static str,
        transform: Box<dyn Transformer>,
    ) -> usize {
        let index = self.nodes.len();
        self.nodes.push(WorkflowNode::Transform {
            step: WorkflowStep::new(name, transform),
            next: None,
        });
        if self.entry.is_none() {
            self.entry = Some(index);
        }
        index
    }

    fn link_from_tail(&mut self, next_index: usize) {
        if self.entry.is_none() {
            self.entry = Some(next_index);
        }

        if let Some(tail) = self.tail {
            match &mut self.nodes[tail] {
                WorkflowNode::Transform { next, .. }
                | WorkflowNode::RepeatUntil { next, .. }
                | WorkflowNode::SelectBest { next, .. } => {
                    if next.is_none() {
                        *next = Some(next_index);
                    }
                }
                WorkflowNode::Condition { .. } => {}
            }
        }
    }

    fn embed_workflow(&mut self, workflow: CompilerWorkflow) -> Option<usize> {
        let entry = workflow.entry?;
        let offset = self.nodes.len();
        let mapped_entry = entry + offset;

        for node in workflow.nodes {
            self.nodes.push(match node {
                WorkflowNode::Transform { step, next } => WorkflowNode::Transform {
                    step,
                    next: next.map(|value| value + offset),
                },
                WorkflowNode::Condition {
                    name,
                    predicate,
                    on_true,
                    on_false,
                } => WorkflowNode::Condition {
                    name,
                    predicate,
                    on_true: on_true.map(|value| value + offset),
                    on_false: on_false.map(|value| value + offset),
                },
                WorkflowNode::RepeatUntil {
                    name,
                    transform,
                    max_iters,
                    should_continue,
                    next,
                } => WorkflowNode::RepeatUntil {
                    name,
                    transform,
                    max_iters,
                    should_continue,
                    next: next.map(|value| value + offset),
                },
                WorkflowNode::SelectBest {
                    name,
                    branches,
                    scorer,
                    next,
                } => WorkflowNode::SelectBest {
                    name,
                    branches,
                    scorer,
                    next: next.map(|value| value + offset),
                },
            });
        }

        Some(mapped_entry)
    }
}

fn append_step_report(report: &mut WorkflowReport, step: WorkflowStepReport) {
    report.changed |= step.changed;
    report.executed_steps += 1;
    report.notes.extend(step.notes.iter().cloned());
    report.diagnostics.extend(step.diagnostics.iter().cloned());
    if let Some(change) = step.stats_change.clone() {
        merge_stats_change(report, change);
    }
    report.steps.push(step);
}

fn merge_stats_change(report: &mut WorkflowReport, change: TransformStatsChange) {
    match report.stats_change.as_mut() {
        Some(existing) => existing.merge(&change),
        None => report.stats_change = Some(change),
    }
}

pub fn build_workflow(preset: CompilePreset, _options: &CompileOptions) -> CompilerWorkflow {
    match preset {
        CompilePreset::LogicalOptimize => CompilerWorkflow::new("logical.optimize")
            .with_transform(Box::new(KnowledgeRewriter::production())),
        CompilePreset::TargetLowering => CompilerWorkflow::new("target.lowering"),
        CompilePreset::ExecutionReady => CompilerWorkflow::new("execution.ready"),
    }
}

#[cfg(test)]
#[path = "./workflow_test.rs"]
mod workflow_test;
