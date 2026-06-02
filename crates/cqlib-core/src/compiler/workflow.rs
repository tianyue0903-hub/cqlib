// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Workflow-level orchestration for the new compiler optimization pipeline.
//!
//! The workflow is a staged composition layer, not an optimization algorithm.
//! It resolves target constraints, runs completed compiler transforms in a
//! deterministic order, and records only the postconditions it can actually
//! verify with the compiler capabilities currently implemented.

use crate::circuit::{Circuit, Instruction};
use crate::compiler::CompilerError;
use crate::compiler::transform::{
    Canonicalizer, KnowledgeRewriter, RewriteConfig, TransformResult, Transformer,
};

use super::{CompileConfig, CompileMode, CompileResult};

/// Per-step execution record produced by a workflow run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowStepReport {
    /// Coarse workflow stage, following the staged-pass-manager model.
    pub stage: &'static str,
    /// Workflow-local step name.
    pub name: &'static str,
    /// Whether this step changed the circuit representation.
    pub changed: bool,
    /// Whether the step was intentionally skipped.
    pub skipped: bool,
    /// Optional skip or configuration note.
    pub reason: Option<String>,
}

/// Compiler optimization workflow built from completed compiler transforms.
pub struct CompilerWorkflow {
    config: CompileConfig,
}

impl CompilerWorkflow {
    /// Creates a compiler workflow from a complete configuration.
    pub const fn new(config: CompileConfig) -> Self {
        Self { config }
    }

    /// Returns the workflow configuration.
    pub const fn config(&self) -> &CompileConfig {
        &self.config
    }

    /// Runs the workflow over `circuit` and returns the rebuilt circuit plus
    /// execution metadata.
    pub fn run(&self, circuit: &Circuit) -> Result<CompileResult, CompilerError> {
        let resolved_target = self.resolve_target_basis()?;

        let mut current = circuit.clone();
        let mut changed = false;
        let mut steps = Vec::new();
        self.record_pre_init(&mut steps, resolved_target.as_ref());

        match self.config.mode {
            CompileMode::Normal => {
                self.run_normal_stages(&mut current, &mut changed, &mut steps, &resolved_target)?
            }
            CompileMode::Enhanced => {
                self.run_enhanced_stages(&mut current, &mut changed, &mut steps, &resolved_target)?
            }
        }

        Ok(CompileResult {
            circuit: current,
            changed,
            mode: self.config.mode,
            steps,
        })
    }

    fn run_normal_stages(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
        target_basis: &Option<Vec<Instruction>>,
    ) -> Result<(), CompilerError> {
        apply_transform(
            current,
            changed,
            steps,
            "init",
            "canonicalize.input",
            &Canonicalizer::production(),
        )?;
        apply_transform(
            current,
            changed,
            steps,
            "optimization",
            "optimize.light",
            &KnowledgeRewriter::production(),
        )?;
        self.apply_target_translation(
            current,
            changed,
            steps,
            target_basis,
            RewriteConfig::lowering(),
        )?;
        apply_transform(
            current,
            changed,
            steps,
            "output",
            "canonicalize.output",
            &Canonicalizer::production(),
        )?;
        Ok(())
    }

    fn run_enhanced_stages(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
        target_basis: &Option<Vec<Instruction>>,
    ) -> Result<(), CompilerError> {
        apply_transform(
            current,
            changed,
            steps,
            "init",
            "canonicalize.input",
            &Canonicalizer::production(),
        )?;
        apply_rewrite(
            current,
            changed,
            steps,
            "optimization",
            "optimize.pre_translation",
            RewriteConfig::production()
                .with_max_rounds(16)
                .with_max_window_ops(32)
                .with_max_pattern_len(12),
        )?;
        self.apply_target_translation(
            current,
            changed,
            steps,
            target_basis,
            RewriteConfig::lowering()
                .with_max_rounds(16)
                .with_max_window_ops(32)
                .with_max_pattern_len(12),
        )?;
        let cleanup_rewrite_config = match target_basis.as_deref() {
            Some(target_basis) => RewriteConfig::production()
                .with_max_rounds(16)
                .with_max_window_ops(32)
                .with_max_pattern_len(12)
                .with_target_instructions(target_basis.to_vec())?,
            None => RewriteConfig::production()
                .with_max_rounds(16)
                .with_max_window_ops(32)
                .with_max_pattern_len(12),
        };
        apply_rewrite(
            current,
            changed,
            steps,
            "optimization",
            "optimize.cleanup",
            cleanup_rewrite_config,
        )?;
        apply_transform(
            current,
            changed,
            steps,
            "optimization",
            "canonicalize.mid",
            &Canonicalizer::production(),
        )?;
        let final_config = match target_basis.as_deref() {
            Some(target_basis) => {
                RewriteConfig::production().with_target_instructions(target_basis.to_vec())?
            }
            None => RewriteConfig::production(),
        };
        apply_rewrite(
            current,
            changed,
            steps,
            "optimization",
            "optimize.final",
            final_config,
        )?;
        apply_transform(
            current,
            changed,
            steps,
            "output",
            "canonicalize.output",
            &Canonicalizer::production(),
        )?;
        Ok(())
    }

    fn apply_target_translation(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
        target_basis: &Option<Vec<Instruction>>,
        config: RewriteConfig,
    ) -> Result<(), CompilerError> {
        let Some(target_basis) = target_basis.as_deref() else {
            steps.push(WorkflowStepReport {
                stage: "translation",
                name: "translate.target_basis",
                changed: false,
                skipped: true,
                reason: Some("no target basis configured".to_string()),
            });
            return Ok(());
        };

        apply_rewrite(
            current,
            changed,
            steps,
            "translation",
            "translate.target_basis",
            config.with_target_instructions(target_basis.to_vec())?,
        )
    }

    fn resolve_target_basis(&self) -> Result<Option<Vec<Instruction>>, CompilerError> {
        if let Some(target_basis) = &self.config.target_basis {
            validate_target_basis_config(&target_basis)?;
            return Ok(Some(target_basis.to_vec()));
        }

        let Some(device) = &self.config.device else {
            return Ok(None);
        };
        let target_basis = device.native_gates();
        if target_basis.is_empty() {
            return Ok(None);
        }

        validate_target_basis_config(target_basis)?;
        Ok(Some(target_basis.to_vec()))
    }

    fn record_pre_init(
        &self,
        steps: &mut Vec<WorkflowStepReport>,
        target_basis: Option<&Vec<Instruction>>,
    ) {
        let reason = match (
            self.config.target_basis.is_some(),
            self.config.device.is_some(),
            target_basis,
        ) {
            (true, _, Some(basis)) => Some(format!(
                "resolved explicit target basis with {} instructions",
                basis.len()
            )),
            (false, true, Some(basis)) => Some(format!(
                "resolved device native target basis with {} instructions",
                basis.len()
            )),
            (false, true, None) => {
                Some("target device has no native gates; basis lowering disabled".to_string())
            }
            (false, false, None) => Some("no target constraints configured".to_string()),
            _ => None,
        };

        steps.push(WorkflowStepReport {
            stage: "pre_init",
            name: "resolve.target",
            changed: false,
            skipped: false,
            reason,
        });
    }
}

fn apply_transform(
    current: &mut Circuit,
    workflow_changed: &mut bool,
    steps: &mut Vec<WorkflowStepReport>,
    stage: &'static str,
    name: &'static str,
    transform: &dyn Transformer,
) -> Result<(), CompilerError> {
    let TransformResult { circuit, changed } = transform.transform(current)?;
    *current = circuit;
    *workflow_changed |= changed;
    steps.push(WorkflowStepReport {
        stage,
        name,
        changed,
        skipped: false,
        reason: None,
    });
    Ok(())
}

fn apply_rewrite(
    current: &mut Circuit,
    workflow_changed: &mut bool,
    steps: &mut Vec<WorkflowStepReport>,
    stage: &'static str,
    name: &'static str,
    config: RewriteConfig,
) -> Result<(), CompilerError> {
    apply_transform(
        current,
        workflow_changed,
        steps,
        stage,
        name,
        &KnowledgeRewriter::new(config),
    )
}

fn validate_target_basis_config(target_basis: &[Instruction]) -> Result<(), CompilerError> {
    if target_basis.is_empty() {
        return Err(CompilerError::InvalidInput(
            "workflow target basis must not be empty".to_string(),
        ));
    }

    for instruction in target_basis {
        if !matches!(
            instruction,
            Instruction::Standard(_) | Instruction::McGate(_)
        ) {
            return Err(CompilerError::InvalidInput(format!(
                "unsupported workflow target instruction {instruction:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "./workflow_test.rs"]
mod workflow_test;
