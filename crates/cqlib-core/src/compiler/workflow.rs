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

use crate::circuit::{Circuit, ControlFlow, Instruction, Operation};
use crate::compiler::CompilerError;
use crate::compiler::resource::ResourceLimits;
use crate::compiler::transform::decompose::expand_definitions;
use crate::compiler::transform::decompose::mc_gate::{McGateDecomposeConfig, decompose_mc_gates};
use crate::compiler::transform::decompose::unitary::decompose::{
    UnitaryDecomposeConfig, decompose_unitaries,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkflowStep {
    stage: &'static str,
    name: &'static str,
}

const STEP_RESOLVE_TARGET: WorkflowStep = WorkflowStep {
    stage: "pre_init",
    name: "resolve.target",
};
const STEP_VALIDATE_RESOURCES: WorkflowStep = WorkflowStep {
    stage: "pre_init",
    name: "validate.resources",
};
const STEP_CANONICALIZE_INPUT: WorkflowStep = WorkflowStep {
    stage: "init",
    name: "canonicalize.input",
};
const STEP_DECOMPOSE_DEFINITIONS: WorkflowStep = WorkflowStep {
    stage: "init",
    name: "decompose.definitions",
};
const STEP_OPTIMIZE_PRE_DECOMPOSITION: WorkflowStep = WorkflowStep {
    stage: "optimization",
    name: "optimize.pre_decomposition",
};
const STEP_DECOMPOSE_UNITARY: WorkflowStep = WorkflowStep {
    stage: "translation",
    name: "decompose.unitary",
};
const STEP_DECOMPOSE_MC_GATES: WorkflowStep = WorkflowStep {
    stage: "translation",
    name: "decompose.mc_gates",
};
const STEP_CANONICALIZE_AFTER_DECOMPOSITION: WorkflowStep = WorkflowStep {
    stage: "optimization",
    name: "canonicalize.after_decomposition",
};
const STEP_OPTIMIZE_POST_DECOMPOSITION: WorkflowStep = WorkflowStep {
    stage: "optimization",
    name: "optimize.post_decomposition",
};
const STEP_TRANSLATE_TARGET_BASIS: WorkflowStep = WorkflowStep {
    stage: "translation",
    name: "translate.target_basis",
};
const STEP_OPTIMIZE_TARGET_CLEANUP: WorkflowStep = WorkflowStep {
    stage: "optimization",
    name: "optimize.target_cleanup",
};
const STEP_CANONICALIZE_OUTPUT: WorkflowStep = WorkflowStep {
    stage: "output",
    name: "canonicalize.output",
};

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
        self.validate_resources(circuit, &mut steps)?;

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
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_INPUT,
            &Canonicalizer::production(),
        )?;
        self.apply_definition_decomposition(current, changed, steps)?;
        apply_rewrite(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_PRE_DECOMPOSITION,
            RewriteConfig::production(),
        )?;
        self.apply_unitary_decomposition(current, changed, steps)?;
        self.apply_mc_gate_decomposition(current, changed, steps)?;
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_AFTER_DECOMPOSITION,
            &Canonicalizer::production(),
        )?;
        apply_rewrite(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_POST_DECOMPOSITION,
            RewriteConfig::production(),
        )?;
        self.apply_target_translation(
            current,
            changed,
            steps,
            target_basis,
            RewriteConfig::lowering(),
        )?;
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_OUTPUT,
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
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_INPUT,
            &Canonicalizer::production(),
        )?;
        self.apply_definition_decomposition(current, changed, steps)?;
        apply_rewrite(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_PRE_DECOMPOSITION,
            enhanced_rewrite_config(RewriteConfig::production()),
        )?;
        self.apply_unitary_decomposition(current, changed, steps)?;
        self.apply_mc_gate_decomposition(current, changed, steps)?;
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_AFTER_DECOMPOSITION,
            &Canonicalizer::production(),
        )?;
        apply_rewrite(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_POST_DECOMPOSITION,
            enhanced_rewrite_config(RewriteConfig::production()),
        )?;
        self.apply_target_translation(
            current,
            changed,
            steps,
            target_basis,
            enhanced_rewrite_config(RewriteConfig::lowering()),
        )?;
        let cleanup_rewrite_config = match target_basis.as_deref() {
            Some(target_basis) => enhanced_rewrite_config(RewriteConfig::production())
                .with_target_instructions(target_basis.to_vec())?,
            None => enhanced_rewrite_config(RewriteConfig::production()),
        };
        apply_rewrite(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_TARGET_CLEANUP,
            cleanup_rewrite_config,
        )?;
        apply_transformer(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_OUTPUT,
            &Canonicalizer::production(),
        )?;
        Ok(())
    }

    fn apply_definition_decomposition(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        if !contains_circuit_backed_definition(current.operations()) {
            record_unchanged_step(steps, STEP_DECOMPOSE_DEFINITIONS);
            return Ok(());
        }

        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_DECOMPOSE_DEFINITIONS,
            |circuit| {
                Ok(TransformResult {
                    circuit: expand_definitions(circuit)?,
                    changed: true,
                })
            },
        )
    }

    fn apply_unitary_decomposition(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        if !contains_unitary_gate(current.operations()) {
            record_unchanged_step(steps, STEP_DECOMPOSE_UNITARY);
            return Ok(());
        }

        apply_circuit_transform(current, changed, steps, STEP_DECOMPOSE_UNITARY, |circuit| {
            Ok(TransformResult {
                circuit: decompose_unitaries(circuit, UnitaryDecomposeConfig::default())?,
                changed: true,
            })
        })
    }

    fn apply_mc_gate_decomposition(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        let config = self.mc_gate_decompose_config();
        if !contains_mc_gate(current.operations()) {
            record_unchanged_step(steps, STEP_DECOMPOSE_MC_GATES);
            return Ok(());
        }

        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_DECOMPOSE_MC_GATES,
            |circuit| decompose_mc_gates(circuit, config),
        )
    }

    fn mc_gate_decompose_config(&self) -> McGateDecomposeConfig {
        McGateDecomposeConfig {
            resource_policy: self.config.resource_policy,
            resource_limits: self.resource_limits(),
        }
    }

    fn resource_limits(&self) -> ResourceLimits {
        ResourceLimits {
            max_total_qubits: self
                .config
                .device
                .as_ref()
                .map(|device| device.num_usable_qubits()),
        }
    }

    fn validate_resources(
        &self,
        circuit: &Circuit,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        let resource_limits = self.resource_limits();
        validate_logical_width(circuit, resource_limits)?;
        steps.push(WorkflowStepReport {
            stage: STEP_VALIDATE_RESOURCES.stage,
            name: STEP_VALIDATE_RESOURCES.name,
            changed: false,
            skipped: false,
            reason: resource_limits
                .max_total_qubits
                .map(|capacity| format!("target capacity permits {capacity} total logical qubits")),
        });
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
                stage: STEP_TRANSLATE_TARGET_BASIS.stage,
                name: STEP_TRANSLATE_TARGET_BASIS.name,
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
            STEP_TRANSLATE_TARGET_BASIS,
            config.with_target_instructions(target_basis.to_vec())?,
        )
    }

    fn resolve_target_basis(&self) -> Result<Option<Vec<Instruction>>, CompilerError> {
        if let Some(target_basis) = &self.config.target_basis {
            validate_workflow_target_basis_config(&target_basis)?;
            return Ok(Some(target_basis.to_vec()));
        }

        let Some(device) = &self.config.device else {
            return Ok(None);
        };
        let target_basis = device.native_gates();
        if target_basis.is_empty() {
            return Ok(None);
        }

        validate_workflow_target_basis_config(target_basis)?;
        Ok(Some(target_basis.to_vec()))
    }

    fn record_pre_init(
        &self,
        steps: &mut Vec<WorkflowStepReport>,
        target_basis: Option<&Vec<Instruction>>,
    ) {
        let reason = if let Some(basis) = target_basis {
            if self.config.target_basis.is_some() {
                Some(format!(
                    "resolved explicit target basis with {} instructions",
                    basis.len()
                ))
            } else {
                Some(format!(
                    "resolved device native target basis with {} instructions",
                    basis.len()
                ))
            }
        } else if self.config.device.is_some() {
            Some("target device has no native gates; basis lowering disabled".to_string())
        } else {
            Some("no target constraints configured".to_string())
        };

        steps.push(WorkflowStepReport {
            stage: STEP_RESOLVE_TARGET.stage,
            name: STEP_RESOLVE_TARGET.name,
            changed: false,
            skipped: false,
            reason,
        });
    }
}

fn record_unchanged_step(steps: &mut Vec<WorkflowStepReport>, step: WorkflowStep) {
    steps.push(WorkflowStepReport {
        stage: step.stage,
        name: step.name,
        changed: false,
        skipped: false,
        reason: None,
    });
}

fn apply_transformer(
    current: &mut Circuit,
    workflow_changed: &mut bool,
    steps: &mut Vec<WorkflowStepReport>,
    step: WorkflowStep,
    transform: &dyn Transformer,
) -> Result<(), CompilerError> {
    apply_circuit_transform(current, workflow_changed, steps, step, |circuit| {
        transform.transform(circuit)
    })
}

fn apply_circuit_transform(
    current: &mut Circuit,
    workflow_changed: &mut bool,
    steps: &mut Vec<WorkflowStepReport>,
    step: WorkflowStep,
    transform: impl FnOnce(&Circuit) -> Result<TransformResult, CompilerError>,
) -> Result<(), CompilerError> {
    let TransformResult { circuit, changed } = transform(current)?;
    *current = circuit;
    *workflow_changed |= changed;
    steps.push(WorkflowStepReport {
        stage: step.stage,
        name: step.name,
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
    step: WorkflowStep,
    config: RewriteConfig,
) -> Result<(), CompilerError> {
    apply_transformer(
        current,
        workflow_changed,
        steps,
        step,
        &KnowledgeRewriter::new(config),
    )
}

fn enhanced_rewrite_config(config: RewriteConfig) -> RewriteConfig {
    config
        .with_max_rounds(16)
        .with_max_window_ops(32)
        .with_max_pattern_len(12)
}

fn contains_circuit_backed_definition(operations: &[Operation]) -> bool {
    operations
        .iter()
        .any(|operation| match &operation.instruction {
            Instruction::CircuitGate(_) => true,
            Instruction::UnitaryGate(gate) => gate.circuit().is_some(),
            Instruction::ControlFlowGate(flow) => contains_circuit_backed_definition_in_flow(flow),
            _ => false,
        })
}

fn contains_circuit_backed_definition_in_flow(flow: &ControlFlow) -> bool {
    match flow {
        ControlFlow::IfElse(gate) => {
            contains_circuit_backed_definition(gate.true_body())
                || gate
                    .false_body()
                    .is_some_and(contains_circuit_backed_definition)
        }
        ControlFlow::WhileLoop(gate) => contains_circuit_backed_definition(gate.body()),
    }
}

fn contains_unitary_gate(operations: &[Operation]) -> bool {
    operations
        .iter()
        .any(|operation| match &operation.instruction {
            Instruction::UnitaryGate(_) => true,
            Instruction::ControlFlowGate(flow) => contains_unitary_gate_in_flow(flow),
            _ => false,
        })
}

fn contains_unitary_gate_in_flow(flow: &ControlFlow) -> bool {
    match flow {
        ControlFlow::IfElse(gate) => {
            contains_unitary_gate(gate.true_body())
                || gate.false_body().is_some_and(contains_unitary_gate)
        }
        ControlFlow::WhileLoop(gate) => contains_unitary_gate(gate.body()),
    }
}

fn contains_mc_gate(operations: &[Operation]) -> bool {
    operations
        .iter()
        .any(|operation| match &operation.instruction {
            Instruction::McGate(_) => true,
            Instruction::ControlFlowGate(flow) => contains_mc_gate_in_flow(flow),
            _ => false,
        })
}

fn contains_mc_gate_in_flow(flow: &ControlFlow) -> bool {
    match flow {
        ControlFlow::IfElse(gate) => {
            contains_mc_gate(gate.true_body()) || gate.false_body().is_some_and(contains_mc_gate)
        }
        ControlFlow::WhileLoop(gate) => contains_mc_gate(gate.body()),
    }
}

fn validate_logical_width(
    circuit: &Circuit,
    resource_limits: ResourceLimits,
) -> Result<(), CompilerError> {
    if let Some(max_total_qubits) = resource_limits.max_total_qubits {
        if circuit.qubits().len() > max_total_qubits {
            return Err(CompilerError::InvalidInput(format!(
                "source circuit uses {} logical qubits but target capacity is {max_total_qubits}",
                circuit.qubits().len()
            )));
        }
    }
    Ok(())
}

fn validate_workflow_target_basis_config(
    target_basis: &[Instruction],
) -> Result<(), CompilerError> {
    if target_basis.is_empty() {
        return Err(CompilerError::InvalidInput(
            "workflow target basis must not be empty".to_string(),
        ));
    }

    // Rewrite lowering can represent `McGate` as a target instruction, but the
    // current workflow decomposes all multi-controlled gates before target-basis
    // translation. Native multi-controlled target support therefore needs an
    // explicit workflow policy before it can be accepted here.
    for instruction in target_basis {
        if !matches!(instruction, Instruction::Standard(_)) {
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
