// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
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
//!
//! The normal workflow follows the stable pass order:
//! canonicalize input, expand circuit-backed definitions, apply production
//! knowledge rewrite, decompose unitary and multi-controlled gates,
//! canonicalize again, optimize the decomposed circuit, optionally route on a
//! device, optionally translate to the resolved target basis, and canonicalize
//! the output.
//!
//! The enhanced workflow uses the same required correctness stages but raises
//! rewrite budgets, uses stronger SABRE trial settings, performs a
//! post-routing cleanup pass, and adds a target-aware cleanup pass after
//! target-basis translation. This keeps `Normal` suitable for predictable
//! production compilation while giving `Enhanced` more chances to recover
//! simplifications exposed by decomposition, routing, and lowering.
//!
//! Stages are deliberately ordered around compiler invariants. Early
//! canonicalization gives later passes a stable representation, definition and
//! high-level gate decomposition remove operations that routing cannot accept,
//! routing runs before final target-basis cleanup because it may insert SWAPs,
//! and the output canonicalizer removes representation noise introduced by
//! previous stages.

use crate::circuit::{Circuit, ControlFlow, Instruction};
use crate::compile::CompilerError;
use crate::compile::resource::ResourceLimits;
use crate::compile::sabre::{SabreConfig, SabreHeuristicConfig, SabreTrialObjective};
use crate::compile::transform::decompose::expand_definitions;
use crate::compile::transform::decompose::mc_gate::{McGateDecomposeConfig, decompose_mc_gates};
use crate::compile::transform::decompose::unitary::decompose::{
    UnitaryDecomposeConfig, decompose_unitaries,
};
use crate::compile::transform::{
    Canonicalizer, KnowledgeRewriter, LayoutObjective, RewriteConfig, TransformResult, Transformer,
    build_physical_layout_graph, route_sabre,
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
const STEP_ROUTE_SABRE: WorkflowStep = WorkflowStep {
    stage: "routing",
    name: "route.sabre",
};
const STEP_OPTIMIZE_POST_ROUTING: WorkflowStep = WorkflowStep {
    stage: "optimization",
    name: "optimize.post_routing",
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
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_INPUT,
            |circuit| Canonicalizer::production().transform(circuit),
        )?;
        self.apply_definition_decomposition(current, changed, steps)?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_PRE_DECOMPOSITION,
            |circuit| KnowledgeRewriter::new(RewriteConfig::production()).transform(circuit),
        )?;
        self.apply_unitary_decomposition(current, changed, steps)?;
        self.apply_mc_gate_decomposition(current, changed, steps)?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_AFTER_DECOMPOSITION,
            |circuit| Canonicalizer::production().transform(circuit),
        )?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_POST_DECOMPOSITION,
            |circuit| KnowledgeRewriter::new(RewriteConfig::production()).transform(circuit),
        )?;
        self.apply_layout_and_routing(current, changed, steps, CompileMode::Normal)?;
        self.apply_target_translation(
            current,
            changed,
            steps,
            target_basis,
            RewriteConfig::lowering(),
        )?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_OUTPUT,
            |circuit| Canonicalizer::production().transform(circuit),
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
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_INPUT,
            |circuit| Canonicalizer::production().transform(circuit),
        )?;
        self.apply_definition_decomposition(current, changed, steps)?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_PRE_DECOMPOSITION,
            |circuit| {
                KnowledgeRewriter::new(
                    RewriteConfig::production()
                        .with_max_rounds(16)
                        .with_max_window_ops(32)
                        .with_max_pattern_len(12),
                )
                .transform(circuit)
            },
        )?;
        self.apply_unitary_decomposition(current, changed, steps)?;
        self.apply_mc_gate_decomposition(current, changed, steps)?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_AFTER_DECOMPOSITION,
            |circuit| Canonicalizer::production().transform(circuit),
        )?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_POST_DECOMPOSITION,
            |circuit| {
                KnowledgeRewriter::new(
                    RewriteConfig::production()
                        .with_max_rounds(16)
                        .with_max_window_ops(32)
                        .with_max_pattern_len(12),
                )
                .transform(circuit)
            },
        )?;
        self.apply_layout_and_routing(current, changed, steps, CompileMode::Enhanced)?;
        self.apply_post_routing_cleanup(current, changed, steps)?;
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
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_TARGET_CLEANUP,
            |circuit| KnowledgeRewriter::new(cleanup_rewrite_config).transform(circuit),
        )?;
        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_CANONICALIZE_OUTPUT,
            |circuit| Canonicalizer::production().transform(circuit),
        )?;
        Ok(())
    }

    fn apply_definition_decomposition(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        let mut operation_stack = vec![current.operations()];
        let mut has_circuit_backed_definition = false;
        while let Some(operations) = operation_stack.pop() {
            if operations
                .iter()
                .any(|operation| match &operation.instruction {
                    Instruction::CircuitGate(_) => true,
                    Instruction::UnitaryGate(gate) => gate.circuit().is_some(),
                    Instruction::ControlFlowGate(flow) => {
                        match flow {
                            ControlFlow::IfElse(gate) => {
                                operation_stack.push(gate.true_body());
                                if let Some(false_body) = gate.false_body() {
                                    operation_stack.push(false_body);
                                }
                            }
                            ControlFlow::WhileLoop(gate) => operation_stack.push(gate.body()),
                        }
                        false
                    }
                    _ => false,
                })
            {
                has_circuit_backed_definition = true;
                break;
            }
        }

        if !has_circuit_backed_definition {
            steps.push(WorkflowStepReport {
                stage: STEP_DECOMPOSE_DEFINITIONS.stage,
                name: STEP_DECOMPOSE_DEFINITIONS.name,
                changed: false,
                skipped: false,
                reason: None,
            });
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
        let mut operation_stack = vec![current.operations()];
        let mut has_unitary_gate = false;
        while let Some(operations) = operation_stack.pop() {
            if operations
                .iter()
                .any(|operation| match &operation.instruction {
                    Instruction::UnitaryGate(_) => true,
                    Instruction::ControlFlowGate(flow) => {
                        match flow {
                            ControlFlow::IfElse(gate) => {
                                operation_stack.push(gate.true_body());
                                if let Some(false_body) = gate.false_body() {
                                    operation_stack.push(false_body);
                                }
                            }
                            ControlFlow::WhileLoop(gate) => operation_stack.push(gate.body()),
                        }
                        false
                    }
                    _ => false,
                })
            {
                has_unitary_gate = true;
                break;
            }
        }

        if !has_unitary_gate {
            steps.push(WorkflowStepReport {
                stage: STEP_DECOMPOSE_UNITARY.stage,
                name: STEP_DECOMPOSE_UNITARY.name,
                changed: false,
                skipped: false,
                reason: None,
            });
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
        let mut operation_stack = vec![current.operations()];
        let mut has_mc_gate = false;
        while let Some(operations) = operation_stack.pop() {
            if operations
                .iter()
                .any(|operation| match &operation.instruction {
                    Instruction::McGate(_) => true,
                    Instruction::ControlFlowGate(flow) => {
                        match flow {
                            ControlFlow::IfElse(gate) => {
                                operation_stack.push(gate.true_body());
                                if let Some(false_body) = gate.false_body() {
                                    operation_stack.push(false_body);
                                }
                            }
                            ControlFlow::WhileLoop(gate) => operation_stack.push(gate.body()),
                        }
                        false
                    }
                    _ => false,
                })
            {
                has_mc_gate = true;
                break;
            }
        }

        if !has_mc_gate {
            steps.push(WorkflowStepReport {
                stage: STEP_DECOMPOSE_MC_GATES.stage,
                name: STEP_DECOMPOSE_MC_GATES.name,
                changed: false,
                skipped: false,
                reason: None,
            });
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
        if let Some(max_total_qubits) = resource_limits.max_total_qubits {
            if circuit.qubits().len() > max_total_qubits {
                return Err(CompilerError::InvalidInput(format!(
                    "source circuit uses {} logical qubits but target capacity is {max_total_qubits}",
                    circuit.qubits().len()
                )));
            }
        }
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

    fn apply_layout_and_routing(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
        mode: CompileMode,
    ) -> Result<(), CompilerError> {
        let Some(device) = self.config.device.as_ref() else {
            steps.push(WorkflowStepReport {
                stage: STEP_ROUTE_SABRE.stage,
                name: STEP_ROUTE_SABRE.name,
                changed: false,
                skipped: true,
                reason: Some("no target device configured".to_string()),
            });
            return Ok(());
        };

        let physical = build_physical_layout_graph(device)?;
        let objective = match mode {
            CompileMode::Normal => LayoutObjective::auto_from_physical(&physical),
            CompileMode::Enhanced => {
                if physical.has_fidelity_data() {
                    LayoutObjective::fidelity_required(&physical)?
                } else {
                    LayoutObjective::topology_only()
                }
            }
        };
        let config = sabre_config_for_mode(mode, self.config.seed);
        let routed = route_sabre(current, device, &objective, &config)?;
        let route_changed = routed.changed;
        *current = routed.circuit;
        *changed |= route_changed;

        steps.push(WorkflowStepReport {
            stage: STEP_ROUTE_SABRE.stage,
            name: STEP_ROUTE_SABRE.name,
            changed: route_changed,
            skipped: false,
            reason: Some(format!(
                "inserted {} swap operations using {} routing trials",
                routed.swap_count, routed.diagnostics.trials_evaluated
            )),
        });
        Ok(())
    }

    fn apply_post_routing_cleanup(
        &self,
        current: &mut Circuit,
        changed: &mut bool,
        steps: &mut Vec<WorkflowStepReport>,
    ) -> Result<(), CompilerError> {
        if self.config.device.is_none() {
            steps.push(WorkflowStepReport {
                stage: STEP_OPTIMIZE_POST_ROUTING.stage,
                name: STEP_OPTIMIZE_POST_ROUTING.name,
                changed: false,
                skipped: true,
                reason: Some("routing was skipped".to_string()),
            });
            return Ok(());
        }

        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_OPTIMIZE_POST_ROUTING,
            |circuit| {
                KnowledgeRewriter::new(
                    RewriteConfig::production()
                        .with_max_rounds(16)
                        .with_max_window_ops(32)
                        .with_max_pattern_len(12),
                )
                .transform(circuit)
            },
        )
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

        apply_circuit_transform(
            current,
            changed,
            steps,
            STEP_TRANSLATE_TARGET_BASIS,
            |circuit| {
                KnowledgeRewriter::new(config.with_target_instructions(target_basis.to_vec())?)
                    .transform(circuit)
            },
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

fn sabre_config_for_mode(mode: CompileMode, seed: Option<u32>) -> SabreConfig {
    let mut config = SabreConfig {
        seed: seed.map(u64::from),
        ..SabreConfig::default()
    };

    if mode == CompileMode::Enhanced {
        config.layout_trials = 24;
        config.refinement_iterations = 2;
        config.layout_scoring_trials = 3;
        config.routing_trials = 12;
        config.trial_objective = SabreTrialObjective::SwapThenDepth;
        config.heuristic = SabreHeuristicConfig {
            lookahead_weights: vec![0.5, 0.25],
            ..SabreHeuristicConfig::default()
        };
    }

    config
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
