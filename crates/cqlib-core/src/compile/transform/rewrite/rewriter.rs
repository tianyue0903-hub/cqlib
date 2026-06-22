// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Transformer entry point and circuit rebuild logic for knowledge rewrite.

use crate::circuit::operation::ValueOperation;
use crate::circuit::{
    Circuit, CircuitParam, ClassicalControlOp, Instruction, Operation, Parameter, ParameterValue,
    Qubit, StandardGate, ValueClassicalControlOp, ValueControlBody, ValueInstruction,
    ValueSwitchCase,
};
use crate::compile::error::CompilerError;
use crate::compile::knowledge::library::RuleLibrary;
use crate::compile::knowledge::matcher::KnowledgeInstructionKey as RewriteInstructionKey;
use crate::compile::transform::rewrite::basis::{TargetContext, validate_final_target};
use crate::compile::transform::rewrite::config::RewriteConfig;
use crate::compile::transform::rewrite::matcher::{
    CompiledRuleSet, ReplacementItem, RewritePatch, resolve_operation_param,
    select_rewrites_in_context,
};

use crate::compile::transform::rebuild::{CircuitRebuildContext, ClassicalRemap};
use crate::compile::transform::{TransformResult, Transformer};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

static BUILTIN_COMPILED_RULES: OnceLock<Result<Arc<CompiledRuleSet>, String>> = OnceLock::new();

/// Aggregate statistics produced by one knowledge rewrite run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnowledgeRewriteStats {
    /// Number of fixpoint rounds actually executed.
    pub rounds_executed: u8,
    /// Number of selected rule patches emitted into rebuilt sequences.
    pub rules_applied: usize,
    /// Number of operation sequences whose selected patch set was non-empty.
    pub changed_sequences: usize,
    /// Whether the run observed a stable round before hitting `max_rounds`.
    pub reached_fixpoint: bool,
}

impl KnowledgeRewriteStats {
    fn merge_round(&mut self, other: &RoundStats) {
        self.rules_applied += other.rules_applied;
        self.changed_sequences += other.changed_sequences;
    }
}

/// Public result for running the rewriter directly.
#[derive(Debug, Clone)]
pub struct KnowledgeRewriteResult {
    pub circuit: Circuit,
    pub changed: bool,
    pub stats: KnowledgeRewriteStats,
}

#[derive(Debug, Clone, Default)]
struct RoundStats {
    rules_applied: usize,
    changed_sequences: usize,
    representation_changes: usize,
}

impl RoundStats {
    fn changed(&self) -> bool {
        self.rules_applied > 0 || self.representation_changes > 0
    }
}

/// Transformer that optimizes circuits using the compiler knowledge base.
#[derive(Debug, Clone)]
pub struct KnowledgeRewriter {
    config: RewriteConfig,
}

impl KnowledgeRewriter {
    /// Creates a knowledge rewriter with the supplied configuration.
    pub fn new(config: RewriteConfig) -> Self {
        Self { config }
    }

    /// Creates a knowledge rewriter using conservative production defaults.
    pub fn production() -> Self {
        Self::new(RewriteConfig::production())
    }

    /// Creates a knowledge rewriter using explicit lowering defaults.
    pub fn lowering() -> Self {
        Self::new(RewriteConfig::lowering())
    }

    pub const fn config(&self) -> &RewriteConfig {
        &self.config
    }

    /// Runs knowledge-based local rewrite to a fixpoint or round limit.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::{Circuit, Qubit};
    /// use cqlib_core::compile::transform::KnowledgeRewriter;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.x(Qubit::new(0)).unwrap();
    /// circuit.x(Qubit::new(0)).unwrap();
    ///
    /// let result = KnowledgeRewriter::production().run(&circuit).unwrap();
    /// assert!(result.stats.rounds_executed >= 1);
    /// assert!(result.stats.reached_fixpoint);
    /// let _rewritten = result.circuit;
    /// ```
    pub fn run(&self, circuit: &Circuit) -> Result<KnowledgeRewriteResult, CompilerError> {
        if self.config.max_rounds() == 0 {
            return Err(CompilerError::InvalidInput(
                "rewrite max_rounds must be greater than zero".to_string(),
            ));
        }

        let rules = builtin_compiled_rules()?;
        let target_context = TargetContext::from_config(&self.config, rules.as_ref())?;

        let mut current = circuit.clone();
        let mut aggregate = KnowledgeRewriteStats::default();
        let mut changed = false;

        for round in 1..=self.config.max_rounds() {
            aggregate.rounds_executed = round;
            let (next, round_stats) = RoundRewriter::run(
                &current,
                rules.as_ref(),
                &self.config,
                target_context.as_ref(),
            )?;
            if !round_stats.changed() {
                aggregate.reached_fixpoint = true;
                break;
            }

            changed = true;
            aggregate.merge_round(&round_stats);
            current = next;
        }

        validate_final_target(&current, &self.config)?;

        Ok(KnowledgeRewriteResult {
            circuit: current,
            changed,
            stats: aggregate,
        })
    }
}

fn builtin_compiled_rules() -> Result<Arc<CompiledRuleSet>, CompilerError> {
    match BUILTIN_COMPILED_RULES.get_or_init(|| {
        let library = RuleLibrary::builtin_rules().map_err(|err| err.to_string())?;
        CompiledRuleSet::from_library(library)
            .map(Arc::new)
            .map_err(|err| err.to_string())
    }) {
        Ok(rules) => Ok(Arc::clone(rules)),
        Err(message) => Err(CompilerError::InvariantViolation(message.clone())),
    }
}

// Transformer integration exposes only the generic transform result; direct
// callers should use `KnowledgeRewriter::run` when rewrite statistics matter.
impl Transformer for KnowledgeRewriter {
    fn name(&self) -> &'static str {
        "knowledge_rewrite"
    }

    fn transform(
        &self,
        circuit: &Circuit,
        _analysis: Option<&crate::compile::transform::CircuitAnalysis>,
    ) -> Result<TransformResult, CompilerError> {
        let result = self.run(circuit)?;
        Ok(TransformResult {
            circuit: result.circuit,
            changed: result.changed,
        })
    }
}

/// Rewrites a circuit with the supplied configuration.
pub fn rewrite_circuit(
    circuit: &Circuit,
    config: RewriteConfig,
) -> Result<KnowledgeRewriteResult, CompilerError> {
    KnowledgeRewriter::new(config).run(circuit)
}

enum SequenceTarget<'a> {
    TopLevel {
        output: &'a mut Vec<ValueOperation>,
        phase_delta: &'a mut Parameter,
    },
    ControlFlowBody {
        output: &'a mut Vec<ValueOperation>,
        phase_delta: &'a mut Parameter,
    },
}

struct RoundRewriter<'a> {
    source: &'a Circuit,
    rules: &'a CompiledRuleSet,
    config: &'a RewriteConfig,
    target_context: Option<&'a TargetContext>,
    rebuild: CircuitRebuildContext,
    stats: RoundStats,
}

impl<'a> RoundRewriter<'a> {
    fn run(
        source: &'a Circuit,
        rules: &'a CompiledRuleSet,
        config: &'a RewriteConfig,
        target_context: Option<&'a TargetContext>,
    ) -> Result<(Circuit, RoundStats), CompilerError> {
        let mut rewriter = Self {
            source,
            rules,
            config,
            target_context,
            rebuild: CircuitRebuildContext::new(source),
            stats: RoundStats::default(),
        };
        let mut phase_delta = Parameter::from(0.0);
        let root_classical = rewriter.rebuild.root_classical().clone();
        let mut operations = Vec::with_capacity(source.operations().len());

        rewriter.apply_sequence(
            source.operations(),
            &root_classical,
            SequenceTarget::TopLevel {
                output: &mut operations,
                phase_delta: &mut phase_delta,
            },
        )?;
        let global_phase = &source.global_phase() + &phase_delta;
        let circuit = rewriter
            .rebuild
            .finish(source.qubits(), operations, global_phase)?;

        Ok((circuit, rewriter.stats))
    }

    fn apply_sequence(
        &mut self,
        operations: &[Operation],
        classical_remap: &ClassicalRemap,
        mut target: SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        let mut cursor = 0;
        while cursor < operations.len() {
            if RewriteInstructionKey::from_instruction(&operations[cursor].instruction).is_none() {
                self.emit_original_operation(&operations[cursor], classical_remap, &mut target)?;
                cursor += 1;
                continue;
            }

            let block_start = cursor;
            while cursor < operations.len()
                && RewriteInstructionKey::from_instruction(&operations[cursor].instruction)
                    .is_some()
            {
                cursor += 1;
            }

            let block = &operations[block_start..cursor];
            let patches = select_rewrites_in_context(
                self.source,
                block,
                self.rules,
                self.config,
                self.target_context,
            )?;
            if patches.is_empty() {
                for operation in block {
                    self.emit_original_operation(operation, classical_remap, &mut target)?;
                }
            } else {
                self.stats.changed_sequences += 1;
                self.emit_rewritten_block(block, patches, classical_remap, &mut target)?;
            }
        }

        Ok(())
    }

    fn emit_rewritten_block(
        &mut self,
        block: &[Operation],
        patches: Vec<RewritePatch>,
        classical_remap: &ClassicalRemap,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        let mut patches_by_start = HashMap::new();
        let mut skipped_positions = vec![false; block.len()];
        for patch in patches {
            for &position in &patch.matched_positions {
                let Some(skipped) = skipped_positions.get_mut(position) else {
                    return Err(CompilerError::InvariantViolation(format!(
                        "rewrite patch matched position {position} outside block of length {}",
                        block.len()
                    )));
                };
                *skipped = true;
            }
            patches_by_start.insert(patch.first_position, patch);
        }

        for (position, operation) in block.iter().enumerate() {
            if let Some(patch) = patches_by_start.remove(&position) {
                self.stats.rules_applied += 1;
                for replacement in &patch.replacements {
                    self.emit_replacement(replacement, target)?;
                }
            }

            if skipped_positions[position] {
                continue;
            }

            self.emit_operation(
                operation.instruction.clone(),
                operation.qubits.clone(),
                operation.params.as_slice(),
                operation.label.clone(),
                classical_remap,
                target,
            )?;
        }

        Ok(())
    }

    fn emit_original_operation(
        &mut self,
        operation: &Operation,
        classical_remap: &ClassicalRemap,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if !self.config.recurses_control_flow() {
            return self.emit_preserved_operation(operation, classical_remap, target);
        }

        if let Instruction::ClassicalControl(control) = &operation.instruction {
            let instruction = self.rewrite_control_flow(control, classical_remap)?;
            let qubits = instruction.used_qubits().into_iter().collect();
            return self.emit_value_operation(
                ValueInstruction::ClassicalControl(instruction),
                qubits,
                CircuitRebuildContext::resolve_source_params(
                    self.source,
                    operation.params.as_slice(),
                )?,
                operation.label.clone(),
                target,
            );
        }

        self.emit_preserved_operation(operation, classical_remap, target)
    }

    fn rewrite_control_flow(
        &mut self,
        control: &ClassicalControlOp,
        classical_remap: &ClassicalRemap,
    ) -> Result<ValueClassicalControlOp, CompilerError> {
        let cc = match control {
            ClassicalControlOp::If(op) => {
                let mut then_body = Vec::with_capacity(op.then_body().operations().len());
                let mut then_phase = Parameter::from(0.0);
                self.apply_sequence(
                    op.then_body().operations(),
                    classical_remap,
                    SequenceTarget::ControlFlowBody {
                        output: &mut then_body,
                        phase_delta: &mut then_phase,
                    },
                )?;
                self.prepend_body_phase(&mut then_body, then_phase);

                let else_body = op
                    .else_body()
                    .map(|body| {
                        let mut rewritten = Vec::with_capacity(body.operations().len());
                        let mut body_phase = Parameter::from(0.0);
                        self.apply_sequence(
                            body.operations(),
                            classical_remap,
                            SequenceTarget::ControlFlowBody {
                                output: &mut rewritten,
                                phase_delta: &mut body_phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut rewritten, body_phase);
                        Ok::<_, CompilerError>(rewritten)
                    })
                    .transpose()?;

                ValueClassicalControlOp::If {
                    condition: classical_remap.remap_expr(op.condition())?,
                    then_body: ValueControlBody::new(then_body),
                    else_body: else_body.map(ValueControlBody::new),
                }
            }
            ClassicalControlOp::While(op) => {
                let mut body = Vec::with_capacity(op.body().operations().len());
                let mut body_phase = Parameter::from(0.0);
                self.apply_sequence(
                    op.body().operations(),
                    classical_remap,
                    SequenceTarget::ControlFlowBody {
                        output: &mut body,
                        phase_delta: &mut body_phase,
                    },
                )?;
                self.prepend_body_phase(&mut body, body_phase);

                ValueClassicalControlOp::While {
                    condition: classical_remap.remap_expr(op.condition())?,
                    body: ValueControlBody::new(body),
                }
            }
            ClassicalControlOp::For(op) => {
                let mut body = Vec::with_capacity(op.body().operations().len());
                let mut body_phase = Parameter::from(0.0);
                self.apply_sequence(
                    op.body().operations(),
                    classical_remap,
                    SequenceTarget::ControlFlowBody {
                        output: &mut body,
                        phase_delta: &mut body_phase,
                    },
                )?;
                self.prepend_body_phase(&mut body, body_phase);

                ValueClassicalControlOp::For {
                    var: classical_remap.remap_var(op.var())?,
                    start: classical_remap.remap_expr(op.start())?,
                    stop: classical_remap.remap_expr(op.stop())?,
                    step: classical_remap.remap_expr(op.step())?,
                    body: ValueControlBody::new(body),
                }
            }
            ClassicalControlOp::Switch(op) => {
                let cases = op
                    .cases()
                    .iter()
                    .map(|case| {
                        let mut rewritten = Vec::with_capacity(case.body().operations().len());
                        let mut body_phase = Parameter::from(0.0);
                        self.apply_sequence(
                            case.body().operations(),
                            classical_remap,
                            SequenceTarget::ControlFlowBody {
                                output: &mut rewritten,
                                phase_delta: &mut body_phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut rewritten, body_phase);
                        Ok::<_, CompilerError>(ValueSwitchCase::new(
                            case.value(),
                            ValueControlBody::new(rewritten),
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let default = op
                    .default()
                    .map(|body| {
                        let mut rewritten = Vec::with_capacity(body.operations().len());
                        let mut body_phase = Parameter::from(0.0);
                        self.apply_sequence(
                            body.operations(),
                            classical_remap,
                            SequenceTarget::ControlFlowBody {
                                output: &mut rewritten,
                                phase_delta: &mut body_phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut rewritten, body_phase);
                        Ok::<_, CompilerError>(ValueControlBody::new(rewritten))
                    })
                    .transpose()?;

                ValueClassicalControlOp::Switch {
                    target: classical_remap.remap_expr(op.target())?,
                    cases,
                    default,
                }
            }
            ClassicalControlOp::Break => ValueClassicalControlOp::Break,
            ClassicalControlOp::Continue => ValueClassicalControlOp::Continue,
        };

        Ok(cc)
    }

    fn emit_operation(
        &mut self,
        instruction: Instruction,
        qubits: SmallVec<[Qubit; 3]>,
        params: &[CircuitParam],
        label: Option<Box<str>>,
        classical_remap: &ClassicalRemap,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if Self::is_gphase_instruction(&instruction) {
            if matches!(target, SequenceTarget::TopLevel { .. }) {
                self.stats.representation_changes += 1;
            }
            Self::accumulate_phase(target, self.source_gphase_param(params)?);
            return Ok(());
        }

        let instruction = self
            .rebuild
            .remap_non_control_instruction(&instruction, classical_remap)?;
        let params = CircuitRebuildContext::resolve_source_params(self.source, params)?;
        self.emit_value_operation(instruction, qubits, params, label, target)
    }

    fn emit_preserved_operation(
        &mut self,
        operation: &Operation,
        classical_remap: &ClassicalRemap,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if Self::is_gphase_instruction(&operation.instruction) {
            if matches!(target, SequenceTarget::TopLevel { .. }) {
                self.stats.representation_changes += 1;
            }
            Self::accumulate_phase(target, self.source_gphase_param(&operation.params)?);
            return Ok(());
        }

        let operation =
            self.rebuild
                .remap_preserved_operation(self.source, operation, classical_remap)?;
        Self::push_value_operation(target, operation);
        Ok(())
    }

    fn emit_value_operation(
        &mut self,
        instruction: ValueInstruction,
        qubits: SmallVec<[Qubit; 3]>,
        params: SmallVec<[ParameterValue; 1]>,
        label: Option<Box<str>>,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        Self::push_value_operation(
            target,
            ValueOperation {
                instruction,
                qubits,
                params,
                label,
            },
        );
        Ok(())
    }

    fn emit_replacement(
        &mut self,
        replacement: &ReplacementItem,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if Self::is_gphase_instruction(&replacement.instruction) {
            Self::accumulate_phase(target, Self::replacement_gphase_param(replacement)?);
            return Ok(());
        }

        let params = replacement.params.iter().cloned().collect();
        self.emit_value_operation(
            ValueInstruction::from_instruction(replacement.instruction.clone()),
            replacement.qubits.clone(),
            params,
            None,
            target,
        )
    }

    fn prepend_body_phase(&mut self, body: &mut Vec<ValueOperation>, phase: Parameter) {
        if phase.is_zero() {
            return;
        }

        body.insert(
            0,
            ValueOperation {
                instruction: ValueInstruction::from_instruction(Instruction::Standard(
                    StandardGate::GPhase,
                )),
                qubits: SmallVec::new(),
                params: smallvec::smallvec![ParameterValue::from(phase)],
                label: None,
            },
        );
    }

    fn source_gphase_param(&self, params: &[CircuitParam]) -> Result<Parameter, CompilerError> {
        if params.len() != 1 {
            return Err(CompilerError::InvariantViolation(
                "GPhase operation must contain one parameter".to_string(),
            ));
        }
        resolve_operation_param(self.source, &params[0])
    }

    fn replacement_gphase_param(replacement: &ReplacementItem) -> Result<Parameter, CompilerError> {
        let phase = replacement.params.first().ok_or_else(|| {
            CompilerError::InvariantViolation(
                "GPhase replacement must contain one parameter".to_string(),
            )
        })?;
        Ok(match phase {
            ParameterValue::Fixed(value) => Parameter::from(*value),
            ParameterValue::Param(parameter) => parameter.clone(),
        })
    }

    fn push_value_operation(target: &mut SequenceTarget<'_>, operation: ValueOperation) {
        match target {
            SequenceTarget::TopLevel { output, .. }
            | SequenceTarget::ControlFlowBody { output, .. } => output.push(operation),
        }
    }

    fn accumulate_phase(target: &mut SequenceTarget<'_>, phase: Parameter) {
        match target {
            SequenceTarget::TopLevel { phase_delta, .. } => {
                **phase_delta = &**phase_delta + &phase;
            }
            SequenceTarget::ControlFlowBody { phase_delta, .. } => {
                **phase_delta = &**phase_delta + &phase;
            }
        }
    }

    fn is_gphase_instruction(instruction: &Instruction) -> bool {
        matches!(instruction, Instruction::Standard(StandardGate::GPhase))
    }
}
