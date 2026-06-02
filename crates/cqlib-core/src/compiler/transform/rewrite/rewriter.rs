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

//! Transformer entry point and circuit rebuild logic for knowledge rewrite.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, Parameter,
    ParameterValue, Qubit, StandardGate, WhileLoopGate,
};
use crate::compiler::error::CompilerError;
use crate::compiler::knowledge::library::RuleLibrary;
use crate::compiler::knowledge::matcher::KnowledgeInstructionKey as RewriteInstructionKey;
use crate::compiler::transform::rewrite::basis::{TargetContext, validate_final_target};
use crate::compiler::transform::rewrite::config::RewriteConfig;
use crate::compiler::transform::rewrite::matcher::{
    CompiledRuleSet, ReplacementItem, RewritePatch, resolve_operation_param,
    select_rewrites_in_context,
};

use crate::compiler::transform::{TransformResult, Transformer};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

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
    pub fn run(&self, circuit: &Circuit) -> Result<KnowledgeRewriteResult, CompilerError> {
        if self.config.max_rounds() == 0 {
            return Err(CompilerError::InvalidInput(
                "rewrite max_rounds must be greater than zero".to_string(),
            ));
        }

        let library = RuleLibrary::builtin_rules()
            .map_err(|err| CompilerError::InvariantViolation(err.to_string()))?;
        let rules = CompiledRuleSet::from_library(library)?;
        let target_context = TargetContext::from_config(&self.config, &rules)?;

        let mut current = circuit.clone();
        let mut aggregate = KnowledgeRewriteStats::default();
        let mut changed = false;

        for round in 1..=self.config.max_rounds() {
            aggregate.rounds_executed = round;
            let (next, round_stats) =
                RoundRewriter::run(&current, &rules, &self.config, target_context.as_ref())?;
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

impl Transformer for KnowledgeRewriter {
    fn transform(&self, circuit: &Circuit) -> Result<TransformResult, CompilerError> {
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
        phase_delta: &'a mut Parameter,
    },
    ControlFlowBody {
        output: &'a mut Vec<Operation>,
        phase_delta: &'a mut Parameter,
    },
}

struct RoundRewriter<'a> {
    source: &'a Circuit,
    rules: &'a CompiledRuleSet,
    config: &'a RewriteConfig,
    target_context: Option<&'a TargetContext>,
    rebuilt: Circuit,
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
            rebuilt: Circuit::from_qubits(source.qubits())?,
            stats: RoundStats::default(),
        };
        let mut phase_delta = Parameter::from(0.0);

        rewriter.apply_sequence(
            source.operations(),
            SequenceTarget::TopLevel {
                phase_delta: &mut phase_delta,
            },
        )?;
        rewriter
            .rebuilt
            .set_global_phase(&source.global_phase() + &phase_delta);

        Ok((rewriter.rebuilt, rewriter.stats))
    }

    fn apply_sequence(
        &mut self,
        operations: &[Operation],
        mut target: SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        let mut cursor = 0;
        while cursor < operations.len() {
            if RewriteInstructionKey::from_instruction(&operations[cursor].instruction).is_none() {
                self.emit_original_operation(&operations[cursor], &mut target)?;
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
                    self.emit_original_operation(operation, &mut target)?;
                }
            } else {
                self.stats.changed_sequences += 1;
                self.emit_rewritten_block(block, patches, &mut target)?;
            }
        }

        Ok(())
    }

    fn emit_rewritten_block(
        &mut self,
        block: &[Operation],
        patches: Vec<RewritePatch>,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        let mut patches_by_start = HashMap::new();
        let mut skipped_positions = HashSet::new();
        for patch in patches {
            for &position in &patch.matched_positions {
                skipped_positions.insert(position);
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

            if skipped_positions.contains(&position) {
                continue;
            }

            self.emit_operation(
                operation.instruction.clone(),
                operation.qubits.clone(),
                operation.params.as_slice(),
                operation.label.clone(),
                target,
            )?;
        }

        Ok(())
    }

    fn emit_original_operation(
        &mut self,
        operation: &Operation,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if !self.config.recurses_control_flow() {
            return self.emit_operation(
                operation.instruction.clone(),
                operation.qubits.clone(),
                operation.params.as_slice(),
                operation.label.clone(),
                target,
            );
        }

        let rewritten_instruction = match &operation.instruction {
            Instruction::ControlFlowGate(flow) => Some(self.rewrite_control_flow(flow)?),
            _ => None,
        };

        if let Some(instruction) = rewritten_instruction {
            let qubits = Self::control_flow_operation_qubits(&instruction);
            self.emit_operation(
                instruction,
                qubits,
                operation.params.as_slice(),
                operation.label.clone(),
                target,
            )
        } else {
            self.emit_operation(
                operation.instruction.clone(),
                operation.qubits.clone(),
                operation.params.as_slice(),
                operation.label.clone(),
                target,
            )
        }
    }

    fn rewrite_control_flow(&mut self, flow: &ControlFlow) -> Result<Instruction, CompilerError> {
        let flow = match flow {
            ControlFlow::IfElse(gate) => {
                let mut true_body = Vec::with_capacity(gate.true_body().len());
                let mut true_phase = Parameter::from(0.0);
                self.apply_sequence(
                    gate.true_body(),
                    SequenceTarget::ControlFlowBody {
                        output: &mut true_body,
                        phase_delta: &mut true_phase,
                    },
                )?;
                self.prepend_body_phase(&mut true_body, true_phase);

                let false_body = gate
                    .false_body()
                    .map(|body| {
                        let mut rewritten = Vec::with_capacity(body.len());
                        let mut body_phase = Parameter::from(0.0);
                        self.apply_sequence(
                            body,
                            SequenceTarget::ControlFlowBody {
                                output: &mut rewritten,
                                phase_delta: &mut body_phase,
                            },
                        )?;
                        self.prepend_body_phase(&mut rewritten, body_phase);
                        Ok::<_, CompilerError>(rewritten)
                    })
                    .transpose()?;

                ControlFlow::IfElse(IfElseGate::new(gate.condition(), true_body, false_body))
            }
            ControlFlow::WhileLoop(gate) => {
                let mut body = Vec::with_capacity(gate.body().len());
                let mut body_phase = Parameter::from(0.0);
                self.apply_sequence(
                    gate.body(),
                    SequenceTarget::ControlFlowBody {
                        output: &mut body,
                        phase_delta: &mut body_phase,
                    },
                )?;
                self.prepend_body_phase(&mut body, body_phase);

                ControlFlow::WhileLoop(WhileLoopGate::new(gate.condition(), body))
            }
        };

        Ok(Instruction::ControlFlowGate(flow))
    }

    fn emit_operation(
        &mut self,
        instruction: Instruction,
        qubits: SmallVec<[Qubit; 3]>,
        params: &[CircuitParam],
        label: Option<Box<str>>,
        target: &mut SequenceTarget<'_>,
    ) -> Result<(), CompilerError> {
        if Self::is_gphase_instruction(&instruction) {
            if matches!(target, SequenceTarget::TopLevel { .. }) {
                self.stats.representation_changes += 1;
            }
            Self::accumulate_phase(target, self.source_gphase_param(params)?);
            return Ok(());
        }

        match target {
            SequenceTarget::TopLevel { .. } => {
                let param_values = params
                    .iter()
                    .map(|param| {
                        resolve_operation_param(self.source, param).map(ParameterValue::from)
                    })
                    .collect::<Result<SmallVec<[_; 3]>, _>>()?;
                self.rebuilt
                    .append(instruction, qubits, param_values, label.as_deref())?;
            }
            SequenceTarget::ControlFlowBody { output, .. } => {
                let params = self.intern_source_params(params)?;
                output.push(Operation {
                    instruction,
                    qubits,
                    params,
                    label,
                });
            }
        }

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

        match target {
            SequenceTarget::TopLevel { .. } => {
                self.rebuilt.append(
                    replacement.instruction.clone(),
                    replacement.qubits.clone(),
                    replacement.params.clone(),
                    None,
                )?;
            }
            SequenceTarget::ControlFlowBody { output, .. } => {
                let params = self.intern_replacement_params(&replacement.params);
                output.push(Operation {
                    instruction: replacement.instruction.clone(),
                    qubits: replacement.qubits.clone(),
                    params,
                    label: None,
                });
            }
        }

        Ok(())
    }

    fn prepend_body_phase(&mut self, body: &mut Vec<Operation>, phase: Parameter) {
        if phase.is_zero() {
            return;
        }

        let phase = ParameterValue::from(phase);
        let params = self.intern_replacement_params(std::slice::from_ref(&phase));
        body.insert(
            0,
            Operation {
                instruction: Instruction::Standard(StandardGate::GPhase),
                qubits: SmallVec::new(),
                params,
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

    fn intern_source_params(
        &mut self,
        params: &[CircuitParam],
    ) -> Result<SmallVec<[CircuitParam; 1]>, CompilerError> {
        params
            .iter()
            .map(|param| {
                let parameter = resolve_operation_param(self.source, param)?;
                Ok(self.intern_parameter(parameter))
            })
            .collect()
    }

    fn intern_replacement_params(
        &mut self,
        params: &[ParameterValue],
    ) -> SmallVec<[CircuitParam; 1]> {
        params
            .iter()
            .cloned()
            .map(|param| match param {
                ParameterValue::Fixed(value) => CircuitParam::Fixed(value),
                ParameterValue::Param(parameter) => self.intern_parameter(parameter),
            })
            .collect()
    }

    fn intern_parameter(&mut self, parameter: Parameter) -> CircuitParam {
        if let Ok(value) = parameter.evaluate(&None) {
            CircuitParam::Fixed(value)
        } else {
            let (index, _) = self.rebuilt.add_parameter(parameter);
            CircuitParam::Index(index as u32)
        }
    }

    fn control_flow_operation_qubits(instruction: &Instruction) -> SmallVec<[Qubit; 3]> {
        let mut qubits = SmallVec::new();
        let mut push_unique = |qubit| {
            if !qubits.contains(&qubit) {
                qubits.push(qubit);
            }
        };

        match instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                for operation in gate.true_body() {
                    for &qubit in &operation.qubits {
                        push_unique(qubit);
                    }
                }
                if let Some(false_body) = gate.false_body() {
                    for operation in false_body {
                        for &qubit in &operation.qubits {
                            push_unique(qubit);
                        }
                    }
                }
                push_unique(gate.condition().qubit);
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                for operation in gate.body() {
                    for &qubit in &operation.qubits {
                        push_unique(qubit);
                    }
                }
                push_unique(gate.condition().qubit);
            }
            _ => {}
        }
        qubits
    }

    fn accumulate_phase(target: &mut SequenceTarget<'_>, phase: Parameter) {
        match target {
            SequenceTarget::TopLevel { phase_delta } => {
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
