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
//!
//! This module bridges the local matcher and the compiler workflow.  It owns the
//! fixpoint loop, splits circuits into rewrite-safe standard-gate blocks, recurs
//! into control-flow bodies when configured, and rebuilds a new circuit from the
//! selected patches.  The matcher decides which local rewrites are legal; this
//! module decides where those rewrites are applied in the circuit structure.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, Parameter,
    ParameterValue, Qubit, StandardGate, WhileLoopGate,
};
use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
use crate::compiler::context::{CompilerContext, ContextChangeSet};
use crate::compiler::error::CompilerError;
use crate::compiler::knowledge::library::RuleLibrary;
use crate::compiler::transform::{TransformDescriptor, TransformOutcome, Transformer};
use indexmap::IndexSet;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

use super::config::RewriteConfig;
use super::matcher::{
    CompiledRuleSet, ReplacementItem, RewritePatch, is_rewrite_safe_operation,
    resolve_parameter_value, select_rewrites,
};

/// Aggregate statistics produced by one knowledge rewrite run.
///
/// These statistics are currently reported through the transform note.  They are
/// aggregated across all fixpoint rounds and include rewrites in nested
/// control-flow bodies.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnowledgeRewriteStats {
    /// Number of fixpoint rounds actually executed.
    pub rounds_executed: u8,
    /// Number of selected rule patches emitted into rebuilt sequences.
    pub rules_applied: usize,
    /// Number of operation sequences whose selected patch set was non-empty.
    pub changed_sequences: usize,
}

impl KnowledgeRewriteStats {
    /// Adds statistics from one completed round to the aggregate run stats.
    fn merge_round(&mut self, other: &RoundStats) {
        self.rules_applied += other.rules_applied;
        self.changed_sequences += other.changed_sequences;
    }
}

/// Per-round mutable statistics.
#[derive(Debug, Clone, Default)]
struct RoundStats {
    rules_applied: usize,
    changed_sequences: usize,
}

/// Result of applying one rewrite round to a circuit.
#[derive(Debug, Clone)]
struct RoundResult {
    circuit: Circuit,
    changed: bool,
    stats: RoundStats,
}

/// Transformer that optimizes circuits using the compiler knowledge base.
///
/// The transformer is intentionally stateless apart from its configuration.  The
/// builtin knowledge library is loaded when the transform runs, then compiled
/// into matcher indexes for that run.
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

    /// Returns the active rewrite configuration.
    pub const fn config(&self) -> &RewriteConfig {
        &self.config
    }
}

static KNOWLEDGE_REWRITE_DESCRIPTOR: TransformDescriptor = TransformDescriptor::new(
    "rewrite.knowledge",
    "Applies knowledge-base local equivalence rewrites",
)
.supports_control_flow(true)
.supports_symbolic_parameters(true)
.modifies_circuit();

impl Transformer for KnowledgeRewriter {
    /// Returns the static workflow descriptor for the knowledge rewriter.
    fn descriptor(&self) -> &'static TransformDescriptor {
        &KNOWLEDGE_REWRITE_DESCRIPTOR
    }

    /// Runs knowledge-based local rewrite to a fixpoint or round limit.
    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        if self.config.max_rounds() == 0 {
            return Err(CompilerError::InvalidContextState(
                "rewrite max_rounds must be greater than zero".to_string(),
            ));
        }

        // Load and compile the knowledge library once per transform run.  The
        // compiled form owns the hot lookup structures used by all rounds.
        let library =
            RuleLibrary::builtin_rules().map_err(|err| CompilerError::TransformFailed {
                name: self.descriptor().name,
                reason: err.to_string(),
            })?;
        let rules = CompiledRuleSet::from_library(library)?;

        let mut current = ctx.circuit().clone();
        let mut aggregate = KnowledgeRewriteStats::default();
        let mut stabilized = false;

        // Repeatedly rebuild from the previous circuit until no rules apply.
        for round in 1..=self.config.max_rounds() {
            aggregate.rounds_executed = round;
            let result = run_round(&current, &rules, &self.config)?;
            if !result.changed {
                stabilized = true;
                break;
            }
            aggregate.merge_round(&result.stats);
            current = result.circuit;
        }

        // No applied rules means the compiler context remains unchanged.
        if aggregate.rules_applied == 0 {
            return Ok(TransformOutcome::unchanged());
        }

        *ctx.circuit_mut() = current;
        let mut outcome = TransformOutcome::changed()
            .with_changes(
                ContextChangeSet::circuit_changed()
                    .with_cfg_structure_changed(true)
                    .with_parameter_table_changed(true),
            )
            .with_note(format!(
                "rewrite: applied {} knowledge rules across {} changed sequences in {} rounds",
                aggregate.rules_applied, aggregate.changed_sequences, aggregate.rounds_executed
            ));

        // A changed final round without a following stable round means the
        // configured iteration bound was reached before proving convergence.
        if !stabilized {
            outcome = outcome.with_diagnostic(CompileDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: "compiler.rewrite.round_limit_reached",
                message: format!(
                    "knowledge rewrite stopped after {} rounds before proving stability",
                    aggregate.rounds_executed
                ),
            });
        }

        Ok(outcome)
    }
}

/// Applies one rewrite round and returns the rebuilt circuit.
fn run_round(
    circuit: &Circuit,
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
) -> Result<RoundResult, CompilerError> {
    let qubits: IndexSet<_> = circuit.qubits().into_iter().collect();
    let mut rebuilt = Circuit::from_parts(
        qubits,
        circuit.symbols().clone(),
        circuit.parameters().clone(),
        Vec::new(),
        circuit.global_phase_param().clone(),
    );
    let mut stats = RoundStats::default();
    let mut phase_delta = Parameter::from(0.0);

    // Rewrite the top-level operation list into a fresh circuit, accumulating
    // replacement global phase separately.
    apply_sequence(
        circuit,
        circuit.operations(),
        &mut rebuilt,
        rules,
        config,
        SequenceTarget::TopLevel {
            phase_delta: &mut phase_delta,
        },
        &mut stats,
    )?;

    // Top-level replacement GPhase operations are represented as global phase
    // metadata instead of ordinary circuit operations.
    if !phase_delta.is_zero() {
        rebuilt.set_global_phase(circuit.global_phase() + phase_delta);
    }

    Ok(RoundResult {
        circuit: rebuilt,
        changed: stats.rules_applied > 0,
        stats,
    })
}

/// Destination for emitted operations while rebuilding a sequence.
///
/// Top-level emission appends into the rebuilt circuit.  Control-flow-body
/// emission appends raw operations into the body vector that will be installed in
/// a rebuilt control-flow instruction.
enum SequenceTarget<'a> {
    /// Top-level circuit output plus accumulated global phase delta.
    TopLevel { phase_delta: &'a mut Parameter },
    /// Output vector for a nested control-flow body.
    ControlFlowBody { output: &'a mut Vec<Operation> },
}

/// Rewrites one operation sequence into the selected output target.
///
/// A sequence is split into maximal contiguous blocks of standard-gate
/// operations.  Non-standard operations are emitted unchanged, except that
/// control-flow instructions may have their bodies recursively rewritten.
fn apply_sequence(
    source: &Circuit,
    operations: &[Operation],
    rebuilt: &mut Circuit,
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
    mut target: SequenceTarget<'_>,
    stats: &mut RoundStats,
) -> Result<(), CompilerError> {
    let mut cursor = 0;
    while cursor < operations.len() {
        // Non-standard operations are hard boundaries for local rewrite blocks.
        if !is_rewrite_safe_operation(&operations[cursor]) {
            emit_original_operation(
                source,
                &operations[cursor],
                rebuilt,
                config,
                &mut target,
                rules,
                stats,
            )?;
            cursor += 1;
            continue;
        }

        // Gather one maximal standard-gate block and let the matcher choose
        // non-overlapping rewrites inside that block.
        let block_start = cursor;
        while cursor < operations.len() && is_rewrite_safe_operation(&operations[cursor]) {
            cursor += 1;
        }
        let block = &operations[block_start..cursor];
        let selected = select_rewrites(source, block, rules, config)?;
        if selected.is_empty() {
            for operation in block {
                emit_original_operation(
                    source,
                    operation,
                    rebuilt,
                    config,
                    &mut target,
                    rules,
                    stats,
                )?;
            }
        } else {
            stats.changed_sequences += 1;
            emit_rewritten_block(source, block, selected.patches, rebuilt, &mut target, stats)?;
        }
    }

    Ok(())
}

/// Emits one rewritten standard-gate block.
///
/// Patches are keyed by their first matched source position.  When iteration
/// reaches that position, replacements are emitted.  Every matched source
/// position is skipped so it is not copied after its replacement.
fn emit_rewritten_block(
    source: &Circuit,
    block: &[Operation],
    patches: Vec<RewritePatch>,
    rebuilt: &mut Circuit,
    target: &mut SequenceTarget<'_>,
    stats: &mut RoundStats,
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
            stats.rules_applied += 1;
            for replacement in &patch.replacements {
                emit_replacement(rebuilt, replacement, target)?;
            }
        }

        if skipped_positions.contains(&position) {
            continue;
        }

        emit_original_operation_without_recursion(source, operation, rebuilt, target)?;
    }

    Ok(())
}

/// Emits an original operation, optionally rewriting nested control-flow bodies.
///
/// Control-flow operations are not matched as local rewrite items, but their
/// bodies can be recursively processed when enabled by configuration.  All other
/// operations are copied directly.
fn emit_original_operation(
    source: &Circuit,
    operation: &Operation,
    rebuilt: &mut Circuit,
    config: &RewriteConfig,
    target: &mut SequenceTarget<'_>,
    rules: &CompiledRuleSet,
    stats: &mut RoundStats,
) -> Result<(), CompilerError> {
    if !config.recurses_control_flow() {
        return emit_original_operation_without_recursion(source, operation, rebuilt, target);
    }

    let rewritten_instruction = match &operation.instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            let mut true_body = Vec::with_capacity(gate.true_body().len());
            apply_sequence(
                source,
                gate.true_body(),
                rebuilt,
                rules,
                config,
                SequenceTarget::ControlFlowBody {
                    output: &mut true_body,
                },
                stats,
            )?;

            let false_body = gate
                .false_body()
                .map(|body| {
                    let mut rewritten = Vec::with_capacity(body.len());
                    apply_sequence(
                        source,
                        body,
                        rebuilt,
                        rules,
                        config,
                        SequenceTarget::ControlFlowBody {
                            output: &mut rewritten,
                        },
                        stats,
                    )?;
                    Ok::<_, CompilerError>(rewritten)
                })
                .transpose()?;

            Some(Instruction::ControlFlowGate(ControlFlow::IfElse(
                IfElseGate::new(gate.condition(), true_body, false_body),
            )))
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            let mut body = Vec::with_capacity(gate.body().len());
            apply_sequence(
                source,
                gate.body(),
                rebuilt,
                rules,
                config,
                SequenceTarget::ControlFlowBody { output: &mut body },
                stats,
            )?;

            Some(Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                WhileLoopGate::new(gate.condition(), body),
            )))
        }
        _ => None,
    };

    if let Some(instruction) = rewritten_instruction {
        let qubits = control_flow_operation_qubits(&instruction);
        emit_operation_parts(
            source,
            rebuilt,
            target,
            instruction,
            qubits,
            operation.params.as_slice(),
            operation.label.clone(),
        )
    } else {
        emit_original_operation_without_recursion(source, operation, rebuilt, target)
    }
}

/// Recomputes the qubit list for a rebuilt control-flow operation.
///
/// Rewriting can remove every operation in a body, so copying the original qubit
/// list may leave stale body qubits attached to the enclosing operation.  The
/// condition qubit is always retained.
fn control_flow_operation_qubits(instruction: &Instruction) -> SmallVec<[Qubit; 3]> {
    let mut qubits = SmallVec::new();
    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            collect_operation_qubits(gate.true_body(), &mut qubits);
            if let Some(false_body) = gate.false_body() {
                collect_operation_qubits(false_body, &mut qubits);
            }
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            collect_operation_qubits(gate.body(), &mut qubits);
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        _ => {}
    }
    qubits
}

/// Collects all operation qubits into `output` without duplicates.
fn collect_operation_qubits(operations: &[Operation], output: &mut SmallVec<[Qubit; 3]>) {
    for operation in operations {
        for &qubit in &operation.qubits {
            push_unique_qubit(output, qubit);
        }
    }
}

/// Pushes a qubit only if it is not already present.
fn push_unique_qubit(output: &mut SmallVec<[Qubit; 3]>, qubit: Qubit) {
    if !output.contains(&qubit) {
        output.push(qubit);
    }
}

/// Emits an operation exactly as represented in the source sequence.
fn emit_original_operation_without_recursion(
    source: &Circuit,
    operation: &Operation,
    rebuilt: &mut Circuit,
    target: &mut SequenceTarget<'_>,
) -> Result<(), CompilerError> {
    emit_operation_parts(
        source,
        rebuilt,
        target,
        operation.instruction.clone(),
        operation.qubits.clone(),
        operation.params.as_slice(),
        operation.label.clone(),
    )
}

/// Emits operation parts into either the top-level circuit or a body vector.
///
/// Top-level emission resolves parameter indices against the source circuit and
/// interns parameters in the rebuilt circuit through `Circuit::append`.
/// Control-flow body emission preserves the body operation representation.
fn emit_operation_parts(
    source: &Circuit,
    rebuilt: &mut Circuit,
    target: &mut SequenceTarget<'_>,
    instruction: Instruction,
    qubits: SmallVec<[crate::circuit::Qubit; 3]>,
    params: &[CircuitParam],
    label: Option<Box<str>>,
) -> Result<(), CompilerError> {
    match target {
        SequenceTarget::TopLevel { .. } => {
            let param_values = params
                .iter()
                .map(|param| resolve_parameter_value(source, param))
                .collect::<Result<SmallVec<[_; 3]>, _>>()?;
            rebuilt.append(instruction, qubits, param_values, label.as_deref())?;
        }
        SequenceTarget::ControlFlowBody { output } => output.push(Operation {
            instruction,
            qubits,
            params: params.iter().cloned().collect(),
            label,
        }),
    }

    Ok(())
}

/// Emits one instantiated replacement operation.
///
/// Replacement `GPhase` has special policy: at top level it is accumulated into
/// circuit global phase metadata, while inside control-flow bodies it is dropped
/// because the current IR has no branch-local global phase storage.
fn emit_replacement(
    rebuilt: &mut Circuit,
    replacement: &ReplacementItem,
    target: &mut SequenceTarget<'_>,
) -> Result<(), CompilerError> {
    if matches!(
        replacement.instruction,
        Instruction::Standard(StandardGate::GPhase)
    ) {
        let phase = replacement.params.first().ok_or_else(|| {
            CompilerError::InvariantViolation(
                "GPhase replacement must contain one parameter".to_string(),
            )
        })?;
        match target {
            SequenceTarget::TopLevel { phase_delta } => {
                **phase_delta = &**phase_delta + &parameter_value_to_parameter(phase);
                return Ok(());
            }
            SequenceTarget::ControlFlowBody { .. } => return Ok(()),
        }
    }

    match target {
        SequenceTarget::TopLevel { .. } => {
            rebuilt.append(
                replacement.instruction.clone(),
                replacement.qubits.clone(),
                replacement.params.clone(),
                None,
            )?;
        }
        SequenceTarget::ControlFlowBody { output } => {
            let params = replacement
                .params
                .iter()
                .cloned()
                .map(|param| parameter_value_to_circuit_param(rebuilt, param))
                .collect();
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

/// Converts a rewrite parameter value into a parameter expression.
fn parameter_value_to_parameter(value: &ParameterValue) -> Parameter {
    match value {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => parameter.clone(),
    }
}

/// Interns a replacement parameter in the rebuilt circuit when needed.
fn parameter_value_to_circuit_param(circuit: &mut Circuit, value: ParameterValue) -> CircuitParam {
    match value {
        ParameterValue::Fixed(value) => CircuitParam::Fixed(value),
        ParameterValue::Param(parameter) => {
            let (index, _) = circuit.add_parameter(parameter);
            CircuitParam::Index(index as u32)
        }
    }
}
