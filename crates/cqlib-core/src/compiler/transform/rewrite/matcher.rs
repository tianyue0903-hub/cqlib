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

//! Rule compilation and dependency-aware sequence matching.
//!
//! The matcher is the local decision engine for knowledge rewrite.  It compiles
//! raw knowledge-base rules into a first-gate index, scans one standard-gate
//! block at a time, and returns a non-overlapping set of rewrite patches.  It is
//! dependency-aware rather than purely adjacent: a later pattern item may be
//! matched across intervening operations when the commutation oracle proves that
//! the skipped operations can safely move around both the matched source and the
//! instantiated replacement.

use crate::circuit::{
    Circuit, CircuitParam, Instruction, Operation, Parameter, ParameterValue, Qubit, StandardGate,
};
use crate::compiler::error::CompilerError;
use crate::compiler::knowledge::library::{RuleKind, RuleLibrary};
use crate::compiler::knowledge::rule::{Condition, Rule, RuleItem};
use crate::compiler::transform::rewrite::config::{GPhaseCost, LocalRewriteCost, RewriteConfig};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

const PARAMETER_TOLERANCE: f64 = 1e-12;

/// A rewrite rule prepared for repeated matching.
///
/// The compiled form caches static metadata used by hot matching paths:
/// rule kind, pattern size, touched rule-local qubit count, and gate sets used
/// for target-basis filtering.
pub(crate) struct CompiledRule {
    /// Stable rule id from the source [`RuleLibrary`].
    id: usize,
    /// Coarse compiler use-case for policy filtering.
    kind: RuleKind,
    /// Number of source operations in the match block.
    match_len: usize,
    /// Number of distinct rule-local qubits referenced by match or rewrite.
    qubit_count: usize,
    /// Static operation-count delta used as a tie-breaker.
    static_cost_delta: isize,
    /// Distinct standard gates appearing in the match side.
    match_gates: SmallVec<[StandardGate; 4]>,
    /// Distinct standard gates appearing in the rewrite side.
    rewrite_gates: SmallVec<[StandardGate; 4]>,
    /// The validated runtime rule.
    rule: Rule,
}

/// Compiled rule collection with a first-gate candidate index.
///
/// `first_gate_map` keeps candidate lookup cheap for each anchor operation.
/// Commutation rules are not applied as ordinary rewrite patches; they are
/// extracted into [`CommutationOracle`] and used only to justify skipped
/// operations during non-adjacent matching.
pub(crate) struct CompiledRuleSet {
    rules: Vec<CompiledRule>,
    first_gate_map: HashMap<StandardGate, SmallVec<[usize; 8]>>,
    commutation: CommutationOracle,
}

/// One operation emitted by a rewrite target.
///
/// Replacements use concrete circuit qubits and instantiated parameters, so
/// they can be emitted without consulting rule-local bindings again.
#[derive(Debug, Clone)]
pub(crate) struct ReplacementItem {
    pub(crate) instruction: Instruction,
    pub(crate) qubits: SmallVec<[Qubit; 3]>,
    pub(crate) params: SmallVec<[ParameterValue; 3]>,
}

/// A selected replacement for matched operation positions in one block.
///
/// `matched_positions` are positions within the current block, not global
/// circuit operation indices.  Rebuild logic uses them to suppress the matched
/// source operations and emit replacements at `first_position`.
#[derive(Debug, Clone)]
pub(crate) struct RewritePatch {
    pub(crate) rule_id: usize,
    static_cost_delta: isize,
    pub(crate) first_position: usize,
    pub(crate) matched_positions: Vec<usize>,
    pub(crate) replacements: Vec<ReplacementItem>,
}

impl RewritePatch {
    /// Returns the precomputed rewrite-length minus match-length delta.
    fn static_cost_delta(&self) -> isize {
        self.static_cost_delta
    }
}

/// All non-overlapping rewrites selected for one operation block.
#[derive(Debug, Clone, Default)]
pub(crate) struct SelectedRewrites {
    pub(crate) patches: Vec<RewritePatch>,
}

impl SelectedRewrites {
    /// Returns whether this block has no accepted rewrite patches.
    pub(crate) fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }
}

/// Mutable bindings accumulated while matching one rule instance.
///
/// Qubits are tracked in both directions to guarantee a one-to-one mapping from
/// rule-local labels to concrete circuit qubits.  Parameter bindings map rule
/// symbols such as `theta` to concrete circuit parameters.
#[derive(Clone, Default)]
struct MatchState {
    qubits: HashMap<u32, Qubit>,
    reverse_qubits: HashMap<Qubit, u32>,
    params: HashMap<String, Parameter>,
}

/// Candidate patch plus before/after cost used by greedy selection.
#[derive(Clone)]
struct CandidatePatch {
    patch: RewritePatch,
    before: LocalRewriteCost,
    after: LocalRewriteCost,
}

/// Read-only commutation prover derived from `RuleKind::Commute` rules.
///
/// The oracle has two built-in proofs: disjoint operations commute, and any
/// `GPhase` commutes.  All other same-qubit proofs must be represented as
/// explicit two-operation swap rules in the knowledge base.
#[derive(Clone, Default)]
struct CommutationOracle {
    patterns: Vec<CommutationPattern>,
}

/// One `A; B -> B; A` rule normalized as an ordered commutation pattern.
#[derive(Clone)]
struct CommutationPattern {
    lhs: RuleItem,
    rhs: RuleItem,
}

/// Borrowed operation view used by the commutation matcher.
#[derive(Clone, Copy)]
struct ConcreteOperation<'a> {
    instruction: &'a Instruction,
    qubits: &'a [Qubit],
    params: &'a [Parameter],
}

/// Preprocessed view of one contiguous standard-gate block.
///
/// Circuit parameter indices are resolved once up front so every candidate rule
/// sees parameters in value form.  The gate set and qubit count serve as cheap
/// static filters before the expensive matcher runs.
struct BlockContext<'a> {
    operations: &'a [Operation],
    resolved_params: Vec<SmallVec<[Parameter; 3]>>,
    gate_set: HashSet<StandardGate>,
    qubit_count: usize,
}

impl<'a> BlockContext<'a> {
    /// Builds a block context and resolves all operation parameters.
    fn new(circuit: &'a Circuit, operations: &'a [Operation]) -> Result<Self, CompilerError> {
        let mut resolved_params = Vec::with_capacity(operations.len());
        let mut touched_qubits = HashSet::new();
        let mut gate_set = HashSet::new();

        for operation in operations {
            let params = operation
                .params
                .iter()
                .map(|param| resolve_operation_param(circuit, param))
                .collect::<Result<SmallVec<[_; 3]>, _>>()?;
            resolved_params.push(params);

            for &qubit in &operation.qubits {
                touched_qubits.insert(qubit);
            }
            if let Instruction::Standard(gate) = operation.instruction {
                gate_set.insert(gate);
            }
        }

        Ok(Self {
            operations,
            resolved_params,
            gate_set,
            qubit_count: touched_qubits.len(),
        })
    }

    /// Returns the number of operations in this block.
    fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns the operation at a block-local position.
    fn operation(&self, position: usize) -> &Operation {
        &self.operations[position]
    }

    /// Returns resolved parameters for the operation at a block-local position.
    fn params(&self, position: usize) -> &[Parameter] {
        &self.resolved_params[position]
    }
}

impl CompiledRuleSet {
    /// Compiles a rule library into matcher data structures.
    ///
    /// Rule ids are preserved as library-local indices so diagnostics and tests
    /// can refer back to the original rule ordering.
    pub(crate) fn from_library(library: &RuleLibrary) -> Result<Self, CompilerError> {
        let mut rules = Vec::with_capacity(library.len());
        let mut first_gate_map: HashMap<StandardGate, SmallVec<[usize; 8]>> = HashMap::new();
        let kind_by_id = build_kind_index(library);
        let commutation = CommutationOracle::from_library(library, &kind_by_id);

        for (index, rule) in library.rules().iter().cloned().enumerate() {
            let kind = kind_by_id.get(&index).copied().unwrap_or(RuleKind::Other);
            push_compiled_rule(&mut rules, &mut first_gate_map, index, kind, rule)?;
        }

        Ok(Self {
            rules,
            first_gate_map,
            commutation,
        })
    }

    /// Returns candidate compiled-rule indices for an anchor gate.
    fn candidates_for_first_gate(&self, gate: StandardGate) -> &[usize] {
        self.first_gate_map
            .get(&gate)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns a compiled rule by compiled-rule index.
    fn get(&self, index: usize) -> &CompiledRule {
        &self.rules[index]
    }
}

impl CommutationOracle {
    /// Extracts commutation patterns from commute-kind library rules.
    fn from_library(library: &RuleLibrary, kind_by_id: &HashMap<usize, RuleKind>) -> Self {
        let mut patterns = Vec::new();
        for (index, rule) in library.rules().iter().enumerate() {
            if kind_by_id.get(&index).copied() != Some(RuleKind::Commute) {
                continue;
            }
            if let Some(pattern) = commutation_pattern_from_rule(rule) {
                patterns.push(pattern);
            }
        }
        Self { patterns }
    }

    /// Returns whether two block operations commute.
    fn operations_commute(
        &self,
        block: &BlockContext<'_>,
        lhs_position: usize,
        rhs_position: usize,
    ) -> Result<bool, CompilerError> {
        let lhs = block.operation(lhs_position);
        let rhs = block.operation(rhs_position);
        self.concrete_operations_commute(
            ConcreteOperation {
                instruction: &lhs.instruction,
                qubits: &lhs.qubits,
                params: block.params(lhs_position),
            },
            ConcreteOperation {
                instruction: &rhs.instruction,
                qubits: &rhs.qubits,
                params: block.params(rhs_position),
            },
        )
    }

    /// Returns whether a source operation commutes with an instantiated replacement.
    fn operation_commutes_with_replacement(
        &self,
        block: &BlockContext<'_>,
        operation_position: usize,
        replacement: &ReplacementItem,
    ) -> Result<bool, CompilerError> {
        let operation = block.operation(operation_position);
        let replacement_params = replacement_parameters(replacement);
        self.concrete_operations_commute(
            ConcreteOperation {
                instruction: &operation.instruction,
                qubits: &operation.qubits,
                params: block.params(operation_position),
            },
            ConcreteOperation {
                instruction: &replacement.instruction,
                qubits: &replacement.qubits,
                params: &replacement_params,
            },
        )
    }

    /// Proves commutation for two concrete operations.
    fn concrete_operations_commute(
        &self,
        lhs: ConcreteOperation<'_>,
        rhs: ConcreteOperation<'_>,
    ) -> Result<bool, CompilerError> {
        if operation_is_gphase(lhs.instruction)
            || operation_is_gphase(rhs.instruction)
            || !operations_touch(lhs.qubits, rhs.qubits)
        {
            return Ok(true);
        }

        for pattern in &self.patterns {
            if commutation_pattern_matches(pattern, lhs, rhs)?
                || commutation_pattern_matches(pattern, rhs, lhs)?
            {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Adds one validated runtime rule to the compiled rule set under construction.
///
/// The rule is indexed by its first match gate and stores the original library
/// id so selected patches can report the source rule.
fn push_compiled_rule(
    rules: &mut Vec<CompiledRule>,
    first_gate_map: &mut HashMap<StandardGate, SmallVec<[usize; 8]>>,
    id: usize,
    kind: RuleKind,
    rule: Rule,
) -> Result<(), CompilerError> {
    let first_gate = first_gate(&rule)?;
    let match_len = rule.operations.len();
    let rewrite_len = rule.target.len();
    let qubit_count = compiled_rule_qubit_count(&rule);
    let match_gates = collect_rule_gates(&rule.operations);
    let rewrite_gates = collect_rule_gates(&rule.target);
    first_gate_map
        .entry(first_gate)
        .or_default()
        .push(rules.len());
    rules.push(CompiledRule {
        id,
        kind,
        match_len,
        qubit_count,
        static_cost_delta: rewrite_len as isize - match_len as isize,
        match_gates,
        rewrite_gates,
        rule,
    });
    Ok(())
}

/// Returns whether an operation may participate in local rewrite matching.
///
/// Only standard gates are considered safe.  Directives, delays, measurement,
/// reset, multi-control gate wrappers, and control-flow gates are handled by the
/// rewriter as block boundaries.
pub(crate) fn is_rewrite_safe_operation(operation: &Operation) -> bool {
    matches!(operation.instruction, Instruction::Standard(_))
}

/// Selects a non-overlapping set of rewrite patches for one operation block.
///
/// Candidate generation is independent for every anchor position.  After
/// matching, candidates are sorted by local replacement cost and selected
/// greedily while rejecting any candidate whose matched source positions overlap
/// a previously selected patch.
pub(crate) fn select_rewrites(
    circuit: &Circuit,
    operations: &[Operation],
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
) -> Result<SelectedRewrites, CompilerError> {
    if config.target_gates().is_some_and(|gates| gates.is_empty()) {
        return Err(CompilerError::InvalidContextState(
            "rewrite target gate set must not be empty".to_string(),
        ));
    }

    let block = BlockContext::new(circuit, operations)?;
    let mut candidates = Vec::new();

    for anchor in 0..block.len() {
        let operation = block.operation(anchor);
        if config.skips_labeled_ops() && operation.label.is_some() {
            continue;
        }
        let first_gate = match operation.instruction {
            Instruction::Standard(gate) => gate,
            _ => continue,
        };

        // Use the first-gate index to avoid considering rules that cannot start
        // at this anchor operation.
        for &rule_index in rules.candidates_for_first_gate(first_gate) {
            let compiled = rules.get(rule_index);
            if !rule_passes_static_filters(compiled, config, &block) {
                continue;
            }
            if let Some(candidate) =
                try_match_rule(&block, anchor, compiled, &rules.commutation, config)?
            {
                candidates.push(candidate);
            }
        }
    }

    // Prefer candidates that produce the lowest local cost, then choose stable
    // deterministic tie-breakers so repeated runs produce identical patches.
    candidates.sort_by(|lhs, rhs| {
        lhs.after
            .cmp(&rhs.after)
            .then_with(|| lhs.before.cmp(&rhs.before).reverse())
            .then_with(|| {
                rhs.patch
                    .matched_positions
                    .len()
                    .cmp(&lhs.patch.matched_positions.len())
            })
            .then_with(|| {
                lhs.patch
                    .static_cost_delta()
                    .cmp(&rhs.patch.static_cost_delta())
            })
            .then_with(|| lhs.patch.first_position.cmp(&rhs.patch.first_position))
            .then_with(|| lhs.patch.rule_id.cmp(&rhs.patch.rule_id))
    });

    // Greedily keep the best candidates while enforcing source-position
    // disjointness.
    let mut occupied_positions = HashSet::new();
    let mut patches = Vec::new();
    for candidate in candidates {
        if candidate
            .patch
            .matched_positions
            .iter()
            .any(|position| occupied_positions.contains(position))
        {
            continue;
        }
        occupied_positions.extend(candidate.patch.matched_positions.iter().copied());
        patches.push(candidate.patch);
    }

    patches.sort_by_key(|patch| patch.first_position);
    Ok(SelectedRewrites { patches })
}

/// Builds a rule-id to rule-kind lookup from the library kind indexes.
fn build_kind_index(library: &RuleLibrary) -> HashMap<usize, RuleKind> {
    let mut index = HashMap::new();
    for kind in [
        RuleKind::Simplify,
        RuleKind::Cancel,
        RuleKind::Merge,
        RuleKind::Commute,
        RuleKind::Decompose,
        RuleKind::Canonicalize,
        RuleKind::HardwareNative,
        RuleKind::Other,
    ] {
        for id in library.rules_by_kind(kind) {
            index.insert(id.as_usize(), kind);
        }
    }
    index
}

/// Returns the first standard gate in a non-empty rule match block.
fn first_gate(rule: &Rule) -> Result<StandardGate, CompilerError> {
    match rule.operations.first().map(|item| &item.instruction) {
        Some(Instruction::Standard(gate)) => Ok(*gate),
        Some(other) => Err(CompilerError::InvariantViolation(format!(
            "rewrite rule contains unsupported first instruction {other:?}"
        ))),
        None => Err(CompilerError::InvariantViolation(
            "rewrite rule contains an empty match block".to_string(),
        )),
    }
}

/// Counts distinct rule-local qubits referenced by a rule.
fn compiled_rule_qubit_count(rule: &Rule) -> usize {
    let mut qubits = HashSet::new();
    for item in rule.operations.iter().chain(&rule.target) {
        qubits.extend(item.qubits.iter().copied());
    }
    qubits.len()
}

/// Collects distinct standard gates from rule items while preserving order.
fn collect_rule_gates(items: &[RuleItem]) -> SmallVec<[StandardGate; 4]> {
    let mut gates = SmallVec::new();
    for item in items {
        if let Instruction::Standard(gate) = item.instruction
            && !gates.contains(&gate)
        {
            gates.push(gate);
        }
    }
    gates
}

/// Extracts a two-operation `A; B -> B; A` commute pattern from a rule.
///
/// Conditional commute rules are ignored because the current oracle is a simple
/// structural prover used on hot matching paths.
fn commutation_pattern_from_rule(rule: &Rule) -> Option<CommutationPattern> {
    if rule.operations.len() != 2
        || rule.target.len() != 2
        || rule
            .conditions
            .as_ref()
            .is_some_and(|conditions| !conditions.is_empty())
    {
        return None;
    }

    if !rule_items_equivalent(&rule.operations[0], &rule.target[1])
        || !rule_items_equivalent(&rule.operations[1], &rule.target[0])
    {
        return None;
    }

    Some(CommutationPattern {
        lhs: rule.operations[0].clone(),
        rhs: rule.operations[1].clone(),
    })
}

/// Returns whether a concrete operation pair matches a commutation pattern.
fn commutation_pattern_matches(
    pattern: &CommutationPattern,
    lhs: ConcreteOperation<'_>,
    rhs: ConcreteOperation<'_>,
) -> Result<bool, CompilerError> {
    let mut state = MatchState::default();
    Ok(match_commutation_item(&pattern.lhs, lhs, &mut state)?
        && match_commutation_item(&pattern.rhs, rhs, &mut state)?)
}

/// Matches one item of a commutation pattern against a concrete operation.
fn match_commutation_item(
    item: &RuleItem,
    concrete: ConcreteOperation<'_>,
    state: &mut MatchState,
) -> Result<bool, CompilerError> {
    if !instructions_equivalent(concrete.instruction, &item.instruction) {
        return Ok(false);
    }
    if concrete.qubits.len() != item.qubits.len() {
        return Ok(false);
    }

    for (&rule_qubit, &actual_qubit) in item.qubits.iter().zip(concrete.qubits) {
        if let Some(bound) = state.qubits.get(&rule_qubit) {
            if *bound != actual_qubit {
                return Ok(false);
            }
        } else if let Some(other_rule_qubit) = state.reverse_qubits.get(&actual_qubit) {
            if *other_rule_qubit != rule_qubit {
                return Ok(false);
            }
        } else {
            state.qubits.insert(rule_qubit, actual_qubit);
            state.reverse_qubits.insert(actual_qubit, rule_qubit);
        }
    }

    let rule_params = item.params.as_deref().unwrap_or(&[]);
    if concrete.params.len() != rule_params.len() {
        return Ok(false);
    }
    for (rule_param, actual) in rule_params.iter().zip(concrete.params) {
        if !match_parameter(rule_param, actual, &mut state.params)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Returns whether two rule items are structurally equivalent.
fn rule_items_equivalent(lhs: &RuleItem, rhs: &RuleItem) -> bool {
    instructions_equivalent(&lhs.instruction, &rhs.instruction)
        && lhs.qubits == rhs.qubits
        && rule_item_params(lhs).len() == rule_item_params(rhs).len()
        && rule_item_params(lhs)
            .iter()
            .zip(rule_item_params(rhs))
            .all(|(lhs, rhs)| parameter_values_equivalent(lhs, rhs))
}

/// Returns the parameters attached to a rule item.
fn rule_item_params(item: &RuleItem) -> &[ParameterValue] {
    item.params.as_deref().unwrap_or(&[])
}

/// Returns whether two instructions are the same supported standard gate.
fn instructions_equivalent(lhs: &Instruction, rhs: &Instruction) -> bool {
    matches!(
        (lhs, rhs),
        (Instruction::Standard(lhs), Instruction::Standard(rhs)) if lhs == rhs
    )
}

/// Returns whether an instruction is a standard global-phase operation.
fn operation_is_gphase(instruction: &Instruction) -> bool {
    matches!(instruction, Instruction::Standard(StandardGate::GPhase))
}

/// Returns whether two qubit lists share at least one qubit.
fn operations_touch(lhs: &[Qubit], rhs: &[Qubit]) -> bool {
    lhs.iter().any(|qubit| rhs.contains(qubit))
}

/// Converts replacement parameters to `Parameter` values for commutation checks.
fn replacement_parameters(replacement: &ReplacementItem) -> SmallVec<[Parameter; 3]> {
    replacement
        .params
        .iter()
        .map(parameter_value_to_parameter)
        .collect()
}

/// Returns whether two rule parameter values are equivalent.
fn parameter_values_equivalent(lhs: &ParameterValue, rhs: &ParameterValue) -> bool {
    parameters_equivalent(
        &parameter_value_to_parameter(lhs),
        &parameter_value_to_parameter(rhs),
    )
}

/// Converts a runtime parameter value into a symbolic-or-fixed parameter.
fn parameter_value_to_parameter(value: &ParameterValue) -> Parameter {
    match value {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => parameter.clone(),
    }
}

/// Applies cheap rule filters that do not require pattern matching.
fn rule_passes_static_filters(
    rule: &CompiledRule,
    config: &RewriteConfig,
    block: &BlockContext<'_>,
) -> bool {
    rule.kind != RuleKind::Commute
        && config.allows_kind(rule.kind)
        && rule.match_len <= config.max_pattern_len()
        && rule.qubit_count <= block.qubit_count
        && rule_passes_target_filter(rule, config.target_gates(), block)
}

/// Applies target-basis filtering for a compiled rule.
///
/// Match gates must be present in the current block, and rewrite gates must be
/// target gates.  Replacement `GPhase` is allowed implicitly because it is not
/// emitted as an ordinary top-level operation.
fn rule_passes_target_filter(
    rule: &CompiledRule,
    target_gates: Option<&[StandardGate]>,
    block: &BlockContext<'_>,
) -> bool {
    let Some(target_gates) = target_gates else {
        return true;
    };

    rule.match_gates
        .iter()
        .all(|gate| block.gate_set.contains(gate))
        && rule
            .rewrite_gates
            .iter()
            .all(|gate| *gate == StandardGate::GPhase || target_gates.contains(gate))
}

/// Attempts to match one compiled rule at one anchor position.
///
/// The matcher is intentionally greedy inside a rule: for each subsequent rule
/// item it accepts the first position in the configured window that matches
/// structurally and can be reached through commuting skipped operations.  Rule
/// conditions are checked only after the full structural match is found.
fn try_match_rule(
    block: &BlockContext<'_>,
    anchor: usize,
    compiled: &CompiledRule,
    commutation: &CommutationOracle,
    config: &RewriteConfig,
) -> Result<Option<CandidatePatch>, CompilerError> {
    let rule = &compiled.rule;
    let mut state = MatchState::default();

    // Step 1: bind the first rule item to the anchor operation.
    if !match_item(block, anchor, &rule.operations[0], &mut state, config)? {
        return Ok(None);
    }

    let mut matched_positions = vec![anchor];
    let mut skipped_positions = Vec::new();
    let mut cursor = anchor + 1;
    for item in rule.operations.iter().skip(1) {
        let mut found = None;
        let limit = block.len().min(cursor + config.max_window_ops());

        // Step 2: scan forward for the next pattern item within the window.
        for position in cursor..limit {
            if !candidate_gate_matches(block.operation(position), item) {
                continue;
            }
            // Step 3: any source operations skipped by non-adjacent matching must
            // commute with the partial match and the current candidate.
            if !can_skip_between(
                block,
                cursor..position,
                &matched_positions,
                position,
                commutation,
                config,
            )? {
                continue;
            }

            let mut next_state = state.clone();
            if match_item(block, position, item, &mut next_state, config)? {
                found = Some((position, next_state));
                break;
            }
        }

        let Some((position, next_state)) = found else {
            return Ok(None);
        };
        state = next_state;
        skipped_positions.extend(cursor..position);
        matched_positions.push(position);
        cursor = position + 1;
    }

    // Step 4: apply symbolic/numeric rule conditions after all parameters have
    // been bound by the structural match.
    if !conditions_hold(rule.conditions.as_deref(), &state.params)? {
        return Ok(None);
    }

    // Step 5: instantiate the rewrite target using the matched qubit and
    // parameter bindings.
    let replacements = instantiate_target(&rule.target, &state)?;
    // Step 6: replacements must also commute with skipped operations; otherwise
    // emitting them at the first matched position would change behavior.
    if !replacements_commute_with_skipped(block, &skipped_positions, &replacements, commutation)? {
        return Ok(None);
    }

    // Step 7: accept only rewrites permitted by the configured local cost model.
    let before = cost_for_operation_positions(block, &matched_positions, config.target_gates());
    let after = cost_for_replacements(&replacements, config.target_gates());
    if !config.allows_rewrite(compiled.kind, before, after) {
        return Ok(None);
    }

    let first_position = matched_positions[0];
    Ok(Some(CandidatePatch {
        before,
        after,
        patch: RewritePatch {
            rule_id: compiled.id,
            static_cost_delta: compiled.static_cost_delta,
            first_position,
            matched_positions,
            replacements,
        },
    }))
}

/// Checks whether an operation has the same standard gate as a rule item.
fn candidate_gate_matches(operation: &Operation, item: &RuleItem) -> bool {
    match (&operation.instruction, &item.instruction) {
        (Instruction::Standard(lhs), Instruction::Standard(rhs)) => lhs == rhs,
        _ => false,
    }
}

/// Returns whether skipped operations may be crossed by a non-adjacent match.
///
/// Operations that do not touch any relevant qubits are ignored.  Touching
/// operations must commute with all previously matched operations and with the
/// current candidate position.
fn can_skip_between(
    block: &BlockContext<'_>,
    skipped: Range<usize>,
    matched_positions: &[usize],
    candidate_position: usize,
    commutation: &CommutationOracle,
    config: &RewriteConfig,
) -> Result<bool, CompilerError> {
    if skipped.is_empty() {
        return Ok(true);
    }

    let mut relevant = HashSet::new();
    for &position in matched_positions {
        relevant.extend(block.operation(position).qubits.iter().copied());
    }
    relevant.extend(block.operation(candidate_position).qubits.iter().copied());

    for skipped_position in skipped {
        let skipped_operation = block.operation(skipped_position);
        if !skipped_operation
            .qubits
            .iter()
            .any(|qubit| relevant.contains(qubit))
        {
            continue;
        }

        if config.skips_labeled_ops() && skipped_operation.label.is_some() {
            return Ok(false);
        }
        for &matched_position in matched_positions {
            if !commutation.operations_commute(block, skipped_position, matched_position)? {
                return Ok(false);
            }
        }
        if !commutation.operations_commute(block, skipped_position, candidate_position)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Returns whether instantiated replacements may be emitted before skipped ops.
fn replacements_commute_with_skipped(
    block: &BlockContext<'_>,
    skipped_positions: &[usize],
    replacements: &[ReplacementItem],
    commutation: &CommutationOracle,
) -> Result<bool, CompilerError> {
    if skipped_positions.is_empty() || replacements.is_empty() {
        return Ok(true);
    }

    for &skipped_position in skipped_positions {
        for replacement in replacements {
            if !operations_touch(
                &block.operation(skipped_position).qubits,
                &replacement.qubits,
            ) {
                continue;
            }
            if !commutation.operation_commutes_with_replacement(
                block,
                skipped_position,
                replacement,
            )? {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Matches one rule item against one operation and updates match bindings.
///
/// Qubit labels must form a one-to-one mapping.  Parameter symbols are bound on
/// first use, and repeated uses must be provably equivalent to the existing
/// binding.
fn match_item(
    block: &BlockContext<'_>,
    position: usize,
    item: &RuleItem,
    state: &mut MatchState,
    config: &RewriteConfig,
) -> Result<bool, CompilerError> {
    let operation = block.operation(position);
    if config.skips_labeled_ops() && operation.label.is_some() {
        return Ok(false);
    }
    if !candidate_gate_matches(operation, item) {
        return Ok(false);
    }
    if operation.qubits.len() != item.qubits.len() {
        return Ok(false);
    }

    for (&rule_qubit, &actual_qubit) in item.qubits.iter().zip(&operation.qubits) {
        if let Some(bound) = state.qubits.get(&rule_qubit) {
            if *bound != actual_qubit {
                return Ok(false);
            }
        } else if let Some(other_rule_qubit) = state.reverse_qubits.get(&actual_qubit) {
            if *other_rule_qubit != rule_qubit {
                return Ok(false);
            }
        } else {
            state.qubits.insert(rule_qubit, actual_qubit);
            state.reverse_qubits.insert(actual_qubit, rule_qubit);
        }
    }

    let rule_params = item.params.as_deref().unwrap_or(&[]);
    if operation.params.len() != rule_params.len() {
        return Ok(false);
    }
    for (rule_param, actual) in rule_params.iter().zip(block.params(position)) {
        if !match_parameter(rule_param, actual, &mut state.params)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Matches one rule parameter pattern against one concrete operation parameter.
///
/// A bare symbol binds to the actual parameter.  A non-bare expression can only
/// match after all referenced symbols have already been bound.
fn match_parameter(
    rule_param: &ParameterValue,
    actual: &Parameter,
    bindings: &mut HashMap<String, Parameter>,
) -> Result<bool, CompilerError> {
    match rule_param {
        ParameterValue::Fixed(value) => Ok(parameters_equivalent(&Parameter::from(*value), actual)),
        ParameterValue::Param(pattern) => {
            if let Some(symbol) = single_symbol(pattern) {
                if let Some(bound) = bindings.get(&symbol) {
                    return Ok(parameters_equivalent(bound, actual));
                }
                bindings.insert(symbol, actual.clone());
                return Ok(true);
            }

            let substituted = substitute_bindings(pattern, bindings);
            if substituted.get_symbols().is_empty() {
                return Ok(parameters_equivalent(&substituted, actual));
            }

            Ok(false)
        }
    }
}

/// Returns a symbol name when a parameter is exactly one symbolic variable.
fn single_symbol(parameter: &Parameter) -> Option<String> {
    let symbols = parameter.get_symbols();
    if symbols.len() != 1 {
        return None;
    }
    let symbol = symbols.into_iter().next()?;
    let direct = Parameter::symbol(&symbol);
    if parameter == &direct {
        return Some(symbol);
    }
    parameter
        .simplify()
        .ok()
        .filter(|simplified| simplified == &direct)
        .map(|_| symbol)
}

/// Evaluates all rule conditions under the current parameter bindings.
///
/// Conditions are conjunctive.  `Eq` uses exact simplified expression equality
/// or concrete numeric equality within tolerance; `EqMod` additionally accepts
/// numeric integer multiples of the modulus.
fn conditions_hold(
    conditions: Option<&[Condition]>,
    bindings: &HashMap<String, Parameter>,
) -> Result<bool, CompilerError> {
    for condition in conditions.unwrap_or(&[]) {
        let holds = match condition {
            Condition::Eq(lhs, rhs) => {
                let lhs = substitute_bindings(lhs, bindings);
                let rhs = substitute_bindings(rhs, bindings);
                parameters_equivalent(&lhs, &rhs)
            }
            Condition::EqMod(lhs, rhs, modulus) => {
                let lhs = substitute_bindings(lhs, bindings);
                let rhs = substitute_bindings(rhs, bindings);
                let modulus = substitute_bindings(modulus, bindings);
                equivalent_modulo(&lhs, &rhs, &modulus)
            }
        };
        if !holds {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Instantiates a rule rewrite target into concrete replacement operations.
fn instantiate_target(
    target: &[RuleItem],
    state: &MatchState,
) -> Result<Vec<ReplacementItem>, CompilerError> {
    let mut replacements = Vec::with_capacity(target.len());

    for item in target {
        let qubits = item
            .qubits
            .iter()
            .map(|rule_qubit| {
                state.qubits.get(rule_qubit).copied().ok_or_else(|| {
                    CompilerError::InvariantViolation(format!(
                        "rewrite target referenced unbound qubit {rule_qubit}"
                    ))
                })
            })
            .collect::<Result<SmallVec<[_; 3]>, _>>()?;
        let params = item
            .params
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|value| instantiate_parameter_value(value, &state.params))
            .collect::<SmallVec<[_; 3]>>();

        replacements.push(ReplacementItem {
            instruction: item.instruction.clone(),
            qubits,
            params,
        });
    }

    Ok(replacements)
}

/// Instantiates one rule parameter value under matched symbol bindings.
fn instantiate_parameter_value(
    value: &ParameterValue,
    bindings: &HashMap<String, Parameter>,
) -> ParameterValue {
    let parameter = match value {
        ParameterValue::Fixed(value) => Parameter::from(*value),
        ParameterValue::Param(parameter) => substitute_bindings(parameter, bindings),
    };
    ParameterValue::from(parameter)
}

/// Replaces every bound symbol in a parameter expression and simplifies it.
fn substitute_bindings(parameter: &Parameter, bindings: &HashMap<String, Parameter>) -> Parameter {
    let mut substituted = parameter.clone();
    for (symbol, value) in bindings {
        substituted = substituted.replace(symbol, value.clone());
    }
    substituted.simplify().unwrap_or(substituted)
}

/// Resolves a circuit operation parameter into a concrete parameter expression.
pub(crate) fn resolve_operation_param(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<Parameter, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(Parameter::from(*value)),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .ok_or_else(|| {
                CompilerError::InvalidContextState(format!(
                    "invalid rewrite parameter index {}",
                    index
                ))
            }),
    }
}

/// Resolves a circuit operation parameter into a runtime parameter value.
pub(crate) fn resolve_parameter_value(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<ParameterValue, CompilerError> {
    Ok(ParameterValue::from(resolve_operation_param(
        circuit, param,
    )?))
}

/// Returns whether two parameters are provably equivalent for rewrite purposes.
fn parameters_equivalent(lhs: &Parameter, rhs: &Parameter) -> bool {
    let lhs = lhs.simplify().unwrap_or_else(|_| lhs.clone());
    let rhs = rhs.simplify().unwrap_or_else(|_| rhs.clone());
    if lhs == rhs {
        return true;
    }

    match (lhs.evaluate(&None), rhs.evaluate(&None)) {
        (Ok(lhs), Ok(rhs)) => (lhs - rhs).abs() <= PARAMETER_TOLERANCE,
        _ => false,
    }
}

/// Returns whether two parameters are equivalent modulo a third parameter.
fn equivalent_modulo(lhs: &Parameter, rhs: &Parameter, modulus: &Parameter) -> bool {
    let diff = (lhs - rhs).simplify().unwrap_or_else(|_| lhs - rhs);
    if parameters_equivalent(&diff, &Parameter::from(0.0)) {
        return true;
    }

    let Ok(diff_value) = diff.evaluate(&None) else {
        return false;
    };
    let Ok(modulus_value) = modulus.evaluate(&None) else {
        return false;
    };
    if modulus_value.abs() <= PARAMETER_TOLERANCE {
        return false;
    }

    let ratio = diff_value / modulus_value;
    (ratio - ratio.round()).abs() <= PARAMETER_TOLERANCE
}

/// Computes local cost for matched source operation positions.
fn cost_for_operation_positions(
    block: &BlockContext<'_>,
    positions: &[usize],
    target_gates: Option<&[StandardGate]>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for &position in positions {
        let operation = block.operation(position);
        if let Instruction::Standard(gate) = operation.instruction {
            let counted = cost.add_gate(
                gate,
                operation.qubits.len(),
                operation.params.len(),
                GPhaseCost::ExplicitOperation,
                target_gates,
            );
            if counted {
                update_depth_estimate(&mut cost, &mut depths, &operation.qubits);
            }
        }
    }
    cost
}

/// Computes local cost for instantiated replacement operations.
fn cost_for_replacements(
    replacements: &[ReplacementItem],
    target_gates: Option<&[StandardGate]>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for replacement in replacements {
        if let Instruction::Standard(gate) = replacement.instruction {
            let counted = cost.add_gate(
                gate,
                replacement.qubits.len(),
                replacement.params.len(),
                GPhaseCost::ImplicitReplacement,
                target_gates,
            );
            if counted {
                update_depth_estimate(&mut cost, &mut depths, &replacement.qubits);
            }
        }
    }
    cost
}

/// Updates a greedy local depth estimate for one operation.
///
/// The estimate tracks the latest depth assigned to each touched qubit and
/// places the next operation after the maximum depth of its unique qubits.
fn update_depth_estimate(
    cost: &mut LocalRewriteCost,
    depths: &mut HashMap<Qubit, usize>,
    qubits: &[Qubit],
) {
    if qubits.is_empty() {
        return;
    }

    let mut unique = SmallVec::<[Qubit; 3]>::new();
    for &qubit in qubits {
        if !unique.contains(&qubit) {
            unique.push(qubit);
        }
    }

    let next_depth = unique
        .iter()
        .filter_map(|qubit| depths.get(qubit))
        .max()
        .copied()
        .unwrap_or(0)
        + 1;
    for qubit in unique {
        depths.insert(qubit, next_depth);
    }
    cost.depth_estimate = cost.depth_estimate.max(next_depth);
}
