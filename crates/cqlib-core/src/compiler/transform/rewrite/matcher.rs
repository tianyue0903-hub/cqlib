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
//! raw knowledge-base rules into a first-instruction index, scans one rewrite-safe
//! block at a time, and returns a non-overlapping set of rewrite patches.  It is
//! dependency-aware rather than purely adjacent: a later pattern item may be
//! matched across intervening operations when the commutation oracle proves that
//! the skipped operations can safely move around both the matched source and the
//! instantiated replacement.

use crate::circuit::{
    Circuit, CircuitParam, Instruction, Operation, Parameter, ParameterValue, Qubit,
};
use crate::compiler::commutation::{CommutationChecker, CommutationConfig};
use crate::compiler::error::CompilerError;
use crate::compiler::knowledge::library::{RuleKind, RuleLibrary};
pub(crate) use crate::compiler::knowledge::matcher::KnowledgeInstructionKey as RewriteInstructionKey;
use crate::compiler::knowledge::matcher::{
    ConcreteOperationView, MatchBindings, conditions_hold as knowledge_conditions_hold,
    instantiate_target as knowledge_instantiate_target,
    match_rule_item as knowledge_match_rule_item,
};
use crate::compiler::knowledge::rule::{Rule, RuleItem};
use crate::compiler::transform::rewrite::config::{GPhaseCost, LocalRewriteCost, RewriteConfig};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::target::TargetContext;

/// A rewrite rule prepared for repeated matching.
///
/// The compiled form caches static metadata used by hot matching paths:
/// rule kind, pattern size, touched rule-local qubit count, and instruction
/// keys used for target-basis filtering.
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
    /// Match-side instruction key for each source rule item.
    source_keys: SmallVec<[RewriteInstructionKey; 8]>,
    /// Distinct rewrite-safe instructions appearing in the match side.
    match_keys: SmallVec<[RewriteInstructionKey; 4]>,
    /// Distinct rewrite-safe instructions appearing in the rewrite side.
    rewrite_keys: SmallVec<[RewriteInstructionKey; 4]>,
    /// The validated runtime rule.
    rule: Rule,
}

/// Compiled rule collection with a first-instruction candidate index.
///
/// `first_key_map` keeps candidate lookup cheap for each anchor operation.
/// Commutation rules are not applied as ordinary rewrite patches; the compiled
/// checker is used only to justify skipped operations during non-adjacent
/// matching.
pub(crate) struct CompiledRuleSet {
    rules: Vec<CompiledRule>,
    first_key_map: HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>>,
    commutation: CommutationChecker,
}

/// Read-only rule summary used by target-basis lowerability analysis.
pub(super) struct LowerableRuleView<'a> {
    pub(super) kind: RuleKind,
    pub(super) source_keys: &'a [RewriteInstructionKey],
    pub(super) rewrite_keys: &'a [RewriteInstructionKey],
    pub(super) has_conditions: bool,
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
    key: RewriteInstructionKey,
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

/// Candidate patch plus before/after cost used by greedy selection.
#[derive(Clone)]
struct CandidatePatch {
    patch: RewritePatch,
    before: LocalRewriteCost,
    after: LocalRewriteCost,
}

/// Preprocessed view of one contiguous rewrite-safe operation block.
///
/// Circuit parameter indices are resolved once up front so every candidate rule
/// sees parameters in value form.  The instruction set and qubit count serve as
/// cheap static filters before the expensive matcher runs.
struct BlockContext<'a> {
    operations: &'a [Operation],
    instruction_keys: Vec<RewriteInstructionKey>,
    resolved_params: Vec<SmallVec<[Parameter; 3]>>,
    instruction_set: HashSet<RewriteInstructionKey>,
    qubit_count: usize,
}

impl<'a> BlockContext<'a> {
    /// Builds a block context and resolves all operation parameters.
    fn new(circuit: &'a Circuit, operations: &'a [Operation]) -> Result<Self, CompilerError> {
        let mut resolved_params = Vec::with_capacity(operations.len());
        let mut instruction_keys = Vec::with_capacity(operations.len());
        let mut touched_qubits = HashSet::new();
        let mut instruction_set = HashSet::new();

        for operation in operations {
            let key = RewriteInstructionKey::from_instruction(&operation.instruction).ok_or_else(
                || {
                    CompilerError::InvariantViolation(format!(
                        "rewrite block contains unsupported instruction {:?}",
                        operation.instruction
                    ))
                },
            )?;
            let params = operation
                .params
                .iter()
                .map(|param| resolve_operation_param(circuit, param))
                .collect::<Result<SmallVec<[_; 3]>, _>>()?;
            resolved_params.push(params);

            for &qubit in &operation.qubits {
                touched_qubits.insert(qubit);
            }
            instruction_set.insert(key.clone());
            instruction_keys.push(key);
        }

        Ok(Self {
            operations,
            instruction_keys,
            resolved_params,
            instruction_set,
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

    /// Returns the rewrite key for the operation at a block-local position.
    fn key(&self, position: usize) -> &RewriteInstructionKey {
        &self.instruction_keys[position]
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
        let mut first_key_map: HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>> =
            HashMap::new();
        let kind_by_id = build_kind_index(library);
        let commutation = CommutationChecker::from_library(library, rewrite_commutation_config());

        for (index, rule) in library.rules().iter().cloned().enumerate() {
            let kind = kind_by_id.get(&index).copied().unwrap_or(RuleKind::Other);
            push_compiled_rule(&mut rules, &mut first_key_map, index, kind, rule)?;
        }

        Ok(Self {
            rules,
            first_key_map,
            commutation,
        })
    }

    /// Returns candidate compiled-rule indices for an anchor instruction.
    fn candidates_for_first_instruction(&self, key: &RewriteInstructionKey) -> &[usize] {
        self.first_key_map
            .get(key)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    /// Returns a compiled rule by compiled-rule index.
    fn get(&self, index: usize) -> &CompiledRule {
        &self.rules[index]
    }

    pub(super) fn lowerable_rule_views(&self) -> impl Iterator<Item = LowerableRuleView<'_>> {
        self.rules.iter().map(|rule| LowerableRuleView {
            kind: rule.kind,
            source_keys: &rule.source_keys,
            rewrite_keys: &rule.rewrite_keys,
            has_conditions: rule
                .rule
                .conditions
                .as_ref()
                .is_some_and(|conditions| !conditions.is_empty()),
        })
    }
}

fn rewrite_commutation_config() -> CommutationConfig {
    CommutationConfig {
        enable_rule_oracle: true,
        enable_matrix_fallback: false,
        max_matrix_qubits: 0,
    }
}

/// Adds one validated runtime rule to the compiled rule set under construction.
///
/// The rule is indexed by its first match instruction and stores the original
/// library id so selected patches can report the source rule.
fn push_compiled_rule(
    rules: &mut Vec<CompiledRule>,
    first_key_map: &mut HashMap<RewriteInstructionKey, SmallVec<[usize; 8]>>,
    id: usize,
    kind: RuleKind,
    rule: Rule,
) -> Result<(), CompilerError> {
    if rule.operations.is_empty() {
        return Err(CompilerError::InvariantViolation(
            "rewrite rule contains an empty match block".to_string(),
        ));
    }
    let match_len = rule.operations.len();
    let rewrite_len = rule.target.len();
    let mut rule_qubits = HashSet::new();
    for item in rule.operations.iter().chain(&rule.target) {
        rule_qubits.extend(item.qubits.iter().copied());
    }
    let qubit_count = rule_qubits.len();
    let mut source_keys = SmallVec::<[RewriteInstructionKey; 8]>::new();
    let mut match_keys = SmallVec::<[RewriteInstructionKey; 4]>::new();
    let mut rewrite_keys = SmallVec::<[RewriteInstructionKey; 4]>::new();
    for item in &rule.operations {
        let key = RewriteInstructionKey::from_instruction(&item.instruction).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "rewrite rule contains unsupported instruction {:?}",
                item.instruction
            ))
        })?;
        if !match_keys.contains(&key) {
            match_keys.push(key.clone());
        }
        source_keys.push(key);
    }
    for item in &rule.target {
        let key = RewriteInstructionKey::from_instruction(&item.instruction).ok_or_else(|| {
            CompilerError::InvariantViolation(format!(
                "rewrite rule contains unsupported instruction {:?}",
                item.instruction
            ))
        })?;
        if !rewrite_keys.contains(&key) {
            rewrite_keys.push(key.clone());
        }
    }
    first_key_map
        .entry(source_keys[0].clone())
        .or_default()
        .push(rules.len());
    rules.push(CompiledRule {
        id,
        kind,
        match_len,
        qubit_count,
        static_cost_delta: rewrite_len as isize - match_len as isize,
        source_keys,
        match_keys,
        rewrite_keys,
        rule,
    });
    Ok(())
}

/// Selects a non-overlapping set of rewrite patches for one operation block.
///
/// Candidate generation is independent for every anchor position.  After
/// matching, candidates are sorted by local replacement cost, not by maximum
/// before/after improvement, and selected greedily while rejecting any candidate
/// whose matched source positions overlap a previously selected patch.
#[cfg(test)]
pub(crate) fn select_rewrites(
    circuit: &Circuit,
    operations: &[Operation],
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
) -> Result<SelectedRewrites, CompilerError> {
    let target_context = TargetContext::from_config(config, rules)?;
    select_rewrites_in_context(circuit, operations, rules, config, target_context.as_ref())
}

pub(super) fn select_rewrites_in_context(
    circuit: &Circuit,
    operations: &[Operation],
    rules: &CompiledRuleSet,
    config: &RewriteConfig,
    target_context: Option<&TargetContext>,
) -> Result<SelectedRewrites, CompilerError> {
    let block = BlockContext::new(circuit, operations)?;
    let mut candidates = Vec::new();

    for anchor in 0..block.len() {
        let operation = block.operation(anchor);
        if config.skips_labeled_ops() && operation.label.is_some() {
            continue;
        }
        let first_key = block.key(anchor);

        // Use the first-instruction index to avoid considering rules that cannot
        // start at this anchor operation.
        for &rule_index in rules.candidates_for_first_instruction(first_key) {
            let compiled = rules.get(rule_index);
            if !rule_passes_static_filters(compiled, config, &block, target_context) {
                continue;
            }
            if let Some(candidate) = try_match_rule(
                &block,
                anchor,
                compiled,
                &rules.commutation,
                config,
                target_context,
            )? {
                candidates.push(candidate);
            }
        }
    }

    // Prefer candidates that produce the lowest replacement local cost, then
    // choose stable deterministic tie-breakers so repeated runs produce
    // identical patches.
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
                    .static_cost_delta
                    .cmp(&rhs.patch.static_cost_delta)
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

/// Applies cheap rule filters that do not require pattern matching.
fn rule_passes_static_filters(
    rule: &CompiledRule,
    config: &RewriteConfig,
    block: &BlockContext<'_>,
    target_context: Option<&TargetContext>,
) -> bool {
    rule.kind != RuleKind::Commute
        && config.allows_kind(rule.kind)
        && rule.match_len <= config.max_pattern_len()
        && rule.qubit_count <= block.qubit_count
        && rule_passes_target_filter(rule, target_context, block)
}

/// Applies target-basis filtering for a compiled rule.
///
/// Match instructions must be present in the current block.  Rewrite
/// instructions may be physical targets or lowerable intermediates, except
/// replacement `GPhase` which is allowed implicitly because sequence emission
/// handles it as phase metadata or discards it in control-flow bodies.
fn rule_passes_target_filter(
    rule: &CompiledRule,
    target_context: Option<&TargetContext>,
    block: &BlockContext<'_>,
) -> bool {
    let Some(target_context) = target_context else {
        return true;
    };

    rule.match_keys
        .iter()
        .all(|key| block.instruction_set.contains(key))
        && rule
            .rewrite_keys
            .iter()
            .all(|key| target_context.allows_rewrite_key(key))
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
    commutation: &CommutationChecker,
    config: &RewriteConfig,
    target_context: Option<&TargetContext>,
) -> Result<Option<CandidatePatch>, CompilerError> {
    let rule = &compiled.rule;
    let mut bindings = MatchBindings::new();

    // Step 1: bind the first rule item to the anchor operation.
    if !match_item(
        block,
        anchor,
        &rule.operations[0],
        &compiled.source_keys[0],
        &mut bindings,
        config,
    )? {
        return Ok(None);
    }

    let mut matched_positions = vec![anchor];
    let mut skipped_positions = Vec::new();
    let mut cursor = anchor + 1;
    for (item, item_key) in rule.operations.iter().zip(&compiled.source_keys).skip(1) {
        let mut found = None;
        let limit = block.len().min(cursor + config.max_window_ops());

        // Step 2: scan forward for the next pattern item within the window.
        for position in cursor..limit {
            if block.key(position) != item_key {
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

            let mut next_bindings = bindings.clone();
            if match_item(block, position, item, item_key, &mut next_bindings, config)? {
                found = Some((position, next_bindings));
                break;
            }
        }

        let Some((position, next_bindings)) = found else {
            return Ok(None);
        };
        bindings = next_bindings;
        skipped_positions.extend(cursor..position);
        matched_positions.push(position);
        cursor = position + 1;
    }

    // Step 4: apply symbolic/numeric rule conditions after all parameters have
    // been bound by the structural match.
    if !knowledge_conditions_hold(rule.conditions.as_deref(), &bindings) {
        return Ok(None);
    }

    // Step 5: each skipped operation is emitted after the replacement, so it must
    // commute with every later matched source operation that it crosses.
    if !skipped_sources_commute_with_future_matches(
        block,
        &skipped_positions,
        &matched_positions,
        commutation,
        config,
    ) {
        return Ok(None);
    }

    // Step 6: instantiate the rewrite target using the matched qubit and
    // parameter bindings.
    let replacements = knowledge_instantiate_target(&rule.target, &bindings)
        .map_err(|error| CompilerError::InvariantViolation(error.to_string()))?
        .into_iter()
        .map(|item| ReplacementItem {
            instruction: item.instruction,
            qubits: item.qubits,
            params: item.params,
            key: item.key,
        })
        .collect::<Vec<_>>();
    // Step 7: replacements must also commute with skipped operations; otherwise
    // emitting them at the first matched position would change behavior.
    if !replacements_commute_with_skipped(block, &skipped_positions, &replacements, commutation)? {
        return Ok(None);
    }

    // Step 8: accept only rewrites permitted by the configured local cost model.
    let before = cost_for_operation_positions(block, &matched_positions, target_context);
    let after = cost_for_replacements(&replacements, target_context);
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
    commutation: &CommutationChecker,
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
            if !operations_commute(block, skipped_position, matched_position, commutation) {
                return Ok(false);
            }
        }
        if !operations_commute(block, skipped_position, candidate_position, commutation) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Rechecks skipped source operations against future matched source operations.
///
/// Incremental matching only proves that a skipped operation can reach the next
/// matched item.  For patterns with three or more operations, the same skipped
/// operation may later need to cross additional matched items before the whole
/// source block can be replaced at `first_position`.
fn skipped_sources_commute_with_future_matches(
    block: &BlockContext<'_>,
    skipped_positions: &[usize],
    matched_positions: &[usize],
    commutation: &CommutationChecker,
    config: &RewriteConfig,
) -> bool {
    for &skipped_position in skipped_positions {
        let skipped_operation = block.operation(skipped_position);
        for &matched_position in matched_positions {
            if matched_position <= skipped_position
                || !skipped_operation
                    .qubits
                    .iter()
                    .any(|qubit| block.operation(matched_position).qubits.contains(qubit))
            {
                continue;
            }
            if config.skips_labeled_ops() && skipped_operation.label.is_some() {
                return false;
            }
            if !operations_commute(block, skipped_position, matched_position, commutation) {
                return false;
            }
        }
    }

    true
}

/// Returns whether instantiated replacements may be emitted before skipped ops.
fn replacements_commute_with_skipped(
    block: &BlockContext<'_>,
    skipped_positions: &[usize],
    replacements: &[ReplacementItem],
    commutation: &CommutationChecker,
) -> Result<bool, CompilerError> {
    if skipped_positions.is_empty() || replacements.is_empty() {
        return Ok(true);
    }

    for &skipped_position in skipped_positions {
        for replacement in replacements {
            let skipped_qubits = &block.operation(skipped_position).qubits;
            if !skipped_qubits
                .iter()
                .any(|qubit| replacement.qubits.contains(qubit))
            {
                continue;
            }
            if !operation_commutes_with_replacement(
                block,
                skipped_position,
                replacement,
                commutation,
            ) {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Returns whether two block operations commute exactly.
fn operations_commute(
    block: &BlockContext<'_>,
    lhs_position: usize,
    rhs_position: usize,
    commutation: &CommutationChecker,
) -> bool {
    let lhs = block.operation(lhs_position);
    let rhs = block.operation(rhs_position);
    commutation
        .check(
            &lhs.instruction,
            &lhs.qubits,
            block.params(lhs_position),
            &rhs.instruction,
            &rhs.qubits,
            block.params(rhs_position),
        )
        .is_some_and(|result| result.is_exact())
}

/// Returns whether a source operation commutes exactly with a replacement.
fn operation_commutes_with_replacement(
    block: &BlockContext<'_>,
    operation_position: usize,
    replacement: &ReplacementItem,
    commutation: &CommutationChecker,
) -> bool {
    let operation = block.operation(operation_position);
    let replacement_params = replacement
        .params
        .iter()
        .map(|value| match value {
            ParameterValue::Fixed(value) => Parameter::from(*value),
            ParameterValue::Param(parameter) => parameter.clone(),
        })
        .collect::<SmallVec<[_; 3]>>();

    commutation
        .check(
            &operation.instruction,
            &operation.qubits,
            block.params(operation_position),
            &replacement.instruction,
            &replacement.qubits,
            &replacement_params,
        )
        .is_some_and(|result| result.is_exact())
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
    item_key: &RewriteInstructionKey,
    bindings: &mut MatchBindings,
    config: &RewriteConfig,
) -> Result<bool, CompilerError> {
    let operation = block.operation(position);
    if config.skips_labeled_ops() && operation.label.is_some() {
        return Ok(false);
    }
    if block.key(position) != item_key {
        return Ok(false);
    }

    knowledge_match_rule_item(
        item,
        ConcreteOperationView {
            instruction: &operation.instruction,
            qubits: &operation.qubits,
            params: block.params(position),
        },
        bindings,
    )
    .map_err(|error| CompilerError::InvariantViolation(error.to_string()))
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

/// Computes local cost for matched source operation positions.
fn cost_for_operation_positions(
    block: &BlockContext<'_>,
    positions: &[usize],
    target_context: Option<&TargetContext>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for &position in positions {
        let operation = block.operation(position);
        add_instruction_cost(
            &mut cost,
            &mut depths,
            block.key(position),
            &operation.qubits,
            operation.params.len(),
            GPhaseCost::ExplicitOperation,
            target_context,
        );
    }
    cost
}

/// Computes local cost for instantiated replacement operations.
fn cost_for_replacements(
    replacements: &[ReplacementItem],
    target_context: Option<&TargetContext>,
) -> LocalRewriteCost {
    let mut cost = LocalRewriteCost::default();
    let mut depths = HashMap::new();

    for replacement in replacements {
        add_instruction_cost(
            &mut cost,
            &mut depths,
            &replacement.key,
            &replacement.qubits,
            replacement.params.len(),
            GPhaseCost::ImplicitReplacement,
            target_context,
        );
    }
    cost
}

/// Adds one rewrite-safe instruction to the local cost tuple and depth estimate.
fn add_instruction_cost(
    cost: &mut LocalRewriteCost,
    depths: &mut HashMap<Qubit, usize>,
    key: &RewriteInstructionKey,
    qubits: &[Qubit],
    param_count: usize,
    gphase_cost: GPhaseCost,
    target_context: Option<&TargetContext>,
) {
    let target_supported = match target_context {
        Some(target_context) => target_context.physically_supports(key),
        None => true,
    };
    let standard_gate = match key {
        RewriteInstructionKey::Standard(gate) => Some(*gate),
        RewriteInstructionKey::McGate(_) => None,
    };
    let counted = cost.add_gate_like(
        standard_gate,
        target_supported,
        qubits.len(),
        param_count,
        gphase_cost,
    );
    if counted {
        if let Some(target_context) = target_context {
            cost.lowering_distance = cost
                .lowering_distance
                .saturating_add(target_context.lowering_distance(key));
        }
        update_depth_estimate(cost, depths, qubits);
    }
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
