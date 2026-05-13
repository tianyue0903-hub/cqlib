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

//! Rule library and indexes for compiler knowledge-base rules.
//!
//! A [`RuleLibrary`] owns validated rules, assigns stable [`RuleId`] values,
//! and precomputes the metadata and first-gate index needed by matchers. DSL
//! parsing remains in `rule_dsl`; this module only delegates to it.

use crate::circuit::{Instruction, StandardGate};
use crate::compiler::knowledge::rule::{Rule, RuleItem, RuleValidationError};
use crate::compiler::knowledge::rule_dsl::load::{
    LoadError, load_rules_from_file, load_rules_from_str,
};
use smallvec::SmallVec;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;

static BUILTIN_RULES: OnceLock<Result<RuleLibrary, RuleLibraryError>> = OnceLock::new();

/// Stable identifier assigned to a rule when it is inserted into a library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleId(usize);

impl RuleId {
    /// Returns the rule's current library-local index.
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// Coarse compiler use-case for a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleKind {
    Simplify,
    Cancel,
    Merge,
    Commute,
    Decompose,
    Canonicalize,
    HardwareNative,
    Other,
}

/// Precomputed metadata used by rule selection and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleMetadata {
    pub id: RuleId,
    pub kind: RuleKind,
    pub pattern_len: usize,
    pub rewrite_len: usize,
    pub qubit_count: usize,
    pub first_gate: StandardGate,
    pub cost_delta: isize,
    pub has_conditions: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GateMask(u64);

impl GateMask {
    fn from_gates(gates: &[StandardGate]) -> Self {
        let mut mask = Self::default();
        for &gate in gates {
            mask.insert(gate);
        }
        mask
    }

    fn from_rule_items(items: &[RuleItem]) -> Self {
        let mut mask = Self::default();
        for item in items {
            let gate = match &item.instruction {
                Instruction::Standard(gate) => *gate,
                other => unreachable!("validated rule contains unsupported instruction: {other:?}"),
            };
            mask.insert(gate);
        }
        mask
    }

    fn insert(&mut self, gate: StandardGate) {
        let bit = 1u64
            .checked_shl(gate as u32)
            .expect("standard gate discriminant must fit in GateMask");
        self.0 |= bit;
    }

    fn contains_all(self, required: Self) -> bool {
        required.0 & !self.0 == 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RuleGateMetadata {
    match_gate_mask: GateMask,
    rewrite_gate_mask: GateMask,
}

/// Errors produced while building or extending a rule library.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RuleLibraryError {
    #[error("invalid rule {name}: {source}")]
    InvalidRule {
        name: String,
        source: RuleValidationError,
    },
    #[error("duplicate rule name: {0}")]
    DuplicateRuleName(String),
    #[error("failed to load rules: {0}")]
    Load(LoadError),
}

/// Validated rule collection with indexes for matcher candidate lookup.
#[derive(Debug, Clone, Default)]
pub struct RuleLibrary {
    rules: Vec<Rule>,
    metadata: Vec<RuleMetadata>,
    gate_metadata: Vec<RuleGateMetadata>,
    name_map: HashMap<String, RuleId>,
    first_gate_map: HashMap<StandardGate, SmallVec<[RuleId; 8]>>,
    kind_map: HashMap<RuleKind, SmallVec<[RuleId; 8]>>,
}

impl RuleLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn builtin_rules() -> Result<&'static RuleLibrary, RuleLibraryError> {
        match BUILTIN_RULES.get_or_init(load_builtin_rules) {
            Ok(library) => Ok(library),
            Err(err) => Err(err.clone()),
        }
    }

    pub fn from_rules(rules: Vec<Rule>, kind: RuleKind) -> Result<Self, RuleLibraryError> {
        let mut library = Self::new();
        library.extend_rules(rules, kind)?;
        Ok(library)
    }

    pub fn from_dsl_str(source: &str, kind: RuleKind) -> Result<Self, RuleLibraryError> {
        Self::from_rules(
            load_rules_from_str(source).map_err(RuleLibraryError::Load)?,
            kind,
        )
    }

    pub fn from_dsl_file(path: impl AsRef<Path>, kind: RuleKind) -> Result<Self, RuleLibraryError> {
        Self::from_rules(
            load_rules_from_file(path).map_err(RuleLibraryError::Load)?,
            kind,
        )
    }

    pub fn add_rule(
        &mut self,
        rule: Rule,
        kind: RuleKind,
        validate: bool,
    ) -> Result<RuleId, RuleLibraryError> {
        if validate {
            rule.validate()
                .map_err(|source| RuleLibraryError::InvalidRule {
                    name: rule.name.clone(),
                    source,
                })?;
        }

        if self.name_map.contains_key(&rule.name) {
            return Err(RuleLibraryError::DuplicateRuleName(rule.name));
        }

        let id = RuleId(self.rules.len());
        let metadata = build_metadata(id, kind, &rule);
        let gate_metadata = build_gate_metadata(&rule);

        self.name_map.insert(rule.name.clone(), id);
        self.first_gate_map
            .entry(metadata.first_gate)
            .or_default()
            .push(id);
        self.kind_map.entry(kind).or_default().push(id);
        self.metadata.push(metadata);
        self.gate_metadata.push(gate_metadata);
        self.rules.push(rule);

        Ok(id)
    }

    pub fn extend_rules(
        &mut self,
        rules: Vec<Rule>,
        kind: RuleKind,
    ) -> Result<SmallVec<[RuleId; 8]>, RuleLibraryError> {
        self.extend_rules_with_validation(rules, kind, true)
    }

    fn extend_rules_with_validation(
        &mut self,
        rules: Vec<Rule>,
        kind: RuleKind,
        validate: bool,
    ) -> Result<SmallVec<[RuleId; 8]>, RuleLibraryError> {
        let mut updated = self.clone();
        let mut ids = SmallVec::new();

        for rule in rules {
            ids.push(updated.add_rule(rule, kind, validate)?);
        }

        *self = updated;
        Ok(ids)
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    pub fn get(&self, id: RuleId) -> Option<&Rule> {
        self.rules.get(id.0)
    }

    pub fn metadata(&self, id: RuleId) -> Option<&RuleMetadata> {
        self.metadata.get(id.0)
    }

    pub fn id_by_name(&self, name: &str) -> Option<RuleId> {
        self.name_map.get(name).copied()
    }

    pub fn get_by_name(&self, name: &str) -> Option<&Rule> {
        self.id_by_name(name).and_then(|id| self.get(id))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.name_map.contains_key(name)
    }

    pub fn candidates_for_first_gate(&self, gate: StandardGate) -> &[RuleId] {
        self.first_gate_map
            .get(&gate)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    pub fn rules_by_kind(&self, kind: RuleKind) -> &[RuleId] {
        self.kind_map
            .get(&kind)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
    }

    pub fn filter_rule_ids_by_gates(
        &self,
        op_gates: &[StandardGate],
        target_gates: &[StandardGate],
    ) -> SmallVec<[RuleId; 16]> {
        let op_mask = GateMask::from_gates(op_gates);
        let target_mask = GateMask::from_gates(target_gates);
        let mut ids = SmallVec::new();

        for (index, metadata) in self.gate_metadata.iter().enumerate() {
            if op_mask.contains_all(metadata.match_gate_mask)
                && target_mask.contains_all(metadata.rewrite_gate_mask)
            {
                ids.push(RuleId(index));
            }
        }

        ids
    }
}

fn build_metadata(id: RuleId, kind: RuleKind, rule: &Rule) -> RuleMetadata {
    let first_gate = match &rule.operations[0].instruction {
        Instruction::Standard(gate) => *gate,
        other => unreachable!("validated rule contains unsupported instruction: {other:?}"),
    };
    let pattern_len = rule.operations.len();
    let rewrite_len = rule.target.len();

    RuleMetadata {
        id,
        kind,
        pattern_len,
        rewrite_len,
        qubit_count: qubit_count(rule),
        first_gate,
        cost_delta: rewrite_len as isize - pattern_len as isize,
        has_conditions: rule
            .conditions
            .as_ref()
            .is_some_and(|conditions| !conditions.is_empty()),
    }
}

fn build_gate_metadata(rule: &Rule) -> RuleGateMetadata {
    RuleGateMetadata {
        match_gate_mask: GateMask::from_rule_items(&rule.operations),
        rewrite_gate_mask: GateMask::from_rule_items(&rule.target),
    }
}

fn qubit_count(rule: &Rule) -> usize {
    rule.operations
        .iter()
        .chain(&rule.target)
        .flat_map(|item| item.qubits.iter().copied())
        .collect::<BTreeSet<_>>()
        .len()
}

fn load_builtin_rules() -> Result<RuleLibrary, RuleLibraryError> {
    let mut library = RuleLibrary::new();

    for (source, kind) in BUILTIN_RULE_SOURCES {
        let rules = load_rules_from_str(source).map_err(RuleLibraryError::Load)?;
        library.extend_rules_with_validation(rules, *kind, false)?;
    }

    Ok(library)
}

const BUILTIN_RULE_SOURCES: &[(&str, RuleKind)] = &[
    (include_str!("rules/normalize.rule"), RuleKind::Canonicalize),
    (include_str!("rules/cancel.rule"), RuleKind::Cancel),
    (include_str!("rules/merge.rule"), RuleKind::Merge),
    (include_str!("rules/identity.rule"), RuleKind::Simplify),
    (include_str!("rules/specialize.rule"), RuleKind::Simplify),
    (include_str!("rules/commutation.rule"), RuleKind::Commute),
    (
        include_str!("rules/decompose_ccx.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_controlled_pauli.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_controlled_rotation.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_fsim.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_ising.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_phase.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_qcis.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_single_clifford.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_single_rotation.rule"),
        RuleKind::Decompose,
    ),
    (
        include_str!("rules/decompose_swap.rule"),
        RuleKind::Decompose,
    ),
];

#[cfg(test)]
#[path = "./library_test.rs"]
mod library_test;
