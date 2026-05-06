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

//! Runtime model for knowledge-base rewrite rules.
//!
//! This module describes what a rule is.  Rule equivalence checking lives in
//! `rule_equivalence` because matrix construction, parameter sampling, and
//! numerical comparison are validation strategies rather than rule structure.

use crate::circuit::{Instruction, Parameter, ParameterValue, StandardGate};
use smallvec::SmallVec;
use std::collections::{BTreeSet, HashSet};

#[derive(Debug, Clone)]
pub struct RuleItem {
    pub instruction: Instruction,
    pub qubits: SmallVec<[u32; 3]>,
    pub params: Option<SmallVec<[ParameterValue; 1]>>,
}

impl RuleItem {
    pub fn standard(gate: StandardGate, qubits: &[u32], params: Vec<ParameterValue>) -> Self {
        Self {
            instruction: Instruction::Standard(gate),
            qubits: SmallVec::from_slice(qubits),
            params: if params.is_empty() {
                None
            } else {
                Some(SmallVec::from_vec(params))
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum Condition {
    Eq(Parameter, Parameter),
    EqMod(Parameter, Parameter, Parameter), // lhs == rhs mod modulus
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub operations: SmallVec<[RuleItem; 4]>,
    pub conditions: Option<SmallVec<[Condition; 2]>>,
    pub target: SmallVec<[RuleItem; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleItemContext {
    Match,
    Rewrite,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuleValidationError {
    #[error("rule match block is empty")]
    EmptyMatch,
    #[error("unsupported {context:?} instruction: {instruction}")]
    UnsupportedInstruction {
        context: RuleItemContext,
        instruction: String,
    },
    #[error("wrong {context:?} qubit count for {gate:?}: expected {expected}, got {got}")]
    WrongQubitCount {
        context: RuleItemContext,
        gate: StandardGate,
        expected: usize,
        got: usize,
    },
    #[error("wrong {context:?} parameter count for {gate:?}: expected {expected}, got {got}")]
    WrongParamCount {
        context: RuleItemContext,
        gate: StandardGate,
        expected: usize,
        got: usize,
    },
    #[error("duplicate {context:?} qubit {qubit} in {gate:?}")]
    DuplicateQubit {
        context: RuleItemContext,
        gate: StandardGate,
        qubit: u32,
    },
    #[error("rewrite qubit {qubit} is not bound by the match block")]
    UnboundRewriteQubit { qubit: u32 },
    #[error("rewrite symbol {symbol} is not bound by the match block")]
    UnboundRewriteSymbol { symbol: String },
    #[error("condition symbol {symbol} is not bound by the match block")]
    UnboundConditionSymbol { symbol: String },
    #[error("qubit labels must be dense from 0, found {labels:?}")]
    NonDenseQubitLabels { labels: Vec<u32> },
}

impl Rule {
    /// Helper: build a simple Rule with no conditions.
    pub fn new(name: &str, ops: Vec<RuleItem>, target: Vec<RuleItem>) -> Rule {
        Rule {
            name: name.to_string(),
            operations: SmallVec::from_vec(ops),
            conditions: None,
            target: SmallVec::from_vec(target),
        }
    }

    /// Validate structural invariants required by the rule runtime.
    ///
    /// This checks whether a rule is well-formed enough to be dumped, verified,
    /// and eventually matched. It does not prove that the rewrite preserves
    /// quantum semantics; matrix equivalence checking lives in
    /// `rule_equivalence`.
    pub fn validate(&self) -> Result<(), RuleValidationError> {
        if self.operations.is_empty() {
            return Err(RuleValidationError::EmptyMatch);
        }

        for item in &self.operations {
            validate_rule_item(item, RuleItemContext::Match)?;
        }
        for item in &self.target {
            validate_rule_item(item, RuleItemContext::Rewrite)?;
        }

        let match_qubits = collect_qubits(&self.operations);
        validate_dense_qubit_labels(&match_qubits)?;

        for item in &self.target {
            for &qubit in &item.qubits {
                if !match_qubits.contains(&qubit) {
                    return Err(RuleValidationError::UnboundRewriteQubit { qubit });
                }
            }
        }

        let bound_symbols = collect_item_symbols(&self.operations);
        for symbol in collect_item_symbols(&self.target) {
            if !bound_symbols.contains(&symbol) {
                return Err(RuleValidationError::UnboundRewriteSymbol { symbol });
            }
        }

        if let Some(conditions) = &self.conditions {
            for symbol in collect_condition_symbols(conditions) {
                if !bound_symbols.contains(&symbol) {
                    return Err(RuleValidationError::UnboundConditionSymbol { symbol });
                }
            }
        }

        Ok(())
    }

    /// Determine the number of qubits from the rule's operations and target.
    pub(crate) fn num_qubits(&self) -> usize {
        let mut op_qubits: HashSet<u32> = self
            .operations
            .iter()
            .flat_map(|op| op.qubits.iter().copied())
            .collect();
        let max_qubit_target: HashSet<u32> = self
            .target
            .iter()
            .flat_map(|op| op.qubits.iter().copied())
            .collect();
        op_qubits.extend(max_qubit_target);
        // Minimum 1 for GPhase-only rules.
        op_qubits.len().max(1)
    }

    /// Collect all free symbol names from parameters and conditions.
    pub(crate) fn collect_free_symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::new();

        for op in self.operations.iter().chain(&self.target) {
            if let Some(params) = &op.params {
                for p in params {
                    if let ParameterValue::Param(p) = p {
                        symbols.extend(p.get_symbols());
                    }
                }
            }
        }

        if let Some(conditions) = &self.conditions {
            for cond in conditions {
                match cond {
                    Condition::Eq(a, b) => {
                        symbols.extend(a.get_symbols());
                        symbols.extend(b.get_symbols());
                    }
                    Condition::EqMod(a, b, m) => {
                        symbols.extend(a.get_symbols());
                        symbols.extend(b.get_symbols());
                        symbols.extend(m.get_symbols());
                    }
                }
            }
        }

        // Remove built-in constants (symb_anafis stores π as "pi" internally).
        symbols.remove("π");
        symbols.remove("pi");
        symbols.remove("e");
        symbols
    }
}

fn validate_rule_item(
    item: &RuleItem,
    context: RuleItemContext,
) -> Result<(), RuleValidationError> {
    let gate = match &item.instruction {
        Instruction::Standard(gate) => *gate,
        other => {
            return Err(RuleValidationError::UnsupportedInstruction {
                context,
                instruction: format!("{other:?}"),
            });
        }
    };

    let expected_qubits = gate.num_qubits();
    if item.qubits.len() != expected_qubits {
        return Err(RuleValidationError::WrongQubitCount {
            context,
            gate,
            expected: expected_qubits,
            got: item.qubits.len(),
        });
    }

    let expected_params = gate.num_params();
    let got_params = item.params.as_ref().map_or(0, SmallVec::len);
    if got_params != expected_params {
        return Err(RuleValidationError::WrongParamCount {
            context,
            gate,
            expected: expected_params,
            got: got_params,
        });
    }

    let mut seen = SmallVec::<[u32; 3]>::new();
    for &qubit in &item.qubits {
        if seen.contains(&qubit) {
            return Err(RuleValidationError::DuplicateQubit {
                context,
                gate,
                qubit,
            });
        }
        seen.push(qubit);
    }

    Ok(())
}

fn collect_qubits(items: &[RuleItem]) -> BTreeSet<u32> {
    items
        .iter()
        .flat_map(|item| item.qubits.iter().copied())
        .collect()
}

fn validate_dense_qubit_labels(labels: &BTreeSet<u32>) -> Result<(), RuleValidationError> {
    for (expected, actual) in labels.iter().copied().enumerate() {
        if expected as u32 != actual {
            return Err(RuleValidationError::NonDenseQubitLabels {
                labels: labels.iter().copied().collect(),
            });
        }
    }
    Ok(())
}

fn collect_item_symbols(items: &[RuleItem]) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for item in items {
        if let Some(params) = &item.params {
            for param in params {
                if let ParameterValue::Param(param) = param {
                    symbols.extend(param.get_symbols());
                }
            }
        }
    }
    remove_builtin_symbols(&mut symbols);
    symbols
}

fn collect_condition_symbols(conditions: &[Condition]) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for condition in conditions {
        match condition {
            Condition::Eq(lhs, rhs) => {
                symbols.extend(lhs.get_symbols());
                symbols.extend(rhs.get_symbols());
            }
            Condition::EqMod(lhs, rhs, modulus) => {
                symbols.extend(lhs.get_symbols());
                symbols.extend(rhs.get_symbols());
                symbols.extend(modulus.get_symbols());
            }
        }
    }
    remove_builtin_symbols(&mut symbols);
    symbols
}

fn remove_builtin_symbols(symbols: &mut HashSet<String>) {
    symbols.remove("π");
    symbols.remove("pi");
    symbols.remove("e");
}

#[cfg(test)]
#[path = "./rule_test.rs"]
mod rule_test;
