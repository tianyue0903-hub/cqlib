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

//! Target-basis policy for knowledge rewrite.
//!
//! The central invariant is that `physical_keys` are the user's requested final
//! basis, while `lowerable_ranks` are only a search aid.  Lowerable instructions
//! may appear on an intermediate RHS, but they must never be counted as final
//! target support.

use crate::circuit::{Circuit, ControlFlow, Instruction, Operation, StandardGate};
use crate::compiler::error::CompilerError;
use crate::compiler::knowledge::library::RuleKind;
use crate::compiler::transform::rewrite::config::{RewriteConfig, RewriteMode, TargetInstruction};
use std::collections::{HashMap, HashSet};

use super::matcher::{CompiledRuleSet, RewriteInstructionKey};

const NON_LOWERABLE_DISTANCE: usize = 1_000_000;
const MAX_FINAL_TARGET_EXAMPLES: usize = 3;

/// Target-basis lookup used by matcher filtering and local cost.
pub(super) struct TargetContext {
    physical: PhysicalTargetBasis,
    lowerable_ranks: HashMap<RewriteInstructionKey, usize>,
}

/// User-requested final instruction basis.
///
/// This type deliberately has no lowerable state.  Any check using it is asking
/// whether an operation may remain in the finished circuit, not whether a rule
/// may pass through it while searching.
pub(super) struct PhysicalTargetBasis {
    keys: HashSet<RewriteInstructionKey>,
}

impl PhysicalTargetBasis {
    pub(super) fn from_config(config: &RewriteConfig) -> Result<Option<Self>, CompilerError> {
        let Some(target_instructions) = config.target_instructions() else {
            return Ok(None);
        };
        if target_instructions.is_empty() {
            return Err(CompilerError::InvalidContextState(
                "rewrite target instruction basis must not be empty".to_string(),
            ));
        }

        let mut keys = HashSet::with_capacity(target_instructions.len());
        for instruction in target_instructions {
            match instruction {
                TargetInstruction::Standard(gate) => {
                    keys.insert(RewriteInstructionKey::Standard(*gate));
                }
                TargetInstruction::McGate(gate) => {
                    keys.insert(RewriteInstructionKey::McGate(gate.clone()));
                }
                TargetInstruction::Unsupported(description) => {
                    return Err(CompilerError::InvalidContextState(format!(
                        "unsupported rewrite target instruction {description}"
                    )));
                }
            }
        }

        Ok(Some(Self { keys }))
    }

    fn supports(&self, key: &RewriteInstructionKey) -> bool {
        self.keys.contains(key)
    }
}

impl TargetContext {
    pub(super) fn from_config(
        config: &RewriteConfig,
        rules: &CompiledRuleSet,
    ) -> Result<Option<Self>, CompilerError> {
        let Some(physical) = PhysicalTargetBasis::from_config(config)? else {
            return Ok(None);
        };

        let mut lowerable_ranks = physical
            .keys
            .iter()
            .cloned()
            .map(|key| (key, 0))
            .collect::<HashMap<_, _>>();

        loop {
            let mut changed = false;
            for rule in rules.lowerable_rule_views() {
                if rule.kind == RuleKind::Commute
                    || !config.allows_kind(rule.kind)
                    || rule.source_keys.len() != 1
                    || rule.has_conditions
                {
                    continue;
                }

                let mut max_rewrite_rank = 0usize;
                let mut rewrite_is_lowerable = true;
                for key in rule.rewrite_keys {
                    if matches!(key, RewriteInstructionKey::Standard(StandardGate::GPhase)) {
                        continue;
                    }
                    if let Some(rank) = lowerable_ranks.get(key) {
                        max_rewrite_rank = max_rewrite_rank.max(*rank);
                    } else {
                        rewrite_is_lowerable = false;
                        break;
                    }
                }
                if !rewrite_is_lowerable {
                    continue;
                }

                let candidate_rank = max_rewrite_rank.saturating_add(1);
                let source_key = rule.source_keys[0].clone();
                match lowerable_ranks.get_mut(&source_key) {
                    Some(existing_rank) if candidate_rank < *existing_rank => {
                        *existing_rank = candidate_rank;
                        changed = true;
                    }
                    Some(_) => {}
                    None => {
                        lowerable_ranks.insert(source_key, candidate_rank);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Ok(Some(Self {
            physical,
            lowerable_ranks,
        }))
    }

    pub(super) fn allows_rewrite_key(&self, key: &RewriteInstructionKey) -> bool {
        matches!(key, RewriteInstructionKey::Standard(StandardGate::GPhase))
            || self.lowerable_ranks.contains_key(key)
    }

    pub(super) fn physically_supports(&self, key: &RewriteInstructionKey) -> bool {
        self.physical.supports(key)
    }

    pub(super) fn lowering_distance(&self, key: &RewriteInstructionKey) -> usize {
        if self.physically_supports(key) {
            return 0;
        }
        self.lowerable_ranks
            .get(key)
            .copied()
            .unwrap_or(NON_LOWERABLE_DISTANCE)
    }
}

pub(super) fn validate_final_target(
    circuit: &Circuit,
    config: &RewriteConfig,
    transform_name: &'static str,
) -> Result<(), CompilerError> {
    if config.mode() != RewriteMode::Lowering || config.target_instructions().is_none() {
        return Ok(());
    }

    let Some(physical_target) = PhysicalTargetBasis::from_config(config)? else {
        return Ok(());
    };

    let mut scan = FinalTargetScan::default();
    scan_operations(
        circuit.operations(),
        &physical_target,
        config.recurses_control_flow(),
        &mut scan,
    );

    if scan.control_flow_ops > 0 {
        return Err(CompilerError::InvalidContextState(
            "rewrite cannot prove final target instruction basis while recurse_control_flow is disabled and control-flow operations are present".to_string(),
        ));
    }
    if scan.unsupported_gate_like_ops == 0 {
        return Ok(());
    }

    let mut reason = format!(
        "target instruction basis not satisfied: {} gate-like operations remain outside the physical target basis",
        scan.unsupported_gate_like_ops
    );
    if !scan.examples.is_empty() {
        reason.push_str(&format!(" (examples: {})", scan.examples.join(", ")));
    }
    Err(CompilerError::TransformFailed {
        name: transform_name,
        reason,
    })
}

#[derive(Default)]
struct FinalTargetScan {
    unsupported_gate_like_ops: usize,
    control_flow_ops: usize,
    examples: Vec<String>,
}

fn scan_operations(
    operations: &[Operation],
    physical_target: &PhysicalTargetBasis,
    recurse_control_flow: bool,
    scan: &mut FinalTargetScan,
) {
    for operation in operations {
        match &operation.instruction {
            Instruction::Standard(_) | Instruction::McGate(_) => {
                let key = RewriteInstructionKey::from_instruction(&operation.instruction)
                    .expect("standard and multi-controlled gates are rewrite instruction keys");
                if !physical_target.supports(&key) {
                    scan.add_unsupported(&operation.instruction);
                }
            }
            Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
                scan.add_unsupported(&operation.instruction);
            }
            Instruction::ControlFlowGate(flow) => {
                if recurse_control_flow {
                    match flow {
                        ControlFlow::IfElse(gate) => {
                            scan_operations(
                                gate.true_body(),
                                physical_target,
                                recurse_control_flow,
                                scan,
                            );
                            if let Some(false_body) = gate.false_body() {
                                scan_operations(
                                    false_body,
                                    physical_target,
                                    recurse_control_flow,
                                    scan,
                                );
                            }
                        }
                        ControlFlow::WhileLoop(gate) => {
                            scan_operations(
                                gate.body(),
                                physical_target,
                                recurse_control_flow,
                                scan,
                            );
                        }
                    }
                } else {
                    scan.control_flow_ops += 1;
                }
            }
            Instruction::Directive(_) | Instruction::Delay => {}
        }
    }
}

impl FinalTargetScan {
    fn add_unsupported(&mut self, instruction: &Instruction) {
        self.unsupported_gate_like_ops += 1;
        if self.examples.len() < MAX_FINAL_TARGET_EXAMPLES {
            self.examples.push(instruction.to_string());
        }
    }
}
