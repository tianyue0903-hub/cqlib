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

//! Single-operation canonicalization utilities.
//!
//! This module provides helpers for:
//! - resolving `CircuitParam` references into semantic `Parameter` values
//! - detecting and dropping trivial no-ops
//! - canonicalizing barrier qubit lists (sorting and deduplication)
//! - comparing and merging barrier scopes and labels
//!
//! These helpers are used by both the top-level linear scan (`linear.rs`) and
//! the control-flow body rebuild path so that canonicalization behavior stays
//! identical regardless of where an operation appears in the circuit.

use crate::circuit::{
    Circuit, CircuitParam, Directive, Instruction, Operation, Parameter, ParameterValue,
    StandardGate,
};
use crate::compiler::error::CompilerError;
use smallvec::SmallVec;

use super::config::CanonicalizeConfig;

/// Resolves an operation-local parameter reference into the semantic
/// `Parameter` value it denotes.
///
/// This keeps structural canonicalization honest about the distinction between:
/// - `Circuit.parameters`: the circuit-wide parameter pool
/// - `Operation.params`: compact `CircuitParam` references into that pool
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
                    "invalid control-flow body parameter index {}",
                    index
                ))
            }),
    }
}

/// Resolves a `CircuitParam` into a `ParameterValue` for use in a
/// `PendingOperation`.
pub(crate) fn resolve_parameter_value(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<ParameterValue, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(*value)),
        CircuitParam::Index(_) => Ok(resolve_operation_param(circuit, param)?.into()),
    }
}

/// Returns the first resolved parameter of an operation, if any.
pub(crate) fn operation_first_param(
    circuit: &Circuit,
    operation: &Operation,
) -> Result<Option<Parameter>, CompilerError> {
    operation
        .params
        .first()
        .map(|param| resolve_operation_param(circuit, param))
        .transpose()
}

/// Determines whether an operation is a trivial no-op that can be dropped.
///
/// Hard-coded rules (kept as `match` arms because the set is small and stable):
/// - Unlabeled `I` gate — exactly identity.
/// - Unlabeled `Delay(0)` — zero-duration delay has no effect.
/// - Unlabeled `RX/RY/RZ/Phase/RXX/RYY/RZZ/RZX(0)` — rotation by zero is identity.
/// - Unlabeled barrier with an empty qubit set — conveys no synchronization info.
///
/// Labeled operations are retained even when they are quantum-semantic no-ops
/// because operation labels are user-visible metadata.
pub(crate) fn should_drop_operation(
    circuit: &Circuit,
    operation: &Operation,
    instruction: &Instruction,
    qubits: &[crate::circuit::Qubit],
    config: &CanonicalizeConfig,
) -> Result<bool, CompilerError> {
    if !config.drops_trivial_noops() {
        return Ok(false);
    }

    if operation.label.is_some() {
        return Ok(false);
    }

    if is_barrier_instruction(instruction) && qubits.is_empty() {
        return Ok(true);
    }

    Ok(match instruction {
        Instruction::Standard(StandardGate::I) => true,
        Instruction::Delay => {
            operation_first_param(circuit, operation)?.is_some_and(|param| param.is_zero())
        }
        Instruction::Standard(
            StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::Phase
            | StandardGate::RXX
            | StandardGate::RYY
            | StandardGate::RZZ
            | StandardGate::RZX,
        ) => operation_first_param(circuit, operation)?.is_some_and(|param| param.is_zero()),
        _ => false,
    })
}

/// Returns `true` if the instruction is a barrier directive.
pub(crate) fn is_barrier_instruction(instruction: &Instruction) -> bool {
    matches!(instruction, Instruction::Directive(Directive::Barrier))
}

/// Sorts and deduplicates the qubit list for barrier instructions.
///
/// Non-barrier instructions are returned unchanged.
pub(crate) fn canonicalize_barrier_qubits(
    instruction: &Instruction,
    qubits: &SmallVec<[crate::circuit::Qubit; 3]>,
) -> SmallVec<[crate::circuit::Qubit; 3]> {
    if !is_barrier_instruction(instruction) {
        return qubits.clone();
    }

    let mut qubits = qubits.clone();
    qubits.sort_unstable_by_key(|qubit| qubit.id());
    qubits.dedup();
    qubits
}

/// Relationship between two barrier qubit sets for merge decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BarrierRelation {
    Equal,
    LeftSuperset,
    RightSuperset,
    DisjointOrOverlap,
}

/// Compares two sorted barrier qubit scopes to decide mergeability.
///
/// Returns:
/// - `Equal` — identical sets.
/// - `LeftSuperset` — `lhs` strictly contains `rhs`.
/// - `RightSuperset` — `rhs` strictly contains `lhs`.
/// - `DisjointOrOverlap` — neither is a superset of the other.
pub(crate) fn compare_barrier_scope(
    lhs: &[crate::circuit::Qubit],
    rhs: &[crate::circuit::Qubit],
) -> BarrierRelation {
    if lhs == rhs {
        return BarrierRelation::Equal;
    }

    let lhs_contains_rhs = rhs.iter().all(|qubit| lhs.contains(qubit));
    let rhs_contains_lhs = lhs.iter().all(|qubit| rhs.contains(qubit));

    match (lhs_contains_rhs, rhs_contains_lhs) {
        (true, false) => BarrierRelation::LeftSuperset,
        (false, true) => BarrierRelation::RightSuperset,
        _ => BarrierRelation::DisjointOrOverlap,
    }
}

/// Merges two operation labels, preserving uniqueness and original order.
///
/// Labels are split on `" | "`, deduplicated while keeping the first-seen
/// order, and re-joined with the same separator. This prevents duplicate
/// metadata when two barriers are collapsed into one.
pub(crate) fn merge_operation_labels(
    primary: Option<Box<str>>,
    absorbed: Option<Box<str>>,
) -> Option<Box<str>> {
    match (primary, absorbed) {
        (None, None) => None,
        (Some(label), None) | (None, Some(label)) => Some(label),
        (Some(primary), Some(absorbed)) => {
            if primary == absorbed {
                return Some(primary);
            }

            let mut merged = Vec::new();
            for part in split_merged_label(&primary) {
                if !merged.iter().any(|existing| *existing == part) {
                    merged.push(part.to_string());
                }
            }
            for part in split_merged_label(&absorbed) {
                if !merged.iter().any(|existing| *existing == part) {
                    merged.push(part.to_string());
                }
            }

            Some(merged.join(" | ").into_boxed_str())
        }
    }
}

/// Splits a merged label into its constituent parts.
fn split_merged_label(label: &str) -> impl Iterator<Item = &str> {
    label.split(" | ")
}
