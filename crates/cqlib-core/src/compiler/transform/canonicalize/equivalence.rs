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

//! Equivalence checking for canonicalization change detection.
//!
//! Canonicalization rebuilds circuits from scratch using public `Circuit`
//! APIs. To detect whether a round actually changed anything, we compare the
//! rebuilt circuit against the original using a conservative equivalence check.
///
/// This check is scoped to **canonical representation changes** — it is NOT
/// full quantum semantic equivalence. It compares parameter tables and symbol
/// tables structurally, and resolves operation parameters semantically so that
/// equivalent fixed values are recognized across rebuilds. Benign structural
/// differences that do not affect operation semantics (such as parameter pool
/// reordering) are still reported as changes when the tables differ.
use crate::circuit::gate::UnitaryMatrix;
use crate::circuit::{Circuit, ControlFlow, Instruction, Operation};
use num_complex::Complex64;

use super::linear::PendingOperation;
use super::ops::resolve_operation_param;

/// Conservative equivalence check for detecting whether a canonicalization round changed anything.
///
/// This reports `true` only when the two circuits are indistinguishable in
/// their canonical representation:
/// - qubit sets, global phase, symbols, and parameters are compared structurally
/// - operations are compared with semantically resolved parameters
/// - instructions are compared recursively into control-flow bodies and nested gates
///
/// It is intentionally **not** full quantum semantic equivalence. Benign
/// structural differences (e.g. harmless parameter-table reordering) are still
/// reported as changes when the tables differ.
pub(crate) fn circuits_equivalent_for_canonicalize(before: &Circuit, after: &Circuit) -> bool {
    before.qubits() == after.qubits()
        && before.global_phase() == after.global_phase()
        && before.symbols() == after.symbols()
        && before.parameters() == after.parameters()
        && operations_equivalent(before.operations(), after.operations(), before, after)
}

/// Compares two operation sequences for canonicalization round tracking.
pub(crate) fn operations_equivalent(
    lhs: &[Operation],
    rhs: &[Operation],
    lhs_circuit: &Circuit,
    rhs_circuit: &Circuit,
) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }

    lhs.iter().zip(rhs).all(|(lhs, rhs)| {
        lhs.qubits == rhs.qubits
            && lhs.label.as_deref() == rhs.label.as_deref()
            && instructions_equivalent(&lhs.instruction, &rhs.instruction, lhs_circuit, rhs_circuit)
            && params_equivalent(&lhs.params, &rhs.params, lhs_circuit, rhs_circuit)
    })
}

/// Compares two instructions for canonicalization round tracking.
///
/// Recurses into control-flow bodies and nested `CircuitGate` / `UnitaryGate`
/// circuits. Standard gates, multi-controlled gates, and directives are
/// compared structurally. Unitary matrices use a small floating-point
/// tolerance (`1e-14`) because rebuilds may introduce negligible numeric drift.
pub(crate) fn instructions_equivalent(
    lhs: &Instruction,
    rhs: &Instruction,
    lhs_circuit: &Circuit,
    rhs_circuit: &Circuit,
) -> bool {
    match (lhs, rhs) {
        (
            Instruction::ControlFlowGate(ControlFlow::IfElse(lhs)),
            Instruction::ControlFlowGate(ControlFlow::IfElse(rhs)),
        ) => {
            lhs.condition() == rhs.condition()
                && operations_equivalent(lhs.true_body(), rhs.true_body(), lhs_circuit, rhs_circuit)
                && match (lhs.false_body(), rhs.false_body()) {
                    (Some(lhs), Some(rhs)) => {
                        operations_equivalent(lhs, rhs, lhs_circuit, rhs_circuit)
                    }
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(lhs)),
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(rhs)),
        ) => {
            lhs.condition() == rhs.condition()
                && operations_equivalent(lhs.body(), rhs.body(), lhs_circuit, rhs_circuit)
        }
        (Instruction::Standard(lhs), Instruction::Standard(rhs)) => lhs == rhs,
        (Instruction::McGate(lhs), Instruction::McGate(rhs)) => lhs == rhs,
        (Instruction::Directive(lhs), Instruction::Directive(rhs)) => lhs == rhs,
        (Instruction::Delay, Instruction::Delay) => true,
        (Instruction::CircuitGate(lhs), Instruction::CircuitGate(rhs)) => {
            lhs.name() == rhs.name()
                && lhs.num_qubits() == rhs.num_qubits()
                && lhs.num_params() == rhs.num_params()
                && circuits_equivalent_for_canonicalize(
                    lhs.circuit().circuit(),
                    rhs.circuit().circuit(),
                )
        }
        (Instruction::UnitaryGate(lhs), Instruction::UnitaryGate(rhs)) => {
            lhs.label() == rhs.label()
                && lhs.num_qubits() == rhs.num_qubits()
                && unitary_matrix_reprs_equivalent(lhs, rhs)
                && match (lhs.circuit(), rhs.circuit()) {
                    (Some(lhs), Some(rhs)) => {
                        circuits_equivalent_for_canonicalize(lhs.circuit(), rhs.circuit())
                    }
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => false,
    }
}

/// Compares two optional unitary matrices with a tight floating-point tolerance.
///
/// Uses `1e-14` for both real and imaginary parts to tolerate negligible
/// numeric drift that can appear after circuit serialization or rebuild.
pub(crate) fn unitary_matrices_equivalent(
    lhs: Option<&ndarray::Array2<Complex64>>,
    rhs: Option<&ndarray::Array2<Complex64>>,
) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            lhs.shape() == rhs.shape()
                && lhs.iter().zip(rhs.iter()).all(|(lhs, rhs)| {
                    (lhs.re - rhs.re).abs() <= 1e-14 && (lhs.im - rhs.im).abs() <= 1e-14
                })
        }
        (None, None) => true,
        _ => false,
    }
}

fn unitary_matrix_reprs_equivalent(
    lhs: &crate::circuit::UnitaryGate,
    rhs: &crate::circuit::UnitaryGate,
) -> bool {
    match (lhs.matrix_repr(), rhs.matrix_repr()) {
        (Some(UnitaryMatrix::Numeric(lhs)), Some(UnitaryMatrix::Numeric(rhs))) => {
            unitary_matrices_equivalent(Some(lhs), Some(rhs))
        }
        (Some(UnitaryMatrix::Symbolic(lhs_matrix)), Some(UnitaryMatrix::Symbolic(rhs_matrix))) => {
            lhs.matrix_params() == rhs.matrix_params() && lhs_matrix == rhs_matrix
        }
        (None, None) => true,
        _ => false,
    }
}

/// Compares two parameter lists by resolving them into semantic `Parameter`
/// values.
pub(crate) fn params_equivalent(
    lhs: &[crate::circuit::CircuitParam],
    rhs: &[crate::circuit::CircuitParam],
    lhs_circuit: &Circuit,
    rhs_circuit: &Circuit,
) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(lhs, rhs)| {
            match (
                resolve_operation_param(lhs_circuit, lhs),
                resolve_operation_param(rhs_circuit, rhs),
            ) {
                (Ok(lhs), Ok(rhs)) => lhs == rhs,
                _ => false,
            }
        })
}

/// Compares an original operation sequence against a pending (rewritten)
/// operation sequence.
pub(crate) fn pending_operations_equivalent(
    original: &[Operation],
    rewritten: &[PendingOperation],
    circuit: &Circuit,
) -> bool {
    original.len() == rewritten.len()
        && original.iter().zip(rewritten).all(|(lhs, rhs)| {
            lhs.qubits == rhs.qubits
                && lhs.label.as_deref() == rhs.label.as_deref()
                && instructions_equivalent(&lhs.instruction, &rhs.instruction, circuit, circuit)
                && pending_params_equivalent(&lhs.params, &rhs.params, circuit)
        })
}

/// Compares original `CircuitParam` references against resolved
/// `ParameterValue`s.
pub(crate) fn pending_params_equivalent(
    lhs: &[crate::circuit::CircuitParam],
    rhs: &[crate::circuit::ParameterValue],
    circuit: &Circuit,
) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(lhs, rhs)| {
            let Ok(lhs) = resolve_operation_param(circuit, lhs) else {
                return false;
            };
            match rhs {
                crate::circuit::ParameterValue::Fixed(value) => {
                    lhs == crate::circuit::Parameter::from(*value)
                }
                crate::circuit::ParameterValue::Param(param) => lhs == *param,
            }
        })
}
