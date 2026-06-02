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

//! Representation equality used for canonicalization fixed-point detection.

use crate::circuit::{Circuit, CircuitParam, ControlFlow, Instruction, Operation};

pub(super) fn circuits_equivalent_for_canonicalize(lhs: &Circuit, rhs: &Circuit) -> bool {
    lhs.qubits() == rhs.qubits()
        && lhs.symbols() == rhs.symbols()
        && lhs.parameters() == rhs.parameters()
        && lhs.global_phase() == rhs.global_phase()
        && operation_slices_equal(lhs.operations(), rhs.operations())
}

fn operation_slices_equal(lhs: &[Operation], rhs: &[Operation]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(lhs, rhs)| operations_equal(lhs, rhs))
}

fn operations_equal(lhs: &Operation, rhs: &Operation) -> bool {
    instructions_equal(&lhs.instruction, &rhs.instruction)
        && lhs.qubits == rhs.qubits
        && circuit_params_equal(&lhs.params, &rhs.params)
        && lhs.label.as_deref() == rhs.label.as_deref()
}

fn instructions_equal(lhs: &Instruction, rhs: &Instruction) -> bool {
    match (lhs, rhs) {
        (Instruction::Standard(lhs), Instruction::Standard(rhs)) => lhs == rhs,
        (Instruction::McGate(lhs), Instruction::McGate(rhs)) => lhs == rhs,
        (Instruction::Directive(lhs), Instruction::Directive(rhs)) => lhs == rhs,
        (Instruction::Delay, Instruction::Delay) => true,
        (Instruction::CircuitGate(lhs), Instruction::CircuitGate(rhs)) => {
            lhs.name() == rhs.name()
                && lhs.num_qubits() == rhs.num_qubits()
                && lhs.num_params() == rhs.num_params()
        }
        (Instruction::UnitaryGate(lhs), Instruction::UnitaryGate(rhs)) => lhs == rhs,
        (
            Instruction::ControlFlowGate(ControlFlow::IfElse(lhs)),
            Instruction::ControlFlowGate(ControlFlow::IfElse(rhs)),
        ) => {
            lhs.condition() == rhs.condition()
                && operation_slices_equal(lhs.true_body(), rhs.true_body())
                && match (lhs.false_body(), rhs.false_body()) {
                    (Some(lhs), Some(rhs)) => operation_slices_equal(lhs, rhs),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(lhs)),
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(rhs)),
        ) => lhs.condition() == rhs.condition() && operation_slices_equal(lhs.body(), rhs.body()),
        _ => false,
    }
}

fn circuit_params_equal(lhs: &[CircuitParam], rhs: &[CircuitParam]) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(lhs, rhs)| match (lhs, rhs) {
            (CircuitParam::Fixed(lhs), CircuitParam::Fixed(rhs)) => lhs == rhs,
            (CircuitParam::Index(lhs), CircuitParam::Index(rhs)) => lhs == rhs,
            _ => false,
        })
}
