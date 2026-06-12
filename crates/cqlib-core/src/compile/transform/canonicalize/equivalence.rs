// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Representation equality used for canonicalization fixed-point detection.
//!
//! This module defines equivalence only for the canonicalizer's convergence
//! loop. Two circuits are equivalent here when their compiler IR
//! representation is identical in every field that canonicalization is allowed
//! to stabilize: qubit order, symbol and parameter tables, global phase,
//! operation sequence, labels, operands, and recursively stored control-flow
//! bodies.
//!
//! This is deliberately not unitary equivalence, semantic equivalence, or
//! control-flow logical equivalence. For example, two different gate sequences
//! with the same matrix are not equivalent for this module unless the
//! canonicalizer has already rewritten them to the same representation. Keeping
//! this predicate representation-based makes fixed-point detection cheap,
//! deterministic, and aligned with the pass contract.

use crate::circuit::{
    Circuit, CircuitParam, ClassicalControlOp, ClassicalDataOp, ControlBody, Instruction, Operation,
};

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
        (Instruction::ClassicalData(lhs), Instruction::ClassicalData(rhs)) => {
            classical_data_equal(lhs, rhs)
        }
        (Instruction::ClassicalControl(lhs), Instruction::ClassicalControl(rhs)) => {
            classical_control_equal(lhs, rhs)
        }
        _ => false,
    }
}

fn classical_data_equal(lhs: &ClassicalDataOp, rhs: &ClassicalDataOp) -> bool {
    match (lhs, rhs) {
        (
            ClassicalDataOp::Store {
                target: lhs_target,
                value: lhs_value,
            },
            ClassicalDataOp::Store {
                target: rhs_target,
                value: rhs_value,
            },
        ) => lhs_target == rhs_target && lhs_value == rhs_value,
        (
            ClassicalDataOp::MeasureBit { result: lhs },
            ClassicalDataOp::MeasureBit { result: rhs },
        )
        | (
            ClassicalDataOp::MeasureBits { result: lhs },
            ClassicalDataOp::MeasureBits { result: rhs },
        ) => lhs == rhs,
        _ => false,
    }
}

fn classical_control_equal(lhs: &ClassicalControlOp, rhs: &ClassicalControlOp) -> bool {
    match (lhs, rhs) {
        (ClassicalControlOp::If(lhs), ClassicalControlOp::If(rhs)) => {
            lhs.condition() == rhs.condition()
                && bodies_equal(lhs.then_body(), rhs.then_body())
                && match (lhs.else_body(), rhs.else_body()) {
                    (Some(lhs), Some(rhs)) => bodies_equal(lhs, rhs),
                    (None, None) => true,
                    _ => false,
                }
        }
        (ClassicalControlOp::While(lhs), ClassicalControlOp::While(rhs)) => {
            lhs.condition() == rhs.condition() && bodies_equal(lhs.body(), rhs.body())
        }
        (ClassicalControlOp::For(lhs), ClassicalControlOp::For(rhs)) => {
            lhs.var() == rhs.var()
                && lhs.start() == rhs.start()
                && lhs.stop() == rhs.stop()
                && lhs.step() == rhs.step()
                && bodies_equal(lhs.body(), rhs.body())
        }
        (ClassicalControlOp::Switch(lhs), ClassicalControlOp::Switch(rhs)) => {
            lhs.target() == rhs.target()
                && lhs.cases().len() == rhs.cases().len()
                && lhs.cases().iter().zip(rhs.cases()).all(|(lhs, rhs)| {
                    lhs.value() == rhs.value() && bodies_equal(lhs.body(), rhs.body())
                })
                && match (lhs.default(), rhs.default()) {
                    (Some(lhs), Some(rhs)) => bodies_equal(lhs, rhs),
                    (None, None) => true,
                    _ => false,
                }
        }
        (ClassicalControlOp::Break, ClassicalControlOp::Break)
        | (ClassicalControlOp::Continue, ClassicalControlOp::Continue) => true,
        _ => false,
    }
}

fn bodies_equal(lhs: &ControlBody, rhs: &ControlBody) -> bool {
    operation_slices_equal(lhs.operations(), rhs.operations())
}

fn circuit_params_equal(lhs: &[CircuitParam], rhs: &[CircuitParam]) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(lhs, rhs)| match (lhs, rhs) {
            (CircuitParam::Fixed(lhs), CircuitParam::Fixed(rhs)) => lhs == rhs,
            (CircuitParam::Index(lhs), CircuitParam::Index(rhs)) => lhs == rhs,
            _ => false,
        })
}
