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

use super::{ForOp, IfOp, SwitchOp, WhileOp};
use crate::circuit::{ClassicalValue, ClassicalVar, Qubit};
use std::collections::BTreeSet;
use std::fmt;

/// Expression-based classical control-flow operation.
#[derive(Debug, Clone)]
pub enum ClassicalControlOp {
    If(IfOp),
    While(WhileOp),
    For(ForOp),
    Switch(SwitchOp),
    Break,
    Continue,
}

impl ClassicalControlOp {
    /// Returns classical variables read by the operation's controlling expressions.
    pub fn classical_var_reads(&self) -> BTreeSet<ClassicalVar> {
        match self {
            Self::If(op) => op.classical_var_reads(),
            Self::While(op) => op.classical_var_reads(),
            Self::For(op) => op.classical_var_reads(),
            Self::Switch(op) => op.classical_var_reads(),
            Self::Break | Self::Continue => BTreeSet::new(),
        }
    }

    /// Returns immutable classical values read by the operation's controlling expressions.
    pub fn classical_value_reads(&self) -> BTreeSet<ClassicalValue> {
        match self {
            Self::If(op) => op.classical_value_reads(),
            Self::While(op) => op.classical_value_reads(),
            Self::For(op) => op.classical_value_reads(),
            Self::Switch(op) => op.classical_value_reads(),
            Self::Break | Self::Continue => BTreeSet::new(),
        }
    }

    /// Returns classical variables written directly by this operation.
    pub fn classical_writes(&self) -> BTreeSet<ClassicalVar> {
        match self {
            Self::For(op) => op.classical_writes(),
            Self::If(_) | Self::While(_) | Self::Switch(_) | Self::Break | Self::Continue => {
                BTreeSet::new()
            }
        }
    }

    /// Returns qubits used by the operation's structured bodies.
    pub fn used_qubits(&self) -> BTreeSet<Qubit> {
        match self {
            Self::If(op) => op.used_qubits(),
            Self::While(op) => op.used_qubits(),
            Self::For(op) => op.used_qubits(),
            Self::Switch(op) => op.used_qubits(),
            Self::Break | Self::Continue => BTreeSet::new(),
        }
    }
}

impl fmt::Display for ClassicalControlOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::If(_) => write!(f, "if"),
            Self::While(_) => write!(f, "while"),
            Self::For(_) => write!(f, "for"),
            Self::Switch(_) => write!(f, "switch"),
            Self::Break => write!(f, "break"),
            Self::Continue => write!(f, "continue"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ClassicalControlOp;
    use crate::circuit::{
        CircuitId, ClassicalExpr, ClassicalType, ClassicalValue, ClassicalVar, ControlBody, ForOp,
        IfOp, SwitchCase, SwitchOp, WhileOp,
    };

    #[test]
    fn break_and_continue_have_no_resource_dependencies() {
        assert!(ClassicalControlOp::Break.classical_var_reads().is_empty());
        assert!(ClassicalControlOp::Break.classical_value_reads().is_empty());
        assert!(ClassicalControlOp::Break.classical_writes().is_empty());
        assert!(ClassicalControlOp::Break.used_qubits().is_empty());
        assert!(
            ClassicalControlOp::Continue
                .classical_var_reads()
                .is_empty()
        );
        assert!(
            ClassicalControlOp::Continue
                .classical_value_reads()
                .is_empty()
        );
        assert!(ClassicalControlOp::Continue.classical_writes().is_empty());
        assert!(ClassicalControlOp::Continue.used_qubits().is_empty());
    }

    #[test]
    fn control_op_forwards_classical_var_and_value_reads() {
        let circuit_id = CircuitId::new();
        let bit = ClassicalVar::new(circuit_id, 1, ClassicalType::Bit);
        let value = ClassicalValue::new(circuit_id, 2, ClassicalType::Bit);
        let condition = ClassicalExpr::try_and(
            ClassicalExpr::bit_to_bool(bit.expr()).unwrap(),
            ClassicalExpr::bit_to_bool(value.expr()).unwrap(),
        )
        .unwrap();
        let op = IfOp::new(condition, ControlBody::new(vec![]), None).unwrap();
        let op = ClassicalControlOp::If(op);

        assert!(op.classical_var_reads().contains(&bit));
        assert!(op.classical_value_reads().contains(&value));
    }

    #[test]
    fn display_reports_operation_kind() {
        let condition = ClassicalExpr::bool_literal(true);
        let body = ControlBody::new(vec![]);
        let if_op = IfOp::new(condition.clone(), body.clone(), None).unwrap();
        let while_op = WhileOp::new(condition, body.clone()).unwrap();
        let loop_var = ClassicalVar::new(CircuitId::new(), 0, ClassicalType::uint(8).unwrap());
        let for_op = ForOp::new(
            loop_var,
            ClassicalExpr::uint_literal(8, 0).unwrap(),
            ClassicalExpr::uint_literal(8, 2).unwrap(),
            ClassicalExpr::uint_literal(8, 1).unwrap(),
            body.clone(),
        )
        .unwrap();
        let switch_op = SwitchOp::new(
            ClassicalExpr::uint_literal(2, 1).unwrap(),
            vec![SwitchCase::new(1, body)],
            None,
        )
        .unwrap();

        assert_eq!(ClassicalControlOp::If(if_op).to_string(), "if");
        assert_eq!(ClassicalControlOp::While(while_op).to_string(), "while");
        assert_eq!(ClassicalControlOp::For(for_op).to_string(), "for");
        assert_eq!(ClassicalControlOp::Switch(switch_op).to_string(), "switch");
        assert_eq!(ClassicalControlOp::Break.to_string(), "break");
        assert_eq!(ClassicalControlOp::Continue.to_string(), "continue");
    }
}
