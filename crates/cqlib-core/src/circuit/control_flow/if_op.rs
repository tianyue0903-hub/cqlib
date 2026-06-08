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

use super::{ClassicalExpr, ControlBody};
use crate::circuit::{CircuitError, ClassicalType, ClassicalVar, Qubit};
use std::collections::BTreeSet;

/// Conditional execution controlled by a boolean classical expression.
#[derive(Debug, Clone)]
pub struct IfOp {
    condition: ClassicalExpr,
    then_body: ControlBody,
    else_body: Option<ControlBody>,
}

impl IfOp {
    /// Creates an expression-based `if` operation.
    pub fn new(
        condition: ClassicalExpr,
        then_body: ControlBody,
        else_body: Option<ControlBody>,
    ) -> Result<Self, CircuitError> {
        if condition.ty() != ClassicalType::Bool {
            return Err(CircuitError::InvalidOperation(format!(
                "if condition must be Bool, got {:?}",
                condition.ty()
            )));
        }
        Ok(Self {
            condition,
            then_body,
            else_body,
        })
    }

    pub fn condition(&self) -> &ClassicalExpr {
        &self.condition
    }

    pub fn then_body(&self) -> &ControlBody {
        &self.then_body
    }

    pub fn else_body(&self) -> Option<&ControlBody> {
        self.else_body.as_ref()
    }

    pub fn classical_reads(&self) -> BTreeSet<ClassicalVar> {
        self.condition.vars()
    }

    pub fn used_qubits(&self) -> BTreeSet<Qubit> {
        let mut qubits = self.then_body.used_qubits();
        if let Some(else_body) = &self.else_body {
            qubits.extend(else_body.used_qubits());
        }
        qubits
    }
}

#[cfg(test)]
mod tests {
    use super::IfOp;
    use crate::circuit::{ClassicalExpr, ClassicalType, ClassicalVar, ControlBody};

    #[test]
    fn if_requires_bool_condition() {
        let condition = ClassicalExpr::bool_literal(true);
        let op = IfOp::new(condition, ControlBody::new(vec![]), None).unwrap();

        assert!(op.else_body().is_none());
        assert!(
            IfOp::new(
                ClassicalExpr::bit_literal(true),
                ControlBody::new(vec![]),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn if_reports_condition_reads() {
        let bit = ClassicalExpr::var(ClassicalVar::new(3, ClassicalType::Bit));
        let condition = ClassicalExpr::bit_to_bool(bit).unwrap();
        let op = IfOp::new(condition, ControlBody::new(vec![]), None).unwrap();

        assert!(
            op.classical_reads()
                .contains(&ClassicalVar::new(3, ClassicalType::Bit))
        );
    }
}
