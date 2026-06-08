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

/// Loop controlled by a boolean classical expression.
#[derive(Debug, Clone)]
pub struct WhileOp {
    condition: ClassicalExpr,
    body: ControlBody,
}

impl WhileOp {
    /// Creates an expression-based `while` operation.
    pub fn new(condition: ClassicalExpr, body: ControlBody) -> Result<Self, CircuitError> {
        if condition.ty() != ClassicalType::Bool {
            return Err(CircuitError::InvalidOperation(format!(
                "while condition must be Bool, got {:?}",
                condition.ty()
            )));
        }
        Ok(Self { condition, body })
    }

    pub fn condition(&self) -> &ClassicalExpr {
        &self.condition
    }

    pub fn body(&self) -> &ControlBody {
        &self.body
    }

    pub fn classical_reads(&self) -> BTreeSet<ClassicalVar> {
        self.condition.vars()
    }

    pub fn used_qubits(&self) -> BTreeSet<Qubit> {
        self.body.used_qubits()
    }
}

#[cfg(test)]
mod tests {
    use super::WhileOp;
    use crate::circuit::{ClassicalExpr, ControlBody};

    #[test]
    fn while_requires_bool_condition() {
        assert!(WhileOp::new(ClassicalExpr::bool_literal(true), ControlBody::new(vec![])).is_ok());
        assert!(WhileOp::new(ClassicalExpr::bit_literal(true), ControlBody::new(vec![])).is_err());
    }
}
