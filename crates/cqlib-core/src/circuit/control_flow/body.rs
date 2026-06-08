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

use crate::circuit::{Operation, Qubit};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Structured control-flow body.
#[derive(Debug, Clone)]
pub struct ControlBody {
    operations: Arc<Vec<Operation>>,
}

impl ControlBody {
    /// Creates a body from a sequence of operations.
    pub fn new(operations: Vec<Operation>) -> Self {
        Self {
            operations: Arc::new(operations),
        }
    }

    /// Returns the body operations.
    pub fn operations(&self) -> &[Operation] {
        self.operations.as_slice()
    }

    /// Returns all qubits referenced directly by this body.
    pub fn used_qubits(&self) -> BTreeSet<Qubit> {
        let mut qubits = BTreeSet::new();
        for operation in self.operations() {
            qubits.extend(operation.qubits.iter().copied());
        }
        qubits
    }
}

impl From<Vec<Operation>> for ControlBody {
    fn from(operations: Vec<Operation>) -> Self {
        Self::new(operations)
    }
}

#[cfg(test)]
mod tests {
    use super::ControlBody;
    use crate::circuit::{Instruction, Operation, Qubit, StandardGate};
    use smallvec::smallvec;

    #[test]
    fn body_reports_directly_used_qubits() {
        let body = ControlBody::new(vec![
            Operation {
                instruction: Instruction::Standard(StandardGate::CX),
                qubits: smallvec![Qubit::new(0), Qubit::new(2)],
                params: smallvec![],
                label: None,
            },
            Operation {
                instruction: Instruction::Standard(StandardGate::H),
                qubits: smallvec![Qubit::new(1)],
                params: smallvec![],
                label: None,
            },
        ]);

        let qubits = body.used_qubits();
        assert_eq!(qubits.len(), 3);
        assert!(qubits.contains(&Qubit::new(0)));
        assert!(qubits.contains(&Qubit::new(1)));
        assert!(qubits.contains(&Qubit::new(2)));
    }
}
