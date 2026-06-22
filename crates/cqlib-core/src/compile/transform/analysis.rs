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

//! Structural circuit analysis shared by compiler transforms.
//!
//! The analysis reports stable facts about the current IR shape so workflow and
//! transforms can skip inapplicable work without rescanning the full operation
//! tree repeatedly.

use crate::circuit::{Circuit, ClassicalControlOp, Instruction, Operation};

/// Structural facts about a circuit relevant to compiler transforms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CircuitAnalysis {
    pub has_classical_data: bool,
    pub has_classical_control: bool,
    pub has_measurement: bool,
    pub has_classical_values: bool,
    pub has_classical_vars: bool,
    pub has_runtime_classical: bool,
    pub needs_classical_handle_preservation: bool,
    pub has_circuit_gate_definitions: bool,
    pub has_unitary_circuit_definitions: bool,
    pub has_unitary_gates: bool,
    pub has_mc_gates: bool,
}

impl CircuitAnalysis {
    /// Computes structural facts for `circuit`.
    pub fn analyze(circuit: &Circuit) -> Self {
        let mut analysis = Self {
            has_classical_values: !circuit.classical_values().is_empty(),
            has_classical_vars: !circuit.classical_vars().is_empty(),
            ..Self::default()
        };
        analysis.scan_operations(circuit.operations());
        analysis.has_runtime_classical = analysis.has_classical_data
            || analysis.has_classical_control
            || analysis.has_classical_values
            || analysis.has_classical_vars;
        analysis.needs_classical_handle_preservation = analysis.has_runtime_classical;
        analysis
    }

    fn scan_operations(&mut self, operations: &[Operation]) {
        for operation in operations {
            self.scan_operation(operation);
        }
    }

    fn scan_operation(&mut self, operation: &Operation) {
        match &operation.instruction {
            Instruction::ClassicalData(op) => {
                self.has_classical_data = true;
                if op.result().is_some() {
                    self.has_measurement = true;
                }
            }
            Instruction::ClassicalControl(op) => {
                self.has_classical_control = true;
                self.scan_control_flow(op);
            }
            Instruction::CircuitGate(_) => {
                self.has_circuit_gate_definitions = true;
            }
            Instruction::UnitaryGate(gate) => {
                self.has_unitary_gates = true;
                if gate.circuit().is_some() {
                    self.has_unitary_circuit_definitions = true;
                }
            }
            Instruction::McGate(_) => {
                self.has_mc_gates = true;
            }
            _ => {}
        }
    }

    fn scan_control_flow(&mut self, op: &ClassicalControlOp) {
        match op {
            ClassicalControlOp::If(op) => {
                self.scan_operations(op.then_body().operations());
                if let Some(body) = op.else_body() {
                    self.scan_operations(body.operations());
                }
            }
            ClassicalControlOp::While(op) => self.scan_operations(op.body().operations()),
            ClassicalControlOp::For(op) => self.scan_operations(op.body().operations()),
            ClassicalControlOp::Switch(op) => {
                for case in op.cases() {
                    self.scan_operations(case.body().operations());
                }
                if let Some(body) = op.default() {
                    self.scan_operations(body.operations());
                }
            }
            ClassicalControlOp::Break | ClassicalControlOp::Continue => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CircuitAnalysis;
    use crate::circuit::{Circuit, ClassicalExpr, Qubit};

    #[test]
    fn analysis_detects_runtime_classical_and_definitions_recursively() {
        let mut inner = Circuit::new(1);
        let measured = inner.measure(Qubit::new(0)).unwrap();
        inner
            .if_(measured.expr().to_bool().unwrap(), |body| {
                body.x(Qubit::new(0))?;
                Ok(())
            })
            .unwrap();
        let gate = inner.to_gate("measured").unwrap();

        let mut circuit = Circuit::new(1);
        let value = circuit.measure(Qubit::new(0)).unwrap();
        circuit
            .if_(ClassicalExpr::bit_to_bool(value.expr()).unwrap(), |body| {
                body.append(gate.clone(), [Qubit::new(0)], [], None)?;
                Ok(())
            })
            .unwrap();

        let analysis = CircuitAnalysis::analyze(&circuit);
        assert!(analysis.has_measurement);
        assert!(analysis.has_classical_data);
        assert!(analysis.has_classical_control);
        assert!(analysis.has_runtime_classical);
        assert!(analysis.needs_classical_handle_preservation);
        assert!(analysis.has_circuit_gate_definitions);
        assert!(!analysis.has_unitary_circuit_definitions);
        assert!(!analysis.has_mc_gates);
    }
}
