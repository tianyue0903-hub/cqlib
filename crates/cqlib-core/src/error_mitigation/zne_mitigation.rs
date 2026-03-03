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

use crate::circuit::{Circuit, CircuitError, Instruction, Operation, Parameter};
use std::collections::HashSet;

/// Zero-noise extrapolation (ZNE) mitigation helper.
///
/// This mirrors the Python `ZNEMitigation` data model and currently implements
/// only circuit folding.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::gate::{Instruction, StandardGate};
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::error_mitigation::ZNEMitigation;
///
/// let q0 = Qubit::new(0);
/// let q1 = Qubit::new(1);
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(q0).unwrap();
/// circuit.cx(q0, q1).unwrap();
///
/// // Build ZNE with fold levels [0, 1, 2], noise factors [1, 3, 5].
/// let zne = ZNEMitigation::new(circuit, vec![0, 1, 2]);
///
/// // Global folding for each level.
/// let folded_all = zne.fold_circuits(None).unwrap();
/// assert_eq!(folded_all.len(), 3);
///
/// // Selective folding for H only.
/// let folded_h_only = zne
///     .fold_circuits(Some(&[Instruction::Standard(StandardGate::H)]))
///     .unwrap();
/// assert_eq!(folded_h_only.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ZNEMitigation {
    circuit: Circuit,
    fold_levels: Vec<i32>,
    noise_factors: Vec<i32>,
}

impl ZNEMitigation {
    /// Creates a new ZNE mitigation helper.
    ///
    /// `noise_factors` follow the Python implementation: `2 * level + 1`.
    pub fn new(circuit: Circuit, fold_levels: Vec<i32>) -> Self {
        let noise_factors = fold_levels.iter().map(|level| 2 * level + 1).collect();
        Self {
            circuit,
            fold_levels,
            noise_factors,
        }
    }

    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    pub fn fold_levels(&self) -> &[i32] {
        &self.fold_levels
    }

    pub fn noise_factors(&self) -> &[i32] {
        &self.noise_factors
    }

    /// Fold the circuit for each configured level using unitary folding.
    ///
    /// If `gate_set` is `None`, this performs global folding:
    /// `U -> U (U^† U)^level`.
    ///
    /// If `gate_set` is provided, only operations whose instruction name matches
    /// one of the instruction names in `gate_set` are folded.
    pub fn fold_circuits(
        &self,
        gate_set: Option<&[Instruction]>,
    ) -> Result<Vec<Circuit>, CircuitError> {
        self.fold_levels
            .iter()
            .map(|level| self.fold_to_level(*level, gate_set))
            .collect()
    }

    fn fold_to_level(
        &self,
        level: i32,
        gate_set: Option<&[Instruction]>,
    ) -> Result<Circuit, CircuitError> {
        if level < 0 {
            return Err(CircuitError::InvalidControlOperation(
                "Fold level must be non-negative.".to_string(),
            ));
        }

        if level == 0 {
            return Ok(self.circuit.clone());
        }

        match gate_set {
            None => self.fold_all(level as usize),
            Some(gates) => self.fold_selected(level as usize, gates),
        }
    }

    fn fold_all(&self, level: usize) -> Result<Circuit, CircuitError> {
        let mut folded = Circuit::from_qubits(self.circuit.qubits())?;
        let inverse = self.circuit.inverse()?;

        self.append_circuit_ops(&mut folded, &self.circuit)?;
        for _ in 0..level {
            self.append_circuit_ops(&mut folded, &inverse)?;
            self.append_circuit_ops(&mut folded, &self.circuit)?;
        }

        Ok(folded)
    }

    fn fold_selected(
        &self,
        level: usize,
        gate_set: &[Instruction],
    ) -> Result<Circuit, CircuitError> {
        let gate_names: HashSet<String> = gate_set.iter().map(|gate| gate.to_string()).collect();
        let mut folded = Circuit::from_qubits(self.circuit.qubits())?;

        for op in self.circuit.operations() {
            self.append_operation(&mut folded, &self.circuit, op)?;
            if gate_names.contains(&op.instruction.to_string()) {
                for _ in 0..level {
                    let inv = self.invert_operation(op)?;
                    self.append_operation(&mut folded, &inv.0, &inv.1)?;
                    self.append_operation(&mut folded, &self.circuit, op)?;
                }
            }
        }

        Ok(folded)
    }

    fn append_circuit_ops(
        &self,
        target: &mut Circuit,
        source: &Circuit,
    ) -> Result<(), CircuitError> {
        for op in source.operations() {
            self.append_operation(target, source, op)?;
        }
        Ok(())
    }

    fn append_operation(
        &self,
        target: &mut Circuit,
        source: &Circuit,
        op: &Operation,
    ) -> Result<(), CircuitError> {
        let params = op.params.iter().map(|param| match param {
            crate::circuit::param::CircuitParam::Fixed(value) => (*value).into(),
            crate::circuit::param::CircuitParam::Index(index) => {
                source.parameters()[*index as usize].clone().into()
            }
        });

        target.append(
            op.instruction.clone(),
            op.qubits.iter().copied(),
            params,
            op.label.as_deref(),
        )
    }

    fn invert_operation(&self, op: &Operation) -> Result<(Circuit, Operation), CircuitError> {
        let op_params: Vec<Parameter> = op
            .params
            .iter()
            .map(|param| match param {
                crate::circuit::param::CircuitParam::Fixed(value) => Parameter::from(*value),
                crate::circuit::param::CircuitParam::Index(index) => {
                    self.circuit.parameters()[*index as usize].clone()
                }
            })
            .collect();

        let (inv_inst, inv_params) = op
            .instruction
            .inverse(&op_params)
            .ok_or(CircuitError::IrreversibleOperation)?;

        let mut inv_circuit = Circuit::from_qubits(self.circuit.qubits())?;
        inv_circuit.append(
            inv_inst,
            op.qubits.iter().copied(),
            inv_params.into_iter().map(Into::into),
            op.label.as_deref(),
        )?;
        let inv_op = inv_circuit.operations()[0].clone();

        Ok((inv_circuit, inv_op))
    }
}

#[cfg(test)]
#[path = "./zne_mitigation_test.rs"]
mod zne_mitigation_test;
