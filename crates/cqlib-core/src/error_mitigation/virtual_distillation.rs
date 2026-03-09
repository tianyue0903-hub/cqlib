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

use std::collections::HashMap;

use thiserror::Error;

use crate::circuit::{Circuit, CircuitError, CircuitParam, ParameterValue, Qubit};

/// Errors raised by [`VirtualDistillation`].
#[derive(Debug, Error, PartialEq)]
pub enum VirtualDistillationError {
    #[error("virtual distillation requires at least 2 copies, got {0}")]
    InvalidCopies(usize),
}

/// Virtual distillation mitigation based on the moment ratio
/// `Tr(O rho^M) / Tr(rho^M)`.
/// Based on: [1] W. J. Huggins et al., “Virtual Distillation for Quantum Error Mitigation,”
///     Phys. Rev. X, vol. 11, no. 4, p. 041036, Nov. 2021, doi: 10.1103/PhysRevX.11.041036.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::error_mitigation::VirtualDistillation;
///
/// let q0 = Qubit::new(0);
/// let mut circuit = Circuit::new(1);
/// circuit.x(q0).unwrap();
///
/// let _vd = VirtualDistillation::new(circuit, 2).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct VirtualDistillation {
    circuit: Circuit,
    copies: usize,
}

impl VirtualDistillation {
    /// Creates a new virtual distillation helper.
    pub fn new(circuit: Circuit, copies: usize) -> Result<Self, VirtualDistillationError> {
        if copies < 2 {
            return Err(VirtualDistillationError::InvalidCopies(copies));
        }

        Ok(Self { circuit, copies })
    }

    /// Builds a copy-swap circuit from the configured base circuit.
    ///
    /// The returned circuit contains:
    /// - `copies` disjoint copies of the base circuit preparation,
    /// - pairwise SWAP operations between the first copy and every additional copy.
    pub fn build_copy_swap_circuit(&self) -> Result<Circuit, CircuitError> {
        let base_circuit = self.circuit.decompose()?;
        let base_width = base_circuit.width();
        let mut copy_swap_circuit = Circuit::new(self.copies * base_width);

        for copy_index in 0..self.copies {
            let copy_offset = copy_index * base_width;
            Self::append_circuit_with_offset(&mut copy_swap_circuit, &base_circuit, copy_offset)?;
        }

        for other_copy in 1..self.copies {
            let first_copy_offset = 0;
            let other_copy_offset = other_copy * base_width;
            for qubit_index in 0..base_width {
                let left = Qubit::new((first_copy_offset + qubit_index) as u32);
                let right = Qubit::new((other_copy_offset + qubit_index) as u32);
                copy_swap_circuit.swap(left, right)?;
            }
        }

        Ok(copy_swap_circuit)
    }

    fn append_circuit_with_offset(
        target_circuit: &mut Circuit,
        source_circuit: &Circuit,
        qubit_offset: usize,
    ) -> Result<(), CircuitError> {
        let source_qubits = source_circuit.qubits();
        let qubit_positions: HashMap<_, _> = source_qubits
            .iter()
            .enumerate()
            .map(|(position, qubit)| (*qubit, position))
            .collect();

        for op in source_circuit.operations() {
            let mapped_qubits: Vec<Qubit> = op
                .qubits
                .iter()
                .map(|qubit| {
                    let position = qubit_positions[qubit];
                    Qubit::new((qubit_offset + position) as u32)
                })
                .collect();
            let mapped_params: Vec<ParameterValue> = op
                .params
                .iter()
                .map(|param| match param {
                    CircuitParam::Fixed(value) => ParameterValue::Fixed(*value),
                    CircuitParam::Index(index) => {
                        source_circuit.parameters()[*index as usize].clone().into()
                    }
                })
                .collect();

            target_circuit.append(
                op.instruction.clone(),
                mapped_qubits,
                mapped_params,
                op.label.as_deref(),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "./virtual_distillation_test.rs"]
mod virtual_distillation_test;
