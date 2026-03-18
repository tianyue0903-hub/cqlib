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

use crate::circuit::{Circuit, CircuitError, CircuitParam, ParameterValue, Qubit};
use crate::error_mitigation::Estimator;
use crate::error_mitigation::ErrorMitigationError;
use crate::qis::{Hamiltonian, Pauli, PauliString};

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
    pub fn new(circuit: Circuit, copies: usize) -> Result<Self, ErrorMitigationError> {
        if copies < 2 {
            return Err(ErrorMitigationError::InvalidCopies(copies));
        }

        Ok(Self { circuit, copies })
    }

    /// Returns the configured number of copies.
    pub fn copies(&self) -> usize {
        self.copies
    }

    /// Updates the configured number of copies.
    pub fn set_copies(&mut self, copies: usize) -> Result<(), ErrorMitigationError> {
        if copies < 2 {
            return Err(ErrorMitigationError::InvalidCopies(copies));
        }

        self.copies = copies;
        Ok(())
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

    /// Expands a Hamiltonian to the full copy-swap circuit width.
    ///
    /// Virtual distillation evaluates the numerator on a wider copy-swap circuit,
    /// so the estimator needs a Hamiltonian of matching width. This helper keeps
    /// the original Pauli operators on their current qubit indices and appends
    /// `n` `Z`s on higher-index qubits.
    fn expand_hamiltonian(hamiltonian: &Hamiltonian, n: usize) -> Hamiltonian {
        if hamiltonian.terms.is_empty() {
            return Hamiltonian::new(hamiltonian.num_qubits + n);
        }

        let expanded_terms = hamiltonian
            .terms
            .iter()
            .map(|(term, coeff)| {
                let mut expanded_term = PauliString::new(hamiltonian.num_qubits + n);
                expanded_term.phase = term.phase;

                for qubit in 0..hamiltonian.num_qubits {
                    let pauli = match (term.x[qubit], term.z[qubit]) {
                        (false, false) => Pauli::I,
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                    };
                    expanded_term.set_pauli(qubit, pauli);
                }

                for qubit in hamiltonian.num_qubits..(hamiltonian.num_qubits + n) {
                    expanded_term.set_pauli(qubit, Pauli::Z);
                }

                (expanded_term, *coeff)
            })
            .collect();

        Hamiltonian::from_list(expanded_terms)
            .expect("expanded Hamiltonian terms should have consistent qubit counts")
    }

    /// Runs the denominator circuit and returns the estimated mean and variance.
    ///
    /// - `shots`: the number of shots to run the circuit.
    /// - `estimator`: an estimator that evaluates the copy-swap circuit, with no
    ///   Hamiltonian (`None`) and the provided shot count.
    ///
    /// # Returns
    ///
    /// - `(mu, var)`: the estimated mean and variance of the denominator circuit.
    pub fn run_denominator_circuit(
        &self,
        shots: usize,
        estimator: &Estimator<'_>,
    ) -> Result<(f64, f64), CircuitError> {
        let denominator_circuit = self.build_copy_swap_circuit()?;

        Ok(estimator(&denominator_circuit, None, Some(shots)))
    }

    /// Runs the numerator circuit and returns the estimated mean and variance.
    ///
    /// - `hamiltonian`: the Hamiltonian to estimate on the copy-swap circuit.
    /// - `shots`: the number of shots to run the circuit.
    /// - `estimator`: an estimator that evaluates the copy-swap circuit, the given
    ///   expanded Hamiltonian, and the provided shot count.
    ///
    /// # Returns
    ///
    /// - `(mu, var)`: the estimated mean and variance of the numerator circuit.
    pub fn run_numerator_circuit(
        &self,
        hamiltonian: &Hamiltonian,
        shots: usize,
        estimator: &Estimator<'_>,
    ) -> Result<(f64, f64), CircuitError> {
        let numerator_circuit = self.build_copy_swap_circuit()?;
        let extra_qubits = (self.copies - 1) * hamiltonian.num_qubits;
        let expanded_hamiltonian = Self::expand_hamiltonian(hamiltonian, extra_qubits);
        if expanded_hamiltonian.num_qubits != numerator_circuit.width() {
            return Err(CircuitError::QubitCountMismatch {
                expected: numerator_circuit.width(),
                actual: expanded_hamiltonian.num_qubits,
            });
        }

        Ok(estimator(
            &numerator_circuit,
            Some(&expanded_hamiltonian),
            Some(shots),
        ))
    }

    /// Runs the virtual distillation circuit and returns the mean and variance.
    ///
    /// - `hamiltonian`: a `qis::Hamiltonian` describing the observable to estimate for the original circuit.
    /// - `shots_numerator`: the number of shots to run the numerator circuit.
    /// - `shots_denominator`: the number of shots to run the denominator circuit.
    /// - `estimator`: an estimator that evaluates circuits and returns `(mu, var)`.
    ///
    /// # Returns
    ///
    /// - `mu_vd`: the mitigated result of the observable on the original circuit.
    /// - `var_vd`: the variance of the mitigated result.
    pub fn run_vd(
        &self,
        hamiltonian: &Hamiltonian,
        shots_numerator: usize,
        shots_denominator: usize,
        estimator: &Estimator<'_>,
    ) -> Result<(f64, f64), ErrorMitigationError> {
        if hamiltonian.num_qubits != self.circuit.width() {
            return Err(ErrorMitigationError::HamiltonianQubitCountMismatch {
                expected: self.circuit.width(),
                actual: hamiltonian.num_qubits,
            });
        }

        let (mu_numerator, var_numerator) =
            self.run_numerator_circuit(hamiltonian, shots_numerator, estimator)?;
        let (mu_denominator, var_denominator) =
            self.run_denominator_circuit(shots_denominator, estimator)?;
        let (mu_vd, var_vd) = self.ratio_mu_var(
            mu_numerator,
            var_numerator,
            mu_denominator,
            var_denominator,
        )?;

        Ok((mu_vd, var_vd))
    }

    fn ratio_mu_var(
        &self,
        mu_numerator: f64,
        var_numerator: f64,
        mu_denominator: f64,
        var_denominator: f64,
    ) -> Result<(f64, f64), ErrorMitigationError> {
        if mu_denominator == 0.0 {
            return Err(ErrorMitigationError::ZeroDenominatorMean);
        }

        let mu_vd = mu_numerator / mu_denominator;

        // D11 from the paper, using Taylor approximation and assuming independence
        // of numerator and denominator.
        let var_vd = var_numerator / mu_denominator.powi(2)
            + mu_numerator.powi(2) * var_denominator / mu_denominator.powi(4);
        Ok((mu_vd, var_vd))
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
