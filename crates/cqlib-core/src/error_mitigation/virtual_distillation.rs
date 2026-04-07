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

    /// Returns the configured number of copies.
    pub fn copies(&self) -> usize {
        self.copies
    }

    /// Updates the configured number of copies.
    pub fn set_copies(&mut self, copies: usize) -> Result<(), VirtualDistillationError> {
        if copies < 2 {
            return Err(VirtualDistillationError::InvalidCopies(copies));
        }

        self.copies = copies;
        Ok(())
    }

    /// Builds a copy-swap circuit from the configured base circuit.
    ///
    /// The returned circuit contains:
    /// - `copies` disjoint copies of the base circuit preparation,
    /// - pairwise SWAP operations between the first copy and every additional copy.
    pub fn build_copy_swap_circuit(
        &self,
        observable_circ: Option<Circuit>,
    ) -> Result<Circuit, CircuitError> {
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

        if let Some(observable_circ) = observable_circ {
            if observable_circ.width() != base_width {
                return Err(CircuitError::QubitCountMismatch {
                    expected: base_width,
                    actual: observable_circ.width(),
                });
            }

            let observable_circ = observable_circ.decompose()?;
            Self::append_circuit_with_offset(&mut copy_swap_circuit, &observable_circ, 0)?;
        }

        Ok(copy_swap_circuit)
    }

    /// Runs the denominator circuit and returns the eigenvalues.
    ///
    /// - `shots`: the number of shots to run the circuit.
    /// - `eigen_calc`: a backend that can run the circuit and return the eigenvalues.
    ///
    /// # Returns
    ///
    /// - `eigen_values`: a vector of eigenvalues.
    pub fn run_denominator_circuit<F>(
        &self,
        shots: usize,
        eigen_calc: F,
    ) -> Result<Vec<f64>, CircuitError>
    where
        F: Fn(&Circuit, usize) -> Vec<f64>,
    {
        let denominator_circuit = self.build_copy_swap_circuit(None)?;

        Ok(eigen_calc(&denominator_circuit, shots))
    }

    /// Runs the numerator circuit and returns the eigenvalues.
    ///
    /// - `observable_circ`: a pauli observable represented as a basis transformation quantum circuit.
    /// - `shots`: the number of shots to run the circuit.
    /// - `eigen_calc`: a backend that can run the circuit and return the eigenvalues.
    ///
    /// # Returns
    ///
    /// - `eigen_values`: a vector of eigenvalues.
    pub fn run_numerator_circuit<F>(
        &self,
        observable_circ: Circuit,
        shots: usize,
        eigen_calc: F,
    ) -> Result<Vec<f64>, CircuitError>
    where
        F: Fn(&Circuit, usize) -> Vec<f64>,
    {
        let numerator_circuit = self.build_copy_swap_circuit(Some(observable_circ))?;

        Ok(eigen_calc(&numerator_circuit, shots))
    }

    /// Runs the virtual distillation circuit and returns the mean and variance.
    ///
    /// - `observables`: a vector of pauli observables expressed as basis transformation quantum circuits.
    /// - `coefficients`: a vector of coefficients for the observables.
    /// - `shots_numerator`: the number of shots to run the numerator circuit.
    /// - `shots_denominator`: the number of shots to run the denominator circuit.
    /// - `eigen_calc`: a backend that can run the circuit and return the eigenvalues.
    ///
    /// # Returns
    ///
    /// - `mu_vd`: the mitigated result of the observable on the original circuit.
    /// - `var_vd`: the variance of the mitigated result.
    pub fn run_vd<F>(
        &self,
        observables: Vec<Circuit>,
        coefficients: Vec<f64>,
        shots_numerator: usize,
        shots_denominator: usize,
        eigen_calc: F,
    ) -> Result<(f64, f64), CircuitError>
    where
        F: Fn(&Circuit, usize) -> Vec<f64>,
    {
        Self::validate_sample_count("shots_numerator", shots_numerator)?;
        Self::validate_sample_count("shots_denominator", shots_denominator)?;

        if observables.len() != coefficients.len() {
            return Err(CircuitError::InvalidOperation(
                "The number of observables and coefficients must be the same".to_string(),
            ));
        }

        let mut weighted_numerator = vec![0.0; shots_numerator];

        for (observable, coefficient) in observables.into_iter().zip(coefficients.into_iter()) {
            let eigen_values =
                self.run_numerator_circuit(observable, shots_numerator, &eigen_calc)?;

            Self::validate_eigen_vector_length(
                "Numerator",
                &eigen_values,
                shots_numerator,
            )?;

            for (weighted_value, eigen_value) in
                weighted_numerator.iter_mut().zip(eigen_values.into_iter())
            {
                *weighted_value += coefficient * eigen_value;
            }
        }

        let denominator_data = self.run_denominator_circuit(shots_denominator, &eigen_calc)?;
        Self::validate_eigen_vector_length(
            "Denominator",
            &denominator_data,
            shots_denominator,
        )?;

        let (mu_numerator, var_numerator) = self.get_mu_var(&weighted_numerator);
        let (mu_denominator, var_denominator) = self.get_mu_var(&denominator_data);
        let (mu_vd, var_vd) = self.ratio_mu_var(
            mu_numerator,
            var_numerator,
            mu_denominator,
            var_denominator,
        )?;

        Ok((mu_vd, var_vd))
    }

    fn validate_sample_count(name: &str, sample_count: usize) -> Result<(), CircuitError> {
        if sample_count == 0 {
            return Err(CircuitError::InvalidOperation(format!(
                "{name} must be greater than 0"
            )));
        }

        Ok(())
    }

    fn validate_eigen_vector_length(
        kind: &str,
        data: &[f64],
        expected_len: usize,
    ) -> Result<(), CircuitError> {
        if data.len() != expected_len {
            return Err(CircuitError::InvalidOperation(format!(
                "{kind} eigenvalue vector length mismatch: expected {expected_len}, got {}",
                data.len()
            )));
        }

        Ok(())
    }

    fn get_mu_var(&self, data: &[f64]) -> (f64, f64) {
        let mu = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / data.len() as f64;
        (mu, var)
    }

    fn ratio_mu_var(
        &self,
        mu_numerator: f64,
        var_numerator: f64,
        mu_denominator: f64,
        var_denominator: f64,
    ) -> Result<(f64, f64), CircuitError> {
        if mu_denominator == 0.0 {
            return Err(CircuitError::InvalidOperation(
                "Virtual distillation denominator mean is zero".to_string(),
            ));
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
