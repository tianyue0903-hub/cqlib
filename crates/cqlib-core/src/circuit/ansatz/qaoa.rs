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

//! # QAOA Ansatz Module
//!
//! This module implements the Quantum Approximate Optimization Algorithm (QAOA)
//! ansatz. QAOA is a variational quantum algorithm designed for solving
//! combinatorial optimization problems.
//!
//! ## Algorithm Overview
//!
//! The QAOA circuit alternates between:
//! - **Cost Layer**: Applies $e^{-i\gamma H_C}$ where $H_C$ is the problem Hamiltonian
//! - **Mixer Layer**: Applies $e^{-i\beta H_B}$ where $H_B$ is typically $\sum X_i$
//!
//! By optimizing the parameters $(\gamma, \beta)$, QAOA approximates the ground
//! state of the cost Hamiltonian.

use super::traits::Ansatz;
use crate::circuit::Parameter;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use crate::qis::evolution::PauliEvolution;
use crate::qis::hamiltonian::Hamiltonian;
use crate::qis::pauli::{Pauli, PauliString};

/// The QAOA (Quantum Approximate Optimization Algorithm) Ansatz.
///
/// QAOA is a widely used variational quantum algorithm for solving combinatorial
/// optimization problems. The ansatz consists of alternating layers of a cost
/// Hamiltonian $H_C$ and a mixer Hamiltonian $H_B$.
///
/// For a given number of layers $p$, the circuit applies:
/// $$ U(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \prod_{j=1}^{p} e^{-i \beta_j H_B} e^{-i \gamma_j H_C} $$
///
/// By default, the mixer Hamiltonian $H_B$ is the sum of Pauli X operators
/// on all qubits: $H_B = \sum_i X_i$.
#[derive(Debug, Clone)]
pub struct QAOAAnsatz {
    cost_operator: Hamiltonian,
    mixer_operator: Hamiltonian,
    reps: usize,
    initial_state: Option<Circuit>,
}

impl QAOAAnsatz {
    /// Creates a new QAOA Ansatz from a given cost operator (Hamiltonian).
    ///
    /// By default, it uses the standard X-mixer ($\sum X_i$) and `reps = 1`.
    /// The number of qubits is inferred from the cost operator.
    ///
    /// # Errors
    ///
    /// Returns `CircuitError::InvalidOperation` if there's an internal error building
    /// the default mixer Hamiltonian.
    pub fn new(mut cost_operator: Hamiltonian) -> Result<Self, CircuitError> {
        // Normalize Pauli phases into the coefficients before Hermiticity validation.
        // For example, 1.0 * (+iX) becomes i * X and is correctly rejected later,
        // while i * (-iX) becomes 1.0 * X and remains a valid Hermitian term.
        cost_operator.simplify();
        let num_qubits = cost_operator.num_qubits;

        // Build the default X-mixer: H_B = \sum_{i=0}^{n-1} X_i
        // Directly set Pauli X at each qubit index to avoid relying on
        // the string parsing convention (highest-index-first).
        let mut mixer_operator = Hamiltonian::new(num_qubits);
        for i in 0..num_qubits {
            let mut pauli = PauliString::new(num_qubits);
            pauli.set_pauli(i, Pauli::X);
            mixer_operator.add_term(pauli, 1.0.into()).map_err(|e| {
                CircuitError::InvalidOperation(format!(
                    "Failed to add term to mixer Hamiltonian: {:?}",
                    e
                ))
            })?;
        }

        Ok(Self {
            cost_operator,
            mixer_operator,
            reps: 1,
            initial_state: None,
        })
    }

    /// Sets the number of alternating layers (depth $p$).
    pub fn reps(mut self, reps: usize) -> Self {
        self.reps = reps;
        self
    }

    /// Overrides the default mixer Hamiltonian.
    ///
    /// The custom mixer must act on the same number of qubits as the cost operator.
    pub fn mixer(mut self, mut mixer_operator: Hamiltonian) -> Result<Self, CircuitError> {
        if mixer_operator.num_qubits != self.cost_operator.num_qubits {
            return Err(CircuitError::QubitCountMismatch {
                expected: self.cost_operator.num_qubits,
                actual: mixer_operator.num_qubits,
            });
        }
        mixer_operator.simplify();
        self.mixer_operator = mixer_operator;
        Ok(self)
    }

    /// Sets an initial state circuit to be prepended before the QAOA layers.
    ///
    /// By default, QAOA starts in the uniform superposition state $|+\rangle^{\otimes n}$.
    /// If an initial state is provided, it replaces the default Hadamard layer.
    pub fn initial_state(mut self, circuit: Circuit) -> Result<Self, CircuitError> {
        if circuit.num_qubits() != self.cost_operator.num_qubits {
            return Err(CircuitError::QubitCountMismatch {
                expected: self.cost_operator.num_qubits,
                actual: circuit.num_qubits(),
            });
        }
        self.initial_state = Some(circuit);
        Ok(self)
    }
}

impl Ansatz for QAOAAnsatz {
    fn validate(&self) -> Result<(), CircuitError> {
        // Validate mixer operator has same number of qubits as cost operator
        if self.mixer_operator.num_qubits != self.cost_operator.num_qubits {
            return Err(CircuitError::QubitCountMismatch {
                expected: self.cost_operator.num_qubits,
                actual: self.mixer_operator.num_qubits,
            });
        }

        // Validate initial state has same number of qubits if provided
        if let Some(initial_circuit) = &self.initial_state {
            if initial_circuit.num_qubits() != self.cost_operator.num_qubits {
                return Err(CircuitError::QubitCountMismatch {
                    expected: self.cost_operator.num_qubits,
                    actual: initial_circuit.num_qubits(),
                });
            }
        }

        // Validate cost operator has real coefficients (Hermitian requirement)
        for (pauli_str, coeff) in &self.cost_operator.terms {
            if coeff.im.abs() > 1e-10 {
                return Err(CircuitError::InvalidOperation(format!(
                    "Cost Hamiltonian coefficient for {} has non-zero imaginary part ({}). QAOA requires Hermitian Hamiltonian with real coefficients.",
                    pauli_str, coeff.im
                )));
            }
        }

        // Validate mixer operator has real coefficients
        for (pauli_str, coeff) in &self.mixer_operator.terms {
            if coeff.im.abs() > 1e-10 {
                return Err(CircuitError::InvalidOperation(format!(
                    "Mixer Hamiltonian coefficient for {} has non-zero imaginary part ({}). QAOA requires Hermitian Hamiltonian with real coefficients.",
                    pauli_str, coeff.im
                )));
            }
        }

        Ok(())
    }

    /// Builds the parameterized QAOA circuit.
    ///
    /// Parameters are generated using the provided prefix. For example, if `prefix` is "p",
    /// the parameters will be named "p_gamma_0", "p_beta_0", "p_gamma_1", "p_beta_1", etc.
    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        self.validate()?;
        // 1. Prepare initial state
        let mut circuit = if let Some(initial_circuit) = &self.initial_state {
            initial_circuit.clone()
        } else {
            let n = self.num_qubits();
            let mut c = Circuit::new(n);
            let qubits = c.qubits();
            for q in &qubits {
                c.h(*q)?;
            }
            c
        };

        let qubits = circuit.qubits();

        // 2. Apply alternating layers of Cost and Mixer
        for layer in 0..self.reps {
            // Create parameters for this layer
            let gamma_name = format!("{}_gamma_{}", prefix, layer);
            let beta_name = format!("{}_beta_{}", prefix, layer);

            let gamma_param = Parameter::try_from(gamma_name.as_str())
                .map_err(|_| CircuitError::InvalidParameterValue(layer * 2, f64::NAN))?;
            let beta_param = Parameter::try_from(beta_name.as_str())
                .map_err(|_| CircuitError::InvalidParameterValue(layer * 2 + 1, f64::NAN))?;

            // Apply e^{-i \gamma H_C}
            // For H_C = \sum c_k P_k, we evolve each term by angle = 2 * c_k * gamma
            // We use the existing pauli_evolution logic: U = e^{-i \theta/2 P}, so \theta = 2 * c_k * gamma
            for (pauli_str, coeff) in &self.cost_operator.terms {
                let term_angle = gamma_param.clone() * (2.0 * coeff.re);
                circuit.pauli_evolution(pauli_str, term_angle, &qubits)?;
            }

            // Apply e^{-i \beta H_B}
            for (pauli_str, coeff) in &self.mixer_operator.terms {
                let term_angle = beta_param.clone() * (2.0 * coeff.re);
                circuit.pauli_evolution(pauli_str, term_angle, &qubits)?;
            }
        }

        Ok(circuit)
    }

    /// QAOA requires 2 parameters per layer: one for the cost Hamiltonian ($\gamma$)
    /// and one for the mixer Hamiltonian ($\beta$).
    fn num_parameters(&self) -> usize {
        self.reps * 2
    }

    fn num_qubits(&self) -> usize {
        self.cost_operator.num_qubits
    }
}

#[cfg(test)]
#[path = "qaoa_test.rs"]
mod qaoa_test;
