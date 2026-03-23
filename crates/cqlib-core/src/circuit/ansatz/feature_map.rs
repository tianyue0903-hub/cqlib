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

//! # Feature Map Module
//!
//! This module provides feature encoding circuits for quantum machine learning.
//! Feature maps encode classical data into quantum states, enabling quantum
//! computers to process classical information.
//!
//! ## Available Encodings
//!
//! - [`AngleEncoding`]: Simple tensor product encoding using rotation gates
//! - [`ZZFeatureMap`]: Entangling feature map with ZZ interactions

use super::traits::Ansatz;
use super::two_local::EntanglementTopology;
use crate::circuit::Instruction;
use crate::circuit::Parameter;
use crate::circuit::ParameterValue;
use crate::circuit::bit::Qubit;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::StandardGate;
use crate::qis::evolution::PauliEvolution;

/// Angle Encoding (also known as Tensor Product Encoding).
///
/// Encodes $N$ classical features into $N$ qubits by applying parameterized
/// single-qubit rotations. A common choice is applying $R_X(x_i)$ or $R_Y(x_i)$
/// to each qubit $i$.
///
/// This is the simplest feature map, mapping features independently.
#[derive(Debug, Clone)]
pub struct AngleEncoding {
    num_qubits: usize,
    rotation_gate: StandardGate,
}

impl AngleEncoding {
    /// Creates a new AngleEncoding using the specified rotation gate.
    /// Typically, `StandardGate::RX` or `StandardGate::RY` is used.
    pub fn new(num_qubits: usize, rotation_gate: StandardGate) -> Self {
        Self {
            num_qubits,
            rotation_gate,
        }
    }
}

impl Ansatz for AngleEncoding {
    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        let mut circuit = Circuit::new(self.num_qubits);

        for q in 0..self.num_qubits {
            let param_name = format!("{}_{}", prefix, q);
            let param = Parameter::try_from(param_name.as_str())
                .map_err(|_| CircuitError::InvalidParameterValue(q, f64::NAN))?;

            circuit.append(
                Instruction::Standard(self.rotation_gate),
                vec![Qubit::new(q as u32)],
                vec![ParameterValue::Param(param)],
                None,
            )?;
        }

        Ok(circuit)
    }

    fn num_parameters(&self) -> usize {
        self.num_qubits
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// ZZ Feature Map (also known as the IQP Encoding).
///
/// A second-order Pauli-Z evolution circuit widely used in Quantum Kernel Methods.
/// It encodes classical data using single-qubit Z-rotations and two-qubit ZZ-entanglement,
/// making the feature space highly non-linear and difficult to simulate classically.
///
/// For data vector $x$, a single layer applies:
/// 1. Hadamard gates on all qubits.
/// 2. $R_Z(2 \cdot x_i)$ on each qubit $i$.
/// 3. $e^{-i (2 \cdot (\pi - x_i)(\pi - x_j)) Z_i Z_j}$ for entangled pairs $(i,j)$.
///
/// Note: The specific encoding function $\phi(x)$ can vary. We use a common variation:
/// $\phi_i(x) = x_i$ for single qubits, and $\phi_{ij}(x) = (\pi - x_i)(\pi - x_j)$ for pairs.
#[derive(Debug, Clone)]
pub struct ZZFeatureMap {
    num_qubits: usize,
    reps: usize,
    entanglement: EntanglementTopology,
}

impl ZZFeatureMap {
    /// Creates a new ZZFeatureMap.
    ///
    /// By default, it uses `reps = 2` and `EntanglementTopology::Full`.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            reps: 2,
            entanglement: EntanglementTopology::Full,
        }
    }

    /// Sets the number of repetition layers.
    pub fn reps(mut self, reps: usize) -> Self {
        self.reps = reps;
        self
    }

    /// Sets the entanglement topology.
    pub fn entanglement(mut self, topology: EntanglementTopology) -> Self {
        self.entanglement = topology;
        self
    }
}

impl Ansatz for ZZFeatureMap {
    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        let mut circuit = Circuit::new(self.num_qubits);
        let qubits = circuit.qubits();

        // Prepare the base parameters [x_0, x_1, ... x_{n-1}]
        let mut x_params = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            let param_name = format!("{}_{}", prefix, i);
            let param = Parameter::try_from(param_name.as_str())
                .map_err(|_| CircuitError::InvalidParameterValue(i, f64::NAN))?;
            x_params.push(param);
        }

        // Generate the pairs based on the chosen topology
        let pairs = self.entanglement.generate_pairs(self.num_qubits);

        for _layer in 0..self.reps {
            // 1. Initial Hadamard layer
            for q in 0..self.num_qubits {
                circuit.h(Qubit::new(q as u32))?;
            }

            // 2. Single qubit phase encoding: RZ(2 * x_i)
            // U = e^{-i * x_i * Z} = RZ(2 * x_i)
            for (i, item) in x_params.iter().enumerate().take(self.num_qubits) {
                let angle = item.clone() * 2.0;
                circuit.rz(Qubit::new(i as u32), ParameterValue::Param(angle))?;
            }

            // 3. Two qubit ZZ entanglement: e^{-i * 2 * (\pi - x_i)(\pi - x_j) * Z_i Z_j}
            if !pairs.is_empty() {
                for &(i, j) in &pairs {
                    if i < self.num_qubits && j < self.num_qubits {
                        // Let phi_{ij} = (\pi - x_i) * (\pi - x_j)
                        // Evolution: R_ZZ(\theta) = e^{-i (\theta/2) Z_i Z_j}
                        // So we want \theta/2 = 2 * phi_{ij} => \theta = 4 * phi_{ij}

                        let pi_param = Parameter::pi();
                        let xi = x_params[i].clone();
                        let xj = x_params[j].clone();

                        // construct (\pi - x_i) * (\pi - x_j)
                        let phi_ij = (pi_param.clone() - xi) * (pi_param - xj);
                        let angle = phi_ij * 4.0;

                        // Using pauli evolution for ZZ
                        let mut pauli_str_chars = vec!['I'; self.num_qubits];
                        pauli_str_chars[i] = 'Z';
                        pauli_str_chars[j] = 'Z';
                        let pauli_str: String = pauli_str_chars.into_iter().collect();

                        circuit.pauli_evolution(
                            &pauli_str.parse().unwrap(),
                            ParameterValue::Param(angle),
                            &qubits,
                        )?;
                    }
                }
            }
        }

        Ok(circuit)
    }

    fn num_parameters(&self) -> usize {
        self.num_qubits
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
#[path = "feature_map_test.rs"]
mod feature_map_test;
