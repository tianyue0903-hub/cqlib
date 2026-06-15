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
//! - [`PauliFeatureMap`]: General-purpose feature map with arbitrary Pauli strings

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
use crate::qis::pauli::{Pauli, PauliString};

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
    fn validate(&self) -> Result<(), CircuitError> {
        if self.num_qubits == 0 {
            return Err(CircuitError::InvalidOperation(
                "AngleEncoding requires at least 1 qubit".to_string(),
            ));
        }
        // Only single-parameter single-qubit rotation gates are valid
        if !matches!(
            self.rotation_gate,
            StandardGate::RX | StandardGate::RY | StandardGate::RZ | StandardGate::Phase
        ) {
            return Err(CircuitError::InvalidOperation(format!(
                "AngleEncoding rotation_gate must be a single-parameter single-qubit gate \
                 (RX, RY, RZ, or Phase), got {:?}",
                self.rotation_gate
            )));
        }
        Ok(())
    }

    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        self.validate()?;

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
    fn validate(&self) -> Result<(), CircuitError> {
        if self.num_qubits == 0 {
            return Err(CircuitError::InvalidOperation(
                "ZZFeatureMap requires at least 1 qubit".to_string(),
            ));
        }
        if let EntanglementTopology::Custom(pairs) = &self.entanglement {
            use std::collections::HashSet;
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for (i, j) in pairs {
                if *i >= self.num_qubits || *j >= self.num_qubits {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains out-of-bounds index: \
                         ({}, {}) for {} qubits",
                        i, j, self.num_qubits
                    )));
                }
                if i == j {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains self-loop ({}, {})",
                        i, j
                    )));
                }
                // Undirected duplicate check
                let edge = if i < j { (*i, *j) } else { (*j, *i) };
                if !seen.insert(edge) {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains duplicate edge ({}, {})",
                        i, j
                    )));
                }
            }
        }
        Ok(())
    }

    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        self.validate()?;

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
                        let mut pauli_str = PauliString::new(self.num_qubits);
                        pauli_str.set_pauli(i, Pauli::Z);
                        pauli_str.set_pauli(j, Pauli::Z);

                        circuit.pauli_evolution(
                            &pauli_str,
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
        if self.reps == 0 {
            return 0;
        }
        self.num_qubits
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Pauli Feature Map: A general-purpose data encoding circuit using Pauli evolution.
///
/// The PauliFeatureMap encodes classical data into quantum states through parameterized
/// Pauli evolution gates. It supports arbitrary Pauli strings (e.g., "Z", "ZZ", "ZZZ")
/// and flexible entanglement topologies.
///
/// # Mathematical Foundation
///
/// For a k-local Pauli template P and feature vector x, the encoding applies:
///
/// | Locality | Evolution | Angle θ (passed to `pauli_evolution`) |
/// |----------|-----------|----------------------------------------|
/// | k = 1    | $e^{-i x_i P}$ | $2 x_i$ |
/// | k ≥ 2    | $e^{-i 2 \prod_{j \in S}(\pi - x_j) P}$ | $4 \prod_{j \in S}(\pi - x_j)$ |
///
/// where `pauli_evolution(θ)` implements $e^{-i \frac{\theta}{2} P}$.
///
/// The non-linear k-local mapping $\prod_{j \in S}(\pi - x_j)$ creates a rich
/// feature space through the kernel trick, making the kernel function hard to
/// evaluate classically for large circuits.
///
/// # Architecture
///
/// The circuit structure for each repetition layer:
///
/// 1. **Hadamard Layer**: Apply $H$ gates to all qubits to create superposition.
/// 2. **k-local Pauli Evolution**: For each Pauli template P and each k-tuple
///    of qubit indices $(q_0, \ldots, q_{k-1})$ from the entanglement topology,
///    apply $e^{-i \phi(x_{q_0}, \ldots, x_{q_{k-1}}) P}$.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::ansatz::{Ansatz, PauliFeatureMap, EntanglementTopology};
/// use cqlib_core::qis::pauli::PauliString;
///
/// // 3-qubit feature map with Z (1-local) + ZZ (2-local) + ZZZ (3-local)
/// let feature_map = PauliFeatureMap::new(3)
///     .reps(1)
///     .paulis(vec![
///         (PauliString::from("Z"),   "Z".to_string()),
///         (PauliString::from("ZZ"),  "ZZ".to_string()),
///         (PauliString::from("ZZZ"), "ZZZ".to_string()),
///     ])
///     .entanglement(EntanglementTopology::Full);
///
/// let circuit = feature_map.build_circuit("x").unwrap();
/// assert_eq!(feature_map.num_parameters(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct PauliFeatureMap {
    /// Number of qubits (features) in the feature map.
    num_qubits: usize,
    /// Number of repetition layers for the encoding.
    reps: usize,
    /// List of Pauli strings to use for evolution, with their labels.
    paulis: Vec<(PauliString, String)>,
    /// Entanglement topology for 2-local interactions.
    entanglement: EntanglementTopology,
    /// Prefix for parameter names (default: "x").
    parameter_prefix: String,
}

impl PauliFeatureMap {
    /// Creates a new PauliFeatureMap with default configuration.
    ///
    /// # Default Configuration
    ///
    /// - `reps`: 2 layers
    /// - `paulis`: ["Z", "ZZ"] (single-qubit Z and two-qubit ZZ interactions)
    /// - `entanglement`: [`Full`](EntanglementTopology::Full) all-to-all connectivity
    /// - `parameter_prefix`: "x"
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits (features) in the feature map.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cqlib_core::circuit::ansatz::{Ansatz, PauliFeatureMap};
    ///
    /// let fm = PauliFeatureMap::new(4);
    /// assert_eq!(fm.num_qubits(), 4);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            reps: 2,
            paulis: vec![
                (PauliString::from("Z"), "Z".to_string()),
                (PauliString::from("ZZ"), "ZZ".to_string()),
            ],
            entanglement: EntanglementTopology::Full,
            parameter_prefix: "x".to_string(),
        }
    }

    /// Sets the number of repetition layers.
    ///
    /// Each repetition applies the full encoding sequence (Hadamard + Pauli evolutions).
    /// More repetitions increase the expressiveness of the feature map but also
    /// increase circuit depth.
    ///
    /// # Arguments
    ///
    /// * `reps` - The number of repetition layers.
    pub fn reps(mut self, reps: usize) -> Self {
        self.reps = reps;
        self
    }

    /// Sets the Pauli strings to use for evolution.
    ///
    /// Each Pauli string specifies a type of interaction:
    /// - Single-character strings (e.g., "Z", "X", "Y") create 1-local interactions.
    /// - Multi-character strings (e.g., "ZZ", "XY") create multi-qubit interactions.
    ///
    /// # Arguments
    ///
    /// * `paulis` - A vector of `(PauliString, label)` tuples.
    pub fn paulis(mut self, paulis: Vec<(PauliString, String)>) -> Self {
        self.paulis = paulis;
        self
    }

    /// Sets the entanglement topology for 2-local interactions.
    ///
    /// The topology determines which qubit pairs are connected by 2-local
    /// Pauli evolution gates.
    ///
    /// # Arguments
    ///
    /// * `topology` - The entanglement topology.
    pub fn entanglement(mut self, topology: EntanglementTopology) -> Self {
        self.entanglement = topology;
        self
    }

    /// Sets the parameter name prefix.
    ///
    /// Parameter names are generated as `{prefix}_{index}` (e.g., "x_0", "x_1").
    ///
    /// # Arguments
    ///
    /// * `prefix` - The parameter name prefix.
    pub fn parameter_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.parameter_prefix = prefix.into();
        self
    }
}

impl Ansatz for PauliFeatureMap {
    fn validate(&self) -> Result<(), CircuitError> {
        // Check that num_qubits > 0
        if self.num_qubits == 0 {
            return Err(CircuitError::InvalidOperation(
                "PauliFeatureMap requires at least 1 qubit".to_string(),
            ));
        }

        // Check that pauli strings have length <= num_qubits
        for (pauli_str, _) in &self.paulis {
            if pauli_str.num_qubits > self.num_qubits {
                return Err(CircuitError::InvalidOperation(format!(
                    "Pauli string '{}' has length {} which exceeds num_qubits {}",
                    pauli_str, pauli_str.num_qubits, self.num_qubits
                )));
            }
        }

        // If Custom topology, validate all indices and reject ambiguous edges.
        if let EntanglementTopology::Custom(pairs) = &self.entanglement {
            use std::collections::HashSet;
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for (i, j) in pairs {
                if *i >= self.num_qubits || *j >= self.num_qubits {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains out-of-bounds index: ({}, {}) for {} qubits",
                        i, j, self.num_qubits
                    )));
                }
                if i == j {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains self-loop ({}, {})",
                        i, j
                    )));
                }
                let edge = if i < j { (*i, *j) } else { (*j, *i) };
                if !seen.insert(edge) {
                    return Err(CircuitError::InvalidOperation(format!(
                        "Custom entanglement topology contains duplicate edge ({}, {})",
                        i, j
                    )));
                }
            }
        }

        Ok(())
    }

    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError> {
        // Validate configuration first
        self.validate()?;

        let mut circuit = Circuit::new(self.num_qubits);
        let qubits = circuit.qubits();

        // Use provided prefix if non-empty, otherwise use self.parameter_prefix
        let effective_prefix = if prefix.is_empty() {
            &self.parameter_prefix
        } else {
            prefix
        };

        // Prepare the base parameters [x_0, x_1, ..., x_{n-1}]
        let mut x_params = Vec::with_capacity(self.num_qubits);
        for i in 0..self.num_qubits {
            let param_name = format!("{}_{}", effective_prefix, i);
            let param = Parameter::try_from(param_name.as_str())
                .map_err(|_| CircuitError::InvalidParameterValue(i, f64::NAN))?;
            x_params.push(param);
        }

        // Apply the encoding for each repetition layer
        for _layer in 0..self.reps {
            // Step 1: Initial Hadamard layer to create superposition
            for q in 0..self.num_qubits {
                circuit.h(Qubit::new(q as u32))?;
            }

            // Step 2: Apply Pauli evolution for each Pauli string
            for (pauli_str, _) in &self.paulis {
                // Determine the support (non-identity positions) of the template Pauli string.
                let support = pauli_str.support();
                let k = support.len();

                // An all-identity Pauli string contributes only a global phase; skip it.
                if k == 0 {
                    continue;
                }

                // Extract the Pauli operator type at each support position.
                let template_ops: Vec<Pauli> = support
                    .iter()
                    .map(|&idx| pauli_str.get_pauli(idx))
                    .collect();

                // Generate all k-tuples of circuit qubit indices for this k-local interaction.
                // For k=1: each single qubit independently.
                // For k≥2: k-tuples from the chosen entanglement topology.
                let tuples = self.entanglement.generate_k_tuples(k, self.num_qubits);

                for tuple in &tuples {
                    // Build a full-length PauliString that places each template operator
                    // at the corresponding circuit qubit position in the tuple.
                    let mut full_pauli = PauliString::new(self.num_qubits);
                    for (pos, &qubit_idx) in tuple.iter().enumerate() {
                        full_pauli.set_pauli(qubit_idx, template_ops[pos]);
                    }

                    // Compute the encoding angle θ passed to `pauli_evolution`,
                    // which implements e^{-i θ/2 P}:
                    //
                    //   k=1: θ = 2·x_i        →  e^{-i x_i P}
                    //   k≥2: θ = 4·∏(π−x_j)  →  e^{-i 2·∏(π−x_j) P}
                    let angle = if k == 1 {
                        x_params[tuple[0]].clone() * 2.0
                    } else {
                        let pi = Parameter::pi();
                        let mut product = pi.clone() - x_params[tuple[0]].clone();
                        for &qi in &tuple[1..] {
                            product = product * (pi.clone() - x_params[qi].clone());
                        }
                        product * 4.0
                    };

                    circuit.pauli_evolution(&full_pauli, ParameterValue::Param(angle), &qubits)?;
                }
            }
        }

        Ok(circuit)
    }

    fn num_parameters(&self) -> usize {
        // The number of parameters equals the number of features (qubits)
        // Each qubit has one input feature parameter x_i
        if self.reps == 0 {
            return 0;
        }
        self.num_qubits
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
#[path = "feature_map_test.rs"]
mod feature_map_test;
