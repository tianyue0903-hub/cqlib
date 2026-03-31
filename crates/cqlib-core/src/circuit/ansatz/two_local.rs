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

//! Hardware-efficient ansatze with alternating rotation and entanglement layers.
//!
//! This module provides the [`TwoLocal`] ansatz and related components. TwoLocal
//! is one of the most widely used ansatz architectures in variational quantum
//! algorithms due to its flexibility and hardware efficiency.
//!
//! # TwoLocal Architecture
//!
//! The TwoLocal ansatz consists of alternating layers:
//!
//! 1. **Rotation Layers**: Single-qubit parameterized gates (e.g., `RY`, `RZ`, `RX`).
//! 2. **Entanglement Layers**: Two-qubit gates creating correlations between qubits
//!    (e.g., `CX`, `CZ`).
//!
//! The pattern is: `[Rotation] → [Entanglement] → [Rotation] → [Entanglement] → ...`
//!
//! # Entanglement Topologies
//!
//! Different problem structures benefit from different entanglement patterns:
//!
//! - [`Linear`][`EntanglementTopology::Linear`]: Nearest-neighbor interactions.
//!   Best for near-term devices with limited connectivity.
//!
//! - [`Circular`][`EntanglementTopology::Circular`]: Linear + wrap-around edge.
//!   Adds periodic boundary conditions.
//!
//! - [`Full`][`EntanglementTopology::Full`]: All-to-all connectivity.
//!   Most expressive but requires many gates.
//!
//! - [`Custom`][`EntanglementTopology::Custom`]: User-defined qubit pairs.
//!   Allows problem-specific tailoring.
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal, EntanglementTopology};
//! use cqlib_core::circuit::gate::StandardGate;
//!
//! // Create an EfficientSU2-style ansatz
//! let ansatz = TwoLocal::new(4)
//!     .reps(3)
//!     .rotation_gates(vec![StandardGate::RY, StandardGate::RZ])
//!     .entanglement(EntanglementTopology::Circular)
//!     .entanglement_gate(StandardGate::CX);
//!
//! let circuit = ansatz.build_circuit("theta").unwrap();
//! ```

use super::traits::Ansatz;
use crate::circuit::Parameter;
use crate::circuit::ParameterValue;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::StandardGate;
use crate::circuit::{Instruction, Qubit};

/// Returns all k-combinations of `{0, 1, ..., n-1}` in lexicographic order.
fn k_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    k_combinations_helper(0, n, k, &mut combo, &mut result);
    result
}

fn k_combinations_helper(
    start: usize,
    n: usize,
    remaining: usize,
    combo: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(combo.clone());
        return;
    }
    for i in start..=(n - remaining) {
        combo.push(i);
        k_combinations_helper(i + 1, n, remaining - 1, combo, result);
        combo.pop();
    }
}

/// Defines the topology of the entanglement layer in a TwoLocal ansatz.
///
/// The entanglement topology determines which pairs of qubits are connected
/// by two-qubit gates in each entanglement layer. This choice significantly
/// affects the expressibility of the ansatz and the hardware resources required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntanglementTopology {
    /// Linear nearest-neighbor entanglement.
    ///
    /// Creates entanglement between adjacent qubits: (0,1), (1,2), ..., (n-2, n-1).
    /// This topology is hardware-efficient for devices with linear connectivity.
    Linear,
    /// Circular nearest-neighbor entanglement.
    ///
    /// Like [`Linear`](Self::Linear) but adds a wrap-around edge (n-1, 0).
    /// Creates a ring topology suitable for periodic boundary conditions.
    /// For 2 qubits, this is identical to Linear.
    Circular,
    /// Full all-to-all entanglement.
    ///
    /// Creates entanglement between every pair of qubits.
    /// Most expressive but requires O(n²) gates per layer.
    Full,
    /// Custom explicit entanglement pairs.
    ///
    /// Allows users to specify exactly which qubit pairs should be entangled.
    /// Useful for problem-specific ansatz design or hardware-aware compilation.
    Custom(Vec<(usize, usize)>),
}

impl EntanglementTopology {
    /// Generates all k-tuples of qubit indices based on the entanglement topology.
    ///
    /// This generalizes [`generate_pairs`](Self::generate_pairs) (k=2) to arbitrary
    /// locality k. It is used by [`PauliFeatureMap`][super::feature_map::PauliFeatureMap]
    /// to apply k-local Pauli evolution gates.
    ///
    /// # Behavior per topology
    ///
    /// | Topology | k=1 | k=2 | k≥3 |
    /// |----------|-----|-----|-----|
    /// | Linear | each qubit | (0,1),(1,2),… | consecutive windows of width k |
    /// | Circular | each qubit | linear + wrap | all rotational windows mod n |
    /// | Full | each qubit | all C(n,2) | all C(n,k) |
    /// | Custom | each qubit | custom pairs | falls back to Full (C(n,k)) |
    ///
    /// Returns an empty vec if `k == 0`, `num_qubits == 0`, or `k > num_qubits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::EntanglementTopology;
    ///
    /// let full = EntanglementTopology::Full;
    /// let triples = full.generate_k_tuples(3, 4);
    /// assert_eq!(triples, vec![
    ///     vec![0, 1, 2],
    ///     vec![0, 1, 3],
    ///     vec![0, 2, 3],
    ///     vec![1, 2, 3],
    /// ]);
    ///
    /// let linear = EntanglementTopology::Linear;
    /// let triples = linear.generate_k_tuples(3, 4);
    /// assert_eq!(triples, vec![vec![0, 1, 2], vec![1, 2, 3]]);
    /// ```
    pub fn generate_k_tuples(&self, k: usize, num_qubits: usize) -> Vec<Vec<usize>> {
        if k == 0 || num_qubits == 0 || k > num_qubits {
            return vec![];
        }
        if k == 1 {
            return (0..num_qubits).map(|i| vec![i]).collect();
        }
        if k == 2 {
            return self
                .generate_pairs(num_qubits)
                .into_iter()
                .map(|(a, b)| vec![a, b])
                .collect();
        }
        // k >= 3
        match self {
            EntanglementTopology::Linear => {
                (0..=num_qubits - k).map(|i| (i..i + k).collect()).collect()
            }
            // For Circular k≥3, generates n rotational windows: [i, i+1, ..., i+k-1] (mod n).
            // Note: when k == n, all n windows contain the same set of qubits (just rotated),
            // so they produce the same Pauli string and each applies an identical evolution gate.
            // This rotational symmetry is intentional for periodic-boundary feature maps.
            EntanglementTopology::Circular => (0..num_qubits)
                .map(|i| (0..k).map(|j| (i + j) % num_qubits).collect())
                .collect(),
            // Custom topology only defines pairs; fall back to Full for k > 2.
            EntanglementTopology::Full | EntanglementTopology::Custom(_) => {
                k_combinations(num_qubits, k)
            }
        }
    }

    /// Generates a list of qubit pairs based on the specified topology.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The total number of qubits in the circuit.
    ///
    /// # Returns
    ///
    /// A vector of tuples `(control, target)` representing qubit pairs to entangle.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::EntanglementTopology;
    ///
    /// let topology = EntanglementTopology::Linear;
    /// let pairs = topology.generate_pairs(4);
    /// assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
    ///
    /// let circular = EntanglementTopology::Circular;
    /// let pairs = circular.generate_pairs(4);
    /// assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    /// ```
    pub fn generate_pairs(&self, num_qubits: usize) -> Vec<(usize, usize)> {
        match self {
            EntanglementTopology::Linear => (0..num_qubits.saturating_sub(1))
                .map(|i| (i, i + 1))
                .collect(),
            EntanglementTopology::Circular => {
                let mut p: Vec<(usize, usize)> = (0..num_qubits.saturating_sub(1))
                    .map(|i| (i, i + 1))
                    .collect();
                if num_qubits > 2 {
                    p.push((num_qubits - 1, 0));
                }
                p
            }
            EntanglementTopology::Full => {
                let mut p = Vec::new();
                for i in 0..num_qubits {
                    for j in (i + 1)..num_qubits {
                        p.push((i, j));
                    }
                }
                p
            }
            EntanglementTopology::Custom(pairs) => pairs.clone(),
        }
    }
}

/// The TwoLocal ansatz, a versatile hardware-efficient parameterized circuit.
///
/// TwoLocal is a popular ansatz architecture consisting of alternating layers
/// of single-qubit rotations and two-qubit entanglement gates. It serves as
/// the foundation for many common ansatz patterns including RealAmplitudes
/// and EfficientSU2.
///
/// # Architecture
///
/// The circuit structure follows this pattern:
///
/// ```text
/// Layer 0:    [R]───────[R]───────[R]───────[R]    (Rotation)
///               │       │       │       │
/// Layer 0.5:   └─[E]───┘       └─[E]───┘        (Entanglement)
///
/// Layer 1:    [R]───────[R]───────[R]───────[R]
///               │       │       │       │
/// Layer 1.5:   └─[E]───┘       └─[E]───┘
/// ...
/// ```
///
/// Where `[R]` represents parameterized rotation gates and `[E]` represents
/// entanglement gates.
///
/// # Configuration
///
/// Use the builder pattern to configure the ansatz:
///
/// - [`reps`](Self::reps): Number of repetition layers
/// - [`rotation_gates`](Self::rotation_gates): Single-qubit gates (e.g., `[RY]`, `[RY, RZ]`)
/// - [`entanglement_gate`](Self::entanglement_gate): Two-qubit gate (e.g., `CX`, `CZ`)
/// - [`entanglement`](Self::entanglement): Connectivity pattern
/// - [`skip_final_rotation_layer`](Self::skip_final_rotation_layer): Omit final rotation
///
/// # Parameter Count
///
/// The total number of parameters is:
/// `layers × num_qubits × rotation_gates.len()`
///
/// where `layers = reps` if `skip_final_rotation_layer` is `true`,
/// otherwise `layers = reps + 1`.
#[derive(Debug, Clone)]
pub struct TwoLocal {
    num_qubits: usize,
    reps: usize,
    rotation_gates: Vec<StandardGate>,
    entanglement_gate: StandardGate,
    entanglement: EntanglementTopology,
    skip_final_rotation_layer: bool,
}

impl TwoLocal {
    /// Creates a new TwoLocal ansatz with sensible defaults.
    ///
    /// # Default Configuration
    ///
    /// - `reps`: 3 layers
    /// - `rotation_gates`: `[RY]`
    /// - `entanglement_gate`: `CX`
    /// - `entanglement`: [`Linear`](EntanglementTopology::Linear)
    /// - `skip_final_rotation_layer`: false
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the ansatz.
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal};
    ///
    /// let ansatz = TwoLocal::new(5);
    /// assert_eq!(ansatz.num_qubits(), 5);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            reps: 3,
            rotation_gates: vec![StandardGate::RY],
            entanglement_gate: StandardGate::CX,
            entanglement: EntanglementTopology::Linear,
            skip_final_rotation_layer: false,
        }
    }

    /// Sets the number of repetition layers.
    ///
    /// Each repetition consists of one rotation layer followed by one
    /// entanglement layer. The default is 3.
    ///
    /// # Arguments
    ///
    /// * `reps` - The number of repetitions (must be >= 0).
    pub fn reps(mut self, reps: usize) -> Self {
        self.reps = reps;
        self
    }

    /// Sets the rotation gates used in the rotation layers.
    ///
    /// Each qubit receives all specified rotation gates in sequence.
    /// Valid choices are single-parameter single-qubit rotation gates:
    /// - `[RX]`, `[RY]`, `[RZ]` individually
    /// - `[RY, RZ]` for EfficientSU2 ansatz
    /// - `[RX, RY, RZ]` for full SU(2) coverage
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of single-qubit parameterized gates (RX, RY, or RZ).
    ///
    /// # Errors
    ///
    /// Validation is performed in [`build_circuit`](Ansatz::build_circuit).
    /// Returns `CircuitError::InvalidOperation` if any gate is not RX, RY, or RZ.
    pub fn rotation_gates(mut self, gates: Vec<StandardGate>) -> Self {
        self.rotation_gates = gates;
        self
    }

    /// Sets the entanglement gate used in the entanglement layers.
    ///
    /// Valid choices are two-qubit controlled gates that create entanglement:
    /// - `CX` (CNOT): Standard choice, universal for entanglement (default)
    /// - `CY`: Controlled-Y gate
    /// - `CZ`: Symmetric, native on some hardware
    ///
    /// # Arguments
    ///
    /// * `gate` - A two-qubit controlled entanglement gate (CX, CY, or CZ).
    ///
    /// # Errors
    ///
    /// Validation is performed in [`build_circuit`](Ansatz::build_circuit).
    /// Returns `CircuitError::InvalidOperation` if the gate is not CX, CY, or CZ.
    pub fn entanglement_gate(mut self, gate: StandardGate) -> Self {
        self.entanglement_gate = gate;
        self
    }

    /// Sets the entanglement topology.
    ///
    /// See [`EntanglementTopology`] for available options.
    ///
    /// # Arguments
    ///
    /// * `topology` - The desired connectivity pattern.
    pub fn entanglement(mut self, topology: EntanglementTopology) -> Self {
        self.entanglement = topology;
        self
    }

    /// Sets whether to skip the final rotation layer.
    ///
    /// The default TwoLocal architecture adds a final rotation layer after
    /// the last entanglement layer (total `reps + 1` rotation layers).
    /// When enabled, this final rotation layer is omitted, meaning the
    /// ansatz ends with an entanglement layer and has exactly `reps`
    /// rotation layers.
    ///
    /// # Arguments
    ///
    /// * `skip` - If `true`, omit the final rotation layer.
    pub fn skip_final_rotation_layer(mut self, skip: bool) -> Self {
        self.skip_final_rotation_layer = skip;
        self
    }
}

impl Ansatz for TwoLocal {
    fn validate(&self) -> Result<(), CircuitError> {
        if self.num_qubits == 0 {
            return Err(CircuitError::InvalidOperation(
                "TwoLocal requires at least 1 qubit".to_string(),
            ));
        }

        // Validate rotation gates: must be RX, RY, or RZ
        for gate in &self.rotation_gates {
            if !matches!(gate, StandardGate::RX | StandardGate::RY | StandardGate::RZ) {
                return Err(CircuitError::InvalidOperation(format!(
                    "Rotation gates must be RX, RY, or RZ, got {:?}",
                    gate
                )));
            }
        }

        // Validate entanglement gate: must be CX, CY, or CZ
        if !matches!(
            self.entanglement_gate,
            StandardGate::CX | StandardGate::CY | StandardGate::CZ
        ) {
            return Err(CircuitError::InvalidOperation(format!(
                "Entanglement gate must be CX, CY, or CZ, got {:?}",
                self.entanglement_gate
            )));
        }

        // Validate Custom topology pairs
        if let EntanglementTopology::Custom(pairs) = &self.entanglement {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for (c, t) in pairs {
                // Check for self-loop
                if c == t {
                    return Err(CircuitError::InvalidOperation(format!(
                        "EntanglementTopology::Custom contains self-loop ({}, {})",
                        c, t
                    )));
                }
                // Check index bounds
                if *c >= self.num_qubits || *t >= self.num_qubits {
                    return Err(CircuitError::InvalidOperation(format!(
                        "EntanglementTopology::Custom contains out-of-bounds index: ({}, {}) for {} qubits",
                        c, t, self.num_qubits
                    )));
                }
                // Check for duplicate edges (undirected: (c,t) == (t,c))
                let edge = if c < t { (*c, *t) } else { (*t, *c) };
                if !seen.insert(edge) {
                    return Err(CircuitError::InvalidOperation(format!(
                        "EntanglementTopology::Custom contains duplicate edge ({}, {})",
                        c, t
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
        let mut param_idx = 0;

        let num_layers = if self.skip_final_rotation_layer {
            self.reps
        } else {
            self.reps + 1
        };

        for layer in 0..num_layers {
            // Rotation Layer
            for q in 0..self.num_qubits {
                for gate in &self.rotation_gates {
                    let param_name = format!("{}_{}", prefix, param_idx);
                    let param = Parameter::try_from(param_name.as_str())
                        .map_err(|_| CircuitError::InvalidParameterValue(param_idx, f64::NAN))?;

                    circuit.append(
                        Instruction::Standard(*gate),
                        vec![Qubit::new(q as u32)],
                        vec![ParameterValue::Param(param)],
                        None,
                    )?;
                    param_idx += 1;
                }
            }

            // Entanglement Layer
            if layer < self.reps {
                let pairs = self.entanglement.generate_pairs(self.num_qubits);

                for (c, t) in pairs {
                    circuit.append(
                        Instruction::Standard(self.entanglement_gate),
                        vec![Qubit::new(c as u32), Qubit::new(t as u32)],
                        vec![],
                        None,
                    )?;
                }
            }
        }

        Ok(circuit)
    }

    fn num_parameters(&self) -> usize {
        let layers = if self.skip_final_rotation_layer {
            self.reps
        } else {
            self.reps + 1
        };
        layers * self.num_qubits * self.rotation_gates.len()
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
