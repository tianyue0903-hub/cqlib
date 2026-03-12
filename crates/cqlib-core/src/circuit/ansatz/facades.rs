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

//! Convenient constructor functions for common ansatz patterns.
//!
//! This module provides high-level factory functions that create pre-configured
//! [`TwoLocal`] ansatze for common use cases in variational quantum algorithms.
//!
//! # Available Ansatze
//!
//! - [`real_amplitudes`]: Ansatz with real-valued amplitudes, suitable for chemistry
//!   and optimization problems where the wavefunction can be constrained to real numbers.
//!
//! - [`efficient_su2`]: Hardware-efficient ansatz spanning SU(2), widely used in
//!   VQE and Quantum Machine Learning.
//!
//! # Comparison
//!
//! | Ansatz | Rotation Gates | Entanglement | Use Case |
//! |--------|---------------|--------------|----------|
//! | RealAmplitudes | RY | CX | Chemistry, QAOA with real amplitudes |
//! | EfficientSU2 | RY, RZ | CX | General VQE, QML |
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::ansatz::{real_amplitudes, efficient_su2, EntanglementTopology, Ansatz};
//!
//! // Create a RealAmplitudes ansatz for 4 qubits
//! let ra = real_amplitudes(4, 3, EntanglementTopology::Linear);
//! assert_eq!(ra.num_parameters(), 16); // 4 qubits * 4 layers
//!
//! // Create an EfficientSU2 ansatz
//! let su2 = efficient_su2(3, 2, EntanglementTopology::Full);
//! assert_eq!(su2.num_parameters(), 18); // 3 qubits * 2 gates * 3 layers
//! ```

use super::two_local::{EntanglementTopology, TwoLocal};
use crate::circuit::gate::StandardGate;

/// Creates a RealAmplitudes ansatz.
///
/// This is a heuristic trial wave function used as an ansatz in chemistry applications or
/// QAOA where the wave function is constrained to have real amplitudes (i.e., no imaginary parts).
///
/// The ansatz consists of alternating layers of $R_Y$ rotations and $CX$ entanglements.
/// This structure ensures that the resulting statevector has only real components when
/// starting from $|0\rangle^{\otimes n}$.
///
/// # Mathematical Structure
///
/// For $n$ qubits and $r$ repetitions:
///
/// $$|\psi(\theta)\rangle = \left( \prod_{i=0}^{n-1} R_Y(\theta_{r,i}) \right) \prod_{l=0}^{r-1} \left[ \left( \prod_{(i,j) \in E} CX_{i,j} \right) \left( \prod_{i=0}^{n-1} R_Y(\theta_{l,i}) \right) \right] |0\rangle^{\otimes n}$$
///
/// where $E$ is the set of edges defined by the entanglement topology.
///
/// # Arguments
///
/// * `num_qubits` - The number of qubits in the ansatz. Must be at least 1.
/// * `reps` - The number of repetition layers. Each repetition adds a rotation
///   layer and an entanglement layer. The final rotation layer is always included.
/// * `entanglement` - The topology of the entanglement (e.g., Linear, Full).
///   See [`EntanglementTopology`] for options.
///
/// # Returns
///
/// A [`TwoLocal`] instance configured for Real Amplitudes with:
/// - Rotation gates: `[RY]`
/// - Entanglement gate: `CX`
/// - Total parameters: `(reps + 1) × num_qubits`
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::ansatz::{real_amplitudes, EntanglementTopology, Ansatz};
///
/// // 3 qubits, 2 reps, linear entanglement
/// let ansatz = real_amplitudes(3, 2, EntanglementTopology::Linear);
///
/// assert_eq!(ansatz.num_qubits(), 3);
/// assert_eq!(ansatz.num_parameters(), 9); // (2+1) * 3 = 9
///
/// let circuit = ansatz.build_circuit("theta").unwrap();
/// ```
pub fn real_amplitudes(
    num_qubits: usize,
    reps: usize,
    entanglement: EntanglementTopology,
) -> TwoLocal {
    TwoLocal::new(num_qubits)
        .reps(reps)
        .rotation_gates(vec![StandardGate::RY])
        .entanglement_gate(StandardGate::CX)
        .entanglement(entanglement)
}

/// Creates an EfficientSU2 ansatz.
///
/// A hardware-efficient, heuristic ansatz that consists of layers of single-qubit
/// operations spanning SU(2) and $CX$ entanglements. It is widely used in VQE and
/// Quantum Machine Learning models due to its expressibility and relatively low
/// circuit depth.
///
/// The ansatz consists of alternating layers of $[R_Y, R_Z]$ rotations and $CX$ entanglements.
/// Using both $R_Y$ and $R_Z$ allows the ansatz to span the full SU(2) space for each qubit.
///
/// # Mathematical Structure
///
/// For $n$ qubits and $r$ repetitions:
///
/// $$|\psi(\theta, \phi)\rangle = \left( \prod_{i=0}^{n-1} R_Z(\phi_{r,i}) R_Y(\theta_{r,i}) \right) \prod_{l=0}^{r-1} \left[ \left( \prod_{(i,j) \in E} CX_{i,j} \right) \left( \prod_{i=0}^{n-1} R_Z(\phi_{l,i}) R_Y(\theta_{l,i}) \right) \right] |0\rangle^{\otimes n}$$
///
/// where $E$ is the set of edges defined by the entanglement topology.
///
/// # Arguments
///
/// * `num_qubits` - The number of qubits in the ansatz. Must be at least 1.
/// * `reps` - The number of repetition layers. Each repetition adds a rotation
///   layer and an entanglement layer. The final rotation layer is always included.
/// * `entanglement` - The topology of the entanglement (e.g., Linear, Full).
///   See [`EntanglementTopology`] for options.
///
/// # Returns
///
/// A [`TwoLocal`] instance configured for Efficient SU2 with:
/// - Rotation gates: `[RY, RZ]`
/// - Entanglement gate: `CX`
/// - Total parameters: `(reps + 1) × num_qubits × 2`
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::ansatz::{efficient_su2, EntanglementTopology, Ansatz};
///
/// // 2 qubits, 1 rep, full entanglement
/// let ansatz = efficient_su2(2, 1, EntanglementTopology::Full);
///
/// assert_eq!(ansatz.num_qubits(), 2);
/// assert_eq!(ansatz.num_parameters(), 8); // (1+1) * 2 * 2 = 8
///
/// let circuit = ansatz.build_circuit("p").unwrap();
/// ```
pub fn efficient_su2(
    num_qubits: usize,
    reps: usize,
    entanglement: EntanglementTopology,
) -> TwoLocal {
    TwoLocal::new(num_qubits)
        .reps(reps)
        .rotation_gates(vec![StandardGate::RY, StandardGate::RZ])
        .entanglement_gate(StandardGate::CX)
        .entanglement(entanglement)
}

#[cfg(test)]
#[path = "facades_test.rs"]
mod facades_test;
