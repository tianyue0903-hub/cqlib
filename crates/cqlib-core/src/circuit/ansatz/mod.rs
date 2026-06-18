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

//! Quantum Ansatz library for variational quantum algorithms.
//!
//! This module provides parameterized quantum circuit templates (ansatze) commonly
//! used in variational quantum algorithms such as VQE (Variational Quantum Eigensolver),
//! QAOA (Quantum Approximate Optimization Algorithm), and Quantum Machine Learning.
//!
//! # Core Concepts
//!
//! An **ansatz** is a parameterized quantum circuit that serves as a trial wavefunction
//! or feature map. The parameters are optimized iteratively by a classical optimizer
//! to minimize a cost function (e.g., the expectation value of a Hamiltonian).
//!
//! # Module Structure
//!
//! - [`traits`]: Defines the core [`Ansatz`] trait that all ansatze implement.
//! - [`two_local`]: Hardware-efficient ansatze with alternating rotation and entanglement layers.
//! - [`facades`]: Convenient constructors for common ansatz patterns (RealAmplitudes, EfficientSU2).
//! - [`feature_map`]: Data encoding circuits, including [`BasisEncoding`], [`AngleEncoding`],
//!   [`ZFeatureMap`], [`IQPFeatureMap`], [`ZZFeatureMap`], and [`PauliFeatureMap`].
//! - [`layers`]: Layer-style templates, including [`BasicEntanglerLayers`] and
//!   [`StronglyEntanglingLayers`].
//! - [`hamiltonian_evolution`]: Hamiltonian time-evolution ansatz utilities.
//! - [`qaoa`]: The Quantum Approximate Optimization Algorithm ansatz.
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal, EntanglementTopology};
//!
//! // Create a TwoLocal ansatz with 3 qubits, linear entanglement
//! let ansatz = TwoLocal::new(3)
//!     .reps(2)
//!     .entanglement(EntanglementTopology::Linear);
//!
//! // Build the parameterized circuit
//! let circuit = ansatz.build_circuit("theta").unwrap();
//!
//! // The ansatz has a fixed number of parameters
//! assert_eq!(ansatz.num_parameters(), 9); // 3 qubits * 3 layers
//! ```

pub mod facades;
pub mod feature_map;
pub mod hamiltonian_evolution;
pub mod layers;
pub mod qaoa;
pub mod traits;
pub mod two_local;

pub use facades::{efficient_su2, pauli_feature_map, real_amplitudes, zz_feature_map};
pub use feature_map::{
    AngleEncoding, BasisEncoding, IQPFeatureMap, PauliFeatureMap, ZFeatureMap, ZZFeatureMap,
};
pub use hamiltonian_evolution::{EvolutionInfo, EvolutionStrategy, PauliEvolutionAnsatz};
pub use layers::{BasicEntanglerLayers, StronglyEntanglingLayers};
pub use qaoa::QAOAAnsatz;
pub use traits::Ansatz;
pub use two_local::{EntanglementTopology, TwoLocal};
