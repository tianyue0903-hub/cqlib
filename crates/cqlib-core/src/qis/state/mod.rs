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

//! Quantum state representations for simulation.
//!
//! This module provides different representations of quantum states:
//!
//! - [`Statevector`]: Represents pure quantum states as a vector of complex amplitudes.
//!   Efficient for simulating ideal quantum circuits without noise.
//!
//! - [`DensityMatrix`]: Represents mixed quantum states as a density matrix.
//!   Capable of simulating both pure and mixed states, including quantum channels
//!   via Kraus operators.
//!
//! - [`DensityMatrixNoise`]: Extends the density matrix simulator with a configurable
//!   noise model for realistic quantum simulations including gate errors and readout noise.
//!
//! # Choosing a Simulator
//!
//! | Simulator | Use Case | Performance | Noise Support |
//! |-----------|----------|-------------|---------------|
//! | [`Statevector`] | Ideal circuits, large qubit counts | Fastest | No |
//! | [`DensityMatrix`] | Mixed states, quantum channels | Moderate | Kraus operators only |
//! | [`DensityMatrixNoise`] | Realistic device simulation | Slower | Full noise model |
//!
//! # Examples
//!
//! ```rust
//! use cqlib_core::qis::{Statevector, DensityMatrix};
//!
//! // Pure state simulation
//! let mut sv = Statevector::new(3);
//! sv.apply_h(0);
//! sv.apply_cx(0, 1);
//! let probs_sv = sv.probabilities();
//!
//! // Mixed state simulation
//! let mut dm = DensityMatrix::new(3);
//! dm.apply_h(0);
//! dm.apply_cx(0, 1);
//! let probs_dm = dm.probabilities();
//!
//! // Results should be identical for pure states
//! assert!((probs_sv[0] - probs_dm[0]).abs() < 1e-10);
//! ```

pub mod density_matrix;
pub mod density_matrix_noise;
pub mod statevector;

pub use density_matrix::DensityMatrix;
pub use density_matrix_noise::DensityMatrixNoise;
pub use statevector::Statevector;

#[cfg(test)]
#[path = "state_test.rs"]
mod state_test;
