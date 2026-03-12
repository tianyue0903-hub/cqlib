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

//! Quantum Information Science (QIS) module for quantum simulation.
//!
//! This module provides core quantum computing primitives including:
//! - State representations: [`Statevector`] for pure states and [`DensityMatrix`] for mixed states
//! - Noise modeling: [`DensityMatrixNoise`] for realistic quantum simulations
//! - Observables: [`Hamiltonian`] and [`PauliString`] for expectation value calculations
//! - Pauli operators: [`Pauli`] and [`Phase`] for quantum error correction and stabilizer formalism
//!
//! # Module Structure
//!
//! - [`state`]: Quantum state representations (statevector and density matrix)
//! - [`pauli`]: Pauli operators and Pauli strings with symplectic encoding
//! - [`hamiltonian`]: Hamiltonian construction from Pauli strings
//! - [`observable`]: Trait for computing expectation values
//! - [`error`]: Error types for QIS operations
//!
//! # Examples
//!
//! Creating and manipulating quantum states:
//!
//! ```rust
//! use cqlib_core::qis::{Statevector, DensityMatrix};
//!
//! // Create a Bell state using statevector
//! let mut sv = Statevector::new(2);
//! sv.apply_h(0);
//! sv.apply_cx(0, 1);
//!
//! // Create the same state using density matrix
//! let mut dm = DensityMatrix::new(2);
//! dm.apply_h(0);
//! dm.apply_cx(0, 1);
//! ```
//!
//! Working with observables:
//!
//! ```rust
//! use cqlib_core::qis::{Hamiltonian, PauliString, Observable};
//!
//! // Create a Hamiltonian H = 0.5 * ZZ + 0.3 * XX
//! let mut h = Hamiltonian::new(2);
//! h.add_term("ZZ".into(), 0.5.into()).unwrap();
//! h.add_term("XX".into(), 0.3.into()).unwrap();
//! ```

pub mod error;
pub mod evolution;
pub mod hamiltonian;
pub mod metrics;
pub mod observable;
pub mod pauli;
pub mod state;

pub use error::{PauliStringParseError, QisError};
pub use evolution::{PauliEvolution, TrotterMode};
pub use hamiltonian::Hamiltonian;
pub use observable::Observable;
pub use pauli::{Pauli, PauliString, Phase};
pub use state::density_matrix::DensityMatrix;
pub use state::density_matrix_noise::DensityMatrixNoise;
pub use state::statevector::Statevector;
