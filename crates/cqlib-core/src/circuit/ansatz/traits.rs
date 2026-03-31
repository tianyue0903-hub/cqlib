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

//! Core trait definitions for variational quantum ansatze.
//!
//! This module defines the [`Ansatz`] trait, which is the fundamental abstraction
//! for all parameterized quantum circuits in variational quantum algorithms.
//!
//! # The Ansatz Pattern
//!
//! In variational quantum computing, an ansatz provides:
//!
//! 1. **Parameterization**: A way to generate quantum circuits with tunable parameters.
//! 2. **Structure**: A specific circuit architecture (e.g., alternating layers, specific gates).
//! 3. **Metrics**: Information about the number of parameters and qubits required.
//!
//! Implementors of this trait can be used interchangeably in variational algorithms.

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;

/// Core trait for all parameterized variational quantum ansatze.
///
/// This trait defines the interface that any variational quantum circuit template
/// must implement to be usable within the variational quantum algorithm framework.
///
/// # Type Parameters
///
/// Implementations are typically concrete types (structs) that hold configuration
/// for the ansatz (number of qubits, layers, entanglement patterns, etc.).
///
/// # Example
///
/// ```
/// use cqlib_core::circuit::ansatz::Ansatz;
/// use cqlib_core::circuit::ansatz::TwoLocal;
///
/// fn use_ansatz<A: Ansatz>(ansatz: &A) {
///     println!("Using {} qubits and {} parameters",
///         ansatz.num_qubits(),
///         ansatz.num_parameters()
///     );
/// }
/// ```
pub trait Ansatz {
    /// Validates the ansatz configuration without building the circuit.
    ///
    /// This method checks whether the ansatz configuration is valid, allowing
    /// early detection of errors before expensive circuit construction. The
    /// default implementation performs no validation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the configuration is valid.
    /// * `Err(CircuitError)` if the configuration is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal};
    ///
    /// let ansatz = TwoLocal::new(2);
    /// assert!(ansatz.validate().is_ok());
    /// ```
    fn validate(&self) -> Result<(), CircuitError> {
        Ok(())
    }

    /// Builds the ansatz circuit and returns it.
    ///
    /// This method constructs the actual quantum circuit based on the ansatz
    /// configuration. All rotation angles in the circuit are represented as
    /// symbolic parameters that can be bound to concrete values later.
    ///
    /// Implementations should call [`validate`](Self::validate) at the start
    /// of this method to ensure configuration validity.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix string used to generate parameter names (e.g., "theta").
    ///   This allows multiple ansatze in the same circuit to have distinct parameter names.
    ///
    /// # Returns
    ///
    /// * `Result<Circuit, CircuitError>` - The constructed parameterized circuit,
    ///   or an error if the circuit cannot be built (e.g., invalid configuration).
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal};
    ///
    /// let ansatz = TwoLocal::new(2);
    /// let circuit = ansatz.build_circuit("theta").unwrap();
    ///
    /// // The circuit contains symbolic parameters like "theta_0", "theta_1", etc.
    /// ```
    fn build_circuit(&self, prefix: &str) -> Result<Circuit, CircuitError>;

    /// Returns the number of independent parameters required by this ansatz.
    ///
    /// This is the number of classical optimization variables that need to be
    /// tuned during the variational optimization process.
    ///
    /// # Returns
    ///
    /// The total count of independent scalar parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal};
    ///
    /// let ansatz = TwoLocal::new(3).reps(2);
    /// // 3 qubits * 3 layers (2 reps + 1 final) = 9 parameters
    /// assert_eq!(ansatz.num_parameters(), 9);
    /// ```
    fn num_parameters(&self) -> usize;

    /// Returns the number of qubits this ansatz acts upon.
    ///
    /// # Returns
    ///
    /// The number of quantum bits required to execute the ansatz circuit.
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::ansatz::{Ansatz, TwoLocal};
    ///
    /// let ansatz = TwoLocal::new(4);
    /// assert_eq!(ansatz.num_qubits(), 4);
    /// ```
    fn num_qubits(&self) -> usize;
}
