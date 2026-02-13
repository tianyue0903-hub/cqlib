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

//! Circuit-Based Gate Definitions
//!
//! This module provides [`CircuitGate`] and [`FrozenCircuit`], allowing
//! quantum circuits to be used as reusable gate components within other circuits.
//! This enables hierarchical circuit construction and custom composite gates.

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use indexmap::IndexSet;
use std::sync::Arc;

/// An immutable, frozen circuit for use in gate definitions.
///
/// `FrozenCircuit` wraps a [`Circuit`] in an immutable container, ensuring
/// that the circuit definition cannot be modified after creation. This is
/// essential for maintaining consistency when circuits are used as gate
/// definitions.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::Circuit;
/// use cqlib_core::circuit::gate::FrozenCircuit;
///
/// // Create a circuit and freeze it
/// let circuit = Circuit::new(2);
/// let frozen = FrozenCircuit::new(circuit);
///
/// assert_eq!(frozen.circuit().qubits().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FrozenCircuit {
    pub(crate) circuit: Circuit,
}

impl FrozenCircuit {
    /// Creates a new frozen circuit from a [`Circuit`].
    ///
    /// The circuit is moved into the frozen container and cannot be
    /// modified afterwards.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The circuit to freeze.
    ///
    /// # Returns
    ///
    /// A new `FrozenCircuit` wrapping the provided circuit.
    pub fn new(circuit: Circuit) -> Self {
        Self { circuit }
    }

    /// Returns a reference to the inner circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::FrozenCircuit;
    ///
    /// let circuit = Circuit::new(3);
    /// let frozen = FrozenCircuit::new(circuit);
    ///
    /// assert_eq!(frozen.circuit().qubits().len(), 3);
    /// ```
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
}

/// A composite gate defined by a quantum circuit.
///
/// `CircuitGate` encapsulates a frozen circuit as a reusable gate operation.
/// When used in a circuit, its parameters are mapped positionally to the
/// symbolic parameters of the inner circuit.
///
/// # Parameter Resolution Logic
///
/// It is important to distinguish between the symbols defined inside the `CircuitGate`
/// and the arguments passed to it from the outside.
///
/// **Example Flow:**
///
/// 1. **Internal Expression**: The inner circuit contains a gate with a parameter expression, e.g., `theta + 1`.
/// 2. **Binding**: The `CircuitGate` usage defines a mapping from an external argument to the internal symbol, e.g., `theta` $\leftarrow$ `2 * x`.
/// 3. **Evaluation**:
///    If the external argument `x` is `0.5`:
///    - First, the binding is resolved: `theta` becomes `2 * 0.5 = 1.0`.
///    - Then, the internal circuit evaluates its expression using this value: `1.0 + 1 = 2.0`.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::Circuit;
/// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
///
/// // Create a simple circuit and wrap as a CircuitGate
/// let circuit = Circuit::new(2);
/// let frozen = FrozenCircuit::new(circuit);
/// let gate = CircuitGate::new("Empty2Qubit", frozen).unwrap();
///
/// assert_eq!(gate.num_qubits(), 2);
/// assert_eq!(gate.name(), "Empty2Qubit");
/// ```
#[derive(Debug, Clone)]
pub struct CircuitGate {
    /// The name/label of this composite gate.
    pub name: Arc<String>,
    pub(crate) num_qubits: usize,
    pub(crate) num_params: usize,
    pub(crate) circuit: Arc<FrozenCircuit>,
}

impl CircuitGate {
    /// Creates a new circuit-based gate.
    ///
    /// Extracts the qubit count and parameter count from the frozen circuit.
    ///
    /// # Arguments
    ///
    /// * `name` - A descriptive name for the gate.
    /// * `circuit` - The frozen circuit defining the gate operation.
    ///
    /// # Returns
    ///
    /// - `Ok(CircuitGate)`: The new gate if successful.
    /// - `Err(CircuitError)`: If the circuit cannot be used as a gate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(2);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("Bell", frozen).unwrap();
    ///
    /// assert_eq!(gate.num_qubits(), 2);
    /// ```
    pub fn new(name: impl Into<String>, circuit: FrozenCircuit) -> Result<Self, CircuitError> {
        let num_qubits = circuit.circuit.qubits().len();
        let num_params = circuit.circuit.symbols().len();

        Ok(Self {
            name: Arc::new(name.into()),
            num_qubits,
            num_params,
            circuit: Arc::new(circuit),
        })
    }

    /// Returns the set of symbolic parameter names used in the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(1);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("NoParams", frozen).unwrap();
    ///
    /// let symbols = gate.symbols();
    /// assert!(symbols.is_empty());
    /// ```
    pub fn symbols(&self) -> IndexSet<String> {
        self.circuit.circuit.symbols().clone()
    }

    /// Returns the number of qubits this gate acts on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(3);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("ThreeQubitOp", frozen).unwrap();
    /// assert_eq!(gate.num_qubits(), 3);
    /// ```
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the number of parameters this gate accepts.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(1);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("NoParams", frozen).unwrap();
    ///
    /// assert_eq!(gate.num_params(), 0);
    /// ```
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Returns a clone of the internal frozen circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(2);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("Test", frozen).unwrap();
    ///
    /// let inner = gate.circuit();
    /// assert_eq!(inner.circuit().qubits().len(), 2);
    /// ```
    pub fn circuit(&self) -> Arc<FrozenCircuit> {
        self.circuit.clone()
    }

    /// Returns the name of this circuit gate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(2);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("MyCustomGate", frozen).unwrap();
    /// assert_eq!(gate.name(), "MyCustomGate");
    /// ```
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Computes the inverse of this circuit gate.
    ///
    /// Creates a new `CircuitGate` with the circuit inverted and appends "_dg"
    /// to the name.
    ///
    /// # Returns
    ///
    /// - `Ok(CircuitGate)`: The inverted gate.
    /// - `Err(CircuitError)`: If the circuit cannot be inverted.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::circuit::gate::{CircuitGate, FrozenCircuit};
    ///
    /// let circuit = Circuit::new(1);
    /// let frozen = FrozenCircuit::new(circuit);
    /// let gate = CircuitGate::new("Empty", frozen).unwrap();
    ///
    /// // Note: empty circuits are trivially invertible
    /// let inverse = gate.inverse().unwrap();
    /// assert_eq!(inverse.name(), "Empty_dg");
    /// ```
    pub fn inverse(&self) -> Result<Self, CircuitError> {
        let inverted_circuit = self.circuit.circuit.inverse()?;
        let frozen_inverted = FrozenCircuit {
            circuit: inverted_circuit,
        };
        CircuitGate::new(format!("{}_dg", self.name), frozen_inverted)
    }
}
