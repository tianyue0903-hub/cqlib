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

//! Python Bindings for Circuit-Based Gates
//!
//! This module provides Python bindings for [`CircuitGate`] from cqlib-core.
//! It allows quantum circuits to be used as reusable gate components.
//!
//! # Key Components
//!
//! - [`PyCircuitGate`]: A composite gate defined by a quantum circuit.

use crate::circuit::PyCircuit;
use cqlib_core::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for `CircuitGate`.
///
/// Represents a composite gate defined by a quantum circuit.
/// Allows hierarchical circuit construction and custom composite gates.
#[pyclass(name = "CircuitGate", module = "cqlib.circuit.gate")]
#[derive(Clone, Debug)]
pub struct PyCircuitGate {
    pub inner: CircuitGate,
}

#[pymethods]
impl PyCircuitGate {
    /// Creates a new circuit-based gate.
    ///
    /// # Arguments
    ///
    /// * `name` - A descriptive name for the gate.
    /// * `circuit` - The circuit to use as the gate definition.
    ///
    /// # Returns
    ///
    /// A new `CircuitGate`.
    ///
    /// # Examples
    ///
    /// ```python
    /// from cqlib import Circuit, CircuitGate
    ///
    /// circuit = Circuit(2)
    /// gate = CircuitGate("Bell", circuit)
    /// ```
    #[new]
    fn new(name: String, circuit: PyCircuit) -> PyResult<Self> {
        let frozen = FrozenCircuit::new(circuit.inner);
        CircuitGate::new(name, frozen)
            .map(|gate| Self { inner: gate })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns the name of this circuit gate.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Returns the number of qubits this gate acts on.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of parameters this gate accepts.
    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Returns the set of symbolic parameter names used in the circuit.
    ///
    /// # Returns
    ///
    /// A list of parameter names.
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols().into_iter().collect()
    }

    /// Computes the inverse of this circuit gate.
    ///
    /// Creates a new `CircuitGate` with the circuit inverted and appends "_dg" to the name.
    ///
    /// # Returns
    ///
    /// A new gate representing the inverse operation.
    fn inverse(&self) -> PyResult<Self> {
        self.inner
            .inverse()
            .map(|gate| Self { inner: gate })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl From<CircuitGate> for PyCircuitGate {
    fn from(gate: CircuitGate) -> Self {
        Self { inner: gate }
    }
}

impl From<&CircuitGate> for PyCircuitGate {
    fn from(gate: &CircuitGate) -> Self {
        Self {
            inner: gate.clone(),
        }
    }
}

impl From<PyCircuitGate> for CircuitGate {
    fn from(gate: PyCircuitGate) -> Self {
        gate.inner
    }
}

impl From<&PyCircuitGate> for CircuitGate {
    fn from(gate: &PyCircuitGate) -> Self {
        gate.inner.clone()
    }
}
