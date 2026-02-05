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

use cqlib_core::circuit::gate::circuit_gate::CircuitGate;
use pyo3::prelude::*;

#[pyclass(name = "CircuitGate", module = "cqlib.circuit.gates")]
#[derive(Clone, Debug)]
pub struct PyCircuitGate {
    pub inner: CircuitGate,
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
