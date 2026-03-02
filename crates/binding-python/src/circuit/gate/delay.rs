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

//! Quantum Delay Operation
//!
//! This module provides Python bindings for delay, representing a time delay
//! in quantum circuit execution.

use cqlib_core::circuit::gate::delay::DelayOp;
use pyo3::prelude::*;

#[pyclass(name = "Delay", module = "cqlib.circuit.gate")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyDelay {
    pub(crate) inner: DelayOp,
}

impl From<DelayOp> for PyDelay {
    fn from(inner: DelayOp) -> Self {
        Self { inner }
    }
}

impl From<PyDelay> for DelayOp {
    fn from(py: PyDelay) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyDelay {
    /// Creates a new delay operation.
    ///
    /// The delay unit is 0.5 nanoseconds (aligned with common quantum
    /// control hardware timing resolutions).
    #[new]
    fn new() -> Self {
        PyDelay { inner: DelayOp {} }
    }

    fn __repr__(&self) -> String {
        "delay".to_string()
    }

    fn __str__(&self) -> String {
        "delay".to_string()
    }
}

impl Default for PyDelay {
    fn default() -> Self {
        Self::new()
    }
}
