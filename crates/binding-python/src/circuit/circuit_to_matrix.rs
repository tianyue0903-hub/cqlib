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

//! Python Bindings for Circuit to Matrix Conversion
//!
//! This module provides a function to convert a quantum circuit to its unitary matrix representation.

use crate::circuit::PyCircuit;
use crate::circuit::error::CircuitError as PyCircuitError;
use cqlib_core::circuit::circuit_to_matrix;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Converts a quantum circuit to its unitary matrix representation.
///
/// This function computes the unitary matrix corresponding to the entire quantum circuit.
/// The matrix size is 2^n x 2^n where n is the number of qubits.
///
/// # Arguments
///
/// * `circuit` - The quantum circuit to convert.
/// * `qubits_order` - Optional custom qubit ordering for the output matrix.
///
/// # Returns
///
/// A 2D numpy array (dtype=complex128) representing the unitary matrix.
///
/// # Raises
///
/// CircuitError if the circuit contains non-unitary operations (e.g., Measure, Reset).
///
/// # Examples
///
/// ```python
/// import numpy as np
/// from cqlib import Circuit, circuit_to_matrix
///
/// circuit = Circuit(2)
/// circuit.h(0)
/// circuit.cx(0, 1)
///
/// matrix = circuit_to_matrix(circuit)
/// print(matrix.shape)  # (4, 4)
/// ```
#[pyfunction(name = "circuit_to_matrix", signature = (circuit, qubits_order=None))]
pub fn py_circuit_to_matrix<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    qubits_order: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    // Clone circuit data for thread-safe access without holding GIL
    let circuit_inner = circuit.inner.clone();
    let order = qubits_order.clone();
    // Release GIL during potentially expensive matrix computation
    let result = py.detach(move || circuit_to_matrix(&circuit_inner, order.as_deref()));
    result
        .map(|arr| arr.into_pyarray(py))
        .map_err(|e| PyCircuitError::new_err(e.to_string()))
}
