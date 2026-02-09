use crate::circuit::PyCircuit;
use cqlib_core::circuit::circuit_to_matrix;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction(name = "circuit_to_matrix", signature = (circuit, qubits_order=None))]
pub fn py_circuit_to_matrix<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    qubits_order: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    circuit_to_matrix(&circuit.inner, qubits_order.as_ref())
        .map(|arr| arr.into_pyarray(py))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
