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

use crate::circuit::PyCircuit;
use cqlib_core::ir::qcis_loads;
use cqlib_core::ir::{qasm2_dump, qasm2_dumps};
use cqlib_core::ir::{qasm2_load, qasm2_loads};
use cqlib_core::ir::{qcis_dump, qcis_dumps};
use pyo3::prelude::*;

// QASM2 functions
#[pyfunction(name = "qasm2_loads")]
pub fn py_qasm2_loads(qasm: &str) -> PyResult<PyCircuit> {
    match qasm2_loads(qasm) {
        Ok(circuit) => Ok(PyCircuit { inner: circuit }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "QASM parse error: {}",
            e
        ))),
    }
}

#[pyfunction(name = "qasm2_load")]
pub fn py_qasm2_load(path: &str) -> PyResult<PyCircuit> {
    match qasm2_load(path) {
        Ok(circuit) => Ok(PyCircuit { inner: circuit }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "QASM load error: {}",
            e
        ))),
    }
}

#[pyfunction(name = "qasm2_dumps")]
pub fn py_qasm2_dumps(circuit: &PyCircuit) -> PyResult<String> {
    match qasm2_dumps(&circuit.inner) {
        Ok(qasm) => Ok(qasm),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "QASM dump error: {}",
            e
        ))),
    }
}

#[pyfunction(name = "qasm2_dump")]
pub fn py_qasm2_dump(circuit: &PyCircuit, path: &str) -> PyResult<()> {
    Ok(qasm2_dump(&circuit.inner, path)?)
}

// QCIS functions
#[pyfunction(name = "qcis_loads")]
pub fn py_qcis_loads(qcis: &str) -> PyResult<PyCircuit> {
    match qcis_loads(qcis) {
        Ok(c) => Ok(PyCircuit { inner: c }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "QCIS parse error: {}",
            e
        ))),
    }
}

#[pyfunction(name = "qcis_load")]
pub fn py_qcis_load(path: &str) -> PyResult<PyCircuit> {
    let content = std::fs::read_to_string(path)?;
    match qcis_loads(&content) {
        Ok(c) => Ok(PyCircuit { inner: c }),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "QCIS parse error: {}",
            e
        ))),
    }
}

#[pyfunction(name = "qcis_dumps")]
pub fn py_qcis_dumps(circuit: &PyCircuit) -> PyResult<String> {
    qcis_dumps(&circuit.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("QCIS dump error: {}", e))
    })
}

#[pyfunction(name = "qcis_dump")]
pub fn py_qcis_dump(circuit: &PyCircuit, path: &str) -> PyResult<()> {
    let path_buf = std::path::PathBuf::from(path);
    qcis_dump(&circuit.inner, &path_buf).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("QCIS dump error: {}", e))
    })
}
