use crate::circuit::PyCircuit;
use cqlib_core::ir::{qasm2_dump, qasm2_dumps};
use cqlib_core::ir::{qasm2_load, qasm2_loads};
use cqlib_core::ir::{qcis_dump, qcis_dumps};
use cqlib_core::ir::{qcis_load, qcis_loads};
use pyo3::prelude::*;

// QASM2 functions
#[pyfunction(name = "qasm2_loads")]
pub fn py_qasm2_loads(qasm: &str) -> PyCircuit {
    PyCircuit {
        inner: qasm2_loads(qasm).unwrap(),
    }
}

#[pyfunction(name = "qasm2_load")]
pub fn py_qasm2_load(path: &str) -> PyCircuit {
    PyCircuit {
        inner: qasm2_load(path).unwrap(),
    }
}

#[pyfunction(name = "qasm2_dumps")]
pub fn py_qasm2_dumps(circuit: &PyCircuit) -> PyResult<String> {
    Ok(qasm2_dumps(&circuit.inner).unwrap())
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
