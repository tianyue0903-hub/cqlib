use crate::circuit::PyCircuit;
use cqlib_core::ir::{qasm2_dump, qasm2_dumps};
use cqlib_core::ir::{qasm2_load, qasm2_loads};
use pyo3::prelude::*;

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
