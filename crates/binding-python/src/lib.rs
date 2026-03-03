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

pub mod circuit;
pub mod compile;
pub mod ir;

use pyo3::prelude::*;

use crate::circuit::gate::{
    PyCircuitGate, PyConditionView, PyIfElseGate, PyMcGate, PyStandardGate, PyUnitaryGate,
    PyWhileLoopGate,
};
use circuit::circuit_to_matrix;
use circuit::{PyCircuit, PyInstruction, PyOperation, PyParameter, PyQubit};
use compile::{PySabreConfig, PyTopology};

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_native")]
fn binding_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQubit>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyStandardGate>()?;
    m.add_class::<PyUnitaryGate>()?;
    m.add_class::<PyMcGate>()?;
    m.add_class::<PyCircuitGate>()?;
    m.add_class::<PyOperation>()?;
    m.add_class::<PyInstruction>()?;
    m.add_class::<PyTopology>()?;
    m.add_class::<PySabreConfig>()?;
    m.add_class::<PyIfElseGate>()?;
    m.add_class::<PyWhileLoopGate>()?;
    m.add_class::<PyConditionView>()?;

    m.add_function(wrap_pyfunction!(ir::py_qasm2_load, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qasm2_loads, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qasm2_dump, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qasm2_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qcis_load, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qcis_loads, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qcis_dump, m)?)?;
    m.add_function(wrap_pyfunction!(ir::py_qcis_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(
        circuit_to_matrix::py_circuit_to_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_is_subgraph_isomorphic, m)?)?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_find_initial_layout, m)?)?;
    m.add_function(wrap_pyfunction!(
        compile::py_vf2_find_initial_layout_candidates,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_map, m)?)?;
    m.add_function(wrap_pyfunction!(compile::py_map_with_vf2_sabre, m)?)?;

    // Register static gate instances (H, X, etc.) to StandardGate class
    circuit::gate::standard::register_gates(m)?;

    Ok(())
}
