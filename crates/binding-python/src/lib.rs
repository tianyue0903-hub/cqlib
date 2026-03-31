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
pub mod device;
pub mod ir;
pub mod qis;
pub mod visualization;

use compile::{
    PyCliffordRzOptimization, PyGaConfig, PySabreConfig, PyTemplateMatching, PyTemplateOptimization,
};
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_native")]
fn binding_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all circuit classes, gates, helpers, and ansatz submodule
    circuit::register_circuit_module(m)?;
    // Register IR module with qasm2 and qcis submodules
    ir::register_ir_module(m)?;
    device::register_device_module(m)?;
    qis::register_qis_module(m)?;

    // Compile utilities
    m.add_class::<PySabreConfig>()?;
    m.add_class::<PyTemplateMatching>()?;
    m.add_class::<PyTemplateOptimization>()?;
    m.add_class::<PyCliffordRzOptimization>()?;
    m.add_class::<PyGaConfig>()?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_is_subgraph_isomorphic, m)?)?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_find_initial_layout, m)?)?;
    m.add_function(wrap_pyfunction!(
        compile::py_vf2_find_initial_layout_candidates,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compile::py_vf2_map, m)?)?;
    m.add_function(wrap_pyfunction!(compile::py_map_with_vf2_sabre, m)?)?;
    m.add_function(wrap_pyfunction!(compile::py_map_with_ga, m)?)?;
    // Register visualization functions
    m.add_function(wrap_pyfunction!(visualization::py_draw_text, m)?)?;
    m.add_function(wrap_pyfunction!(visualization::py_draw_figure, m)?)?;

    Ok(())
}
