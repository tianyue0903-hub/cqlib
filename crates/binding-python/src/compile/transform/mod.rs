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

//! Python bindings for reusable compiler transforms.

mod canonicalize;
pub mod decompose;
pub mod layout;
pub mod result;

use pyo3::prelude::*;

use canonicalize::{
    PyCanonicalizeConfig, PyCanonicalizeResult, PyCanonicalizer, py_canonicalize_circuit,
};
pub(crate) use result::PyTransformResult;

/// Registers transform bindings as `_native.compile.transform`.
pub(crate) fn register_transform_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "transform")?;

    m.add_class::<PyCanonicalizeConfig>()?;
    m.add_class::<PyCanonicalizer>()?;
    m.add_class::<PyCanonicalizeResult>()?;
    m.add_class::<PyTransformResult>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_canonicalize_circuit, &m)?)?;

    decompose::register_decompose_module(&m)?;
    layout::register_layout_module(&m)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.transform", &m)?;

    Ok(())
}
