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

//! Python bindings for compiler commutation proofs.

pub mod checker;

pub use checker::{
    PyCommutation, PyCommutationChecker, PyCommutationConfig, py_algebraic_commutation,
    py_check_commutation,
};

use pyo3::prelude::*;

/// Registers commutation bindings as `_native.compile.commutation`.
pub(crate) fn register_commutation_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "commutation")?;

    m.add_class::<PyCommutation>()?;
    m.add_class::<PyCommutationConfig>()?;
    m.add_class::<PyCommutationChecker>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_check_commutation, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_algebraic_commutation, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.commutation", &m)?;

    Ok(())
}
