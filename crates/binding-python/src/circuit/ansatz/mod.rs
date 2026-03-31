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

//! Python bindings for the `cqlib_core::circuit::ansatz` module.
//!
//! Provides parameterized quantum circuit templates for variational algorithms
//! and quantum machine learning, accessible from Python as `cqlib.circuit.ansatz`.

pub mod facades;
pub mod feature_map;
pub mod qaoa;
pub mod two_local;

pub use feature_map::{PyAngleEncoding, PyPauliFeatureMap, PyZZFeatureMap};
pub use qaoa::PyQAOAAnsatz;
pub use two_local::{PyEntanglementTopology, PyTwoLocal};

use pyo3::prelude::*;

/// Registers the `ansatz` submodule on the given parent module.
///
/// Adds all ansatz classes and facade functions as `parent.ansatz`.
pub(crate) fn register_ansatz_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "ansatz")?;

    // Classes
    m.add_class::<PyEntanglementTopology>()?;
    m.add_class::<PyTwoLocal>()?;
    m.add_class::<PyAngleEncoding>()?;
    m.add_class::<PyZZFeatureMap>()?;
    m.add_class::<PyPauliFeatureMap>()?;
    m.add_class::<PyQAOAAnsatz>()?;

    // Facade functions
    m.add_function(wrap_pyfunction!(facades::real_amplitudes, &m)?)?;
    m.add_function(wrap_pyfunction!(facades::efficient_su2, &m)?)?;
    m.add_function(wrap_pyfunction!(facades::zz_feature_map, &m)?)?;
    m.add_function(wrap_pyfunction!(facades::pauli_feature_map, &m)?)?;

    parent.add_submodule(&m)?;

    // Make `from cqlib._native.circuit.ansatz import X` work
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.circuit.ansatz", &m)?;

    Ok(())
}
