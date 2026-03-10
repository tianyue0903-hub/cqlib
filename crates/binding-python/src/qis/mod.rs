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

//! Python bindings for cqlib-core quantum information (qis) module.

use pyo3::prelude::*;

pub mod pauli;

/// Register the qis submodule.
pub fn register_qis_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let qis_module = PyModule::new(parent.py(), "qis")?;

    qis_module.add_class::<pauli::PyPhase>()?;
    qis_module.add_class::<pauli::PyPauli>()?;
    qis_module.add_class::<pauli::PyPauliString>()?;

    parent.add_submodule(&qis_module)?;
    Ok(())
}
