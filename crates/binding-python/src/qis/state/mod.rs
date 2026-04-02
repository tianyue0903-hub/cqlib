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

//! Python bindings for cqlib-core quantum state simulation.

use pyo3::prelude::*;

pub mod density_matrix;
pub mod density_matrix_noise;
pub mod statevector;

/// Register the state submodule.
pub fn register_state_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let state_module = PyModule::new(parent.py(), "state")?;

    state_module.add_class::<statevector::PyStatevector>()?;
    state_module.add_class::<density_matrix::PyDensityMatrix>()?;
    state_module.add_class::<density_matrix_noise::PyDensityMatrixNoise>()?;

    parent.add_submodule(&state_module)?;
    Ok(())
}
