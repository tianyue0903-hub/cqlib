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

//! Python bindings for compiler ancillary-resource management.

pub mod error;
pub mod manager;
pub mod model;
pub mod policy;

pub use manager::PyResourceManager;
pub use model::{PyAncillaRequirement, PyResourceLease, PyResourcePlan, PyResourceRequest};
pub use policy::{PyResourceLimits, PyResourcePolicy};

use pyo3::prelude::*;

/// Registers resource bindings as `_native.compile.resource`.
pub(crate) fn register_resource_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "resource")?;

    error::register_errors(&m)?;
    m.add_class::<PyAncillaRequirement>()?;
    m.add_class::<PyResourcePolicy>()?;
    m.add_class::<PyResourceLimits>()?;
    m.add_class::<PyResourceRequest>()?;
    m.add_class::<PyResourcePlan>()?;
    m.add_class::<PyResourceLease>()?;
    m.add_class::<PyResourceManager>()?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.resource", &m)?;

    Ok(())
}
