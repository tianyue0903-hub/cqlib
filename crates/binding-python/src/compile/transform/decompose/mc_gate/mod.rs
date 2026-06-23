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

mod composite;
mod families;
mod mc_su2;
mod mcx;

use crate::circuit::PyValueOperation;
pub(super) use crate::compile::error::compiler_error_to_py_err as compiler_error;
use cqlib_core::circuit::ValueOperation;
use pyo3::prelude::*;

pub(super) fn into_py_operations(operations: Vec<ValueOperation>) -> Vec<PyValueOperation> {
    operations.into_iter().map(Into::into).collect()
}

/// Registers exact multi-controlled synthesis primitives.
pub(crate) fn register_mc_gate_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "mc_gate")?;

    composite::register(&m)?;
    families::register(&m)?;
    mc_su2::register(&m)?;
    mcx::register(&m)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.transform.decompose.mc_gate", &m)?;

    Ok(())
}
