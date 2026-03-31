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

pub mod ansatz;
pub mod bit;
pub mod circuit_impl;
pub mod circuit_to_matrix;
pub mod gate;
pub mod instruction;
pub mod operation;
pub mod parameter;

pub use bit::PyQubit;
pub use circuit_impl::PyCircuit;
pub use gate::{
    PyCircuitGate, PyConditionView, PyControlFlow, PyDirective, PyIfElseGate, PyMcGate,
    PyStandardGate, PyUnitaryGate, PyWhileLoopGate,
};
pub use instruction::PyInstruction;
pub use operation::PyOperation;
pub use parameter::PyParameter;

use pyo3::prelude::*;

/// Registers all circuit classes, gate classes, helper functions, and the
/// ansatz submodule as `_native.circuit`.
pub(crate) fn register_circuit_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "circuit")?;

    // Core circuit types
    m.add_class::<PyQubit>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyParameter>()?;
    m.add_class::<PyOperation>()?;
    m.add_class::<PyInstruction>()?;
    // Gate classes and static gate instances
    gate::register_gate_classes(&m)?;
    // circuit_to_matrix helper function
    m.add_function(wrap_pyfunction!(
        circuit_to_matrix::py_circuit_to_matrix,
        &m
    )?)?;
    // Ansatz submodule
    ansatz::register_ansatz_module(&m)?;

    // Attach as `_native.circuit` and make it importable by Python
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.circuit", &m)?;

    Ok(())
}
