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

use pyo3::prelude::*;

pub mod circuit_gate;
pub mod control_flow;
pub mod directive;
pub mod mc_gate;
pub mod standard;
pub mod unitary;

pub use circuit_gate::PyCircuitGate;
pub use control_flow::{PyConditionView, PyControlFlow, PyIfElseGate, PyWhileLoopGate};
pub use directive::PyDirective;
pub use mc_gate::PyMcGate;
pub use standard::PyStandardGate;
pub use unitary::PyUnitaryGate;

/// Registers all gate classes and static gate instances on `m`.
pub(crate) fn register_gate_classes(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gate")?;

    m.add_class::<PyStandardGate>()?;
    m.add_class::<PyUnitaryGate>()?;
    m.add_class::<PyMcGate>()?;
    m.add_class::<PyCircuitGate>()?;
    m.add_class::<PyControlFlow>()?;
    m.add_class::<PyIfElseGate>()?;
    m.add_class::<PyWhileLoopGate>()?;
    m.add_class::<PyConditionView>()?;
    m.add_class::<PyDirective>()?;
    // Register static gate instances (H, X, CX, etc.) onto StandardGate
    standard::register_gates(&m)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.circuit.gate", &m)?;
    Ok(())
}
