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

//! Python bindings for cqlib-core device module.

use pyo3::prelude::*;

pub mod common;
pub mod device_impl;
pub mod noise;
pub mod result;

/// Register the device submodule.
pub fn register_device_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let device_module = PyModule::new(parent.py(), "device")?;

    // Re-export compiler Topology type so users can construct Device directly from cqlib.device.
    if let Ok(topology_type) = parent.getattr("Topology") {
        device_module.add("Topology", topology_type)?;
    }

    device_module.add_class::<device_impl::PyInstructionProp>()?;
    device_module.add_class::<device_impl::PyQubitProp>()?;
    device_module.add_class::<device_impl::PyEdgeProp>()?;
    device_module.add_class::<device_impl::PyDevice>()?;
    device_module.add_class::<device_impl::PyLayout>()?;
    device_module.add_class::<noise::PySingleQubitNoise>()?;
    device_module.add_class::<noise::PyTwoQubitNoise>()?;
    device_module.add_class::<noise::PyReadoutError>()?;
    device_module.add_class::<noise::PyOperationKey>()?;
    device_module.add_class::<noise::PyNoiseModel>()?;
    device_module.add_class::<result::PyOutcome>()?;
    device_module.add_class::<result::PyStatus>()?;
    device_module.add_class::<result::PyExecutionResult>()?;

    parent.add_submodule(&device_module)?;
    Ok(())
}
