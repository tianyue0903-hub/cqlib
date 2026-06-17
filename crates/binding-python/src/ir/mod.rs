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

//! Python bindings for Intermediate Representation (IR) module.
//!
//! This module provides Python bindings for parsing and serializing quantum circuit
//! formats including OpenQASM 2.0, OpenQASM 3.0, and QCIS.
//!
//! # Supported Formats
//!
//! | Format | Load (Parse) | Dump (Serialize) |
//! |--------|--------------|------------------|
//! | OpenQASM 2.0 | `qasm2.load`, `qasm2.loads` | `qasm2.dump`, `qasm2.dumps` |
//! | OpenQASM 3.0 | `qasm3.load`, `qasm3.loads` | `qasm3.dump`, `qasm3.dumps` |
//! | QCIS | `qcis.load`, `qcis.loads` | `qcis.dump`, `qcis.dumps` |
//!
//! # Usage Example
//!
//! ```python
//! from cqlib.ir import qasm2, qcis
//!
//! # Parse OpenQASM 2.0
//! qasm = '''OPENQASM 2.0;
//! include "qelib1.inc";
//! qreg q[2];
//! h q[0];
//! cx q[0], q[1];
//! '''
//! circuit = qasm2.loads(qasm)
//!
//! # Convert to QCIS format
//! qcis_str = qcis.dumps(circuit)
//! print(qcis_str)  # H Q0\nCZ Q0 Q1\n...
//! ```

use pyo3::prelude::*;

pub mod qasm2;
pub mod qasm3;
pub mod qcis;

fn register_sys_module(module: &Bound<'_, PyModule>, name: &str) -> PyResult<()> {
    module
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(name, module)
}

/// Register the ir submodule with qasm2, qasm3, and qcis submodules.
pub fn register_ir_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let ir_module = PyModule::new(parent.py(), "ir")?;

    // Register qasm2 submodule under ir
    let qasm2_module = PyModule::new(parent.py(), "qasm2")?;
    qasm2_module.add_function(wrap_pyfunction!(qasm2::py_qasm2_load, &qasm2_module)?)?;
    qasm2_module.add_function(wrap_pyfunction!(qasm2::py_qasm2_loads, &qasm2_module)?)?;
    qasm2_module.add_function(wrap_pyfunction!(qasm2::py_qasm2_dump, &qasm2_module)?)?;
    qasm2_module.add_function(wrap_pyfunction!(qasm2::py_qasm2_dumps, &qasm2_module)?)?;
    ir_module.add_submodule(&qasm2_module)?;
    register_sys_module(&qasm2_module, "cqlib._native.ir.qasm2")?;

    // Register qasm3 submodule under ir
    let qasm3_module = PyModule::new(parent.py(), "qasm3")?;
    qasm3_module.add_function(wrap_pyfunction!(qasm3::py_qasm3_load, &qasm3_module)?)?;
    qasm3_module.add_function(wrap_pyfunction!(qasm3::py_qasm3_loads, &qasm3_module)?)?;
    qasm3_module.add_function(wrap_pyfunction!(qasm3::py_qasm3_dump, &qasm3_module)?)?;
    qasm3_module.add_function(wrap_pyfunction!(qasm3::py_qasm3_dumps, &qasm3_module)?)?;
    ir_module.add_submodule(&qasm3_module)?;
    register_sys_module(&qasm3_module, "cqlib._native.ir.qasm3")?;

    // Register qcis submodule under ir
    let qcis_module = PyModule::new(parent.py(), "qcis")?;
    qcis_module.add_function(wrap_pyfunction!(qcis::py_qcis_load, &qcis_module)?)?;
    qcis_module.add_function(wrap_pyfunction!(qcis::py_qcis_loads, &qcis_module)?)?;
    qcis_module.add_function(wrap_pyfunction!(qcis::py_qcis_dump, &qcis_module)?)?;
    qcis_module.add_function(wrap_pyfunction!(qcis::py_qcis_dumps, &qcis_module)?)?;
    ir_module.add_submodule(&qcis_module)?;
    register_sys_module(&qcis_module, "cqlib._native.ir.qcis")?;

    parent.add_submodule(&ir_module)?;
    register_sys_module(&ir_module, "cqlib._native.ir")?;
    Ok(())
}
