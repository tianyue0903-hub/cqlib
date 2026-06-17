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

//! Python exception types for circuit bindings.
//!
//! Rust errors remain strongly typed inside `cqlib-core`. At the Python boundary
//! they are converted into a small cqlib-specific exception hierarchy so callers
//! can catch errors by domain without depending on Rust enum representations.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(cqlib.circuit, CqlibError, PyException);
create_exception!(cqlib.circuit, CircuitError, CqlibError);
create_exception!(cqlib.circuit, ParameterError, CqlibError);
create_exception!(cqlib.circuit, QubitError, CqlibError);

/// Registers the exception hierarchy on `_native.circuit`.
pub(crate) fn register_errors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CqlibError", module.py().get_type::<CqlibError>())?;
    module.add("CircuitError", module.py().get_type::<CircuitError>())?;
    module.add("ParameterError", module.py().get_type::<ParameterError>())?;
    module.add("QubitError", module.py().get_type::<QubitError>())?;
    Ok(())
}
