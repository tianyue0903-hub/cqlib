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

//! Python exception types for error-mitigation bindings.

use crate::circuit::error::{CircuitError, CqlibError};
use cqlib_core::error_mitigation::ErrorMitigationError as CoreErrorMitigationError;
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(cqlib.error_mitigation, ErrorMitigationError, CqlibError);

pub(crate) fn error_mitigation_error_to_py_err(error: CoreErrorMitigationError) -> PyErr {
    let message = error.to_string();
    match error {
        CoreErrorMitigationError::Circuit(_) => CircuitError::new_err(message),
        CoreErrorMitigationError::Qis(_) => PyValueError::new_err(message),
        _ => ErrorMitigationError::new_err(message),
    }
}

pub(super) fn register_errors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "ErrorMitigationError",
        module.py().get_type::<ErrorMitigationError>(),
    )?;
    Ok(())
}
