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

use crate::circuit::error::CqlibError;
use cqlib_core::compile::resource::ResourceError as CoreResourceError;
use pyo3::create_exception;
use pyo3::prelude::*;

create_exception!(cqlib.compile.resource, ResourceError, CqlibError);
create_exception!(
    cqlib.compile.resource,
    ResourceUnavailableError,
    ResourceError
);

/// Maps core resource failures without forcing planners to parse messages.
pub(super) fn resource_error_to_py_err(error: CoreResourceError) -> PyErr {
    match error {
        CoreResourceError::InsufficientResources { .. }
        | CoreResourceError::CapacityExceeded { .. }
        | CoreResourceError::QubitIdOverflow => {
            ResourceUnavailableError::new_err(error.to_string())
        }
        _ => ResourceError::new_err(error.to_string()),
    }
}

pub(super) fn register_errors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ResourceError", module.py().get_type::<ResourceError>())?;
    module.add(
        "ResourceUnavailableError",
        module.py().get_type::<ResourceUnavailableError>(),
    )?;
    Ok(())
}
