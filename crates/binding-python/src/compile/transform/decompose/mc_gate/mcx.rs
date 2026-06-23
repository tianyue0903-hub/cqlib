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

use super::{compiler_error, into_py_operations};
use crate::circuit::PyValueOperation;
use crate::circuit::bit::{PyIntListOrQubitList, PyIntOrQubit};
use cqlib_core::circuit::Qubit;
use cqlib_core::compile::transform::decompose::mc_gate::{
    decompose_mcx_1_clean_b95, decompose_mcx_1_clean_kg24, decompose_mcx_1_dirty,
    decompose_mcx_2_clean, decompose_mcx_2_dirty, decompose_mcx_n_clean, decompose_mcx_n_dirty,
    decompose_mcx_no_aux, decompose_mcx_small,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

macro_rules! mcx_basic {
    ($rust_name:ident, $python_name:literal, $core:path) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            py.detach(move || $core(&controls, target))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

macro_rules! mcx_many_ancillas {
    ($rust_name:ident, $python_name:literal, $core:path, $argument:ident) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
            $argument: PyIntListOrQubitList,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            let $argument: Vec<Qubit> = $argument.into();
            py.detach(move || $core(&controls, target, &$argument))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

macro_rules! mcx_one_ancilla {
    ($rust_name:ident, $python_name:literal, $core:path, $argument:ident) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
            $argument: PyIntOrQubit,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            let $argument: Qubit = $argument.into();
            py.detach(move || $core(&controls, target, $argument))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

mcx_basic!(
    py_decompose_mcx_small,
    "decompose_mcx_small",
    decompose_mcx_small
);
mcx_basic!(
    py_decompose_mcx_no_aux,
    "decompose_mcx_no_aux",
    decompose_mcx_no_aux
);
mcx_many_ancillas!(
    py_decompose_mcx_n_clean,
    "decompose_mcx_n_clean",
    decompose_mcx_n_clean,
    clean_ancillas
);
mcx_many_ancillas!(
    py_decompose_mcx_n_dirty,
    "decompose_mcx_n_dirty",
    decompose_mcx_n_dirty,
    dirty_ancillas
);
mcx_one_ancilla!(
    py_decompose_mcx_1_clean_b95,
    "decompose_mcx_1_clean_b95",
    decompose_mcx_1_clean_b95,
    clean_ancilla
);
mcx_one_ancilla!(
    py_decompose_mcx_1_clean_kg24,
    "decompose_mcx_1_clean_kg24",
    decompose_mcx_1_clean_kg24,
    clean_ancilla
);
mcx_one_ancilla!(
    py_decompose_mcx_1_dirty,
    "decompose_mcx_1_dirty",
    decompose_mcx_1_dirty,
    dirty_ancilla
);

#[pyfunction(name = "decompose_mcx_2_clean")]
fn py_decompose_mcx_2_clean(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas: [Qubit; 2] =
        Vec::<Qubit>::from(clean_ancillas)
            .try_into()
            .map_err(|values: Vec<Qubit>| {
                PyValueError::new_err(format!(
                    "decompose_mcx_2_clean requires exactly 2 clean ancillas, got {}",
                    values.len()
                ))
            })?;
    py.detach(move || decompose_mcx_2_clean(&controls, target, clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_mcx_2_dirty")]
fn py_decompose_mcx_2_dirty(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    dirty_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let dirty_ancillas: [Qubit; 2] =
        Vec::<Qubit>::from(dirty_ancillas)
            .try_into()
            .map_err(|values: Vec<Qubit>| {
                PyValueError::new_err(format!(
                    "decompose_mcx_2_dirty requires exactly 2 dirty ancillas, got {}",
                    values.len()
                ))
            })?;
    py.detach(move || decompose_mcx_2_dirty(&controls, target, dirty_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_small, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_n_dirty, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_mcx_1_clean_b95,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_mcx_1_clean_kg24,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_1_dirty, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_2_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mcx_2_dirty, module)?)?;
    Ok(())
}
