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
use crate::circuit::gate::PyStandardGate;
use crate::circuit::operation::extract_parameter_value;
use cqlib_core::circuit::{ParameterValue, Qubit};
use cqlib_core::compile::transform::decompose::mc_gate::{
    decompose_mc_rzz_n_clean, decompose_mc_rzz_no_aux, decompose_pauli_1_clean_b95,
    decompose_pauli_1_clean_kg24, decompose_pauli_1_dirty, decompose_pauli_2_clean,
    decompose_pauli_2_dirty, decompose_pauli_n_clean, decompose_pauli_n_dirty,
    decompose_pauli_no_aux, decompose_pauli_rotation_n_clean, decompose_pauli_rotation_no_aux,
    decompose_pauli_small, decompose_rotation_n_clean, decompose_rotation_no_aux,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

macro_rules! pauli_basic {
    ($rust_name:ident, $python_name:literal, $core:path) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            pauli: PyStandardGate,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            py.detach(move || $core(pauli.inner, &controls, target))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

macro_rules! pauli_many_ancillas {
    ($rust_name:ident, $python_name:literal, $core:path, $argument:ident) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            pauli: PyStandardGate,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
            $argument: PyIntListOrQubitList,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            let $argument: Vec<Qubit> = $argument.into();
            py.detach(move || $core(pauli.inner, &controls, target, &$argument))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

macro_rules! pauli_one_ancilla {
    ($rust_name:ident, $python_name:literal, $core:path, $argument:ident) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            pauli: PyStandardGate,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
            $argument: PyIntOrQubit,
        ) -> PyResult<Vec<PyValueOperation>> {
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            let $argument: Qubit = $argument.into();
            py.detach(move || $core(pauli.inner, &controls, target, $argument))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

pauli_basic!(
    py_decompose_pauli_small,
    "decompose_pauli_small",
    decompose_pauli_small
);
pauli_basic!(
    py_decompose_pauli_no_aux,
    "decompose_pauli_no_aux",
    decompose_pauli_no_aux
);
pauli_many_ancillas!(
    py_decompose_pauli_n_clean,
    "decompose_pauli_n_clean",
    decompose_pauli_n_clean,
    clean_ancillas
);
pauli_many_ancillas!(
    py_decompose_pauli_n_dirty,
    "decompose_pauli_n_dirty",
    decompose_pauli_n_dirty,
    dirty_ancillas
);
pauli_one_ancilla!(
    py_decompose_pauli_1_clean_b95,
    "decompose_pauli_1_clean_b95",
    decompose_pauli_1_clean_b95,
    clean_ancilla
);
pauli_one_ancilla!(
    py_decompose_pauli_1_clean_kg24,
    "decompose_pauli_1_clean_kg24",
    decompose_pauli_1_clean_kg24,
    clean_ancilla
);
pauli_one_ancilla!(
    py_decompose_pauli_1_dirty,
    "decompose_pauli_1_dirty",
    decompose_pauli_1_dirty,
    dirty_ancilla
);

fn two_ancillas(value: PyIntListOrQubitList, name: &str) -> PyResult<[Qubit; 2]> {
    Vec::<Qubit>::from(value)
        .try_into()
        .map_err(|values: Vec<Qubit>| {
            PyValueError::new_err(format!(
                "{name} requires exactly 2 ancillas, got {}",
                values.len()
            ))
        })
}

#[pyfunction(name = "decompose_pauli_2_clean")]
fn py_decompose_pauli_2_clean(
    py: Python<'_>,
    pauli: PyStandardGate,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas = two_ancillas(clean_ancillas, "decompose_pauli_2_clean")?;
    py.detach(move || decompose_pauli_2_clean(pauli.inner, &controls, target, clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_pauli_2_dirty")]
fn py_decompose_pauli_2_dirty(
    py: Python<'_>,
    pauli: PyStandardGate,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    dirty_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let dirty_ancillas = two_ancillas(dirty_ancillas, "decompose_pauli_2_dirty")?;
    py.detach(move || decompose_pauli_2_dirty(pauli.inner, &controls, target, dirty_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_rotation_no_aux")]
fn py_decompose_rotation_no_aux(
    py: Python<'_>,
    rotation: PyStandardGate,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    py.detach(move || decompose_rotation_no_aux(rotation.inner, &theta, &controls, target))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_rotation_n_clean")]
fn py_decompose_rotation_n_clean(
    py: Python<'_>,
    rotation: PyStandardGate,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || {
        decompose_rotation_n_clean(rotation.inner, &theta, &controls, target, &clean_ancillas)
    })
    .map(into_py_operations)
    .map_err(compiler_error)
}

#[pyfunction(name = "decompose_pauli_rotation_no_aux")]
fn py_decompose_pauli_rotation_no_aux(
    py: Python<'_>,
    rotation: PyStandardGate,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    py.detach(move || {
        decompose_pauli_rotation_no_aux(rotation.inner, &theta, &controls, first, second)
    })
    .map(into_py_operations)
    .map_err(compiler_error)
}

#[pyfunction(name = "decompose_pauli_rotation_n_clean")]
fn py_decompose_pauli_rotation_n_clean(
    py: Python<'_>,
    rotation: PyStandardGate,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || {
        decompose_pauli_rotation_n_clean(
            rotation.inner,
            &theta,
            &controls,
            first,
            second,
            &clean_ancillas,
        )
    })
    .map(into_py_operations)
    .map_err(compiler_error)
}

#[pyfunction(name = "decompose_mc_rzz_no_aux")]
fn py_decompose_mc_rzz_no_aux(
    py: Python<'_>,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    py.detach(move || decompose_mc_rzz_no_aux(&theta, &controls, first, second))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_mc_rzz_n_clean")]
fn py_decompose_mc_rzz_n_clean(
    py: Python<'_>,
    theta: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let theta: ParameterValue = extract_parameter_value(&theta)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || decompose_mc_rzz_n_clean(&theta, &controls, first, second, &clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_small, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_n_dirty, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_pauli_1_clean_b95,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_pauli_1_clean_kg24,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_1_dirty, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_2_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_pauli_2_dirty, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_rotation_no_aux,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_rotation_n_clean,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_pauli_rotation_no_aux,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_pauli_rotation_n_clean,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mc_rzz_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_mc_rzz_n_clean, module)?)?;
    Ok(())
}
