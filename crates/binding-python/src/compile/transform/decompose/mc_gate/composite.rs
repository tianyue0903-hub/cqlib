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
    decompose_fsim_n_clean, decompose_fsim_no_aux, decompose_hadamard_n_clean,
    decompose_hadamard_no_aux, decompose_phase_n_clean, decompose_phase_no_aux,
    decompose_qcis_n_clean, decompose_qcis_no_aux, decompose_swap_n_clean, decompose_swap_no_aux,
    decompose_unitary_n_clean, decompose_unitary_no_aux,
};
use pyo3::prelude::*;

fn extract_params(values: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<ParameterValue>> {
    values.iter().map(extract_parameter_value).collect()
}

#[pyfunction(name = "decompose_hadamard_no_aux")]
fn py_decompose_hadamard_no_aux(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    py.detach(move || decompose_hadamard_no_aux(&controls, target))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_hadamard_n_clean")]
fn py_decompose_hadamard_n_clean(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || decompose_hadamard_n_clean(&controls, target, &clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_swap_no_aux")]
fn py_decompose_swap_no_aux(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    py.detach(move || decompose_swap_no_aux(&controls, first, second))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_swap_n_clean")]
fn py_decompose_swap_n_clean(
    py: Python<'_>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || decompose_swap_n_clean(&controls, first, second, &clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_phase_no_aux")]
#[pyo3(signature = (phase, theta, controls, target))]
fn py_decompose_phase_no_aux(
    py: Python<'_>,
    phase: PyStandardGate,
    theta: Option<Bound<'_, PyAny>>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let theta = theta
        .map(|value| extract_parameter_value(&value))
        .transpose()?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    py.detach(move || decompose_phase_no_aux(phase.inner, theta.as_ref(), &controls, target))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_phase_n_clean")]
#[pyo3(signature = (phase, theta, controls, target, clean_ancillas))]
fn py_decompose_phase_n_clean(
    py: Python<'_>,
    phase: PyStandardGate,
    theta: Option<Bound<'_, PyAny>>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let theta = theta
        .map(|value| extract_parameter_value(&value))
        .transpose()?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || {
        decompose_phase_n_clean(
            phase.inner,
            theta.as_ref(),
            &controls,
            target,
            &clean_ancillas,
        )
    })
    .map(into_py_operations)
    .map_err(compiler_error)
}

macro_rules! gate_params_no_aux {
    ($rust_name:ident, $python_name:literal, $core:path) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            gate: PyStandardGate,
            params: Vec<Bound<'_, PyAny>>,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
        ) -> PyResult<Vec<PyValueOperation>> {
            let params = extract_params(params)?;
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            py.detach(move || $core(gate.inner, &params, &controls, target))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

macro_rules! gate_params_n_clean {
    ($rust_name:ident, $python_name:literal, $core:path) => {
        #[pyfunction(name = $python_name)]
        fn $rust_name(
            py: Python<'_>,
            gate: PyStandardGate,
            params: Vec<Bound<'_, PyAny>>,
            controls: PyIntListOrQubitList,
            target: PyIntOrQubit,
            clean_ancillas: PyIntListOrQubitList,
        ) -> PyResult<Vec<PyValueOperation>> {
            let params = extract_params(params)?;
            let controls: Vec<Qubit> = controls.into();
            let target: Qubit = target.into();
            let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
            py.detach(move || $core(gate.inner, &params, &controls, target, &clean_ancillas))
                .map(into_py_operations)
                .map_err(compiler_error)
        }
    };
}

gate_params_no_aux!(
    py_decompose_qcis_no_aux,
    "decompose_qcis_no_aux",
    decompose_qcis_no_aux
);
gate_params_n_clean!(
    py_decompose_qcis_n_clean,
    "decompose_qcis_n_clean",
    decompose_qcis_n_clean
);

#[pyfunction(name = "decompose_fsim_no_aux")]
fn py_decompose_fsim_no_aux(
    py: Python<'_>,
    params: Vec<Bound<'_, PyAny>>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let params = extract_params(params)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    py.detach(move || decompose_fsim_no_aux(&params, &controls, first, second))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_fsim_n_clean")]
fn py_decompose_fsim_n_clean(
    py: Python<'_>,
    params: Vec<Bound<'_, PyAny>>,
    controls: PyIntListOrQubitList,
    first: PyIntOrQubit,
    second: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let params = extract_params(params)?;
    let controls: Vec<Qubit> = controls.into();
    let first: Qubit = first.into();
    let second: Qubit = second.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || decompose_fsim_n_clean(&params, &controls, first, second, &clean_ancillas))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_unitary_no_aux")]
fn py_decompose_unitary_no_aux(
    py: Python<'_>,
    theta: Bound<'_, PyAny>,
    phi: Bound<'_, PyAny>,
    lambda_: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
) -> PyResult<Vec<PyValueOperation>> {
    let theta = extract_parameter_value(&theta)?;
    let phi = extract_parameter_value(&phi)?;
    let lambda = extract_parameter_value(&lambda_)?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    py.detach(move || decompose_unitary_no_aux(&theta, &phi, &lambda, &controls, target))
        .map(into_py_operations)
        .map_err(compiler_error)
}

#[pyfunction(name = "decompose_unitary_n_clean")]
fn py_decompose_unitary_n_clean(
    py: Python<'_>,
    theta: Bound<'_, PyAny>,
    phi: Bound<'_, PyAny>,
    lambda_: Bound<'_, PyAny>,
    controls: PyIntListOrQubitList,
    target: PyIntOrQubit,
    clean_ancillas: PyIntListOrQubitList,
) -> PyResult<Vec<PyValueOperation>> {
    let theta = extract_parameter_value(&theta)?;
    let phi = extract_parameter_value(&phi)?;
    let lambda = extract_parameter_value(&lambda_)?;
    let controls: Vec<Qubit> = controls.into();
    let target: Qubit = target.into();
    let clean_ancillas: Vec<Qubit> = clean_ancillas.into();
    py.detach(move || {
        decompose_unitary_n_clean(&theta, &phi, &lambda, &controls, target, &clean_ancillas)
    })
    .map(into_py_operations)
    .map_err(compiler_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_hadamard_no_aux,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_hadamard_n_clean,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_swap_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_swap_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_phase_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_phase_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_qcis_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_qcis_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_fsim_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_fsim_n_clean, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(py_decompose_unitary_no_aux, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        py_decompose_unitary_n_clean,
        module
    )?)?;
    Ok(())
}
