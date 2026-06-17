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

//! Python bindings for cqlib-core metrics module.
//!
//! This module exposes quantum information metrics such as fidelity,
//! purity, and trace distance to Python.

use cqlib_core::qis::metrics as core_metrics;
use pyo3::prelude::*;

use crate::qis::qis_error_to_py_err;
use crate::qis::state::density_matrix::PyDensityMatrix;
use crate::qis::state::statevector::PyStatevector;

/// Calculates the purity of a pure quantum state.
///
/// Args:
///     sv (Statevector): The statevector.
///
/// Returns:
///     float: The purity value (theoretically 1.0 for normalized pure states).
///
/// Raises:
///     ValueError: If the calculation fails.
#[pyfunction]
pub fn purity_pure(sv: &PyStatevector) -> PyResult<f64> {
    core_metrics::purity_pure(&sv.inner).map_err(qis_error_to_py_err)
}

/// Calculates the purity of a mixed quantum state.
///
/// Args:
///     dm (DensityMatrix): The density matrix.
///
/// Returns:
///     float: The purity value in [1/2^N, 1.0].
///
/// Raises:
///     ValueError: If the calculation fails.
#[pyfunction]
pub fn purity_mixed(dm: &PyDensityMatrix) -> PyResult<f64> {
    core_metrics::purity_mixed(&dm.inner).map_err(qis_error_to_py_err)
}

/// Calculates the state fidelity between two pure quantum states.
///
/// Args:
///     sv1 (Statevector): The first statevector.
///     sv2 (Statevector): The second statevector.
///
/// Returns:
///     float: The fidelity value in [0.0, 1.0].
///
/// Raises:
///     ValueError: If the number of qubits do not match.
#[pyfunction]
pub fn state_fidelity_pure(sv1: &PyStatevector, sv2: &PyStatevector) -> PyResult<f64> {
    core_metrics::state_fidelity_pure(&sv1.inner, &sv2.inner).map_err(qis_error_to_py_err)
}

/// Calculates the trace distance between two pure quantum states.
///
/// Args:
///     sv1 (Statevector): The first statevector.
///     sv2 (Statevector): The second statevector.
///
/// Returns:
///     float: The trace distance in [0.0, 1.0].
///
/// Raises:
///     ValueError: If the number of qubits do not match.
#[pyfunction]
pub fn trace_distance_pure(sv1: &PyStatevector, sv2: &PyStatevector) -> PyResult<f64> {
    core_metrics::trace_distance_pure(&sv1.inner, &sv2.inner).map_err(qis_error_to_py_err)
}

/// Calculates the state fidelity between a pure state and a mixed state.
///
/// Args:
///     sv (Statevector): The pure state.
///     dm (DensityMatrix): The mixed state.
///
/// Returns:
///     float: The fidelity value in [0.0, 1.0].
///
/// Raises:
///     ValueError: If the number of qubits do not match.
#[pyfunction]
pub fn state_fidelity_pure_mixed(sv: &PyStatevector, dm: &PyDensityMatrix) -> PyResult<f64> {
    core_metrics::state_fidelity_pure_mixed(&sv.inner, &dm.inner).map_err(qis_error_to_py_err)
}

/// Calculates the von Neumann entropy of a mixed state.
///
/// Args:
///     dm (DensityMatrix): The density matrix.
///
/// Returns:
///     float: The entropy in bits (base-2 logarithm).
///
/// Raises:
///     ValueError: If eigendecomposition fails.
#[pyfunction]
pub fn entropy(dm: &PyDensityMatrix) -> PyResult<f64> {
    core_metrics::entropy(&dm.inner).map_err(qis_error_to_py_err)
}

/// Calculates the trace distance between two mixed quantum states.
///
/// Args:
///     dm1 (DensityMatrix): The first density matrix.
///     dm2 (DensityMatrix): The second density matrix.
///
/// Returns:
///     float: The trace distance.
///
/// Raises:
///     ValueError: If the number of qubits do not match.
#[pyfunction]
pub fn trace_distance_mixed(dm1: &PyDensityMatrix, dm2: &PyDensityMatrix) -> PyResult<f64> {
    core_metrics::trace_distance_mixed(&dm1.inner, &dm2.inner).map_err(qis_error_to_py_err)
}

/// Calculates the state fidelity between two mixed quantum states.
///
/// Args:
///     dm1 (DensityMatrix): The first density matrix.
///     dm2 (DensityMatrix): The second density matrix.
///
/// Returns:
///     float: The fidelity value.
///
/// Raises:
///     ValueError: If the number of qubits do not match or eigendecomposition fails.
#[pyfunction]
pub fn state_fidelity_mixed(dm1: &PyDensityMatrix, dm2: &PyDensityMatrix) -> PyResult<f64> {
    core_metrics::state_fidelity_mixed(&dm1.inner, &dm2.inner).map_err(qis_error_to_py_err)
}

/// Performs the partial transpose operation on a density matrix.
///
/// Args:
///     dm (DensityMatrix): The input density matrix.
///     target_qubits (list[int]): The qubit indices specifying the subsystem to transpose.
///
/// Returns:
///     DensityMatrix: The partially transposed density matrix.
///
/// Raises:
///     ValueError: If any target qubit index is out of bounds.
#[pyfunction]
pub fn partial_transpose(
    dm: &PyDensityMatrix,
    target_qubits: Vec<usize>,
) -> PyResult<PyDensityMatrix> {
    let new_dm =
        core_metrics::partial_transpose(&dm.inner, &target_qubits).map_err(qis_error_to_py_err)?;
    Ok(PyDensityMatrix { inner: new_dm })
}

/// Calculates the logarithmic negativity of a bipartite quantum state.
///
/// Args:
///     dm (DensityMatrix): The density matrix.
///     sys_a (list[int]): The qubit indices comprising subsystem A.
///
/// Returns:
///     float: The logarithmic negativity value (>= 0).
///
/// Raises:
///     ValueError: If subsystem indices are invalid or eigendecomposition fails.
#[pyfunction]
pub fn logarithmic_negativity(dm: &PyDensityMatrix, sys_a: Vec<usize>) -> PyResult<f64> {
    core_metrics::logarithmic_negativity(&dm.inner, &sys_a).map_err(qis_error_to_py_err)
}

/// Register the metrics module with Python.
pub fn register_metrics_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let metrics_module = PyModule::new(parent.py(), "metrics")?;

    metrics_module.add_function(wrap_pyfunction!(purity_pure, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(purity_mixed, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(state_fidelity_pure, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(trace_distance_pure, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(
        state_fidelity_pure_mixed,
        &metrics_module
    )?)?;
    metrics_module.add_function(wrap_pyfunction!(entropy, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(trace_distance_mixed, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(state_fidelity_mixed, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(partial_transpose, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(logarithmic_negativity, &metrics_module)?)?;

    parent.add_submodule(&metrics_module)?;
    Ok(())
}
