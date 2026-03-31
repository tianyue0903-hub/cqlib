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

//! Python-exposed convenience constructors for common ansatz patterns.

use cqlib_core::circuit::ansatz::facades;
use pyo3::prelude::*;

use crate::qis::pauli::PyPauliString;

use super::feature_map::{PyPauliFeatureMap, PyZZFeatureMap};
use super::two_local::{PyEntanglementTopology, PyTwoLocal};

/// Creates a RealAmplitudes ansatz.
///
/// A hardware-efficient heuristic ansatz with a single RY rotation layer and
/// CX entanglement. This is widely used as a baseline in VQE experiments.
///
/// Structure: [RY layer] → [CX entanglement] × reps → [final RY layer]
///
/// Args:
///     num_qubits: Number of qubits (≥ 1).
///     reps: Number of [Rotation + Entanglement] layers.
///     entanglement: Connectivity topology (default: Linear).
///
/// Returns:
///     A TwoLocal configured as RealAmplitudes. Total parameters = (reps+1) * num_qubits.
///
/// Examples:
///     >>> from cqlib.circuit.ansatz import real_amplitudes, EntanglementTopology
///     >>> a = real_amplitudes(3, 2, EntanglementTopology.linear())
///     >>> a.num_parameters()
///     9
#[pyfunction]
pub fn real_amplitudes(
    num_qubits: usize,
    reps: usize,
    entanglement: PyRef<'_, PyEntanglementTopology>,
) -> PyTwoLocal {
    facades::real_amplitudes(num_qubits, reps, entanglement.inner.clone()).into()
}

/// Creates an EfficientSU2 ansatz.
///
/// A hardware-efficient ansatz spanning SU(2) on each qubit via [RY, RZ] rotation
/// layers and CX entanglement. Widely used in VQE and Quantum Machine Learning.
///
/// Structure: [RY+RZ layer] → [CX entanglement] × reps → [final RY+RZ layer]
///
/// Args:
///     num_qubits: Number of qubits (≥ 1).
///     reps: Number of [Rotation + Entanglement] layers.
///     entanglement: Connectivity topology.
///
/// Returns:
///     A TwoLocal configured as EfficientSU2. Total parameters = (reps+1) * num_qubits * 2.
///
/// Examples:
///     >>> from cqlib.circuit.ansatz import efficient_su2, EntanglementTopology
///     >>> a = efficient_su2(2, 1, EntanglementTopology.full())
///     >>> a.num_parameters()
///     8
#[pyfunction]
pub fn efficient_su2(
    num_qubits: usize,
    reps: usize,
    entanglement: PyRef<'_, PyEntanglementTopology>,
) -> PyTwoLocal {
    facades::efficient_su2(num_qubits, reps, entanglement.inner.clone()).into()
}

/// Creates a ZZFeatureMap.
///
/// A second-order Pauli-Z feature map (Z single-qubit + ZZ two-qubit interactions).
/// Widely used for quantum kernel methods in quantum machine learning.
///
/// Args:
///     num_qubits: Number of qubits (= number of input features, ≥ 1).
///     reps: Number of repetition layers.
///     entanglement: Connectivity for ZZ interactions (default: Full).
///
/// Returns:
///     A ZZFeatureMap. Always has `num_qubits` parameters.
///
/// Examples:
///     >>> from cqlib.circuit.ansatz import zz_feature_map, EntanglementTopology
///     >>> fm = zz_feature_map(3, 2, EntanglementTopology.full())
///     >>> fm.num_parameters()
///     3
#[pyfunction]
pub fn zz_feature_map(
    num_qubits: usize,
    reps: usize,
    entanglement: PyRef<'_, PyEntanglementTopology>,
) -> PyZZFeatureMap {
    facades::zz_feature_map(num_qubits, reps, entanglement.inner.clone()).into()
}

/// Creates a PauliFeatureMap with custom Pauli strings.
///
/// A general-purpose data encoding circuit using Pauli evolution gates.
/// Supports arbitrary Pauli strings and entanglement topologies.
///
/// Args:
///     num_qubits: Number of qubits (= number of input features, ≥ 1).
///     reps: Number of repetition layers.
///     paulis: List of PauliString templates. Each string's non-identity count
///             determines the locality k of that interaction.
///     entanglement: Connectivity topology for multi-qubit interactions.
///
/// Returns:
///     A PauliFeatureMap. Always has `num_qubits` parameters.
///
/// Raises:
///     ValueError: If any Pauli string is incompatible with the configuration.
///
/// Examples:
///     >>> from cqlib.circuit.ansatz import pauli_feature_map, EntanglementTopology
///     >>> from cqlib import PauliString
///     >>> fm = pauli_feature_map(3, 2, [PauliString("Z"), PauliString("ZZ")],
///     ...                        EntanglementTopology.full())
///     >>> fm.num_parameters()
///     3
#[pyfunction]
pub fn pauli_feature_map(
    num_qubits: usize,
    reps: usize,
    paulis: Vec<PyRef<'_, PyPauliString>>,
    entanglement: PyRef<'_, PyEntanglementTopology>,
) -> PyPauliFeatureMap {
    let rust_paulis: Vec<_> = paulis
        .iter()
        .map(|p| (p.inner.clone(), p.inner.to_string()))
        .collect();
    facades::pauli_feature_map(num_qubits, reps, rust_paulis, entanglement.inner.clone()).into()
}
