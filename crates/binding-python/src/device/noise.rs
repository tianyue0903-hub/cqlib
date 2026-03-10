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

use super::common::{parse_pauli, pauli_to_name, py_id_to_qubit};
use crate::circuit::PyStandardGate;
use cqlib_core::device::{NoiseModel, OperationKey, ReadoutError, SingleQubitNoise, TwoQubitNoise};
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[pyclass(name = "SingleQubitNoise", module = "cqlib.device")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PySingleQubitNoise {
    pub(crate) inner: SingleQubitNoise,
}

impl From<SingleQubitNoise> for PySingleQubitNoise {
    fn from(inner: SingleQubitNoise) -> Self {
        Self { inner }
    }
}

impl From<PySingleQubitNoise> for SingleQubitNoise {
    fn from(value: PySingleQubitNoise) -> Self {
        value.inner
    }
}

#[pymethods]
impl PySingleQubitNoise {
    #[staticmethod]
    fn bit_flip(p: f64) -> Self {
        Self {
            inner: SingleQubitNoise::BitFlip(p),
        }
    }

    #[staticmethod]
    fn phase_flip(p: f64) -> Self {
        Self {
            inner: SingleQubitNoise::PhaseFlip(p),
        }
    }

    #[staticmethod]
    fn pauli(px: f64, py: f64, pz: f64) -> Self {
        Self {
            inner: SingleQubitNoise::Pauli { px, py, pz },
        }
    }

    #[staticmethod]
    fn depolarizing(p: f64) -> Self {
        Self {
            inner: SingleQubitNoise::Depolarizing(p),
        }
    }

    #[staticmethod]
    fn amplitude_damping(gamma: f64) -> Self {
        Self {
            inner: SingleQubitNoise::AmplitudeDamping(gamma),
        }
    }

    #[staticmethod]
    fn phase_damping(lambda_: f64) -> Self {
        Self {
            inner: SingleQubitNoise::PhaseDamping(lambda_),
        }
    }

    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn to_kraus<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<Complex64>>> {
        self.inner
            .to_kraus()
            .into_iter()
            .map(|k| k.to_pyarray(py))
            .collect()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            SingleQubitNoise::BitFlip(_) => "bit_flip",
            SingleQubitNoise::PhaseFlip(_) => "phase_flip",
            SingleQubitNoise::Pauli { .. } => "pauli",
            SingleQubitNoise::Depolarizing(_) => "depolarizing",
            SingleQubitNoise::AmplitudeDamping(_) => "amplitude_damping",
            SingleQubitNoise::PhaseDamping(_) => "phase_damping",
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SingleQubitNoise::BitFlip(p) => format!("SingleQubitNoise.bit_flip({})", p),
            SingleQubitNoise::PhaseFlip(p) => format!("SingleQubitNoise.phase_flip({})", p),
            SingleQubitNoise::Pauli { px, py, pz } => {
                format!("SingleQubitNoise.pauli({}, {}, {})", px, py, pz)
            }
            SingleQubitNoise::Depolarizing(p) => format!("SingleQubitNoise.depolarizing({})", p),
            SingleQubitNoise::AmplitudeDamping(gamma) => {
                format!("SingleQubitNoise.amplitude_damping({})", gamma)
            }
            SingleQubitNoise::PhaseDamping(lambda_) => {
                format!("SingleQubitNoise.phase_damping({})", lambda_)
            }
        }
    }
}

#[pyclass(name = "TwoQubitNoise", module = "cqlib.device")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyTwoQubitNoise {
    pub(crate) inner: TwoQubitNoise,
}

impl From<TwoQubitNoise> for PyTwoQubitNoise {
    fn from(inner: TwoQubitNoise) -> Self {
        Self { inner }
    }
}

impl From<PyTwoQubitNoise> for TwoQubitNoise {
    fn from(value: PyTwoQubitNoise) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyTwoQubitNoise {
    #[staticmethod]
    fn depolarizing(p: f64) -> Self {
        Self {
            inner: TwoQubitNoise::Depolarizing(p),
        }
    }

    #[staticmethod]
    fn independent(q0_noise: PySingleQubitNoise, q1_noise: PySingleQubitNoise) -> Self {
        Self {
            inner: TwoQubitNoise::Independent {
                q0_noise: q0_noise.inner,
                q1_noise: q1_noise.inner,
            },
        }
    }

    #[staticmethod]
    fn correlated_pauli(op_q0: String, op_q1: String, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: TwoQubitNoise::CorrelatedPauli {
                op_q0: parse_pauli(&op_q0)?,
                op_q1: parse_pauli(&op_q1)?,
                p,
            },
        })
    }

    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn to_kraus<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<Complex64>>> {
        self.inner
            .to_kraus()
            .into_iter()
            .map(|k| k.to_pyarray(py))
            .collect()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            TwoQubitNoise::Depolarizing(_) => "depolarizing",
            TwoQubitNoise::Independent { .. } => "independent",
            TwoQubitNoise::CorrelatedPauli { .. } => "correlated_pauli",
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            TwoQubitNoise::Depolarizing(p) => format!("TwoQubitNoise.depolarizing({})", p),
            TwoQubitNoise::Independent { q0_noise, q1_noise } => format!(
                "TwoQubitNoise.independent({}, {})",
                PySingleQubitNoise::from(q0_noise).__repr__(),
                PySingleQubitNoise::from(q1_noise).__repr__()
            ),
            TwoQubitNoise::CorrelatedPauli { op_q0, op_q1, p } => format!(
                "TwoQubitNoise.correlated_pauli('{}', '{}', {})",
                pauli_to_name(op_q0),
                pauli_to_name(op_q1),
                p
            ),
        }
    }
}

#[pyclass(name = "ReadoutError", module = "cqlib.device")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyReadoutError {
    pub(crate) inner: ReadoutError,
}

impl From<ReadoutError> for PyReadoutError {
    fn from(inner: ReadoutError) -> Self {
        Self { inner }
    }
}

impl From<PyReadoutError> for ReadoutError {
    fn from(value: PyReadoutError) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyReadoutError {
    #[new]
    fn new(p_0_given_1: f64, p_1_given_0: f64) -> Self {
        Self {
            inner: ReadoutError {
                p_0_given_1,
                p_1_given_0,
            },
        }
    }

    #[getter]
    fn p_0_given_1(&self) -> f64 {
        self.inner.p_0_given_1
    }

    #[getter]
    fn p_1_given_0(&self) -> f64 {
        self.inner.p_1_given_0
    }

    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReadoutError(p_0_given_1={}, p_1_given_0={})",
            self.p_0_given_1(),
            self.p_1_given_0()
        )
    }
}

#[pyclass(name = "OperationKey", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyOperationKey {
    pub(crate) inner: OperationKey,
}

impl From<OperationKey> for PyOperationKey {
    fn from(inner: OperationKey) -> Self {
        Self { inner }
    }
}

impl From<PyOperationKey> for OperationKey {
    fn from(value: PyOperationKey) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyOperationKey {
    #[staticmethod]
    fn new_single(gate: PyStandardGate, q0: usize) -> PyResult<Self> {
        Ok(Self {
            inner: OperationKey::new_single(gate.inner, py_id_to_qubit(q0)?),
        })
    }

    #[staticmethod]
    fn new_double(gate: PyStandardGate, q0: usize, q1: usize) -> PyResult<Self> {
        let q0 = py_id_to_qubit(q0)?;
        let q1 = py_id_to_qubit(q1)?;
        let inner = OperationKey::new_double(gate.inner, q0, q1)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn new_triple(gate: PyStandardGate, q0: usize, q1: usize, q2: usize) -> PyResult<Self> {
        let q0 = py_id_to_qubit(q0)?;
        let q1 = py_id_to_qubit(q1)?;
        let q2 = py_id_to_qubit(q2)?;
        let inner = OperationKey::new_triple(gate.inner, q0, q1, q2)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.inner.qubits().to_vec()
    }

    #[getter]
    fn gate(&self) -> PyStandardGate {
        PyStandardGate::from(*self.inner.gate(), Vec::new())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !other.is_instance_of::<PyOperationKey>() {
            return Ok(false);
        }
        let other = other.extract::<PyOperationKey>()?;
        Ok(self.inner == other.inner)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!(
            "OperationKey(gate={}, qubits={:?})",
            self.inner.gate(),
            self.qubits()
        )
    }
}

#[pyclass(name = "NoiseModel", module = "cqlib.device")]
#[derive(Clone, Debug, Default)]
pub struct PyNoiseModel {
    pub(crate) inner: NoiseModel,
}

impl From<NoiseModel> for PyNoiseModel {
    fn from(inner: NoiseModel) -> Self {
        Self { inner }
    }
}

impl From<PyNoiseModel> for NoiseModel {
    fn from(value: PyNoiseModel) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyNoiseModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: NoiseModel::new(),
        }
    }

    fn add_readout_error(&mut self, qubit: usize, error: PyReadoutError) -> PyResult<()> {
        self.inner
            .add_readout_error(py_id_to_qubit(qubit)?, error.inner)
            .map_err(PyValueError::new_err)
    }

    fn add_single_qubit_error(
        &mut self,
        gate: PyStandardGate,
        qubit: usize,
        noise: PySingleQubitNoise,
    ) -> PyResult<()> {
        self.inner
            .add_single_qubit_error(gate.inner, py_id_to_qubit(qubit)?, noise.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn add_two_qubit_error(
        &mut self,
        gate: PyStandardGate,
        q0: usize,
        q1: usize,
        noise: PyTwoQubitNoise,
    ) -> PyResult<()> {
        self.inner
            .add_two_qubit_error(
                gate.inner,
                py_id_to_qubit(q0)?,
                py_id_to_qubit(q1)?,
                noise.inner,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_readout_error(&self, qubit: usize) -> PyResult<Option<PyReadoutError>> {
        Ok(self
            .inner
            .get_readout_error(&py_id_to_qubit(qubit)?)
            .copied()
            .map(PyReadoutError::from))
    }

    fn get_single_qubit_errors(&self, key: PyOperationKey) -> Option<Vec<PySingleQubitNoise>> {
        self.inner
            .get_single_qubit_errors(&key.inner)
            .cloned()
            .map(|v| v.into_iter().map(PySingleQubitNoise::from).collect())
    }

    fn get_two_qubit_errors(&self, key: PyOperationKey) -> Option<Vec<PyTwoQubitNoise>> {
        self.inner
            .get_two_qubit_errors(&key.inner)
            .cloned()
            .map(|v| v.into_iter().map(PyTwoQubitNoise::from).collect())
    }

    fn __repr__(&self) -> String {
        "NoiseModel()".to_string()
    }
}
