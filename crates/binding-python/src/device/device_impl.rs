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

//! Python bindings for quantum device hardware characteristics.
//!
//! This module provides Python wrappers around the core device types:
//!
//! - [`PyInstructionProp`]: Calibration data for quantum gates (error rates, duration)
//! - [`PyQubitProp`]: Physical properties of individual qubits (T1, T2, readout errors)
//! - [`PyEdgeProp`]: Properties of coupling edges between qubits
//! - [`PyDevice`]: Complete hardware description including topology and calibration data
//!
//! # Example
//!
//! ```python
//! from cqlib.device import Device, Topology, QubitProp
//! from datetime import datetime, timezone
//!
//! # Create device topology
//! topology = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])
//!
//! # Initialize device
//! device = Device("superconducting_qpu", [0, 1, 2], topology)
//!
//! # Set calibration timestamp
//! device.calibration_time = datetime.now(timezone.utc)
//!
//! # Set default coherence times
//! device.default_t1 = 100.0
//! device.default_t2 = 50.0
//!
//! # Add qubit-specific properties
//! prop = QubitProp(readout_error=0.01)
//! prop.t1 = 120.0
//! device.add_qubit_properties(0, prop)
//! ```

use crate::circuit::bit::{PyIntListOrQubitList, PyIntOrQubit};
use crate::circuit::{PyInstruction, PyQubit};
use crate::device::topology::PyTopology;
use cqlib_core::circuit::Qubit;
use cqlib_core::device::{Device, EdgeProp, InstructionProp, QubitProp};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use time::OffsetDateTime;

/// Python wrapper for [`InstructionProp`].
///
/// Represents calibration data for a quantum gate executed on specific qubits,
/// including the gate's error rate (infidelity) and optionally its execution duration.
///
/// # Python Example
///
/// ```python
/// from cqlib.device import InstructionProp, StandardGate
///
/// # Create properties for an H gate with 0.1% error rate
/// prop = InstructionProp(StandardGate.H, error_rate=0.001)
///
/// # Optionally set gate duration in nanoseconds
/// prop.length = 35.0  # 35 ns
/// ```
#[pyclass(name = "InstructionProp", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyInstructionProp {
    pub(crate) inner: InstructionProp,
}

impl From<InstructionProp> for PyInstructionProp {
    fn from(inner: InstructionProp) -> Self {
        Self { inner }
    }
}

impl From<PyInstructionProp> for InstructionProp {
    fn from(value: PyInstructionProp) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyInstructionProp {
    #[new]
    fn new(instruction: PyInstruction, error_rate: f64) -> Self {
        Self {
            inner: InstructionProp::new(instruction.inner, error_rate),
        }
    }

    #[setter]
    fn set_length(&mut self, length: f64) {
        self.inner.set_length(length);
    }

    #[setter]
    fn set_instruction(&mut self, instruction: PyInstruction) {
        self.inner.set_instruction(instruction.inner);
    }

    #[setter]
    fn set_error_rate(&mut self, error_rate: f64) {
        self.inner.set_error_rate(error_rate);
    }

    #[getter]
    fn instruction(&self) -> PyInstruction {
        PyInstruction::from(self.inner.instruction().clone())
    }

    #[getter]
    fn error_rate(&self) -> f64 {
        self.inner.error_rate()
    }

    #[getter]
    fn length(&self) -> Option<f64> {
        self.inner.length()
    }

    fn __repr__(&self) -> String {
        let instruction_name = format!("{}", self.inner.instruction());
        match self.length() {
            Some(length) => format!(
                "InstructionProp(instruction={}, error_rate={}, length={})",
                instruction_name,
                self.error_rate(),
                length
            ),
            None => format!(
                "InstructionProp(instruction={}, error_rate={})",
                instruction_name,
                self.error_rate()
            ),
        }
    }
}

/// Python wrapper for [`QubitProp`].
///
/// Represents the physical properties and operational characteristics of a single
/// quantum qubit, including coherence times (T1, T2), readout errors, and supported
/// native gates.
///
/// # Python Example
///
/// ```python
/// from cqlib.device import QubitProp
///
/// # Create qubit properties with 1% readout error
/// prop = QubitProp(readout_error=0.01)
///
/// # Set coherence times (in microseconds)
/// prop.t1 = 50.0
/// prop.t2 = 30.0
///
/// # Set measurement discrimination errors
/// prop.prob_meas0_prep1 = 0.02  # P(meas 0 | prep 1)
/// prop.prob_meas1_prep0 = 0.01  # P(meas 1 | prep 0)
/// ```
#[pyclass(name = "QubitProp", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyQubitProp {
    pub(crate) inner: QubitProp,
}

impl From<QubitProp> for PyQubitProp {
    fn from(inner: QubitProp) -> Self {
        Self { inner }
    }
}

impl From<PyQubitProp> for QubitProp {
    fn from(value: PyQubitProp) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyQubitProp {
    #[new]
    fn new(readout_error: f64) -> Self {
        Self {
            inner: QubitProp::new(readout_error),
        }
    }

    #[setter]
    fn set_prob_meas0_prep1(&mut self, prob: f64) {
        self.inner.set_prob_meas0_prep1(prob);
    }

    #[setter]
    fn set_prob_meas1_prep0(&mut self, prob: f64) {
        self.inner.set_prob_meas1_prep0(prob);
    }

    #[setter]
    fn set_t1(&mut self, t1: f64) {
        self.inner.set_t1(t1);
    }

    #[setter]
    fn set_t2(&mut self, t2: f64) {
        self.inner.set_t2(t2);
    }

    #[setter]
    fn set_frequency(&mut self, frequency: f64) {
        self.inner.set_frequency(frequency);
    }

    #[setter]
    fn set_native_instruction(&mut self, prop: PyInstructionProp) {
        self.inner.set_native_instruction(prop.inner);
    }

    #[getter]
    fn readout_error(&self) -> f64 {
        self.inner.readout_error()
    }

    #[getter]
    fn prob_meas0_prep1(&self) -> Option<f64> {
        self.inner.prob_meas0_prep1()
    }

    #[getter]
    fn prob_meas1_prep0(&self) -> Option<f64> {
        self.inner.prob_meas1_prep0()
    }

    #[getter]
    fn t1(&self) -> Option<f64> {
        self.inner.t1()
    }

    #[getter]
    fn t2(&self) -> Option<f64> {
        self.inner.t2()
    }

    #[getter]
    fn frequency(&self) -> Option<f64> {
        self.inner.frequency()
    }

    #[getter]
    fn native_instructions(&self) -> Vec<PyInstructionProp> {
        self.inner
            .native_instructions()
            .iter()
            .cloned()
            .map(PyInstructionProp::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "QubitProp(readout_error={}, t1={:?}, t2={:?}, frequency={:?}, native_instructions={})",
            self.readout_error(),
            self.t1(),
            self.t2(),
            self.frequency(),
            self.native_instructions().len()
        )
    }
}

#[pyclass(name = "EdgeProp", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyEdgeProp {
    pub(crate) inner: EdgeProp,
}

impl From<EdgeProp> for PyEdgeProp {
    fn from(inner: EdgeProp) -> Self {
        Self { inner }
    }
}

impl From<PyEdgeProp> for EdgeProp {
    fn from(value: PyEdgeProp) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyEdgeProp {
    #[new]
    fn new() -> Self {
        Self {
            inner: EdgeProp::new(),
        }
    }

    #[setter]
    fn set_native_instruction(&mut self, prop: PyInstructionProp) {
        self.inner.set_native_instruction(prop.inner);
    }

    #[getter]
    fn native_instructions(&self) -> Vec<PyInstructionProp> {
        self.inner
            .native_instructions()
            .iter()
            .cloned()
            .map(PyInstructionProp::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EdgeProp(native_instructions={})",
            self.native_instructions().len()
        )
    }
}

/// Python wrapper for [`Device`].
///
/// Represents a complete quantum device characterization, including topology,
/// qubit properties, coupling edge properties, and default calibration values.
///
/// This is the primary interface for describing quantum hardware to the compiler
/// and simulator.
///
/// # Python Example
///
/// ```python
/// from cqlib.device import Device, Topology, QubitProp
/// from datetime import datetime, timezone
///
/// # Create device topology
/// topology = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])
///
/// # Initialize device
/// device = Device("superconducting_qpu", [0, 1, 2], topology)
///
/// # Set calibration timestamp
/// device.calibration_time = datetime.now(timezone.utc)
///
/// # Set default coherence times
/// device.default_t1 = 100.0
/// device.default_t2 = 50.0
///
/// # Add qubit-specific properties
/// prop = QubitProp(readout_error=0.01)
/// prop.t1 = 120.0
/// device.add_qubit_properties(0, prop)
/// ```
#[pyclass(name = "Device", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyDevice {
    pub(crate) inner: Device,
}

impl From<Device> for PyDevice {
    fn from(inner: Device) -> Self {
        Self { inner }
    }
}

impl From<PyDevice> for Device {
    fn from(value: PyDevice) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(name: String, qubits: PyIntListOrQubitList, topology: PyTopology) -> PyResult<Self> {
        let inner = Device::new(
            name,
            <PyIntListOrQubitList as Into<Vec<Qubit>>>::into(qubits)
                .into_iter()
                .collect(),
            topology.inner,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[setter]
    pub fn set_native_gates(&mut self, gates: Vec<PyInstruction>) {
        let gates = gates.into_iter().map(|g| g.inner).collect();
        self.inner.set_native_gates(gates)
    }

    #[setter]
    fn set_default_t1(&mut self, t1: f64) {
        self.inner.set_default_t1(t1)
    }

    #[setter]
    fn set_default_t2(&mut self, t2: f64) {
        self.inner.set_default_t2(t2)
    }

    #[setter]
    fn set_default_readout_error(&mut self, error: f64) {
        self.inner.set_default_readout_error(error)
    }

    #[setter]
    fn set_default_single_qubit_error(&mut self, error: f64) {
        self.inner.set_default_single_qubit_error(error)
    }

    #[setter]
    fn set_default_two_qubit_error(&mut self, error: f64) {
        self.inner.set_default_two_qubit_error(error)
    }

    #[setter]
    fn set_calibration_time(&mut self, datetime: chrono::DateTime<chrono::Utc>) {
        let timestamp = datetime.timestamp();
        let time = OffsetDateTime::from_unix_timestamp(timestamp)
            .unwrap_or_else(|_| OffsetDateTime::now_utc());
        self.inner.set_calibration_time(time);
    }

    #[getter]
    fn calibration_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.inner.calibration_time().map(|t| {
            chrono::DateTime::from_timestamp(t.unix_timestamp(), 0).unwrap_or_else(chrono::Utc::now)
        })
    }

    fn add_qubit_properties(&mut self, qubit: PyIntOrQubit, props: PyQubitProp) -> PyResult<()> {
        self.inner
            .add_qubit_properties(qubit.into(), props.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn add_edge_properties(
        &mut self,
        control: PyIntOrQubit,
        target: PyIntOrQubit,
        props: PyEdgeProp,
    ) -> PyResult<()> {
        self.inner
            .add_edge_properties(control.into(), target.into(), props.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn qubits(&self) -> Vec<PyQubit> {
        self.inner.qubits().map(PyQubit::from).collect()
    }

    #[getter]
    fn invalid_qubits(&self) -> Vec<PyQubit> {
        self.inner.invalid_qubits().map(PyQubit::from).collect()
    }

    #[setter]
    fn set_invalid_qubits(&mut self, qubits: PyIntListOrQubitList) {
        let qubits: std::collections::HashSet<_> =
            <PyIntListOrQubitList as Into<Vec<Qubit>>>::into(qubits)
                .into_iter()
                .collect();
        self.inner.set_invalid_qubits(qubits);
    }

    #[getter]
    fn topology(&self) -> PyTopology {
        PyTopology {
            inner: self.inner.topology().clone(),
        }
    }

    #[getter]
    fn native_gates(&self) -> Vec<PyInstruction> {
        self.inner
            .native_gates()
            .iter()
            .cloned()
            .map(PyInstruction::from)
            .collect()
    }

    fn qubit_properties(&self, qubit: PyIntOrQubit) -> PyResult<Option<PyQubitProp>> {
        Ok(self
            .inner
            .qubit_properties(qubit.into())
            .cloned()
            .map(PyQubitProp::from))
    }

    fn edge_properties(
        &self,
        control: PyIntOrQubit,
        target: PyIntOrQubit,
    ) -> PyResult<Option<PyEdgeProp>> {
        Ok(self
            .inner
            .edge_properties(control.into(), target.into())
            .cloned()
            .map(PyEdgeProp::from))
    }

    fn get_t1(&self, qubit: PyIntOrQubit) -> PyResult<Option<f64>> {
        Ok(self.inner.get_t1(qubit.into()))
    }

    fn get_t2(&self, qubit: PyIntOrQubit) -> PyResult<Option<f64>> {
        Ok(self.inner.get_t2(qubit.into()))
    }

    fn get_readout_error(&self, qubit: PyIntOrQubit) -> PyResult<Option<f64>> {
        Ok(self.inner.get_readout_error(qubit.into()))
    }

    #[getter]
    fn default_single_qubit_error(&self) -> Option<f64> {
        self.inner.default_single_qubit_error()
    }

    #[getter]
    fn default_two_qubit_error(&self) -> Option<f64> {
        self.inner.default_two_qubit_error()
    }

    fn __repr__(&self) -> String {
        format!("Device(name='{}')", self.name(),)
    }
}
