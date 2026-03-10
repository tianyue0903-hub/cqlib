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

use super::common::{py_id_to_qubit, qubit_to_py_id};
use crate::circuit::PyInstruction;
use crate::compile::PyTopology;
use cqlib_core::circuit::Qubit;
use cqlib_core::device::{Device, EdgeProp, InstructionProp, Layout, QubitProp};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

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

    fn with_length(&self, length: f64) -> Self {
        Self {
            inner: self.inner.clone().with_length(length),
        }
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

    fn with_prob_meas0_prep1(&self, prob: f64) -> Self {
        Self {
            inner: self.inner.clone().with_prob_meas0_prep1(prob),
        }
    }

    fn with_prob_meas1_prep0(&self, prob: f64) -> Self {
        Self {
            inner: self.inner.clone().with_prob_meas1_prep0(prob),
        }
    }

    fn with_t1(&self, t1: f64) -> Self {
        Self {
            inner: self.inner.clone().with_t1(t1),
        }
    }

    fn with_t2(&self, t2: f64) -> Self {
        Self {
            inner: self.inner.clone().with_t2(t2),
        }
    }

    fn with_frequency(&self, frequency: f64) -> Self {
        Self {
            inner: self.inner.clone().with_frequency(frequency),
        }
    }

    fn with_native_instruction(&self, prop: PyInstructionProp) -> Self {
        Self {
            inner: self.inner.clone().with_native_instruction(prop.inner),
        }
    }

    #[getter]
    fn readout_error(&self) -> f64 {
        self.inner.readout_error()
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

    fn with_native_instruction(&self, prop: PyInstructionProp) -> Self {
        Self {
            inner: self.inner.clone().with_native_instruction(prop.inner),
        }
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
    fn new(name: String, topology: PyTopology) -> Self {
        Self {
            inner: Device::new(name, topology.inner),
        }
    }

    fn with_native_gates(&self, gates: Vec<PyInstruction>) -> Self {
        let gates = gates.into_iter().map(|g| g.inner).collect();
        Self {
            inner: self.inner.clone().with_native_gates(gates),
        }
    }

    fn with_default_t1(&self, t1: f64) -> Self {
        Self {
            inner: self.inner.clone().with_default_t1(t1),
        }
    }

    fn with_default_t2(&self, t2: f64) -> Self {
        Self {
            inner: self.inner.clone().with_default_t2(t2),
        }
    }

    fn with_default_readout_error(&self, error: f64) -> Self {
        Self {
            inner: self.inner.clone().with_default_readout_error(error),
        }
    }

    fn with_default_single_qubit_error(&self, error: f64) -> Self {
        Self {
            inner: self.inner.clone().with_default_single_qubit_error(error),
        }
    }

    fn with_default_two_qubit_error(&self, error: f64) -> Self {
        Self {
            inner: self.inner.clone().with_default_two_qubit_error(error),
        }
    }

    fn add_qubit_properties(&mut self, qubit: usize, props: PyQubitProp) -> PyResult<()> {
        let qubit = py_id_to_qubit(qubit)?;
        self.inner
            .add_qubit_properties(qubit, props.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn add_edge_properties(
        &mut self,
        control: usize,
        target: usize,
        props: PyEdgeProp,
    ) -> PyResult<()> {
        let control = py_id_to_qubit(control)?;
        let target = py_id_to_qubit(target)?;
        self.inner
            .add_edge_properties(control, target, props.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.inner.qubits().map(qubit_to_py_id).collect()
    }

    #[getter]
    fn invalid_qubits(&self) -> Vec<usize> {
        self.inner.invalid_qubits().map(qubit_to_py_id).collect()
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

    fn qubit_properties(&self, qubit: usize) -> PyResult<Option<PyQubitProp>> {
        let qubit = py_id_to_qubit(qubit)?;
        Ok(self
            .inner
            .qubit_properties(qubit)
            .cloned()
            .map(PyQubitProp::from))
    }

    fn edge_properties(&self, control: usize, target: usize) -> PyResult<Option<PyEdgeProp>> {
        let control = py_id_to_qubit(control)?;
        let target = py_id_to_qubit(target)?;
        Ok(self
            .inner
            .edge_properties(control, target)
            .cloned()
            .map(PyEdgeProp::from))
    }

    fn get_t1(&self, qubit: usize) -> PyResult<Option<f64>> {
        Ok(self.inner.get_t1(py_id_to_qubit(qubit)?))
    }

    fn get_t2(&self, qubit: usize) -> PyResult<Option<f64>> {
        Ok(self.inner.get_t2(py_id_to_qubit(qubit)?))
    }

    fn get_readout_error(&self, qubit: usize) -> PyResult<Option<f64>> {
        Ok(self.inner.get_readout_error(py_id_to_qubit(qubit)?))
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
        format!(
            "Device(name='{}', qubits={}, invalid_qubits={}, native_gates={})",
            self.name(),
            self.qubits().len(),
            self.invalid_qubits().len(),
            self.native_gates().len()
        )
    }
}

#[pyclass(name = "Layout", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyLayout {
    pub(crate) inner: Layout,
}

impl From<Layout> for PyLayout {
    fn from(inner: Layout) -> Self {
        Self { inner }
    }
}

impl From<PyLayout> for Layout {
    fn from(value: PyLayout) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyLayout {
    #[new]
    #[pyo3(signature = (logical, physical, init_map=None))]
    fn new(
        logical: Vec<usize>,
        physical: Vec<usize>,
        init_map: Option<HashMap<usize, usize>>,
    ) -> PyResult<Self> {
        let logical = logical
            .into_iter()
            .map(py_id_to_qubit)
            .collect::<PyResult<Vec<_>>>()?;
        let physical = physical
            .into_iter()
            .map(py_id_to_qubit)
            .collect::<PyResult<Vec<_>>>()?;
        let init_map = init_map
            .map(|m| {
                m.into_iter()
                    .map(|(v, p)| Ok((py_id_to_qubit(v)?, py_id_to_qubit(p)?)))
                    .collect::<PyResult<HashMap<_, _>>>()
            })
            .transpose()?;

        let inner = Layout::new(logical, physical, init_map)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_logical(&self) -> usize {
        self.inner.num_logical()
    }

    #[getter]
    fn num_ancilla(&self) -> usize {
        self.inner.num_ancilla()
    }

    #[getter]
    fn num_physical(&self) -> usize {
        self.inner.num_physical()
    }

    fn get_physical(&self, virtual_id: usize) -> PyResult<Option<usize>> {
        let virtual_id = py_id_to_qubit(virtual_id)?;
        Ok(self.inner.get_physical(virtual_id).map(qubit_to_py_id))
    }

    fn get_virtual(&self, physical_id: usize) -> PyResult<Option<usize>> {
        let physical_id = py_id_to_qubit(physical_id)?;
        Ok(self.inner.get_virtual(physical_id).map(qubit_to_py_id))
    }

    #[getter]
    fn logical_qubits(&self) -> Vec<usize> {
        self.inner.logical_qubits().map(qubit_to_py_id).collect()
    }

    #[getter]
    fn ancilla_qubits(&self) -> Vec<usize> {
        self.inner.ancilla_qubits().map(qubit_to_py_id).collect()
    }

    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        self.inner.physical_qubits().map(qubit_to_py_id).collect()
    }

    #[getter]
    fn v2p_map(&self) -> HashMap<usize, usize> {
        self.inner
            .v2p_map()
            .iter()
            .map(|(k, v)| (qubit_to_py_id(*k), qubit_to_py_id(*v)))
            .collect()
    }

    #[getter]
    fn p2v_map(&self) -> HashMap<usize, usize> {
        self.inner
            .p2v_map()
            .iter()
            .map(|(k, v)| (qubit_to_py_id(*k), qubit_to_py_id(*v)))
            .collect()
    }

    fn swap_physical(&mut self, phys_a: usize, phys_b: usize) -> PyResult<()> {
        let phys_a = py_id_to_qubit(phys_a)?;
        let phys_b = py_id_to_qubit(phys_b)?;
        let physical_set: HashSet<Qubit> = self.inner.physical_qubits().collect();

        if !physical_set.contains(&phys_a) {
            return Err(PyValueError::new_err(format!(
                "physical qubit {} not in layout",
                qubit_to_py_id(phys_a)
            )));
        }
        if !physical_set.contains(&phys_b) {
            return Err(PyValueError::new_err(format!(
                "physical qubit {} not in layout",
                qubit_to_py_id(phys_b)
            )));
        }

        self.inner.swap_physical(phys_a, phys_b);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Layout(num_logical={}, num_ancilla={}, num_physical={})",
            self.num_logical(),
            self.num_ancilla(),
            self.num_physical()
        )
    }
}
