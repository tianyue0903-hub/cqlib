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

use super::bit::PyQubit;
use super::gates::{PyCircuitGate, PyMcGate, PyStandardGate, PyUnitaryGate};
use super::parameter::PyParameter;
use crate::circuit::PyOperation;
use crate::circuit::operation::PyOperationIter;
use cqlib_core::circuit::gate::Instruction;
use cqlib_core::circuit::param::ParameterValue;
use cqlib_core::circuit::{Circuit, Qubit};
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};

/// A helper enum to accept either a float or a Parameter object from Python.
#[derive(FromPyObject)]
pub enum PyParamLike {
    Float(f64),
    Param(PyParameter),
}

impl From<PyParamLike> for ParameterValue {
    fn from(val: PyParamLike) -> Self {
        match val {
            PyParamLike::Float(f) => ParameterValue::Fixed(f),
            PyParamLike::Param(p) => ParameterValue::Param(p.into_inner()),
        }
    }
}

#[pyclass(name = "Circuit", module = "cqlib.circuit")]
#[derive(Debug, Clone)]
pub struct PyCircuit {
    pub inner: Circuit,
}

impl From<Circuit> for PyCircuit {
    fn from(circuit: Circuit) -> Self {
        PyCircuit { inner: circuit }
    }
}

#[derive(FromPyObject)]
pub enum PyIntQubitList {
    // 1. 单个整数 (比特数)
    NumQubits(usize),
    // 2. 整数列表 (比特索引列表)
    IndexList(Vec<usize>),
    // 3. Qubit 对象列表 (直接拷贝 PyQubit，因为它是 Copy 的)
    QubitList(Vec<PyQubit>),
}

#[pymethods]
impl PyCircuit {
    /// Creates a new quantum circuit.
    ///
    /// The circuit can be initialized in three ways:
    /// 1. By number of qubits: `Circuit(5)` creates a circuit with qubits 0-4.
    /// 2. By list of integers: `Circuit([0, 2, 4])` creates a circuit with specific qubits.
    /// 3. By list of Qubit objects: `Circuit([Qubit(0), Qubit(1)])`.
    ///
    /// Args:
    ///     qubits (Union[int, List[int], List[Qubit]]): The qubits to include in the circuit.
    #[new]
    fn new(qubits: PyIntQubitList) -> PyResult<Self> {
        match qubits {
            PyIntQubitList::NumQubits(num) => Ok(PyCircuit {
                inner: Circuit::new(num),
            }),
            PyIntQubitList::IndexList(indices) => {
                let core_qubits: Vec<Qubit> = indices
                    .into_iter()
                    .map(|idx| Qubit::new(idx as u32))
                    .collect();
                let inner = Circuit::from_qubits(core_qubits)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyCircuit { inner })
            }
            PyIntQubitList::QubitList(qubits) => {
                let core_qubits: Vec<Qubit> = qubits.into_iter().map(|q| q.inner).collect();
                let inner = Circuit::from_qubits(core_qubits)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyCircuit { inner })
            }
        }
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn qubits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.qubits().iter().map(|&q| PyQubit::from(q)))
    }

    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .parameters()
                .iter()
                .map(|p| PyParameter::from(p.clone())),
        )
    }

    /// Appends an instruction to the circuit.
    ///
    /// Args:
    ///     instruction (StandardGate): The gate to append.
    ///     qubits (List[int]): The list of qubit indices to apply the gate to.
    #[pyo3(signature = (instruction, qubits))]
    fn append(&mut self, instruction: PyStandardGate, qubits: Vec<usize>) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits
            .into_iter()
            .map(|idx| Qubit::new(idx as u32))
            .collect();

        let inst = Instruction::Standard(instruction.inner);
        let params_core: Vec<ParameterValue> = instruction
            .params
            .into_iter()
            .map(ParameterValue::Param)
            .collect();

        self.inner
            .append(inst, qubits_core, params_core, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    fn i(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .i(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .h(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .x(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .y(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .z(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .s(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .sdg(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .t(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .tdg(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn x2p(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .x2p(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn x2m(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .x2m(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn y2p(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .y2p(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn y2m(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .y2m(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // --- Parametric Single Qubit Gates ---

    fn rx(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rx(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn ry(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .ry(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rz(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rz(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn phase(&mut self, qubit: usize, lambda: PyParamLike) -> PyResult<()> {
        self.inner
            .phase(Qubit::new(qubit as u32), lambda)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn xy(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn xy2p(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy2p(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn xy2m(&mut self, qubit: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy2m(Qubit::new(qubit as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (qubit, theta, phi, lambda))]
    fn u(
        &mut self,
        qubit: usize,
        theta: PyParamLike,
        phi: PyParamLike,
        lambda: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .u(Qubit::new(qubit as u32), theta, phi, lambda)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rxy(&mut self, qubit: usize, theta: PyParamLike, phi: PyParamLike) -> PyResult<()> {
        self.inner
            .rxy(Qubit::new(qubit as u32), theta, phi)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // --- Two Qubit Gates ---

    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner
            .cx(Qubit::new(control as u32), Qubit::new(target as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner
            .cy(Qubit::new(control as u32), Qubit::new(target as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner
            .cz(Qubit::new(control as u32), Qubit::new(target as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn swap(&mut self, a: usize, b: usize) -> PyResult<()> {
        self.inner
            .swap(Qubit::new(a as u32), Qubit::new(b as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rxx(&mut self, a: usize, b: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rxx(Qubit::new(a as u32), Qubit::new(b as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn ryy(&mut self, a: usize, b: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .ryy(Qubit::new(a as u32), Qubit::new(b as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rzz(&mut self, a: usize, b: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rzz(Qubit::new(a as u32), Qubit::new(b as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn rzx(&mut self, a: usize, b: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rzx(Qubit::new(a as u32), Qubit::new(b as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn fsim(&mut self, a: usize, b: usize, theta: PyParamLike, phi: PyParamLike) -> PyResult<()> {
        self.inner
            .fsim(Qubit::new(a as u32), Qubit::new(b as u32), theta, phi)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // --- Controlled Rotations ---

    fn crx(&mut self, control: usize, target: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .crx(Qubit::new(control as u32), Qubit::new(target as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn cry(&mut self, control: usize, target: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .cry(Qubit::new(control as u32), Qubit::new(target as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn crz(&mut self, control: usize, target: usize, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .crz(Qubit::new(control as u32), Qubit::new(target as u32), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) -> PyResult<()> {
        self.inner
            .ccx(
                Qubit::new(control1 as u32),
                Qubit::new(control2 as u32),
                Qubit::new(target as u32),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (instruction, controls, targets, params=None))]
    pub fn multi_control(
        &mut self,
        instruction: PyStandardGate,
        controls: Vec<usize>,
        targets: Vec<usize>,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let mut ps = vec![];
        if let Some(params) = params {
            for p in params {
                match p {
                    PyParamLike::Float(f) => ps.push(ParameterValue::Fixed(f)),
                    PyParamLike::Param(p) => ps.push(ParameterValue::Param(p.into_inner())),
                }
            }
        }
        self.inner
            .multi_control(instruction.inner, controls, targets, ps)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (instruction, qubits, params=None))]
    pub fn multi_control_gate(
        &mut self,
        instruction: PyMcGate,
        qubits: Vec<usize>,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits.into_iter().map(Qubit::from).collect();
        let inst = Instruction::McGate(Box::new(instruction.inner));
        let params_core: Vec<ParameterValue> = params
            .unwrap_or_default()
            .into_iter()
            .map(ParameterValue::from)
            .collect();

        self.inner
            .append(inst, qubits_core, params_core, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a custom unitary gate to the circuit.
    ///
    /// Args:
    ///     gate (UnitaryGate): The custom unitary gate definition.
    ///     qubits (List[int]): The list of qubit indices to apply the gate to.
    fn unitary(&mut self, gate: PyUnitaryGate, qubits: Vec<usize>) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits.into_iter().map(|q| Qubit::new(q as u32)).collect();
        self.inner
            .unitary(gate.into(), qubits_core)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn measure(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .measure(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn reset(&mut self, qubit: usize) -> PyResult<()> {
        self.inner
            .reset(Qubit::new(qubit as u32))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn barrier(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits.into_iter().map(|q| Qubit::new(q as u32)).collect();
        self.inner
            .barrier(qubits_core)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (instruction, qubits, params=None))]
    fn circuit_gate(
        &mut self,
        instruction: PyCircuitGate,
        qubits: Vec<usize>,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits
            .into_iter()
            .map(|idx| Qubit::new(idx as u32))
            .collect();

        let inst = Instruction::CircuitGate(Box::new(instruction.inner));
        let params_core: Vec<ParameterValue> = params
            .unwrap_or_default()
            .into_iter()
            .map(|p| p.into())
            .collect();
        self.inner
            .append(inst, qubits_core, params_core, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    fn inverse(&self) -> PyResult<Self> {
        let new_inner = self
            .inner
            .inverse()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyCircuit { inner: new_inner })
    }

    #[getter]
    fn operations(&self) -> PyOperationIter {
        PyOperationIter::new(self.inner.operations().to_vec(), 0)
    }

    fn to_gate(&self, name: String) -> PyResult<PyCircuitGate> {
        let instruction = self
            .inner
            .clone()
            .to_gate(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Instruction::CircuitGate(gate) = instruction {
            Ok(PyCircuitGate { inner: *gate })
        } else {
            Err(PyValueError::new_err(
                "Unexpected instruction type returned from to_gate",
            ))
        }
    }

    #[pyo3(signature = (qubits_order=None))]
    fn to_matrix<'py>(
        &self,
        py: Python<'py>,
        qubits_order: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        Ok(self.inner.to_matrix(qubits_order.as_ref()).to_pyarray(py))
    }

    fn decompose(&self) -> Self {
        Self {
            inner: self.inner.decompose(),
        }
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        idx: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ops = self.inner.operations();
        let len = ops.len();

        // 处理单个整数索引
        if let Ok(index) = idx.extract::<isize>() {
            let idx = if index < 0 {
                let neg = len as isize + index;
                if neg < 0 {
                    return Err(PyIndexError::new_err(format!(
                        "Index {} out of range for circuit with {} operations",
                        index, len
                    )));
                }
                neg as usize
            } else {
                if index as usize >= len {
                    return Err(PyIndexError::new_err(format!(
                        "Index {} out of range for circuit with {} operations",
                        index, len
                    )));
                }
                index as usize
            };

            let op = PyOperation::from(ops[idx].clone());
            return op.into_bound_py_any(py);
        }

        if let Ok(slice) = idx.cast_into::<PySlice>() {
            let indices = slice.indices(len as isize)?;

            // indices 是 PySliceIndices 结构体，不是元组
            let mut result = Vec::with_capacity(indices.slicelength);
            let mut i = indices.start;

            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                result.push(PyOperation::from(ops[i as usize].clone()));
                i += indices.step;
            }

            return Ok(PyList::new(py, result)?.into_any());
        }

        Err(PyTypeError::new_err("Index must be integer or slice"))
    }

    fn __len__(&self) -> usize {
        self.inner.operations().len()
    }
}
