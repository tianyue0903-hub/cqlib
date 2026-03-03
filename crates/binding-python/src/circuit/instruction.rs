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

use cqlib_core::circuit::gate::instruction::Instruction;
use pyo3::prelude::*;

use crate::circuit::{
    PyControlFlow, PyDelay, PyDirective, PyMcGate, PyStandardGate, PyUnitaryGate,
};

#[pyclass(name = "Instruction", module = "cqlib.circuit")]
#[derive(Debug, Clone)]
pub struct PyInstruction {
    pub inner: Instruction,
}

impl From<Instruction> for PyInstruction {
    fn from(inner: Instruction) -> Self {
        Self { inner }
    }
}

impl From<PyInstruction> for Instruction {
    fn from(py: PyInstruction) -> Self {
        py.inner
    }
}

#[pymethods]
impl PyInstruction {
    /// Creates an instruction from a standard gate.
    ///
    /// Args:
    ///     gate: The standard gate.
    #[staticmethod]
    fn from_standard_gate(gate: PyStandardGate) -> Self {
        PyInstruction {
            inner: Instruction::Standard(gate.inner),
        }
    }

    /// Creates an instruction from a multi-controlled gate.
    ///
    /// Args:
    ///     gate: The multi-controlled gate.
    #[staticmethod]
    fn from_mc_gate(gate: PyMcGate) -> Self {
        PyInstruction {
            inner: Instruction::McGate(Box::new(gate.inner)),
        }
    }

    /// Creates an instruction from a unitary gate.
    ///
    /// Args:
    ///     gate: The unitary gate.
    #[staticmethod]
    fn from_unitary_gate(gate: PyUnitaryGate) -> Self {
        PyInstruction {
            inner: Instruction::UnitaryGate(Box::new(gate.into())),
        }
    }

    /// Creates a directive instruction (barrier, measure, reset).
    ///
    /// Args:
    ///     directive: The directive.
    #[staticmethod]
    fn from_directive(directive: PyDirective) -> Self {
        PyInstruction {
            inner: Instruction::Directive(directive.inner),
        }
    }

    /// Creates a delay instruction.
    ///
    /// Args:
    ///     delay: The delay operation.
    #[staticmethod]
    fn from_delay(_delay: PyDelay) -> Self {
        PyInstruction {
            inner: Instruction::Delay,
        }
    }

    /// Creates a control flow instruction.
    ///
    /// Args:
    ///     control_flow: The control flow operation.
    #[staticmethod]
    fn from_control_flow(control_flow: PyControlFlow) -> Self {
        PyInstruction {
            inner: Instruction::ControlFlowGate(control_flow.inner),
        }
    }

    /// Returns the name of the instruction.
    #[getter]
    fn name(&self) -> String {
        match &self.inner {
            Instruction::Standard(g) => format!("{}", g),
            Instruction::McGate(g) => format!("{}", g),
            Instruction::UnitaryGate(g) => g.label().to_string(),
            Instruction::CircuitGate(g) => g.name.to_string(),
            Instruction::Directive(d) => format!("{}", d),
            Instruction::ControlFlowGate(g) => format!("{}", g),
            Instruction::Delay => "Delay".to_string(),
        }
    }

    /// Returns the type of the instruction as a string.
    #[getter]
    fn instruction_type(&self) -> String {
        match &self.inner {
            Instruction::Standard(_) => "standard".to_string(),
            Instruction::McGate(_) => "mcgate".to_string(),
            Instruction::UnitaryGate(_) => "unitary".to_string(),
            Instruction::CircuitGate(_) => "circuit".to_string(),
            Instruction::Directive(_) => "directive".to_string(),
            Instruction::Delay => "delay".to_string(),
            Instruction::ControlFlowGate(_) => "control_flow".to_string(),
        }
    }

    /// Returns true if the instruction is a standard gate.
    #[getter]
    fn is_standard(&self) -> bool {
        matches!(self.inner, Instruction::Standard(_))
    }

    /// Returns true if the instruction is a multi-controlled gate.
    #[getter]
    fn is_mcgate(&self) -> bool {
        matches!(self.inner, Instruction::McGate(_))
    }

    /// Returns true if the instruction is a unitary gate.
    #[getter]
    fn is_unitary(&self) -> bool {
        matches!(self.inner, Instruction::UnitaryGate(_))
    }

    /// Returns true if the instruction is a circuit gate.
    #[getter]
    fn is_circuit(&self) -> bool {
        matches!(self.inner, Instruction::CircuitGate(_))
    }

    /// Returns true if the instruction is a directive (measure, barrier, reset).
    #[getter]
    fn is_directive(&self) -> bool {
        matches!(self.inner, Instruction::Directive(_))
    }

    /// Returns the standard gate if this is a standard instruction, None otherwise.
    #[getter]
    fn standard_gate(&self) -> Option<PyStandardGate> {
        match &self.inner {
            Instruction::Standard(g) => Some(PyStandardGate::from(*g, vec![])),
            _ => None,
        }
    }

    /// Returns the multi-controlled gate if this is an mc instruction, None otherwise.
    #[getter]
    fn mc_gate(&self) -> Option<PyMcGate> {
        match &self.inner {
            Instruction::McGate(g) => Some(PyMcGate::new(
                g.num_ctrl_qubits() as u8,
                PyStandardGate::from(*g.base_gate(), vec![]),
            )),
            _ => None,
        }
    }

    /// Returns the unitary gate if this is a unitary instruction, None otherwise.
    #[getter]
    fn unitary_gate(&self) -> Option<PyUnitaryGate> {
        match &self.inner {
            Instruction::UnitaryGate(g) => Some(PyUnitaryGate::from(g.as_ref().clone())),
            _ => None,
        }
    }

    /// Returns true if this is a delay instruction.
    #[getter]
    fn is_delay(&self) -> bool {
        matches!(self.inner, Instruction::Delay)
    }

    /// Returns true if this is a control flow instruction.
    #[getter]
    fn is_control_flow(&self) -> bool {
        matches!(self.inner, Instruction::ControlFlowGate(_))
    }

    /// Returns the directive if this is a directive instruction, None otherwise.
    #[getter]
    fn directive(&self) -> Option<PyDirective> {
        match &self.inner {
            Instruction::Directive(d) => Some(PyDirective::from(*d)),
            _ => None,
        }
    }

    /// Returns the control flow if this is a control flow instruction, None otherwise.
    fn control_flow(&self) -> Option<PyControlFlow> {
        match &self.inner {
            Instruction::ControlFlowGate(cf) => Some(PyControlFlow::from(cf.clone())),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Instruction({})", self.name())
    }

    fn __str__(&self) -> String {
        self.name()
    }
}
