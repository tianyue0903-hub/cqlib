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

use cqlib_core::circuit::gate::{Directive, Instruction};
use cqlib_core::circuit::Parameter;
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[pyclass(name = "Instruction", module = "cqlib.circuit.gates")]
#[derive(Clone, Debug)]
pub struct PyInstruction {
    pub inner: Instruction,
}

#[pymethods]
impl PyInstruction {
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __eq__(&self, other: &PyInstruction) -> bool {
        // Instruction in core doesn't derive PartialEq automatically for all variants in a simple way
        // that matches Python's expectation strictly (e.g. Box<ExtendedGate> pointer vs value).
        // For now, we format debug strings or rely on specific matching if implemented in core.
        // Let's assume for now strict equality is hard without core support, 
        // but for StandardGate variants it works.
        // A simple workaround for this binding:
        format!("{:?}", self.inner) == format!("{:?}", other.inner)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.inner).hash(&mut hasher);
        hasher.finish()
    }

    // --- Properties ---

    #[getter]
    fn num_qubits(&self) -> usize {
        match &self.inner {
            Instruction::Standard(g) => g.num_qubits(),
            Instruction::Extended(g) => g.num_qubits(),
            Instruction::Circuit(g) => g.num_qubits(),
            Instruction::Directive(d) => match d {
                Directive::Measure => 1,
                Directive::Reset => 1,
                Directive::Barrier => 0, // Represents variable/undefined
            },
        }
    }

    #[getter]
    fn num_ctrl_qubits(&self) -> usize {
        match &self.inner {
            Instruction::Standard(g) => g.num_ctrl_qubits(),
            Instruction::Extended(g) => g.num_ctrl_qubits(),
            Instruction::Circuit(_) => 0,
            Instruction::Directive(_) => 0,
        }
    }

    #[getter]
    fn num_params(&self) -> usize {
        match &self.inner {
            Instruction::Standard(g) => g.num_params(),
            Instruction::Extended(g) => g.num_params(),
            Instruction::Circuit(g) => g.num_params(),
            Instruction::Directive(_) => 0,
        }
    }

    /// Returns the unitary matrix of the instruction.
    ///
    /// Args:
    ///     params (List[float], optional): Parameters for the instruction.
    #[pyo3(signature = (params=None))]
    fn matrix(&self, params: Option<Vec<f64>>) -> PyResult<Vec<Vec<Complex64>>> {
        let p = params.unwrap_or_default();
        // Note: Instruction::matrix signature might not check param length strictly in all core impls,
        // but we pass it through.

        match self.inner.matrix(&p) {
            Some(mat_cow) => {
                let mat = mat_cow.view();
                let mut result = Vec::with_capacity(mat.nrows());
                for row in mat.rows() {
                    result.push(row.to_vec());
                }
                Ok(result)
            }
            None => Err(PyValueError::new_err(format!(
                "Instruction {} has no matrix representation (it might be a directive like Measure)",
                self.inner
            ))),
        }
    }

    /// Returns a controlled version of this instruction.
    fn control(&self, num_ctrls: usize) -> PyResult<PyInstruction> {
        match self.inner.control(num_ctrls) {
            Some(controlled) => Ok(PyInstruction { inner: controlled }),
            None => Err(PyValueError::new_err(format!(
                "Cannot control instruction {}",
                self.inner
            ))),
        }
    }

    /// Returns the inverse instruction type.
    ///
    /// Note: This returns the *instruction* that represents the inverse.
    /// For parametric gates, this returns the generic inverse type without specific parameter values bound.
    fn inverse(&self) -> PyResult<PyInstruction> {
        // We use dummy parameters (all zeros) to determine the inverse structure.
        // This is a limitation of the stateless Instruction object in Python.
        // Ideally, we'd need to know the number of params.
        // We can try with 0, 1, 2, 3 parameters and see which one succeeds if we don't know.
        // Or better, check the underlying gate type if possible.

        // Strategy: Try with 3 zeros (covers most standard gates U, etc).
        // Extra params usually ignored by simple gates.
        let dummy_params: Vec<Parameter> = vec![Parameter::from(0.0); 3];

        if let Some((inv_inst, _)) = self.inner.inverse(&dummy_params) {
            Ok(PyInstruction { inner: inv_inst })
        } else {
            Err(PyValueError::new_err(format!(
                "Instruction {} is not invertible",
                self.inner
            )))
        }
    }
}

impl From<Instruction> for PyInstruction {
    fn from(instruction: Instruction) -> Self {
        Self { inner: instruction }
    }
}
