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

use super::instruction::PyInstruction;
use cqlib_core::circuit::Parameter;
use cqlib_core::circuit::gate::{Instruction, StandardGate};
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "StandardGate", module = "cqlib.circuit.gates")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyStandardGate {
    pub inner: StandardGate,
}

#[pymethods]
impl PyStandardGate {
    // --- Static Attributes (The Gates) ---
    #[classattr]
    const I: PyStandardGate = PyStandardGate {
        inner: StandardGate::I,
    };
    #[classattr]
    const H: PyStandardGate = PyStandardGate {
        inner: StandardGate::H,
    };
    #[classattr]
    const X: PyStandardGate = PyStandardGate {
        inner: StandardGate::X,
    };
    #[classattr]
    const Y: PyStandardGate = PyStandardGate {
        inner: StandardGate::Y,
    };
    #[classattr]
    const Z: PyStandardGate = PyStandardGate {
        inner: StandardGate::Z,
    };
    #[classattr]
    const S: PyStandardGate = PyStandardGate {
        inner: StandardGate::S,
    };
    #[classattr]
    const SDG: PyStandardGate = PyStandardGate {
        inner: StandardGate::SDG,
    };
    #[classattr]
    const T: PyStandardGate = PyStandardGate {
        inner: StandardGate::T,
    };
    #[classattr]
    const TDG: PyStandardGate = PyStandardGate {
        inner: StandardGate::TDG,
    };

    #[classattr]
    const RX: PyStandardGate = PyStandardGate {
        inner: StandardGate::RX,
    };
    #[classattr]
    const RY: PyStandardGate = PyStandardGate {
        inner: StandardGate::RY,
    };
    #[classattr]
    const RZ: PyStandardGate = PyStandardGate {
        inner: StandardGate::RZ,
    };
    #[classattr]
    const U: PyStandardGate = PyStandardGate {
        inner: StandardGate::U,
    };

    #[classattr]
    #[pyo3(name = "Phase")]
    const PHASE: PyStandardGate = PyStandardGate {
        inner: StandardGate::Phase,
    };
    #[classattr]
    #[pyo3(name = "GPhase")]
    const GPHASE: PyStandardGate = PyStandardGate {
        inner: StandardGate::GPhase,
    };

    #[classattr]
    const RXX: PyStandardGate = PyStandardGate {
        inner: StandardGate::RXX,
    };
    #[classattr]
    const RXY: PyStandardGate = PyStandardGate {
        inner: StandardGate::RXY,
    };
    #[classattr]
    const RYY: PyStandardGate = PyStandardGate {
        inner: StandardGate::RYY,
    };
    #[classattr]
    const RZX: PyStandardGate = PyStandardGate {
        inner: StandardGate::RZX,
    };
    #[classattr]
    const RZZ: PyStandardGate = PyStandardGate {
        inner: StandardGate::RZZ,
    };

    #[classattr]
    const CX: PyStandardGate = PyStandardGate {
        inner: StandardGate::CX,
    };
    #[classattr]
    const CY: PyStandardGate = PyStandardGate {
        inner: StandardGate::CY,
    };
    #[classattr]
    const CZ: PyStandardGate = PyStandardGate {
        inner: StandardGate::CZ,
    };
    #[classattr]
    const CCX: PyStandardGate = PyStandardGate {
        inner: StandardGate::CCX,
    };
    #[classattr]
    const SWAP: PyStandardGate = PyStandardGate {
        inner: StandardGate::SWAP,
    };

    #[classattr]
    const CRX: PyStandardGate = PyStandardGate {
        inner: StandardGate::CRX,
    };
    #[classattr]
    const CRY: PyStandardGate = PyStandardGate {
        inner: StandardGate::CRY,
    };
    #[classattr]
    const CRZ: PyStandardGate = PyStandardGate {
        inner: StandardGate::CRZ,
    };

    #[classattr]
    const XY: PyStandardGate = PyStandardGate {
        inner: StandardGate::XY,
    };
    #[classattr]
    const X2P: PyStandardGate = PyStandardGate {
        inner: StandardGate::X2P,
    };
    #[classattr]
    const X2M: PyStandardGate = PyStandardGate {
        inner: StandardGate::X2M,
    };
    #[classattr]
    const XY2P: PyStandardGate = PyStandardGate {
        inner: StandardGate::XY2P,
    };
    #[classattr]
    const XY2M: PyStandardGate = PyStandardGate {
        inner: StandardGate::XY2M,
    };
    #[classattr]
    const Y2P: PyStandardGate = PyStandardGate {
        inner: StandardGate::Y2P,
    };
    #[classattr]
    const Y2M: PyStandardGate = PyStandardGate {
        inner: StandardGate::Y2M,
    };

    #[classattr]
    const FSIM: PyStandardGate = PyStandardGate {
        inner: StandardGate::FSIM,
    };

    // --- Properties ---

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_ctrl_qubits(&self) -> usize {
        self.inner.num_ctrl_qubits()
    }

    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __eq__(&self, other: &PyStandardGate) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    // --- Methods ---

    /// Returns the unitary matrix of the gate.
    ///
    /// Args:
    ///     params (List[float], optional): Parameters for the gate (if any).
    #[pyo3(signature = (params=None))]
    fn matrix(&self, params: Option<Vec<f64>>) -> PyResult<Vec<Vec<Complex64>>> {
        let p = params.unwrap_or_default();
        if p.len() != self.inner.num_params() {
            return Err(PyValueError::new_err(format!(
                "Gate {:?} expects {} parameters, got {}",
                self.inner,
                self.inner.num_params(),
                p.len()
            )));
        }

        let mat_cow = self.inner.matrix(&p);
        let mat = mat_cow.view();
        let mut result = Vec::with_capacity(mat.nrows());
        for row in mat.rows() {
            result.push(row.to_vec());
        }
        Ok(result)
    }

    /// Returns a controlled version of this gate.
    ///
    /// Args:
    ///     num_ctrls (int): Number of control qubits to add.
    ///
    /// Returns:
    ///     Instruction: The controlled instruction.
    fn control(&self, num_ctrls: usize) -> PyResult<PyInstruction> {
        let inst: Instruction = self.inner.into();
        match inst.control(num_ctrls) {
            Some(controlled_inst) => Ok(PyInstruction::from(controlled_inst)),
            None => Err(PyValueError::new_err(format!(
                "Cannot control gate {:?}",
                self.inner
            ))),
        }
    }

    /// Returns the type of the inverse gate.
    ///
    /// Note: This returns the *gate type* that represents the inverse.
    /// For parametric gates like RX, the inverse type is still RX (but with -theta).
    /// For S, the inverse type is SDG.
    fn inverse(&self) -> PyStandardGate {
        // Create dummy parameters to satisfy the API.
        // The values don't matter for determining the *type* of the inverse gate in most cases.
        let dummy_params: Vec<Parameter> = (0..self.inner.num_params())
            .map(|_| Parameter::from(0.0))
            .collect();

        if let Some((inv_gate, _)) = self.inner.inverse(&dummy_params) {
            PyStandardGate { inner: inv_gate }
        } else {
            // Should not happen for standard gates usually, but as a fallback return self
            *self
        }
    }
}
