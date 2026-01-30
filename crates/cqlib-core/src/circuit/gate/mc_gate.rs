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

use crate::circuit::Parameter;
use crate::circuit::gate::{StandardGate, gate_matrix};
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex;
use smallvec::SmallVec;
use std::fmt;

#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub struct MCGate {
    num_controls: u8,
    gate: StandardGate,
}

impl MCGate {
    pub fn new(num_controls: u8, gate: StandardGate) -> Self {
        Self { num_controls, gate }
    }

    pub fn matrix(&self, params: &[f64]) -> Cow<'_, Array2<Complex<f64>>> {
        let base_matrix = self.gate.matrix(params);
        if self.num_controls == 0 {
            return base_matrix;
        }
        // Construct controlled matrix
        let controlled = gate_matrix::control_matrix(&base_matrix, self.num_controls as usize);
        Cow::Owned(controlled)
    }

    pub fn inverse(&self, params: &[Parameter]) -> Option<(MCGate, SmallVec<[Parameter; 3]>)> {
        // The inverse of a controlled gate C(U) is C(U†).
        let (inv_gate, inv_params) = self.gate.inverse(params)?;
        Some((
            Self {
                num_controls: self.num_controls,
                gate: inv_gate,
            },
            inv_params,
        ))
    }

    /// Returns the number of control qubits.
    pub fn num_ctrl_qubits(&self) -> usize {
        self.num_controls as usize + self.gate.num_ctrl_qubits()
    }

    /// Returns the total number of qubits (controls + targets).
    pub fn num_qubits(&self) -> usize {
        self.num_controls as usize + self.gate.num_qubits()
    }

    /// Returns the number of parameters required by the gate.
    pub fn num_params(&self) -> usize {
        self.gate.num_params()
    }

    pub fn base_gate(&self) -> &StandardGate {
        &self.gate
    }
}

impl fmt::Display for MCGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.num_controls == 0 {
            write!(f, "{}", self.gate)
        } else {
            write!(f, "C{}-{}", self.num_controls, self.gate)
        }
    }
}
