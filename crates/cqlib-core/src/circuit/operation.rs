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

use crate::circuit::bit::Qubit;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::param::CircuitParam;
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone)]
pub struct Operation {
    pub instruction: Instruction,
    pub qubits: SmallVec<[Qubit; 3]>,
    pub params: SmallVec<[CircuitParam; 1]>,
    pub label: Option<Box<str>>,
}

impl Operation {
    pub fn matrix(&self) -> Result<Cow<'_, Array2<Complex64>>, CircuitError> {
        let mut ps: SmallVec<[f64; 4]> = smallvec![];
        for p in self.params.iter() {
            match p {
                CircuitParam::Fixed(val) => {
                    ps.push(*val);
                }
                CircuitParam::Index(index) => {
                    todo!()
                }
            }
        }
        self.instruction
            .matrix(&ps)
            .ok_or(CircuitError::NoMatrixRepresentation)
    }
}
