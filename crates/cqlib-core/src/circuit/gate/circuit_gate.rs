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

use crate::circuit::circuit::Circuit;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::parameter::impls::Parameter;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FrozenCircuit {
    instructions: Arc<[Instruction]>,
    num_qubits: usize,
    num_params: usize,
    params: Vec<Parameter>,
}

#[derive(Debug, Clone)]
pub struct CircuitGate {
    pub name: Arc<String>,
    num_qubits: usize,
    num_params: usize,
    params: Vec<Parameter>,
    pub(crate) circuit: Arc<FrozenCircuit>,
}

impl CircuitGate {
    pub fn circuit(&self) -> Arc<FrozenCircuit> {
        self.circuit.clone()
    }
}
