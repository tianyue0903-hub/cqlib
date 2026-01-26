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
use crate::circuit::error::CircuitError;
use crate::circuit::parameter::impls::Parameter;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FrozenCircuit {
    pub(crate) circuit: Circuit,
}

#[derive(Debug, Clone)]
pub struct CircuitGate {
    pub name: Arc<String>,
    pub(crate) num_qubits: usize,
    pub(crate) num_params: usize,
    pub(crate) params: Vec<Parameter>,
    pub(crate) circuit: Arc<FrozenCircuit>,
}

impl CircuitGate {
    pub fn new(
        name: impl Into<String>,
        circuit: FrozenCircuit,
        params: Vec<Parameter>,
    ) -> Result<Self, CircuitError> {
        let num_qubits = circuit.circuit.qubits().len();
        let num_params = circuit.circuit.symbols().len();

        if params.len() != num_params {
            return Err(CircuitError::ParameterCountMismatch {
                expected: num_params,
                actual: params.len(),
            });
        }

        Ok(Self {
            name: Arc::new(name.into()),
            num_qubits,
            num_params,
            params,
            circuit: Arc::new(circuit),
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn circuit(&self) -> Arc<FrozenCircuit> {
        self.circuit.clone()
    }

    pub fn inverse(&self) -> Result<Self, CircuitError> {
        let inverted_circuit = self.circuit.circuit.inverse()?;
        let frozen_inverted = FrozenCircuit {
            circuit: inverted_circuit,
        };
        CircuitGate::new(
            format!("{}_dg", self.name),
            frozen_inverted,
            self.params.clone(),
        )
    }
}
