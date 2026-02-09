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

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use indexmap::IndexSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct FrozenCircuit {
    pub(crate) circuit: Circuit,
}

impl FrozenCircuit {
    pub fn new(circuit: Circuit) -> Self {
        Self { circuit }
    }

    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
}

/// A composite gate defined by a quantum circuit.
///
/// `CircuitGate` encapsulates a frozen circuit as a reusable gate operation.
/// When used in a circuit, its parameters are mapped positionally to the
/// symbolic parameters of the inner circuit.
///
/// ### Parameter Resolution Logic
///
/// It is important to distinguish between the symbols defined inside the `CircuitGate`
/// and the arguments passed to it from the outside.
///
/// **Example Flow:**
///
/// 1. **Internal Expression**: The inner circuit contains a gate with a parameter expression, e.g., `theta + 1`.
/// 2. **Binding**: The `CircuitGate` usage defines a mapping from an external argument to the internal symbol, e.g., `theta` $\leftarrow$ `2 * x`.
/// 3. **Evaluation**:
///    If the external argument `x` is `0.5`:
///    - First, the binding is resolved: `theta` becomes `2 * 0.5 = 1.0`.
///    - Then, the internal circuit evaluates its expression using this value: `1.0 + 1 = 2.0`.
#[derive(Debug, Clone)]
pub struct CircuitGate {
    pub name: Arc<String>,
    pub(crate) num_qubits: usize,
    pub(crate) num_params: usize,
    pub(crate) circuit: Arc<FrozenCircuit>,
}

impl CircuitGate {
    pub fn new(name: impl Into<String>, circuit: FrozenCircuit) -> Result<Self, CircuitError> {
        let num_qubits = circuit.circuit.qubits().len();
        let num_params = circuit.circuit.symbols().len();

        Ok(Self {
            name: Arc::new(name.into()),
            num_qubits,
            num_params,
            circuit: Arc::new(circuit),
        })
    }

    pub fn symbols(&self) -> IndexSet<String> {
        self.circuit.circuit.symbols().clone()
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
        CircuitGate::new(format!("{}_dg", self.name), frozen_inverted)
    }
}
