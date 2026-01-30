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

/// 线路的符号，和线路门的符号不等价
///  1. CircuitGate 内部有一个参数表达式：theta + 1
///  2. CircuitGate 定义了一个绑定：theta -> 2 * theta（外部参数映射到内部符号）
///  3. 当外部传入 theta = 0.5 时：
///    • 首先计算 CircuitGate 的绑定：2 * 0.5 = 1.0
///    • 然后用 1.0 替换内部电路的 theta
///    • 最后计算内部电路的参数：1.0 + 1 = 2.0
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
