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

//! Unified Quantum Instruction Type
//!
//! This module defines the [`Instruction`] enum, which serves as the universal wrapper for all
//! operations that can be placed in a quantum circuit. It unifies:
//! - **Standard Gates**: Highly optimized, commonly used gates (e.g., H, CX).
//! - **Extended Gates**: Complex gates like multi-controlled operations or custom unitaries.
//! - **Non-Unitary Operations**: Measurement, Barrier, Reset.
//!
//! By using `Instruction`, the circuit can store a heterogeneous list of operations in a single vector.

use crate::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use crate::circuit::gate::directive::Directive;
use crate::circuit::gate::standard_gate::StandardGate;
use crate::circuit::gate::{MCGate, UnitaryGate, gate_matrix};
use crate::circuit::{Circuit, Parameter};
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// A unified representation of any operation in a quantum circuit.
///
/// This enum acts as a sum type for all possible instructions.
/// - Prefer [`Instruction::Standard`] for common gates to leverage simulator optimizations.
/// - Use [`Instruction::Extended`] for generalized controls or custom matrices.
/// - Use [`Instruction::Operation`] for non-reversible actions (measurement, reset).
#[derive(Debug, Clone)]
pub enum Instruction {
    /// A standard, natively supported quantum gate (e.g., `H`, `CX`).
    Standard(StandardGate),
    /// An extended gate, such as a multi-controlled gate or a user-defined unitary.
    McGate(Box<MCGate>),
    UnitaryGate(Box<UnitaryGate>),
    CircuitGate(Box<CircuitGate>),
    /// A non-unitary operation, such as `Measure`, `Barrier`, or `Reset`.
    Directive(Directive),
}

impl Instruction {
    /// Computes the unitary matrix representation of the instruction.
    ///
    /// # Arguments
    ///
    /// * `params` - A slice of floating-point parameters (e.g., rotation angles).
    ///
    /// # Returns
    ///
    /// - `Some(Cow<Array2>)`: The unitary matrix for `Standard` and `Extended` gates.
    /// - `None`: If the instruction is non-unitary (e.g., `Measure`, `Barrier`, `Reset`).
    pub fn matrix(&self, params: &[f64]) -> Option<Cow<'_, Array2<Complex64>>> {
        match self {
            Instruction::Standard(g) => Some(g.matrix(params)),
            Instruction::McGate(g) => Some(g.matrix(params)),
            Instruction::UnitaryGate(g) => g.matrix().map(Cow::Borrowed),
            Instruction::CircuitGate(_) => todo!(),
            Instruction::Directive(_) => None,
        }
    }

    /// Computes the inverse (Hermitian conjugate) of the instruction.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the instruction instance.
    ///
    /// # Returns
    ///
    /// - `Some(...)`: The inverse instruction and its transformed parameters.
    /// - `None`: If the instruction is non-invertible (e.g., `Measure`) or if the inverse
    ///   cannot be symbolically determined (e.g., some custom unitaries).
    pub fn inverse(&self, params: &[Parameter]) -> Option<(Instruction, SmallVec<[Parameter; 3]>)> {
        match self {
            Instruction::Standard(g) => {
                if let Some((gate, ps)) = g.inverse(params) {
                    Some((Self::Standard(gate), ps))
                } else {
                    None
                }
            }
            Instruction::McGate(g) => {
                if let Some((gate, ps)) = g.inverse(params) {
                    Some((Self::McGate(Box::new(gate)), ps))
                } else {
                    None
                }
            }
            Instruction::UnitaryGate(g) => {
                // Try to invert via circuit representation first
                if let Some(c) = g.circuit().as_ref() {
                    // Invert the internal circuit
                    if let Ok(c_inv) = c.circuit.inverse() {
                        // Create frozen circuit from inverted circuit
                        let frozen_inv = FrozenCircuit { circuit: c_inv };
                        // Create new UnitaryGate with inverted circuit
                        let u_inv =
                            UnitaryGate::new(format!("{}_dg", g.label()).as_str(), g.num_qubits())
                                .with_circuit(Arc::new(frozen_inv));
                        return Some((Self::UnitaryGate(Box::new(u_inv)), SmallVec::new()));
                    }
                }
                None
            }
            Instruction::CircuitGate(circuit_gate) => {
                // if let Ok(inv_gate) = circuit_gate.inverse() {
                //     Some((
                //         Instruction::Circuit(Box::new(inv_gate)),
                //         params.iter().cloned().collect(),
                //     ))
                // } else {
                //     None
                // }
                todo!()
            }
            Instruction::Directive(d) => match d {
                Directive::Barrier => Some((Self::Directive(Directive::Barrier), SmallVec::new())),
                _ => None,
            },
        }
    }

    /// Returns a new instruction that applies the current operation conditioned on extra control qubits.
    ///
    /// This method employs a **canonicalization strategy** to return the most optimized gate representation:
    ///
    /// 1. **Promotion**: If adding controls to a `StandardGate` results in another `StandardGate`
    ///    (e.g., $X \xrightarrow{+1} CX \xrightarrow{+1} CCX$), it returns the upgraded standard gate.
    /// 2. **Fallback**: If no standard equivalent exists (e.g., $C^3-X$), it returns an [`ExtendedGate::MCGate`].
    /// 3. **Aggregation**: If the instruction is already an `MCGate`, the new controls are merged into it.
    ///
    /// # Arguments
    ///
    /// * `num_new_ctrls` - The number of additional control qubits to add.
    ///
    /// # Returns
    ///
    /// - `Some(Instruction)`: The new controlled instruction.
    /// - `None`: If the instruction cannot be controlled (e.g., `Barrier`, `Measure`).
    pub fn control(&self, num_new_ctrls: usize) -> Option<Instruction> {
        if num_new_ctrls == 0 {
            return Some(self.clone());
        }

        // Internal Helper: Decompose a StandardGate into its base gate and current control count.
        // e.g., CX -> (X, 1), CRX -> (RX, 1)
        let decompose_std = |g: StandardGate| -> (StandardGate, usize) {
            match g {
                StandardGate::CX => (StandardGate::X, 1),
                StandardGate::CCX => (StandardGate::X, 2),
                StandardGate::CY => (StandardGate::Y, 1),
                StandardGate::CZ => (StandardGate::Z, 1),
                StandardGate::CRX => (StandardGate::RX, 1),
                StandardGate::CRY => (StandardGate::RY, 1),
                StandardGate::CRZ => (StandardGate::RZ, 1),
                _ => (g, 0),
            }
        };

        // Internal Helper: Try to recompose a base gate and total control count into a StandardGate.
        let try_compose_std = |base: StandardGate, total: usize| -> Option<StandardGate> {
            match (base, total) {
                (StandardGate::X, 1) => Some(StandardGate::CX),
                (StandardGate::X, 2) => Some(StandardGate::CCX),
                (StandardGate::Y, 1) => Some(StandardGate::CY),
                (StandardGate::Z, 1) => Some(StandardGate::CZ),
                (StandardGate::RX, 1) => Some(StandardGate::CRX),
                (StandardGate::RY, 1) => Some(StandardGate::CRY),
                (StandardGate::RZ, 1) => Some(StandardGate::CRZ),
                (g, 0) => Some(g),
                _ => None,
            }
        };

        match self {
            Instruction::Standard(g) => {
                let (base, curr_ctrls) = decompose_std(*g);
                let total_ctrls = curr_ctrls + num_new_ctrls;

                if let Some(std) = try_compose_std(base, total_ctrls) {
                    Some(Instruction::Standard(std))
                } else {
                    Some(Instruction::McGate(Box::from(MCGate::new(
                        total_ctrls as u8,
                        base,
                    ))))
                }
            }
            Instruction::McGate(mc) => {
                let total_ctrls = mc.num_qubits() + num_new_ctrls;
                let base = mc.base_gate().to_owned();

                if let Some(std) = try_compose_std(base, total_ctrls) {
                    Some(Instruction::Standard(std))
                } else {
                    Some(Instruction::McGate(Box::from(MCGate::new(
                        total_ctrls as u8,
                        base,
                    ))))
                }
            }
            Instruction::UnitaryGate(uni) => {
                let mut g = UnitaryGate::new(
                    format!("ctl_{}_{}", num_new_ctrls, uni.label()).as_str(),
                    uni.num_qubits() + num_new_ctrls as u16,
                );
                if let Some(m) = g.matrix() {
                    let controlled = gate_matrix::control_matrix(m, num_new_ctrls);
                    g = g.with_matrix(controlled).unwrap();
                }

                Some(Instruction::UnitaryGate(Box::from(g)))
            }
            Instruction::CircuitGate(_) => todo!(),
            Instruction::Directive(_) => None,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Standard(g) => write!(f, "{}", g),
            Instruction::McGate(g) => write!(f, "{}", g),
            Instruction::UnitaryGate(g) => write!(f, "{}", g),
            Instruction::CircuitGate(_) => todo!(),
            Instruction::Directive(i) => write!(f, "{}", i),
        }
    }
}

impl From<StandardGate> for Instruction {
    fn from(g: StandardGate) -> Self {
        Self::Standard(g)
    }
}

impl From<Directive> for Instruction {
    fn from(d: Directive) -> Self {
        Self::Directive(d)
    }
}
