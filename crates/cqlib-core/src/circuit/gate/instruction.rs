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
//! - **Control Flow**: Conditional and iterative operations (If-Else, While-Loop).
//!
//! By using `Instruction`, the circuit can store a heterogeneous list of operations in a single vector.

use crate::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use crate::circuit::gate::control_flow::ControlFlow;
use crate::circuit::gate::directive::Directive;
use crate::circuit::gate::standard_gate::StandardGate;
use crate::circuit::gate::{MCGate, UnitaryGate, gate_matrix};
use crate::circuit::{Parameter, circuit_to_matrix};
use alloc::borrow::Cow;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::SmallVec;
use std::fmt;
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

    /// Control flow operation for conditional or iterative quantum execution.
    ///
    /// This variant supports:
    /// - [`IfElseGate`]: Conditional execution based on classical measurement outcomes
    /// - [`WhileLoopGate`]: Iterative execution based on classical conditions
    ///
    /// # Important
    ///
    /// Control flow operations are **not unitary** and cannot be represented as a matrix.
    /// They require runtime interpretation and may not be supported by all backends.
    ControlFlowGate(ControlFlow),
    /// I gate in QCIS, represented here as Delay, unit is 0.5ns
    Delay,
}

impl Instruction {
    /// Breaks a standard gate into its canonical base gate plus explicit
    /// control count.
    ///
    /// This helper centralizes the standard-gate promotion table so
    /// instruction-form canonicalization and `control()` share exactly the
    /// same notion of which multi-controlled instructions deserve a dedicated
    /// `StandardGate` representation.
    fn decompose_standard_control(g: StandardGate) -> (StandardGate, usize) {
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
    }

    /// Attempts to recompose a base gate and explicit control count into the
    /// most specific `StandardGate` form supported by the current IR.
    fn compose_standard_control(base: StandardGate, total_controls: usize) -> Option<StandardGate> {
        match (base, total_controls) {
            (StandardGate::X, 1) => Some(StandardGate::CX),
            (StandardGate::X, 2) => Some(StandardGate::CCX),
            (StandardGate::Y, 1) => Some(StandardGate::CY),
            (StandardGate::Z, 1) => Some(StandardGate::CZ),
            (StandardGate::RX, 1) => Some(StandardGate::CRX),
            (StandardGate::RY, 1) => Some(StandardGate::CRY),
            (StandardGate::RZ, 1) => Some(StandardGate::CRZ),
            (gate, 0) => Some(gate),
            _ => None,
        }
    }

    /// Returns the canonical instruction-form representation for this
    /// instruction without changing its semantics.
    ///
    /// This only collapses `McGate` values back to dedicated `StandardGate`
    /// variants when such a representation already exists in the IR. It uses
    /// the same control-promotion table as [`Instruction::control`], preserving
    /// the total control count, target convention, and parameter arity of the
    /// base gate. Operation qubit order is not changed here; callers that
    /// rewrite an operation keep the original qubit list attached to the new
    /// instruction.
    ///
    /// It intentionally does not perform decomposition, target-aware lowering,
    /// or matrix-based equivalence detection.
    pub(crate) fn canonicalize_form(&self) -> Instruction {
        match self {
            Instruction::McGate(mc) => {
                let total_ctrls = mc.num_ctrl_qubits();
                let (base, _) = Self::decompose_standard_control(mc.base_gate().to_owned());

                if let Some(std) = Self::compose_standard_control(base, total_ctrls) {
                    Instruction::Standard(std)
                } else {
                    self.clone()
                }
            }
            _ => self.clone(),
        }
    }

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
            Instruction::Standard(g) => g.matrix(params).ok(),
            Instruction::McGate(g) => g.matrix(params).ok(),
            Instruction::UnitaryGate(g) => g.matrix_for_params(params).ok(),
            Instruction::CircuitGate(g) => circuit_to_matrix(&g.circuit.circuit, None)
                .ok()
                .map(Cow::Owned),
            Instruction::Directive(_) => None,
            Instruction::Delay => None,
            Instruction::ControlFlowGate(_) => None,
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
                        let frozen_inv = FrozenCircuit::new(c_inv);
                        // Create new UnitaryGate with inverted circuit
                        let u_inv = UnitaryGate::new(
                            format!("{}_dg", g.label()).as_str(),
                            g.num_qubits(),
                            g.num_params(),
                        )
                        .with_circuit(Arc::new(frozen_inv))
                        .ok()?;
                        return Some((Self::UnitaryGate(Box::new(u_inv)), SmallVec::new()));
                    }
                }
                None
            }
            Instruction::CircuitGate(circuit_gate) => {
                if let Ok(inv_gate) = circuit_gate.inverse() {
                    Some((
                        Instruction::CircuitGate(Box::new(inv_gate)),
                        params.iter().cloned().collect(),
                    ))
                } else {
                    None
                }
            }
            Instruction::Directive(d) => match d {
                Directive::Barrier => Some((Self::Directive(Directive::Barrier), SmallVec::new())),
                _ => None,
            },
            Instruction::Delay => Some((Self::Delay, SmallVec::new())),
            Instruction::ControlFlowGate(_) => None,
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

        match self {
            Instruction::Standard(g) => {
                let (base, curr_ctrls) = Self::decompose_standard_control(*g);
                let total_ctrls = curr_ctrls + num_new_ctrls;

                if let Some(std) = Self::compose_standard_control(base, total_ctrls) {
                    Some(Instruction::Standard(std))
                } else {
                    Some(Instruction::McGate(Box::from(MCGate::new(
                        total_ctrls as u8,
                        base,
                    ))))
                }
            }
            Instruction::McGate(mc) => {
                let total_ctrls = mc.num_ctrl_qubits() + num_new_ctrls;
                let (base, _) = Self::decompose_standard_control(mc.base_gate().to_owned());

                if let Some(std) = Self::compose_standard_control(base, total_ctrls) {
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
                    uni.num_params(),
                );
                if let Some(m) = uni.matrix() {
                    let controlled = gate_matrix::control_matrix(m, num_new_ctrls);
                    // Handle possible matrix dimension error, return None on failure
                    g = match g.with_matrix(controlled) {
                        Ok(gate) => gate,
                        Err(_) => return None,
                    };
                } else if let Some(m) = uni.symbolic_matrix() {
                    let controlled =
                        crate::circuit::symbolic_matrix::gate::control_matrix(m, num_new_ctrls);
                    let params = uni.matrix_params().unwrap_or(&[]);
                    g = match g.with_symbolic_matrix(params.iter().cloned(), controlled) {
                        Ok(gate) => gate,
                        Err(_) => return None,
                    };
                }
                // Copy circuit field if present
                if let Some(c) = uni.circuit() {
                    g = match g.with_circuit(c.clone()) {
                        Ok(gate) => gate,
                        Err(_) => return None,
                    };
                }

                Some(Instruction::UnitaryGate(Box::from(g)))
            }
            Instruction::CircuitGate(_) => None, // CircuitGate does not support control yet
            Instruction::Directive(_) => None,
            Instruction::Delay => None,
            Instruction::ControlFlowGate(_) => None,
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Standard(g) => write!(f, "{}", g),
            Instruction::McGate(g) => write!(f, "{}", g),
            Instruction::UnitaryGate(g) => write!(f, "{}", g),
            Instruction::CircuitGate(g) => write!(f, "{}", g.name),
            Instruction::Directive(i) => write!(f, "{}", i),
            Instruction::Delay => write!(f, "delay"),
            Instruction::ControlFlowGate(g) => write!(f, "{}", g),
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

impl From<ControlFlow> for Instruction {
    fn from(cf: ControlFlow) -> Self {
        Self::ControlFlowGate(cf)
    }
}

#[cfg(test)]
mod tests {
    use super::Instruction;
    use crate::circuit::gate::{MCGate, StandardGate};

    #[test]
    fn canonicalize_form_collapses_supported_mc_gate_forms() {
        let cases = [
            (MCGate::new(0, StandardGate::X), StandardGate::X),
            (MCGate::new(1, StandardGate::X), StandardGate::CX),
            (MCGate::new(2, StandardGate::X), StandardGate::CCX),
            (MCGate::new(1, StandardGate::CX), StandardGate::CCX),
            (MCGate::new(1, StandardGate::Y), StandardGate::CY),
            (MCGate::new(1, StandardGate::Z), StandardGate::CZ),
            (MCGate::new(1, StandardGate::RX), StandardGate::CRX),
            (MCGate::new(1, StandardGate::RY), StandardGate::CRY),
            (MCGate::new(1, StandardGate::RZ), StandardGate::CRZ),
        ];

        for (mc_gate, expected) in cases {
            let expected_controls = expected.num_ctrl_qubits();
            let expected_qubits = expected.num_qubits();
            let expected_params = expected.num_params();
            let canonical = Instruction::McGate(Box::new(mc_gate)).canonicalize_form();

            let Instruction::Standard(actual) = canonical else {
                panic!("expected standard gate {expected:?}");
            };
            assert_eq!(actual, expected);
            assert_eq!(actual.num_ctrl_qubits(), expected_controls);
            assert_eq!(actual.num_qubits(), expected_qubits);
            assert_eq!(actual.num_params(), expected_params);
        }
    }

    #[test]
    fn canonicalize_form_keeps_non_promotable_mc_gate_forms() {
        let cases = [
            MCGate::new(3, StandardGate::X),
            MCGate::new(1, StandardGate::H),
            MCGate::new(1, StandardGate::RXX),
        ];

        for mc_gate in cases {
            let expected_controls = mc_gate.num_ctrl_qubits();
            let expected_qubits = mc_gate.num_qubits();
            let expected_params = mc_gate.num_params();
            let canonical = Instruction::McGate(Box::new(mc_gate)).canonicalize_form();

            let Instruction::McGate(actual) = canonical else {
                panic!("expected MCGate");
            };
            assert_eq!(actual.num_ctrl_qubits(), expected_controls);
            assert_eq!(actual.num_qubits(), expected_qubits);
            assert_eq!(actual.num_params(), expected_params);
        }
    }

    #[test]
    fn control_on_mc_gate_counts_inherent_base_controls_once() {
        let instruction = Instruction::McGate(Box::new(MCGate::new(1, StandardGate::CX)));

        let controlled = instruction.control(1).unwrap();

        let Instruction::McGate(actual) = controlled else {
            panic!("expected MCGate");
        };
        assert_eq!(actual.base_gate(), &StandardGate::X);
        assert_eq!(actual.num_ctrl_qubits(), 3);
        assert_eq!(actual.num_qubits(), 4);
    }
}
