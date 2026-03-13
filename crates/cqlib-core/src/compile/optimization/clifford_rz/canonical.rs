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

use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use crate::circuit::{Operation, Parameter, Qubit};
use crate::compile::error::CompileError;
use crate::compile::prepared::PreparedOperation;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CanonicalGate {
    H,
    X,
    CX,
    RZ,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CanonicalOp {
    pub(crate) gate: CanonicalGate,
    pub(crate) logical_qubits: SmallVec<[usize; 2]>,
    pub(crate) theta: Option<f64>,
    pub(crate) label: Option<Box<str>>,
}

impl CanonicalOp {
    #[cfg(test)]
    pub(crate) fn h(logical: usize) -> Self {
        Self {
            gate: CanonicalGate::H,
            logical_qubits: smallvec![logical],
            theta: None,
            label: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn x(logical: usize) -> Self {
        Self {
            gate: CanonicalGate::X,
            logical_qubits: smallvec![logical],
            theta: None,
            label: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn cx(control: usize, target: usize) -> Self {
        Self {
            gate: CanonicalGate::CX,
            logical_qubits: smallvec![control, target],
            theta: None,
            label: None,
        }
    }

    pub(crate) fn rz(logical: usize, theta: f64) -> Self {
        Self {
            gate: CanonicalGate::RZ,
            logical_qubits: smallvec![logical],
            theta: Some(theta),
            label: None,
        }
    }

    pub(crate) fn is_rz(&self) -> bool {
        self.gate == CanonicalGate::RZ
    }

    pub(crate) fn theta_value(&self) -> f64 {
        self.theta
            .expect("RZ canonical operation must contain a rotation angle")
    }

    pub(crate) fn with_theta(mut self, theta: f64) -> Self {
        self.theta = Some(theta);
        self
    }

    pub(crate) fn to_operation(&self, logical_qubits: &[Qubit]) -> Operation {
        let qubits = self
            .logical_qubits
            .iter()
            .map(|&logical| logical_qubits[logical])
            .collect();
        let instruction = match self.gate {
            CanonicalGate::H => Instruction::Standard(StandardGate::H),
            CanonicalGate::X => Instruction::Standard(StandardGate::X),
            CanonicalGate::CX => Instruction::Standard(StandardGate::CX),
            CanonicalGate::RZ => Instruction::Standard(StandardGate::RZ),
        };
        let params = match self.theta {
            Some(theta) => smallvec![CircuitParam::Fixed(theta)],
            None => smallvec![],
        };
        Operation {
            instruction,
            qubits,
            params,
            label: self.label.clone(),
        }
    }
}

pub(crate) fn approx_zero(value: f64, tol: f64) -> bool {
    value.abs() <= tol
}

pub(crate) fn normalize_4pi(theta: f64) -> f64 {
    theta.rem_euclid(4.0 * PI)
}

pub(crate) fn approx_angle_eq(lhs: f64, rhs: f64, tol: f64) -> bool {
    (lhs - rhs).abs() <= tol
}

pub(crate) fn canonical_sequence_eq(lhs: &[CanonicalOp], rhs: &[CanonicalOp], tol: f64) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.iter().zip(rhs.iter()).all(|(a, b)| {
        a.gate == b.gate
            && a.logical_qubits == b.logical_qubits
            && match (a.theta, b.theta) {
                (Some(x), Some(y)) => approx_angle_eq(x, y, tol),
                (None, None) => true,
                _ => false,
            }
    })
}

pub(crate) fn try_canonicalize(
    prep_op: &PreparedOperation,
    parameter_pool: &[Parameter],
) -> Result<Option<(CanonicalOp, f64)>, CompileError> {
    let params = match resolve_numeric_params(&prep_op.op, parameter_pool)? {
        Some(params) => params,
        None => return Ok(None),
    };

    let logical_qubits = prep_op.logical_qubits.clone();
    let label = prep_op.op.label.clone();
    let op = match &prep_op.op.instruction {
        Instruction::Standard(StandardGate::H) => Some((
            CanonicalOp {
                gate: CanonicalGate::H,
                logical_qubits,
                theta: None,
                label,
            },
            0.0,
        )),
        Instruction::Standard(StandardGate::X) => Some((
            CanonicalOp {
                gate: CanonicalGate::X,
                logical_qubits,
                theta: None,
                label,
            },
            0.0,
        )),
        Instruction::Standard(StandardGate::CX) => Some((
            CanonicalOp {
                gate: CanonicalGate::CX,
                logical_qubits,
                theta: None,
                label,
            },
            0.0,
        )),
        Instruction::Standard(StandardGate::RZ) => {
            if params.len() != 1 {
                return Err(CompileError::Internal(
                    "RZ gate must resolve to exactly one numeric parameter".to_string(),
                ));
            }
            Some((
                CanonicalOp {
                    gate: CanonicalGate::RZ,
                    logical_qubits,
                    theta: Some(params[0]),
                    label,
                },
                0.0,
            ))
        }
        Instruction::Standard(StandardGate::Z) => Some((
            CanonicalOp {
                gate: CanonicalGate::RZ,
                logical_qubits,
                theta: Some(PI),
                label,
            },
            PI / 2.0,
        )),
        Instruction::Standard(StandardGate::S) => Some((
            CanonicalOp {
                gate: CanonicalGate::RZ,
                logical_qubits,
                theta: Some(PI / 2.0),
                label,
            },
            PI / 4.0,
        )),
        Instruction::Standard(StandardGate::SDG) => Some((
            CanonicalOp {
                gate: CanonicalGate::RZ,
                logical_qubits,
                theta: Some(-PI / 2.0),
                label,
            },
            -PI / 4.0,
        )),
        Instruction::Standard(StandardGate::T) => Some((
            CanonicalOp {
                gate: CanonicalGate::RZ,
                logical_qubits,
                theta: Some(PI / 4.0),
                label,
            },
            PI / 8.0,
        )),
        Instruction::Standard(StandardGate::TDG) => Some((
            CanonicalOp {
                gate: CanonicalGate::RZ,
                logical_qubits,
                theta: Some(-PI / 4.0),
                label,
            },
            -PI / 8.0,
        )),
        _ => None,
    };
    Ok(op)
}

pub(crate) fn exact_rz_rewrite(theta: f64, tol: f64) -> Option<(Option<StandardGate>, f64)> {
    let theta = normalize_4pi(theta);
    let kf = theta / (PI / 4.0);
    let k = kf.round();
    if !approx_angle_eq(kf, k, tol) {
        return None;
    }

    match (k as i64).rem_euclid(16) as i32 {
        0 => Some((None, 0.0)),
        1 => Some((Some(StandardGate::T), -PI / 8.0)),
        2 => Some((Some(StandardGate::S), -PI / 4.0)),
        4 => Some((Some(StandardGate::Z), -PI / 2.0)),
        6 => Some((Some(StandardGate::SDG), 5.0 * PI / 4.0)),
        7 => Some((Some(StandardGate::TDG), 9.0 * PI / 8.0)),
        8 => Some((None, PI)),
        9 => Some((Some(StandardGate::T), 7.0 * PI / 8.0)),
        10 => Some((Some(StandardGate::S), 3.0 * PI / 4.0)),
        12 => Some((Some(StandardGate::Z), PI / 2.0)),
        14 => Some((Some(StandardGate::SDG), PI / 4.0)),
        15 => Some((Some(StandardGate::TDG), PI / 8.0)),
        _ => None,
    }
}

fn resolve_numeric_params(
    op: &Operation,
    parameter_pool: &[Parameter],
) -> Result<Option<SmallVec<[f64; 3]>>, CompileError> {
    let mut out = SmallVec::<[f64; 3]>::with_capacity(op.params.len());
    for param in &op.params {
        match param {
            CircuitParam::Fixed(value) => out.push(*value),
            CircuitParam::Index(index) => {
                let idx = *index as usize;
                let Some(parameter) = parameter_pool.get(idx) else {
                    return Err(CompileError::Internal(format!(
                        "operation references missing parameter index {}",
                        idx
                    )));
                };
                match parameter.evaluate(&None) {
                    Ok(value) => out.push(value),
                    Err(_) => return Ok(None),
                }
            }
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Qubit};
    use crate::compile::prepared::preprocess_circuit;

    #[test]
    fn test_exact_rz_rewrite_pi_maps_to_z() {
        let (gate, phase) = exact_rz_rewrite(PI, 1e-10).unwrap();
        assert_eq!(gate, Some(StandardGate::Z));
        assert!((phase + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonicalize_t_as_rz_and_phase() {
        let mut circuit = Circuit::new(1);
        circuit.t(Qubit::new(0)).unwrap();
        let prepared = preprocess_circuit(&circuit).unwrap();
        let (op, phase) = try_canonicalize(&prepared.operations[0], prepared.parameters.as_slice())
            .unwrap()
            .unwrap();
        assert_eq!(op.gate, CanonicalGate::RZ);
        assert!((op.theta_value() - PI / 4.0).abs() < 1e-10);
        assert!((phase - PI / 8.0).abs() < 1e-10);
    }
}
