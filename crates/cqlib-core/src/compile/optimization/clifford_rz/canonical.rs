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
    pub(crate) fn new(
        gate: CanonicalGate,
        logical_qubits: SmallVec<[usize; 2]>,
        theta: Option<f64>,
    ) -> Self {
        Self {
            gate,
            logical_qubits,
            theta,
            label: None,
        }
    }

    pub(crate) fn with_label(mut self, label: Option<Box<str>>) -> Self {
        self.label = label;
        self
    }

    pub(crate) fn h(logical: usize) -> Self {
        Self::new(CanonicalGate::H, smallvec![logical], None)
    }

    pub(crate) fn x(logical: usize) -> Self {
        Self::new(CanonicalGate::X, smallvec![logical], None)
    }

    pub(crate) fn cx(control: usize, target: usize) -> Self {
        Self::new(CanonicalGate::CX, smallvec![control, target], None)
    }

    pub(crate) fn rz(logical: usize, theta: f64) -> Self {
        Self::new(CanonicalGate::RZ, smallvec![logical], Some(theta))
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
) -> Result<Option<(Vec<CanonicalOp>, f64)>, CompileError> {
    let params = match resolve_numeric_params(&prep_op.op, parameter_pool)? {
        Some(params) => params,
        None => return Ok(None),
    };

    let sequence = match &prep_op.op.instruction {
        Instruction::Standard(gate) => {
            canonicalize_standard_gate(*gate, &prep_op.logical_qubits, &params)?
        }
        _ => return Ok(None),
    };

    let Some((mut ops, phase)) = sequence else {
        return Ok(None);
    };
    if let Some(first) = ops.first_mut() {
        first.label = prep_op.op.label.clone();
    }
    Ok(Some((ops, phase)))
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

fn canonicalize_standard_gate(
    gate: StandardGate,
    logical_qubits: &SmallVec<[usize; 2]>,
    params: &[f64],
) -> Result<Option<(Vec<CanonicalOp>, f64)>, CompileError> {
    let seq = match gate {
        StandardGate::I => Some((Vec::new(), 0.0)),
        StandardGate::H => Some((vec![CanonicalOp::h(logical_qubits[0])], 0.0)),
        StandardGate::X => Some((vec![CanonicalOp::x(logical_qubits[0])], 0.0)),
        StandardGate::Y => Some((
            vec![
                CanonicalOp::x(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], PI),
            ],
            0.0,
        )),
        StandardGate::Z => Some((vec![CanonicalOp::rz(logical_qubits[0], PI)], PI / 2.0)),
        StandardGate::S => Some((vec![CanonicalOp::rz(logical_qubits[0], PI / 2.0)], PI / 4.0)),
        StandardGate::SDG => Some((
            vec![CanonicalOp::rz(logical_qubits[0], -PI / 2.0)],
            -PI / 4.0,
        )),
        StandardGate::T => Some((vec![CanonicalOp::rz(logical_qubits[0], PI / 4.0)], PI / 8.0)),
        StandardGate::TDG => Some((
            vec![CanonicalOp::rz(logical_qubits[0], -PI / 4.0)],
            -PI / 8.0,
        )),
        StandardGate::X2P => Some((
            vec![
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
            ],
            0.0,
        )),
        StandardGate::X2M => Some((
            vec![
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], -PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
            ],
            0.0,
        )),
        StandardGate::Y2P => Some((
            vec![
                CanonicalOp::rz(logical_qubits[0], -PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], PI / 2.0),
            ],
            0.0,
        )),
        StandardGate::Y2M => Some((
            vec![
                CanonicalOp::rz(logical_qubits[0], -PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], -PI / 2.0),
                CanonicalOp::h(logical_qubits[0]),
                CanonicalOp::rz(logical_qubits[0], PI / 2.0),
            ],
            0.0,
        )),
        StandardGate::CX => Some((
            vec![CanonicalOp::cx(logical_qubits[0], logical_qubits[1])],
            0.0,
        )),
        StandardGate::CY => Some((
            vec![
                CanonicalOp::rz(logical_qubits[1], -PI / 2.0),
                CanonicalOp::cx(logical_qubits[0], logical_qubits[1]),
                CanonicalOp::rz(logical_qubits[1], PI / 2.0),
            ],
            0.0,
        )),
        StandardGate::CZ => Some((
            vec![
                CanonicalOp::h(logical_qubits[1]),
                CanonicalOp::cx(logical_qubits[0], logical_qubits[1]),
                CanonicalOp::h(logical_qubits[1]),
            ],
            0.0,
        )),
        StandardGate::SWAP => Some((
            vec![
                CanonicalOp::cx(logical_qubits[0], logical_qubits[1]),
                CanonicalOp::cx(logical_qubits[1], logical_qubits[0]),
                CanonicalOp::cx(logical_qubits[0], logical_qubits[1]),
            ],
            0.0,
        )),
        StandardGate::RZ => {
            let theta = expect_one_numeric_param(gate, params)?;
            Some((vec![CanonicalOp::rz(logical_qubits[0], theta)], 0.0))
        }
        StandardGate::Phase => {
            let theta = expect_one_numeric_param(gate, params)?;
            Some((vec![CanonicalOp::rz(logical_qubits[0], theta)], theta / 2.0))
        }
        _ => None,
    };
    Ok(seq)
}

fn expect_one_numeric_param(gate: StandardGate, params: &[f64]) -> Result<f64, CompileError> {
    if params.len() != 1 {
        return Err(CompileError::Internal(format!(
            "{gate} gate must resolve to exactly one numeric parameter"
        )));
    }
    Ok(params[0])
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
    use crate::circuit::circuit_to_matrix;
    use crate::circuit::param::ParameterValue;
    use crate::circuit::{Circuit, Qubit};
    use crate::compile::prepared::preprocess_circuit;
    use num_complex::Complex64;

    fn matrix_with_global_phase(circuit: &Circuit) -> ndarray::Array2<Complex64> {
        let mut matrix = circuit_to_matrix(circuit, None).unwrap();
        let phase = circuit.global_phase().evaluate(&None).unwrap();
        let factor = Complex64::from_polar(1.0, phase);
        matrix.mapv_inplace(|value| factor * value);
        matrix
    }

    fn assert_matrix_eq(lhs: &Circuit, rhs: &Circuit) {
        let left = matrix_with_global_phase(lhs);
        let right = matrix_with_global_phase(rhs);
        assert_eq!(left.dim(), right.dim());
        for (a, b) in left.iter().zip(right.iter()) {
            assert!(
                (*a - *b).norm() <= 1e-9,
                "matrix mismatch: lhs={:?}, rhs={:?}",
                a,
                b
            );
        }
    }

    fn canonicalized_single_gate(
        gate: StandardGate,
        qubits: &[u32],
        params: &[f64],
    ) -> (Circuit, Circuit) {
        let num_qubits = qubits
            .iter()
            .copied()
            .max()
            .map(|value| value + 1)
            .unwrap_or(1) as usize;
        let mut circuit = Circuit::new(num_qubits);
        let values = params
            .iter()
            .copied()
            .map(ParameterValue::Fixed)
            .collect::<Vec<_>>();
        circuit
            .append(
                Instruction::Standard(gate),
                qubits.iter().copied().map(Qubit::new),
                values,
                None,
            )
            .unwrap();

        let prepared = preprocess_circuit(&circuit).unwrap();
        let (ops, phase) =
            try_canonicalize(&prepared.operations[0], prepared.parameters.as_slice())
                .unwrap()
                .unwrap();

        let mut rewritten = Circuit::new(num_qubits);
        for op in ops {
            rewritten
                .append_operation(op.to_operation(&rewritten.qubits()))
                .unwrap();
        }
        rewritten.set_global_phase(Parameter::from(phase));
        (circuit, rewritten)
    }

    trait AppendOperationExt {
        fn append_operation(&mut self, op: Operation) -> Result<(), crate::circuit::CircuitError>;
    }

    impl AppendOperationExt for Circuit {
        fn append_operation(&mut self, op: Operation) -> Result<(), crate::circuit::CircuitError> {
            use crate::circuit::param::ParameterValue;

            let params = op
                .params
                .iter()
                .map(|param| match param {
                    CircuitParam::Fixed(value) => ParameterValue::Fixed(*value),
                    CircuitParam::Index(_) => unreachable!("canonical ops do not carry indices"),
                })
                .collect::<Vec<_>>();
            self.append(op.instruction, op.qubits, params, op.label.as_deref())
        }
    }

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
        let (ops, phase) =
            try_canonicalize(&prepared.operations[0], prepared.parameters.as_slice())
                .unwrap()
                .unwrap();
        assert_eq!(ops, vec![CanonicalOp::rz(0, PI / 4.0)]);
        assert!((phase - PI / 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonicalize_phase_gate_to_rz_and_phase() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::Phase, &[0], &[0.3]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_y_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::Y, &[0], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_x2p_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::X2P, &[0], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_x2m_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::X2M, &[0], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_y2p_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::Y2P, &[0], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_y2m_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::Y2M, &[0], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_cy_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::CY, &[0, 1], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_cz_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::CZ, &[0, 1], &[]);
        assert_matrix_eq(&source, &rewritten);
    }

    #[test]
    fn test_canonicalize_swap_gate_matrix() {
        let (source, rewritten) = canonicalized_single_gate(StandardGate::SWAP, &[0, 1], &[]);
        assert_matrix_eq(&source, &rewritten);
    }
}
