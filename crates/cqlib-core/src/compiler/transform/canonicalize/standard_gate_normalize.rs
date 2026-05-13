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

//! Standard-gate normalization for canonicalization.
//!
//! This module is not part of the declarative rule knowledge system. It owns
//! canonicalizer-specific policy for fixed numeric standard-gate parameters:
//! angle principal values, tolerance-aware special-angle recognition, and
//! explicit global-phase compensation for strict matrix preservation.

use crate::circuit::{Parameter, ParameterValue, StandardGate};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

const ANGLE_EPS: f64 = 1e-12;

/// Global-phase handling policy for standard-gate normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GlobalPhasePolicy {
    /// Preserve global phase exactly by emitting `GPhase` gates when needed.
    Preserve,
}

/// A normalized standard-gate operation without qubit or label context.
#[derive(Debug, Clone)]
pub(crate) struct NormalizedStandardOp {
    pub(crate) gate: StandardGate,
    pub(crate) params: SmallVec<[ParameterValue; 3]>,
}

impl NormalizedStandardOp {
    fn new(gate: StandardGate, params: SmallVec<[ParameterValue; 3]>) -> Self {
        Self { gate, params }
    }

    fn fixed(gate: StandardGate, params: &[f64]) -> Self {
        Self::new(
            gate,
            params.iter().copied().map(ParameterValue::Fixed).collect(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AngleClass {
    Zero,
    PlusQuarterPi,
    MinusQuarterPi,
    PlusHalfPi,
    MinusHalfPi,
    PlusPi,
    MinusPi,
    PlusTwoPi,
    MinusTwoPi,
    Other(f64),
}

#[derive(Debug, Clone, Copy)]
enum AnglePeriod {
    TwoPi,
    FourPi,
}

impl AnglePeriod {
    fn value(self) -> f64 {
        match self {
            Self::TwoPi => TAU,
            Self::FourPi => 2.0 * TAU,
        }
    }
}

struct AnglePolicy;

impl AnglePolicy {
    fn classify(theta: f64, period: AnglePeriod) -> AngleClass {
        let theta = Self::principal_value(theta, period);
        if Self::close_to(theta, 0.0) {
            AngleClass::Zero
        } else if Self::close_to(theta, FRAC_PI_4) {
            AngleClass::PlusQuarterPi
        } else if Self::close_to(theta, -FRAC_PI_4) {
            AngleClass::MinusQuarterPi
        } else if Self::close_to(theta, FRAC_PI_2) {
            AngleClass::PlusHalfPi
        } else if Self::close_to(theta, -FRAC_PI_2) {
            AngleClass::MinusHalfPi
        } else if Self::close_to(theta, PI) {
            AngleClass::PlusPi
        } else if Self::close_to(theta, -PI) {
            AngleClass::MinusPi
        } else if Self::close_to(theta, TAU) {
            AngleClass::PlusTwoPi
        } else if Self::close_to(theta, -TAU) {
            AngleClass::MinusTwoPi
        } else {
            AngleClass::Other(theta)
        }
    }

    fn principal_value(theta: f64, period: AnglePeriod) -> f64 {
        let period = period.value();
        let mut normalized = theta.rem_euclid(period);
        if normalized > period / 2.0 + ANGLE_EPS {
            normalized -= period;
        }
        if Self::close_to(normalized, 0.0) {
            0.0
        } else if Self::close_to(normalized.abs(), period / 2.0) {
            normalized.signum() * period / 2.0
        } else {
            normalized
        }
    }

    fn close_to(lhs: f64, rhs: f64) -> bool {
        (lhs - rhs).abs() <= ANGLE_EPS
    }
}

struct OpBuilder {
    policy: GlobalPhasePolicy,
}

impl OpBuilder {
    fn new(policy: GlobalPhasePolicy) -> Self {
        Self { policy }
    }

    fn fixed(&self, gate: StandardGate, params: &[f64]) -> Vec<NormalizedStandardOp> {
        vec![NormalizedStandardOp::fixed(gate, params)]
    }

    fn named(&self, gate: StandardGate) -> Vec<NormalizedStandardOp> {
        vec![NormalizedStandardOp::new(gate, smallvec![])]
    }

    fn global_phase(&self, theta: f64) -> Vec<NormalizedStandardOp> {
        match self.policy {
            GlobalPhasePolicy::Preserve => normalize_gphase(theta),
        }
    }

    fn global_phase_then_named(&self, theta: f64, gate: StandardGate) -> Vec<NormalizedStandardOp> {
        let mut ops = self.global_phase(theta);
        ops.push(NormalizedStandardOp::new(gate, smallvec![]));
        ops
    }
}

/// Normalizes one `StandardGate` plus semantic parameters into zero or more
/// equivalent standard-gate operations.
///
/// Symbolic or non-finite parameters are left in their original semantic form.
/// Fixed finite parameters are normalized into canonical standard-gate forms.
pub(crate) fn normalize_standard_gate(
    gate: StandardGate,
    params: &[Parameter],
    policy: GlobalPhasePolicy,
) -> Vec<NormalizedStandardOp> {
    debug_assert_eq!(params.len(), gate.num_params());
    let values = match fixed_values(params) {
        Some(values) => values,
        None => return symbolic_op(gate, params),
    };

    GateFamilyNormalizer::new(policy).normalize(gate, &values)
}

fn fixed_values(params: &[Parameter]) -> Option<SmallVec<[f64; 3]>> {
    params
        .iter()
        .map(|param| {
            let value = param.evaluate(&None).ok()?;
            value.is_finite().then_some(value)
        })
        .collect()
}

fn symbolic_op(gate: StandardGate, params: &[Parameter]) -> Vec<NormalizedStandardOp> {
    vec![NormalizedStandardOp::new(
        gate,
        params.iter().cloned().map(ParameterValue::from).collect(),
    )]
}

struct GateFamilyNormalizer {
    ops: OpBuilder,
}

impl GateFamilyNormalizer {
    fn new(policy: GlobalPhasePolicy) -> Self {
        Self {
            ops: OpBuilder::new(policy),
        }
    }

    fn normalize(&self, gate: StandardGate, params: &[f64]) -> Vec<NormalizedStandardOp> {
        match gate {
            StandardGate::I => vec![],
            StandardGate::GPhase => normalize_gphase(params[0]),
            StandardGate::Phase => self.normalize_phase(params[0]),
            StandardGate::RX | StandardGate::RY | StandardGate::RZ => {
                self.normalize_single_pauli_rotation(gate, params[0])
            }
            StandardGate::RXX | StandardGate::RYY | StandardGate::RZZ | StandardGate::RZX => {
                self.normalize_two_pauli_rotation(gate, params[0])
            }
            StandardGate::RXY => self.normalize_rxy(params[0], params[1]),
            StandardGate::XY | StandardGate::XY2P | StandardGate::XY2M => {
                self.normalize_periodic_phase_param_gate(gate, params[0])
            }
            StandardGate::CRX | StandardGate::CRY | StandardGate::CRZ => {
                self.normalize_controlled_rotation(gate, params[0])
            }
            StandardGate::FSIM => self.normalize_fsim(params[0], params[1]),
            _ => self.ops.fixed(gate, params),
        }
    }

    fn normalize_phase(&self, lambda: f64) -> Vec<NormalizedStandardOp> {
        match AnglePolicy::classify(lambda, AnglePeriod::TwoPi) {
            AngleClass::Zero => vec![],
            AngleClass::PlusHalfPi => self.ops.named(StandardGate::S),
            AngleClass::MinusHalfPi => self.ops.named(StandardGate::SDG),
            AngleClass::PlusPi | AngleClass::MinusPi => self.ops.named(StandardGate::Z),
            AngleClass::PlusQuarterPi => self.ops.named(StandardGate::T),
            AngleClass::MinusQuarterPi => self.ops.named(StandardGate::TDG),
            AngleClass::Other(lambda) => self.ops.fixed(StandardGate::Phase, &[lambda]),
            AngleClass::PlusTwoPi | AngleClass::MinusTwoPi => unreachable!("2π period collapsed"),
        }
    }

    fn normalize_single_pauli_rotation(
        &self,
        gate: StandardGate,
        theta: f64,
    ) -> Vec<NormalizedStandardOp> {
        match AnglePolicy::classify(theta, AnglePeriod::FourPi) {
            AngleClass::Zero => vec![],
            AngleClass::PlusTwoPi | AngleClass::MinusTwoPi => self.ops.global_phase(PI),
            AngleClass::PlusPi => self.normalize_pauli_pi(gate, false),
            AngleClass::MinusPi => self.normalize_pauli_pi(gate, true),
            AngleClass::PlusHalfPi => match gate {
                StandardGate::RX => self.ops.named(StandardGate::X2P),
                StandardGate::RY => self.ops.named(StandardGate::Y2P),
                StandardGate::RZ => self.ops.fixed(gate, &[FRAC_PI_2]),
                _ => unreachable!(),
            },
            AngleClass::MinusHalfPi => match gate {
                StandardGate::RX => self.ops.named(StandardGate::X2M),
                StandardGate::RY => self.ops.named(StandardGate::Y2M),
                StandardGate::RZ => self.ops.fixed(gate, &[-FRAC_PI_2]),
                _ => unreachable!(),
            },
            AngleClass::Other(theta) => self.ops.fixed(gate, &[theta]),
            AngleClass::PlusQuarterPi | AngleClass::MinusQuarterPi => self.ops.fixed(
                gate,
                &[AnglePolicy::principal_value(theta, AnglePeriod::FourPi)],
            ),
        }
    }

    fn normalize_pauli_pi(
        &self,
        rotation_gate: StandardGate,
        negative: bool,
    ) -> Vec<NormalizedStandardOp> {
        let phase = if negative { FRAC_PI_2 } else { -FRAC_PI_2 };
        let pauli = match rotation_gate {
            StandardGate::RX => StandardGate::X,
            StandardGate::RY => StandardGate::Y,
            StandardGate::RZ => StandardGate::Z,
            _ => unreachable!(),
        };
        self.ops.global_phase_then_named(phase, pauli)
    }

    fn normalize_two_pauli_rotation(
        &self,
        gate: StandardGate,
        theta: f64,
    ) -> Vec<NormalizedStandardOp> {
        match AnglePolicy::classify(theta, AnglePeriod::FourPi) {
            AngleClass::Zero => vec![],
            AngleClass::PlusTwoPi | AngleClass::MinusTwoPi => self.ops.global_phase(PI),
            AngleClass::Other(theta) => self.ops.fixed(gate, &[theta]),
            _ => self.ops.fixed(
                gate,
                &[AnglePolicy::principal_value(theta, AnglePeriod::FourPi)],
            ),
        }
    }

    fn normalize_rxy(&self, theta: f64, phi: f64) -> Vec<NormalizedStandardOp> {
        let phi = AnglePolicy::principal_value(phi, AnglePeriod::TwoPi);
        match AnglePolicy::classify(theta, AnglePeriod::FourPi) {
            AngleClass::Zero => vec![],
            AngleClass::PlusTwoPi | AngleClass::MinusTwoPi => self.ops.global_phase(PI),
            AngleClass::PlusPi => self.ops.fixed(StandardGate::XY, &[phi]),
            AngleClass::MinusPi => vec![
                NormalizedStandardOp::fixed(StandardGate::XY, &[phi]),
                NormalizedStandardOp::fixed(StandardGate::GPhase, &[PI]),
            ],
            AngleClass::PlusHalfPi => self.ops.fixed(StandardGate::XY2P, &[phi]),
            AngleClass::MinusHalfPi => self.ops.fixed(StandardGate::XY2M, &[phi]),
            AngleClass::Other(theta) => self.ops.fixed(StandardGate::RXY, &[theta, phi]),
            AngleClass::PlusQuarterPi | AngleClass::MinusQuarterPi => self.ops.fixed(
                StandardGate::RXY,
                &[
                    AnglePolicy::principal_value(theta, AnglePeriod::FourPi),
                    phi,
                ],
            ),
        }
    }

    fn normalize_periodic_phase_param_gate(
        &self,
        gate: StandardGate,
        theta: f64,
    ) -> Vec<NormalizedStandardOp> {
        self.ops.fixed(
            gate,
            &[AnglePolicy::principal_value(theta, AnglePeriod::TwoPi)],
        )
    }

    fn normalize_controlled_rotation(
        &self,
        gate: StandardGate,
        theta: f64,
    ) -> Vec<NormalizedStandardOp> {
        match AnglePolicy::classify(theta, AnglePeriod::FourPi) {
            AngleClass::Zero => vec![],
            AngleClass::Other(theta) => self.ops.fixed(gate, &[theta]),
            _ => self.ops.fixed(
                gate,
                &[AnglePolicy::principal_value(theta, AnglePeriod::FourPi)],
            ),
        }
    }

    fn normalize_fsim(&self, theta: f64, phi: f64) -> Vec<NormalizedStandardOp> {
        let theta = AnglePolicy::principal_value(theta, AnglePeriod::TwoPi);
        let phi = AnglePolicy::principal_value(phi, AnglePeriod::TwoPi);
        if AnglePolicy::close_to(theta, 0.0) && AnglePolicy::close_to(phi, 0.0) {
            vec![]
        } else {
            self.ops.fixed(StandardGate::FSIM, &[theta, phi])
        }
    }
}

fn normalize_gphase(theta: f64) -> Vec<NormalizedStandardOp> {
    let theta = AnglePolicy::principal_value(theta, AnglePeriod::TwoPi);
    if AnglePolicy::close_to(theta, 0.0) {
        vec![]
    } else {
        vec![NormalizedStandardOp::fixed(StandardGate::GPhase, &[theta])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Zip};
    use num::complex::Complex64;

    fn fixed_params(op: &NormalizedStandardOp) -> Vec<f64> {
        op.params
            .iter()
            .map(|param| match param {
                ParameterValue::Fixed(value) => *value,
                ParameterValue::Param(_) => panic!("expected fixed parameter"),
            })
            .collect()
    }

    fn close_to(lhs: f64, rhs: f64) -> bool {
        AnglePolicy::close_to(lhs, rhs)
    }

    fn normalize(gate: StandardGate, params: &[f64]) -> Vec<NormalizedStandardOp> {
        let params: Vec<_> = params.iter().copied().map(Parameter::from).collect();
        normalize_standard_gate(gate, &params, GlobalPhasePolicy::Preserve)
    }

    fn assert_matrix_preserved(gate: StandardGate, params: &[f64]) {
        let expected = gate.matrix(params).unwrap().as_ref().clone();
        let actual = sequence_matrix(gate.num_qubits(), &normalize(gate, params));
        assert_matrix_close(&expected, &actual);
    }

    fn sequence_matrix(num_qubits: usize, ops: &[NormalizedStandardOp]) -> Array2<Complex64> {
        let dim = 1usize << num_qubits;
        let mut matrix = Array2::eye(dim);
        for op in ops {
            if op.gate == StandardGate::GPhase {
                let theta = fixed_params(op)[0];
                let phase = Complex64::new(theta.cos(), theta.sin());
                matrix.mapv_inplace(|value| value * phase);
            } else {
                let gate_matrix = op.gate.matrix(&fixed_params(op)).unwrap().as_ref().clone();
                matrix = gate_matrix.dot(&matrix);
            }
        }
        matrix
    }

    fn assert_matrix_close(lhs: &Array2<Complex64>, rhs: &Array2<Complex64>) {
        assert_eq!(lhs.shape(), rhs.shape());
        Zip::from(lhs).and(rhs).for_each(|lhs, rhs| {
            assert!(
                (*lhs - *rhs).norm() <= 1e-10,
                "matrix mismatch: {lhs:?} vs {rhs:?}"
            );
        });
    }

    #[test]
    fn rx_two_pi_preserves_global_phase() {
        let ops = normalize(StandardGate::RX, &[TAU]);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::GPhase);
        assert!(close_to(fixed_params(&ops[0])[0], PI));
        assert_matrix_preserved(StandardGate::RX, &[TAU]);
    }

    #[test]
    fn phase_special_angles_fold_to_named_gates() {
        let cases = [
            (FRAC_PI_2, StandardGate::S),
            (-FRAC_PI_2, StandardGate::SDG),
            (FRAC_PI_4, StandardGate::T),
            (-FRAC_PI_4, StandardGate::TDG),
            (PI, StandardGate::Z),
        ];

        for (angle, gate) in cases {
            let ops = normalize(StandardGate::Phase, &[angle]);
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0].gate, gate);
            assert_matrix_preserved(StandardGate::Phase, &[angle]);
        }
    }

    #[test]
    fn rxy_special_angles_fold_to_xy_family() {
        let cases = [
            (PI, StandardGate::XY),
            (FRAC_PI_2, StandardGate::XY2P),
            (-FRAC_PI_2, StandardGate::XY2M),
        ];

        for (theta, gate) in cases {
            let ops = normalize(StandardGate::RXY, &[theta, TAU + 0.25]);
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0].gate, gate);
            assert!(close_to(fixed_params(&ops[0])[0], 0.25));
            assert_matrix_preserved(StandardGate::RXY, &[theta, TAU + 0.25]);
        }
    }

    #[test]
    fn controlled_rotation_two_pi_is_not_global_phase() {
        let ops = normalize(StandardGate::CRZ, &[TAU]);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::CRZ);
        assert!(close_to(fixed_params(&ops[0])[0], TAU));
        assert_matrix_preserved(StandardGate::CRZ, &[TAU]);
    }

    // --- Edge case tests ---

    fn param_value(op: &NormalizedStandardOp, idx: usize) -> ParameterValue {
        op.params[idx].clone()
    }

    #[test]
    fn nan_parameter_falls_back_to_symbolic() {
        let ops = normalize_standard_gate(
            StandardGate::RX,
            &[Parameter::from(f64::NAN)],
            GlobalPhasePolicy::Preserve,
        );
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::RX);
        assert!(
            matches!(param_value(&ops[0], 0), ParameterValue::Param(_)),
            "NaN should produce a symbolic Param, not Fixed"
        );
    }

    #[test]
    fn infinity_parameter_falls_back_to_symbolic() {
        let ops = normalize_standard_gate(
            StandardGate::RX,
            &[Parameter::from(f64::INFINITY)],
            GlobalPhasePolicy::Preserve,
        );
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::RX);
        assert!(
            matches!(param_value(&ops[0], 0), ParameterValue::Param(_)),
            "Infinity should produce a symbolic Param, not Fixed"
        );
    }

    #[test]
    fn angle_eps_boundary_below_eps_is_zero() {
        let ops = normalize(StandardGate::RX, &[1e-13]);
        assert!(
            ops.is_empty(),
            "1e-13 should be within ANGLE_EPS and eliminated"
        );
    }

    #[test]
    fn angle_eps_boundary_at_eps_is_zero() {
        let ops = normalize(StandardGate::RX, &[1e-12]);
        assert!(ops.is_empty(), "1e-12 (== ANGLE_EPS) should be eliminated");
    }

    #[test]
    fn angle_eps_boundary_above_eps_is_preserved() {
        let ops = normalize(StandardGate::RX, &[1e-11]);
        assert_eq!(ops.len(), 1, "1e-11 should NOT be eliminated");
        assert_eq!(ops[0].gate, StandardGate::RX);
    }

    #[test]
    fn negative_zero_is_eliminated() {
        let ops = normalize(StandardGate::RX, &[-0.0]);
        assert!(ops.is_empty(), "-0.0 should be eliminated as zero");
    }

    #[test]
    fn rxy_negative_pi_produces_xy_plus_gphase() {
        let ops = normalize(StandardGate::RXY, &[-PI, 0.5]);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].gate, StandardGate::XY);
        assert!(close_to(fixed_params(&ops[0])[0], 0.5));
        assert_eq!(ops[1].gate, StandardGate::GPhase);
        assert!(close_to(fixed_params(&ops[1])[0], PI));
        assert_matrix_preserved(StandardGate::RXY, &[-PI, 0.5]);
    }

    #[test]
    fn phase_negative_pi_folds_to_z() {
        let ops = normalize(StandardGate::Phase, &[-PI]);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::Z);
        assert_matrix_preserved(StandardGate::Phase, &[-PI]);
    }

    #[test]
    fn pauli_pi_rewrites_preserve_strict_matrix_with_gphase() {
        for (gate, theta) in [
            (StandardGate::RX, PI),
            (StandardGate::RX, -PI),
            (StandardGate::RY, PI),
            (StandardGate::RY, -PI),
            (StandardGate::RZ, PI),
            (StandardGate::RZ, -PI),
        ] {
            assert_matrix_preserved(gate, &[theta]);
        }
    }

    #[test]
    fn two_pauli_two_pi_rewrites_preserve_strict_matrix_with_gphase() {
        for gate in [
            StandardGate::RXX,
            StandardGate::RYY,
            StandardGate::RZZ,
            StandardGate::RZX,
        ] {
            let ops = normalize(gate, &[TAU]);
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0].gate, StandardGate::GPhase);
            assert_matrix_preserved(gate, &[TAU]);
        }
    }

    #[test]
    fn fsim_zero_zero_is_eliminated() {
        assert!(normalize(StandardGate::FSIM, &[TAU, -TAU]).is_empty());
    }

    #[test]
    fn symbolic_parameter_keeps_original_gate() {
        let theta = Parameter::symbol("theta");
        let ops = normalize_standard_gate(StandardGate::RX, &[theta], GlobalPhasePolicy::Preserve);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].gate, StandardGate::RX);
        assert!(matches!(param_value(&ops[0], 0), ParameterValue::Param(_)));
    }
}
