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

use super::checker::{Commutation, CommutationChecker, CommutationConfig, CommutationResult};
use crate::circuit::{Instruction, Parameter, Qubit, StandardGate, UnitaryGate};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, PI};

fn algebra_only_checker() -> CommutationChecker {
    CommutationChecker::with_config(CommutationConfig {
        enable_rule_oracle: false,
        enable_matrix_fallback: false,
        ..CommutationConfig::default()
    })
}

fn assert_exact(result: CommutationResult) {
    assert_eq!(result, Some(Commutation::Exact));
}

fn assert_pi_phase(result: CommutationResult) {
    let Some(Commutation::UpToGlobalPhase(phase)) = result else {
        panic!("expected global phase commutation");
    };
    assert!((phase.evaluate(&None).unwrap() - PI).abs() < 1e-10);
}

#[test]
fn identity_commutes_exactly() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::I),
        &[Qubit::new(0)],
        &[],
        &Instruction::Standard(StandardGate::H),
        &[Qubit::new(0)],
        &[],
    );

    assert_exact(result);
}

#[test]
fn disjoint_operations_commute_exactly() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::H),
        &[Qubit::new(0)],
        &[],
        &Instruction::Standard(StandardGate::X),
        &[Qubit::new(1)],
        &[],
    );

    assert_exact(result);
}

#[test]
fn symbolic_rz_family_commutes_exactly() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::symbol("b")],
    );

    assert_exact(result);
}

#[test]
fn symbolic_rx_ry_is_not_proven_commuting() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RX),
        &[Qubit::new(0)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RY),
        &[Qubit::new(0)],
        &[Parameter::symbol("b")],
    );

    assert!(result.is_none());
}

#[test]
fn controlled_rule_commutes_cx_with_rz_on_control() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::CX),
        &[Qubit::new(0), Qubit::new(1)],
        &[],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );

    assert_exact(result);
}

#[test]
fn algebraic_checker_proves_controlled_axis_without_rule_or_matrix() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::CX),
        &[Qubit::new(0), Qubit::new(1)],
        &[],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );

    assert_exact(result);
}

#[test]
fn pauli_interactions_use_symplectic_commutation() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RXX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RZZ),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("b")],
    );

    assert_exact(result);
}

#[test]
fn matrix_fallback_returns_global_phase_for_x_z() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::X),
        &[Qubit::new(0)],
        &[],
        &Instruction::Standard(StandardGate::Z),
        &[Qubit::new(0)],
        &[],
    );

    assert_pi_phase(result);
}

#[test]
fn h_x_does_not_commute_even_up_to_global_phase() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::H),
        &[Qubit::new(0)],
        &[],
        &Instruction::Standard(StandardGate::X),
        &[Qubit::new(0)],
        &[],
    );

    assert!(result.is_none());
}

#[test]
fn pi_rotations_use_pauli_product_phase() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RX),
        &[Qubit::new(0)],
        &[Parameter::from(PI)],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::from(PI)],
    );

    assert_pi_phase(result);
}

#[test]
fn symbolic_same_axis_rotations_commute_algebraically() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RXX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RXX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("b")],
    );

    assert_exact(result);
}

#[test]
fn symbolic_anti_commuting_rotations_are_conservative() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RXX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RZX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("b")],
    );

    assert!(result.is_none());
}

#[test]
fn rxy_same_planar_axis_commutes_algebraically() {
    let checker = algebra_only_checker();
    let phi = Parameter::symbol("phi");
    let result = checker.check(
        &Instruction::Standard(StandardGate::RXY),
        &[Qubit::new(0)],
        &[Parameter::symbol("a"), phi.clone()],
        &Instruction::Standard(StandardGate::RXY),
        &[Qubit::new(0)],
        &[Parameter::symbol("b"), phi],
    );

    assert_exact(result);
}

#[test]
fn rxy_pi_orthogonal_axes_returns_global_phase() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::RXY),
        &[Qubit::new(0)],
        &[Parameter::from(PI), Parameter::from(0.0)],
        &Instruction::Standard(StandardGate::RXY),
        &[Qubit::new(0)],
        &[Parameter::from(PI), Parameter::from(FRAC_PI_2)],
    );

    assert_pi_phase(result);
}

#[test]
fn controlled_axis_target_rotation_commutes_without_rule_or_matrix() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::CRX),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a")],
        &Instruction::Standard(StandardGate::RX),
        &[Qubit::new(1)],
        &[Parameter::symbol("b")],
    );

    assert_exact(result);
}

#[test]
fn controlled_axis_wrong_target_axis_is_conservative() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::CX),
        &[Qubit::new(0), Qubit::new(1)],
        &[],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(1)],
        &[Parameter::symbol("theta")],
    );

    assert!(result.is_none());
}

#[test]
fn ccx_control_and_target_axis_commute_algebraically() {
    let checker = algebra_only_checker();
    let control_result = checker.check(
        &Instruction::Standard(StandardGate::CCX),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        &[],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(1)],
        &[Parameter::symbol("theta")],
    );
    let target_result = checker.check(
        &Instruction::Standard(StandardGate::CCX),
        &[Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        &[],
        &Instruction::Standard(StandardGate::RX),
        &[Qubit::new(2)],
        &[Parameter::symbol("theta")],
    );

    assert_exact(control_result);
    assert_exact(target_result);
}

#[test]
fn fsim_family_commutes_on_same_pair() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::FSIM),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a"), Parameter::symbol("b")],
        &Instruction::Standard(StandardGate::FSIM),
        &[Qubit::new(1), Qubit::new(0)],
        &[Parameter::symbol("c"), Parameter::symbol("d")],
    );

    assert_exact(result);
}

#[test]
fn fsim_commutes_with_symmetric_diagonal_family() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::FSIM),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a"), Parameter::symbol("b")],
        &Instruction::Standard(StandardGate::RZZ),
        &[Qubit::new(1), Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );

    assert_exact(result);
}

#[test]
fn fsim_with_single_rz_is_conservative() {
    let checker = algebra_only_checker();
    let result = checker.check(
        &Instruction::Standard(StandardGate::FSIM),
        &[Qubit::new(0), Qubit::new(1)],
        &[Parameter::symbol("a"), Parameter::symbol("b")],
        &Instruction::Standard(StandardGate::RZ),
        &[Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );

    assert!(result.is_none());
}

#[test]
fn swap_commutes_with_symmetric_pauli_interaction_but_not_rzx() {
    let checker = algebra_only_checker();
    let symmetric = checker.check(
        &Instruction::Standard(StandardGate::SWAP),
        &[Qubit::new(0), Qubit::new(1)],
        &[],
        &Instruction::Standard(StandardGate::RXX),
        &[Qubit::new(1), Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );
    let asymmetric = checker.check(
        &Instruction::Standard(StandardGate::SWAP),
        &[Qubit::new(0), Qubit::new(1)],
        &[],
        &Instruction::Standard(StandardGate::RZX),
        &[Qubit::new(1), Qubit::new(0)],
        &[Parameter::symbol("theta")],
    );

    assert_exact(symmetric);
    assert!(asymmetric.is_none());
}

#[test]
fn symbolic_u_with_x_is_conservative() {
    let checker = CommutationChecker::builtin();
    let result = checker.check(
        &Instruction::Standard(StandardGate::U),
        &[Qubit::new(0)],
        &[
            Parameter::symbol("theta"),
            Parameter::symbol("phi"),
            Parameter::symbol("lambda"),
        ],
        &Instruction::Standard(StandardGate::X),
        &[Qubit::new(0)],
        &[],
    );

    assert!(result.is_none());
}

#[test]
fn matrix_fallback_respects_max_qubits() {
    let checker = CommutationChecker::with_config(CommutationConfig {
        max_matrix_qubits: 4,
        ..CommutationConfig::default()
    });
    let identity = Array2::<Complex64>::eye(32);
    let wide_identity = UnitaryGate::new("WideIdentity", 5, 0)
        .with_matrix(identity)
        .unwrap();
    let result = checker.check(
        &Instruction::UnitaryGate(Box::new(wide_identity)),
        &[
            Qubit::new(0),
            Qubit::new(1),
            Qubit::new(2),
            Qubit::new(3),
            Qubit::new(4),
        ],
        &[],
        &Instruction::Standard(StandardGate::H),
        &[Qubit::new(0)],
        &[],
    );

    assert!(result.is_none());
}
