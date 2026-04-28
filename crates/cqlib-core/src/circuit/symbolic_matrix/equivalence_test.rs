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

use super::*;
use crate::circuit::symbolic_matrix::equivalence::{
    circuits_equivalent, symbolic_matrices_equivalent,
};
use crate::circuit::symbolic_matrix::gate::circuit_to_symbolic_matrix;
use crate::circuit::{Circuit, Parameter, Qubit};
use std::f64::consts::PI;

#[test]
fn test_symbolic_matrices_equivalent_up_to_global_phase() {
    let mut rx = Circuit::new(1);
    rx.rx(Qubit::new(0), PI).unwrap();
    let rx_matrix = circuit_to_symbolic_matrix(&rx, None).unwrap();

    let mut x = Circuit::new(1);
    x.x(Qubit::new(0)).unwrap();
    let x_matrix = circuit_to_symbolic_matrix(&x, None).unwrap();

    assert!(symbolic_matrices_equivalent(&rx_matrix, &x_matrix).unwrap());
}

#[test]
fn test_circuits_equivalent_up_to_global_phase() {
    let mut lhs = Circuit::new(1);
    lhs.rz(Qubit::new(0), PI).unwrap();

    let mut rhs = Circuit::new(1);
    rhs.z(Qubit::new(0)).unwrap();
    rhs.set_global_phase(Parameter::from(0.73));

    assert!(circuits_equivalent(&lhs, &rhs, None).unwrap());
}

#[test]
fn test_circuits_not_equivalent_up_to_global_phase() {
    let mut lhs = Circuit::new(1);
    lhs.h(Qubit::new(0)).unwrap();

    let mut rhs = Circuit::new(1);
    rhs.x(Qubit::new(0)).unwrap();

    assert!(!circuits_equivalent(&lhs, &rhs, None).unwrap());
}

#[test]
fn test_phase_equivalence_propagates_qubit_order_error() {
    let lhs = Circuit::new(2);
    let rhs = Circuit::new(2);
    let err = circuits_equivalent(&lhs, &rhs, Some(&[0])).unwrap_err();

    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}
