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
use crate::qis::hamiltonian::Hamiltonian;
use crate::qis::pauli::{PauliString, Phase};
use num_complex::Complex64;
use std::collections::HashMap;

#[test]
fn test_expectation_probs_identity() {
    let mut h = Hamiltonian::new(2);
    let p_id = PauliString::new(2); // +II
    h.add_term(p_id, Complex64::new(2.5, 0.0)).unwrap();

    // measurements can be empty, because identity doesn't require compatible measurements
    let measurements = vec![];
    let exp = h.expectation_probs(&measurements).unwrap();
    assert!((exp - 2.5).abs() < 1e-10);
}

#[test]
fn test_expectation_probs_single_measurement() {
    let mut h = Hamiltonian::new(2);
    let mut p_z = PauliString::new(2);
    p_z.set_pauli(0, crate::qis::pauli::Pauli::Z); // Z on qubit 0 (IZ)
    h.add_term(p_z.clone(), Complex64::new(1.0, 0.0)).unwrap();

    // Measurement base ZZ
    let mut m_zz = PauliString::new(2);
    m_zz.set_pauli(0, crate::qis::pauli::Pauli::Z);
    m_zz.set_pauli(1, crate::qis::pauli::Pauli::Z);

    let mut probs = HashMap::new();
    probs.insert("00".to_string(), 0.5); // both +1
    probs.insert("01".to_string(), 0.5); // q1=0(+1), q0=1(-1)

    let measurements = vec![(m_zz, probs)];

    let exp = h.expectation_probs(&measurements).unwrap();
    // For "00", q0 is '0' (idx 0), parity of q0 is 0. Eigenvalue is 1. Prob 0.5 -> 0.5
    // For "01", q0 is '1' (idx 1), parity of q0 is 1. Eigenvalue is -1. Prob 0.5 -> -0.5
    // Result: 0.0
    assert!((exp - 0.0).abs() < 1e-10);
}

#[test]
fn test_expectation_probs_multiple_terms() {
    let mut h = Hamiltonian::new(2);
    // Term 1: 2.0 * ZI (Z on qubit 1)
    let mut p_z1 = PauliString::new(2);
    p_z1.set_pauli(1, crate::qis::pauli::Pauli::Z);
    h.add_term(p_z1.clone(), Complex64::new(2.0, 0.0)).unwrap();

    // Term 2: 3.0 * IZ (Z on qubit 0)
    let mut p_z0 = PauliString::new(2);
    p_z0.set_pauli(0, crate::qis::pauli::Pauli::Z);
    h.add_term(p_z0.clone(), Complex64::new(3.0, 0.0)).unwrap();

    // Term 3: 4.0 * ZZ
    let mut p_zz = PauliString::new(2);
    p_zz.set_pauli(0, crate::qis::pauli::Pauli::Z);
    p_zz.set_pauli(1, crate::qis::pauli::Pauli::Z);
    h.add_term(p_zz.clone(), Complex64::new(4.0, 0.0)).unwrap();

    // A single ZZ measurement is compatible with all three!
    let mut probs = HashMap::new();
    // "10": q1='1'(Z1=-1), q0='0'(Z0=+1). Parity for Z1 is 1, for Z0 is 0, for ZZ is 1.
    probs.insert("10".to_string(), 1.0);

    let measurements = vec![(p_zz.clone(), probs)];

    let exp = h.expectation_probs(&measurements).unwrap();
    // Term 1 (ZI): Z1 = -1 -> 2.0 * (-1) = -2.0
    // Term 2 (IZ): Z0 = +1 -> 3.0 * (+1) = +3.0
    // Term 3 (ZZ): Z1*Z0 = -1 -> 4.0 * (-1) = -4.0
    // Total: -2.0 + 3.0 - 4.0 = -3.0
    assert!((exp - (-3.0)).abs() < 1e-10);
}

#[test]
fn test_expectation_probs_missing_measurement() {
    let mut h = Hamiltonian::new(1);
    let mut p_x = PauliString::new(1);
    p_x.set_pauli(0, crate::qis::pauli::Pauli::X);
    h.add_term(p_x, Complex64::new(1.0, 0.0)).unwrap();

    // Only provide Z measurement
    let mut m_z = PauliString::new(1);
    m_z.set_pauli(0, crate::qis::pauli::Pauli::Z);
    let mut probs = HashMap::new();
    probs.insert("0".to_string(), 1.0);

    let measurements = vec![(m_z, probs)];
    let res = h.expectation_probs(&measurements);
    assert!(res.is_err());
}

#[test]
fn test_expectation_probs_invalid_state_string() {
    let mut h = Hamiltonian::new(1);
    let mut p_z = PauliString::new(1);
    p_z.set_pauli(0, crate::qis::pauli::Pauli::Z);
    h.add_term(p_z.clone(), Complex64::new(1.0, 0.0)).unwrap();

    let mut probs1 = HashMap::new();
    probs1.insert("00".to_string(), 1.0); // Wrong length
    let res1 = h.expectation_probs(&[(p_z.clone(), probs1)]);
    assert!(res1.is_err());

    let mut probs2 = HashMap::new();
    probs2.insert("2".to_string(), 1.0); // Invalid character
    let res2 = h.expectation_probs(&[(p_z.clone(), probs2)]);
    assert!(res2.is_err());
}

#[test]
fn test_expectation_probs_with_global_phase() {
    let mut h = Hamiltonian::new(1);
    let mut p_z = PauliString::new(1);
    p_z.set_pauli(0, crate::qis::pauli::Pauli::Z);
    p_z.phase = Phase::Minus; // -Z

    // Coeff is 2.0. So total is -2.0 * Z
    h.add_term(p_z.clone(), Complex64::new(2.0, 0.0)).unwrap();

    let mut probs = HashMap::new();
    probs.insert("0".to_string(), 1.0); // Z=1

    let mut m_z = PauliString::new(1);
    m_z.set_pauli(0, crate::qis::pauli::Pauli::Z);

    let exp = h.expectation_probs(&[(m_z, probs)]).unwrap();
    // 2.0 * (-1.0) * (1.0) = -2.0
    assert!((exp - (-2.0)).abs() < 1e-10);
}
