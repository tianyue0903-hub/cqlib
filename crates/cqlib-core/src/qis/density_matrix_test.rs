// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::Circuit;
use crate::qis::DensityMatrix;
use approx::assert_relative_eq;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_from_state_normalization() {
    // Correctly normalized |1> state: [0.0, 1.0]
    let state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let dm = DensityMatrix::from_state(1, state);

    assert_relative_eq!(dm.data[0].re, 0.0);
    assert_relative_eq!(dm.data[3].re, 1.0);
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
#[should_panic(expected = "Initial state must be normalized")]
fn test_from_state_not_normalized() {
    let state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
    let _ = DensityMatrix::from_state(1, state);
}

#[test]
fn test_probabilities() {
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0);

    let probs = dm.probabilities();
    assert_eq!(probs.len(), 2);
    assert_relative_eq!(probs[0], 0.5);
    assert_relative_eq!(probs[1], 0.5);
}

#[test]
fn test_rz_phase_correction() {
    // Before fix, apply_rz(PI) applied PI/2.
    // Now apply_rz(PI) applies PI, which should flip the sign of off-diagonals perfectly.
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0); // |+> state -> [[0.5, 0.5], [0.5, 0.5]]

    // Apply RZ(PI). |+> goes to |->.
    // Density matrix for |-> is [[0.5, -0.5], [-0.5, 0.5]]
    dm.apply_rz(0, PI);

    assert_relative_eq!(dm.data[0].re, 0.5); // |0><0|
    assert_relative_eq!(dm.data[1].re, -0.5); // |0><1| (flat index 1)
    assert_relative_eq!(dm.data[2].re, -0.5); // |1><0| (flat index 2)
    assert_relative_eq!(dm.data[3].re, 0.5); // |1><1|
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_from_circuit_bell_state() {
    let mut circuit = Circuit::new(2);
    circuit.h(0.into());
    circuit.cx(0.into(), 1.into());

    let dm = DensityMatrix::from_circuit(&circuit).unwrap();

    // Equivalent manual preparation
    let mut dm_manual = DensityMatrix::new(2);
    dm_manual.apply_h(0);
    dm_manual.apply_cx(0, 1);

    for i in 0..16 {
        assert_relative_eq!(dm.data[i].re, dm_manual.data[i].re);
        assert_relative_eq!(dm.data[i].im, dm_manual.data[i].im);
    }
}

#[test]
fn test_ccx_gate() {
    // Prepare |110> state
    let mut dm = DensityMatrix::new(3);
    dm.apply_x(0); // ctrl 1
    dm.apply_x(1); // ctrl 2
    // target 2 is 0

    // Apply CCX
    dm.apply_ccx(0, 1, 2);

    let probs = dm.probabilities();
    // |111> is index 7 (in 0,1,2 little-endian mapping, 1*1 + 1*2 + 1*4 = 7)
    assert_relative_eq!(probs[7], 1.0);
}

#[test]
fn test_swap_gate() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_x(0); // state |10>

    dm.apply_swap(0, 1); // should become |01>

    let probs = dm.probabilities();
    assert_relative_eq!(probs[1], 0.0); // |10> -> 0
    assert_relative_eq!(probs[2], 1.0); // |01> -> 1
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_from_density_matrix_state() {
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0); // |0><0|
    state[3] = Complex64::new(0.5, 0.0); // |1><1|
    let dm = DensityMatrix::from_density_matrix_state(1, state);
    assert_relative_eq!(dm.trace().re, 1.0);
    assert_relative_eq!(dm.probabilities()[0], 0.5);
    assert_relative_eq!(dm.probabilities()[1], 0.5);
}

#[test]
#[should_panic(expected = "Density matrix must have trace 1")]
fn test_from_density_matrix_state_invalid_trace() {
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    let _ = DensityMatrix::from_density_matrix_state(1, state);
}

#[test]
fn test_partial_trace_bell_state() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0);
    dm.apply_cx(0, 1);

    // Tracing out qubit 1 should leave qubit 0 in a maximally mixed state: I/2.
    let reduced_dm = dm.partial_trace(&[0]);
    assert_eq!(reduced_dm.num_qubits, 1);

    let probs = reduced_dm.probabilities();
    assert_relative_eq!(probs[0], 0.5);
    assert_relative_eq!(probs[1], 0.5);

    // Off-diagonals should be 0.
    assert_relative_eq!(reduced_dm.data[1].re, 0.0);
    assert_relative_eq!(reduced_dm.data[1].im, 0.0);
    assert_relative_eq!(reduced_dm.data[2].re, 0.0);
    assert_relative_eq!(reduced_dm.data[2].im, 0.0);
}

#[test]
#[should_panic(expected = "Qubit index out of bounds in partial trace")]
fn test_partial_trace_out_of_bounds() {
    let dm = DensityMatrix::new(2);
    let _ = dm.partial_trace(&[2]);
}

#[test]
fn test_partial_trace_duplicate_qubits() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0);
    dm.apply_cx(0, 1);

    // Should behave same as &[0] due to deduplication
    let reduced_dm = dm.partial_trace(&[0, 0]);
    assert_eq!(reduced_dm.num_qubits, 1);
    assert_relative_eq!(reduced_dm.probabilities()[0], 0.5);
}

#[test]
fn test_apply_kraus_bit_flip() {
    // Bit flip channel: E(rho) = (1-p) rho + p X rho X
    let p: f64 = 0.3;
    let k0 = vec![
        Complex64::new((1.0 - p).sqrt(), 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new((1.0 - p).sqrt(), 0.0),
    ];
    let k1 = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(p.sqrt(), 0.0),
        Complex64::new(p.sqrt(), 0.0),
        Complex64::new(0.0, 0.0),
    ];

    let mut dm = DensityMatrix::new(1);
    // Initial state |0><0|
    dm.apply_kraus(&[k0, k1], &[0]);

    let probs = dm.probabilities();
    assert_relative_eq!(probs[0], 1.0 - p); // |0><0| probability
    assert_relative_eq!(probs[1], p); // |1><1| probability
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_zeros_and_add_assign() {
    let mut dm1 = DensityMatrix::zeros(1);
    let mut dm2 = DensityMatrix::zeros(1);
    dm1.data[0] = Complex64::new(0.5, 0.0);
    dm2.data[3] = Complex64::new(0.5, 0.0);

    dm1 += dm2;
    assert_relative_eq!(dm1.trace().re, 1.0);
    assert_relative_eq!(dm1.probabilities()[0], 0.5);
    assert_relative_eq!(dm1.probabilities()[1], 0.5);
}

#[test]
fn test_single_qubit_gates() {
    let mut dm = DensityMatrix::new(1);

    // Test X gate
    dm.apply_x(0);
    assert_relative_eq!(dm.probabilities()[1], 1.0);

    // Test Y gate (|1> -> -i|0>) => density matrix is |0><0|
    dm.apply_y(0);
    assert_relative_eq!(dm.probabilities()[0], 1.0);

    // Test Z gate (|0> -> |0>)
    dm.apply_z(0);
    assert_relative_eq!(dm.probabilities()[0], 1.0);

    // Test S gate
    dm.apply_h(0);
    dm.apply_s(0);
    // |+> -> (|0> + i|1>)/sqrt(2)
    // dm.data[1] is |0><1| = (1/sqrt(2)) * (-i/sqrt(2)) = -0.5 i
    // dm.data[2] is |1><0| = (i/sqrt(2)) * (1/sqrt(2)) = 0.5 i
    assert_relative_eq!(dm.data[1].im, -0.5);
    assert_relative_eq!(dm.data[2].im, 0.5);
}

#[test]
fn test_two_qubit_gates_cz() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0);
    dm.apply_h(1);

    dm.apply_cz(0, 1);
    // State is |++> -> (|00> + |01> + |10> - |11>)/2
    let probs = dm.probabilities();
    for p in probs {
        assert_relative_eq!(p, 0.25);
    }

    // Check off-diagonal, e.g., |00><11|
    // row = 00 = 0, col = 11 = 3 -> index = 0 * 4 + 3 = 3
    // value = 0.5 * -0.5 = -0.25
    assert_relative_eq!(dm.data[3].re, -0.25);
}
