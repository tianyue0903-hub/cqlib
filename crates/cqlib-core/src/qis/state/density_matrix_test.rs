// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::{Circuit, Qubit, StandardGate};
use crate::qis::hamiltonian::Hamiltonian;
use crate::qis::pauli::{Pauli, PauliString};
use crate::qis::state::density_matrix::DensityMatrix;
use crate::qis::state::statevector::Statevector;
use approx::assert_relative_eq;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_from_state_normalization() {
    // Correctly normalized |1> state: [0.0, 1.0]
    let state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let dm = DensityMatrix::from_state(1, state).unwrap();

    assert_relative_eq!(dm.data[0].re, 0.0);
    assert_relative_eq!(dm.data[3].re, 1.0);
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_from_state_not_normalized() {
    let state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
    let result = DensityMatrix::from_state(1, state);
    assert!(result.is_err());
}

#[test]
fn test_probabilities() {
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0).unwrap();

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
    dm.apply_h(0).unwrap(); // |+> state -> [[0.5, 0.5], [0.5, 0.5]]

    // Apply RZ(PI). |+> goes to |->.
    // Density matrix for |-> is [[0.5, -0.5], [-0.5, 0.5]]
    dm.apply_rz(0, PI).unwrap();

    assert_relative_eq!(dm.data[0].re, 0.5); // |0><0|
    assert_relative_eq!(dm.data[1].re, -0.5); // |0><1| (flat index 1)
    assert_relative_eq!(dm.data[2].re, -0.5); // |1><0| (flat index 2)
    assert_relative_eq!(dm.data[3].re, 0.5); // |1><1|
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_from_circuit_bell_state() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let dm = DensityMatrix::from_circuit(&circuit).unwrap();

    // Equivalent manual preparation
    let mut dm_manual = DensityMatrix::new(2);
    dm_manual.apply_h(0).unwrap();
    dm_manual.apply_cx(0, 1).unwrap();

    for i in 0..16 {
        assert_relative_eq!(dm.data[i].re, dm_manual.data[i].re);
        assert_relative_eq!(dm.data[i].im, dm_manual.data[i].im);
    }
}

#[test]
fn test_from_circuit_ignores_terminal_measurement_declarations() {
    use crate::circuit::Qubit;
    use crate::device::Outcome;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let out = circuit
        .measure_bits([Qubit::new(1), Qubit::new(0)])
        .unwrap();

    let dm = DensityMatrix::from_circuit(&circuit).unwrap();
    let probs = dm.probs(&out).unwrap();

    assert_eq!(probs.len(), 2);
    assert_relative_eq!(probs[&Outcome::from_bitstring("00").unwrap()], 0.5);
    assert_relative_eq!(probs[&Outcome::from_bitstring("11").unwrap()], 0.5);
}

#[test]
fn test_apply_circuit_bell_state() {
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let mut dm = DensityMatrix::new(2);
    dm.apply_circuit(&circuit).unwrap();

    // Equivalent manual preparation
    let mut dm_manual = DensityMatrix::new(2);
    dm_manual.apply_h(0).unwrap();
    dm_manual.apply_cx(0, 1).unwrap();

    for i in 0..16 {
        assert_relative_eq!(dm.data[i].re, dm_manual.data[i].re);
        assert_relative_eq!(dm.data[i].im, dm_manual.data[i].im);
    }
}

#[test]
fn test_apply_circuit_dimension_mismatch() {
    let mut dm = DensityMatrix::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();

    assert!(dm.apply_circuit(&circuit).is_err());
}

#[test]
fn test_apply_circuit_reset_directive() {
    let mut circuit = Circuit::new(1);
    circuit.x(Qubit::new(0)).unwrap();
    circuit.reset(Qubit::new(0)).unwrap();

    let mut dm = DensityMatrix::new(1);
    dm.apply_circuit(&circuit).unwrap();

    let probs = dm.probabilities();
    assert_relative_eq!(probs[0], 1.0);
    assert_relative_eq!(probs[1], 0.0);
}

#[test]
fn test_apply_circuit_classical_control_flow_error() {
    use crate::circuit::ClassicalExpr;

    let mut circuit = Circuit::new(1);
    circuit
        .if_(ClassicalExpr::bool_literal(true), |body| {
            body.x(Qubit::new(0))?;
            Ok(())
        })
        .unwrap();

    let mut dm = DensityMatrix::new(1);
    assert!(matches!(
        dm.apply_circuit(&circuit),
        Err(crate::qis::QisError::UnsupportedOperation(_))
    ));
}

#[test]
fn test_ccx_gate() {
    // Prepare |110> state
    let mut dm = DensityMatrix::new(3);
    dm.apply_x(0).unwrap(); // ctrl 1
    dm.apply_x(1).unwrap(); // ctrl 2
    // target 2 is 0

    // Apply CCX
    dm.apply_ccx(0, 1, 2).unwrap();

    let probs = dm.probabilities();
    // |111> is index 7 (in 0,1,2 little-endian mapping, 1*1 + 1*2 + 1*4 = 7)
    assert_relative_eq!(probs[7], 1.0);
}

#[test]
fn test_swap_gate() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_x(0).unwrap(); // state |10>

    dm.apply_swap(0, 1).unwrap(); // should become |01>

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
    let dm = DensityMatrix::from_density_matrix_state(1, state).unwrap();
    assert_relative_eq!(dm.trace().re, 1.0);
    assert_relative_eq!(dm.probabilities()[0], 0.5);
    assert_relative_eq!(dm.probabilities()[1], 0.5);
}

#[test]
fn test_from_density_matrix_state_invalid_trace() {
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    let result = DensityMatrix::from_density_matrix_state(1, state);
    assert!(result.is_err());
}

#[test]
fn test_partial_trace_bell_state() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

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
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

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
    dm.apply_kraus(&[k0, k1], &[0]).unwrap();

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
    dm.apply_x(0).unwrap();
    assert_relative_eq!(dm.probabilities()[1], 1.0);

    // Test Y gate (|1> -> -i|0>) => density matrix is |0><0|
    dm.apply_y(0).unwrap();
    assert_relative_eq!(dm.probabilities()[0], 1.0);

    // Test Z gate (|0> -> |0>)
    dm.apply_z(0).unwrap();
    assert_relative_eq!(dm.probabilities()[0], 1.0);

    // Test S gate
    dm.apply_h(0).unwrap();
    dm.apply_s(0).unwrap();
    // |+> -> (|0> + i|1>)/sqrt(2)
    // dm.data[1] is |0><1| = (1/sqrt(2)) * (-i/sqrt(2)) = -0.5 i
    // dm.data[2] is |1><0| = (i/sqrt(2)) * (1/sqrt(2)) = 0.5 i
    assert_relative_eq!(dm.data[1].im, -0.5);
    assert_relative_eq!(dm.data[2].im, 0.5);
}

#[test]
fn test_two_qubit_gates_cz() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_h(1).unwrap();

    dm.apply_cz(0, 1).unwrap();
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

#[test]
fn test_expectation_z_on_zero() {
    // |0⟩ state, ⟨Z⟩ = 1
    let dm = DensityMatrix::new(1);
    let mut ps = PauliString::new(1);
    ps.set_pauli(0, Pauli::Z);
    let h = Hamiltonian::from_pauli(ps);

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, 1.0);
}

#[test]
fn test_expectation_z_on_one() {
    // |1⟩ state, ⟨Z⟩ = -1
    let mut dm = DensityMatrix::new(1);
    dm.apply_x(0).unwrap();

    let mut ps = PauliString::new(1);
    ps.set_pauli(0, Pauli::Z);
    let h = Hamiltonian::from_pauli(ps);

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, -1.0);
}

#[test]
fn test_expectation_x_on_plus() {
    // |+⟩ state, ⟨X⟩ = 1
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0).unwrap();

    let mut ps = PauliString::new(1);
    ps.set_pauli(0, Pauli::X);
    let h = Hamiltonian::from_pauli(ps);

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, 1.0);
}

#[test]
fn test_expectation_multi_qubit() {
    // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    // For H = Z⊗I, ⟨H⟩ = 0
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

    let mut ps = PauliString::new(2);
    ps.set_pauli(0, Pauli::Z); // Z on qubit 0
    let h = Hamiltonian::from_pauli(ps);

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, 0.0, epsilon = 1e-10);
}

#[test]
fn test_expectation_zz_on_bell() {
    // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    // For H = Z⊗Z, ⟨H⟩ = 1
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

    let mut ps = PauliString::new(2);
    ps.set_pauli(0, Pauli::Z);
    ps.set_pauli(1, Pauli::Z);
    let h = Hamiltonian::from_pauli(ps);

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, 1.0);
}

#[test]
fn test_expectation_with_coefficient() {
    // |0⟩ state, ⟨2Z⟩ = 2
    let dm = DensityMatrix::new(1);
    let mut ps = PauliString::new(1);
    ps.set_pauli(0, Pauli::Z);
    let h = Hamiltonian::from_list(vec![(ps, Complex64::new(2.0, 0.0))]).unwrap();

    let exp = dm.expectation(&h).unwrap();
    assert_relative_eq!(exp, 2.0);
}

#[test]
fn test_expectation_qubit_mismatch() {
    let dm = DensityMatrix::new(1);
    let h = Hamiltonian::new(2); // 2 qubit Hamiltonian

    let result = dm.expectation(&h);
    assert!(result.is_err(), "Should error on qubit mismatch");
}

#[test]
fn test_expectation_sv_dm_consistency() {
    // Compare Statevector and DensityMatrix expectation values

    // Create the same state in both simulators
    let mut sv = Statevector::new(2);
    sv.apply_h(0).unwrap();
    sv.apply_cx(0, 1).unwrap();

    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

    // Create Hamiltonian H = X⊗X + Z⊗Z
    let mut ps_xx = PauliString::new(2);
    ps_xx.set_pauli(0, Pauli::X);
    ps_xx.set_pauli(1, Pauli::X);

    let mut ps_zz = PauliString::new(2);
    ps_zz.set_pauli(0, Pauli::Z);
    ps_zz.set_pauli(1, Pauli::Z);

    let h = Hamiltonian::from_list(vec![
        (ps_xx, Complex64::new(1.0, 0.0)),
        (ps_zz, Complex64::new(1.0, 0.0)),
    ])
    .unwrap();

    let exp_sv = sv.expectation(&h).unwrap();
    let exp_dm = dm.expectation(&h).unwrap();

    assert_relative_eq!(exp_sv, exp_dm, epsilon = 1e-10);
}

#[test]
fn test_is_hermitian_valid() {
    // Valid pure state |+><+|
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0).unwrap();
    assert!(dm.is_hermitian(1e-10));
}

#[test]
fn test_is_hermitian_invalid() {
    // Non-Hermitian matrix: |0><1| (no |1><0|)
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[1] = Complex64::new(1.0, 0.0); // |0><1| element
    // Missing |1><0| element - not Hermitian
    state[3] = Complex64::new(1.0, 0.0); // |1><1| to make trace 2

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(!dm.is_hermitian(1e-10));
}

#[test]
fn test_is_hermitian_complex_off_diagonal() {
    // Valid Hermitian with complex off-diagonal: (|0><1| + |1><0|)/2
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[1] = Complex64::new(0.5, 0.0); // |0><1|
    state[2] = Complex64::new(0.5, 0.0); // |1><0| = conjugate of |0><1|
    state[3] = Complex64::new(0.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(dm.is_hermitian(1e-10));
}

#[test]
fn test_is_hermitian_requires_real_diagonal() {
    // Diagonal must be real for Hermitian matrix
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.1); // Complex diagonal - invalid
    state[3] = Complex64::new(0.5, -0.1); // Conjugate won't help

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(!dm.is_hermitian(1e-10));
}

#[test]
fn test_is_positive_semidefinite_valid() {
    // Valid mixed state: I/2
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[3] = Complex64::new(0.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(dm.is_positive_semidefinite_approx(1e-10));
}

#[test]
fn test_is_positive_semidefinite_negative_diagonal() {
    // Negative diagonal element
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(-0.5, 0.0);
    state[3] = Complex64::new(1.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(!dm.is_positive_semidefinite_approx(1e-10));
}

#[test]
fn test_validate_physical_valid_mixed_state() {
    // Valid mixed state: (|0><0| + |1><1|)/2
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[3] = Complex64::new(0.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    assert!(dm.validate_physical(1e-10).is_ok());
}

#[test]
fn test_validate_physical_non_hermitian() {
    // Non-Hermitian matrix should fail validation
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[1] = Complex64::new(0.5, 0.0); // Only |0><1|
    state[3] = Complex64::new(0.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    let result = dm.validate_physical(1e-10);
    assert!(result.is_err());
    // Check it's the right error type
    match result {
        Err(crate::qis::error::QisError::NotHermitian) => (), // Expected
        _ => panic!("Expected NotHermitian error"),
    }
}

#[test]
fn test_validate_physical_not_normalized() {
    // Hermitian and PSD but wrong trace
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(1.0, 0.0);
    state[3] = Complex64::new(1.0, 0.0); // Trace = 2

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };
    let result = dm.validate_physical(1e-10);
    assert!(result.is_err());
}

#[test]
fn test_from_density_matrix_state_valid() {
    // Valid maximally mixed state
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[3] = Complex64::new(0.5, 0.0);

    let result = DensityMatrix::from_density_matrix_state(1, state);
    assert!(result.is_ok());
    let dm = result.unwrap();
    assert_relative_eq!(dm.trace().re, 1.0);
}

#[test]
fn test_from_density_matrix_state_non_hermitian() {
    // Non-Hermitian should be rejected
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[1] = Complex64::new(0.3, 0.2); // Asymmetric
    state[3] = Complex64::new(0.5, 0.0);

    let result = DensityMatrix::from_density_matrix_state(1, state);
    assert!(result.is_err());
}

#[test]
fn test_from_density_matrix_state_bell_state() {
    // Valid Bell state density matrix: |Φ+><Φ+|
    // |Φ+> = (|00> + |11>)/√2
    // ρ = (|00><00| + |00><11| + |11><00| + |11><11|)/2
    let size = 16; // 4^2
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0); // |00><00|
    state[3] = Complex64::new(0.5, 0.0); // |00><11|
    state[12] = Complex64::new(0.5, 0.0); // |11><00|
    state[15] = Complex64::new(0.5, 0.0); // |11><11|

    let result = DensityMatrix::from_density_matrix_state(2, state);
    assert!(result.is_ok());
}

#[test]
fn test_pure_state_density_matrix_is_valid() {
    // |+> state created via gates should be valid
    let mut dm = DensityMatrix::new(1);
    dm.apply_h(0).unwrap();

    // Verify it's valid
    assert!(dm.is_hermitian(1e-10));
    assert!(dm.is_positive_semidefinite_approx(1e-10));
    assert!(dm.validate_physical(1e-10).is_ok());
}

#[test]
fn test_tolerance_handling() {
    // Matrix that is "almost" Hermitian but slightly asymmetric
    let size = 4;
    let mut state = vec![Complex64::new(0.0, 0.0); size];
    state[0] = Complex64::new(0.5, 0.0);
    state[1] = Complex64::new(0.5 + 1e-8, 0.0); // Slightly different from conjugate of [2]
    state[2] = Complex64::new(0.5, 0.0);
    state[3] = Complex64::new(0.5, 0.0);

    let dm = DensityMatrix {
        data: state,
        num_qubits: 1,
    };

    // Should pass with loose tolerance (difference 1e-8 < 1e-7)
    assert!(dm.is_hermitian(1e-7));
    // Should fail with strict tolerance (difference 1e-8 > 1e-10)
    assert!(!dm.is_hermitian(1e-10));
}

#[test]
fn test_apply_standard_gate_arity_mismatch() {
    let mut dm = DensityMatrix::new(2);
    let result = dm.apply_standard_gate(StandardGate::CX, &[0], &[]);
    assert!(
        result.is_err(),
        "CX with 1 qubit should fail, got {:?}",
        result
    );
}

#[test]
fn test_apply_standard_gate_params_mismatch() {
    let mut dm = DensityMatrix::new(1);
    let result = dm.apply_standard_gate(StandardGate::RX, &[0], &[]);
    assert!(
        result.is_err(),
        "RX without parameters should fail, got {:?}",
        result
    );
}

#[test]
fn test_measure_out_of_bounds() {
    let mut dm = DensityMatrix::new(2);
    let result = dm.measure(99);
    assert!(
        result.is_err(),
        "measure on out-of-bounds qubit should fail, got {:?}",
        result
    );
}

#[test]
fn test_measure_deterministic_zero_and_one() {
    let mut zero = DensityMatrix::new(1);
    assert_eq!(zero.measure(0).unwrap(), false);
    assert_relative_eq!(zero.data[0].re, 1.0);
    assert_relative_eq!(zero.data[3].re, 0.0);

    let mut one = DensityMatrix::new(1);
    one.apply_x(0).unwrap();
    assert_eq!(one.measure(0).unwrap(), true);
    assert_relative_eq!(one.data[0].re, 0.0);
    assert_relative_eq!(one.data[3].re, 1.0);
}

#[test]
fn test_measure_plus_statistics_and_collapse() {
    let shots = 1000;
    let mut ones = 0;
    for _ in 0..shots {
        let mut dm = DensityMatrix::new(1);
        dm.apply_h(0).unwrap();
        let outcome = dm.measure(0).unwrap();
        if outcome {
            ones += 1;
            assert!(dm.data[0].norm() < 1e-10);
            assert_relative_eq!(dm.data[3].re, 1.0);
        } else {
            assert!((dm.data[0].re - 1.0).abs() < 1e-10);
            assert!(dm.data[3].norm() < 1e-10);
        }
    }
    assert!(
        (350..=650).contains(&ones),
        "|+> measurement should be near 50/50; ones={ones}/{shots}"
    );
}

#[test]
fn test_measure_all_bit_packing() {
    let mut dm = DensityMatrix::new(4);
    dm.apply_x(0).unwrap();
    dm.apply_x(2).unwrap();
    let outcome = dm.measure_all();
    assert!(outcome.is_one(0));
    assert!(!outcome.is_one(1));
    assert!(outcome.is_one(2));
    assert!(!outcome.is_one(3));
}

#[test]
fn test_sample_shots_zero_and_bell_distribution() {
    let mut dm = DensityMatrix::new(2);
    dm.apply_h(0).unwrap();
    dm.apply_cx(0, 1).unwrap();

    assert!(dm.sample_shots(0).is_empty());

    let shots = dm.sample_shots(1000);
    assert_eq!(shots.len(), 1000);
    let ones = shots
        .iter()
        .filter(|outcome| outcome.is_one(0) && outcome.is_one(1))
        .count();
    assert!(
        shots
            .iter()
            .all(|outcome| outcome.is_one(0) == outcome.is_one(1))
    );
    assert!(
        (350..=650).contains(&ones),
        "Bell samples should be near 50/50 between 00 and 11; ones={ones}"
    );
}

#[test]
fn test_sample_uses_measurement_qubit_order() {
    use crate::circuit::Qubit;
    use crate::device::{Outcome, Status};

    let mut circuit = Circuit::new(2);
    circuit.x(Qubit::new(0)).unwrap();
    let out = circuit
        .measure_bits([Qubit::new(1), Qubit::new(0)])
        .unwrap();

    let dm = DensityMatrix::from_circuit(&circuit).unwrap();
    let result = dm.sample(&out, 16).unwrap();

    assert_eq!(result.shots(), 16);
    assert_eq!(result.num_qubits(), 2);
    assert_eq!(result.qubits(), &vec![Qubit::new(1), Qubit::new(0)]);
    assert_eq!(result.status(), &Status::Completed);
    assert_eq!(
        result.counts().get(&Outcome::from_bitstring("10").unwrap()),
        Some(&16)
    );
    assert_eq!(
        result.probabilities().as_ref().unwrap()[&Outcome::from_bitstring("10").unwrap()],
        1.0
    );
}

#[test]
fn test_probs_marginalizes_unmeasured_qubits() {
    use crate::device::Outcome;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let out = circuit.measure(Qubit::new(0)).unwrap();

    let dm = DensityMatrix::from_circuit(&circuit).unwrap();
    let probs = dm.probs(&out).unwrap();

    assert_eq!(probs.len(), 2);
    assert_relative_eq!(probs[&Outcome::from_bitstring("0").unwrap()], 0.5);
    assert_relative_eq!(probs[&Outcome::from_bitstring("1").unwrap()], 0.5);
}

#[test]
fn test_sample_rejects_measurement_qubit_outside_state() {
    use crate::circuit::Circuit;

    let dm = DensityMatrix::new(1);
    let mut circuit = Circuit::new(3);
    let measurement = circuit.measure(Qubit::new(2)).unwrap();

    assert!(matches!(
        dm.sample(&measurement, 1),
        Err(crate::qis::QisError::IndexOutOfBounds { index: 2, max: 0 })
    ));
}

#[test]
fn test_zero_qubit_initialization_and_measure_boundary() {
    let mut dm = DensityMatrix::new(0);
    assert_eq!(dm.num_qubits, 0);
    assert_eq!(dm.data.len(), 1);
    assert_relative_eq!(dm.data[0].re, 1.0);
    assert_eq!(dm.probabilities(), vec![1.0]);
    assert!(dm.measure(0).is_err());
    dm.apply_gphase(std::f64::consts::PI);
}
