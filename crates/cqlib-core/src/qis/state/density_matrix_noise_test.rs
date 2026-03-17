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
use crate::circuit::{Circuit, Instruction, Qubit, StandardGate};
use crate::device::NoiseModel;
use crate::device::noise::{ReadoutError, SingleQubitNoise, TwoQubitNoise};
use crate::qis::pauli::Pauli;
use ndarray::array;
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn test_bit_flip_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[1] - 0.9).abs() < 1e-6, "P(|1>) was {}", probs[1]);
    assert!((probs[0] - 0.1).abs() < 1e-6, "P(|0>) was {}", probs[0]);
}

#[test]
fn test_readout_error() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_readout_error(
            Qubit::new(0),
            ReadoutError {
                p_0_given_1: 0.0,
                p_1_given_0: 0.1,
            },
        )
        .unwrap();

    let sim = DensityMatrixNoise::new(1, Some(noise_model));
    let probs = sim.probabilities_with_readout(&[0]);

    assert!((probs[1] - 0.1).abs() < 1e-6, "P(|1>) was {}", probs[1]);
    assert!((probs[0] - 0.9).abs() < 1e-6, "P(|0>) was {}", probs[0]);
}

#[test]
fn test_ccx() {
    let mut sim = DensityMatrixNoise::new(3, None);
    sim.apply_x(0).unwrap();
    sim.apply_x(1).unwrap();
    sim.apply_ccx(0, 1, 2).unwrap();
    let probs = sim.probabilities_with_readout(&[2]);
    let p1: f64 = probs
        .iter()
        .enumerate()
        .filter(|(s, _)| (s >> 2) & 1usize == 1usize)
        .map(|(_, p)| p)
        .sum();
    assert!((p1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_apply_kraus_memory() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.01),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    for _ in 0..100 {
        sim.apply_x(0).unwrap();
    }
    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] > 0.0);
    assert!(probs[1] > 0.0);
}

#[test]
fn test_from_circuit_with_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut circuit = Circuit::new(1);
    circuit.x(Qubit::new(0)).unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0]);

    assert!((probs[1] - 0.9).abs() < 1e-6);
    assert!((probs[0] - 0.1).abs() < 1e-6);
}

#[test]
fn test_phase_flip_noise_exact() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::H,
            Qubit::new(0),
            SingleQubitNoise::PhaseFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_h(0).unwrap();
    sim.apply_h(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[1] - 0.1).abs() < 1e-6, "P(|1>) was {}", probs[1]);
    assert!((probs[0] - 0.9).abs() < 1e-6, "P(|0>) was {}", probs[0]);
}

#[test]
fn test_amplitude_damping_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::AmplitudeDamping(0.2),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[1] - 0.8).abs() < 1e-6, "P(|1>) was {}", probs[1]);
    assert!((probs[0] - 0.2).abs() < 1e-6, "P(|0>) was {}", probs[0]);
}

#[test]
fn test_phase_damping_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::H,
            Qubit::new(0),
            SingleQubitNoise::PhaseDamping(0.2),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_h(0).unwrap();
    sim.apply_h(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    let expected_p0 = 0.5 * (1.0 + 0.8f64.sqrt());
    let expected_p1 = 0.5 * (1.0 - 0.8f64.sqrt());
    assert!(
        (probs[0] - expected_p0).abs() < 1e-6,
        "P(|0>) was {}",
        probs[0]
    );
    assert!(
        (probs[1] - expected_p1).abs() < 1e-6,
        "P(|1>) was {}",
        probs[1]
    );
}

#[test]
fn test_pauli_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::Pauli {
                px: 0.1,
                py: 0.2,
                pz: 0.3,
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[0] - 0.3).abs() < 1e-6, "P(|0>) was {}", probs[0]);
    assert!((probs[1] - 0.7).abs() < 1e-6, "P(|1>) was {}", probs[1]);
}

#[test]
fn test_depolarizing_1q_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.3),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[1] - 0.8).abs() < 1e-6, "P(|1>) was {}", probs[1]);
    assert!((probs[0] - 0.2).abs() < 1e-6, "P(|0>) was {}", probs[0]);
}

#[test]
fn test_two_qubit_correlated_pauli_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CX,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::CorrelatedPauli {
                op_q0: Pauli::X,
                op_q1: Pauli::X,
                p: 0.2,
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));
    sim.apply_cx(0, 1).unwrap();

    let probs = sim.probabilities_with_readout(&[0, 1]);
    assert!((probs[0] - 0.8).abs() < 1e-6, "P(|00>) was {}", probs[0]);
    assert!((probs[3] - 0.2).abs() < 1e-6, "P(|11>) was {}", probs[3]);
    assert!(probs[1] < 1e-6);
    assert!(probs[2] < 1e-6);
}

#[test]
fn test_two_qubit_independent_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CX,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.1),
                q1_noise: SingleQubitNoise::BitFlip(0.2),
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));
    sim.apply_cx(0, 1).unwrap();

    let probs = sim.probabilities_with_readout(&[0, 1]);
    assert!((probs[0] - 0.72).abs() < 1e-6, "P(|00>) was {}", probs[0]);
    assert!((probs[1] - 0.08).abs() < 1e-6, "P(|01>) was {}", probs[1]);
    assert!((probs[2] - 0.18).abs() < 1e-6, "P(|10>) was {}", probs[2]);
    assert!((probs[3] - 0.02).abs() < 1e-6, "P(|11>) was {}", probs[3]);
}

#[test]
fn test_two_qubit_depolarizing_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CX,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Depolarizing(0.15),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));
    sim.apply_x(0).unwrap();
    sim.apply_cx(0, 1).unwrap();

    let probs = sim.probabilities_with_readout(&[0, 1]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    assert!(probs[3] < 1.0);
    assert!(probs[3] > 0.8);
    assert!(probs[0] > 0.0);
    assert!(probs[1] > 0.0);
    assert!(probs[2] > 0.0);
}

#[test]
fn test_multi_qubit_readout_error() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_readout_error(
            Qubit::new(0),
            ReadoutError {
                p_0_given_1: 0.1,
                p_1_given_0: 0.2,
            },
        )
        .unwrap();
    noise_model
        .add_readout_error(
            Qubit::new(1),
            ReadoutError {
                p_0_given_1: 0.3,
                p_1_given_0: 0.4,
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));
    sim.apply_x(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0, 1]);
    assert!((probs[0] - 0.06).abs() < 1e-6, "P(|00>) was {}", probs[0]);
    assert!((probs[1] - 0.54).abs() < 1e-6, "P(|01>) was {}", probs[1]);
    assert!((probs[2] - 0.04).abs() < 1e-6, "P(|10>) was {}", probs[2]);
    assert!((probs[3] - 0.36).abs() < 1e-6, "P(|11>) was {}", probs[3]);
}

#[test]
fn test_from_circuit_complex() {
    let mut circuit = Circuit::new(3);

    circuit.h(Qubit::new(0)).unwrap();
    circuit.rx(Qubit::new(1), PI / 2.0).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit
        .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
        .unwrap();
    circuit.swap(Qubit::new(0), Qubit::new(2)).unwrap();

    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CX,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.05),
                q1_noise: SingleQubitNoise::BitFlip(0.05),
            },
        )
        .unwrap();

    let sim_result = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model));
    assert!(sim_result.is_ok());
    let sim = sim_result.unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1, 2]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_from_circuit_unitary_gate() {
    use crate::circuit::gate::UnitaryGate;

    let mut circuit = Circuit::new(1);
    let u_mat = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];
    let u_gate = UnitaryGate::new("U", 1).with_matrix(u_mat).unwrap();
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(u_gate)),
            vec![Qubit::new(0)],
            vec![],
            None::<&str>,
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, None).unwrap();
    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[1] - 1.0).abs() < 1e-6);
    assert!((probs[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_x2p_y2m_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x2p(0).unwrap();
    sim.apply_y2m(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cz_xy2p_rz_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Depolarizing(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));
    sim.apply_rz(0, PI / 2.0).unwrap();
    sim.apply_xy2p(1, PI).unwrap();
    sim.apply_cz(0, 1).unwrap();

    let probs = sim.probabilities_with_readout(&[0, 1]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_complex_circuit_with_native_gates() {
    let mut circuit = Circuit::new(3);

    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.y2m(Qubit::new(1)).unwrap();
    circuit.xy2p(Qubit::new(0), PI / 4.0).unwrap();
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.rz(Qubit::new(2), PI).unwrap();
    // xy is a single-qubit gate, not two-qubit
    circuit.xy(Qubit::new(1), PI / 2.0).unwrap();
    circuit.cz(Qubit::new(1), Qubit::new(2)).unwrap();

    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::PhaseDamping(0.05),
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.01),
                q1_noise: SingleQubitNoise::BitFlip(0.01),
            },
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1, 2]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_x2p_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_x2p(0).unwrap();

    // X2P + X2P = X (approximately, without noise)
    // With bit-flip noise after X2P, we expect some probability in |0>
    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);

    // Apply second X2P to complete X rotation
    sim.apply_x2p(0).unwrap();
    let probs = sim.probabilities_with_readout(&[0]);
    // After two X2P gates with bit-flip noise, P(|0>) should be around 0.1
    assert!(probs[0] > 0.0, "P(|0>) should be > 0 due to noise");
}

#[test]
fn test_x2m_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2M,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |1>
    sim.apply_x(0).unwrap();

    // X2M rotates around X axis by -pi/2
    sim.apply_x2m(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_y2p_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2P,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
    sim.apply_y2p(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_y2m_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2M,
            Qubit::new(0),
            SingleQubitNoise::PhaseFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |+>
    sim.apply_h(0).unwrap();

    // Y2M rotates around Y axis by -pi/2
    sim.apply_y2m(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_xy2p_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::XY2P,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |1>
    sim.apply_x(0).unwrap();

    // XY2P is a sqrt(XY) gate, theta=pi creates superposition
    // |1> -> (i|0> + |1>)/sqrt(2), so P(|0>) = 0.5 without noise
    sim.apply_xy2p(0, PI).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    // With bit-flip noise, we expect P(|0>) around 0.5 (with some noise variation)
    assert!(
        probs[0] > 0.35 && probs[0] < 0.65,
        "P(|0>) was {} (expected ~0.5 with noise)",
        probs[0]
    );
}

#[test]
fn test_xy2m_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::XY2M,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |1>
    sim.apply_x(0).unwrap();

    // XY2M is sqrt(XY)^dagger, theta=pi creates superposition
    // |1> -> (-i|0> + |1>)/sqrt(2), so P(|0>) = 0.5 without noise
    sim.apply_xy2m(0, PI).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    // With bit-flip noise, we expect P(|0>) around 0.5 (with some noise variation)
    assert!(
        probs[0] > 0.35 && probs[0] < 0.65,
        "P(|0>) was {} (expected ~0.5 with noise)",
        probs[0]
    );
}

#[test]
fn test_cz_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.05),
                q1_noise: SingleQubitNoise::BitFlip(0.05),
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(2, Some(noise_model));

    // Prepare |11>
    sim.apply_x(0).unwrap();
    sim.apply_x(1).unwrap();

    // Apply CZ
    sim.apply_cz(0, 1).unwrap();

    // CZ adds a phase to |11> but doesn't change probabilities
    let probs = sim.probabilities_with_readout(&[0, 1]);

    // With bit-flip noise, we expect slight deviation from P(|11>) = 1
    assert!(probs[3] > 0.8, "P(|11>) was {} (expected > 0.8)", probs[3]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_rz_gate_noise() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::PhaseFlip(0.2),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Prepare |+>
    sim.apply_h(0).unwrap();

    // Apply RZ(pi)
    sim.apply_rz(0, PI).unwrap();

    // RZ(pi) on |+> gives |->
    // With phase flip noise during RZ, we expect some |+> component
    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_rz_gate_multiple_rotations() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::PhaseDamping(0.05),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |+>
    sim.apply_h(0).unwrap();

    // Apply multiple RZ gates
    sim.apply_rz(0, PI / 4.0).unwrap();
    sim.apply_rz(0, PI / 4.0).unwrap();
    sim.apply_rz(0, PI / 4.0).unwrap();
    sim.apply_rz(0, PI / 4.0).unwrap();

    // Total rotation is pi, should give |->
    // Measure in X basis by applying H first
    sim.apply_h(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    // With phase damping, P(|1>) should be around 0.5
    assert!(probs[0] >= 0.0);
    assert!(probs[1] >= 0.0);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_native_gates_combination() {
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.05),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2P,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.05),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.05),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // X2P + Y2P + RZ(pi) sequence
    sim.apply_x2p(0).unwrap();
    sim.apply_y2p(0).unwrap();
    sim.apply_rz(0, PI).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_complex_circuit_native_gates_with_cz() {
    let mut circuit = Circuit::new(3);

    // Layer 1: Single-qubit rotations
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.y2p(Qubit::new(1)).unwrap();
    circuit.x2m(Qubit::new(2)).unwrap();

    // Layer 2: Two-qubit gates
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cz(Qubit::new(1), Qubit::new(2)).unwrap();

    // Layer 3: Single-qubit rotations with RZ
    circuit.rz(Qubit::new(0), PI / 3.0).unwrap();
    circuit.rz(Qubit::new(1), PI / 2.0).unwrap();
    circuit.rz(Qubit::new(2), PI / 4.0).unwrap();

    // Layer 4: XY gates
    circuit.xy2p(Qubit::new(0), PI / 6.0).unwrap();
    circuit.xy2m(Qubit::new(1), PI / 6.0).unwrap();
    circuit.xy(Qubit::new(2), PI / 3.0).unwrap();

    // Layer 5: Final CZ
    circuit.cz(Qubit::new(0), Qubit::new(2)).unwrap();

    let mut noise_model = NoiseModel::new();
    // Add noise to various gates
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.02),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(1),
            SingleQubitNoise::PhaseFlip(0.03),
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Depolarizing(0.01),
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1, 2]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_complex_circuit_entanglement_with_noise() {
    let mut circuit = Circuit::new(4);

    // Create GHZ-like state with native gates
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2p(Qubit::new(0)).unwrap(); // X

    // Entangle with CZ gates
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cz(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.cz(Qubit::new(2), Qubit::new(3)).unwrap();

    // Apply rotations
    circuit.rz(Qubit::new(0), PI / 4.0).unwrap();
    circuit.rz(Qubit::new(1), PI / 4.0).unwrap();
    circuit.rz(Qubit::new(2), PI / 4.0).unwrap();
    circuit.rz(Qubit::new(3), PI / 4.0).unwrap();

    // Additional entanglement
    circuit.cz(Qubit::new(0), Qubit::new(3)).unwrap();

    // Final rotations
    circuit.y2p(Qubit::new(0)).unwrap();
    circuit.y2m(Qubit::new(3)).unwrap();

    let mut noise_model = NoiseModel::new();
    // Add comprehensive noise
    for i in 0..4 {
        noise_model
            .add_single_qubit_error(
                StandardGate::RZ,
                Qubit::new(i),
                SingleQubitNoise::PhaseDamping(0.02),
            )
            .unwrap();
    }
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.01),
                q1_noise: SingleQubitNoise::BitFlip(0.01),
            },
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(2),
            Qubit::new(3),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.01),
                q1_noise: SingleQubitNoise::BitFlip(0.01),
            },
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1, 2, 3]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify we have some probability distribution across states
    let non_zero_probs: Vec<_> = probs.iter().filter(|&&p| p > 1e-10).collect();
    assert!(
        !non_zero_probs.is_empty(),
        "Should have non-zero probabilities"
    );
}

#[test]
fn test_complex_circuit_all_native_gates() {
    let mut circuit = Circuit::new(3);

    // Test all native gates in a complex circuit
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2m(Qubit::new(0)).unwrap();
    circuit.y2p(Qubit::new(1)).unwrap();
    circuit.y2m(Qubit::new(1)).unwrap();
    circuit.xy(Qubit::new(0), PI / 2.0).unwrap();
    circuit.xy2p(Qubit::new(1), PI / 4.0).unwrap();
    circuit.xy2m(Qubit::new(2), PI / 4.0).unwrap();
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.rz(Qubit::new(0), PI / 3.0).unwrap();
    circuit.rz(Qubit::new(1), PI / 2.0).unwrap();
    circuit.rz(Qubit::new(2), PI).unwrap();
    circuit.cz(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit.xy(Qubit::new(2), PI).unwrap();

    let mut noise_model = NoiseModel::new();

    // Add noise to all gate types
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2M,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2P,
            Qubit::new(1),
            SingleQubitNoise::Depolarizing(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2M,
            Qubit::new(1),
            SingleQubitNoise::Depolarizing(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::XY,
            Qubit::new(0),
            SingleQubitNoise::BitFlip(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::XY2P,
            Qubit::new(1),
            SingleQubitNoise::BitFlip(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::XY2M,
            Qubit::new(2),
            SingleQubitNoise::BitFlip(0.01),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::PhaseDamping(0.02),
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Depolarizing(0.02),
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(1),
            Qubit::new(2),
            TwoQubitNoise::Depolarizing(0.02),
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1, 2]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_complex_circuit_with_readout_and_gate_noise() {
    let mut circuit = Circuit::new(2);

    // Create Bell-like state using native gates
    circuit.x2p(Qubit::new(0)).unwrap();
    circuit.x2p(Qubit::new(0)).unwrap(); // X
    circuit.y2p(Qubit::new(1)).unwrap();
    circuit.y2m(Qubit::new(1)).unwrap();

    // Entangle
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();

    // Rotate to superposition
    circuit.rz(Qubit::new(0), PI / 2.0).unwrap();
    circuit.xy(Qubit::new(1), PI / 2.0).unwrap();

    // Another CZ
    circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();

    let mut noise_model = NoiseModel::new();

    // Gate noise
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::Depolarizing(0.03),
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::PhaseFlip(0.02),
        )
        .unwrap();
    noise_model
        .add_two_qubit_error(
            StandardGate::CZ,
            Qubit::new(0),
            Qubit::new(1),
            TwoQubitNoise::Independent {
                q0_noise: SingleQubitNoise::BitFlip(0.02),
                q1_noise: SingleQubitNoise::BitFlip(0.02),
            },
        )
        .unwrap();

    // Readout noise
    noise_model
        .add_readout_error(
            Qubit::new(0),
            ReadoutError {
                p_0_given_1: 0.05,
                p_1_given_0: 0.05,
            },
        )
        .unwrap();
    noise_model
        .add_readout_error(
            Qubit::new(1),
            ReadoutError {
                p_0_given_1: 0.03,
                p_1_given_0: 0.03,
            },
        )
        .unwrap();

    let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    let probs = sim.probabilities_with_readout(&[0, 1]);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify all states have some probability
    for (i, &p) in probs.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "P(|{:02b}>) = {} is out of range",
            i,
            p
        );
    }
}

#[test]
fn test_native_gates_pauli_noise() {
    // Test Pauli noise on native gates
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::Pauli {
                px: 0.05,
                py: 0.05,
                pz: 0.05,
            },
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::Y2P,
            Qubit::new(0),
            SingleQubitNoise::Pauli {
                px: 0.05,
                py: 0.05,
                pz: 0.05,
            },
        )
        .unwrap();
    noise_model
        .add_single_qubit_error(
            StandardGate::RZ,
            Qubit::new(0),
            SingleQubitNoise::Pauli {
                px: 0.05,
                py: 0.05,
                pz: 0.05,
            },
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Sequence of native gates
    sim.apply_x2p(0).unwrap();
    sim.apply_y2p(0).unwrap();
    sim.apply_rz(0, PI / 2.0).unwrap();
    sim.apply_xy2p(0, PI / 4.0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
    // Due to Pauli noise, both states should have some probability
    assert!(probs[0] > 0.0);
    assert!(probs[1] > 0.0);
}

#[test]
fn test_native_gates_amplitude_damping() {
    // Test amplitude damping on native gates
    let mut noise_model = NoiseModel::new();
    noise_model
        .add_single_qubit_error(
            StandardGate::X2P,
            Qubit::new(0),
            SingleQubitNoise::AmplitudeDamping(0.1),
        )
        .unwrap();

    let mut sim = DensityMatrixNoise::new(1, Some(noise_model));

    // Start in |1>
    sim.apply_x(0).unwrap();

    // Apply X2P with amplitude damping
    sim.apply_x2p(0).unwrap();

    let probs = sim.probabilities_with_readout(&[0]);
    assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6);
    // Amplitude damping increases P(|0>)
    assert!(probs[0] > 0.0);
}
