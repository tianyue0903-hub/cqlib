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

use crate::circuit::gate::gate_matrix;
use crate::compiler::transform::decompose::two_qubit_kak::kak_decompose;
use ndarray::linalg::kron;
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

const RECONSTRUCTION_EPS: f64 = 1e-7;
const CARTAN_EPS: f64 = 1e-7;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn assert_cartan_close(
    matrix: &ndarray::Array2<Complex64>,
    expected_a: f64,
    expected_b: f64,
    expected_c: f64,
) {
    let decomp = kak_decompose(matrix).unwrap_or_else(|e| panic!("KAK failed: {e}"));
    assert!(
        (decomp.a - expected_a).abs() < CARTAN_EPS,
        "a={} expected {}",
        decomp.a,
        expected_a
    );
    assert!(
        (decomp.b - expected_b).abs() < CARTAN_EPS,
        "b={} expected {}",
        decomp.b,
        expected_b
    );
    assert!(
        (decomp.c - expected_c).abs() < CARTAN_EPS,
        "c={} expected {}",
        decomp.c,
        expected_c
    );
}

/// Construct a random SU(2) matrix using Euler angles.
fn random_su2(itheta: f64, iphi: f64, ilambda: f64) -> ndarray::Array2<Complex64> {
    let (st, ct) = (itheta / 2.0).sin_cos();
    let eip = Complex64::from_polar(1.0, iphi);
    let eil = Complex64::from_polar(1.0, ilambda);
    ndarray::array![[ct * eil, -st * eip.conj()], [st * eip, ct * eil.conj()]]
}

/// Small deterministic PRNG for fixed-seed numerical tests.
#[derive(Clone)]
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / SCALE
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    fn complex_normal(&mut self) -> Complex64 {
        Complex64::new(self.normal(), self.normal())
    }
}

/// Full assertion: reconstruction + Weyl chamber + local gates are SU(2).
fn assert_kak_valid(original: &ndarray::Array2<Complex64>) {
    let decomp = kak_decompose(original).unwrap_or_else(|e| panic!("KAK failed: {e}"));

    // 1. Weyl chamber constraints
    assert!(
        decomp.a >= -1e-8 && decomp.a <= FRAC_PI_4 + 1e-8,
        "a = {} not in [0, π/4]",
        decomp.a
    );
    assert!(
        decomp.b >= -1e-8 && decomp.b <= decomp.a + 1e-8,
        "b = {} > a = {}",
        decomp.b,
        decomp.a
    );
    assert!(
        decomp.c.abs() <= decomp.b + 1e-8,
        "|c| = {} > b = {}",
        decomp.c.abs(),
        decomp.b
    );
    if (decomp.a - FRAC_PI_4).abs() < 1e-8 {
        assert!(decomp.c >= -1e-8, "a=π/4 but c={} < 0", decomp.c);
    }

    // 2. Local gates are SU(2) — det ≈ 1
    for (name, m) in [
        ("K1l", &decomp.k1l),
        ("K1r", &decomp.k1r),
        ("K2l", &decomp.k2l),
        ("K2r", &decomp.k2r),
    ] {
        let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
        assert!(
            (det - Complex64::new(1.0, 0.0)).norm() < 1e-8,
            "{name} det={det}, expected 1"
        );
    }

    // 3. Reconstruction
    let k1 = kron(&decomp.k1l, &decomp.k1r);
    let k2 = kron(&decomp.k2l, &decomp.k2r);
    let rxx = gate_matrix::rxx_gate(-2.0 * decomp.a);
    let ryy = gate_matrix::ryy_gate(-2.0 * decomp.b);
    let rzz = gate_matrix::rzz_gate(-2.0 * decomp.c);
    let core = rxx.dot(&ryy.dot(&rzz.dot(&k2)));
    let left = k1.dot(&core);
    let phase = Complex64::from_polar(1.0, decomp.global_phase);
    let reconstructed = left.mapv(|v| phase * v);

    let mut max_diff = 0.0_f64;
    for i in 0..4 {
        for j in 0..4 {
            max_diff = max_diff.max((reconstructed[[i, j]] - original[[i, j]]).norm());
        }
    }
    assert!(
        max_diff < RECONSTRUCTION_EPS,
        "reconstruction error {max_diff} > {RECONSTRUCTION_EPS}"
    );
}

/// Construct a random SU(4) by generating 4 random SU(2) gates and Cartan coordinates,
/// then building U = exp(iφ) (K1l⊗K1r) exp(i(aXX+bYY+cZZ)) (K2l⊗K2r).
fn make_random_su4(seed: u64) -> ndarray::Array2<Complex64> {
    // Simple deterministic pseudo-random from seed
    let f = |s: u64, i: u64| -> f64 {
        let x = s.wrapping_mul(6364136223846793005).wrapping_add(i);
        (x as f64) / (u64::MAX as f64)
    };
    let k1l = random_su2(f(seed, 0) * PI, f(seed, 1) * TAU, f(seed, 2) * TAU);
    let k1r = random_su2(f(seed, 3) * PI, f(seed, 4) * TAU, f(seed, 5) * TAU);
    let k2l = random_su2(f(seed, 6) * PI, f(seed, 7) * TAU, f(seed, 8) * TAU);
    let k2r = random_su2(f(seed, 9) * PI, f(seed, 10) * TAU, f(seed, 11) * TAU);
    let a = f(seed, 12) * FRAC_PI_4;
    let b = f(seed, 13) * a;
    let c = f(seed, 14) * b * (if f(seed, 15) > 0.5 { 1.0 } else { -1.0 });

    let k1 = kron(&k1l, &k1r);
    let k2 = kron(&k2l, &k2r);
    let rxx = gate_matrix::rxx_gate(-2.0 * a);
    let ryy = gate_matrix::ryy_gate(-2.0 * b);
    let rzz = gate_matrix::rzz_gate(-2.0 * c);
    let core = rxx.dot(&ryy.dot(&rzz.dot(&k2)));
    k1.dot(&core)
}

fn cartan_core(a: f64, b: f64, c: f64) -> ndarray::Array2<Complex64> {
    let rxx = gate_matrix::rxx_gate(-2.0 * a);
    let ryy = gate_matrix::ryy_gate(-2.0 * b);
    let rzz = gate_matrix::rzz_gate(-2.0 * c);
    rxx.dot(&ryy.dot(&rzz))
}

fn make_constructed_kak(
    seed: u64,
    a: f64,
    b: f64,
    c: f64,
    global_phase: f64,
) -> ndarray::Array2<Complex64> {
    let f = |s: u64, i: u64| -> f64 {
        let x = s
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493)
            .wrapping_add(i);
        (x as f64) / (u64::MAX as f64)
    };
    let k1l = random_su2(f(seed, 0) * PI, f(seed, 1) * TAU, f(seed, 2) * TAU);
    let k1r = random_su2(f(seed, 3) * PI, f(seed, 4) * TAU, f(seed, 5) * TAU);
    let k2l = random_su2(f(seed, 6) * PI, f(seed, 7) * TAU, f(seed, 8) * TAU);
    let k2r = random_su2(f(seed, 9) * PI, f(seed, 10) * TAU, f(seed, 11) * TAU);

    let k1 = kron(&k1l, &k1r);
    let k2 = kron(&k2l, &k2r);
    let body = k1.dot(&cartan_core(a, b, c).dot(&k2));
    let phase = Complex64::from_polar(1.0, global_phase);
    body.mapv(|value| phase * value)
}

fn make_random_unitary4(seed: u64) -> ndarray::Array2<Complex64> {
    let mut rng = TestRng::new(seed);
    let mut q: ndarray::Array2<Complex64> = ndarray::Array2::zeros((4, 4));

    for col in 0..4 {
        let mut v = [Complex64::new(0.0, 0.0); 4];
        for entry in &mut v {
            *entry = rng.complex_normal();
        }

        for prev_col in 0..col {
            let mut projection = Complex64::new(0.0, 0.0);
            for row in 0..4 {
                projection += q[[row, prev_col]].conj() * v[row];
            }
            for row in 0..4 {
                v[row] -= projection * q[[row, prev_col]];
            }
        }

        let norm = v.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            norm > 1e-12,
            "random unitary generation produced dependent column"
        );
        for row in 0..4 {
            q[[row, col]] = v[row] / norm;
        }
    }

    q
}

fn controlled_phase_matrix(theta: f64) -> ndarray::Array2<Complex64> {
    ndarray::array![
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::from_polar(1.0, theta)
        ]
    ]
}

/// iSWAP gate matrix.
fn iswap_matrix() -> ndarray::Array2<Complex64> {
    ndarray::array![
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    ]
}

/// sqrt(SWAP) gate matrix.
fn sqrt_swap_matrix() -> ndarray::Array2<Complex64> {
    let p = Complex64::new(0.5, 0.5);
    let m = Complex64::new(0.5, -0.5);
    ndarray::array![
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [Complex64::new(0.0, 0.0), p, m, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), m, p, Complex64::new(0.0, 0.0)],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    ]
}

/// sqrt(iSWAP) gate matrix.
fn sqrt_iswap_matrix() -> ndarray::Array2<Complex64> {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    ndarray::array![
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(s2, 0.0),
            Complex64::new(0.0, s2),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, s2),
            Complex64::new(s2, 0.0),
            Complex64::new(0.0, 0.0)
        ],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    ]
}

// ---------------------------------------------------------------------------
// Tests: standard gates
// ---------------------------------------------------------------------------

#[test]
fn test_identity() {
    assert_kak_valid(&ndarray::Array2::eye(4));
}

#[test]
fn test_cx() {
    assert_kak_valid(&gate_matrix::CX_GATE.clone());
}

#[test]
fn test_cz() {
    assert_kak_valid(&gate_matrix::CZ_GATE.clone());
}

#[test]
fn test_swap() {
    assert_kak_valid(&gate_matrix::SWAP_GATE.clone());
}

#[test]
fn test_sqrt_swap() {
    assert_kak_valid(&sqrt_swap_matrix());
}

#[test]
fn test_iswap() {
    assert_kak_valid(&iswap_matrix());
}

#[test]
fn test_sqrt_iswap() {
    assert_kak_valid(&sqrt_iswap_matrix());
}

// ---------------------------------------------------------------------------
// Tests: tensor products (0-CNOT class)
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_hx() {
    assert_kak_valid(&kron(
        &gate_matrix::H_GATE.clone(),
        &gate_matrix::X_GATE.clone(),
    ));
}

#[test]
fn test_tensor_xi() {
    assert_kak_valid(&kron(
        &gate_matrix::X_GATE.clone(),
        &ndarray::Array2::eye(2),
    ));
}

#[test]
fn test_tensor_ix() {
    assert_kak_valid(&kron(
        &ndarray::Array2::eye(2),
        &gate_matrix::X_GATE.clone(),
    ));
}

#[test]
fn test_tensor_hs() {
    let s = gate_matrix::S_GATE.clone();
    assert_kak_valid(&kron(&gate_matrix::H_GATE.clone(), &s));
}

// ---------------------------------------------------------------------------
// Tests: rotation gates with various angles
// ---------------------------------------------------------------------------

#[test]
fn test_rxx_pi4() {
    assert_kak_valid(&gate_matrix::rxx_gate(FRAC_PI_4));
}

#[test]
fn test_ryy_pi4() {
    assert_kak_valid(&gate_matrix::ryy_gate(FRAC_PI_4));
}

#[test]
fn test_rzz_pi4() {
    assert_kak_valid(&gate_matrix::rzz_gate(FRAC_PI_4));
}

#[test]
fn test_rxx_small() {
    assert_kak_valid(&gate_matrix::rxx_gate(0.01));
}

#[test]
fn test_rxx_pi2() {
    assert_kak_valid(&gate_matrix::rxx_gate(FRAC_PI_2));
}

#[test]
fn test_ryy_pi2() {
    assert_kak_valid(&gate_matrix::ryy_gate(FRAC_PI_2));
}

#[test]
fn test_rzz_pi2() {
    assert_kak_valid(&gate_matrix::rzz_gate(FRAC_PI_2));
}

// ---------------------------------------------------------------------------
// Tests: global phase
// ---------------------------------------------------------------------------

#[test]
fn test_global_phase() {
    let phase = Complex64::from_polar(1.0, 0.7);
    let u: ndarray::Array2<Complex64> = ndarray::Array2::eye(4).mapv(|v: Complex64| phase * v);
    assert_kak_valid(&u);
}

// ---------------------------------------------------------------------------
// Tests: error cases
// ---------------------------------------------------------------------------

#[test]
fn test_wrong_size() {
    let m3 = ndarray::Array2::eye(3);
    assert!(kak_decompose(&m3).is_err());
}

#[test]
fn test_non_unitary() {
    let m = ndarray::Array2::from_elem((4, 4), Complex64::new(1.0, 0.0));
    assert!(kak_decompose(&m).is_err());
}

#[test]
fn test_non_finite() {
    let mut m = ndarray::Array2::eye(4);
    m[[0, 0]] = Complex64::new(f64::NAN, 0.0);
    assert!(kak_decompose(&m).is_err());
}

// ---------------------------------------------------------------------------
// Tests: known Cartan coordinates
// ---------------------------------------------------------------------------

#[test]
fn test_cartan_identity() {
    let decomp = kak_decompose(&ndarray::Array2::eye(4)).unwrap();
    assert!(decomp.a.abs() < 1e-8, "identity a={}", decomp.a);
    assert!(decomp.b.abs() < 1e-8, "identity b={}", decomp.b);
    assert!(decomp.c.abs() < 1e-8, "identity c={}", decomp.c);
}

#[test]
fn test_cartan_cx() {
    let decomp = kak_decompose(&gate_matrix::CX_GATE.clone()).unwrap();
    assert!(
        (decomp.a - FRAC_PI_4).abs() < 1e-8,
        "CX a={} expected π/4",
        decomp.a
    );
    assert!(decomp.b.abs() < 1e-8, "CX b={}", decomp.b);
    assert!(decomp.c.abs() < 1e-8, "CX c={}", decomp.c);
}

#[test]
fn test_cartan_cz() {
    let decomp = kak_decompose(&gate_matrix::CZ_GATE.clone()).unwrap();
    assert!(
        (decomp.a - FRAC_PI_4).abs() < 1e-8,
        "CZ a={} expected π/4",
        decomp.a
    );
    assert!(decomp.b.abs() < 1e-8, "CZ b={}", decomp.b);
    assert!(decomp.c.abs() < 1e-8, "CZ c={}", decomp.c);
}

#[test]
fn test_cartan_swap() {
    let decomp = kak_decompose(&gate_matrix::SWAP_GATE.clone()).unwrap();
    assert!(
        (decomp.a - FRAC_PI_4).abs() < 1e-8,
        "SWAP a={} expected π/4",
        decomp.a
    );
    assert!(
        (decomp.b - FRAC_PI_4).abs() < 1e-8,
        "SWAP b={} expected π/4",
        decomp.b
    );
    assert!(
        (decomp.c - FRAC_PI_4).abs() < 1e-8,
        "SWAP c={} expected π/4",
        decomp.c
    );
}

#[test]
fn test_cartan_iswap() {
    let decomp = kak_decompose(&iswap_matrix()).unwrap();
    assert!(
        (decomp.a - FRAC_PI_4).abs() < 1e-8,
        "iSWAP a={} expected π/4",
        decomp.a
    );
    assert!(
        (decomp.b - FRAC_PI_4).abs() < 1e-8,
        "iSWAP b={} expected π/4",
        decomp.b
    );
    assert!(decomp.c.abs() < 1e-8, "iSWAP c={}", decomp.c);
}

#[test]
fn test_cartan_sqrt_swap() {
    let decomp = kak_decompose(&sqrt_swap_matrix()).unwrap();
    assert!(
        (decomp.a - FRAC_PI_4 / 2.0).abs() < CARTAN_EPS,
        "sqrt(SWAP) a={}",
        decomp.a
    );
    assert!(
        (decomp.b - FRAC_PI_4 / 2.0).abs() < CARTAN_EPS,
        "sqrt(SWAP) b={}",
        decomp.b
    );
    assert!(
        (decomp.c.abs() - FRAC_PI_4 / 2.0).abs() < CARTAN_EPS,
        "sqrt(SWAP) c={}",
        decomp.c
    );
}

#[test]
fn test_cartan_rxx_ryy_rzz_generic_angles() {
    assert_cartan_close(&gate_matrix::rxx_gate(0.37), 0.185, 0.0, 0.0);
    assert_cartan_close(&gate_matrix::ryy_gate(-0.41), 0.205, 0.0, 0.0);
    assert_cartan_close(&gate_matrix::rzz_gate(0.23), 0.115, 0.0, 0.0);
}

#[test]
fn test_cartan_controlled_phase() {
    let theta = PI / 3.0;
    assert_cartan_close(&controlled_phase_matrix(theta), theta / 4.0, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Tests: Weyl chamber boundary and near-boundary coordinates
// ---------------------------------------------------------------------------

#[test]
fn test_weyl_boundary_identity() {
    assert_cartan_close(&cartan_core(0.0, 0.0, 0.0), 0.0, 0.0, 0.0);
}

#[test]
fn test_weyl_boundary_cnot_class() {
    assert_cartan_close(&cartan_core(FRAC_PI_4, 0.0, 0.0), FRAC_PI_4, 0.0, 0.0);
}

#[test]
fn test_weyl_boundary_iswap_class() {
    assert_cartan_close(
        &cartan_core(FRAC_PI_4, FRAC_PI_4, 0.0),
        FRAC_PI_4,
        FRAC_PI_4,
        0.0,
    );
}

#[test]
fn test_weyl_boundary_swap_class() {
    assert_cartan_close(
        &cartan_core(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4),
        FRAC_PI_4,
        FRAC_PI_4,
        FRAC_PI_4,
    );
}

#[test]
fn test_weyl_near_equal_a_b() {
    let a = 0.31;
    let b = a - 1e-6;
    let c = 0.12;
    assert_cartan_close(&cartan_core(a, b, c), a, b, c);
}

#[test]
fn test_weyl_near_b_equals_abs_c() {
    let a = 0.53;
    let b = 0.19;
    let c = -(b - 1e-6);
    assert_cartan_close(&cartan_core(a, b, c), a, b, c);
}

#[test]
fn test_weyl_near_zero_c() {
    let a = 0.42;
    let b = 0.17;
    let c = 1e-9;
    assert_cartan_close(&cartan_core(a, b, c), a, b, 0.0);
}

#[test]
fn test_weyl_near_pi_over_four() {
    let a = FRAC_PI_4 - 1e-6;
    let b = 0.11;
    let c = 0.03;
    assert_cartan_close(&cartan_core(a, b, c), a, b, c);
}

// ---------------------------------------------------------------------------
// Tests: random SU(4) via constructed Cartan coordinates
// ---------------------------------------------------------------------------

#[test]
fn test_random_su4_batch() {
    for seed in 0..200 {
        let u = make_random_su4(seed);
        assert_kak_valid(&u);
    }
}

#[test]
fn test_fixed_seed_constructed_kak_property_batch() {
    for seed in 10_000..10_500 {
        let mut rng = TestRng::new(seed);
        let a = 0.02 + rng.next_f64() * (FRAC_PI_4 - 0.04);
        let b = 0.01 + rng.next_f64() * (a - 0.015);
        let c_abs = rng.next_f64() * (b - 0.005);
        let c = if rng.next_f64() < 0.5 { c_abs } else { -c_abs };
        let phase = (rng.next_f64() - 0.5) * TAU;

        let u = make_constructed_kak(seed, a, b, c, phase);
        assert_kak_valid(&u);
        assert_cartan_close(&u, a, b, c);
    }
}

#[test]
fn test_fixed_seed_haar_unitary_property_batch() {
    for seed in 20_000..20_200 {
        let u = make_random_unitary4(seed);
        assert_kak_valid(&u);
    }
}

#[test]
#[ignore = "slow KAK stress test for local production qualification"]
fn test_slow_constructed_kak_property_batch() {
    for seed in 100_000..110_000 {
        let mut rng = TestRng::new(seed);
        let a = 0.02 + rng.next_f64() * (FRAC_PI_4 - 0.04);
        let b = 0.01 + rng.next_f64() * (a - 0.015);
        let c_abs = rng.next_f64() * (b - 0.005);
        let c = if rng.next_f64() < 0.5 { c_abs } else { -c_abs };
        let phase = (rng.next_f64() - 0.5) * TAU;

        let u = make_constructed_kak(seed, a, b, c, phase);
        assert_kak_valid(&u);
        assert_cartan_close(&u, a, b, c);
    }
}

#[test]
#[ignore = "slow KAK stress test for local production qualification"]
fn test_slow_haar_unitary_property_batch() {
    for seed in 200_000..210_000 {
        let u = make_random_unitary4(seed);
        assert_kak_valid(&u);
    }
}

// ---------------------------------------------------------------------------
// Tests: random SU(2)⊗SU(2) (0-CNOT class)
// ---------------------------------------------------------------------------

#[test]
fn test_random_tensor_product() {
    let f = |s: u64, i: u64| -> f64 {
        let x = s.wrapping_mul(6364136223846793005).wrapping_add(i);
        (x as f64) / (u64::MAX as f64)
    };
    for seed in 0..50 {
        let l = random_su2(f(seed, 0) * PI, f(seed, 1) * TAU, f(seed, 2) * TAU);
        let r = random_su2(f(seed, 3) * PI, f(seed, 4) * TAU, f(seed, 5) * TAU);
        let u = kron(&l, &r);
        assert_kak_valid(&u);
        let decomp = kak_decompose(&u).unwrap();
        assert!(decomp.a.abs() < 1e-7, "tensor product a={}", decomp.a);
        assert!(decomp.b.abs() < 1e-7, "tensor product b={}", decomp.b);
        assert!(decomp.c.abs() < 1e-7, "tensor product c={}", decomp.c);
    }
}

// ---------------------------------------------------------------------------
// Tests: boundary cases near degeneracies
// ---------------------------------------------------------------------------

#[test]
fn test_rxx_near_zero() {
    assert_kak_valid(&gate_matrix::rxx_gate(1e-6));
}

#[test]
fn test_rxx_near_pi_over_4() {
    assert_kak_valid(&gate_matrix::rxx_gate(FRAC_PI_4 - 1e-6));
}

#[test]
fn test_rxx_near_pi_over_2() {
    assert_kak_valid(&gate_matrix::rxx_gate(FRAC_PI_2 - 1e-6));
}
