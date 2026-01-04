// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::prelude::*;
use num_complex::Complex;
use std::f64::consts::FRAC_1_SQRT_2;
use std::sync::LazyLock;

const ZERO: Complex<f64> = Complex::new(0., 0.);
const ONE: Complex<f64> = Complex::new(1., 0.);
const I: Complex<f64> = Complex::new(0., 1.);

// Common Constants
// 1/sqrt(2) * (1+i)
const EXP_I_PI_4: Complex<f64> = Complex::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
// 1/sqrt(2) * (1-i)
const EXP_NEG_I_PI_4: Complex<f64> = Complex::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2);
// const HALF: Complex<f64> = Complex::new(0.5, 0.);
// const HALF_I: Complex<f64> = Complex::new(0., 0.5);

// H = 1/sqrt(2) * [[1, 1], [1, -1]]
pub static H_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ONE], [ONE, -ONE]] * FRAC_1_SQRT_2);

// I = [[1, 0], [0, 1]]
pub static I_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, ONE]]);

// iSWAP = [
//     [1, 0, 0, 0],
//     [0, 0, i, 0],
//     [0, i, 0, 0],
//     [0, 0, 0, 1]
// ]
pub static ISWAP_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ZERO, I, ZERO],
        [ZERO, I, ZERO, ZERO],
        [ZERO, ZERO, ZERO, ONE]
    ]
});

// S = [[1, 0], [0, i]]
pub static S_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, I]]);

// SDG = S† = [[1, 0], [0, -i]]
pub static SDG_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, -I]]);

// SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
pub static SWAP_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ZERO, ONE, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ZERO, ONE]
    ]
});

// T = [[1, 0], [0, exp(i*pi/4)]]
pub static T_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, EXP_I_PI_4]]);

// TDG = T† = [[1, 0], [0, exp(-i*pi/4)]]
pub static TDG_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, EXP_NEG_I_PI_4]]);

// X = [[0, 1], [1, 0]]
pub static X_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ZERO, ONE], [ONE, ZERO]]);

// X2P (sqrt(X)) =  1/sqrt(2) * [[1, -i], [-i, 1]]
pub static X2P_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, -I], [-I, ONE]] * FRAC_1_SQRT_2);

// X2M (sqrt(X)†) = 1/sqrt(2) * [[1, i], [i, 1]]
pub static X2M_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, I], [I, ONE]] * FRAC_1_SQRT_2);

// Y = [[0, -i], [i, 0]]
pub static Y_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| array![[ZERO, -I], [I, ZERO]]);

// Y2P (sqrt(Y)) = 1/sqrt(2) * [[1, -1], [1, 1]]
pub static Y2P_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, -ONE], [ONE, ONE]] * FRAC_1_SQRT_2);

// Y2M (sqrt(Y)†) = 1/sqrt(2) * [[1, 1], [-1, 1]]
pub static Y2M_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ONE], [-ONE, ONE]] * FRAC_1_SQRT_2);

// Z = [[1, 0], [0, -1]]
pub static Z_GATE: LazyLock<Array2<Complex<f64>>> =
    LazyLock::new(|| array![[ONE, ZERO], [ZERO, -ONE]]);

// CX (CNOT) = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
pub static CX_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ZERO, ONE],
        [ZERO, ZERO, ONE, ZERO]
    ]
});

// CCX (Toffoli)
pub static CCX_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    let mut m = Array2::eye(8);
    // Swap last two diagonal elements to implement X on target if 11
    // Basis: 000, 001, 010, 011, 100, 101, 110, 111
    // 110 -> 111, 111 -> 110
    // Indices: 6 and 7
    m[[6, 6]] = ZERO;
    m[[6, 7]] = ONE;
    m[[7, 6]] = ONE;
    m[[7, 7]] = ZERO;
    m
});

// CY(Control-Y) = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -i], [0, 0, i, 0]]
pub static CY_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ZERO, -I],
        [ZERO, ZERO, I, ZERO]
    ]
});

// CZ(Control-Z) = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
pub static CZ_GATE: LazyLock<Array2<Complex<f64>>> = LazyLock::new(|| {
    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ONE, ZERO],
        [ZERO, ZERO, ZERO, -ONE]
    ]
});

// RX = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
#[inline]
pub fn rx_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let neg_i_sin = Complex::new(0.0, -sin_val);
    array![[cos, neg_i_sin], [neg_i_sin, cos]]
}

// RXX(theta) = [
//     [cos(theta/2), 0, 0, -i*sin(theta/2)],
//     [0, cos(theta/2), -i*sin(theta/2), 0],
//     [0, -i*sin(theta/2), cos(theta/2), 0],
//     [-i*sin(theta/2), 0, 0, cos(theta/2)]
// ]
#[inline]
pub fn rxx_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let neg_i_sin = Complex::new(0.0, -sin_val);
    array![
        [cos, ZERO, ZERO, neg_i_sin],
        [ZERO, cos, neg_i_sin, ZERO],
        [ZERO, neg_i_sin, cos, ZERO],
        [neg_i_sin, ZERO, ZERO, cos]
    ]
}

// RXY(theta, phi) = [
//     [cos(theta/2), -i*exp(-i*phi)*sin(theta/2)],
//     [-i*exp(i*phi)*sin(theta/2), cos(theta/2)]
// ]
#[inline]
pub fn rxy_gate(theta: f64, phi: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let sin = Complex::new(sin_val, 0.0);

    let (phi_sin, phi_cos) = phi.sin_cos();
    let exp_neg_i_phi = Complex::new(phi_cos, -phi_sin);
    let exp_i_phi = Complex::new(phi_cos, phi_sin);

    // -i * exp(-i*phi) * sin(theta/2)
    // = -i * (cos(phi) - i*sin(phi)) * sin(theta/2)
    // = (-i*cos(phi) - sin(phi)) * sin(theta/2)
    let term1 = -I * exp_neg_i_phi * sin;

    // -i * exp(i*phi) * sin(theta/2)
    // = -i * (cos(phi) + i*sin(phi)) * sin(theta/2)
    // = (-i*cos(phi) + sin(phi)) * sin(theta/2)
    let term2 = -I * exp_i_phi * sin;

    array![[cos, term1], [term2, cos]]
}

// RY = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
#[inline]
pub fn ry_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let sin = Complex::new(sin_val, 0.0);
    let neg_sin = -sin;
    array![[cos, neg_sin], [sin, cos]]
}

// RYY = [
//     [cos(theta/2), 0, 0, i*sin(theta/2)],
//     [0, cos(theta/2), -i*sin(theta/2), 0],
//     [0, -i*sin(theta/2), cos(theta/2), 0],
//     [i*sin(theta/2), 0, 0, cos(theta/2)]
// ]
#[inline]
pub fn ryy_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let i_sin = Complex::new(0.0, sin_val);
    let neg_i_sin = -i_sin;
    array![
        [cos, ZERO, ZERO, i_sin],
        [ZERO, cos, neg_i_sin, ZERO],
        [ZERO, neg_i_sin, cos, ZERO],
        [i_sin, ZERO, ZERO, cos]
    ]
}

// RZ = [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]
#[inline]
pub fn rz_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let exp_neg = Complex::new(cos_val, -sin_val);
    let exp_pos = Complex::new(cos_val, sin_val);
    array![[exp_neg, ZERO], [ZERO, exp_pos]]
}

// RZX = [
//     [cos(theta/2), -i*sin(theta/2), 0, 0],
//     [-i*sin(theta/2), cos(theta/2), 0, 0],
//     [0, 0, cos(theta/2), i*sin(theta/2)],
//     [0, 0, i*sin(theta/2), cos(theta/2)]
// ]
#[inline]
pub fn rzx_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let i_sin = Complex::new(0.0, sin_val);
    let neg_i_sin = -i_sin;
    array![
        [cos, neg_i_sin, ZERO, ZERO],
        [neg_i_sin, cos, ZERO, ZERO],
        [ZERO, ZERO, cos, i_sin],
        [ZERO, ZERO, i_sin, cos]
    ]
}

// RZZ = [
//     [exp(-i*theta/2), 0, 0, 0],
//     [0, exp(i*theta/2), 0, 0],
//     [0, 0, exp(i*theta/2), 0],
//     [0, 0, 0, exp(-i*theta/2)]
// ]
#[inline]
pub fn rzz_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let exp_neg = Complex::new(cos_val, -sin_val);
    let exp_pos = Complex::new(cos_val, sin_val);
    array![
        [exp_neg, ZERO, ZERO, ZERO],
        [ZERO, exp_pos, ZERO, ZERO],
        [ZERO, ZERO, exp_pos, ZERO],
        [ZERO, ZERO, ZERO, exp_neg]
    ]
}

// Phase(lambda) = [[1, 0], [0, exp(i*lambda)]]
#[inline]
pub fn phase_gate(lambda: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = lambda.sin_cos();
    let exp_i_lambda = Complex::new(cos_val, sin_val);
    array![[ONE, ZERO], [ZERO, exp_i_lambda]]
}

// GlobalPhase(theta) = [[exp(i*theta), 0], [0, exp(i*theta)]]
#[inline]
pub fn global_phase_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = theta.sin_cos();
    let exp_i_theta = Complex::new(cos_val, sin_val);
    array![[exp_i_theta, ZERO], [ZERO, exp_i_theta]]
}

// U(theta, phi, lambda) = [
//     [cos(theta/2), -exp(i*lambda)*sin(theta/2)],
//     [exp(i*phi)*sin(theta/2), exp(i*(phi+lambda))*cos(theta/2)]
// ]
#[inline]
pub fn u_gate(theta: f64, phi: f64, lambda: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let sin = Complex::new(sin_val, 0.0);

    let (phi_sin, phi_cos) = phi.sin_cos();
    let exp_i_phi = Complex::new(phi_cos, phi_sin);

    let (lam_sin, lam_cos) = lambda.sin_cos();
    let exp_i_lambda = Complex::new(lam_cos, lam_sin);

    let (pl_sin, pl_cos) = (phi + lambda).sin_cos();
    let exp_i_phi_lambda = Complex::new(pl_cos, pl_sin);

    array![
        [cos, -exp_i_lambda * sin],
        [exp_i_phi * sin, exp_i_phi_lambda * cos]
    ]
}

// XY(theta) = [[0, -i*exp(-i*theta)], [-i*exp(i*theta), 0]]
#[inline]
pub fn xy_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = theta.sin_cos();
    let exp_i_theta = Complex::new(cos_val, sin_val);
    let exp_neg_i_theta = Complex::new(cos_val, -sin_val);

    let term1 = -I * exp_neg_i_theta;
    let term2 = -I * exp_i_theta;

    array![[ZERO, term1], [term2, ZERO]]
}

// XY2P(theta) = 1/sqrt(2) * [[1, -i*exp(-i*theta)], [-i*exp(i*theta), 1]]
#[inline]
pub fn xy2p_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = theta.sin_cos();
    let exp_i_theta = Complex::new(cos_val, sin_val);
    let exp_neg_i_theta = Complex::new(cos_val, -sin_val);

    let term1 = -I * exp_neg_i_theta * FRAC_1_SQRT_2;
    let term2 = -I * exp_i_theta * FRAC_1_SQRT_2;
    let diag = Complex::new(FRAC_1_SQRT_2, 0.0);

    array![[diag, term1], [term2, diag]]
}

// XY2M(theta) = 1/sqrt(2) * [[1, i*exp(-i*theta)], [i*exp(i*theta), 1]]
#[inline]
pub fn xy2m_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = theta.sin_cos();
    let exp_i_theta = Complex::new(cos_val, sin_val);
    let exp_neg_i_theta = Complex::new(cos_val, -sin_val);

    let term1 = I * exp_neg_i_theta * FRAC_1_SQRT_2;
    let term2 = I * exp_i_theta * FRAC_1_SQRT_2;
    let diag = Complex::new(FRAC_1_SQRT_2, 0.0);

    array![[diag, term1], [term2, diag]]
}

// CRX(theta) = [
//     [1, 0, 0, 0],
//     [0, 1, 0, 0],
//     [0, 0, cos(theta/2), -i*sin(theta/2)],
//     [0, 0, -i*sin(theta/2), cos(theta/2)]
// ]
#[inline]
pub fn crx_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let neg_i_sin = Complex::new(0.0, -sin_val);

    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, cos, neg_i_sin],
        [ZERO, ZERO, neg_i_sin, cos]
    ]
}

// CRY(theta) = [
//     [1, 0, 0, 0],
//     [0, 1, 0, 0],
//     [0, 0, cos(theta/2), -sin(theta/2)],
//     [0, 0, sin(theta/2), cos(theta/2)]
// ]
#[inline]
pub fn cry_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let cos = Complex::new(cos_val, 0.0);
    let sin = Complex::new(sin_val, 0.0);
    let neg_sin = -sin;

    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, cos, neg_sin],
        [ZERO, ZERO, sin, cos]
    ]
}

// CRZ(theta) = [
//     [1, 0, 0, 0],
//     [0, 1, 0, 0],
//     [0, 0, exp(-i*theta/2), 0],
//     [0, 0, 0, exp(i*theta/2)]
// ]
#[inline]
pub fn crz_gate(theta: f64) -> Array2<Complex<f64>> {
    let (sin_val, cos_val) = (theta / 2.0).sin_cos();
    let exp_neg = Complex::new(cos_val, -sin_val);
    let exp_pos = Complex::new(cos_val, sin_val);

    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, exp_neg, ZERO],
        [ZERO, ZERO, ZERO, exp_pos]
    ]
}

// fSim(theta, phi) = [
//     [1, 0, 0, 0],
//     [0, cos(theta), -i*sin(theta), 0],
//     [0, -i*sin(theta), cos(theta), 0],
//     [0, 0, 0, exp(-i*phi)]
// ]
#[inline]
pub fn fsim_gate(theta: f64, phi: f64) -> Array2<Complex<f64>> {
    let (sin_theta, cos_theta) = theta.sin_cos();
    let cos_t = Complex::new(cos_theta, 0.0);
    let neg_i_sin_t = Complex::new(0.0, -sin_theta);

    let (sin_phi, cos_phi) = phi.sin_cos();
    let exp_neg_i_phi = Complex::new(cos_phi, -sin_phi);

    array![
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, cos_t, neg_i_sin_t, ZERO],
        [ZERO, neg_i_sin_t, cos_t, ZERO],
        [ZERO, ZERO, ZERO, exp_neg_i_phi]
    ]
}

#[cfg(test)]
#[path = "./gate_matrix_test.rs"]
mod gate_matrix_test;
