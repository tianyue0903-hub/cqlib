use crate::circuit::Parameter;
use crate::circuit::gate::StandardGate;
use crate::compile::gate_transform::transform_rules::decomposed_gate::DecomposedGate;
use smallvec::{SmallVec, smallvec};

use std::f64::consts::PI;

fn fixed(value: f64) -> Parameter {
    Parameter::from(value)
}

/// DoubleQubitRule provides transformation rules between two-qubit gates.
///
/// Gates in consideration: RXX, RYY, RZX, RZZ, CX, CY, CZ
///
/// Categories (gates in the same category are equivalent under 1-qubit gates):
/// - CX category (key: CX): CX, CY, CZ
/// - RZZ category (key: RZZ): RXX, RYY, RZX, RZZ
/// - FSim category (key: FSim): FSim
///
/// Following the minimal necessary principle, only rules between:
/// 1. Key gates of different categories (CX <-> RZZ)
/// 2. Key gate and members of the same category
///
/// are defined. Other transformations can be composed from these base rules.
pub struct DoubleQubitRule {
    pub name: String,
}

impl DoubleQubitRule {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    // ========================================================================
    // Rules between categories (CX <-> RZZ, CX <-> FSIM, RZZ <-> FSIM)
    // ========================================================================

    /// CX = (global phase) · H(t) · RZ(-π/2)(c) · RZ(-π/2)(t) · RZZ(π/2) · H(t)
    pub fn cx2rzz_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::RZ, smallvec![fixed(-PI / 2.0)], 0);
        result.push_single(StandardGate::RZ, smallvec![fixed(-PI / 2.0)], 1);
        result.push_two(StandardGate::RZZ, smallvec![fixed(PI / 2.0)], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = CX · RZ(θ)(t) · CX
    pub fn rzz2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![theta], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);

        result
    }

    /// CX = H(1) · FSIM(0, π) · H(1)
    pub fn cx2fsim_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::FSIM, smallvec![fixed(0.0), fixed(PI)], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// FSIM(θ, φ) decomposition into CX-based sequence.
    pub fn fsim2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let mut result = DecomposedGate::new();

        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_two(StandardGate::CX, smallvec![], 1, 0);
        result.push_single(StandardGate::RZ, smallvec![-1.0 * theta.clone()], 0);
        result.push_two(StandardGate::CX, smallvec![], 1, 0);
        result.push_single(StandardGate::RZ, smallvec![theta], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::Phase, smallvec![phi.clone() / -2.0], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::Phase, smallvec![phi.clone() / 2.0], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![phi / -2.0], 0);

        result
    }

    /// RZZ(θ) = H(1) · FSIM(0, π) · RX(θ)(1) · FSIM(0, π) · H(1)
    pub fn rzz2fsim_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::FSIM, smallvec![fixed(0.0), fixed(PI)], 0, 1);
        result.push_single(StandardGate::RX, smallvec![theta], 1);
        result.push_two(StandardGate::FSIM, smallvec![fixed(0.0), fixed(PI)], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// FSIM(θ, φ) decomposition into RZZ-based sequence.
    pub fn fsim2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta.clone()], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 0);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI)], 0);
        result.push_single(StandardGate::RZ, smallvec![fixed(-PI / 2.0)], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI)], 1);
        result.push_single(StandardGate::RY, smallvec![phi.clone() / -2.0], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![fixed(PI / 2.0)], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![fixed(-PI / 2.0)], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::Phase, smallvec![phi.clone() / 2.0], 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![fixed(PI / 2.0)], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RY, smallvec![fixed(-PI / 2.0)], 0);
        result.push_single(StandardGate::RZ, smallvec![phi / -2.0], 0);

        result
    }

    // ========================================================================
    // Rules within CX category (CX, CY, CZ)
    // Only rules between key gate (CX) and members are defined.
    // ========================================================================

    /// CX = S†(t) · CY · S(t)
    /// (Derived from CY = S(t) · CX · S†(t))
    pub fn cx2cy_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::S, smallvec![], 1);
        result.push_two(StandardGate::CY, smallvec![], 0, 1);
        result.push_single(StandardGate::SDG, smallvec![], 1);

        result
    }

    /// CY = S(t) · CX · S†(t)
    pub fn cy2cx_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::SDG, smallvec![], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::S, smallvec![], 1);

        result
    }

    /// CX = H(t) · CZ · H(t)
    pub fn cx2cz_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::CZ, smallvec![], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// CZ = H(t) · CX · H(t)
    pub fn cz2cx_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    // ========================================================================
    // Rules within RZZ category (RXX, RYY, RZX, RZZ)
    // Only rules between key gate (RZZ) and members are defined.
    // ========================================================================

    /// RZZ(θ) = H(0) · H(1) · RXX(θ) · H(0) · H(1)
    pub fn rzz2rxx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RXX, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result
    }

    /// RXX(θ) = H(0) · H(1) · RZZ(θ) · H(1) · H(0)
    pub fn rxx2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result
    }

    /// RZZ(θ) = RX(-π/2)(0) · RX(-π/2)(1) · RYY(θ) · RX(π/2)(1) · RX(π/2)(0)
    pub fn rzz2ryy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 0);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 1);
        result.push_two(StandardGate::RYY, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 1);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 0);

        result
    }

    /// RYY(θ) = RX(π/2)(0) · RX(π/2)(1) · RZZ(θ) · RX(-π/2)(1) · RX(-π/2)(0)
    pub fn ryy2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 0);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 0);

        result
    }

    /// RZZ(θ) = H(1) · RZX(θ) · H(1)
    pub fn rzz2rzx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZX, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZX(θ) = H(1) · RZZ(θ) · H(1)
    pub fn rzx2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · CRZ(-2θ)
    pub fn rzz2crz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_two(StandardGate::CRZ, smallvec![-2.0 * theta], 0, 1);

        result
    }

    /// CRZ(θ) = RZ(θ/2)(1) · RZZ(-θ/2)
    pub fn crz2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · H(1) · CRX(-2θ) · H(1)
    /// (Derived from CRX = H(t) · CRZ · H(t) and RZZ → CRZ rule)
    pub fn rzz2crx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::CRX, smallvec![-2.0 * theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// CRX(θ) = H(1) · RZ(θ/2)(1) · RZZ(-θ/2) · H(1)
    /// (Derived from CRX = H(t) · CRZ · H(t) and CRZ → RZZ rule)
    pub fn crx2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · RX(-π/2)(1) · CRY(-2θ) · RX(π/2)(1)
    pub fn rzz2cry_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 1);
        result.push_two(StandardGate::CRY, smallvec![-2.0 * theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 1);

        result
    }

    /// CRY(θ) = RX(π/2)(1) · RZ(θ/2)(1) · RZZ(-θ/2) · RX(-π/2)(1)
    pub fn cry2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();

        result.push_single(StandardGate::RX, smallvec![fixed(PI / 2.0)], 1);
        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);
        result.push_single(StandardGate::RX, smallvec![fixed(-PI / 2.0)], 1);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Parameter;
    use ndarray::prelude::*;
    use num::complex::Complex;
    use num::complex::ComplexFloat;
    use rand::Rng;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    fn complex_inner_product(vec1: &[Complex<f64>], vec2: &[Complex<f64>]) -> Complex<f64> {
        vec1.iter()
            .zip(vec2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    fn is_matrix_differ_by_phase(
        matrix1: &Array2<Complex<f64>>,
        matrix2: &Array2<Complex<f64>>,
    ) -> bool {
        let vec1: Vec<Complex<f64>> = matrix1.iter().copied().collect();
        let vec2: Vec<Complex<f64>> = matrix2.iter().copied().collect();
        let inner: Complex<f64> = complex_inner_product(&vec1, &vec2);
        let inner_abs: f64 = inner.abs();
        let vec1_norm: f64 = complex_inner_product(&vec1, &vec1).re.sqrt();
        let vec2_norm: f64 = complex_inner_product(&vec2, &vec2).re.sqrt();

        let cos_vec = inner_abs / (vec1_norm * vec2_norm);
        (cos_vec - 1.0).abs() < 1e-10
    }

    fn gate_expand_rust(
        gate: StandardGate,
        params: &SmallVec<[Parameter; 3]>,
        gate_qubits: Vec<i32>,
        expand_qubits: i32,
    ) -> Array2<Complex<f64>> {
        // assert!(gate.control_num + gate.target_num == 1_i32, "Only support expand single qubit gates into two qubit gates.");
        let expand_mat_shape = 1 << expand_qubits;
        let mut xor_value = expand_mat_shape - 1;
        let mut expand_mat: Array2<Complex<f64>> =
            Array2::<Complex<f64>>::zeros((expand_mat_shape, expand_mat_shape));

        let gq_len: i32 = gate_qubits.len() as i32;
        assert!(
            gq_len == gate_qubits.len() as i32,
            "The given gate qubits must equal to gate's require."
        );
        for gq in &gate_qubits {
            assert!(
                *gq >= 0 && *gq < expand_qubits,
                "The given gate_qubits must be positive and less than expand qubits."
            );
            xor_value ^= 1 << (expand_qubits - 1 - gq);
        }

        let mut expand_vec: Array1<usize> = Array1::zeros(expand_mat_shape);
        for i in 0..expand_mat_shape {
            let mut nowi: usize = 0;
            for (gq_idx, gq) in gate_qubits.iter().enumerate() {
                let k: i32 = expand_qubits - 1 - gq;
                if (1 << k) & i != 0 {
                    nowi += 1 << (gq_len - 1 - gq_idx as i32);
                }
            }
            expand_vec[i] = nowi;
        }

        // let gate_mat: &Array2<Complex<f64>> = &gate.matrix_rust(bindings);
        let gate_params: SmallVec<[f64; 3]> =
            params.iter().map(|p| p.evaluate(&None).unwrap()).collect();
        let gate_mat = gate
            .matrix(&gate_params)
            .expect("two-qubit gate matrix should be well-formed")
            .to_owned();
        for ii in 0..expand_mat_shape {
            for jj in 0..expand_mat_shape {
                if ii & xor_value == jj & xor_value {
                    expand_mat[[ii, jj]] = gate_mat[[expand_vec[ii], expand_vec[jj]]];
                }
            }
        }

        expand_mat
    }

    fn matrix_from_decomposed_gate(decomposed: &DecomposedGate) -> Array2<Complex<f64>> {
        let mut total_u: Array2<Complex<f64>> = Array2::eye(4);

        // Gates are applied in temporal order (left-to-right in circuit),
        // so we use right-multiplication: U = G1 · G2 · G3 · ...
        for op in &decomposed.ops {
            let gate_mat = gate_expand_rust(op.gate, &op.params, op.qubits.iter().copied().collect(), 2);
            total_u = gate_mat.dot(&total_u);
        }
        total_u
    }

    fn test_verbose() -> bool {
        std::env::var("CQLIB_TEST_VERBOSE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    }

    fn assert_rule_decomposition(
        rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
        rule_name: &str,
    ) {
        let decomposed = rule(gate, params);
        let gate_params: SmallVec<[f64; 3]> =
            params.iter().map(|p| p.evaluate(&None).unwrap()).collect();
        let original_matrix = gate
            .matrix(&gate_params)
            .expect("two-qubit gate matrix should be well-formed")
            .to_owned();
        let decomposed_matrix = matrix_from_decomposed_gate(&decomposed);

        if test_verbose() {
            println!("Testing rule: {}", rule_name);
            println!("Original gate: {:?}", gate);
            println!("Decomposed gates: {:?}", decomposed.ops.len());
            println!("Original matrix:\n{:?}", original_matrix);
            println!("Decomposed matrix:\n{:?}", decomposed_matrix);
        }

        assert!(
            is_matrix_differ_by_phase(&original_matrix, &decomposed_matrix),
            "Rule {} failed for gate {:?}",
            rule_name,
            gate
        );
    }

    fn special_pi_over_8_angles() -> Vec<f64> {
        (0..16).map(|k| k as f64 * PI / 8.0).collect()
    }

    fn assert_rule_special_angles_1(
        rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
        gate: StandardGate,
        rule_name: &str,
    ) {
        for theta in special_pi_over_8_angles() {
            assert_rule_decomposition(
                rule,
                &gate,
                &smallvec![Parameter::from(theta)],
                rule_name,
            );
        }
    }

    fn assert_rule_special_angles_2(
        rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
        gate: StandardGate,
        rule_name: &str,
    ) {
        let special_angles = special_pi_over_8_angles();
        for theta in &special_angles {
            for phi in &special_angles {
                assert_rule_decomposition(
                    rule,
                    &gate,
                    &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                    rule_name,
                );
            }
        }
    }

    // ========================================================================
    // Tests for rules between categories
    // ========================================================================

    #[test]
    fn test_cx2rzz_rule() {
        let gate = StandardGate::CX;
        assert_rule_decomposition(
            DoubleQubitRule::cx2rzz_rule,
            &gate,
            &smallvec![],
            "cx2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2cx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2cx_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2cx_rule",
            );
        }
        assert_rule_special_angles_1(DoubleQubitRule::rzz2cx_rule, StandardGate::RZZ, "rzz2cx_rule");
    }

    #[test]
    fn test_cx2fsim_rule() {
        let gate = StandardGate::CX;
        assert_rule_decomposition(
            DoubleQubitRule::cx2fsim_rule,
            &gate,
            &smallvec![],
            "cx2fsim_rule",
        );
    }

    #[test]
    fn test_fsim2cx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            let phi = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::fsim2cx_rule,
                &StandardGate::FSIM,
                &smallvec![Parameter::from(theta), Parameter::from(phi)],
                "fsim2cx_rule",
            );
        }
        assert_rule_special_angles_2(
            DoubleQubitRule::fsim2cx_rule,
            StandardGate::FSIM,
            "fsim2cx_rule",
        );
    }

    #[test]
    fn test_rzz2fsim_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2fsim_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2fsim_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2fsim_rule,
            StandardGate::RZZ,
            "rzz2fsim_rule",
        );
    }

    #[test]
    fn test_fsim2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            let phi = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::fsim2rzz_rule,
                &StandardGate::FSIM,
                &smallvec![Parameter::from(theta), Parameter::from(phi)],
                "fsim2rzz_rule",
            );
        }
        assert_rule_special_angles_2(
            DoubleQubitRule::fsim2rzz_rule,
            StandardGate::FSIM,
            "fsim2rzz_rule",
        );
    }

    // ========================================================================
    // Tests for rules within CX category
    // ========================================================================

    #[test]
    fn test_cx2cy_rule() {
        let gate = StandardGate::CX;
        assert_rule_decomposition(
            DoubleQubitRule::cx2cy_rule,
            &gate,
            &smallvec![],
            "cx2cy_rule",
        );
    }

    #[test]
    fn test_cy2cx_rule() {
        let gate = StandardGate::CY;
        assert_rule_decomposition(
            DoubleQubitRule::cy2cx_rule,
            &gate,
            &smallvec![],
            "cy2cx_rule",
        );
    }

    #[test]
    fn test_cx2cz_rule() {
        let gate = StandardGate::CX;
        assert_rule_decomposition(
            DoubleQubitRule::cx2cz_rule,
            &gate,
            &smallvec![],
            "cx2cz_rule",
        );
    }

    #[test]
    fn test_cz2cx_rule() {
        let gate = StandardGate::CZ;
        assert_rule_decomposition(
            DoubleQubitRule::cz2cx_rule,
            &gate,
            &smallvec![],
            "cz2cx_rule",
        );
    }

    // ========================================================================
    // Tests for rules within RZZ category
    // ========================================================================

    #[test]
    fn test_rzz2rxx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2rxx_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2rxx_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2rxx_rule,
            StandardGate::RZZ,
            "rzz2rxx_rule",
        );
    }

    #[test]
    fn test_rxx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rxx2rzz_rule,
                &StandardGate::RXX,
                &smallvec![Parameter::from(theta)],
                "rxx2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rxx2rzz_rule,
            StandardGate::RXX,
            "rxx2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2ryy_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2ryy_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2ryy_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2ryy_rule,
            StandardGate::RZZ,
            "rzz2ryy_rule",
        );
    }

    #[test]
    fn test_ryy2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::ryy2rzz_rule,
                &StandardGate::RYY,
                &smallvec![Parameter::from(theta)],
                "ryy2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::ryy2rzz_rule,
            StandardGate::RYY,
            "ryy2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2rzx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2rzx_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2rzx_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2rzx_rule,
            StandardGate::RZZ,
            "rzz2rzx_rule",
        );
    }

    #[test]
    fn test_rzx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzx2rzz_rule,
                &StandardGate::RZX,
                &smallvec![Parameter::from(theta)],
                "rzx2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzx2rzz_rule,
            StandardGate::RZX,
            "rzx2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2crz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2crz_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2crz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2crz_rule,
            StandardGate::RZZ,
            "rzz2crz_rule",
        );
    }

    #[test]
    fn test_crz2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::crz2rzz_rule,
                &StandardGate::CRZ,
                &smallvec![Parameter::from(theta)],
                "crz2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::crz2rzz_rule,
            StandardGate::CRZ,
            "crz2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2crx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2crx_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2crx_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2crx_rule,
            StandardGate::RZZ,
            "rzz2crx_rule",
        );
    }

    #[test]
    fn test_crx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::crx2rzz_rule,
                &StandardGate::CRX,
                &smallvec![Parameter::from(theta)],
                "crx2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::crx2rzz_rule,
            StandardGate::CRX,
            "crx2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2cry_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2cry_rule,
                &StandardGate::RZZ,
                &smallvec![Parameter::from(theta)],
                "rzz2cry_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::rzz2cry_rule,
            StandardGate::RZZ,
            "rzz2cry_rule",
        );
    }

    #[test]
    fn test_cry2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::cry2rzz_rule,
                &StandardGate::CRY,
                &smallvec![Parameter::from(theta)],
                "cry2rzz_rule",
            );
        }
        assert_rule_special_angles_1(
            DoubleQubitRule::cry2rzz_rule,
            StandardGate::CRY,
            "cry2rzz_rule",
        );
    }

    #[test]
    fn test_rzz2crz_rule_preserves_symbolic_parameter() {
        let theta = Parameter::symbol("theta");
        let decomposed = DoubleQubitRule::rzz2crz_rule(&StandardGate::RZZ, &smallvec![theta]);

        assert_eq!(decomposed.ops.len(), 2);
        assert!(decomposed.ops[0].params[0].get_symbols().contains("theta"));
        assert!(decomposed.ops[1].params[0].get_symbols().contains("theta"));

        let mut bindings = HashMap::new();
        bindings.insert("theta", 0.7);

        assert!((decomposed.ops[0].params[0].evaluate(&Some(bindings.clone())).unwrap() - 0.7).abs() < 1e-10);
        assert!((decomposed.ops[1].params[0].evaluate(&Some(bindings)).unwrap() + 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_fsim2cx_rule_preserves_symbolic_parameters() {
        let theta = Parameter::symbol("theta");
        let phi = Parameter::symbol("phi");
        let decomposed =
            DoubleQubitRule::fsim2cx_rule(&StandardGate::FSIM, &smallvec![theta, phi]);

        let symbolic_params: Vec<_> = decomposed
            .ops
            .iter()
            .flat_map(|op| op.params.iter())
            .filter(|param| !param.get_symbols().is_empty())
            .map(|param| param.get_symbols())
            .collect();

        assert!(symbolic_params.iter().any(|symbols| symbols.contains("theta")));
        assert!(symbolic_params.iter().any(|symbols| symbols.contains("phi")));
    }
}
