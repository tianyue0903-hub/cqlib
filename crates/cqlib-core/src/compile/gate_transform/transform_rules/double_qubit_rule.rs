use crate::circuit::gate::StandardGate;
use smallvec::{SmallVec, smallvec};

use std::f64::consts::PI;

/// Represents a decomposed two-qubit gate with qubit indices.
/// Each gate in `gates` has corresponding qubit indices in `qubits`.
/// For single-qubit gates: one index (applied to that qubit)
/// For two-qubit gates: two indices [control, target] or [qubit0, qubit1]
#[derive(Debug, Clone)]
pub struct DecomposedTwoQubitGate {
    pub gates: Vec<(StandardGate, SmallVec<[f64; 3]>)>,
    pub qubits: Vec<Vec<i32>>,
}

impl DecomposedTwoQubitGate {
    pub fn new() -> Self {
        DecomposedTwoQubitGate {
            gates: Vec::new(),
            qubits: Vec::new(),
        }
    }

    pub fn push_single(&mut self, gate: StandardGate, params: SmallVec<[f64; 3]>, qubit: i32) {
        self.gates.push((gate, params));
        self.qubits.push(vec![qubit]);
    }

    pub fn push_two(
        &mut self,
        gate: StandardGate,
        params: SmallVec<[f64; 3]>,
        qubit0: i32,
        qubit1: i32,
    ) {
        self.gates.push((gate, params));
        self.qubits.push(vec![qubit0, qubit1]);
    }
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
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::RZ, smallvec![-PI / 2.0], 0);
        result.push_single(StandardGate::RZ, smallvec![-PI / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![PI / 2.0], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = CX · RZ(θ)(t) · CX
    pub fn rzz2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![theta], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);

        result
    }

    /// CX = H(1) · FSIM(0, π) · H(1)
    pub fn cx2fsim_rule(
        _gate: &StandardGate,
        _parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::FSIM, smallvec![0.0, PI], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// FSIM(θ, φ) decomposition into CX-based sequence.
    pub fn fsim2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_two(StandardGate::CX, smallvec![], 1, 0);
        result.push_single(StandardGate::RZ, smallvec![-theta], 0);
        result.push_two(StandardGate::CX, smallvec![], 1, 0);
        result.push_single(StandardGate::RZ, smallvec![theta], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::Phase, smallvec![-phi / 2.0], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::Phase, smallvec![phi / 2.0], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![-phi / 2.0], 0);

        result
    }

    /// RZZ(θ) = H(1) · FSIM(0, π) · RX(θ)(1) · FSIM(0, π) · H(1)
    pub fn rzz2fsim_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::FSIM, smallvec![0.0, PI], 0, 1);
        result.push_single(StandardGate::RX, smallvec![theta], 1);
        result.push_two(StandardGate::FSIM, smallvec![0.0, PI], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// FSIM(θ, φ) decomposition into RZZ-based sequence.
    pub fn fsim2rzz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 0);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![-PI], 0);
        result.push_single(StandardGate::RZ, smallvec![-PI / 2.0], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![-PI], 1);
        result.push_single(StandardGate::RY, smallvec![-phi / 2.0], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![PI / 2.0], 0, 1);
        result.push_single(StandardGate::RZ, smallvec![-PI / 2.0], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::Phase, smallvec![phi / 2.0], 1);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![PI / 2.0], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RY, smallvec![-PI / 2.0], 0);
        result.push_single(StandardGate::RZ, smallvec![-phi / 2.0], 0);

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
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::S, smallvec![], 1);
        result.push_two(StandardGate::CY, smallvec![], 0, 1);
        result.push_single(StandardGate::SDG, smallvec![], 1);

        result
    }

    /// CY = S(t) · CX · S†(t)
    pub fn cy2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::SDG, smallvec![], 1);
        result.push_two(StandardGate::CX, smallvec![], 0, 1);
        result.push_single(StandardGate::S, smallvec![], 1);

        result
    }

    /// CX = H(t) · CZ · H(t)
    pub fn cx2cz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::CZ, smallvec![], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// CZ = H(t) · CX · H(t)
    pub fn cz2cx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let mut result = DecomposedTwoQubitGate::new();

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
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RXX, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result
    }

    /// RXX(θ) = H(0) · H(1) · RZZ(θ) · H(1) · H(0)
    pub fn rxx2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::H, smallvec![], 0);
        result
    }

    /// RZZ(θ) = RX(-π/2)(0) · RX(-π/2)(1) · RYY(θ) · RX(π/2)(1) · RX(π/2)(0)
    pub fn rzz2ryy_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 0);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 1);
        result.push_two(StandardGate::RYY, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 1);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 0);

        result
    }

    /// RYY(θ) = RX(π/2)(0) · RX(π/2)(1) · RZZ(θ) · RX(-π/2)(1) · RX(-π/2)(0)
    pub fn ryy2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 0);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 1);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 0);

        result
    }

    /// RZZ(θ) = H(1) · RZX(θ) · H(1)
    pub fn rzz2rzx_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZX, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZX(θ) = H(1) · RZZ(θ) · H(1)
    pub fn rzx2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · CRZ(-2θ)
    pub fn rzz2crz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_two(StandardGate::CRZ, smallvec![theta * -2.0], 0, 1);

        result
    }

    /// CRZ(θ) = RZ(θ/2)(1) · RZZ(-θ/2)
    pub fn crz2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · H(1) · CRX(-2θ) · H(1)
    /// (Derived from CRX = H(t) · CRZ · H(t) and RZZ → CRZ rule)
    pub fn rzz2crx_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_two(StandardGate::CRX, smallvec![theta * -2.0], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// CRX(θ) = H(1) · RZ(θ/2)(1) · RZZ(-θ/2) · H(1)
    /// (Derived from CRX = H(t) · CRZ · H(t) and CRZ → RZZ rule)
    pub fn crx2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::H, smallvec![], 1);
        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);
        result.push_single(StandardGate::H, smallvec![], 1);

        result
    }

    /// RZZ(θ) = RZ(θ)(1) · RX(-π/2)(1) · CRY(-2θ) · RX(π/2)(1)
    pub fn rzz2cry_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RZ, smallvec![theta.clone()], 1);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 1);
        result.push_two(StandardGate::CRY, smallvec![theta * -2.0], 0, 1);
        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 1);

        result
    }

    /// CRY(θ) = RX(π/2)(1) · RZ(θ/2)(1) · RZZ(-θ/2) · RX(-π/2)(1)
    pub fn cry2rzz_rule(
        gate: &StandardGate,
        parameters: &SmallVec<[f64; 3]>,
    ) -> DecomposedTwoQubitGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedTwoQubitGate::new();

        result.push_single(StandardGate::RX, smallvec![PI / 2.0], 1);
        result.push_single(StandardGate::RZ, smallvec![theta.clone() / 2.0], 1);
        result.push_two(StandardGate::RZZ, smallvec![theta / -2.0], 0, 1);
        result.push_single(StandardGate::RX, smallvec![-PI / 2.0], 1);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use num::complex::Complex;
    use num::complex::ComplexFloat;
    use rand::Rng;
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
        params: &SmallVec<[f64; 3]>,
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
        let gate_mat = gate.matrix(params).to_owned();
        for ii in 0..expand_mat_shape {
            for jj in 0..expand_mat_shape {
                if ii & xor_value == jj & xor_value {
                    expand_mat[[ii, jj]] = gate_mat[[expand_vec[ii], expand_vec[jj]]];
                }
            }
        }

        expand_mat
    }

    fn matrix_from_decomposed_gate(decomposed: &DecomposedTwoQubitGate) -> Array2<Complex<f64>> {
        let mut total_u: Array2<Complex<f64>> = Array2::eye(4);

        // Gates are applied in temporal order (left-to-right in circuit),
        // so we use right-multiplication: U = G1 · G2 · G3 · ...
        for ((gate, params), qubits) in decomposed.gates.iter().zip(decomposed.qubits.iter()) {
            let gate_mat = gate_expand_rust(*gate, params, qubits.clone(), 2);
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
        rule: fn(&StandardGate, &SmallVec<[f64; 3]>) -> DecomposedTwoQubitGate,
        gate: &StandardGate,
        params: &SmallVec<[f64; 3]>,
        rule_name: &str,
    ) {
        let decomposed = rule(gate, params);
        let original_matrix = gate.matrix(params).to_owned();
        let decomposed_matrix = matrix_from_decomposed_gate(&decomposed);

        if test_verbose() {
            println!("Testing rule: {}", rule_name);
            println!("Original gate: {:?}", gate);
            println!("Decomposed gates: {:?}", decomposed.gates.len());
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
                &smallvec![theta],
                "rzz2cx_rule",
            );
        }
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
                &smallvec![theta, phi],
                "fsim2cx_rule",
            );
        }
    }

    #[test]
    fn test_rzz2fsim_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2fsim_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2fsim_rule",
            );
        }
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
                &smallvec![theta, phi],
                "fsim2rzz_rule",
            );
        }
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
                &smallvec![theta],
                "rzz2rxx_rule",
            );
        }
    }

    #[test]
    fn test_rxx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rxx2rzz_rule,
                &StandardGate::RXX,
                &smallvec![theta],
                "rxx2rzz_rule",
            );
        }
    }

    #[test]
    fn test_rzz2ryy_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2ryy_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2ryy_rule",
            );
        }
    }

    #[test]
    fn test_ryy2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::ryy2rzz_rule,
                &StandardGate::RYY,
                &smallvec![theta],
                "ryy2rzz_rule",
            );
        }
    }

    #[test]
    fn test_rzz2rzx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2rzx_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2rzx_rule",
            );
        }
    }

    #[test]
    fn test_rzx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzx2rzz_rule,
                &StandardGate::RZX,
                &smallvec![theta],
                "rzx2rzz_rule",
            );
        }
    }

    #[test]
    fn test_rzz2crz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2crz_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2crz_rule",
            );
        }
    }

    #[test]
    fn test_crz2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::crz2rzz_rule,
                &StandardGate::CRZ,
                &smallvec![theta],
                "crz2rzz_rule",
            );
        }
    }

    #[test]
    fn test_rzz2crx_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2crx_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2crx_rule",
            );
        }
    }

    #[test]
    fn test_crx2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::crx2rzz_rule,
                &StandardGate::CRX,
                &smallvec![theta],
                "crx2rzz_rule",
            );
        }
    }

    #[test]
    fn test_rzz2cry_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::rzz2cry_rule,
                &StandardGate::RZZ,
                &smallvec![theta],
                "rzz2cry_rule",
            );
        }
    }

    #[test]
    fn test_cry2rzz_rule() {
        let mut rng = rand::rng();
        for _ in 0..5 {
            let theta = rng.random_range(-PI..PI);
            assert_rule_decomposition(
                DoubleQubitRule::cry2rzz_rule,
                &StandardGate::CRY,
                &smallvec![theta],
                "cry2rzz_rule",
            );
        }
    }
}
