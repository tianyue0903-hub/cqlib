use crate::circuit::gate::StandardGate;

use ndarray::prelude::*;
use num::complex::Complex;
use num::complex::ComplexFloat;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;

pub struct SingleQubitRule {
    pub name: String,
    rule: fn(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)>,
}

impl SingleQubitRule {
    pub fn new(name: String) -> Self {
        match name.as_str() {
            "u3_rule" => Self {
                name,
                rule: Self::u3_rule,
            },
            "zxz_rule" => Self {
                name,
                rule: Self::zxz_rule,
            },
            "zyz_rule" => Self {
                name,
                rule: Self::zyz_rule,
            },
            "xyx_rule" => Self {
                name,
                rule: Self::xyx_rule,
            },
            "hrz_rule" => Self {
                name,
                rule: Self::hrz_rule,
            },
            "xsxrz_rule" => Self {
                name,
                rule: Self::xsxrz_rule,
            },
            "sxypmrz_rule" => Self {
                name,
                rule: Self::sxypmrz_rule,
            },
            _ => panic!("Invalid single qubit rule name"),
        }
    }

    pub fn execute(
        &self,
        unitary: &Array2<Complex<f64>>,
    ) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        (self.rule)(unitary)
    }

    fn check2pi(theta: f64, eps: f64) -> bool {
        // check if theta is a multiple of 2π
        let multiple: f64 = (theta / (2.0 * PI)).round();

        return (theta - multiple * 2.0 * PI).abs() < eps;
    }

    fn checkpi(theta: f64, eps: f64) -> bool {
        // check if theta is of π + 2kπ
        let theta_mod: f64 = (theta).rem_euclid(2.0 * PI);
        return (theta_mod - PI).abs() < eps;
    }

    fn check_plus_half_pi(theta: f64, eps: f64) -> bool {
        // check if theta is of π/2 + 2kπ
        let theta_mod: f64 = (theta).rem_euclid(2.0 * PI);
        return (theta_mod - PI / 2.0).abs() < eps;
    }

    fn check_minus_half_pi(theta: f64, eps: f64) -> bool {
        // check if theta is of -π/2 + 2kπ
        let theta_mod: f64 = (theta).rem_euclid(2.0 * PI);
        return (theta_mod - 3.0 * PI / 2.0).abs() < eps;
    }

    /// Determinant of a 2×2 complex matrix [[a, b], [c, d]]: det = a*d - b*c.
    fn det_2x2(matrix: &Array2<Complex<f64>>) -> Complex<f64> {
        matrix[(0, 0)] * matrix[(1, 1)] - matrix[(0, 1)] * matrix[(1, 0)]
    }

    fn u2_to_su2(matrix: &mut Array2<Complex<f64>>) -> Complex<f64> {
        /* Modify input single qubit gate in place from U2 to SU2 and return the global phase */
        let phase: Complex<f64> = Self::det_2x2(matrix).sqrt();
        let eps: f64 = 1e-13;

        if (phase - 1.0).abs() > eps {
            matrix.mapv_inplace(|x| x * phase.conj());
        }
        return phase;
    }

    fn close_to_identity(matrix: &Array2<Complex<f64>>) -> bool {
        // check if matrix is close to identity up to a global phase
        let eps: f64 = 1e-13;

        // For U ≈ e^{iφ} I, off-diagonal terms are ~0 and diagonal terms are equal.
        matrix[(0, 1)].norm() < eps
            && matrix[(1, 0)].norm() < eps
            && (matrix[(0, 0)] - matrix[(1, 1)]).norm() < eps
    }

    fn zxz_type_arg(unitary: &Array2<Complex<f64>>) -> (f64, f64, f64, f64, f64) {
        let beta_plus_delta: f64 = 2.0 * unitary[(1, 1)].arg();
        let beta_minus_delta: f64 = 2.0 * unitary[(1, 0)].arg() + PI;
        let gamma_input = Self::clamp_acos(unitary[(0, 0)].abs());
        let gamma: f64 = 2.0 * gamma_input.acos();
        let beta: f64 = (beta_plus_delta + beta_minus_delta) / 2.0;
        let delta: f64 = beta_plus_delta - beta;
        (beta_plus_delta, beta_minus_delta, gamma, beta, delta)
    }

    /// Clamp value to [-1, 1] so acos never receives out-of-range input (avoids NaN).
    fn clamp_acos(x: f64) -> f64 {
        x.clamp(-1.0, 1.0)
    }

    fn zyz_type_arg(unitary: &Array2<Complex<f64>>) -> (f64, f64, f64) {
        let eps: f64 = 1e-13;
        let mut beta_plus_delta: f64 = 0.0;
        let mut beta_minus_delta: f64 = 0.0;
        let gamma: f64;

        if unitary[(0, 0)].abs() > unitary[(0, 1)].abs() && unitary[(0, 1)].abs() > eps {
            let acos_arg = Self::clamp_acos(2.0 * (unitary[(0, 0)] * unitary[(1, 1)]).re - 1.0);
            gamma = Self::clamp_acos(acos_arg).acos();
        } else {
            let acos_arg = Self::clamp_acos(2.0 * (unitary[(0, 1)] * unitary[(1, 0)]).re + 1.0);
            gamma = Self::clamp_acos(acos_arg).acos();
        }

        if unitary[(0, 0)].abs() > eps {
            let cos_half = (gamma / 2.0).cos();
            beta_plus_delta = if cos_half.abs() > eps {
                -(unitary[(0, 0)] / cos_half).arg() * 2.0
            } else {
                0.0
            };
        }
        if unitary[(0, 1)].abs() > eps {
            let sin_half = (gamma / 2.0).sin();
            beta_minus_delta = if sin_half.abs() > eps {
                (unitary[(1, 0)] / sin_half).arg() * 2.0
            } else {
                0.0
            };
        }

        let beta: f64 = (beta_plus_delta + beta_minus_delta) / 2.0;
        let delta: f64 = beta_plus_delta - beta;
        (delta, gamma, beta)
    }

    fn u3_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        /* Deccompose input 2x2 unitary into U3 gate.
         */
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        let eps: f64 = 1e-6;

        if Self::close_to_identity(&unitary) {
            return vec![];
        }

        let angle: f64 = unitary[(0, 0)].arg();
        let z: Complex<f64> = Complex::new(0.0, 1.0 * angle).exp();
        unitary = unitary / z;

        let mut theta: f64 = Self::clamp_acos(unitary[(0, 0)].re).acos();
        let sin_theta: f64 = theta.sin();

        let mut lambda: f64 = 0.0;
        let mut phi: f64;

        if sin_theta.abs() >= eps {
            lambda = (unitary[(0, 1)] / -sin_theta).arg();
            phi = (unitary[(1, 0)] / sin_theta).arg();
        } else {
            phi = (unitary[(1, 1)] / theta.cos()).arg();
        }

        if SingleQubitRule::check2pi(theta, eps) {
            theta = 0.0;
        }

        if SingleQubitRule::check2pi(lambda, eps) {
            lambda = 0.0;
        }

        if SingleQubitRule::check2pi(phi, eps) {
            phi = 0.0;
        }

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if !Self::check2pi(2.0 * theta, eps)
            | !Self::check2pi(phi, eps)
            | !Self::check2pi(lambda, eps)
        {
            decomposed_gates.push((StandardGate::U, smallvec![2.0 * theta, phi, lambda]));
        }
        decomposed_gates
    }

    fn zxz_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        if Self::close_to_identity(&unitary) {
            return vec![];
        }
        Self::u2_to_su2(&mut unitary);
        let eps: f64 = 1e-13;
        let (_beta_plus_delta, _beta_minus_delta, gamma, beta, delta) =
            Self::zxz_type_arg(&unitary);

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if Self::check2pi(gamma, eps) {
            if !Self::check2pi(delta + beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta + beta]));
            }
        } else {
            if !Self::check2pi(delta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta]));
            }
            decomposed_gates.push((StandardGate::RX, smallvec![gamma]));
            if !Self::check2pi(beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![beta]));
            }
        }
        decomposed_gates
    }

    fn hrz_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        if Self::close_to_identity(&unitary) {
            return vec![];
        }
        Self::u2_to_su2(&mut unitary);
        let eps: f64 = 1e-13;
        let (_beta_plus_delta, _beta_minus_delta, gamma, beta, delta) =
            Self::zxz_type_arg(&unitary);
        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if Self::check2pi(gamma, eps) {
            if !Self::check2pi(delta + beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta + beta]));
            }
        } else {
            if !Self::check2pi(delta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta]));
            }
            decomposed_gates.push((StandardGate::H, smallvec![]));
            decomposed_gates.push((StandardGate::RZ, smallvec![gamma]));
            decomposed_gates.push((StandardGate::H, smallvec![]));
            if !Self::check2pi(beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![beta]));
            }
        }
        decomposed_gates
    }

    fn xyx_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let input_matrix: Array2<Complex<f64>> = unitary.clone();
        let mut unitary: Array2<Complex<f64>> = array![
            [
                0.5 * (input_matrix[(0, 0)]
                    + input_matrix[(0, 1)]
                    + input_matrix[(1, 0)]
                    + input_matrix[(1, 1)]),
                0.5 * (input_matrix[(0, 0)] - input_matrix[(0, 1)] + input_matrix[(1, 0)]
                    - input_matrix[(1, 1)])
            ],
            [
                0.5 * (input_matrix[(0, 0)] + input_matrix[(0, 1)]
                    - input_matrix[(1, 0)]
                    - input_matrix[(1, 1)]),
                0.5 * (input_matrix[(0, 0)] - input_matrix[(0, 1)] - input_matrix[(1, 0)]
                    + input_matrix[(1, 1)])
            ]
        ];
        let eps: f64 = 1e-13;

        Self::u2_to_su2(&mut unitary);

        let (delta, gamma, beta) = Self::zyz_type_arg(&unitary);

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if Self::check2pi(gamma, eps) {
            if !Self::check2pi(delta + beta, eps) {
                decomposed_gates.push((StandardGate::RX, smallvec![delta + beta]));
            }
        } else {
            if !Self::check2pi(delta, eps) {
                decomposed_gates.push((StandardGate::RX, smallvec![delta]));
            }
            decomposed_gates.push((StandardGate::RY, smallvec![-gamma]));
            if !Self::check2pi(beta, eps) {
                decomposed_gates.push((StandardGate::RX, smallvec![beta]));
            }
        }
        decomposed_gates
    }

    fn zyz_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        let eps: f64 = 1e-13;

        Self::u2_to_su2(&mut unitary);

        let (delta, gamma, beta) = Self::zyz_type_arg(&unitary);

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if Self::check2pi(gamma, eps) {
            if !Self::check2pi(delta + beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta + beta]));
            }
        } else {
            if !Self::check2pi(delta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta]));
            }
            decomposed_gates.push((StandardGate::RY, smallvec![gamma]));
            if !Self::check2pi(beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![beta]));
            }
        }
        decomposed_gates
    }

    fn xsxrz_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        let eps: f64 = 1e-13;

        Self::u2_to_su2(&mut unitary);

        let (delta, gamma, beta) = Self::zyz_type_arg(&unitary);

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();
        if Self::check2pi(gamma, eps) {
            if !Self::check2pi(delta + beta, eps) {
                decomposed_gates.push((StandardGate::RZ, smallvec![delta + beta]));
            }
        } else {
            if !Self::check2pi(gamma - PI, eps) {
                if !Self::check2pi(delta, eps) {
                    decomposed_gates.push((StandardGate::RZ, smallvec![delta]));
                }
                decomposed_gates.push((StandardGate::X2P, smallvec![]));
                decomposed_gates.push((StandardGate::RZ, smallvec![gamma - PI]));
                decomposed_gates.push((StandardGate::X2P, smallvec![]));
                if !Self::check2pi(beta + PI, eps) {
                    decomposed_gates.push((StandardGate::RZ, smallvec![beta + PI]));
                }
            } else {
                if !Self::check2pi(delta - beta - PI, eps) {
                    decomposed_gates.push((StandardGate::RZ, smallvec![delta - beta - PI]));
                }
                decomposed_gates.push((StandardGate::X, smallvec![]));
            }
        }

        decomposed_gates
    }

    fn sxypmrz_rule(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)> {
        let mut unitary: Array2<Complex<f64>> = unitary.clone();
        let eps: f64 = 1e-13;

        Self::u2_to_su2(&mut unitary);

        let (delta, gamma, beta) = Self::zyz_type_arg(&unitary);

        let mut decomposed_gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::new();

        if !Self::check2pi(delta, eps) {
            decomposed_gates.push((StandardGate::RZ, smallvec![delta]));
        }

        if !Self::check2pi(gamma, eps) {
            if Self::check_plus_half_pi(gamma, eps) {
                decomposed_gates.push((StandardGate::Y2P, smallvec![]));
            } else if Self::check_minus_half_pi(gamma, eps) {
                decomposed_gates.push((StandardGate::Y2M, smallvec![]));
            } else {
                decomposed_gates.push((StandardGate::X2P, smallvec![]));
                decomposed_gates.push((StandardGate::RZ, smallvec![gamma]));
                decomposed_gates.push((StandardGate::X2M, smallvec![]));
            }
        }

        if !Self::check2pi(beta, eps) {
            decomposed_gates.push((StandardGate::RZ, smallvec![beta]));
        }

        decomposed_gates
    }
}

#[cfg(test)]
mod test_single_qubit_rule {
    use super::*;
    use rand::Rng;
    // lazy lock is used to initialize the CLIFFORD_STRINGS vector only once
    use std::sync::LazyLock;

    const PAULI_CASE: [&str; 4] = ["I", "X", "Y", "Z"];

    static CLIFFORD_STRINGS: LazyLock<Vec<String>> = LazyLock::new(|| one_q_clifford_string(6));

    struct CliffordString {
        hs_string: String,
        can_extend: bool,
    }

    fn recursive_clifford_gen(
        mut input_list: Vec<CliffordString>,
        depth: u32,
    ) -> Vec<CliffordString> {
        if depth == 0 {
            return input_list;
        }

        let mut output_list: Vec<CliffordString> = Vec::new();
        let extensions = ["H", "S"];

        for i in 0..input_list.len() {
            if !input_list[i].can_extend {
                continue;
            }

            let mut all_combinations_exist = true;
            let base_string = input_list[i].hs_string.clone();

            for ext in &extensions {
                // Try prepending
                let prepended = format!("{}{}", ext, base_string);
                // Try appending
                let appended = format!("{}{}", base_string, ext);

                for new_string in [prepended, appended] {
                    let new_matrix = get_hs_matrix(&new_string);

                    // Check against all matrices in input_list
                    let mut found_equal = false;
                    for existing in &input_list {
                        let existing_matrix = get_hs_matrix(&existing.hs_string);
                        if is_matrix_differ_by_phase(&new_matrix, &existing_matrix) {
                            found_equal = true;
                            break;
                        }
                    }

                    // Check against all matrices in output_list
                    if !found_equal {
                        for existing in &output_list {
                            let existing_matrix = get_hs_matrix(&existing.hs_string);
                            if is_matrix_differ_by_phase(&new_matrix, &existing_matrix) {
                                found_equal = true;
                                break;
                            }
                        }
                    }

                    // If no equal found, add to output_list
                    if !found_equal {
                        output_list.push(CliffordString {
                            hs_string: new_string,
                            can_extend: true,
                        });
                        all_combinations_exist = false;
                    }
                }
            }

            // If all combinations matched existing strings, mark as non-extendable
            if all_combinations_exist {
                input_list[i].can_extend = false;
            }
        }

        // Prepend input_list to output_list
        let mut result = input_list;
        result.append(&mut output_list);

        // Recurse with depth - 1
        recursive_clifford_gen(result, depth - 1)
    }

    fn one_q_clifford_string(max_depth: u32) -> Vec<String> {
        let initial_list = vec![
            CliffordString {
                hs_string: "H".to_string(),
                can_extend: true,
            },
            CliffordString {
                hs_string: "S".to_string(),
                can_extend: true,
            },
        ];
        let result = recursive_clifford_gen(initial_list, max_depth);
        result.iter().map(|s| s.hs_string.clone()).collect()
    }

    /// Build a random 2×2 unitary using cqlib U gate: U(θ, φ, λ) with random angles.
    fn get_random_unitary() -> Array2<Complex<f64>> {
        let mut rng = rand::rng();
        let theta = rng.random_range(0.0..PI);
        let phi = rng.random_range(0.0..(2.0 * PI));
        let lam = rng.random_range(0.0..(2.0 * PI));
        StandardGate::U
            .matrix(&[theta, phi, lam])
            .expect("U matrix should be well-formed")
            .into_owned()
    }

    fn format_complex(c: &Complex<f64>) -> String {
        let eps = 1e-10;
        let re = if c.re.abs() < eps { 0.0 } else { c.re };
        let im = if c.im.abs() < eps { 0.0 } else { c.im };

        if im.abs() < eps {
            format!("{:>8.4}", re)
        } else if re.abs() < eps {
            format!("{:>8.4}i", im)
        } else if im >= 0.0 {
            format!("{:.4}+{:.4}i", re, im)
        } else {
            format!("{:.4}{:.4}i", re, im)
        }
    }

    fn print_matrix(name: &str, matrix: &Array2<Complex<f64>>) {
        println!("{}:", name);
        for row in matrix.rows() {
            let row_str: Vec<String> = row.iter().map(|c| format_complex(c)).collect();
            println!("  [{}]", row_str.join(", "));
        }
    }

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
        // let phase = inner / inner.abs();
        let vec1_norm: f64 = complex_inner_product(&vec1, &vec1).re.sqrt();
        let vec2_norm: f64 = complex_inner_product(&vec2, &vec2).re.sqrt();

        let cos_vec = inner_abs / (vec1_norm * vec2_norm);
        (cos_vec - 1.0).abs() < 1e-12
    }

    fn matrix_from_gate_vec(
        gates: &Vec<(StandardGate, SmallVec<[f64; 3]>)>,
    ) -> Array2<Complex<f64>> {
        let mut total_u = StandardGate::I
            .matrix(&[])
            .expect("identity matrix should be well-formed")
            .into_owned();

        for (gate, param) in gates {
            total_u = gate
                .matrix(param)
                .expect("single-qubit gate matrix should be well-formed")
                .dot(&total_u);
        }
        total_u
    }

    fn get_pauli_matrix(pauli_string: &str) -> Array2<Complex<f64>> {
        let gate_type = match pauli_string {
            "I" => StandardGate::I,
            "X" => StandardGate::X,
            "Y" => StandardGate::Y,
            "Z" => StandardGate::Z,
            _ => panic!("Invalid Pauli string"),
        };
        gate_type
            .matrix(&[])
            .expect("Pauli matrix should be well-formed")
            .into_owned()
    }

    fn get_hs_matrix(hs_string: &str) -> Array2<Complex<f64>> {
        // Build gates in reverse order from the string
        let gates: Vec<(StandardGate, SmallVec<[f64; 3]>)> = hs_string
            .chars()
            .map(|c| match c.to_ascii_uppercase() {
                'H' => (StandardGate::H, SmallVec::new()),
                'S' => (StandardGate::S, SmallVec::new()),
                _ => panic!(
                    "Invalid character in HS string: expected 'H' or 'S', got '{}'",
                    c
                ),
            })
            .collect();

        matrix_from_gate_vec(&gates)
    }

    fn test_verbose() -> bool {
        std::env::var("CQLIB_TEST_VERBOSE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    }

    fn assert_rule_decomposition(
        rule: fn(&Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)>,
        unitary: &Array2<Complex<f64>>,
        rule_name: &str,
        case_name: &str,
    ) {
        let gates = rule(unitary);
        if test_verbose() {
            if !case_name.is_empty() {
                println!("rule {rule_name} on {case_name}:");
            } else {
                println!("rule {rule_name}:");
            }
            println!("input unitary: {unitary:?}");
            println!("gates: {gates:?}");
        }

        let composite = matrix_from_gate_vec(&gates);
        let error_msg: String;
        if !case_name.is_empty() {
            error_msg = format!("rule {rule_name} on {case_name} did not match up to phase");
        } else {
            error_msg = format!("rule {rule_name} did not match up to phase");
        }

        assert!(
            is_matrix_differ_by_phase(unitary, &composite),
            "{}",
            error_msg
        );
    }

    fn test_rule_random_u(
        rule: fn(&Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)>,
        rule_name: &str,
        reps: usize,
    ) {
        for _ in 0..reps {
            let unitary = get_random_unitary();
            assert_rule_decomposition(rule, &unitary, rule_name, "");
        }
    }

    fn test_rule_clifford(
        rule: fn(&Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)>,
        rule_name: &str,
    ) {
        for case in PAULI_CASE {
            let matrix = get_pauli_matrix(case);
            assert_rule_decomposition(rule, &matrix, rule_name, case);
        }

        for case in CLIFFORD_STRINGS.iter() {
            let matrix = get_hs_matrix(case);
            assert_rule_decomposition(rule, &matrix, rule_name, case);
        }
    }

    #[test]
    fn test_matrix_equal_func() {
        let mut rng = rand::rng();
        let ref_mat = StandardGate::Y
            .matrix(&[])
            .expect("Y matrix should be well-formed")
            .into_owned();

        for _ in 0..10 {
            let random_phase = Complex::new(0.0, rng.random_range(-PI..PI)).exp();
            let test_mat = ref_mat.clone().into_owned() * random_phase;
            let is_equal = is_matrix_differ_by_phase(&ref_mat, &test_mat);

            assert!(is_equal, "ref_mat and test_mat are not equal");
        }

        for _ in 0..10 {
            let random_phase = Complex::new(0.0, rng.random_range(-PI..PI)).exp();
            let test_mat = StandardGate::X
                .matrix(&[])
                .expect("X matrix should be well-formed")
                .into_owned()
                * random_phase;
            let is_not_equal = !is_matrix_differ_by_phase(&ref_mat, &test_mat);

            assert!(is_not_equal, "ref_mat and test_mat are equal");
        }
    }

    #[test]
    fn random_u_u3_rule() {
        test_rule_random_u(SingleQubitRule::u3_rule, "u3_rule", 5);
    }

    #[test]
    fn random_u_zxz_rule() {
        test_rule_random_u(SingleQubitRule::zxz_rule, "zxz_rule", 5);
    }

    #[test]
    fn random_u_zyz_rule() {
        test_rule_random_u(SingleQubitRule::zyz_rule, "zyz_rule", 5);
    }

    #[test]
    fn random_u_xyx_rule() {
        test_rule_random_u(SingleQubitRule::xyx_rule, "xyx_rule", 5);
    }

    #[test]
    fn random_u_hrz_rule() {
        test_rule_random_u(SingleQubitRule::hrz_rule, "hrz_rule", 5);
    }

    #[test]
    fn random_u_xsxrz_rule() {
        test_rule_random_u(SingleQubitRule::xsxrz_rule, "xsxrz_rule", 5);
    }

    #[test]
    fn random_u_sxypmrz_rule() {
        test_rule_random_u(SingleQubitRule::sxypmrz_rule, "sxypmrz_rule", 5);
    }

    #[test]
    fn clifford_u3_rule() {
        test_rule_clifford(SingleQubitRule::u3_rule, "u3_rule");
    }

    #[test]
    fn clifford_zxz_rule() {
        test_rule_clifford(SingleQubitRule::zxz_rule, "zxz_rule");
    }

    #[test]
    fn clifford_zyz_rule() {
        test_rule_clifford(SingleQubitRule::zyz_rule, "zyz_rule");
    }

    #[test]
    fn clifford_xyx_rule() {
        test_rule_clifford(SingleQubitRule::xyx_rule, "xyx_rule");
    }

    #[test]
    fn clifford_hrz_rule() {
        test_rule_clifford(SingleQubitRule::hrz_rule, "hrz_rule");
    }

    #[test]
    fn clifford_xsxrz_rule() {
        test_rule_clifford(SingleQubitRule::xsxrz_rule, "xsxrz_rule");
    }

    #[test]
    fn clifford_sxypmrz_rule() {
        test_rule_clifford(SingleQubitRule::sxypmrz_rule, "sxypmrz_rule");
    }
}
