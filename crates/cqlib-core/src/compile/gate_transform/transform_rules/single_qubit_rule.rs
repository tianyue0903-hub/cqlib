use crate::circuit::gate::StandardGate;

use ndarray::prelude::*;
use num::complex::Complex;
use num::complex::ComplexFloat;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;

type SingleQubitDecomposeFn =
    fn(unitary: &Array2<Complex<f64>>) -> Vec<(StandardGate, SmallVec<[f64; 3]>)>;

pub struct SingleQubitRule {
    pub name: String,
    rule: SingleQubitDecomposeFn,
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

        (theta - multiple * 2.0 * PI).abs() < eps
    }

    fn check_plus_half_pi(theta: f64, eps: f64) -> bool {
        // check if theta is of π/2 + 2kπ
        let theta_mod: f64 = (theta).rem_euclid(2.0 * PI);
        (theta_mod - PI / 2.0).abs() < eps
    }

    fn check_minus_half_pi(theta: f64, eps: f64) -> bool {
        // check if theta is of -π/2 + 2kπ
        let theta_mod: f64 = (theta).rem_euclid(2.0 * PI);
        (theta_mod - 3.0 * PI / 2.0).abs() < eps
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
        phase
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
        let gamma: f64 =
            if unitary[(0, 0)].abs() > unitary[(0, 1)].abs() && unitary[(0, 1)].abs() > eps {
                let acos_arg = Self::clamp_acos(2.0 * (unitary[(0, 0)] * unitary[(1, 1)]).re - 1.0);
                Self::clamp_acos(acos_arg).acos()
            } else {
                let acos_arg = Self::clamp_acos(2.0 * (unitary[(0, 1)] * unitary[(1, 0)]).re + 1.0);
                Self::clamp_acos(acos_arg).acos()
            };

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
        unitary /= z;

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
#[path = "single_qubit_rule_test.rs"]
mod single_qubit_rule_test;
