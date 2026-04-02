use super::*;
use ndarray::prelude::*;
use num::complex::Complex;
use num::complex::ComplexFloat;
use rand::Rng;
use smallvec::smallvec;
use std::collections::HashMap;
use std::f64::consts::PI;

fn complex_inner_product(vec1: &[Complex<f64>], vec2: &[Complex<f64>]) -> Complex<f64> {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| a.conj() * b)
        .sum()
}

fn is_matrix_differ_by_phase(matrix1: &Array2<Complex<f64>>, matrix2: &Array2<Complex<f64>>) -> bool {
    let vec1: Vec<Complex<f64>> = matrix1.iter().copied().collect();
    let vec2: Vec<Complex<f64>> = matrix2.iter().copied().collect();
    let inner = complex_inner_product(&vec1, &vec2);
    let inner_abs = inner.abs();
    let vec1_norm = complex_inner_product(&vec1, &vec1).re.sqrt();
    let vec2_norm = complex_inner_product(&vec2, &vec2).re.sqrt();
    let cos_vec = inner_abs / (vec1_norm * vec2_norm);
    (cos_vec - 1.0).abs() < 1e-12
}

fn matrix_from_decomposed_gate(decomposed: &DecomposedGate) -> Array2<Complex<f64>> {
    let mut total_u = StandardGate::I
        .matrix(&[])
        .expect("identity matrix should be well-formed")
        .into_owned();

    for op in &decomposed.ops {
        assert_eq!(
            op.qubits.as_slice(),
            &[0],
            "Expected single-qubit decomposition on qubit 0"
        );
        let gate_params: SmallVec<[f64; 3]> =
            op.params.iter().map(|p| p.evaluate(&None).unwrap()).collect();
        total_u = op
            .gate
            .matrix(&gate_params)
            .expect("single-qubit gate matrix should be well-formed")
            .dot(&total_u);
    }

    total_u
}

fn assert_rule_decomposition(
    rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
    gate: StandardGate,
    params: &SmallVec<[Parameter; 3]>,
    rule_name: &str,
) {
    let decomposed = rule(&gate, params);
    let gate_params: SmallVec<[f64; 3]> = params.iter().map(|p| p.evaluate(&None).unwrap()).collect();
    let original_matrix = gate
        .matrix(&gate_params)
        .expect("single-qubit gate matrix should be well-formed")
        .to_owned();
    let decomposed_matrix = matrix_from_decomposed_gate(&decomposed);

    assert!(
        is_matrix_differ_by_phase(&original_matrix, &decomposed_matrix),
        "Rule {} failed for gate {:?}",
        rule_name,
        gate
    );
}

fn special_angles() -> Vec<f64> {
    vec![0.0, PI / 2.0, PI, 2.0 * PI, -0.5, 1.234]
}

fn test_random_angles_1(
    rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
    gate: StandardGate,
    rule_name: &str,
    reps: usize,
) {
    let mut rng = rand::rng();
    for _ in 0..reps {
        let theta = rng.random_range(-PI..PI);
        assert_rule_decomposition(rule, gate, &smallvec![Parameter::from(theta)], rule_name);
    }
}

fn test_random_angles_2(
    rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
    gate: StandardGate,
    rule_name: &str,
    reps: usize,
) {
    let mut rng = rand::rng();
    for _ in 0..reps {
        let theta = rng.random_range(-PI..PI);
        let phi = rng.random_range(-PI..PI);
        assert_rule_decomposition(
            rule,
            gate,
            &smallvec![Parameter::from(theta), Parameter::from(phi)],
            rule_name,
        );
    }
}

fn test_random_angles_3(
    rule: fn(&StandardGate, &SmallVec<[Parameter; 3]>) -> DecomposedGate,
    gate: StandardGate,
    rule_name: &str,
    reps: usize,
) {
    let mut rng = rand::rng();
    for _ in 0..reps {
        let theta = rng.random_range(-PI..PI);
        let phi = rng.random_range(-PI..PI);
        let lambda = rng.random_range(-PI..PI);
        assert_rule_decomposition(
            rule,
            gate,
            &smallvec![
                Parameter::from(theta),
                Parameter::from(phi),
                Parameter::from(lambda)
            ],
            rule_name,
        );
    }
}

#[test]
fn test_rx_u_rxy_rules() {
    for theta in special_angles() {
        assert_rule_decomposition(
            ParamTransformRule::rx2u_rule,
            StandardGate::RX,
            &smallvec![Parameter::from(theta)],
            "rx2u_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::rx2rxy_rule,
            StandardGate::RX,
            &smallvec![Parameter::from(theta)],
            "rx2rxy_rule",
        );
    }
}

#[test]
fn test_rxy_u_rules() {
    let angles = special_angles();
    for theta in &angles {
        for phi in &angles {
            assert_rule_decomposition(
                ParamTransformRule::rxy2u_rule,
                StandardGate::RXY,
                &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                "rxy2u_rule",
            );
            assert_rule_decomposition(
                ParamTransformRule::rxy2rx_rule,
                StandardGate::RXY,
                &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                "rxy2rx_rule",
            );
            assert_rule_decomposition(
                ParamTransformRule::rxy2xy_rule,
                StandardGate::RXY,
                &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                "rxy2xy_rule",
            );
            assert_rule_decomposition(
                ParamTransformRule::rxy2xy2p_rule,
                StandardGate::RXY,
                &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                "rxy2xy2p_rule",
            );
            assert_rule_decomposition(
                ParamTransformRule::rxy2xy2m_rule,
                StandardGate::RXY,
                &smallvec![Parameter::from(*theta), Parameter::from(*phi)],
                "rxy2xy2m_rule",
            );
        }
    }

    for phi in &angles {
        assert_rule_decomposition(
            ParamTransformRule::xy2rxy_rule,
            StandardGate::XY,
            &smallvec![Parameter::from(*phi)],
            "xy2rxy_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::xy2p2rxy_rule,
            StandardGate::XY2P,
            &smallvec![Parameter::from(*phi)],
            "xy2p2rxy_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::xy2m2rxy_rule,
            StandardGate::XY2M,
            &smallvec![Parameter::from(*phi)],
            "xy2m2rxy_rule",
        );

    }
}

#[test]
fn test_u_rx_rxy_rules() {
    let angles = special_angles();
    for theta in &angles {
        for phi in &angles {
            for lambda in &angles {
                assert_rule_decomposition(
                    ParamTransformRule::u2rx_rule,
                    StandardGate::U,
                    &smallvec![
                        Parameter::from(*theta),
                        Parameter::from(*phi),
                        Parameter::from(*lambda)
                    ],
                    "u2rx_rule",
                );
                assert_rule_decomposition(
                    ParamTransformRule::u2rxy_rule,
                    StandardGate::U,
                    &smallvec![
                        Parameter::from(*theta),
                        Parameter::from(*phi),
                        Parameter::from(*lambda)
                    ],
                    "u2rxy_rule",
                );
            }
        }
    }
}

#[test]
fn test_pairwise_rx_ry_rz_rules() {
    for theta in special_angles() {
        let params = smallvec![Parameter::from(theta)];
        assert_rule_decomposition(
            ParamTransformRule::rx2ry_rule,
            StandardGate::RX,
            &params,
            "rx2ry_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::ry2rx_rule,
            StandardGate::RY,
            &params,
            "ry2rx_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::rx2rz_rule,
            StandardGate::RX,
            &params,
            "rx2rz_rule",
        );
        assert_rule_decomposition(
            ParamTransformRule::rz2rx_rule,
            StandardGate::RZ,
            &params,
            "rz2rx_rule",
        );
    }
}

#[test]
fn test_rx_u_rxy_rules_random_angles() {
    test_random_angles_1(ParamTransformRule::rx2u_rule, StandardGate::RX, "rx2u_rule", 10);
    test_random_angles_1(
        ParamTransformRule::rx2rxy_rule,
        StandardGate::RX,
        "rx2rxy_rule",
        10,
    );
    test_random_angles_1(
        ParamTransformRule::rx2ry_rule,
        StandardGate::RX,
        "rx2ry_rule",
        10,
    );
    test_random_angles_1(
        ParamTransformRule::rx2rz_rule,
        StandardGate::RX,
        "rx2rz_rule",
        10,
    );
    test_random_angles_1(
        ParamTransformRule::ry2rx_rule,
        StandardGate::RY,
        "ry2rx_rule",
        10,
    );
    test_random_angles_1(
        ParamTransformRule::rz2rx_rule,
        StandardGate::RZ,
        "rz2rx_rule",
        10,
    );
}

#[test]
fn test_rxy_u_rules_random_angles() {
    test_random_angles_2(
        ParamTransformRule::rxy2u_rule,
        StandardGate::RXY,
        "rxy2u_rule",
        10,
    );
    test_random_angles_2(
        ParamTransformRule::rxy2rx_rule,
        StandardGate::RXY,
        "rxy2rx_rule",
        10,
    );
    test_random_angles_2(
        ParamTransformRule::rxy2xy_rule,
        StandardGate::RXY,
        "rxy2xy_rule",
        10,
    );
    test_random_angles_2(
        ParamTransformRule::rxy2xy2p_rule,
        StandardGate::RXY,
        "rxy2xy2p_rule",
        10,
    );
    test_random_angles_2(
        ParamTransformRule::rxy2xy2m_rule,
        StandardGate::RXY,
        "rxy2xy2m_rule",
        10,
    );
    test_random_angles_1(ParamTransformRule::xy2rxy_rule, StandardGate::XY, "xy2rxy_rule", 10);
    test_random_angles_1(
        ParamTransformRule::xy2p2rxy_rule,
        StandardGate::XY2P,
        "xy2p2rxy_rule",
        10,
    );
    test_random_angles_1(
        ParamTransformRule::xy2m2rxy_rule,
        StandardGate::XY2M,
        "xy2m2rxy_rule",
        10,
    );

}

#[test]
fn test_u_rx_rxy_rules_random_angles() {
    test_random_angles_3(ParamTransformRule::u2rx_rule, StandardGate::U, "u2rx_rule", 10);
    test_random_angles_3(
        ParamTransformRule::u2rxy_rule,
        StandardGate::U,
        "u2rxy_rule",
        10,
    );
}

#[test]
fn test_symbolic_params_preserved_for_rxy2u() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let decomposed = ParamTransformRule::rxy2u_rule(&StandardGate::RXY, &smallvec![theta, phi]);

    assert_eq!(decomposed.ops.len(), 1);
    assert_eq!(decomposed.ops[0].gate, StandardGate::U);

    let param_symbols: Vec<_> = decomposed.ops[0]
        .params
        .iter()
        .map(|p| p.get_symbols())
        .collect();
    assert!(param_symbols.iter().any(|symbols| symbols.contains("theta")));
    assert!(param_symbols.iter().any(|symbols| symbols.contains("phi")));
}

#[test]
fn test_symbolic_params_preserved_for_u2rx() {
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let lambda = Parameter::symbol("lambda");
    let decomposed =
        ParamTransformRule::u2rx_rule(&StandardGate::U, &smallvec![theta, phi, lambda]);

    let symbolic_params: Vec<_> = decomposed
        .ops
        .iter()
        .flat_map(|op| op.params.iter())
        .filter(|param| !param.get_symbols().is_empty())
        .map(|param| param.get_symbols())
        .collect();

    assert!(symbolic_params.iter().any(|symbols| symbols.contains("theta")));
    assert!(symbolic_params.iter().any(|symbols| symbols.contains("phi")));
    assert!(symbolic_params.iter().any(|symbols| symbols.contains("lambda")));

    let mut bindings = HashMap::new();
    bindings.insert("theta", 0.7);
    bindings.insert("phi", 0.4);
    bindings.insert("lambda", -1.1);

    let evaluated: Vec<f64> = decomposed
        .ops
        .iter()
        .flat_map(|op| op.params.iter())
        .map(|param| param.evaluate(&Some(bindings.clone())).unwrap())
        .collect();

    assert!(evaluated.iter().any(|v| (v - (0.4 + PI / 2.0)).abs() < 1e-10));
    assert!(evaluated.iter().any(|v| (v - 0.7).abs() < 1e-10));
    assert!(evaluated.iter().any(|v| (v - (-1.1 - PI / 2.0)).abs() < 1e-10));
}
