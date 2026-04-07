use super::*;
use rand::Rng;
use std::f64::consts::PI;
use std::sync::LazyLock;

const PAULI_CASE: [&str; 4] = ["I", "X", "Y", "Z"];

static CLIFFORD_STRINGS: LazyLock<Vec<String>> = LazyLock::new(|| one_q_clifford_string(6));

struct CliffordString {
    hs_string: String,
    can_extend: bool,
}

fn recursive_clifford_gen(mut input_list: Vec<CliffordString>, depth: u32) -> Vec<CliffordString> {
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
            let prepended = format!("{}{}", ext, base_string);
            let appended = format!("{}{}", base_string, ext);

            for new_string in [prepended, appended] {
                let new_matrix = get_hs_matrix(&new_string);

                let mut found_equal = false;
                for existing in &input_list {
                    let existing_matrix = get_hs_matrix(&existing.hs_string);
                    if is_matrix_differ_by_phase(&new_matrix, &existing_matrix) {
                        found_equal = true;
                        break;
                    }
                }

                if !found_equal {
                    for existing in &output_list {
                        let existing_matrix = get_hs_matrix(&existing.hs_string);
                        if is_matrix_differ_by_phase(&new_matrix, &existing_matrix) {
                            found_equal = true;
                            break;
                        }
                    }
                }

                if !found_equal {
                    output_list.push(CliffordString {
                        hs_string: new_string,
                        can_extend: true,
                    });
                    all_combinations_exist = false;
                }
            }
        }

        if all_combinations_exist {
            input_list[i].can_extend = false;
        }
    }

    let mut result = input_list;
    result.append(&mut output_list);

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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    let vec1_norm: f64 = complex_inner_product(&vec1, &vec1).re.sqrt();
    let vec2_norm: f64 = complex_inner_product(&vec2, &vec2).re.sqrt();

    let cos_vec = inner_abs / (vec1_norm * vec2_norm);
    (cos_vec - 1.0).abs() < 1e-12
}

fn matrix_from_gate_vec(gates: &Vec<(StandardGate, SmallVec<[f64; 3]>)>) -> Array2<Complex<f64>> {
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
