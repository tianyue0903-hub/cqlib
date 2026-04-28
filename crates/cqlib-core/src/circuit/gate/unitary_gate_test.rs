use super::*;
use crate::circuit::circuit_to_matrix;
use crate::circuit::{Circuit, Parameter, Qubit};
use ndarray::array;
use std::f64::consts::PI;
use std::sync::Arc;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

// Construction and basic getters

#[test]
fn test_new_initializes_all_getters() {
    let gate = UnitaryGate::new("MyGate", 3, 2);
    assert_eq!(gate.label(), "MyGate");
    assert_eq!(gate.num_qubits(), 3);
    assert_eq!(gate.num_params(), 2);
    assert!(gate.matrix().is_none());
    assert!(gate.circuit().is_none());
    assert!(!gate.has_parameterized_matrix());
}

// with_matrix

#[test]
fn test_with_matrix_success() {
    let h = c(1.0 / f64::sqrt(2.0), 0.0);
    let matrix = array![[h, h], [h, -h]];
    let gate = UnitaryGate::new("Hadamard", 1, 0)
        .with_matrix(matrix.clone())
        .unwrap();

    assert!(gate.matrix().is_some());
    let mat = gate.matrix().unwrap();
    assert_eq!(mat.shape(), &[2, 2]);
    assert_eq!(mat[[0, 0]], h);
    assert_eq!(mat[[0, 1]], h);
    assert_eq!(mat[[1, 0]], h);
    assert_eq!(mat[[1, 1]], -h);
}

#[test]
fn test_with_matrix_rejects_parameterized_gate() {
    let matrix = Array2::eye(2).mapv(|v| c(v, 0.0));
    let err = UnitaryGate::new("BadStatic", 1, 1)
        .with_matrix(matrix)
        .unwrap_err();

    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 0,
            actual: 1
        }
    ));
}

#[test]
fn test_with_matrix_rejects_wrong_dimensions() {
    // 1-qubit gate expects 2x2, but we give 4x4
    let matrix = Array2::eye(4).mapv(|v| c(v, 0.0));
    let err = UnitaryGate::new("WrongDim", 1, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));

    // 2-qubit gate expects 4x4, but we give 2x2
    let matrix = Array2::eye(2).mapv(|v| c(v, 0.0));
    let err = UnitaryGate::new("WrongDim", 2, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_with_matrix_rejects_non_square() {
    let matrix = array![
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
    ];
    let err = UnitaryGate::new("NonSquare", 1, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

// with_parameterized_matrix

#[test]
fn test_with_parameterized_matrix_success() {
    let gate = UnitaryGate::new("Rx", 1, 1)
        .with_parameterized_matrix(|params| {
            let theta = params[0];
            let cos = f64::cos(theta / 2.0);
            let sin = f64::sin(theta / 2.0);
            array![[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)],]
        })
        .unwrap();

    assert!(gate.has_parameterized_matrix());
    assert!(gate.matrix().is_none());
}

#[test]
fn test_parameterized_matrix_validates_generated_shape() {
    let gate = UnitaryGate::new("BadFactory", 1, 1)
        .with_parameterized_matrix(|_| array![[c(1.0, 0.0)]])
        .unwrap();

    let err = gate.matrix_for_params(&[0.1]).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

// matrix_for_params – static matrix path

#[test]
fn test_matrix_for_params_returns_borrowed_for_static_matrix() {
    let matrix = Array2::eye(2).mapv(|v| c(v, 0.0));
    let gate = UnitaryGate::new("Id", 1, 0).with_matrix(matrix).unwrap();

    let result = gate.matrix_for_params(&[]).unwrap();
    assert!(matches!(result, Cow::Borrowed(_)));
    assert_eq!(result.nrows(), 2);
}

// matrix_for_params – parameterized matrix path

#[test]
fn test_matrix_for_params_returns_owned_for_parameterized_matrix() {
    let gate = UnitaryGate::new("Rx", 1, 1)
        .with_parameterized_matrix(|params| {
            let theta = params[0];
            let cos = f64::cos(theta / 2.0);
            let sin = f64::sin(theta / 2.0);
            array![[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)],]
        })
        .unwrap();

    let result = gate.matrix_for_params(&[PI]).unwrap();
    assert!(matches!(result, Cow::Owned(_)));
    // RX(pi) ~= -i X  => off-diagonals should be ~ -1, diagonals ~ 0
    assert!((result[[0, 0]].re).abs() < 1e-10);
    assert!((result[[0, 1]].re).abs() < 1e-10);
    assert!((result[[0, 1]].im + 1.0).abs() < 1e-10);
    assert!((result[[1, 0]].re).abs() < 1e-10);
    assert!((result[[1, 0]].im + 1.0).abs() < 1e-10);
    assert!((result[[1, 1]].re).abs() < 1e-10);
}

// matrix_for_params – circuit-backed path

#[test]
fn test_matrix_for_params_circuit_backed_static() {
    let mut inner = Circuit::new(1);
    inner.h(Qubit::new(0)).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));

    let gate = UnitaryGate::new("HCircuit", 1, 0)
        .with_circuit(frozen)
        .unwrap();

    let result = gate.matrix_for_params(&[]).unwrap();
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.shape(), &[2, 2]);

    let expected = 1.0 / f64::sqrt(2.0);
    assert!((result[[0, 0]].re - expected).abs() < 1e-10);
    assert!((result[[0, 1]].re - expected).abs() < 1e-10);
    assert!((result[[1, 0]].re - expected).abs() < 1e-10);
    assert!((result[[1, 1]].re + expected).abs() < 1e-10);
}

#[test]
fn test_matrix_for_params_circuit_backed_parameterized() {
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));

    let gate = UnitaryGate::new("RxCircuit", 1, 1)
        .with_circuit(frozen)
        .unwrap();

    let result = gate.matrix_for_params(&[PI]).unwrap();
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.shape(), &[2, 2]);
    // RX(pi) ~= -i X
    assert!((result[[0, 1]].im + 1.0).abs() < 1e-10);
    assert!((result[[1, 0]].im + 1.0).abs() < 1e-10);
}

// matrix_for_params – parameter validation

#[test]
fn test_matrix_for_params_wrong_param_count() {
    let matrix = Array2::eye(2).mapv(|v| c(v, 0.0));
    let gate = UnitaryGate::new("Id", 1, 0).with_matrix(matrix).unwrap();

    // Too many params for a 0-param gate
    let err = gate.matrix_for_params(&[1.0]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 0,
            actual: 1
        }
    ));

    // Too few params for a parameterized gate
    let gate = UnitaryGate::new("Rx", 1, 1)
        .with_parameterized_matrix(|params| {
            let theta = params[0];
            let cos = f64::cos(theta / 2.0);
            let sin = f64::sin(theta / 2.0);
            array![[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)],]
        })
        .unwrap();

    let err = gate.matrix_for_params(&[]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 1,
            actual: 0
        }
    ));
}

#[test]
fn test_matrix_for_params_rejects_nan() {
    let gate = UnitaryGate::new("Rx", 1, 1)
        .with_parameterized_matrix(|params| {
            let theta = params[0];
            let cos = f64::cos(theta / 2.0);
            let sin = f64::sin(theta / 2.0);
            array![[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)],]
        })
        .unwrap();

    let err = gate.matrix_for_params(&[f64::NAN]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::InvalidParameterValue(idx, val)
    if idx == 0 && val.is_nan()));
}

#[test]
fn test_matrix_for_params_rejects_infinity() {
    let gate = UnitaryGate::new("Rx", 1, 1)
        .with_parameterized_matrix(|params| {
            let theta = params[0];
            let cos = f64::cos(theta / 2.0);
            let sin = f64::sin(theta / 2.0);
            array![[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)],]
        })
        .unwrap();

    let err = gate.matrix_for_params(&[f64::INFINITY]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::InvalidParameterValue(idx, val)
    if idx == 0 && val.is_infinite()));

    let err = gate.matrix_for_params(&[f64::NEG_INFINITY]).unwrap_err();
    assert!(matches!(
        err,
        CircuitError::InvalidParameterValue(idx, val)
    if idx == 0 && val.is_infinite()));
}

#[test]
fn test_matrix_for_params_no_representation() {
    let gate = UnitaryGate::new("Empty", 1, 0);
    let err = gate.matrix_for_params(&[]).unwrap_err();
    assert!(matches!(err, CircuitError::NoMatrixRepresentation));
}

// with_circuit

#[test]
fn test_with_circuit_success() {
    let mut inner = Circuit::new(2);
    inner.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));

    let gate = UnitaryGate::new("CNOT", 2, 0)
        .with_circuit(frozen.clone())
        .unwrap();
    assert!(gate.circuit().is_some());
    assert_eq!(gate.circuit().as_ref().unwrap().circuit().qubits().len(), 2);
}

#[test]
fn test_with_circuit_validates_signature() {
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), Parameter::symbol("theta")).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));

    let gate = UnitaryGate::new("CircuitBacked", 1, 1)
        .with_circuit(frozen.clone())
        .unwrap();
    assert_eq!(gate.num_params(), 1);

    let err = UnitaryGate::new("WrongParamCount", 1, 0)
        .with_circuit(frozen.clone())
        .unwrap_err();
    assert!(matches!(
        err,
        CircuitError::ParameterCountMismatch {
            expected: 0,
            actual: 1
        }
    ));

    let err = UnitaryGate::new("WrongQubitCount", 2, 1)
        .with_circuit(frozen)
        .unwrap_err();
    assert!(matches!(
        err,
        CircuitError::QubitCountMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

// Identity (Eq, Hash, Clone)

#[test]
fn test_equality_based_on_uuid() {
    let gate_a = UnitaryGate::new("A", 1, 0);
    let gate_b = UnitaryGate::new("A", 1, 0);
    assert_ne!(
        gate_a, gate_b,
        "Two independently created gates must not be equal"
    );

    let gate_a_clone = gate_a.clone();
    assert_eq!(gate_a, gate_a_clone, "Clone must share the same identity");
}

#[test]
fn test_hash_consistent_with_equality() {
    use std::collections::hash_map::DefaultHasher;

    let gate = UnitaryGate::new("HashTest", 2, 3);
    let clone = gate.clone();

    let mut hasher_a = DefaultHasher::new();
    gate.hash(&mut hasher_a);
    let hash_a = hasher_a.finish();

    let mut hasher_b = DefaultHasher::new();
    clone.hash(&mut hasher_b);
    let hash_b = hasher_b.finish();

    assert_eq!(hash_a, hash_b);
}

// Formatting (Display, Debug)

#[test]
fn test_display_outputs_label() {
    let gate = UnitaryGate::new("FooBar", 1, 0);
    assert_eq!(format!("{}", gate), "FooBar");
}

#[test]
fn test_debug_contains_struct_name_and_fields() {
    let gate = UnitaryGate::new("DebugGate", 1, 0);
    let debug = format!("{:?}", gate);
    assert!(debug.starts_with("UnitaryGate {"));
    assert!(debug.contains("id"));
    assert!(debug.contains("DebugGate"));
    assert!(debug.contains("num_qubits: 1"));
    assert!(debug.contains("num_params: 0"));
    assert!(debug.contains("matrix: None"));
    assert!(debug.contains("parameterized_matrix: None"));
    assert!(debug.contains("circuit: None"));
}

#[test]
fn test_debug_parameterized_matrix_shows_placeholder() {
    let gate = UnitaryGate::new("ParamGate", 1, 1)
        .with_parameterized_matrix(|_| Array2::eye(2).mapv(|v| c(v, 0.0)))
        .unwrap();
    let debug = format!("{:?}", gate);
    assert!(debug.contains("parameterized_matrix: Some(\"<matrix_fn>\")"));
}

// A. Unitarity and numeric validation

#[test]
fn test_with_matrix_rejects_non_unitary() {
    // Dimension is correct (2x2) but not unitary: first row norm != 1
    let matrix = array![[c(2.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)],];
    let err = UnitaryGate::new("NonUnitary", 1, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_with_matrix_rejects_matrix_with_nan() {
    let matrix = array![[c(f64::NAN, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)],];
    let err = UnitaryGate::new("NanMat", 1, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_with_matrix_rejects_matrix_with_inf() {
    let matrix = array![
        [c(f64::INFINITY, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0)],
    ];
    let err = UnitaryGate::new("InfMat", 1, 0)
        .with_matrix(matrix)
        .unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_matrix_for_params_rejects_non_unitary_from_factory() {
    let gate = UnitaryGate::new("BadFactory", 1, 1)
        .with_parameterized_matrix(|_| {
            array![[c(2.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
        })
        .unwrap();
    let err = gate.matrix_for_params(&[0.0]).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

#[test]
fn test_matrix_for_params_rejects_nan_from_factory() {
    let gate = UnitaryGate::new("BadFactory", 1, 1)
        .with_parameterized_matrix(|_| {
            array![[c(f64::NAN, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
        })
        .unwrap();
    let err = gate.matrix_for_params(&[0.0]).unwrap_err();
    assert!(matches!(err, CircuitError::InvalidOperation(_)));
}

// B. Combination behavior and edge cases

#[test]
fn test_matrix_and_circuit_both_set_prefers_matrix() {
    let mut inner = Circuit::new(1);
    inner.x(Qubit::new(0)).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));

    let matrix = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];
    let gate = UnitaryGate::new("Both", 1, 0)
        .with_matrix(matrix)
        .unwrap()
        .with_circuit(frozen)
        .unwrap();

    let result = gate.matrix_for_params(&[]).unwrap();
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_parameterized_matrix_on_zero_params_gate() {
    let gate = UnitaryGate::new("ZeroParam", 1, 0)
        .with_parameterized_matrix(|_| Array2::eye(2).mapv(|v| c(v, 0.0)))
        .unwrap();
    let result = gate.matrix_for_params(&[]).unwrap();
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result[[0, 0]], c(1.0, 0.0));
    assert_eq!(result[[1, 1]], c(1.0, 0.0));
}

// C. Circuit-backed edge cases

#[test]
fn test_circuit_backed_with_measure_fails() {
    let mut inner = Circuit::new(1);
    inner.measure(Qubit::new(0)).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));
    let gate = UnitaryGate::new("MeasureGate", 1, 0)
        .with_circuit(frozen)
        .unwrap();
    let err = gate.matrix_for_params(&[]).unwrap_err();
    assert!(matches!(err, CircuitError::NoMatrixRepresentation));
}

#[test]
fn test_circuit_backed_with_reset_fails() {
    let mut inner = Circuit::new(1);
    inner.reset(Qubit::new(0)).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));
    let gate = UnitaryGate::new("ResetGate", 1, 0)
        .with_circuit(frozen)
        .unwrap();
    let err = gate.matrix_for_params(&[]).unwrap_err();
    assert!(matches!(err, CircuitError::NoMatrixRepresentation));
}

#[test]
fn test_circuit_backed_global_phase_preserved() {
    let mut inner = Circuit::new(1);
    inner.x(Qubit::new(0)).unwrap();
    inner.set_global_phase((PI / 2.0).into());
    let frozen = Arc::new(FrozenCircuit::new(inner));
    let gate = UnitaryGate::new("PhaseX", 1, 0)
        .with_circuit(frozen)
        .unwrap();

    let result = gate.matrix_for_params(&[]).unwrap();
    // X with global phase PI/2 => i * X
    assert!((result[[0, 1]].re).abs() < 1e-10);
    assert!((result[[0, 1]].im - 1.0).abs() < 1e-10);
    assert!((result[[1, 0]].re).abs() < 1e-10);
    assert!((result[[1, 0]].im - 1.0).abs() < 1e-10);
    assert!((result[[0, 0]].norm()).abs() < 1e-10);
    assert!((result[[1, 1]].norm()).abs() < 1e-10);
}

#[test]
fn test_circuit_backed_multi_param_binding_order() {
    let mut inner = Circuit::new(1);
    inner.rx(Qubit::new(0), Parameter::symbol("alpha")).unwrap();
    inner.ry(Qubit::new(0), Parameter::symbol("beta")).unwrap();
    let frozen = Arc::new(FrozenCircuit::new(inner));
    let gate = UnitaryGate::new("MultiParam", 1, 2)
        .with_circuit(frozen)
        .unwrap();

    // alpha = PI/2, beta = 0 => RX(pi/2) then RY(0) = RX(pi/2)
    let result = gate.matrix_for_params(&[PI / 2.0, 0.0]).unwrap();
    let expected_cos = 1.0 / f64::sqrt(2.0);
    assert!((result[[0, 0]].re - expected_cos).abs() < 1e-10);
    assert!((result[[0, 0]].im).abs() < 1e-10);
    assert!((result[[0, 1]].re).abs() < 1e-10);
    assert!((result[[0, 1]].im + expected_cos).abs() < 1e-10);
    assert!((result[[1, 0]].re).abs() < 1e-10);
    assert!((result[[1, 0]].im + expected_cos).abs() < 1e-10);
    assert!((result[[1, 1]].re - expected_cos).abs() < 1e-10);
    assert!((result[[1, 1]].im).abs() < 1e-10);
}

// D. Multi-qubit expansion and integration

#[test]
fn test_three_qubit_unitary_matrix_expansion() {
    let mat = Array2::eye(8).mapv(|v| c(v, 0.0));
    let u_gate = UnitaryGate::new("Id3", 3, 0).with_matrix(mat).unwrap();

    let mut circuit = Circuit::new(3);
    circuit
        .unitary(u_gate, vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)])
        .unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_eq!(matrix.shape(), &[8, 8]);
    for i in 0..8 {
        for j in 0..8 {
            if i == j {
                assert!((matrix[[i, j]] - c(1.0, 0.0)).norm() < 1e-10);
            } else {
                assert!(matrix[[i, j]].norm() < 1e-10);
            }
        }
    }
}

#[test]
fn test_custom_unitary_in_composite_circuit() {
    let mat = array![[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)],];
    let u_gate = UnitaryGate::new("CustomX", 1, 0).with_matrix(mat).unwrap();

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.unitary(u_gate, vec![Qubit::new(1)]).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

    let matrix = circuit_to_matrix(&circuit, None).unwrap();
    assert_eq!(matrix.shape(), &[4, 4]);

    // Verify the circuit is unitary
    let conj_t = matrix.t().mapv(|x| x.conj());
    let product = conj_t.dot(&matrix);
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) };
            assert!((product[[i, j]] - expected).norm() < 1e-10);
        }
    }
}
