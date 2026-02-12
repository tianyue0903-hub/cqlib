use super::*;
use crate::circuit::Circuit;
use crate::circuit::Qubit;
use crate::circuit::param::ParameterValue;
use crate::circuit::parameter::Parameter;

#[test]
fn test_dump_simple_gates() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.x(q0).unwrap();
    c.y(q1).unwrap();
    c.z(q0).unwrap();
    c.h(q1).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"X Q0
Y Q1
Z Q0
H Q1
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_native_qcis_gates() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.x2p(q0).unwrap();
    c.x2m(q1).unwrap();
    c.cz(q0, q1).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"X2P Q0
X2M Q1
CZ Q0 Q1
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_xy_gates() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.xy2p(q0, 1.0).unwrap();
    c.xy2m(q1, std::f64::consts::PI / 2.0).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"XY2P Q0 1
XY2M Q1 pi/2
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_parameterized_gates() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.rx(q0, 1.0).unwrap();
    c.ry(q1, std::f64::consts::PI / 2.0).unwrap();
    c.rz(q0, std::f64::consts::PI).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"RX Q0 1
RY Q1 pi/2
RZ Q0 pi
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_rxy_gate() {
    let mut c = Circuit::new(1);
    let q0 = Qubit::new(0);

    c.rxy(q0, 1.0, 2.0).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = "RXY Q0 1 2\n";
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_s_and_t_gates() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.s(q0).unwrap();
    c.sdg(q1).unwrap();
    c.t(q0).unwrap();
    c.tdg(q1).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"S Q0
SD Q1
T Q0
TD Q1
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_measurement() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.measure(q0).unwrap();
    c.measure(q1).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"M Q0
M Q1
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_barrier() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.barrier(vec![q0, q1]).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = "B Q0 Q1\n";
    assert_eq!(qcis, expected);
}

#[test]
fn test_dump_delay() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.delay(q0, ParameterValue::Fixed(100.0)).unwrap();
    c.delay(q1, ParameterValue::Fixed(50.0)).unwrap();

    let qcis = dumps(&c).unwrap();
    let expected = r#"I Q0 100
I Q1 50
"#;
    assert_eq!(qcis, expected);
}

#[test]
fn test_format_float() {
    assert_eq!(format_float(0.0), "0");
    assert_eq!(format_float(1.0), "1");
    assert_eq!(format_float(-1.0), "-1");
    assert_eq!(format_float(std::f64::consts::PI), "pi");
    assert_eq!(format_float(std::f64::consts::PI / 2.0), "pi/2");
    assert_eq!(format_float(-std::f64::consts::PI / 2.0), "-pi/2");
    assert_eq!(format_float(std::f64::consts::PI / 4.0), "pi/4");
    assert_eq!(format_float(42.0), "42");
    assert_eq!(format_float(3.14159), "3.14159");
}

#[test]
fn test_roundtrip_load_dump() {
    use crate::ir::qcis::load::loads;

    let original_qcis = r#"
        RX Q0 1.0
        RY Q1 pi/2
        CZ Q0 Q1
        M Q0 Q1
    "#;

    let circuit = loads(original_qcis).unwrap();
    let dumped_qcis = dumps(&circuit).unwrap();

    // Load the dumped QCIS and verify it works
    let circuit2 = loads(&dumped_qcis).unwrap();
    assert_eq!(circuit2.num_qubits(), 2);
}

#[test]
fn test_unsupported_gate_cx() {
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    c.cx(q0, q1).unwrap();

    // CX is not natively supported by QCIS
    let result = dumps(&c);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("CX"));
    assert!(err_msg.contains("compile"));
}

#[test]
fn test_dump_symbolic_parameters() {
    // Test that symbolic parameters are exported for visual inspection
    // Note: This is for visualization only, QCIS backend does not support symbolic parameters
    let mut c = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);

    let theta = Parameter::try_from("theta").unwrap();
    let phi = Parameter::try_from("phi").unwrap();

    c.rx(q0, theta.clone()).unwrap();
    c.ry(q1, phi.clone()).unwrap();
    c.rz(q0, theta.clone() + 0.5).unwrap();
    c.rxy(q1, theta.clone(), phi.clone()).unwrap();

    let qcis = dumps(&c).unwrap();

    // Verify symbolic parameters appear in output
    let expected = r#"RX Q0 theta
RY Q1 phi
RZ Q0 theta + 0.5
RXY Q1 theta phi
"#;
    assert_eq!(qcis, expected);
}
