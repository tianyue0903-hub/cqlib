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

use super::*;
use std::f64::consts::PI;

#[test]
fn test_parse_number() {
    let p = parse_parameter("1.0").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    let p = parse_parameter("3.14").unwrap();
    assert!((p.evaluate(&None).unwrap() - 3.14).abs() < 1e-10);

    let p = parse_parameter("42").unwrap();
    assert!((p.evaluate(&None).unwrap() - 42.0).abs() < 1e-10);

    // Scientific notation
    let p = parse_parameter("1e3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1000.0).abs() < 1e-10);

    let p = parse_parameter("1.5e-2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 0.015).abs() < 1e-10);
}

#[test]
fn test_parse_constants() {
    let p = parse_parameter("pi").unwrap();
    assert!((p.evaluate(&None).unwrap() - PI).abs() < 1e-10);

    let p = parse_parameter("e").unwrap();
    assert!((p.evaluate(&None).unwrap() - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_parse_addition() {
    let p = parse_parameter("1+2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 3.0).abs() < 1e-10);

    let p = parse_parameter("1 + 2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_parse_subtraction() {
    let p = parse_parameter("5-3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);

    let p = parse_parameter("3-5").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-2.0)).abs() < 1e-10);
}

#[test]
fn test_parse_multiplication() {
    let p = parse_parameter("2*3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 6.0).abs() < 1e-10);

    let p = parse_parameter("2 * 3.5").unwrap();
    assert!((p.evaluate(&None).unwrap() - 7.0).abs() < 1e-10);
}

#[test]
fn test_parse_division() {
    let p = parse_parameter("6/2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 3.0).abs() < 1e-10);

    let p = parse_parameter("5/2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.5).abs() < 1e-10);
}

#[test]
fn test_parse_modulo() {
    let p = parse_parameter("10%3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    let p = parse_parameter("7.5 % 2.5").unwrap();
    assert!((p.evaluate(&None).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_parse_power() {
    let p = parse_parameter("2^3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 8.0).abs() < 1e-10);

    let p = parse_parameter("3^2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 9.0).abs() < 1e-10);

    let p = parse_parameter("4^0.5").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_parse_pi_expression() {
    // pi/2
    let p = parse_parameter("pi/2").unwrap();
    assert!((p.evaluate(&None).unwrap() - PI / 2.0).abs() < 1e-10);

    // pi/2+1
    let p = parse_parameter("pi/2+1").unwrap();
    assert!((p.evaluate(&None).unwrap() - (PI / 2.0 + 1.0)).abs() < 1e-10);

    // 2*pi
    let p = parse_parameter("2*pi").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0 * PI).abs() < 1e-10);
}

#[test]
fn test_parse_parentheses() {
    let p = parse_parameter("(1+2)*3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 9.0).abs() < 1e-10);

    let p = parse_parameter("1+(2*3)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 7.0).abs() < 1e-10);

    let p = parse_parameter("(1+2)*(3+4)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 21.0).abs() < 1e-10);
}

#[test]
fn test_parse_unary_minus() {
    let p = parse_parameter("-5").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-5.0)).abs() < 1e-10);

    let p = parse_parameter("-pi").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-PI)).abs() < 1e-10);

    let p = parse_parameter("3*-2").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-6.0)).abs() < 1e-10);
}

#[test]
fn test_parse_trigonometric() {
    // sin(pi/2) = 1
    let p = parse_parameter("sin(pi/2)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    // cos(0) = 1
    let p = parse_parameter("cos(0)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    // cos(pi) = -1
    let p = parse_parameter("cos(pi)").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-1.0)).abs() < 1e-10);

    // tan(0) = 0
    let p = parse_parameter("tan(0)").unwrap();
    assert!(p.evaluate(&None).unwrap().abs() < 1e-10);
}

#[test]
fn test_parse_inverse_trigonometric() {
    // asin(1) = pi/2
    let p = parse_parameter("asin(1)").unwrap();
    assert!((p.evaluate(&None).unwrap() - PI / 2.0).abs() < 1e-10);

    // acos(1) = 0
    let p = parse_parameter("acos(1)").unwrap();
    assert!(p.evaluate(&None).unwrap().abs() < 1e-10);

    // atan(1) = pi/4
    let p = parse_parameter("atan(1)").unwrap();
    assert!((p.evaluate(&None).unwrap() - PI / 4.0).abs() < 1e-10);
}

#[test]
fn test_parse_other_functions() {
    // abs(-5) = 5
    let p = parse_parameter("abs(-5)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 5.0).abs() < 1e-10);

    // sqrt(4) = 2
    let p = parse_parameter("sqrt(4)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);

    // exp(0) = 1
    let p = parse_parameter("exp(0)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    // ln(e) = 1
    let p = parse_parameter("ln(e)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    // sign(-10) = -1
    let p = parse_parameter("sign(-10)").unwrap();
    assert!((p.evaluate(&None).unwrap() - (-1.0)).abs() < 1e-10);

    // sign(10) = 1
    let p = parse_parameter("sign(10)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_parse_log() {
    // log(100, 10) = 2
    let p = parse_parameter("log(100, 10)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);

    // log(8, 2) = 3
    let p = parse_parameter("log(8, 2)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_parse_nested_functions() {
    // sin(cos(0)) = sin(1)
    let p = parse_parameter("sin(cos(0))").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0f64.sin()).abs() < 1e-10);

    // sqrt(sqrt(16)) = 2
    let p = parse_parameter("sqrt(sqrt(16))").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_parse_complex() {
    // Complex expression: pi/2+1
    let p = parse_parameter("pi/2+1").unwrap();
    assert!((p.evaluate(&None).unwrap() - (PI / 2.0 + 1.0)).abs() < 1e-10);

    // sin(pi/4) * 2
    let p = parse_parameter("sin(pi/4)*2").unwrap();
    assert!((p.evaluate(&None).unwrap() - (PI / 4.0).sin() * 2.0).abs() < 1e-10);

    // 2*sin(pi/6) = 1
    let p = parse_parameter("2*sin(pi/6)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_parse_symbol() {
    let p = parse_parameter("theta").unwrap();
    assert_eq!(p.get_symbols(), vec!["theta"]);

    let p = parse_parameter("x + y").unwrap();
    let symbols = p.get_symbols();
    assert!(symbols.contains(&"x".to_string()));
    assert!(symbols.contains(&"y".to_string()));
}

#[test]
fn test_parse_symbolic_evaluation() {
    use std::collections::HashMap;

    let p = parse_parameter("x + 2 * y").unwrap();
    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 3.0);
    bindings.insert("y".to_string(), 4.0);
    assert!((p.evaluate(&Some(bindings)).unwrap() - 11.0).abs() < 1e-10);
}

#[test]
fn test_parse_function_with_symbol() {
    let p = parse_parameter("sin(theta)").unwrap();
    assert_eq!(p.get_symbols(), vec!["theta"]);
}

#[test]
fn test_parse_errors() {
    // Empty expression
    assert!(matches!(
        parse_parameter(""),
        Err(ParseError::EmptyExpression)
    ));

    // Invalid character
    assert!(parse_parameter("1&2").is_err());

    // Mismatched parentheses
    assert!(matches!(
        parse_parameter("(1+2"),
        Err(ParseError::MismatchedParentheses)
    ));

    // Mismatched parentheses - extra closing
    assert!(parse_parameter("1+2)").is_err());

    // Unknown function
    assert!(parse_parameter("foo(1)").is_err());
}

#[test]
fn test_operator_precedence() {
    // Multiplication before addition
    let p = parse_parameter("2+3*4").unwrap();
    assert!((p.evaluate(&None).unwrap() - 14.0).abs() < 1e-10);

    // Power before multiplication
    let p = parse_parameter("2*3^2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 18.0).abs() < 1e-10);

    // Parentheses override precedence
    let p = parse_parameter("(2+3)*4").unwrap();
    assert!((p.evaluate(&None).unwrap() - 20.0).abs() < 1e-10);
}

#[test]
fn test_display_format() {
    let p = parse_parameter("1+2").unwrap();
    assert_eq!(p.to_string(), "1 + 2");

    let p = parse_parameter("pi/2").unwrap();
    assert_eq!(p.to_string(), "π / 2");

    let p = parse_parameter("sin(x)").unwrap();
    assert_eq!(p.to_string(), "sin(x)");
}

#[test]
fn test_complex_real_world() {
    // RXY gate parameters: theta = pi/2 + 1, phi = 3.14
    let theta = parse_parameter("pi/2+1").unwrap();
    let phi = parse_parameter("3.14").unwrap();

    assert!((theta.evaluate(&None).unwrap() - (PI / 2.0 + 1.0)).abs() < 1e-10);
    assert!((phi.evaluate(&None).unwrap() - 3.14).abs() < 1e-10);

    // RZ(pi/4 + 0.5)
    let rz_param = parse_parameter("pi/4+0.5").unwrap();
    assert!((rz_param.evaluate(&None).unwrap() - (PI / 4.0 + 0.5)).abs() < 1e-10);

    // RX(2*theta) with symbolic theta
    let rx_param = parse_parameter("2*theta").unwrap();
    assert_eq!(rx_param.get_symbols(), vec!["theta"]);
}

#[test]
fn test_scientific_notation() {
    let p = parse_parameter("1e-5").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1e-5).abs() < 1e-15);

    let p = parse_parameter("1.2E5").unwrap();
    assert!((p.evaluate(&None).unwrap() - 120000.0).abs() < 1e-10);
}

#[test]
fn test_functions() {
    let p = parse_parameter("sin(0)").unwrap();
    assert!(p.evaluate(&None).unwrap().abs() < 1e-10);

    let p = parse_parameter("cos(pi)").unwrap();
    assert!((p.evaluate(&None).unwrap() - -1.0).abs() < 1e-10);

    let p = parse_parameter("sqrt(4)").unwrap();
    assert!((p.evaluate(&None).unwrap() - 2.0).abs() < 1e-10);

    let p = parse_parameter("log(8, 2)").unwrap(); // log2(8) = 3
    assert!((p.evaluate(&None).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_pow_mod() {
    let p = parse_parameter("2^3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 8.0).abs() < 1e-10);

    let p = parse_parameter("5 % 2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 1.0).abs() < 1e-10);

    // Right associativity: 2^3^2 = 2^(3^2) = 2^9 = 512
    let p = parse_parameter("2^3^2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 512.0).abs() < 1e-10);
}

#[test]
fn test_complex_precedence() {
    // 1 + 2 * 3 ^ 2 = 1 + 2 * 9 = 1 + 18 = 19
    let p = parse_parameter("1 + 2 * 3 ^ 2").unwrap();
    assert!((p.evaluate(&None).unwrap() - 19.0).abs() < 1e-10);

    // (1+2)*3 = 9
    let p = parse_parameter("(1+2)*3").unwrap();
    assert!((p.evaluate(&None).unwrap() - 9.0).abs() < 1e-10);
}

#[test]
fn test_try_from_parameter() {
    use crate::circuit::parameter::impls::Parameter;

    // Valid expressions should work
    let p1 = Parameter::try_from("theta");
    assert!(p1.is_ok());
    assert_eq!(p1.unwrap().to_string(), "theta");

    let p2 = Parameter::try_from("pi/2 + 1");
    assert!(p2.is_ok());

    let p3 = Parameter::try_from("sin(x)");
    assert!(p3.is_ok());

    // Invalid expressions should return Err
    let err1 = Parameter::try_from("");
    assert!(err1.is_err());

    let err2 = Parameter::try_from("1&2");
    assert!(err2.is_err());

    let err3 = Parameter::try_from("(1+2");
    assert!(err3.is_err());

    let err4 = Parameter::try_from("foo(1)");
    assert!(err4.is_err());

    // String version
    let p5 = Parameter::try_from("lambda".to_string());
    assert!(p5.is_ok());
    assert_eq!(p5.unwrap().to_string(), "lambda");

    let err5 = Parameter::try_from("invalid@expr".to_string());
    assert!(err5.is_err());
}
