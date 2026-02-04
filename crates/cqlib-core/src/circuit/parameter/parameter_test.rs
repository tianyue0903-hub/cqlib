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
use std::collections::HashMap;
use std::f64::consts::{E, PI};

#[test]
fn test_parameter_construction() {
    // From string
    let p1 = Parameter::from("theta");
    assert_eq!(p1.to_string(), "theta");

    // From f64
    let p2 = Parameter::from(3.14);
    assert_eq!(p2.to_string(), "3.14");

    // From integer
    let p3 = Parameter::from(42);
    assert_eq!(p3.to_string(), "42");

    // Default
    let p_default = Parameter::default();
    assert_eq!(p_default.to_string(), "0");

    // Parameter::symbol
    let p_sym = Parameter::symbol("phi");
    assert_eq!(p_sym.to_string(), "phi");
}

#[test]
fn test_parameter_constants() {
    let pi = Parameter::pi();
    assert_eq!(pi.to_string(), "π");
    assert_eq!(pi.evaluate(&None).unwrap(), PI);

    let e = Parameter::e();
    assert_eq!(e.to_string(), "e");
    assert_eq!(e.evaluate(&None).unwrap(), E);
}

#[test]
fn test_parameter_arithmetic_ops() {
    let theta = Parameter::from("theta");
    let phi = Parameter::from("phi");
    let val = Parameter::from(2.0);

    // Add
    let add = theta.clone() + phi.clone();
    assert_eq!(add.to_string(), "theta + phi");

    // Sub
    let sub = theta.clone() - phi.clone();
    assert_eq!(sub.to_string(), "theta - phi");

    // Mul
    let mul = theta.clone() * val.clone();
    assert_eq!(mul.to_string(), "theta * 2");

    // Div
    let div = theta.clone() / val.clone();
    assert_eq!(div.to_string(), "theta / 2");

    // Rem (Mod)
    let rem = theta.clone() % val.clone();
    assert_eq!(rem.to_string(), "theta % 2");
}

#[test]
fn test_parameter_arithmetic_primitive_ops() {
    let theta = Parameter::from("theta");

    // Parameter + f64
    let res: Parameter = theta.clone() + 1.5;
    assert_eq!(res.to_string(), "theta + 1.5");

    // f64 + Parameter
    let res: Parameter = 1.5 + theta.clone();
    assert_eq!(res.to_string(), "1.5 + theta");

    // Parameter - i64
    let res: Parameter = theta.clone() - 10;
    assert_eq!(res.to_string(), "theta - 10");

    // i64 - Parameter
    let res: Parameter = 10 - theta.clone();
    assert_eq!(res.to_string(), "10 - theta");

    // Parameter * f32
    let res: Parameter = theta.clone() * 2.0f32;
    assert_eq!(res.to_string(), "theta * 2");

    // u32 * Parameter
    let res: Parameter = 5u32 * theta.clone();
    assert_eq!(res.to_string(), "5 * theta");

    // Parameter / i32
    let res: Parameter = theta.clone() / 2;
    assert_eq!(res.to_string(), "theta / 2");
}

#[test]
fn test_parameter_reference_ops() {
    let p1 = Parameter::from("p1");
    let p2 = Parameter::from("p2");

    // &Parameter + &Parameter
    let res = &p1 + &p2;
    assert_eq!(res.to_string(), "p1 + p2");

    // Parameter + &Parameter
    let res = p1.clone() + &p2;
    assert_eq!(res.to_string(), "p1 + p2");

    // &Parameter + Parameter
    let res = &p1 + p2.clone();
    assert_eq!(res.to_string(), "p1 + p2");
}

#[test]
fn test_parameter_functions() {
    let x = Parameter::from("x");

    assert_eq!(x.sin().to_string(), "sin(x)");
    assert_eq!(x.cos().to_string(), "cos(x)");
    assert_eq!(x.tan().to_string(), "tan(x)");
    assert_eq!(x.asin().to_string(), "asin(x)");
    assert_eq!(x.acos().to_string(), "acos(x)");
    assert_eq!(x.atan().to_string(), "atan(x)");
    assert_eq!(x.exp().to_string(), "exp(x)");
    assert_eq!(x.ln().to_string(), "ln(x)");
    assert_eq!(x.abs().to_string(), "abs(x)");
    assert_eq!(x.sqrt().to_string(), "sqrt(x)");

    let y = Parameter::from("y");
    assert_eq!(x.pow(&y).to_string(), "x^y");

    // Log with base
    let base = Parameter::from(10.0);
    assert_eq!(x.log(Some(base)).to_string(), "log(x, 10)");
    // Log without base (ln)
    assert_eq!(x.log(None).to_string(), "ln(x)");
}

#[test]
fn test_parameter_evaluation() {
    let x = Parameter::from("x");
    let expr: Parameter = x.clone() * 2.0 + 1.0; // x * 2 + 1

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 3.0);

    let res = expr.evaluate(&Some(bindings)).unwrap();
    assert_eq!(res, 7.0);

    // Missing symbol
    let empty_bindings = HashMap::new();
    let err = expr.evaluate(&Some(empty_bindings));
    assert!(err.is_err());
}

#[test]
fn test_parameter_get_symbols() {
    let x = Parameter::from("x");
    let y = Parameter::from("y");
    let z = Parameter::from("z");

    let expr = (x + y) * z;
    let symbols = expr.get_symbols();

    assert_eq!(symbols.len(), 3);
    assert_eq!(symbols, vec!["x", "y", "z"]);

    // Test caching: call again, should hit cache (though internally opaque)
    let symbols2 = expr.get_symbols();
    assert_eq!(symbols2, symbols);
}

#[test]
fn test_parameter_simplify() {
    let x = Parameter::from("x");

    // 0 + x -> x
    let expr = Parameter::from(0) + x.clone();
    let simplified = expr.simplify(None);
    assert_eq!(simplified.to_string(), "x");

    // x * 1 -> x
    let expr: Parameter = x.clone() * 1.0;
    let simplified = expr.simplify(None);
    assert_eq!(simplified.to_string(), "x");
}

#[test]
fn test_parameter_derivative() {
    let x = Parameter::from("x");
    // d(x^2)/dx = 2*x
    let expr = x.pow(&Parameter::from(2.0));
    let deriv = expr.derivative("x").simplify(None);

    // x^2 -> 2 * x^(2-1) * 1 = 2 * x
    // Exact string match might depend on simplification order, checking evaluation
    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 3.0);
    let val = deriv.evaluate(&Some(bindings)).unwrap();
    assert_eq!(val, 6.0);
}

#[test]
fn test_parameter_replace() {
    let mut p = Parameter::from("x") + Parameter::from("y");
    let z = Parameter::from("z");

    let new_p = p.replace("x", &z);
    assert_eq!(new_p.to_string(), "z + y");
}

#[test]
fn test_parameter_equality() {
    let p1: Parameter = Parameter::from("x") + 1.0;
    let p2: Parameter = Parameter::from("x") + 1.0;
    let p3: Parameter = Parameter::from("x") + 2.0;

    assert_eq!(p1, p2);
    assert_ne!(p1, p3);
}
