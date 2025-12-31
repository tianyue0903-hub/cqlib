// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
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
use std::f64::consts;

#[test]
fn test_sin() {
    let x = ExprNode::Symbol("x".to_string());
    let sin_x = ExprNode::Sin(Arc::from(x));
    assert_eq!(format!("{}", sin_x), "sin(x)");

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), consts::PI);
    let v = sin_x.evaluate(&bindings).unwrap();

    assert!(v.abs() < 1e-10, "failed");
    assert!(sin_x.symbols().contains("x"));
}

#[test]
fn test_cos() {
    let theta = ExprNode::Symbol("theta".to_string());
    let cos_x = ExprNode::Cos(Arc::from(theta));
    assert_eq!(cos_x.to_string(), "cos(theta)");

    let acos_x = ExprNode::ACos(Arc::from(cos_x));
    assert_eq!(acos_x.to_string(), "acos(cos(theta))");

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), consts::PI);
    let v = acos_x.evaluate(&bindings).unwrap();
    assert_eq!(v, consts::PI);
}

#[test]
fn test_tan() {
    let theta = ExprNode::Symbol("theta".to_string());
    let exp = ExprNode::Tan(Arc::from(theta));
    assert_eq!(exp.to_string(), "tan(theta)");

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), 0.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 0.0);

    let exp = ExprNode::ATan(Arc::from(exp));
    assert_eq!(exp.to_string(), "atan(tan(theta))");

    bindings.insert("theta".to_string(), consts::PI / 3.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, consts::PI / 3.0);
    assert!(exp.symbols().contains("theta"));
}

#[test]
fn test_add() {
    let x = ExprNode::Symbol("x".to_string());
    let y = ExprNode::Float(1.0);
    let add = ExprNode::Add(Arc::from(x.clone()), Arc::from(y));
    assert_eq!(add.to_string(), "x + 1");
    assert_eq!(add.to_string(), format!("{}", add));

    let exp = ExprNode::Mod(Arc::from(add), Arc::from(x));
    assert_eq!(exp.to_string(), "(x + 1) % x");

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 1.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 0.0);
}

#[test]
fn test_mul() {
    let x = ExprNode::Symbol("x".to_string());
    let y = ExprNode::Float(2.0);
    let exp = ExprNode::Mul(Arc::from(x.clone()), Arc::from(y));
    assert_eq!(exp.to_string(), "x * 2");

    let exp = ExprNode::Mul(Arc::from(exp), Arc::from(x));
    assert_eq!(exp.to_string(), "x * 2 * x");

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 2.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 8.0);
}

#[test]
fn test_exp() {
    let x = ExprNode::Symbol("x".to_string());
    let exp = ExprNode::Exp(Arc::from(x));
    assert_eq!(exp.to_string(), "exp(x)");

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 1.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, consts::E);

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 0.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 1.0);
}

#[test]
fn test_log() {
    let x = ExprNode::Symbol("x".to_string());
    let base = ExprNode::Symbol("base".to_string());
    let exp = ExprNode::Log(Arc::from(x), Arc::from(base));
    assert_eq!(exp.to_string(), "log(x, base)");
    assert!(exp.symbols().contains("x"));
    assert!(exp.symbols().contains("base"));

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), consts::E);
    bindings.insert("base".to_string(), consts::E);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 1.0);

    bindings.insert("x".to_string(), 1.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 0.0);
}

#[test]
fn test_sqrt() {
    let x = ExprNode::Symbol("x".to_string());
    let exp = ExprNode::Sqrt(Arc::from(x));
    assert_eq!(exp.to_string(), "sqrt(x)");

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 4.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 2.0);

    let exp = ExprNode::Mul(Arc::from(exp.clone()), Arc::from(exp.clone()));
    assert_eq!(exp.to_string(), "sqrt(x) * sqrt(x)");
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 4.0);
}

#[test]
fn test_pow() {
    let x = ExprNode::Symbol("x".to_string());
    let base = ExprNode::Symbol("base".to_string());
    let exp = ExprNode::Pow(Arc::from(base), Arc::from(x));
    assert_eq!(exp.to_string(), "base^x");
    assert!(exp.symbols().contains("base"));

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 0.0);
    bindings.insert("base".to_string(), 3.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 1.0);

    bindings.insert("x".to_string(), 1.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 3.0);
}

#[test]
fn test_div() {
    let x = ExprNode::Symbol("x".to_string());
    let y = ExprNode::Symbol("y".to_string());
    let exp = ExprNode::Div(Arc::from(x), Arc::from(y));
    assert_eq!(exp.to_string(), "x / y");
    assert!(exp.symbols().contains("x"));
    assert!(exp.symbols().contains("y"));

    let mut bindings = HashMap::new();
    bindings.insert("x".to_string(), 0.0);
    bindings.insert("y".to_string(), 3.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 0.0);

    bindings.insert("x".to_string(), 6.0);
    let v = exp.evaluate(&bindings).unwrap();
    assert_eq!(v, 2.0);
}
