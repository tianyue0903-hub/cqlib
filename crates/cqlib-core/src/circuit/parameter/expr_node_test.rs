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

#[test]
fn test_evaluate_partial_edge_cases() {
    let empty_bindings = HashMap::new();

    // ASin(0) -> 0
    let asin_0 = ExprNode::ASin(Arc::new(ExprNode::Integer(0)));
    let res = asin_0.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Integer(0));

    // ASin(1) -> PI/2
    let asin_1 = ExprNode::ASin(Arc::new(ExprNode::Integer(1)));
    let res = asin_1.evaluate_partial(&empty_bindings).unwrap();
    if let ExprNode::Float(v) = res {
        assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    } else {
        panic!("Expected Float(PI/2)");
    }

    // Cos(Pi) -> -1.0
    let cos_pi = ExprNode::Cos(Arc::new(ExprNode::Pi));
    let res = cos_pi.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Float(-1.0));

    // Cos(0) -> 1
    let cos_0 = ExprNode::Cos(Arc::new(ExprNode::Integer(0)));
    let res = cos_0.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Integer(1));

    // ACos(1) -> 0
    let acos_1 = ExprNode::ACos(Arc::new(ExprNode::Integer(1)));
    let res = acos_1.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Integer(0));
    
    // Tan(Pi) -> 0.0
    let tan_pi = ExprNode::Tan(Arc::new(ExprNode::Pi));
    let res = tan_pi.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Float(0.0));

    // ATan(0) -> 0
    let atan_0 = ExprNode::ATan(Arc::new(ExprNode::Integer(0)));
    let res = atan_0.evaluate_partial(&empty_bindings).unwrap();
    assert_eq!(res, ExprNode::Integer(0));
}

#[test]
fn test_partial_substitution_scenarios() {
    let a = ExprNode::Symbol("a".to_string());
    let b = ExprNode::Symbol("b".to_string());
    let c = ExprNode::Symbol("c".to_string());

    // Case 1: a + b, set a = 1.0 -> 1.0 + b
    let expr1 = ExprNode::Add(Arc::new(a.clone()), Arc::new(b.clone()));
    let mut bindings1 = HashMap::new();
    bindings1.insert("a".to_string(), 1.0);
    let res1 = expr1.evaluate_partial(&bindings1).unwrap();
    match res1 {
        ExprNode::Add(lhs, rhs) => {
            assert_eq!(*lhs, ExprNode::Float(1.0));
            assert_eq!(*rhs, b);
        }
        _ => panic!("Expected Add(Float(1.0), Symbol(b)), got {:?}", res1),
    }

    // Case 2: a * b * c, set a = 2.0, b = 3.0 -> 6.0 * c
    // (a * b) * c
    let expr2 = ExprNode::Mul(
        Arc::new(ExprNode::Mul(Arc::new(a.clone()), Arc::new(b.clone()))),
        Arc::new(c.clone())
    );
    let mut bindings2 = HashMap::new();
    bindings2.insert("a".to_string(), 2.0);
    bindings2.insert("b".to_string(), 3.0);
    let res2 = expr2.evaluate_partial(&bindings2).unwrap();
    // (2.0 * 3.0) * c -> 6.0 * c
    match res2 {
        ExprNode::Mul(lhs, rhs) => {
            assert_eq!(*lhs, ExprNode::Float(6.0));
            assert_eq!(*rhs, c);
        }
        _ => panic!("Expected Mul(Float(6.0), Symbol(c)), got {:?}", res2),
    }

    // Case 3: sin(a + b), set b = 0 -> sin(a)
    let expr3 = ExprNode::Sin(Arc::new(ExprNode::Add(Arc::new(a.clone()), Arc::new(b.clone()))));
    let mut bindings3 = HashMap::new();
    bindings3.insert("b".to_string(), 0.0);
    let res3 = expr3.evaluate_partial(&bindings3).unwrap();
    // inner: a + 0 -> a
    // outer: sin(a)
    match res3 {
        ExprNode::Sin(inner) => {
            assert_eq!(*inner, a);
        }
        _ => panic!("Expected Sin(Symbol(a)), got {:?}", res3),
    }

    // Case 4: x * 0 + y, set x = 5 -> y
    let x = ExprNode::Symbol("x".to_string());
    let y = ExprNode::Symbol("y".to_string());
    // (x * 0) + y
    let expr4 = ExprNode::Add(
        Arc::new(ExprNode::Mul(Arc::new(x.clone()), Arc::new(ExprNode::Integer(0)))),
        Arc::new(y.clone())
    );
    let mut bindings4 = HashMap::new();
    bindings4.insert("x".to_string(), 5.0); // substitution
    let res4 = expr4.evaluate_partial(&bindings4).unwrap();
    // 5.0 * 0 -> 0 (Integer or Float)
    // 0 + y -> y
    assert_eq!(res4, y);
}
