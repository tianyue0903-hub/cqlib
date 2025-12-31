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
use std::f64::consts;

#[test]
fn test_add_num() {
    // 5 + 10 = 15
    let s = Parameter::from(5_f64) + Parameter::from(10_f64);
    let r = s.simplify(None);
    assert_eq!(r, Parameter::from(15_f64));

    // (5 + 10) + (9 + 1) = 25
    let s = (Parameter::from(5_f64) + Parameter::from(10_f64))
        + (Parameter::from(9_f64) + Parameter::from(1_f64));
    let r = s.simplify(None);
    assert_eq!(r, Parameter::from(25_f64));
}

#[test]
fn test_add_zero() {
    let x = Parameter::from("x");
    let z = Parameter::from(0_f64);

    // x + 0 = x
    let s = x.clone() + z.clone();
    let r = s.simplify(None);
    assert_eq!(r, x);

    // 0 + x = x
    let s = z.clone() + x.clone();
    let r = s.simplify(None);
    assert_eq!(r, x);

    // x + 0 + x = 2x
    let s = x.clone() + z.clone() + x.clone();
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Integer(2_i64)),
            x.node.clone()
        ))
    );
}

#[test]
fn test_add_self() {
    let x = Parameter::from("x");

    // x + x = 2x
    let s = x.clone() + x.clone();
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Integer(2_i64)),
            x.node.clone()
        ))
    );

    // 2x + x = 3x
    let s = x.clone() + x.clone() + x.clone();
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(3_f64)),
            x.node.clone()
        ))
    );
    // x + 2x = 3x
    let s = x.clone() + (x.clone() + x.clone());
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(3_f64)),
            x.node.clone()
        )),
    );

    // 2.1x + x = 3.1x
    let s = x.clone() * 2.1_f64 + x.clone();
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(3.1_f64)),
            x.node.clone()
        )),
    );

    // 3.1x + 2x + 6x= 11.1x
    let s = x.clone() * 3.1_f64 + x.clone() * 2_f64 + x.clone() * 6_f64;
    let r = s.simplify(None);
    assert_eq!(r.to_string(), "11.1 * x");
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(11.1_f64)),
            x.node.clone()
        )),
    );
}

#[test]
fn test_sub_zero() {
    let x = Parameter::from("x");
    let z = Parameter::from(0_f64);

    // x - 0
    let s = x.clone() - z.clone();
    let r = s.simplify(None);
    assert_eq!(r, x);

    // 0 + 2x
    let s = z.clone() + x.clone() * 2_i64;
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Integer(2_i64)),
            x.node.clone()
        )),
    );
}

#[test]
fn test_sub_num() {
    let x = Parameter::from("x");

    // x - 2x = -1x
    let s = x.clone() - x.clone() * 2_i64;
    let r = s.simplify(None);
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(-1_f64)),
            x.node.clone()
        )),
    );

    // x - x
    let s = x.clone() - x.clone();
    let r = s.simplify(None);
    assert_eq!(r, Parameter::from(0_i64));

    // 9.3x - (-0.2x) = 9.5x
    let s = 9.3_f64 * x.clone() - (-0.2_f64) * x.clone();
    assert_eq!(s.to_string(), "9.3 * x - (-0.2 * x)");
    let r = s.simplify(None);
    assert_eq!(r.to_string(), "9.5 * x");
    assert_eq!(
        r,
        Parameter::new(ExprNode::Mul(
            Arc::new(ExprNode::Float(9.5_f64)),
            x.node.clone()
        )),
    );
}

#[test]
fn test_sin_num() {
    // sin(pi) = 0
    let pi = Parameter::from(consts::PI);
    let s = pi.sin();
    assert_eq!(s.to_string(), "sin(3.141592653589793)");
    let exp = s.simplify(None);
    assert!(exp.evaluate(&None).unwrap().abs() < f64::EPSILON);

    let pi = Parameter::pi();
    let exp = pi.sin() + pi.sin();
    assert_eq!(exp.to_string(), "sin(π) + sin(π)");
    let exp = exp.simplify(None);
    assert_eq!(exp.to_string(), "0")
}

#[test]
fn test_cos_num() {
    // cos(pi) = -1
    let x = Parameter::from(consts::PI);
    let s = x.cos();
    assert_eq!(s.to_string(), "cos(3.141592653589793)");
    let exp = s.simplify(None);
    assert_eq!(exp.evaluate(&None).unwrap(), -1.0);
}

#[test]
fn test_tan() {
    let pi = Parameter::pi();
    let s = pi.tan();
    assert_eq!(s.to_string(), "tan(π)");
    let exp = s.simplify(None);
    assert!(exp.evaluate(&None).unwrap().abs() < f64::EPSILON);

    let x = Parameter::from("x");
    let s = x.tan().atan();
    assert_eq!(s.to_string(), "atan(tan(x))");
    let exp = s.simplify(None);
    assert_eq!(exp.to_string(), "x");
}
