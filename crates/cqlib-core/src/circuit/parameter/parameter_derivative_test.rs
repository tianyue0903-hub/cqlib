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

use crate::circuit::parameter::impls::Parameter;

#[test]
fn test_derivative_constant() {
    let x = "x";

    // d(5)/dx = 0
    let c1 = Parameter::from(5);
    let d_c1 = c1.derivative(x);
    assert_eq!(d_c1, Parameter::from(0));

    // d(3.14)/dx = 0
    let c2 = Parameter::from(3.14);
    let d_c2 = c2.derivative(x);
    assert_eq!(d_c2, Parameter::from(0));

    // d(PI)/dx = 0
    let pi = Parameter::pi();
    let d_pi = pi.derivative(x);
    assert_eq!(d_pi, Parameter::from(0));

    // d(E)/dx = 0
    let e = Parameter::e();
    let d_e = e.derivative(x);
    assert_eq!(d_e, Parameter::from(0));
}

#[test]
fn test_derivative_symbol() {
    let x = Parameter::try_from("x").unwrap();
    let y = Parameter::try_from("y").unwrap();

    // d(x)/dx = 1
    let dx_dx = x.derivative("x");
    assert_eq!(dx_dx, Parameter::from(1));

    // d(y)/dx = 0
    let dy_dx = y.derivative("x");
    assert_eq!(dy_dx, Parameter::from(0));
}

#[test]
fn test_derivative_negation() {
    let x = Parameter::try_from("x").unwrap();
    // d(-x)/dx = -1
    // Workaround: Parameter doesn't implement Neg, so use 0 - x or -1 * x
    let neg_x = Parameter::from(0) - x.clone();
    let d_neg_x = neg_x.derivative("x").simplify(None);
    assert_eq!(d_neg_x, Parameter::from(-1.0));
}

#[test]
fn test_derivative_add_sub() {
    let x = Parameter::try_from("x").unwrap();
    let y = Parameter::try_from("y").unwrap();

    // d(x + y)/dx = 1 + 0 = 1
    let add = x.clone() + y.clone();
    let d_add = add.derivative("x").simplify(None);
    assert_eq!(d_add, Parameter::from(1));

    // d(x - 2x)/dx = 1 - 2 = -1
    let sub = x.clone() - Parameter::from(2.0) * x.clone();
    let d_sub = sub.derivative("x").simplify(None);
    assert_eq!(d_sub, Parameter::from(-1.0));
}

#[test]
fn test_derivative_multiplication() {
    let x = Parameter::try_from("x").unwrap();

    // d(3x)/dx = 3
    let mul_const = Parameter::from(3.0) * x.clone();
    let d_mul_const = mul_const.derivative("x");
    let simplified = d_mul_const.simplify(None);
    // After simplification, it should likely be 3 or 3 * 1 + 0 * x = 3
    assert_eq!(simplified, Parameter::from(3.0));

    // d(x * x)/dx = 1*x + x*1 = 2x
    let square = x.clone() * x.clone();
    let d_square = square.derivative("x").simplify(None);
    assert_eq!(d_square.to_string(), "2 * x");
}

#[test]
fn test_derivative_division() {
    let x = Parameter::try_from("x").unwrap();

    // d(x / 2)/dx = 1/2 = 0.5
    let div_const: Parameter = x.clone() / 2.0;
    let d_div_const = div_const.derivative("x").simplify(None);
    assert_eq!(d_div_const, Parameter::from(0.5));

    // d(1 / x)/dx = -1/x^2
    let inv_x = Parameter::from(1.0) / x.clone();
    let d_inv_x = inv_x.derivative("x").simplify(None);
    assert_eq!(d_inv_x.to_string(), "-1 / (x^2)");
}

#[test]
fn test_derivative_power_constructed() {
    let x = Parameter::try_from("x").unwrap();
    let three = Parameter::from(3.0);

    // x^3
    let x_cubed = x.pow(&three);

    // d(x^3)/dx = 3x^2
    let deriv = x_cubed.derivative("x").simplify(None);
    assert_eq!(deriv.to_string(), "3 * x^2");

    // 2^x
    let two = Parameter::from(2.0);
    let two_pow_x = two.pow(&x);

    // d(2^x)/dx = 2^x * ln(2)
    let deriv_2px = two_pow_x.derivative("x").simplify(None);
    // ln(2) = 0.6931471805599453
    assert_eq!(deriv_2px.to_string(), "0.6931471805599453 * 2^x");
}

#[test]
fn test_derivative_trig() {
    let x = Parameter::try_from("x").unwrap();

    // d(sin(x))/dx = cos(x)
    let sin_x = x.sin();
    let d_sin = sin_x.derivative("x").simplify(None);
    assert_eq!(d_sin.to_string(), "cos(x)");

    // d(cos(x))/dx = -sin(x)
    let cos_x = x.cos();
    let d_cos = cos_x.derivative("x").simplify(None);
    assert_eq!(d_cos.to_string(), "-sin(x)");

    // d(tan(x))/dx = sec^2(x) = 1/cos^2(x)
    let tan_x = x.tan();
    let d_tan = tan_x.derivative("x").simplify(None);
    assert_eq!(d_tan.to_string(), "1 / (cos(x)^2)");
}

#[test]
fn test_derivative_inverse_trig() {
    let x = Parameter::try_from("x").unwrap();

    // d(asin(x))/dx = 1 / sqrt(1 - x^2)
    let asin_x = x.asin();
    let d_asin = asin_x.derivative("x").simplify(None);
    assert_eq!(d_asin.to_string(), "1 / sqrt(1 - x^2)");

    // d(acos(x))/dx = -1 / sqrt(1 - x^2)
    let acos_x = x.acos();
    let d_acos = acos_x.derivative("x");
    assert_eq!(d_acos.to_string(), "-1 / sqrt(1 - x^2)");

    // d(atan(x))/dx = 1 / (1 + x^2)
    let atan_x = x.atan();
    let d_atan = atan_x.derivative("x");
    let simplified = d_atan.simplify(None);
    assert_eq!(simplified.to_string(), "1 / (1 + x^2)");
}

#[test]
fn test_derivative_exp_ln() {
    let x = Parameter::try_from("x").unwrap();

    // d(e^x)/dx = e^x
    let exp_x = x.exp();
    let d_exp = exp_x.derivative("x").simplify(None);
    assert_eq!(d_exp.to_string(), "exp(x)");

    // d(ln(x))/dx = 1/x
    let ln_x = x.ln();
    let d_ln = ln_x.derivative("x");
    assert_eq!(d_ln.simplify(None).to_string(), "1 / x");
}

#[test]
fn test_derivative_chain_rule() {
    let x = Parameter::try_from("x").unwrap();
    let two = Parameter::from(2.0);
    // sin(x^2)
    // d(sin(x^2))/dx = cos(x^2) * 2x
    let x_squared = x.pow(&two);
    let sin_x2 = x_squared.sin();

    let deriv = sin_x2.derivative("x").simplify(None);

    // Updated expectation based on actual output: cos(x^2) * 2 * x
    assert_eq!(deriv.to_string(), "cos(x^2) * 2 * x");
}

#[test]
fn test_derivative_abs_sqrt() {
    let x = Parameter::try_from("x").unwrap();

    // d(|x|)/dx = sign(x)
    let abs_x = x.abs();
    let d_abs = abs_x.derivative("x").simplify(None);
    assert_eq!(d_abs.to_string(), "sign(x)");

    // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
    let sqrt_x = x.sqrt();
    let d_sqrt = sqrt_x.derivative("x").simplify(None);
    assert_eq!(d_sqrt.to_string(), "1 / (2 * sqrt(x))");
}

#[test]
fn test_derivative_product_rule_complex() {
    // x * sin(x)
    // d/dx = 1*sin(x) + x*cos(x) = sin(x) + x*cos(x)
    let x = Parameter::try_from("x").unwrap();
    let expr = x.clone() * x.sin();
    let deriv = expr.derivative("x").simplify(None);

    // Expectation matches actual simplification result from similar logic or standard ordering
    assert_eq!(deriv.to_string(), "sin(x) + x * cos(x)");
}

#[test]
fn test_derivative_quotient_rule_complex() {
    // sin(x) / x
    // d/dx = (cos(x)*x - sin(x)) / x^2
    let x = Parameter::try_from("x").unwrap();
    let expr = x.sin() / x.clone();
    let deriv = expr.derivative("x").simplify(None);

    // Updated expectation based on actual output
    assert_eq!(deriv.to_string(), "(cos(x) * x - sin(x)) / (x^2)");
}

#[test]
fn test_log_arbitrary_base() {
    let x = Parameter::try_from("x").unwrap();
    let ten = Parameter::from(10.0);

    // log(x, 10)
    let log10_x = x.log(Some(ten));

    // d(log(x, 10))/dx = 1 / (x * ln(10))
    // ln(10) ~ 2.302585092994046
    let deriv = log10_x.derivative("x").simplify(None);

    assert_eq!(deriv.to_string(), "1 / (2.302585092994046 * x)");
}

#[test]
fn test_variable_base_log() {
    let x = Parameter::try_from("x").unwrap();

    // log(x, x)
    let log_x_x = x.log(Some(x.clone()));

    // d(log(x, x))/dx
    // Output: 1 / (x * ln(x)) - ln(x) / (x * ln(x)^2)
    // Note: simplify does not fully reduce this to 0.
    let deriv = log_x_x.derivative("x").simplify(None);

    assert_eq!(deriv.to_string(), "1 / (x * ln(x)) - ln(x) / (x * ln(x)^2)");
}
