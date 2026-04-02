use crate::circuit::Parameter;
use crate::circuit::gate::StandardGate;
use crate::compile::gate_transform::transform_rules::decomposed_gate::DecomposedGate;
use smallvec::{SmallVec, smallvec};

use std::f64::consts::PI;

/// ParamTransformRule provides transformation rules for single-qubit parametrized gates.

pub struct ParamTransformRule {
    pub name: String,
}

impl ParamTransformRule {
    pub fn new(name: String) -> Self {
        Self { name }
    }

    // ========================================================================
    // Rules between categories (rx <-> u3/u, u3/u <-> rxy, rxy <-> rx)
    // ========================================================================

    /// rx(theta) = u(theta, -pi/2, pi/2)
    pub fn rx2u_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let mut result = DecomposedGate::new();
        result.push_single(
            StandardGate::U,
            smallvec![theta, Parameter::from(-PI / 2.0), Parameter::from(PI / 2.0)],
            0,
        );
        result
    }

    /// u(theta, phi, lambda) = exp(i*(phi+lambda)/2)
    //    H * RX(phi + pi/2)* H * RX(theta) * H * RX(lambda - pi/2) * H
    pub fn u2rx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let lambda = parameters[2].clone();
        let pi_half: Parameter = Parameter::pi() / 2.0;

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![lambda - pi_half.clone()], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![theta], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![phi + pi_half.clone()], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result
    }

    /// u(theta, phi, lambda) = exp(i*(phi+lambda)/2)
    ///    * H * RXY(phi + pi/2, 0) * H * RXY(theta, 0) * H * RXY(lambda - pi/2, 0) * H
    pub fn u2rxy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let lambda = parameters[2].clone();
        let pi_half: Parameter = Parameter::pi() / 2.0;

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(
            StandardGate::RXY,
            smallvec![lambda - pi_half.clone(), Parameter::from(0.0)],
            0,
        );
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RXY, smallvec![theta, Parameter::from(0.0)], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(
            StandardGate::RXY,
            smallvec![phi + pi_half, Parameter::from(0.0)],
            0,
        );
        result.push_single(StandardGate::H, smallvec![], 0);
        result
    }

    /// rxy(theta, phi) = u(theta, phi - pi/2, pi/2 - phi)
    pub fn rxy2u_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();
        let pi_half: Parameter = Parameter::pi() / 2.0;

        let mut result = DecomposedGate::new();
        result.push_single(
            StandardGate::U,
            smallvec![theta, phi.clone() - pi_half.clone(), pi_half - phi],
            0,
        );
        result
    }

    /// rx(theta) = rxy(theta, 0)
    pub fn rx2rxy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RXY, smallvec![theta, Parameter::from(0.0)], 0);
        result
    }

    /// rxy(theta, phi) = H * RX(phi) * H * RX(theta) * H * RX(-phi) * H
    pub fn rxy2rx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![Parameter::from(0.0) - phi.clone()], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![theta], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result.push_single(StandardGate::RX, smallvec![phi.clone()], 0);
        result.push_single(StandardGate::H, smallvec![], 0);
        result
    }

    // ========================================================================
    // Rules within rx categories (rx, ry, rz)
    // ========================================================================

    /// rx(theta) = rz(-pi/2) * ry(theta) * rz(pi/2)
    pub fn rx2ry_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RZ, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::RY, smallvec![theta], 0);
        result.push_single(StandardGate::RZ, smallvec![Parameter::from(-PI / 2.0)], 0);
        result
    }

    /// ry(theta) = rz(pi/2) * rx(theta) * rz(-pi/2)
    pub fn ry2rx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RZ, smallvec![Parameter::from(-PI / 2.0)], 0);
        result.push_single(StandardGate::RX, smallvec![theta], 0);
        result.push_single(StandardGate::RZ, smallvec![Parameter::from(PI / 2.0)], 0);
        result
    }

    /// rx(theta) = ry(pi/2) * rz(theta) * ry(-pi/2)
    pub fn rx2rz_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RY, smallvec![Parameter::from(-PI / 2.0)], 0);
        result.push_single(StandardGate::RZ, smallvec![theta], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result
    }

    /// rz(theta) = ry(-pi/2) * rx(theta) * ry(pi/2)
    pub fn rz2rx_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::RX, smallvec![theta], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(-PI / 2.0)], 0);
        result
    }

    // ========================================================================
    // Rules within rxy categories (xy, xy2p, xy2m)
    // ========================================================================

    /// xy(phi) = rxy(pi, phi)
    pub fn xy2rxy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let phi = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RXY, smallvec![Parameter::from(PI), phi], 0);
        result
    }

    /// rxy(theta, phi) = xy(phi/2) * ry(pi/2) * xy(theta/2) * ry(pi/2) * xy(-phi/2) * x
    pub fn rxy2xy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::X, smallvec![], 0);
        result.push_single(
            StandardGate::XY,
            smallvec![Parameter::from(0.0) - phi.clone() / 2.0],
            0,
        );
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY, smallvec![theta / 2.0], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY, smallvec![phi / 2.0], 0);
        result
    }

    /// xy2p(phi) = rxy(pi/2, phi)
    pub fn xy2p2rxy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let phi = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RXY, smallvec![Parameter::from(PI / 2.0), phi], 0);
        result
    }

    /// rxy(theta, phi) = xy2p(phi/2)^2 * ry(pi/2) * xy2p(theta/2)^2 * ry(pi/2) * xy2p(-phi/2)^2 * x
    pub fn rxy2xy2p_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::X, smallvec![], 0);
        result.push_single(StandardGate::XY2P, smallvec![Parameter::from(0.0) - phi.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2P, smallvec![Parameter::from(0.0) - phi.clone() / 2.0], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY2P, smallvec![theta.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2P, smallvec![theta / 2.0], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY2P, smallvec![phi.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2P, smallvec![phi / 2.0], 0);
        result
    }

    /// xy2m(phi) = rxy(-pi/2, phi)
    pub fn xy2m2rxy_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let phi = parameters[0].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::RXY, smallvec![Parameter::from(-PI / 2.0), phi], 0);
        result
    }

    /// rxy(theta, phi) = xy2m(phi/2)^2 * ry(pi/2) * xy2m(theta/2)^2 * ry(pi/2) * xy2m(-phi/2)^2 * x
    pub fn rxy2xy2m_rule(
        _gate: &StandardGate,
        parameters: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        let theta = parameters[0].clone();
        let phi = parameters[1].clone();

        let mut result = DecomposedGate::new();
        result.push_single(StandardGate::X, smallvec![], 0);
        result.push_single(StandardGate::XY2M, smallvec![Parameter::from(0.0) - phi.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2M, smallvec![Parameter::from(0.0) - phi.clone() / 2.0], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY2M, smallvec![theta.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2M, smallvec![theta / 2.0], 0);
        result.push_single(StandardGate::RY, smallvec![Parameter::from(PI / 2.0)], 0);
        result.push_single(StandardGate::XY2M, smallvec![phi.clone() / 2.0], 0);
        result.push_single(StandardGate::XY2M, smallvec![phi / 2.0], 0);
        result
    }

}

#[cfg(test)]
#[path = "param_transform_rules_test.rs"]
mod param_transform_rules_test;
