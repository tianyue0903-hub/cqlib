use crate::circuit::gate::StandardGate;
use crate::circuit::parameter::Parameter;
use crate::compile::gate_transform::transform_rules::decomposed_gate::DecomposedGate;
use crate::compile::gate_transform::transform_rules::double_qubit_rule::DoubleQubitRule;
use crate::compile::gate_transform::transform_rules::param_transform_rule::ParamTransformRule;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoQubitTransformRule {
    Cx2Rzz,
    Rzz2Cx,
    Cx2Cy,
    Cy2Cx,
    Cx2Cz,
    Cz2Cx,
    Rzz2Rxx,
    Rxx2Rzz,
    Rzz2Ryy,
    Ryy2Rzz,
    Rzz2Rzx,
    Rzx2Rzz,
    Rzz2Crz,
    Crz2Rzz,
    Rzz2Crx,
    Crx2Rzz,
    Rzz2Cry,
    Cry2Rzz,
    Cx2Fsim,
    Fsim2Cx,
    Fsim2Rzz,
    Rzz2Fsim,
}

impl TwoQubitTransformRule {
    pub fn from_gates(source: &StandardGate, target: &StandardGate) -> Option<Self> {
        match (*source, *target) {
            (StandardGate::CX, StandardGate::RZZ) => Some(Self::Cx2Rzz),
            (StandardGate::RZZ, StandardGate::CX) => Some(Self::Rzz2Cx),
            (StandardGate::CX, StandardGate::CY) => Some(Self::Cx2Cy),
            (StandardGate::CY, StandardGate::CX) => Some(Self::Cy2Cx),
            (StandardGate::CX, StandardGate::CZ) => Some(Self::Cx2Cz),
            (StandardGate::CZ, StandardGate::CX) => Some(Self::Cz2Cx),
            (StandardGate::RZZ, StandardGate::RXX) => Some(Self::Rzz2Rxx),
            (StandardGate::RXX, StandardGate::RZZ) => Some(Self::Rxx2Rzz),
            (StandardGate::RZZ, StandardGate::RYY) => Some(Self::Rzz2Ryy),
            (StandardGate::RYY, StandardGate::RZZ) => Some(Self::Ryy2Rzz),
            (StandardGate::RZZ, StandardGate::RZX) => Some(Self::Rzz2Rzx),
            (StandardGate::RZX, StandardGate::RZZ) => Some(Self::Rzx2Rzz),
            (StandardGate::RZZ, StandardGate::CRZ) => Some(Self::Rzz2Crz),
            (StandardGate::CRZ, StandardGate::RZZ) => Some(Self::Crz2Rzz),
            (StandardGate::RZZ, StandardGate::CRX) => Some(Self::Rzz2Crx),
            (StandardGate::CRX, StandardGate::RZZ) => Some(Self::Crx2Rzz),
            (StandardGate::RZZ, StandardGate::CRY) => Some(Self::Rzz2Cry),
            (StandardGate::CRY, StandardGate::RZZ) => Some(Self::Cry2Rzz),
            (StandardGate::CX, StandardGate::FSIM) => Some(Self::Cx2Fsim),
            (StandardGate::FSIM, StandardGate::CX) => Some(Self::Fsim2Cx),
            (StandardGate::FSIM, StandardGate::RZZ) => Some(Self::Fsim2Rzz),
            (StandardGate::RZZ, StandardGate::FSIM) => Some(Self::Rzz2Fsim),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingleQubitParamTransformRule {
    Rx2U,
    U2Rx,
    U2Rxy,
    Rxy2U,
    Rx2Rxy,
    Rxy2Rx,
    Rx2Ry,
    Ry2Rx,
    Rx2Rz,
    Rz2Rx,
    Xy2Rxy,
    Rxy2Xy,
    Xy2p2Rxy,
    Rxy2Xy2p,
    Xy2m2Rxy,
    Rxy2Xy2m,
}

impl SingleQubitParamTransformRule {
    pub fn from_gates(source: &StandardGate, target: &StandardGate) -> Option<Self> {
        match (*source, *target) {
            (StandardGate::RX, StandardGate::U) => Some(Self::Rx2U),
            (StandardGate::U, StandardGate::RX) => Some(Self::U2Rx),
            (StandardGate::U, StandardGate::RXY) => Some(Self::U2Rxy),
            (StandardGate::RXY, StandardGate::U) => Some(Self::Rxy2U),
            (StandardGate::RX, StandardGate::RXY) => Some(Self::Rx2Rxy),
            (StandardGate::RXY, StandardGate::RX) => Some(Self::Rxy2Rx),
            (StandardGate::RX, StandardGate::RY) => Some(Self::Rx2Ry),
            (StandardGate::RY, StandardGate::RX) => Some(Self::Ry2Rx),
            (StandardGate::RX, StandardGate::RZ) => Some(Self::Rx2Rz),
            (StandardGate::RZ, StandardGate::RX) => Some(Self::Rz2Rx),
            (StandardGate::XY, StandardGate::RXY) => Some(Self::Xy2Rxy),
            (StandardGate::RXY, StandardGate::XY) => Some(Self::Rxy2Xy),
            (StandardGate::XY2P, StandardGate::RXY) => Some(Self::Xy2p2Rxy),
            (StandardGate::RXY, StandardGate::XY2P) => Some(Self::Rxy2Xy2p),
            (StandardGate::XY2M, StandardGate::RXY) => Some(Self::Xy2m2Rxy),
            (StandardGate::RXY, StandardGate::XY2M) => Some(Self::Rxy2Xy2m),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformRuleKind {
    TwoQubit(TwoQubitTransformRule),
    SingleQubitParam(SingleQubitParamTransformRule),
}

pub struct TransformRuleExecutor;

impl TransformRuleExecutor {
    pub fn apply(
        rule: TransformRuleKind,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        match rule {
            TransformRuleKind::TwoQubit(rule) => Self::apply_two_qubit(rule, gate, params),
            TransformRuleKind::SingleQubitParam(rule) => {
                Self::apply_single_qubit_param(rule, gate, params)
            }
        }
    }

    fn apply_two_qubit(
        rule: TwoQubitTransformRule,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        match rule {
            TwoQubitTransformRule::Cx2Rzz => DoubleQubitRule::cx2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Cx => DoubleQubitRule::rzz2cx_rule(gate, params),
            TwoQubitTransformRule::Cx2Cy => DoubleQubitRule::cx2cy_rule(gate, params),
            TwoQubitTransformRule::Cy2Cx => DoubleQubitRule::cy2cx_rule(gate, params),
            TwoQubitTransformRule::Cx2Cz => DoubleQubitRule::cx2cz_rule(gate, params),
            TwoQubitTransformRule::Cz2Cx => DoubleQubitRule::cz2cx_rule(gate, params),
            TwoQubitTransformRule::Rzz2Rxx => DoubleQubitRule::rzz2rxx_rule(gate, params),
            TwoQubitTransformRule::Rxx2Rzz => DoubleQubitRule::rxx2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Ryy => DoubleQubitRule::rzz2ryy_rule(gate, params),
            TwoQubitTransformRule::Ryy2Rzz => DoubleQubitRule::ryy2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Rzx => DoubleQubitRule::rzz2rzx_rule(gate, params),
            TwoQubitTransformRule::Rzx2Rzz => DoubleQubitRule::rzx2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Crz => DoubleQubitRule::rzz2crz_rule(gate, params),
            TwoQubitTransformRule::Crz2Rzz => DoubleQubitRule::crz2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Crx => DoubleQubitRule::rzz2crx_rule(gate, params),
            TwoQubitTransformRule::Crx2Rzz => DoubleQubitRule::crx2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Cry => DoubleQubitRule::rzz2cry_rule(gate, params),
            TwoQubitTransformRule::Cry2Rzz => DoubleQubitRule::cry2rzz_rule(gate, params),
            TwoQubitTransformRule::Cx2Fsim => DoubleQubitRule::cx2fsim_rule(gate, params),
            TwoQubitTransformRule::Fsim2Cx => DoubleQubitRule::fsim2cx_rule(gate, params),
            TwoQubitTransformRule::Fsim2Rzz => DoubleQubitRule::fsim2rzz_rule(gate, params),
            TwoQubitTransformRule::Rzz2Fsim => DoubleQubitRule::rzz2fsim_rule(gate, params),
        }
    }

    fn apply_single_qubit_param(
        rule: SingleQubitParamTransformRule,
        gate: &StandardGate,
        params: &SmallVec<[Parameter; 3]>,
    ) -> DecomposedGate {
        match rule {
            SingleQubitParamTransformRule::Rx2U => ParamTransformRule::rx2u_rule(gate, params),
            SingleQubitParamTransformRule::U2Rx => ParamTransformRule::u2rx_rule(gate, params),
            SingleQubitParamTransformRule::U2Rxy => ParamTransformRule::u2rxy_rule(gate, params),
            SingleQubitParamTransformRule::Rxy2U => ParamTransformRule::rxy2u_rule(gate, params),
            SingleQubitParamTransformRule::Rx2Rxy => {
                ParamTransformRule::rx2rxy_rule(gate, params)
            }
            SingleQubitParamTransformRule::Rxy2Rx => {
                ParamTransformRule::rxy2rx_rule(gate, params)
            }
            SingleQubitParamTransformRule::Rx2Ry => ParamTransformRule::rx2ry_rule(gate, params),
            SingleQubitParamTransformRule::Ry2Rx => ParamTransformRule::ry2rx_rule(gate, params),
            SingleQubitParamTransformRule::Rx2Rz => ParamTransformRule::rx2rz_rule(gate, params),
            SingleQubitParamTransformRule::Rz2Rx => ParamTransformRule::rz2rx_rule(gate, params),
            SingleQubitParamTransformRule::Xy2Rxy => {
                ParamTransformRule::xy2rxy_rule(gate, params)
            }
            SingleQubitParamTransformRule::Rxy2Xy => {
                ParamTransformRule::rxy2xy_rule(gate, params)
            }
            SingleQubitParamTransformRule::Xy2p2Rxy => {
                ParamTransformRule::xy2p2rxy_rule(gate, params)
            }
            SingleQubitParamTransformRule::Rxy2Xy2p => {
                ParamTransformRule::rxy2xy2p_rule(gate, params)
            }
            SingleQubitParamTransformRule::Xy2m2Rxy => {
                ParamTransformRule::xy2m2rxy_rule(gate, params)
            }
            SingleQubitParamTransformRule::Rxy2Xy2m => {
                ParamTransformRule::rxy2xy2m_rule(gate, params)
            }
        }
    }
}
