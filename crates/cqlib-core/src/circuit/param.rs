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

use crate::circuit::Parameter;
use crate::circuit::parameter::expr_node::ExprNode;
use std::f64::consts::{E, PI};

#[derive(Debug, Clone)]
pub enum CircuitParam {
    /// 在线路参数列表中的索引
    Index(u32),
    /// 固定的值
    Fixed(f64),
}

impl From<f64> for CircuitParam {
    fn from(v: f64) -> Self {
        Self::Fixed(v)
    }
}

#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// 在线路参数列表中的索引
    Param(Parameter),
    /// 固定的值
    Fixed(f64),
}

impl From<f64> for ParameterValue {
    fn from(v: f64) -> Self {
        Self::Fixed(v)
    }
}

impl From<i64> for ParameterValue {
    fn from(v: i64) -> Self {
        Self::Fixed(v as f64)
    }
}

impl From<Parameter> for ParameterValue {
    fn from(para: Parameter) -> Self {
        match para.node.as_ref() {
            ExprNode::Integer(i64) => ParameterValue::Fixed(*i64 as f64),
            ExprNode::Float(f64) => ParameterValue::Fixed(*f64),
            ExprNode::Pi => ParameterValue::Fixed(PI),
            ExprNode::E => ParameterValue::Fixed(E),
            _ => {
                if let Ok(v) = para.evaluate(&None) {
                    Self::Fixed(v)
                } else {
                    Self::Param(para)
                }
            }
        }
    }
}
