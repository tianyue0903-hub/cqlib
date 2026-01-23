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

impl From<Parameter> for ParameterValue {
    fn from(para: Parameter) -> Self {
        Self::Param(para)
    }
}
