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

use crate::circuit::Parameter;
use ndarray::prelude::*;
use num::complex::Complex;
use std::fmt;

#[repr(u32)]
#[derive(Eq, Hash, PartialEq, Debug, Default, Clone)]
pub enum GateType {
    #[default]
    H,
    I,
    RX,
    RXX,
    RXY,
    RY,
    RYY,
    RZ,
    RZX,
    RZZ,
    S,
    SDG,
    SWAP,
    ISWAP,
    T,
    TDG,
    U, // U3
    X,
    XY,
    X2P,
    X2M,
    XY2P,
    XY2M,
    Y,
    Y2P,
    Y2M,
    Z,
    Phase,
    GPhase,
    CX,
    CCX,
    CY,
    CZ,
    CRX,
    CRY,
    CRZ,
    FSIM,

    UNITARY,
    BARRIER,
    MEASURE,
}

impl fmt::Display for GateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Gate {
    pub gate_type: GateType,
    pub control_num: i32,
    pub target_num: i32,
    pub parameter_num: i32,
    pub extra_control_num: i32,
    pub parameters: Option<Vec<Parameter>>,
    pub matrix: Option<Array2<Complex<f64>>>,
}
