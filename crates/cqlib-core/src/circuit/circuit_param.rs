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

//! # Circuit Parameters Module
//!
//! This module defines the types used to represent parameters within a circuit and its operations.
//! It bridges the gap between high-level symbolic expressions ([`Parameter`]) and low-level storage ([`CircuitParam`]).

use super::parameter::Parameter;

/// Represents a resolved parameter stored efficiently within a [`Circuit`](crate::circuit::Circuit).
///
/// This enum is designed for memory efficiency in the circuit's operation list.
/// Instead of storing full symbolic expressions in every operation, we either store a raw `f64`
/// (for fixed values) or an index into the circuit's centralized parameter table.
#[derive(Debug, Clone)]
pub enum CircuitParam {
    /// An index pointing to the `parameters` list in the parent [`Circuit`](crate::circuit::Circuit).
    /// This represents a symbolic or shared parameter.
    Index(u32),
    /// A concrete, fixed floating-point value.
    Fixed(f64),
}

impl From<f64> for CircuitParam {
    fn from(v: f64) -> Self {
        Self::Fixed(v)
    }
}

/// A flexible input type for specifying parameters when constructing circuit operations.
///
/// This enum allows users to pass either raw numbers or symbolic `Parameter` objects
/// to gate builders (like `rx`, `ry`, `u`).
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::circuit_param::ParameterValue;
/// use cqlib_core::circuit::Parameter;
///
/// // Create from float
/// let val1: ParameterValue = 1.5.into();
///
/// // Create from integer
/// let val2: ParameterValue = 42.into();
///
/// // Create from symbolic parameter
/// let param = Parameter::symbol("theta");
/// let val3: ParameterValue = param.into();
/// ```
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// A symbolic parameter (which may contain variables like "theta").
    Param(Parameter),
    /// A fixed floating-point value.
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
    /// Converts a `Parameter` into a `ParameterValue`.
    ///
    /// It attempts to eagerly evaluate the parameter if it represents a known constant
    /// (like `Pi`, `E`, or a simple number) to optimize storage.
    fn from(para: Parameter) -> Self {
        if let Ok(p) = para.evaluate(&None) {
            ParameterValue::Fixed(p)
        } else if let Ok(v) = para.evaluate(&None) {
            Self::Fixed(v)
        } else {
            Self::Param(para)
        }
    }
}

impl From<ParameterValue> for Parameter {
    fn from(para: ParameterValue) -> Parameter {
        match para {
            ParameterValue::Param(p) => p,
            ParameterValue::Fixed(f) => Parameter::from(f),
        }
    }
}
