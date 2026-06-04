// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Parameter resolution, simplification, and target-table interning.

use crate::circuit::{Circuit, CircuitParam, Parameter, ParameterValue};
use crate::compiler::CompilerError;

pub(super) fn resolve_parameter(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<Parameter, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => {
            if !value.is_finite() {
                return Err(CompilerError::InvalidInput(format!(
                    "non-finite fixed parameter {value}"
                )));
            }
            Ok(Parameter::from(*value))
        }
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .ok_or_else(|| CompilerError::InvalidInput(format!("missing parameter index {index}"))),
    }
}

pub(super) fn canonical_parameter(param: Parameter) -> Result<Parameter, CompilerError> {
    let simplified = param.simplify().map_err(|error| {
        CompilerError::InvalidInput(format!("parameter simplify failed: {error}"))
    })?;

    if simplified.get_symbols().is_empty() {
        // Constants are stored as fixed values in operations and global phase so
        // that the rebuilt parameter table contains only genuinely symbolic
        // expressions that downstream binding logic needs to track.
        let value = simplified.evaluate(&None).map_err(|error| {
            CompilerError::InvalidInput(format!("constant parameter cannot be evaluated: {error}"))
        })?;
        if !value.is_finite() {
            return Err(CompilerError::InvalidInput(format!(
                "constant parameter evaluates to non-finite value {value}"
            )));
        }
        let value = if value == 0.0 { 0.0 } else { value };
        Ok(Parameter::from(value))
    } else {
        Ok(simplified)
    }
}

pub(super) fn parameter_to_circuit_param(
    circuit: &mut Circuit,
    param: Parameter,
) -> Result<CircuitParam, CompilerError> {
    let param = canonical_parameter(param)?;

    if param.get_symbols().is_empty() {
        // Re-evaluate after simplification instead of preserving the original
        // AST. This gives a stable representation for equivalent constants and
        // normalizes `-0.0` to `0.0`.
        let value = param.evaluate(&None).map_err(|error| {
            CompilerError::InvalidInput(format!("constant parameter cannot be evaluated: {error}"))
        })?;
        if !value.is_finite() {
            return Err(CompilerError::InvalidInput(format!(
                "constant parameter evaluates to non-finite value {value}"
            )));
        }
        let value = if value == 0.0 { 0.0 } else { value };
        Ok(CircuitParam::Fixed(value))
    } else {
        let (index, _) = circuit.add_parameter(param);
        Ok(CircuitParam::Index(index as u32))
    }
}

pub(super) fn circuit_param_to_value(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<ParameterValue, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(*value)),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .map(ParameterValue::Param)
            .ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "canonicalizer produced missing parameter index {index}"
                ))
            }),
    }
}

pub(super) fn parameter_is_exact_zero(param: &Parameter) -> Result<bool, CompilerError> {
    if param.get_symbols().is_empty() {
        let value = param.evaluate(&None).map_err(|error| {
            CompilerError::InvalidInput(format!("constant parameter cannot be evaluated: {error}"))
        })?;
        Ok(value == 0.0)
    } else {
        Ok(false)
    }
}
