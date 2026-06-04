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

//! AST definitions for the rule DSL.
//!
//! This module defines the surface syntax tree ([`RuleDef`], [`GatePattern`])
//! produced by the parser. Lowering to runtime structures is handled by the
//! [`lower`](crate::compile::knowledge::rule_dsl::lower) module.

use crate::circuit::Parameter;
use crate::compile::knowledge::rule::Condition;

/// Surface AST for a single optimization rule.
#[derive(Debug, Clone)]
pub struct RuleDef {
    /// Name of the rule, e.g. `merge_rz`.
    pub name: String,
    /// Gate patterns that must be matched in order.
    pub match_ops: Vec<GatePattern>,
    /// Algebraic conditions on the matched variables.
    pub conditions: Vec<Condition>,
    /// Replacement gate patterns (may be empty for deletion rules).
    pub rewrite_ops: Vec<GatePattern>,
}

/// Gate identifier used by one surface pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateSpec {
    /// Standard gate identifier, e.g. `H`, `RX`, `CX`.
    Standard { gate_name: String },
    /// Multi-controlled gate identifier, e.g. `MCX[3]`.
    MultiControlled {
        /// Base standard gate to control.
        base_gate_name: String,
        /// Number of controls added by the `MC` wrapper.
        added_controls: u8,
    },
}

impl GateSpec {
    /// Returns the canonical DSL spelling of this gate spec.
    pub fn display_name(&self) -> String {
        match self {
            Self::Standard { gate_name } => gate_name.clone(),
            Self::MultiControlled {
                base_gate_name,
                added_controls,
            } => format!("MC{base_gate_name}[{added_controls}]"),
        }
    }
}

/// Surface AST for a single gate pattern.
///
/// A pattern looks like `RZ(a + b) 0` or `MCX[3] 0 1 2 3`:
/// - `gate`   = gate identifier
/// - `params` = `[Parameter("a + b")]`
/// - `qubits` = `[0]`
#[derive(Debug, Clone)]
pub struct GatePattern {
    /// Gate identifier.
    pub gate: GateSpec,
    /// Parameter expressions appearing inside the parentheses.
    pub params: Vec<Parameter>,
    /// Qubit indices appearing after the parentheses.
    pub qubits: Vec<u32>,
}
