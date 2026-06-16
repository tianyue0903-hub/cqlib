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

//! Value-level instruction types for circuit construction.
//!
//! This module defines the **construction IR** — operations that have not yet been assigned to a
//! specific [`Circuit`](crate::circuit::Circuit). Where the storage IR ([`Instruction`],
//! [`Operation`]) uses interned [`CircuitParam::Index`] references into a circuit's parameter
//! table, the construction IR uses [`ParameterValue`] and recursive value-level control-flow
//! bodies.
//!
//! # Two IRs, One Conversion Point
//!
//! ```text
//! ValueInstruction tree ──from_operations()──▶ Instruction / Operation tree
//! ```
//!
//! [`Circuit::from_operations`](crate::circuit::Circuit::from_operations) is the sole bridge.
//! It recursively walks a [`ValueInstruction`] tree, interns every [`ParameterValue`] into the
//! target circuit, and produces a [`ClassicalControlOp`] with [`ControlBody`] bodies whose
//! [`CircuitParam::Index`] values are valid within that circuit.
//!
//! # Why Separate Types?
//!
//! The storage IR [`ControlBody`] wraps `Arc<Vec<Operation>>`. If a [`ValueOperation`] carried
//! [`Instruction::ClassicalControl`] directly, its nested body would contain
//! [`CircuitParam::Index`] values with no source parameter table — a silent correctness bug.
//!
//! By giving the construction IR its own control-flow types ([`ValueClassicalControlOp`] with
//! [`ValueControlBody`]), we ensure that every parameter in a construction-time operation tree
//! is a [`ParameterValue`]. The compiler enforces that indexed parameters never appear in
//! construction IR.

use crate::circuit::ClassicalControlOp;
use crate::circuit::circuit_param::{CircuitParam, ParameterValue};
use crate::circuit::classical::ClassicalVar;
use crate::circuit::classical_expr::ClassicalExpr;
use crate::circuit::control_flow::ControlBody;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::instruction::Instruction;
use crate::circuit::operation::{Operation, ValueOperation};
use alloc::collections::BTreeSet;

/// A value-level control-flow body.
///
/// This is the construction-IR counterpart of [`ControlBody`].
/// Instead of storing indexed [`Operation`] values, it stores
/// [`ValueOperation`] values whose parameters are [`ParameterValue`] objects that have not
/// yet been interned into any circuit.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::value_instruction::ValueControlBody;
/// use cqlib_core::circuit::{ValueOperation, Qubit, StandardGate, ParameterValue};
/// use smallvec::smallvec;
///
/// let body = ValueControlBody::new(vec![
///     ValueOperation::from_standard(
///         StandardGate::H,
///         [Qubit::new(0)],
///         [],
///     ),
///     ValueOperation::from_standard(
///         StandardGate::RX,
///         [Qubit::new(0)],
///         [ParameterValue::from("theta")],
///     ),
/// ]);
/// assert_eq!(body.operations().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ValueControlBody {
    operations: Vec<ValueOperation>,
}

impl ValueControlBody {
    /// Creates a body from a sequence of value-level operations.
    pub fn new(operations: Vec<ValueOperation>) -> Self {
        Self { operations }
    }

    /// Returns the body operations.
    pub fn operations(&self) -> &[ValueOperation] {
        self.operations.as_slice()
    }

    /// Returns every qubit used directly or by nested control-flow operations.
    pub fn used_qubits(&self) -> BTreeSet<crate::circuit::Qubit> {
        let mut qubits = BTreeSet::new();
        for operation in &self.operations {
            qubits.extend(operation.qubits.iter().copied());
            if let ValueInstruction::ClassicalControl(control) = &operation.instruction {
                qubits.extend(control.used_qubits());
            }
        }
        qubits
    }
}

impl From<Vec<ValueOperation>> for ValueControlBody {
    fn from(operations: Vec<ValueOperation>) -> Self {
        Self::new(operations)
    }
}

/// A single case in a value-level [`SwitchOp`](crate::circuit::SwitchOp)-like construct.
///
/// Each case matches an exact unsigned integer value and carries a [`ValueControlBody`]
/// to execute when the switch target equals that value.
#[derive(Debug, Clone)]
pub struct ValueSwitchCase {
    /// The exact unsigned integer value that triggers this case.
    pub value: u128,
    /// The body to execute when the switch target matches `value`.
    pub body: ValueControlBody,
}

impl ValueSwitchCase {
    /// Creates a switch case with the given match value and body.
    pub fn new(value: u128, body: ValueControlBody) -> Self {
        Self { value, body }
    }
}

/// A value-level classical control-flow operation.
///
/// This mirrors [`ClassicalControlOp`] but every
/// nested body is a [`ValueControlBody`] containing [`ValueOperation`] entries with
/// [`ParameterValue`] parameters. There are no [`CircuitParam::Index`] values anywhere
/// in this tree.
///
/// # Variants
///
/// | Variant | Storage-IR equivalent |
/// |---|---|
/// | `If` | [`IfOp`](crate::circuit::IfOp) |
/// | `While` | [`WhileOp`](crate::circuit::WhileOp) |
/// | `For` | [`ForOp`](crate::circuit::ForOp) |
/// | `Switch` | [`SwitchOp`](crate::circuit::SwitchOp) |
/// | `Break` | [`ClassicalControlOp::Break`](crate::circuit::ClassicalControlOp) |
/// | `Continue` | [`ClassicalControlOp::Continue`](crate::circuit::ClassicalControlOp) |
#[derive(Debug, Clone)]
pub enum ValueClassicalControlOp {
    /// Execute `then_body` when `condition` is true, optionally `else_body` when false.
    ///
    /// `condition` must have type [`ClassicalType::Bool`](crate::circuit::ClassicalType).
    If {
        /// Boolean branch condition.
        condition: ClassicalExpr,
        /// Body executed when the condition is true.
        then_body: ValueControlBody,
        /// Optional body executed when the condition is false.
        else_body: Option<ValueControlBody>,
    },
    /// Repeat `body` while `condition` remains true.
    ///
    /// `condition` must have type [`ClassicalType::Bool`](crate::circuit::ClassicalType).
    While {
        /// Boolean loop condition.
        condition: ClassicalExpr,
        /// Body executed while the condition is true.
        body: ValueControlBody,
    },
    /// Iterate `body` over an unsigned half-open range `[start, stop)` with the given `step`.
    ///
    /// `var` must have type [`ClassicalType::UInt`](crate::circuit::ClassicalType).
    /// `start`, `stop`, and `step` must match `var`'s width.
    For {
        /// Mutable unsigned loop variable.
        var: ClassicalVar,
        /// Inclusive initial value.
        start: ClassicalExpr,
        /// Exclusive upper bound.
        stop: ClassicalExpr,
        /// Non-zero iteration increment.
        step: ClassicalExpr,
        /// Loop body.
        body: ValueControlBody,
    },
    /// Select one body by matching `target` against exact case values.
    ///
    /// `target` must have an unsigned integer type. Cases do not fall through.
    Switch {
        /// Unsigned expression matched against case values.
        target: ClassicalExpr,
        /// Exact-value cases in source order.
        cases: Vec<ValueSwitchCase>,
        /// Optional default body.
        default: Option<ValueControlBody>,
    },
    /// Exit the nearest enclosing loop or switch body.
    Break,
    /// Advance to the next iteration of the nearest enclosing loop.
    Continue,
}

impl ValueClassicalControlOp {
    /// Returns every qubit used by the operation's nested control-flow bodies.
    pub fn used_qubits(&self) -> BTreeSet<crate::circuit::Qubit> {
        let mut qubits = BTreeSet::new();
        match self {
            Self::If {
                then_body,
                else_body,
                ..
            } => {
                qubits.extend(then_body.used_qubits());
                if let Some(else_body) = else_body {
                    qubits.extend(else_body.used_qubits());
                }
            }
            Self::While { body, .. } | Self::For { body, .. } => {
                qubits.extend(body.used_qubits());
            }
            Self::Switch { cases, default, .. } => {
                for case in cases {
                    qubits.extend(case.body.used_qubits());
                }
                if let Some(default) = default {
                    qubits.extend(default.used_qubits());
                }
            }
            Self::Break | Self::Continue => {}
        }
        qubits
    }

    /// Returns the classical variables read by the controlling expressions.
    pub fn classical_var_reads(&self) -> Vec<ClassicalVar> {
        match self {
            Self::If { condition, .. } | Self::While { condition, .. } => {
                condition.vars().into_iter().collect()
            }
            Self::For {
                start, stop, step, ..
            } => {
                let mut vars: Vec<_> = start.vars().into_iter().collect();
                vars.extend(stop.vars());
                vars.extend(step.vars());
                vars
            }
            Self::Switch { target, .. } => target.vars().into_iter().collect(),
            Self::Break | Self::Continue => Vec::new(),
        }
    }

    /// Returns the immutable classical values read by the controlling expressions.
    pub fn classical_value_reads(&self) -> Vec<crate::circuit::ClassicalValue> {
        match self {
            Self::If { condition, .. } | Self::While { condition, .. } => {
                condition.values().into_iter().collect()
            }
            Self::For {
                start, stop, step, ..
            } => {
                let mut vals: Vec<_> = start.values().into_iter().collect();
                vals.extend(stop.values());
                vals.extend(step.values());
                vals
            }
            Self::Switch { target, .. } => target.values().into_iter().collect(),
            Self::Break | Self::Continue => Vec::new(),
        }
    }

    /// Returns the classical variables written by this operation.
    ///
    /// Currently only [`For`](Self::For) writes its loop variable.
    pub fn classical_writes(&self) -> Vec<ClassicalVar> {
        match self {
            Self::For { var, .. } => vec![*var],
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ValueInstruction
// ---------------------------------------------------------------------------

/// A value-level instruction for circuit construction.
///
/// `ValueInstruction` is the construction-IR counterpart of [`Instruction`]. It has two
/// families of variants:
///
/// - **`Instruction(Instruction)`** — all non-control-flow operations (gates, directives,
///   classical data). The wrapped [`Instruction`] must not be
///   [`Instruction::ClassicalControl`]; use [`ValueInstruction::ClassicalControl`] for
///   control flow.
/// - **`ClassicalControl(ValueClassicalControlOp)`** — value-level classical control flow
///   whose bodies recursively contain [`ValueOperation`] entries with [`ParameterValue`]
///   parameters.
///
/// # Conversion to Storage IR
///
/// [`Circuit::from_operations`](crate::circuit::Circuit::from_operations) recursively
/// converts every `ValueInstruction` into an [`Instruction`]:
///
/// - `ValueInstruction::Instruction(inst)` → `inst` (pass-through, validated to not be
///   `ClassicalControl`).
/// - `ValueInstruction::ClassicalControl(vcc)` → `Instruction::ClassicalControl(cc)`,
///   where every nested [`ValueControlBody`] is recursively converted to a
///   [`ControlBody`] with interned parameters.
#[derive(Debug, Clone)]
pub enum ValueInstruction {
    /// A non-control-flow instruction. The wrapped [`Instruction`] must not be
    /// [`Instruction::ClassicalControl`].
    Instruction(Instruction),
    /// A value-level classical control-flow operation.
    ClassicalControl(ValueClassicalControlOp),
}

impl ValueInstruction {
    /// Creates a `ValueInstruction` from a non-control-flow [`Instruction`].
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `inst` is [`Instruction::ClassicalControl`]. Use
    /// [`ValueInstruction::ClassicalControl`] for control flow.
    pub fn from_instruction(inst: Instruction) -> Self {
        debug_assert!(
            !matches!(inst, Instruction::ClassicalControl(_)),
            "Instruction::ClassicalControl must use ValueInstruction::ClassicalControl variant"
        );
        Self::Instruction(inst)
    }

    /// Returns `true` if this is a classical control-flow instruction.
    pub fn is_classical_control(&self) -> bool {
        matches!(self, Self::ClassicalControl(_))
    }

    /// Returns `true` if this is a non-control-flow instruction.
    pub fn is_instruction(&self) -> bool {
        matches!(self, Self::Instruction(_))
    }

    /// Returns the wrapped storage instruction when this is a non-control-flow value instruction.
    pub fn as_instruction(&self) -> Option<&Instruction> {
        match self {
            Self::Instruction(inst) => Some(inst),
            Self::ClassicalControl(_) => None,
        }
    }

    /// Consumes this value instruction and returns the wrapped storage instruction
    /// when it is non-control-flow.
    pub fn into_instruction(self) -> Option<Instruction> {
        match self {
            Self::Instruction(inst) => Some(inst),
            Self::ClassicalControl(_) => None,
        }
    }
}

impl From<Instruction> for ValueInstruction {
    /// Converts a non-control-flow [`Instruction`] into a `ValueInstruction`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the instruction is [`Instruction::ClassicalControl`].
    fn from(inst: Instruction) -> Self {
        Self::from_instruction(inst)
    }
}

impl From<ValueClassicalControlOp> for ValueInstruction {
    fn from(op: ValueClassicalControlOp) -> Self {
        Self::ClassicalControl(op)
    }
}

pub(crate) fn storage_operation_to_value<F>(
    operation: Operation,
    resolve_parameter: &F,
) -> Result<ValueOperation, CircuitError>
where
    F: Fn(&CircuitParam) -> Result<ParameterValue, CircuitError>,
{
    fn convert_operation<F>(
        operation: Operation,
        resolve_parameter: &F,
    ) -> Result<ValueOperation, CircuitError>
    where
        F: Fn(&CircuitParam) -> Result<ParameterValue, CircuitError>,
    {
        let params = operation
            .params
            .iter()
            .map(resolve_parameter)
            .collect::<Result<_, _>>()?;
        Ok(ValueOperation {
            instruction: convert_instruction(operation.instruction, resolve_parameter)?,
            qubits: operation.qubits,
            params,
            label: operation.label,
        })
    }

    fn convert_body<F>(
        body: &ControlBody,
        resolve_parameter: &F,
    ) -> Result<ValueControlBody, CircuitError>
    where
        F: Fn(&CircuitParam) -> Result<ParameterValue, CircuitError>,
    {
        body.operations()
            .iter()
            .cloned()
            .map(|operation| convert_operation(operation, resolve_parameter))
            .collect::<Result<Vec<_>, _>>()
            .map(ValueControlBody::new)
    }

    fn convert_instruction<F>(
        instruction: Instruction,
        resolve_parameter: &F,
    ) -> Result<ValueInstruction, CircuitError>
    where
        F: Fn(&CircuitParam) -> Result<ParameterValue, CircuitError>,
    {
        let Instruction::ClassicalControl(op) = instruction else {
            return Ok(ValueInstruction::Instruction(instruction));
        };

        let op = match op {
            ClassicalControlOp::If(op) => ValueClassicalControlOp::If {
                condition: op.condition().clone(),
                then_body: convert_body(op.then_body(), resolve_parameter)?,
                else_body: op
                    .else_body()
                    .map(|body| convert_body(body, resolve_parameter))
                    .transpose()?,
            },
            ClassicalControlOp::While(op) => ValueClassicalControlOp::While {
                condition: op.condition().clone(),
                body: convert_body(op.body(), resolve_parameter)?,
            },
            ClassicalControlOp::For(op) => ValueClassicalControlOp::For {
                var: op.var(),
                start: op.start().clone(),
                stop: op.stop().clone(),
                step: op.step().clone(),
                body: convert_body(op.body(), resolve_parameter)?,
            },
            ClassicalControlOp::Switch(op) => ValueClassicalControlOp::Switch {
                target: op.target().clone(),
                cases: op
                    .cases()
                    .iter()
                    .map(|case| {
                        Ok(ValueSwitchCase::new(
                            case.value(),
                            convert_body(case.body(), resolve_parameter)?,
                        ))
                    })
                    .collect::<Result<_, CircuitError>>()?,
                default: op
                    .default()
                    .map(|body| convert_body(body, resolve_parameter))
                    .transpose()?,
            },
            ClassicalControlOp::Break => ValueClassicalControlOp::Break,
            ClassicalControlOp::Continue => ValueClassicalControlOp::Continue,
        };
        Ok(ValueInstruction::ClassicalControl(op))
    }

    convert_operation(operation, resolve_parameter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::ClassicalType;
    use crate::circuit::bit::Qubit;
    use crate::circuit::gate::StandardGate;
    use std::num::NonZeroU32;

    #[test]
    fn value_control_body_stores_value_operations() {
        let ops = vec![ValueOperation::from_standard(
            StandardGate::H,
            [Qubit::new(0)],
            [],
        )];
        let body = ValueControlBody::new(ops);
        assert_eq!(body.operations().len(), 1);
    }

    #[test]
    fn value_control_body_from_vec() {
        let ops = vec![
            ValueOperation::from_standard(StandardGate::X, [Qubit::new(0)], []),
            ValueOperation::from_standard(StandardGate::Y, [Qubit::new(1)], []),
        ];
        let body: ValueControlBody = ops.into();
        assert_eq!(body.operations().len(), 2);
    }

    #[test]
    fn value_instruction_from_gate_instruction() {
        let vi = ValueInstruction::from_instruction(Instruction::Standard(StandardGate::H));
        assert!(vi.is_instruction());
        assert!(!vi.is_classical_control());
    }

    #[test]
    fn value_instruction_from_classical_control() {
        let vcc = ValueClassicalControlOp::Break;
        let vi = ValueInstruction::from(vcc);
        assert!(vi.is_classical_control());
        assert!(!vi.is_instruction());
    }

    #[test]
    fn value_classical_control_reads_and_writes() {
        let var = ClassicalVar::new(
            crate::circuit::CircuitId::default(),
            0,
            ClassicalType::UInt(NonZeroU32::new(8).unwrap()),
        );
        let start = ClassicalExpr::var(var);
        let stop = ClassicalExpr::uint_literal(8, 10).unwrap();
        let step = ClassicalExpr::uint_literal(8, 1).unwrap();
        let body = ValueControlBody::new(vec![]);

        let vcc = ValueClassicalControlOp::For {
            var,
            start,
            stop,
            step,
            body,
        };

        assert_eq!(vcc.classical_var_reads().len(), 1);
        assert_eq!(vcc.classical_writes().len(), 1);
        assert_eq!(vcc.classical_writes()[0], var);
    }

    #[test]
    fn value_switch_case_construction() {
        let body = ValueControlBody::new(vec![]);
        let case = ValueSwitchCase::new(1, body);
        assert_eq!(case.value, 1);
        assert_eq!(case.body.operations().len(), 0);
    }
}
