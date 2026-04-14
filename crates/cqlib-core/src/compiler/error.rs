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

use crate::circuit::CircuitError;
use crate::circuit::Qubit;
use crate::device::{DeviceError, LayoutError, TopologyError};
use thiserror::Error;

/// Errors produced by compiler infrastructure components.
#[derive(Debug, Error)]
pub enum CompilerError {
    /// Wraps errors from core circuit representations and derived views.
    #[error(transparent)]
    Circuit(#[from] CircuitError),

    /// Wraps device validation and capability errors.
    #[error(transparent)]
    Device(#[from] DeviceError),

    /// Wraps layout construction and update errors.
    #[error(transparent)]
    Layout(#[from] LayoutError),

    /// Wraps topology graph errors when compiler flows manipulate connectivity.
    #[error(transparent)]
    Topology(#[from] TopologyError),

    /// A hardware-aware compiler flow was requested without a target device.
    #[error("compiler context is missing a target device")]
    MissingDevice,

    /// A transform requiring a concrete logical-to-physical mapping was requested
    /// before layout had been established.
    #[error("compiler context is missing a logical-to-physical layout")]
    MissingLayout,

    /// The context holds a combination of states that is inconsistent for compilation.
    #[error("invalid compiler context state: {0}")]
    InvalidContextState(String),

    /// The current workflow or transform does not support control flow in the input circuit.
    #[error("control-flow operations are not supported by this compiler path")]
    UnsupportedControlFlow,

    /// The current compiler path cannot handle the given instruction.
    #[error("unsupported instruction for this compiler path: {instruction}")]
    UnsupportedInstruction { instruction: String },

    /// The current compiler path cannot handle an instruction with this arity.
    #[error("unsupported instruction arity at operation {op_index}: {arity}")]
    UnsupportedArity { arity: usize, op_index: usize },

    /// The target device does not have enough physical qubits for the circuit.
    #[error("target device too small: logical qubits {logical}, physical qubits {physical}")]
    TargetTooSmall { logical: usize, physical: usize },

    /// A logical qubit could not be embedded onto the target device.
    #[error("logical qubit {qubit} cannot be mapped to the target device")]
    QubitNotMappable { qubit: Qubit },

    /// A required two-qubit interaction is unavailable on the target coupling graph.
    #[error("required coupling is unavailable on target device: {control} -> {target}")]
    CouplingUnavailable { control: Qubit, target: Qubit },

    /// The target device supports a coupling but not in the requested direction.
    #[error("direction not supported for instruction {instruction}: {control} -> {target}")]
    DirectionNotSupported {
        control: Qubit,
        target: Qubit,
        instruction: String,
    },

    /// The target device does not natively support the requested instruction on these qubits.
    #[error("native instruction unsupported on target device: {instruction} over {qubits:?}")]
    NativeInstructionUnsupported {
        instruction: String,
        qubits: Vec<Qubit>,
    },

    /// The current layout is incompatible with the selected target device.
    #[error("layout is invalid for target device: {0}")]
    InvalidLayoutForDevice(String),

    /// A pass was invoked before one of its required prerequisites was satisfied.
    #[error("prerequisite not met for {pass}: {requirement}")]
    PrerequisiteNotMet {
        pass: &'static str,
        requirement: &'static str,
    },

    /// An analysis stage failed with additional stage-specific context.
    #[error("analysis {name} failed: {reason}")]
    AnalysisFailed { name: &'static str, reason: String },

    /// A transform stage failed with additional stage-specific context.
    #[error("transform {name} failed: {reason}")]
    TransformFailed { name: &'static str, reason: String },

    /// A workflow failed with additional orchestration context.
    #[error("workflow {name} failed: {reason}")]
    WorkflowFailed { name: &'static str, reason: String },

    /// Internal compiler invariants were violated.
    #[error("compiler invariant violation: {0}")]
    InvariantViolation(String),

    /// Catch-all internal compiler error for impossible or partially implemented states.
    #[error("internal compiler error: {0}")]
    Internal(String),
}
