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

//! Compile-layer errors for mapping and routing passes.

use crate::circuit::{CircuitError, Qubit};
use thiserror::Error;

/// Error type for compile/mapping workflows.
#[derive(Debug, Error)]
pub enum CompileError {
    /// Wrapper around circuit-level errors.
    #[error("Circuit error: {0}")]
    Circuit(#[from] CircuitError),

    /// Failed to construct DAG representation for preprocessing.
    #[error("Failed to build circuit DAG: {0}")]
    DagBuildFailed(String),

    /// DAG entry block is missing.
    #[error("Circuit DAG entry block is missing")]
    MissingEntryBlock,

    /// Control-flow is not supported in this mapper.
    #[error("Control-flow operations are not supported by the mapper")]
    UnsupportedControlFlow,

    /// Unsupported instruction kind encountered.
    #[error("Unsupported instruction at op #{op_index}: {instruction}")]
    UnsupportedInstruction {
        op_index: usize,
        instruction: String,
    },

    /// Unsupported gate arity encountered.
    #[error("Unsupported gate arity {arity} at op #{op_index}; only 1q/2q are supported")]
    UnsupportedArity { op_index: usize, arity: usize },

    /// Topology does not provide enough usable qubits.
    #[error("Topology has insufficient qubits: required {required}, available {available}")]
    TopologyTooSmall { required: usize, available: usize },

    /// Fidelity value is invalid.
    #[error("Invalid fidelity for edge ({u}, {v}): {value}; expected in [0, 1]")]
    InvalidFidelity { u: Qubit, v: Qubit, value: f64 },

    /// Fidelity entry references an unknown topology qubit.
    #[error("Fidelity references a topology-unknown qubit edge ({u}, {v})")]
    UnknownFidelityQubit { u: Qubit, v: Qubit },

    /// Fidelity entry references a non-existent topology edge.
    #[error("Fidelity provided for non-existent topology edge ({u}, {v})")]
    FidelityEdgeNotFound { u: Qubit, v: Qubit },

    /// VF2 cannot find a suitable mapping.
    #[error("VF2 could not find a subgraph-isomorphic mapping")]
    Vf2NoMapping,

    /// SABRE routing got stuck and cannot continue.
    #[error("SABRE routing failed to progress")]
    SabreRoutingStuck,

    /// Internal invariants were broken.
    #[error("Internal compile error: {0}")]
    Internal(String),
}
