// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::Qubit;
use thiserror::Error;

/// Errors raised while managing compiler-visible logical-qubit resources.
///
/// These errors distinguish unsupported requests and capacity limits from stale
/// planner state and internal consistency failures. Callers may use resource
/// availability errors to reject an algorithm candidate and try another one;
/// consistency errors indicate that the compiler state must not be trusted.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResourceError {
    /// A resource request violates its input contract.
    #[error("invalid resource request: {reason}")]
    InvalidRequest {
        /// Human-readable description of the violated request contract.
        reason: String,
    },
    /// The requested number of ancillary qubits cannot be made available under
    /// the active phase and policy.
    #[error(
        "insufficient {requirement} ancillary qubits: requested {requested}, available {available}"
    )]
    InsufficientResources {
        /// Stable requirement name used in diagnostics.
        requirement: &'static str,
        /// Number of ancillary qubits requested by the caller.
        requested: usize,
        /// Number of ancillary qubits available under the active policy.
        available: usize,
    },
    /// A resource operation would exceed the configured total-qubit limit.
    #[error("resource capacity exceeded: requested total {requested_total}, limit {limit}")]
    CapacityExceeded {
        /// Maximum number of logical qubits allowed by the active limits.
        limit: usize,
        /// Total logical-qubit count required by the attempted operation.
        requested_total: usize,
    },
    /// A qubit identifier conflicts with an existing resource or circuit qubit.
    #[error("duplicate resource qubit {qubit}")]
    DuplicateQubit {
        /// Conflicting logical qubit.
        qubit: Qubit,
    },
    /// A plan was created against an older resource-manager snapshot.
    ///
    /// The caller must preview the candidate again before attempting to commit.
    #[error(
        "stale resource plan: plan revision {plan_revision}, current revision {current_revision}"
    )]
    StalePlan {
        /// Revision captured while previewing the plan.
        plan_revision: u64,
        /// Current manager revision.
        current_revision: u64,
    },
    /// A plan was created by a different resource manager.
    #[error("resource plan belongs to another resource manager")]
    ForeignPlan,
    /// A lease belongs to another manager, is unknown, has already been
    /// released, or does not match its active record.
    #[error("unknown or released resource lease {lease_id}")]
    UnknownLease {
        /// Lease identifier supplied by the caller.
        lease_id: u64,
    },
    /// An operation is not valid in the current resource phase.
    #[error(
        "resource operation '{operation}' requires phase {expected}, current phase is {actual}"
    )]
    InvalidPhase {
        /// Stable resource-manager operation name.
        operation: &'static str,
        /// Required phase name.
        expected: &'static str,
        /// Current phase name.
        actual: &'static str,
    },
    /// No fresh logical-qubit identifier can be represented by [`Qubit`].
    #[error("cannot allocate logical ancillary qubit: qubit identifier overflow")]
    QubitIdOverflow,
    /// An internal monotonic identifier can no longer be incremented.
    #[error("resource {counter} counter overflow")]
    CounterOverflow {
        /// Stable name of the exhausted counter.
        counter: &'static str,
    },
    /// The circuit qubit set and manager resource records no longer match.
    ///
    /// This reports a compiler-state consistency failure rather than an
    /// ordinary unavailable-resource condition.
    #[error("circuit and resource manager are inconsistent: {reason}")]
    CircuitResourceMismatch {
        /// Human-readable description of the detected mismatch.
        reason: String,
    },
    /// An idle boundary was reached before all leases were released.
    #[error("resource manager still has {count} active lease(s)")]
    ActiveLeaseRemaining {
        /// Number of active leases that must be released.
        count: usize,
    },
}
