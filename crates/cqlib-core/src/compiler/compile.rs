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

//! Public compiler entry point.
//!
//! [`compile`] runs the configured [`CompilerWorkflow`](super::workflow::CompilerWorkflow)
//! and returns the optimized circuit plus step-level diagnostics. The workflow
//! starts from a logical circuit, applies canonicalization, definition
//! expansion, knowledge-based rewrite, unitary and multi-controlled-gate
//! decomposition, optional device layout/routing, and optional target-basis
//! translation.
//!
//! Target constraints are resolved before any transform runs. An explicit
//! [`CompileConfig::target_basis`] takes precedence over native gates declared
//! by [`CompileConfig::device`]. When a device is present, the workflow also
//! uses its usable qubits and topology for capacity checks and SABRE routing;
//! when no device is present, compilation stays in the logical qubit space.
//!
//! [`CompileMode::Normal`] selects conservative production defaults.
//! [`CompileMode::Enhanced`] keeps the same semantic contract but spends more
//! rewrite and routing effort and runs additional cleanup around routing and
//! target-basis translation.
//!
//! The compiler does not currently perform final directed-coupling
//! legalization. Device routing guarantees undirected physical adjacency for
//! two-qubit operations; direction-specific native lowering remains a separate
//! compiler concern.

use super::workflow::CompilerWorkflow;
use crate::circuit::{Circuit, Instruction};
use crate::compiler::resource::ResourcePolicy;
use crate::compiler::{CompilerError, WorkflowStepReport};
use crate::device::Device;

/// Optimization effort selected for the compiler workflow.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompileMode {
    /// Conservative logical optimization using production pass defaults.
    #[default]
    Normal,
    /// A richer staged workflow with stronger pass budgets and target-aware
    /// cleanup when target constraints are available.
    Enhanced,
}

/// User-facing configuration for [`compile`].
///
/// The configuration describes logical optimization effort, optional target
/// constraints, and the ancillary-resource permissions available before layout.
#[derive(Debug, Clone)]
pub struct CompileConfig {
    /// Optimization workflow mode.
    pub mode: CompileMode,
    /// Explicit standard-gate target basis for final translation.
    ///
    /// When set, this takes precedence over native gates declared by
    /// [`Self::device`]. The current workflow accepts only
    /// [`Instruction::Standard`] entries because multi-controlled gates are
    /// decomposed before target-basis translation.
    pub target_basis: Option<Vec<Instruction>>,
    /// Optional target device used to derive native gates and logical-qubit
    /// capacity, and to run device topology layout/routing.
    ///
    /// When present, the workflow routes the circuit onto usable physical
    /// qubits before target-basis translation. Final directed-gate legalization
    /// remains a separate compiler concern.
    pub device: Option<Device>,
    /// Ancillary-resource permission for pre-layout decomposition passes.
    ///
    /// This controls whether logical clean ancillas may be allocated or dirty
    /// input qubits may be borrowed. Hard target capacity is derived from
    /// [`Self::device`] rather than this policy.
    pub resource_policy: ResourcePolicy,
    /// Optional deterministic seed for heuristic layout/routing passes.
    pub seed: Option<u32>,
}

/// Result returned by [`compile`].
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// Optimized circuit.
    pub circuit: Circuit,
    /// Whether any workflow step changed the input representation.
    pub changed: bool,
    /// Workflow mode used for this run.
    pub mode: CompileMode,
    /// Step-level execution report in run order.
    pub steps: Vec<WorkflowStepReport>,
}

/// Runs the configured compiler workflow over `circuit`.
///
/// The returned result records the optimized circuit and step-level reports in
/// execution order. Errors are reported when a configured target or transform
/// precondition cannot be satisfied.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::compiler::{CompileConfig, CompileMode, compile};
/// use cqlib_core::compiler::resource::ResourcePolicy;
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(Qubit::new(0)).unwrap();
/// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
///
/// let result = compile(
///     &circuit,
///     CompileConfig {
///         mode: CompileMode::Normal,
///         target_basis: None,
///         device: None,
///         resource_policy: ResourcePolicy::default(),
///         seed: Some(7),
///     },
/// )
/// .unwrap();
///
/// assert_eq!(result.mode, CompileMode::Normal);
/// assert!(!result.steps.is_empty());
/// assert_eq!(result.circuit.qubits().len(), 2);
/// ```
pub fn compile(circuit: &Circuit, config: CompileConfig) -> Result<CompileResult, CompilerError> {
    CompilerWorkflow::new(config).run(circuit)
}

#[cfg(test)]
#[path = "./compile_test.rs"]
mod compile_test;
