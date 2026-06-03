// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

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
    /// capacity.
    ///
    /// The current workflow does not perform layout or routing, so device
    /// topology is not validated as a postcondition.
    pub device: Option<Device>,
    /// Ancillary-resource permission for pre-layout decomposition passes.
    ///
    /// This controls whether logical clean ancillas may be allocated or dirty
    /// input qubits may be borrowed. Hard target capacity is derived from
    /// [`Self::device`] rather than this policy.
    pub resource_policy: ResourcePolicy,
    /// Reserved deterministic seed for future heuristic layout/routing passes.
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
pub fn compile(circuit: &Circuit, config: CompileConfig) -> Result<CompileResult, CompilerError> {
    CompilerWorkflow::new(config).run(circuit)
}
