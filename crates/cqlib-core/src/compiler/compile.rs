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

#[derive(Debug, Clone)]
pub struct CompileConfig {
    pub mode: CompileMode,
    pub target_basis: Option<Vec<Instruction>>,
    pub device: Option<Device>,
    pub seed: Option<u32>,
}

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

pub fn compile(circuit: &Circuit, config: CompileConfig) -> Result<CompileResult, CompilerError> {
    CompilerWorkflow::new(config).run(circuit)
}
