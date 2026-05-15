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

//! Target-basis decomposer.

use crate::circuit::Instruction;
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::compiler::transform::descriptor::TransformDescriptor;
use crate::compiler::transform::rewrite::{KnowledgeRewriter, RewriteConfig};
use crate::compiler::transform::{TransformOutcome, Transformer};

use super::config::DecomposeConfig;

/// Compiler transformer that lowers standard gates toward the target device
/// native standard-gate basis or an explicitly configured standard-gate basis.
#[derive(Debug, Clone, Default)]
pub struct Decomposer {
    /// User-facing lowering policy for basis selection and rewrite traversal.
    config: DecomposeConfig,
}

impl Decomposer {
    /// Creates a decomposer with explicit configuration.
    pub fn new(config: DecomposeConfig) -> Self {
        Self { config }
    }

    /// Returns the active decomposition configuration.
    pub const fn config(&self) -> &DecomposeConfig {
        &self.config
    }
}

static DECOMPOSER_DESCRIPTOR: TransformDescriptor = TransformDescriptor::new(
    "decompose.basis",
    "Decomposes gates toward a target standard-gate basis",
)
.supports_control_flow(true)
.supports_symbolic_parameters(true)
.modifies_circuit();

impl Transformer for Decomposer {
    fn descriptor(&self) -> &'static TransformDescriptor {
        &DECOMPOSER_DESCRIPTOR
    }

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        // The lower-level rewriter also expects a positive round count; reject
        // invalid configuration here so errors are reported by the decomposer.
        if self.config.max_rounds() == 0 {
            return Err(CompilerError::InvalidContextState(
                "decompose max_rounds must be greater than zero".to_string(),
            ));
        }

        // Prefer an explicit target basis. If none is configured, derive the
        // standard-gate subset from the active device native instruction list.
        let target_gates = if let Some(gates) = self.config.target_gates() {
            gates.to_vec()
        } else {
            let device = ctx.device().ok_or(CompilerError::MissingDevice)?;
            let mut gates = Vec::new();

            // Devices may list duplicate native gates or non-standard
            // instructions; the target-basis rewriter only accepts distinct
            // standard gates.
            for instruction in device.native_gates() {
                if let Instruction::Standard(gate) = instruction {
                    if !gates.contains(gate) {
                        gates.push(*gate);
                    }
                }
            }

            gates
        };

        if target_gates.is_empty() {
            return Err(CompilerError::InvalidContextState(
                "decompose target standard gate set is empty".to_string(),
            ));
        }

        // Decomposition is implemented by the knowledge-base rewriter in
        // lowering mode. This adapter owns device/default handling; the
        // rewriter owns rule matching, fixpoint iteration, and circuit rebuild.
        let rewrite_config = RewriteConfig::lowering()
            .with_target_gates(target_gates.clone())
            .with_max_rounds(self.config.max_rounds())
            .recurse_control_flow(self.config.recurses_control_flow())
            .skip_labeled_ops(self.config.skips_labeled_ops());

        let rewriter = KnowledgeRewriter::new(rewrite_config);
        let mut outcome = rewriter.transform(ctx)?;
        // Keep the outcome note at this layer so callers can distinguish
        // target-basis decomposition from a direct rewrite pass invocation.
        if outcome.changed {
            outcome.notes.push(format!(
                "decompose: lowered toward target standard gates {:?}",
                target_gates
            ));
        }

        Ok(outcome)
    }
}
