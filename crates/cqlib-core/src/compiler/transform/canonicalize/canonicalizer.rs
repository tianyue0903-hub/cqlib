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

//! Canonicalizer transformer entry point.
//!
//! The canonicalizer is built in layers:
//! - Task 2 normalized symbolic parameters and global phase
//! - Task 3 adds recursive linear-structure canonicalization
//!
//! This separation matters because:
//! - it lets us ship a production-ready canonicalization interface without
//!   mixing in unrelated structural cleanup logic
//! - it reuses current `Circuit` functionality instead of duplicating symbolic
//!   parameter handling in the compiler layer
//! - it keeps later tasks free to add instruction-form and structural rules
//!   without changing the public `Canonicalizer` shape

use crate::circuit::Circuit;
use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
use crate::compiler::context::{CompilerContext, ContextChangeSet};
use crate::compiler::error::CompilerError;
use crate::compiler::transform::{TransformDescriptor, TransformOutcome, Transformer};

use super::config::CanonicalizeConfig;
use super::linear::canonicalize_linear_structure;
use super::parameter_phase::{canonicalize_parameter_phase, parameter_phase_changed};

/// Stable rule identifiers for built-in canonicalization behaviors.
///
/// These identifiers are exposed so that downstream diagnostics, logging, or
/// future configuration surfaces can refer to rules by name. They are
/// **intentionally not** wired into a pluggable rule engine yet — the current
/// canonicalizer still uses simple hard-coded checks in `ops.rs` and
/// `linear.rs` because the rule set is small and the extra abstraction would
/// hurt readability more than it helps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanonicalRuleId {
    NormalizeParameters,
    CanonicalizeInstructionForm,
    MergeAdjacentBarriers,
    DropTrivialNoOps,
}

/// Production canonicalizer entry point.
pub struct Canonicalizer {
    config: CanonicalizeConfig,
}

impl Canonicalizer {
    /// Creates a canonicalizer with the supplied configuration.
    pub const fn new(config: CanonicalizeConfig) -> Self {
        Self { config }
    }

    /// Creates a canonicalizer using production defaults.
    pub const fn production() -> Self {
        Self::new(CanonicalizeConfig::production())
    }

    /// Returns the active canonicalization configuration.
    pub const fn config(&self) -> &CanonicalizeConfig {
        &self.config
    }
}

static CANONICALIZER_DESCRIPTOR: TransformDescriptor = TransformDescriptor::new(
    "canonicalize.standard",
    "Canonicalizes circuit structure into a stable internal form",
)
.supports_control_flow(true)
.supports_symbolic_parameters(true)
.modifies_circuit();

impl Transformer for Canonicalizer {
    fn descriptor(&self) -> &'static TransformDescriptor {
        &CANONICALIZER_DESCRIPTOR
    }

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        let loop_result = run_to_fixpoint_with(ctx.circuit(), &self.config, run_single_round)?;

        if !loop_result.any_parameter_phase_changed && !loop_result.any_structural_changed {
            return Ok(TransformOutcome::unchanged());
        }

        *ctx.circuit_mut() = loop_result.circuit;

        let mut outcome = TransformOutcome::changed().with_changes(
            ContextChangeSet::circuit_changed()
                .with_cfg_structure_changed(loop_result.any_structural_changed)
                .with_parameter_table_changed(loop_result.any_parameter_phase_changed),
        );
        if loop_result.any_parameter_phase_changed {
            outcome =
                outcome.with_note("canonicalize: normalized symbolic parameters and global phase");
        }
        if loop_result.any_structural_changed {
            outcome = outcome.with_note(
                "canonicalize: canonicalized instruction forms, barriers, and trivial no-ops",
            );
        }
        if !loop_result.stabilized {
            outcome = outcome
                .with_note(format!(
                    "canonicalize: reached round limit after {} rounds before proving stability",
                    loop_result.rounds_executed
                ))
                .with_diagnostic(CompileDiagnostic {
                    severity: DiagnosticSeverity::Warning,
                    code: "compiler.canonicalize.round_limit_reached",
                    message: format!(
                        "canonicalization stopped after {} rounds before reaching a fixed point",
                        loop_result.rounds_executed
                    ),
                });
        }

        Ok(outcome)
    }
}

/// Result of one canonicalization round.
///
/// A single round performs parameter-phase canonicalization first, then structural
/// canonicalization. The two flags are tracked independently so the outer fixpoint
/// loop can decide whether another round is needed.
#[derive(Debug, Clone)]
pub(crate) struct SingleRoundResult {
    /// Circuit produced by this round.
    pub(crate) circuit: Circuit,
    /// Whether parameter normalization or global-phase simplification changed anything.
    pub(crate) parameter_phase_changed: bool,
    /// Whether structural rules (instruction-form collapse, barrier merge, no-op drop) changed anything.
    pub(crate) structural_changed: bool,
}

/// Result of the fixpoint loop.
///
/// The canonicalizer runs rounds iteratively because parameter normalization can
/// turn a symbolic angle into `0.0`, which then enables structural cleanup to drop
/// the now-trivial gate in the next round. The loop stops when both phases report
/// no change or when the configured round limit is reached.
#[derive(Debug, Clone)]
pub(crate) struct FixpointResult {
    /// Final circuit after all executed rounds.
    pub(crate) circuit: Circuit,
    /// Whether the parameter phase changed in *any* round.
    pub(crate) any_parameter_phase_changed: bool,
    /// Whether the structure changed in *any* round.
    pub(crate) any_structural_changed: bool,
    /// Number of rounds actually executed.
    pub(crate) rounds_executed: u8,
    /// `true` if the circuit reached a fixed point before the round limit.
    pub(crate) stabilized: bool,
}

/// Runs a single canonicalization round.
///
/// Order matters: parameter-phase canonicalization runs first so that a
/// simplified parameter (e.g. `0.0`) can be picked up by structural rules in
/// the same round. Structural canonicalization is skipped entirely when all
/// structural config flags are disabled.
fn run_single_round(
    circuit: &Circuit,
    config: &CanonicalizeConfig,
) -> Result<SingleRoundResult, CompilerError> {
    let mut canonical = if config.normalizes_parameters() {
        canonicalize_parameter_phase(circuit)?
    } else {
        circuit.clone()
    };
    let parameter_phase_changed = parameter_phase_changed(circuit, &canonical);

    let structural_changed = if config.canonicalizes_instruction_form()
        || config.merges_adjacent_barriers()
        || config.drops_trivial_noops()
    {
        let structural = canonicalize_linear_structure(&canonical, config)?;
        if let Some(rebuilt) = structural.circuit {
            canonical = rebuilt;
        }
        structural.changed
    } else {
        false
    };

    Ok(SingleRoundResult {
        circuit: canonical,
        parameter_phase_changed,
        structural_changed,
    })
}

/// Runs canonicalization rounds until the circuit reaches a fixed point or the round limit.
///
/// Each round can enable further simplifications in the next round (e.g.
/// parameter normalization → structural no-op drop). The loop terminates
/// early when both phases report no change, indicating stability. If the
/// round limit is reached first, a warning diagnostic is produced but the
/// partially canonicalized circuit is still returned.
pub(crate) fn run_to_fixpoint_with<F>(
    initial: &Circuit,
    config: &CanonicalizeConfig,
    mut step: F,
) -> Result<FixpointResult, CompilerError>
where
    F: FnMut(&Circuit, &CanonicalizeConfig) -> Result<SingleRoundResult, CompilerError>,
{
    if config.round_limit() == 0 {
        return Err(CompilerError::InvalidContextState(
            "canonicalize round_limit must be greater than zero".to_string(),
        ));
    }

    let mut current = initial.clone();
    let mut any_parameter_phase_changed = false;
    let mut any_structural_changed = false;
    let mut rounds_executed = 0;

    for round in 1..=config.round_limit() {
        let round_result = step(&current, config)?;
        rounds_executed = round;
        any_parameter_phase_changed |= round_result.parameter_phase_changed;
        any_structural_changed |= round_result.structural_changed;

        if !round_result.parameter_phase_changed && !round_result.structural_changed {
            return Ok(FixpointResult {
                circuit: current,
                any_parameter_phase_changed,
                any_structural_changed,
                rounds_executed,
                stabilized: true,
            });
        }

        current = round_result.circuit;
    }

    Ok(FixpointResult {
        circuit: current,
        any_parameter_phase_changed,
        any_structural_changed,
        rounds_executed,
        stabilized: false,
    })
}
