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

use crate::circuit::Circuit;
use crate::compiler::api::CompilePreset;
use crate::compiler::artifact::diagnostic::CompileDiagnostic;
use crate::compiler::artifact::trace::CompileTrace;
use crate::compiler::workflow::WorkflowReport;
use crate::device::Layout;

/// Stable success-state classification for a completed compile request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileStatus {
    /// Compilation succeeded without warnings and without target-execution claims.
    Succeeded,
    /// Compilation succeeded but emitted one or more warning diagnostics.
    SucceededWithWarnings,
    /// Compilation succeeded and lowered toward a target, but does not guarantee
    /// direct executability on the current backend.
    PartiallyLowered,
    /// Compilation succeeded and produced a result intended for execution.
    ExecutionReady,
}

/// Stable user-facing summary of one compile result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileSummary {
    /// High-level preset requested by the caller.
    pub preset: CompilePreset,
    /// Stable name of the workflow that produced this result.
    pub workflow_name: String,
    /// Name of the selected target device, when compilation was target-aware.
    pub target_name: Option<String>,
    /// Operation count of the input circuit before workflow execution.
    pub input_ops: usize,
    /// Operation count of the final output circuit.
    pub output_ops: usize,
    /// Whether any workflow step reported a semantic change.
    pub changed: bool,
    /// Number of workflow steps that executed.
    pub executed_steps: usize,
    /// Whether the final artifact includes a logical-to-physical layout.
    pub has_layout: bool,
    /// Whether the result is bound to a concrete target backend.
    pub is_target_bound: bool,
}

/// Stable result metadata exposed at the artifact boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArtifactMetadata {
    /// Workflow name recorded during execution, when available.
    pub workflow_name: Option<String>,
    /// Target device name, when compilation used a concrete backend.
    pub target_name: Option<String>,
    /// Workflow- or transform-level tags accumulated during execution.
    pub tags: Vec<String>,
    /// Stable digest of externally visible compile options, when computed.
    pub options_digest: Option<String>,
}

/// Top-level result package returned by a compiler invocation.
///
/// This groups the finalized circuit together with stable result summary,
/// diagnostics, optional execution reports, and output metadata.
#[derive(Debug, Clone)]
pub struct CompileArtifact {
    /// Final circuit produced by the compiler workflow.
    pub circuit: Circuit,
    /// Final logical-to-physical layout when the workflow established one.
    pub layout: Option<Layout>,
    /// Stable success-state classification for the completed compile request.
    pub status: CompileStatus,
    /// Stable user-facing summary of the compile result.
    pub summary: CompileSummary,
    /// Structured compile diagnostics emitted during or after workflow execution.
    pub diagnostics: Vec<CompileDiagnostic>,
    /// Stable result metadata exposed at the artifact boundary.
    pub metadata: ArtifactMetadata,
    /// Aggregate workflow report when report emission was enabled.
    pub report: Option<WorkflowReport>,
    /// Pass-level execution trace when trace emission was enabled.
    pub trace: Option<CompileTrace>,
}

#[cfg(test)]
mod tests {
    use super::{ArtifactMetadata, CompileStatus, CompileSummary};
    use crate::compiler::api::CompilePreset;

    #[test]
    fn compile_summary_tracks_target_facts() {
        let summary = CompileSummary {
            preset: CompilePreset::ExecutionReady,
            workflow_name: "execution.ready".to_string(),
            target_name: Some("mock-qpu".to_string()),
            input_ops: 3,
            output_ops: 4,
            changed: true,
            executed_steps: 2,
            has_layout: true,
            is_target_bound: true,
        };

        assert_eq!(summary.preset, CompilePreset::ExecutionReady);
        assert_eq!(summary.target_name.as_deref(), Some("mock-qpu"));
        assert!(summary.has_layout);
        assert!(summary.is_target_bound);
    }

    #[test]
    fn artifact_metadata_is_decoupled_from_context_metadata() {
        let metadata = ArtifactMetadata {
            workflow_name: Some("logical.optimize".to_string()),
            target_name: None,
            tags: vec!["normalized".to_string()],
            options_digest: Some("digest".to_string()),
        };

        assert_eq!(metadata.workflow_name.as_deref(), Some("logical.optimize"));
        assert_eq!(metadata.tags, vec!["normalized"]);
        assert_eq!(metadata.options_digest.as_deref(), Some("digest"));
    }

    #[test]
    fn compile_status_distinguishes_success_kinds() {
        assert_ne!(
            CompileStatus::Succeeded,
            CompileStatus::SucceededWithWarnings
        );
        assert_ne!(
            CompileStatus::PartiallyLowered,
            CompileStatus::ExecutionReady
        );
    }
}
