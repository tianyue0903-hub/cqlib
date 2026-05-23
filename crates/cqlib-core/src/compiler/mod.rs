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

pub mod analysis;
pub mod api;
pub mod artifact;
pub mod commutation;
pub mod context;
pub mod error;
pub mod knowledge;
pub mod transform;
pub mod verify;
pub mod workflow;

pub use analysis::AnalysisStore;
pub use api::{CompileOptions, CompilePreset, compile};
pub use artifact::{
    ArtifactMetadata, CompileArtifact, CompileDiagnostic, CompileStatus, CompileSummary,
    CompileTrace, DiagnosticSeverity,
};
pub use context::{CompilerContext, ContextChangeSet, ContextMetadata, VerificationConfig};
pub use error::CompilerError;
pub use transform::canonicalize::{CanonicalRuleId, CanonicalizeConfig, Canonicalizer};
pub use transform::resynthesis::{
    ResynthesisBudget, ResynthesisObjective, ResynthesisProfile, ResynthesisScope, Resynthesizer,
};
pub use workflow::{
    CompilerWorkflow, WorkflowReport, WorkflowStep, WorkflowStepReport, build_workflow,
};
