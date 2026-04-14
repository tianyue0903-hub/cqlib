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
use crate::compiler::artifact::{
    ArtifactMetadata, CompileArtifact, CompileDiagnostic, CompileStatus, CompileSummary,
    CompileTrace, DiagnosticSeverity,
};
use crate::compiler::context::{CompilerContext, ContextMetadata};
use crate::compiler::error::CompilerError;
use crate::compiler::workflow::{WorkflowReport, build_workflow};
use crate::device::Device;

/// Stable user-facing compilation path presets.
///
/// Presets choose a standard compiler path without exposing pass-level policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilePreset {
    /// Run device-agnostic logical cleanup and normalization only.
    LogicalOptimize,
    /// Lower the input circuit toward a specific target device or native basis.
    TargetLowering,
    /// Produce a target-bound circuit intended to be ready for execution.
    ExecutionReady,
}

/// High-level compile-time options shared across workflows.
///
/// These options express external compilation policy and reporting preferences.
/// They intentionally avoid exposing low-level pass tuning knobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    /// Emit an aggregate workflow report in the final artifact.
    emit_report: bool,
    /// Emit a pass-level trace in the final artifact.
    emit_trace: bool,
    /// Permit control-flow operations in the input circuit.
    allow_control_flow: bool,
    /// Permit unresolved symbolic parameters in the input circuit.
    allow_symbolic_parameters: bool,
    /// Allow workflows to include optional resynthesis stages.
    enable_resynthesis: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            emit_report: true,
            emit_trace: false,
            allow_control_flow: true,
            allow_symbolic_parameters: true,
            enable_resynthesis: false,
        }
    }
}

impl CompileOptions {
    /// Creates compile options with the default policy:
    /// reports enabled, traces disabled, control flow allowed,
    /// symbolic parameters allowed, and resynthesis disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Controls whether the final artifact includes an aggregate workflow report.
    pub fn with_report(mut self, enabled: bool) -> Self {
        self.emit_report = enabled;
        self
    }

    /// Controls whether the final artifact includes a pass-level execution trace.
    pub fn with_trace(mut self, enabled: bool) -> Self {
        self.emit_trace = enabled;
        self
    }

    /// Controls whether input circuits containing control flow are accepted.
    pub fn allow_control_flow(mut self, allowed: bool) -> Self {
        self.allow_control_flow = allowed;
        self
    }

    /// Controls whether unresolved symbolic parameters are accepted.
    pub fn allow_symbolic_parameters(mut self, allowed: bool) -> Self {
        self.allow_symbolic_parameters = allowed;
        self
    }

    /// Controls whether optional resynthesis stages may be selected by workflows.
    pub fn enable_resynthesis(mut self, enabled: bool) -> Self {
        self.enable_resynthesis = enabled;
        self
    }

    /// Returns whether the final artifact should include a workflow report.
    pub fn emit_report(&self) -> bool {
        self.emit_report
    }

    /// Returns whether the final artifact should include a pass-level trace.
    pub fn emit_trace(&self) -> bool {
        self.emit_trace
    }

    /// Returns whether control-flow operations are allowed in the input circuit.
    pub fn allows_control_flow(&self) -> bool {
        self.allow_control_flow
    }

    /// Returns whether unresolved symbolic parameters are allowed in the input circuit.
    pub fn allows_symbolic_parameters(&self) -> bool {
        self.allow_symbolic_parameters
    }

    /// Returns whether workflows may include optional resynthesis stages.
    pub fn resynthesis_enabled(&self) -> bool {
        self.enable_resynthesis
    }
}

/// Compiles a circuit using the selected preset, optional target device, and
/// optional high-level compile options.
///
/// `device` may be `None` only for [`CompilePreset::LogicalOptimize`].
/// `options = None` is equivalent to [`CompileOptions::default()`].
pub fn compile(
    circuit: Circuit,
    preset: CompilePreset,
    device: Option<Device>,
    options: Option<CompileOptions>,
) -> Result<CompileArtifact, CompilerError> {
    let options = options.unwrap_or_default();
    let input_ops = circuit.operations().len();

    validate_request(&circuit, preset, device.as_ref(), &options)?;

    let mut ctx = match device.as_ref() {
        Some(device) => CompilerContext::with_device(circuit, device.clone()),
        None => CompilerContext::new(circuit),
    };
    let workflow = build_workflow(preset, &options);
    let report = workflow.run(&mut ctx)?;
    let diagnostics = build_diagnostics(preset, &report);
    let summary = build_summary(preset, input_ops, &ctx, &report);
    let status = derive_status(preset, &diagnostics);
    let metadata = build_artifact_metadata(ctx.metadata());

    let trace = options
        .emit_trace()
        .then(|| CompileTrace::from_report(&report));
    let report = options.emit_report().then_some(report);

    Ok(CompileArtifact {
        circuit: ctx.circuit().clone(),
        layout: ctx.layout().cloned(),
        status,
        summary,
        diagnostics,
        metadata,
        report,
        trace,
    })
}

fn build_summary(
    preset: CompilePreset,
    input_ops: usize,
    ctx: &CompilerContext,
    report: &WorkflowReport,
) -> CompileSummary {
    CompileSummary {
        preset,
        workflow_name: report.name.clone(),
        target_name: ctx.metadata().target_name.clone(),
        input_ops,
        output_ops: ctx.circuit().operations().len(),
        changed: report.changed,
        executed_steps: report.executed_steps,
        has_layout: ctx.layout().is_some(),
        is_target_bound: ctx.device().is_some(),
    }
}

fn build_artifact_metadata(metadata: &ContextMetadata) -> ArtifactMetadata {
    ArtifactMetadata {
        workflow_name: metadata.workflow_name.clone(),
        target_name: metadata.target_name.clone(),
        tags: metadata.tags.clone(),
        options_digest: metadata.options_digest.clone(),
    }
}

fn build_diagnostics(preset: CompilePreset, report: &WorkflowReport) -> Vec<CompileDiagnostic> {
    let mut diagnostics = Vec::new();

    if matches!(preset, CompilePreset::TargetLowering) {
        diagnostics.push(CompileDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: "compiler.target.partially_lowered",
            message: "result is target-bound but does not claim execution readiness".to_string(),
        });
    }

    if !report.changed {
        diagnostics.push(CompileDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: "compiler.workflow.no_changes",
            message: "workflow completed without changing the input circuit".to_string(),
        });
    }

    diagnostics
}

fn derive_status(preset: CompilePreset, diagnostics: &[CompileDiagnostic]) -> CompileStatus {
    let has_warnings = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Warning);

    if has_warnings {
        return CompileStatus::SucceededWithWarnings;
    }

    match preset {
        CompilePreset::LogicalOptimize => CompileStatus::Succeeded,
        CompilePreset::TargetLowering => CompileStatus::PartiallyLowered,
        CompilePreset::ExecutionReady => CompileStatus::ExecutionReady,
    }
}

fn validate_request(
    circuit: &Circuit,
    preset: CompilePreset,
    device: Option<&Device>,
    options: &CompileOptions,
) -> Result<(), CompilerError> {
    if matches!(
        preset,
        CompilePreset::TargetLowering | CompilePreset::ExecutionReady
    ) && device.is_none()
    {
        return Err(CompilerError::MissingDevice);
    }

    if !options.allows_control_flow()
        && circuit.operations().iter().any(|op| {
            matches!(
                op.instruction,
                crate::circuit::Instruction::ControlFlowGate(_)
            )
        })
    {
        return Err(CompilerError::UnsupportedControlFlow);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CompileOptions, CompilePreset, compile};
    use crate::circuit::{Circuit, ConditionView, Operation, Qubit, StandardGate};
    use crate::compiler::{CompileStatus, CompilerError};
    use crate::device::{Device, Topology};
    use smallvec::smallvec;
    use std::collections::HashSet;

    fn mock_device(name: &str, qubit_count: usize) -> Device {
        let qubits: Vec<_> = (0..qubit_count).map(|i| Qubit::new(i as u32)).collect();
        let topology = Topology::new(qubits.clone(), vec![]).unwrap();
        Device::new(name, HashSet::from_iter(qubits), topology).unwrap()
    }

    #[test]
    fn compile_options_default_matches_api_contract() {
        let options = CompileOptions::new();

        assert_eq!(options, CompileOptions::default());
        assert!(options.emit_report());
        assert!(!options.emit_trace());
        assert!(options.allows_control_flow());
        assert!(options.allows_symbolic_parameters());
        assert!(!options.resynthesis_enabled());
    }

    #[test]
    fn compile_options_builder_methods_only_change_requested_flags() {
        let options = CompileOptions::new()
            .with_report(false)
            .with_trace(true)
            .allow_control_flow(false)
            .allow_symbolic_parameters(false)
            .enable_resynthesis(true);

        assert!(!options.emit_report());
        assert!(options.emit_trace());
        assert!(!options.allows_control_flow());
        assert!(!options.allows_symbolic_parameters());
        assert!(options.resynthesis_enabled());
    }

    #[test]
    fn compile_returns_complete_artifact_for_empty_logical_workflow() {
        let artifact =
            compile(Circuit::new(2), CompilePreset::LogicalOptimize, None, None).unwrap();

        assert_eq!(artifact.circuit.operations().len(), 0);
        assert!(artifact.layout.is_none());
        assert_eq!(artifact.status, CompileStatus::Succeeded);
        assert_eq!(artifact.summary.preset, CompilePreset::LogicalOptimize);
        assert_eq!(artifact.summary.input_ops, 0);
        assert_eq!(artifact.summary.output_ops, 0);
        assert!(!artifact.summary.changed);
        assert_eq!(artifact.diagnostics.len(), 1);
        assert_eq!(artifact.diagnostics[0].code, "compiler.workflow.no_changes");
        assert_eq!(
            artifact.report.as_ref().map(|report| report.name.as_str()),
            Some("logical.optimize")
        );
        assert!(artifact.trace.is_none());
        assert_eq!(
            artifact.metadata.workflow_name.as_deref(),
            Some("logical.optimize")
        );
    }

    #[test]
    fn compile_with_device_populates_target_metadata() {
        let artifact = compile(
            Circuit::new(1),
            CompilePreset::TargetLowering,
            Some(mock_device("mock-qpu", 1)),
            None,
        )
        .unwrap();

        assert_eq!(artifact.status, CompileStatus::PartiallyLowered);
        assert!(artifact.summary.is_target_bound);
        assert_eq!(artifact.metadata.target_name.as_deref(), Some("mock-qpu"));
        assert_eq!(
            artifact.diagnostics[0].code,
            "compiler.target.partially_lowered"
        );
        assert_eq!(
            artifact.report.as_ref().map(|report| report.name.as_str()),
            Some("target.lowering")
        );
    }

    #[test]
    fn compile_rejects_target_presets_without_device() {
        let err = compile(Circuit::new(1), CompilePreset::TargetLowering, None, None).unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));

        let err = compile(Circuit::new(1), CompilePreset::ExecutionReady, None, None).unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));
    }

    #[test]
    fn compile_omits_report_when_disabled() {
        let artifact = compile(
            Circuit::new(1),
            CompilePreset::LogicalOptimize,
            None,
            Some(CompileOptions::new().with_report(false)),
        )
        .unwrap();
        assert!(artifact.report.is_none());
        assert_eq!(artifact.summary.workflow_name, "logical.optimize");
        assert_eq!(artifact.status, CompileStatus::Succeeded);
    }

    #[test]
    fn compile_emits_trace_when_enabled() {
        let artifact = compile(
            Circuit::new(1),
            CompilePreset::LogicalOptimize,
            None,
            Some(CompileOptions::new().with_trace(true)),
        )
        .unwrap();
        let trace = artifact.trace.expect("trace should be emitted");

        assert_eq!(trace.workflow_name, "logical.optimize");
        assert_eq!(trace.executed_steps, 0);
    }

    #[test]
    fn compile_uses_default_options_when_none_is_provided() {
        let artifact =
            compile(Circuit::new(1), CompilePreset::LogicalOptimize, None, None).unwrap();

        assert!(artifact.report.is_some());
        assert!(artifact.trace.is_none());
        assert_eq!(artifact.summary.input_ops, 0);
    }

    #[test]
    fn execution_ready_preset_sets_execution_ready_status() {
        let artifact = compile(
            Circuit::new(1),
            CompilePreset::ExecutionReady,
            Some(mock_device("mock-qpu", 1)),
            None,
        )
        .unwrap();

        assert_eq!(artifact.status, CompileStatus::ExecutionReady);
        assert!(artifact.summary.is_target_bound);
    }

    #[test]
    fn compile_rejects_control_flow_when_option_disables_it() {
        let mut circuit = Circuit::new(1);
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: StandardGate::X.into(),
                    qubits: smallvec![Qubit::new(0)],
                    params: smallvec![],
                    label: None,
                }],
                None,
            )
            .unwrap();

        let err = compile(
            circuit,
            CompilePreset::LogicalOptimize,
            None,
            Some(CompileOptions::new().allow_control_flow(false)),
        )
        .unwrap_err();

        assert!(matches!(err, CompilerError::UnsupportedControlFlow));
    }
}
