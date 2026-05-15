use super::{CompileOptions, CompilePreset, build_diagnostics, compile};
use crate::circuit::{Circuit, ConditionView, Operation, Parameter, Qubit, StandardGate};
use crate::compiler::analysis::{InstructionStats, LogicalCost};
use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
use crate::compiler::transform::transformer::TransformStatsChange;
use crate::compiler::{CompileStatus, CompilerError, WorkflowReport, WorkflowStepReport};
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
    let artifact = compile(Circuit::new(2), CompilePreset::LogicalOptimize, None, None).unwrap();

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
    assert_eq!(trace.executed_steps, 1);
}

#[test]
fn compile_uses_default_options_when_none_is_provided() {
    let artifact = compile(Circuit::new(1), CompilePreset::LogicalOptimize, None, None).unwrap();

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

#[test]
fn compile_rejects_symbolic_parameters_when_option_disables_them() {
    let mut circuit = Circuit::new(1);
    circuit
        .rx(Qubit::new(0), Parameter::symbol("theta"))
        .unwrap();

    let err = compile(
        circuit,
        CompilePreset::LogicalOptimize,
        None,
        Some(CompileOptions::new().allow_symbolic_parameters(false)),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::UnsupportedInstruction { instruction }
        if instruction == "symbolic parameters"
    ));
}

#[test]
fn build_diagnostics_includes_workflow_diagnostics() {
    let report = WorkflowReport {
        name: "logical.optimize".to_string(),
        changed: true,
        executed_steps: 1,
        steps: vec![WorkflowStepReport {
            name: "test.step".to_string(),
            transform_name: "test.transform".to_string(),
            changed: true,
            notes: vec![],
            diagnostics: vec![CompileDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: "test.workflow.warning",
                message: "workflow warning".to_string(),
            }],
            stats_change: Some(TransformStatsChange::from_parts(
                InstructionStats {
                    total_ops: 1,
                    ..InstructionStats::default()
                },
                InstructionStats::default(),
                LogicalCost {
                    total_ops: 1,
                    ..LogicalCost::default()
                },
                LogicalCost::default(),
            )),
            iteration: None,
            branch: None,
        }],
        notes: vec![],
        diagnostics: vec![CompileDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: "test.workflow.warning",
            message: "workflow warning".to_string(),
        }],
        stats_change: Some(TransformStatsChange::from_parts(
            InstructionStats {
                total_ops: 1,
                ..InstructionStats::default()
            },
            InstructionStats::default(),
            LogicalCost {
                total_ops: 1,
                ..LogicalCost::default()
            },
            LogicalCost::default(),
        )),
    };

    let diagnostics = build_diagnostics(CompilePreset::LogicalOptimize, &report);

    assert_eq!(diagnostics, report.diagnostics);
}
