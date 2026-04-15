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

use crate::compiler::artifact::CompileDiagnostic;
use crate::compiler::transform::transformer::TransformStatsChange;
use crate::compiler::workflow::WorkflowReport;

/// Minimal user-facing compile trace derived from workflow execution.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompileTrace {
    pub workflow_name: String,
    pub executed_steps: usize,
    pub notes: Vec<String>,
    pub diagnostics: Vec<CompileDiagnostic>,
    pub stats_change: Option<TransformStatsChange>,
}

impl CompileTrace {
    pub fn from_report(report: &WorkflowReport) -> Self {
        Self {
            workflow_name: report.name.clone(),
            executed_steps: report.executed_steps,
            notes: report.notes.clone(),
            diagnostics: report.diagnostics.clone(),
            stats_change: report.stats_change.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CompileTrace;
    use crate::compiler::analysis::{InstructionStats, LogicalCost};
    use crate::compiler::artifact::{CompileDiagnostic, DiagnosticSeverity};
    use crate::compiler::transform::transformer::TransformStatsChange;
    use crate::compiler::workflow::WorkflowReport;

    #[test]
    fn compile_trace_is_built_from_workflow_report() {
        let report = WorkflowReport {
            name: "logical.optimize".to_string(),
            changed: true,
            executed_steps: 2,
            steps: vec![],
            notes: vec!["canonicalized".to_string()],
            diagnostics: vec![CompileDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: "test.trace.note",
                message: "traceable".to_string(),
            }],
            stats_change: Some(TransformStatsChange::from_parts(
                InstructionStats {
                    total_ops: 1,
                    ..InstructionStats::default()
                },
                InstructionStats::default(),
                LogicalCost::default(),
                LogicalCost {
                    depth_estimate: 1,
                    ..LogicalCost::default()
                },
            )),
        };

        let trace = CompileTrace::from_report(&report);

        assert_eq!(trace.workflow_name, "logical.optimize");
        assert_eq!(trace.executed_steps, 2);
        assert_eq!(trace.notes, vec!["canonicalized"]);
        assert_eq!(trace.diagnostics, report.diagnostics);
        assert_eq!(trace.stats_change, report.stats_change);
    }
}
