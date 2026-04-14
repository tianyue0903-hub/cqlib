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

use crate::compiler::workflow::WorkflowReport;

/// Minimal user-facing compile trace derived from workflow execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompileTrace {
    pub workflow_name: String,
    pub executed_steps: usize,
    pub notes: Vec<String>,
}

impl CompileTrace {
    pub fn from_report(report: &WorkflowReport) -> Self {
        Self {
            workflow_name: report.name.clone(),
            executed_steps: report.executed_steps,
            notes: report.notes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CompileTrace;
    use crate::compiler::workflow::WorkflowReport;

    #[test]
    fn compile_trace_is_built_from_workflow_report() {
        let report = WorkflowReport {
            name: "logical.optimize".to_string(),
            changed: true,
            executed_steps: 2,
            steps: vec![],
            notes: vec!["canonicalized".to_string()],
        };

        let trace = CompileTrace::from_report(&report);

        assert_eq!(trace.workflow_name, "logical.optimize");
        assert_eq!(trace.executed_steps, 2);
        assert_eq!(trace.notes, vec!["canonicalized"]);
    }
}
