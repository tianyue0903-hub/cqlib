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

/// Severity level for a compile diagnostic reported in the final artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Informational diagnostic that does not indicate degraded output quality.
    Info,
    /// Warning diagnostic indicating a limitation, fallback, or degraded outcome.
    Warning,
}

/// Stable structured diagnostic attached to a compile result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileDiagnostic {
    /// Severity level of the diagnostic.
    pub severity: DiagnosticSeverity,
    /// Stable machine-readable diagnostic code.
    pub code: &'static str,
    /// Human-readable diagnostic message.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::{CompileDiagnostic, DiagnosticSeverity};

    #[test]
    fn diagnostic_preserves_code_and_severity() {
        let diagnostic = CompileDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: "compiler.target.partially_lowered",
            message: "result is target-bound but not guaranteed execution-ready".to_string(),
        };

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert_eq!(diagnostic.code, "compiler.target.partially_lowered");
        assert!(diagnostic.message.contains("execution-ready"));
    }
}
