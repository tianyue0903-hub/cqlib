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

use crate::compiler::analysis::AnalysisKey;
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;

/// Static description of a transformer's requirements and state effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformDescriptor {
    pub name: &'static str,
    pub summary: &'static str,
    pub requires_device: bool,
    pub requires_layout: bool,
    pub required_analyses: &'static [AnalysisKey],
    pub supports_control_flow: bool,
    pub supports_symbolic_parameters: bool,
    pub modifies_circuit: bool,
    pub modifies_layout: bool,
    pub invalidates_layout: bool,
}

impl TransformDescriptor {
    /// Creates a descriptor with conservative defaults for a device-agnostic transformer.
    pub const fn new(name: &'static str, summary: &'static str) -> Self {
        Self {
            name,
            summary,
            requires_device: false,
            requires_layout: false,
            required_analyses: &[],
            supports_control_flow: true,
            supports_symbolic_parameters: true,
            modifies_circuit: false,
            modifies_layout: false,
            invalidates_layout: false,
        }
    }

    pub const fn requires_device(mut self) -> Self {
        self.requires_device = true;
        self
    }

    pub const fn requires_layout(mut self) -> Self {
        self.requires_layout = true;
        self
    }

    pub const fn with_required_analyses(mut self, analyses: &'static [AnalysisKey]) -> Self {
        self.required_analyses = analyses;
        self
    }

    pub const fn supports_control_flow(mut self, supports: bool) -> Self {
        self.supports_control_flow = supports;
        self
    }

    pub const fn supports_symbolic_parameters(mut self, supports: bool) -> Self {
        self.supports_symbolic_parameters = supports;
        self
    }

    pub const fn modifies_circuit(mut self) -> Self {
        self.modifies_circuit = true;
        self
    }

    pub const fn modifies_layout(mut self) -> Self {
        self.modifies_layout = true;
        self
    }

    pub const fn invalidates_layout(mut self) -> Self {
        self.invalidates_layout = true;
        self
    }

    /// Validates that the current compiler context satisfies this descriptor.
    pub fn validate(&self, ctx: &mut CompilerContext) -> Result<(), CompilerError> {
        if self.requires_layout && !self.requires_device {
            return Err(CompilerError::InvalidContextState(format!(
                "transformer {} requires layout without requiring device",
                self.name
            )));
        }

        if self.requires_device && ctx.device().is_none() {
            return Err(CompilerError::MissingDevice);
        }

        if self.requires_layout && ctx.layout().is_none() {
            return Err(CompilerError::MissingLayout);
        }

        if !self.supports_control_flow && ctx.has_control_flow() {
            return Err(CompilerError::UnsupportedControlFlow);
        }

        for &analysis in self.required_analyses {
            ctx.ensure_analysis(analysis)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::TransformDescriptor;
    use crate::circuit::{Circuit, ConditionView, Operation, Qubit, StandardGate};
    use crate::compiler::analysis::AnalysisKey;
    use crate::compiler::context::CompilerContext;
    use crate::compiler::error::CompilerError;
    use smallvec::smallvec;

    #[test]
    fn descriptor_defaults_are_conservative() {
        let descriptor = TransformDescriptor::new("rewrite.cancel", "Cancel adjacent inverses");

        assert_eq!(descriptor.name, "rewrite.cancel");
        assert!(!descriptor.requires_device);
        assert!(!descriptor.requires_layout);
        assert!(descriptor.supports_control_flow);
        assert!(descriptor.supports_symbolic_parameters);
        assert!(!descriptor.modifies_circuit);
        assert!(!descriptor.modifies_layout);
        assert!(!descriptor.invalidates_layout);
        assert!(descriptor.required_analyses.is_empty());
    }

    #[test]
    fn descriptor_builder_sets_requirements_and_effects() {
        let descriptor = TransformDescriptor::new("routing.basic", "Route unsupported couplings")
            .requires_device()
            .requires_layout()
            .with_required_analyses(&[AnalysisKey::CircuitCfg, AnalysisKey::QubitUsage])
            .supports_control_flow(false)
            .supports_symbolic_parameters(false)
            .modifies_circuit()
            .modifies_layout()
            .invalidates_layout();

        assert!(descriptor.requires_device);
        assert!(descriptor.requires_layout);
        assert_eq!(
            descriptor.required_analyses,
            &[AnalysisKey::CircuitCfg, AnalysisKey::QubitUsage]
        );
        assert!(!descriptor.supports_control_flow);
        assert!(!descriptor.supports_symbolic_parameters);
        assert!(descriptor.modifies_circuit);
        assert!(descriptor.modifies_layout);
        assert!(descriptor.invalidates_layout);
    }

    #[test]
    fn validate_requires_device_and_layout() {
        let descriptor = TransformDescriptor::new("routing.basic", "Route unsupported couplings")
            .requires_device()
            .requires_layout();
        let mut ctx = CompilerContext::new(Circuit::new(1));

        assert!(matches!(
            descriptor.validate(&mut ctx),
            Err(CompilerError::MissingDevice)
        ));
    }

    #[test]
    fn validate_rejects_control_flow_when_unsupported() {
        let descriptor = TransformDescriptor::new("rewrite.linear", "Linear-only rewrite")
            .supports_control_flow(false);
        let mut ctx = CompilerContext::new(Circuit::new(1));
        ctx.circuit_mut()
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

        assert!(matches!(
            descriptor.validate(&mut ctx),
            Err(CompilerError::UnsupportedControlFlow)
        ));
    }

    #[test]
    fn validate_materializes_supported_analyses() {
        let descriptor = TransformDescriptor::new("rewrite.cfg", "Uses cfg")
            .with_required_analyses(&[AnalysisKey::CircuitCfg]);
        let mut ctx = CompilerContext::new(Circuit::new(1));

        descriptor.validate(&mut ctx).unwrap();
        assert!(ctx.dag().is_ok());
    }

    #[test]
    fn validate_propagates_analysis_build_failure() {
        let descriptor = TransformDescriptor::new("native.check", "Uses native support")
            .with_required_analyses(&[AnalysisKey::NativeSupportAnalysis]);
        let mut ctx = CompilerContext::new(Circuit::new(1));

        assert!(matches!(
            descriptor.validate(&mut ctx),
            Err(CompilerError::MissingDevice)
        ));
    }
}
