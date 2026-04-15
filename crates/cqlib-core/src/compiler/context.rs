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

use crate::circuit::{Circuit, CircuitCFG, CircuitParam, Instruction};
use crate::compiler::analysis::basis::ContextAnalysis;
use crate::compiler::analysis::{
    AnalysisKey, AnalysisStore, BasisAnalysis, BlockSummary, CostAnalysis, CouplingRequirements,
    InstructionStats, NativeSupportAnalysis, QubitUsage,
};
use crate::compiler::error::CompilerError;
use crate::compiler::verify::{
    CfgVerifier, CircuitVerifier, DefaultCfgVerifier, DefaultCircuitVerifier,
};
use crate::device::{Device, Layout};
use alloc::collections::BTreeSet;
use core::fmt::Debug;
use core::ops::Range;

/// Lightweight workflow metadata attached to the current compiler state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextMetadata {
    pub workflow_name: Option<String>,
    pub target_name: Option<String>,
    pub tags: Vec<String>,
    pub options_digest: Option<String>,
}

/// Verification hooks that may run around workflow and transform execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerificationConfig {
    pub verify_before_workflow: bool,
    pub verify_after_each_changed_transform: bool,
    pub verify_after_workflow: bool,
}

impl VerificationConfig {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn after_each_changed_transform(mut self, enabled: bool) -> Self {
        self.verify_after_each_changed_transform = enabled;
        self
    }

    pub fn before_workflow(mut self, enabled: bool) -> Self {
        self.verify_before_workflow = enabled;
        self
    }

    pub fn after_workflow(mut self, enabled: bool) -> Self {
        self.verify_after_workflow = enabled;
        self
    }

    pub fn any_enabled(&self) -> bool {
        self.verify_before_workflow
            || self.verify_after_each_changed_transform
            || self.verify_after_workflow
    }
}

/// Fine-grained description of what a transform changed in the compiler state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextChangeSet {
    pub circuit_changed: bool,
    pub device_changed: bool,
    pub layout_changed: bool,
    pub mapping_invalidated: bool,
    pub cfg_structure_changed: bool,
    pub parameter_table_changed: bool,
    pub affected_qubits: BTreeSet<crate::circuit::Qubit>,
    pub affected_op_range: Option<Range<usize>>,
}

impl ContextChangeSet {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn circuit_changed() -> Self {
        Self {
            circuit_changed: true,
            ..Self::default()
        }
    }

    pub fn device_changed() -> Self {
        Self {
            device_changed: true,
            ..Self::default()
        }
    }

    pub fn layout_changed() -> Self {
        Self {
            layout_changed: true,
            ..Self::default()
        }
    }

    pub fn mapping_invalidated() -> Self {
        Self {
            mapping_invalidated: true,
            ..Self::default()
        }
    }

    pub fn with_cfg_structure_changed(mut self, changed: bool) -> Self {
        self.cfg_structure_changed = changed;
        self
    }

    pub fn with_parameter_table_changed(mut self, changed: bool) -> Self {
        self.parameter_table_changed = changed;
        self
    }

    pub fn with_affected_qubit(mut self, qubit: crate::circuit::Qubit) -> Self {
        self.affected_qubits.insert(qubit);
        self
    }

    pub fn with_affected_op_range(mut self, range: Range<usize>) -> Self {
        self.affected_op_range = Some(range);
        self
    }

    pub fn extend(&mut self, other: Self) {
        self.circuit_changed |= other.circuit_changed;
        self.device_changed |= other.device_changed;
        self.layout_changed |= other.layout_changed;
        self.mapping_invalidated |= other.mapping_invalidated;
        self.cfg_structure_changed |= other.cfg_structure_changed;
        self.parameter_table_changed |= other.parameter_table_changed;
        self.affected_qubits.extend(other.affected_qubits);
        self.affected_op_range = match (self.affected_op_range.take(), other.affected_op_range) {
            (Some(lhs), Some(rhs)) => Some(lhs.start.min(rhs.start)..lhs.end.max(rhs.end)),
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        };
    }

    pub fn has_effects(&self) -> bool {
        self.circuit_changed
            || self.device_changed
            || self.layout_changed
            || self.mapping_invalidated
            || self.cfg_structure_changed
            || self.parameter_table_changed
            || !self.affected_qubits.is_empty()
            || self.affected_op_range.is_some()
    }
}

/// Shared compiler state managed across workflows, analyses, and transforms.
#[derive(Debug)]
pub struct CompilerContext {
    circuit: Circuit,
    device: Option<Device>,
    layout: Option<Layout>,
    revision: u64,
    metadata: ContextMetadata,
    analysis: AnalysisStore,
    verification: VerificationConfig,
}

impl ContextAnalysis for CircuitCFG {
    const KEY: AnalysisKey = AnalysisKey::CircuitCfg;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(CircuitCFG::from_circuit(&ctx.circuit)?)
    }
}

impl ContextAnalysis for InstructionStats {
    const KEY: AnalysisKey = AnalysisKey::InstructionStats;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(InstructionStats::from_circuit(&ctx.circuit))
    }
}

impl ContextAnalysis for QubitUsage {
    const KEY: AnalysisKey = AnalysisKey::QubitUsage;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(QubitUsage::from_circuit(&ctx.circuit))
    }
}

impl ContextAnalysis for BlockSummary {
    const KEY: AnalysisKey = AnalysisKey::BlockSummary;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(BlockSummary::from_cfg(ctx.cfg()?))
    }
}

impl ContextAnalysis for CouplingRequirements {
    const KEY: AnalysisKey = AnalysisKey::CouplingRequirements;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(CouplingRequirements::from_cfg(ctx.cfg()?))
    }
}

impl ContextAnalysis for BasisAnalysis {
    const KEY: AnalysisKey = AnalysisKey::BasisAnalysis;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        Ok(BasisAnalysis::from_circuit(&ctx.circuit))
    }
}

impl ContextAnalysis for NativeSupportAnalysis {
    const KEY: AnalysisKey = AnalysisKey::NativeSupportAnalysis;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        NativeSupportAnalysis::from_context(ctx)
    }
}

impl ContextAnalysis for CostAnalysis {
    const KEY: AnalysisKey = AnalysisKey::CostAnalysis;

    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError> {
        CostAnalysis::from_context(ctx)
    }
}

impl CompilerContext {
    /// Creates a compiler context for a circuit without a target device.
    pub fn new(circuit: Circuit) -> Self {
        Self {
            circuit,
            device: None,
            layout: None,
            revision: 0,
            metadata: ContextMetadata::default(),
            analysis: AnalysisStore::default(),
            verification: VerificationConfig::default(),
        }
    }

    /// Creates a compiler context for a circuit and target device.
    pub fn with_device(circuit: Circuit, device: Device) -> Self {
        let metadata = ContextMetadata {
            target_name: Some(device.name().to_string()),
            ..ContextMetadata::default()
        };

        Self {
            circuit,
            device: Some(device),
            layout: None,
            revision: 0,
            metadata,
            analysis: AnalysisStore::default(),
            verification: VerificationConfig::default(),
        }
    }

    /// Creates an isolated copy of the compiler state without analysis cache entries.
    pub fn fork(&self) -> Self {
        Self {
            circuit: self.circuit.clone(),
            device: self.device.clone(),
            layout: self.layout.clone(),
            revision: self.revision,
            metadata: self.metadata.clone(),
            analysis: AnalysisStore::default(),
            verification: self.verification.clone(),
        }
    }

    /// Replaces this context with another compiler state snapshot.
    pub fn replace_with(&mut self, other: Self) {
        *self = other;
    }

    /// Returns the current circuit.
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// Returns a mutable reference to the circuit.
    ///
    /// Call [`Self::mark_circuit_changed`] after any semantic or structural mutation.
    pub fn circuit_mut(&mut self) -> &mut Circuit {
        &mut self.circuit
    }

    /// Returns the current target device, if any.
    pub fn device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    /// Returns the current layout, if any.
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// Returns workflow metadata associated with the current state.
    pub fn metadata(&self) -> &ContextMetadata {
        &self.metadata
    }

    /// Returns mutable workflow metadata.
    pub fn metadata_mut(&mut self) -> &mut ContextMetadata {
        &mut self.metadata
    }

    /// Returns the active verification policy.
    pub fn verification_config(&self) -> &VerificationConfig {
        &self.verification
    }

    /// Replaces the active verification policy.
    pub fn set_verification_config(&mut self, config: VerificationConfig) {
        self.verification = config;
    }

    /// Returns the current compiler revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns whether the current circuit contains explicit control-flow operations.
    pub fn has_control_flow(&self) -> bool {
        self.circuit
            .operations()
            .iter()
            .any(|op| matches!(op.instruction, Instruction::ControlFlowGate(_)))
    }

    /// Returns whether the current circuit contains unresolved symbolic parameters.
    pub fn has_symbolic_parameters(&self) -> bool {
        self.circuit
            .operations()
            .iter()
            .flat_map(|op| op.params.iter())
            .any(|param| matches!(param, CircuitParam::Index(_)))
    }

    /// Replaces the target device and invalidates mapping-dependent state.
    pub fn set_device(&mut self, device: Device) {
        self.metadata.target_name = Some(device.name().to_string());
        self.device = Some(device);
        self.layout = None;
        self.apply_changes(ContextChangeSet::device_changed());
    }

    /// Removes the current target device and mapping state.
    pub fn clear_device(&mut self) {
        self.device = None;
        self.layout = None;
        self.metadata.target_name = None;
        self.apply_changes(ContextChangeSet::device_changed());
    }

    /// Sets the current logical-to-physical layout.
    pub fn set_layout(&mut self, layout: Layout) {
        self.layout = Some(layout);
        self.apply_changes(ContextChangeSet::layout_changed());
    }

    /// Clears the current layout.
    pub fn clear_layout(&mut self) {
        self.layout = None;
        self.apply_changes(ContextChangeSet::layout_changed());
    }

    /// Clears mapping state without touching other compiler state.
    pub fn mark_mapping_invalid(&mut self) {
        self.layout = None;
        self.apply_changes(ContextChangeSet::mapping_invalidated());
    }

    /// Marks the circuit as changed and invalidates all cached analyses.
    pub fn mark_circuit_changed(&mut self) {
        self.apply_changes(ContextChangeSet::circuit_changed());
    }

    /// Applies a change set and invalidates only affected analyses.
    pub fn apply_changes(&mut self, changes: ContextChangeSet) {
        if changes.circuit_changed {
            self.revision = self.revision.saturating_add(1);
            self.invalidate_analysis(AnalysisKey::CircuitCfg);
            self.invalidate_analysis(AnalysisKey::InstructionStats);
            self.invalidate_analysis(AnalysisKey::QubitUsage);
            self.invalidate_analysis(AnalysisKey::BasisAnalysis);
            self.invalidate_analysis(AnalysisKey::BlockSummary);
            self.invalidate_analysis(AnalysisKey::CouplingRequirements);
            self.invalidate_analysis(AnalysisKey::NativeSupportAnalysis);
            self.invalidate_analysis(AnalysisKey::CostAnalysis);
            return;
        }

        if changes.device_changed {
            self.invalidate_analysis(AnalysisKey::NativeSupportAnalysis);
            self.invalidate_analysis(AnalysisKey::CostAnalysis);
        }

        if changes.layout_changed || changes.mapping_invalidated {
            self.invalidate_analysis(AnalysisKey::NativeSupportAnalysis);
            self.invalidate_analysis(AnalysisKey::CostAnalysis);
        }
    }

    fn invalidate_analysis(&mut self, key: AnalysisKey) {
        match key {
            AnalysisKey::CircuitCfg => self.analysis.remove::<CircuitCFG>(),
            AnalysisKey::InstructionStats => self.analysis.remove::<InstructionStats>(),
            AnalysisKey::QubitUsage => self.analysis.remove::<QubitUsage>(),
            AnalysisKey::BlockSummary => self.analysis.remove::<BlockSummary>(),
            AnalysisKey::CouplingRequirements => self.analysis.remove::<CouplingRequirements>(),
            AnalysisKey::BasisAnalysis => self.analysis.remove::<BasisAnalysis>(),
            AnalysisKey::NativeSupportAnalysis => self.analysis.remove::<NativeSupportAnalysis>(),
            AnalysisKey::CostAnalysis => self.analysis.remove::<CostAnalysis>(),
        }
    }

    /// Runs the default verifier suite against the current context state.
    pub fn verify(&mut self) -> Result<(), CompilerError> {
        DefaultCircuitVerifier.verify(self.circuit())?;
        let cfg = self.cfg()?;
        DefaultCfgVerifier.verify(cfg)?;
        Ok(())
    }

    /// Returns a cached or newly built control-flow graph view of the circuit.
    pub fn cfg(&mut self) -> Result<&CircuitCFG, CompilerError> {
        self.get_or_build_analysis::<CircuitCFG>()
    }

    /// Backward-compatible alias for [`Self::cfg`].
    pub fn dag(&mut self) -> Result<&CircuitCFG, CompilerError> {
        self.cfg()
    }

    /// Returns cached or newly built circuit-wide instruction statistics.
    pub fn instruction_stats(&mut self) -> Result<&InstructionStats, CompilerError> {
        self.get_or_build_analysis::<InstructionStats>()
    }

    /// Returns cached or newly built per-qubit usage statistics.
    pub fn qubit_usage(&mut self) -> Result<&QubitUsage, CompilerError> {
        self.get_or_build_analysis::<QubitUsage>()
    }

    /// Returns cached or newly built block-level CFG summaries.
    pub fn block_summary(&mut self) -> Result<&BlockSummary, CompilerError> {
        self.get_or_build_analysis::<BlockSummary>()
    }

    /// Returns cached or newly built logical coupling requirements.
    pub fn coupling_requirements(&mut self) -> Result<&CouplingRequirements, CompilerError> {
        self.get_or_build_analysis::<CouplingRequirements>()
    }

    /// Returns cached or newly built normalized instruction-family usage.
    pub fn basis_analysis(&mut self) -> Result<&BasisAnalysis, CompilerError> {
        self.get_or_build_analysis::<BasisAnalysis>()
    }

    /// Returns cached or newly built target-native support diagnostics.
    pub fn native_support_analysis(&mut self) -> Result<&NativeSupportAnalysis, CompilerError> {
        self.get_or_build_analysis::<NativeSupportAnalysis>()
    }

    /// Returns cached or newly built logical and optional target-aware cost estimates.
    pub fn cost_analysis(&mut self) -> Result<&CostAnalysis, CompilerError> {
        self.get_or_build_analysis::<CostAnalysis>()
    }

    fn get_or_build_analysis<T: ContextAnalysis>(&mut self) -> Result<&T, CompilerError> {
        if self.analysis.get::<T>(self.revision).is_none() {
            let built = T::build(self)?;
            self.analysis.insert(self.revision, built);
        }

        Ok(self
            .analysis
            .get::<T>(self.revision)
            .expect("analysis cache must exist immediately after insertion"))
    }

    /// Ensures an analysis exists for the current revision.
    pub fn ensure_analysis(&mut self, key: AnalysisKey) -> Result<(), CompilerError> {
        match key {
            AnalysisKey::CircuitCfg => {
                let _ = self.cfg()?;
            }
            AnalysisKey::InstructionStats => {
                let _ = self.instruction_stats()?;
            }
            AnalysisKey::QubitUsage => {
                let _ = self.qubit_usage()?;
            }
            AnalysisKey::BlockSummary => {
                let _ = self.block_summary()?;
            }
            AnalysisKey::CouplingRequirements => {
                let _ = self.coupling_requirements()?;
            }
            AnalysisKey::BasisAnalysis => {
                let _ = self.basis_analysis()?;
            }
            AnalysisKey::NativeSupportAnalysis => {
                let _ = self.native_support_analysis()?;
            }
            AnalysisKey::CostAnalysis => {
                let _ = self.cost_analysis()?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CompilerContext;
    use crate::circuit::{Circuit, ConditionView, Instruction, Parameter, Qubit};
    use crate::compiler::analysis::AnalysisKey;
    use crate::compiler::error::CompilerError;
    use crate::device::{Device, Layout, Topology};
    use std::collections::HashSet;

    fn mock_device(name: &str, qubit_count: usize) -> Device {
        let qubits: Vec<_> = (0..qubit_count).map(|i| Qubit::new(i as u32)).collect();
        let topology = Topology::new(qubits.clone(), vec![]).unwrap();
        Device::new(name, HashSet::from_iter(qubits), topology).unwrap()
    }

    #[test]
    fn context_new_starts_without_device_or_layout() {
        let context = CompilerContext::new(Circuit::new(2));

        assert!(context.device().is_none());
        assert!(context.layout().is_none());
        assert_eq!(context.revision(), 0);
    }

    #[test]
    fn with_device_sets_target_name_metadata() {
        let context = CompilerContext::with_device(Circuit::new(1), mock_device("mock-qpu", 1));

        assert_eq!(context.metadata().target_name.as_deref(), Some("mock-qpu"));
    }

    #[test]
    fn dag_is_built_lazily_and_cached_per_revision() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let first = context.cfg().unwrap() as *const _;
        let second = context.cfg().unwrap() as *const _;

        assert_eq!(first, second);
    }

    #[test]
    fn instruction_stats_is_built_lazily_and_cached_per_revision() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let first = context.instruction_stats().unwrap() as *const _;
        let second = context.instruction_stats().unwrap() as *const _;

        assert_eq!(first, second);
        assert_eq!(context.instruction_stats().unwrap().total_ops, 0);
    }

    #[test]
    fn qubit_usage_is_built_lazily_and_cached_per_revision() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let first = context.qubit_usage().unwrap() as *const _;
        let second = context.qubit_usage().unwrap() as *const _;

        assert_eq!(first, second);
        assert_eq!(context.qubit_usage().unwrap().total_qubits_touched(), 0);
    }

    #[test]
    fn block_summary_is_built_lazily_and_cached_per_revision() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let first = context.block_summary().unwrap() as *const _;
        let second = context.block_summary().unwrap() as *const _;

        assert_eq!(first, second);
        assert_eq!(context.block_summary().unwrap().num_blocks(), 1);
    }

    #[test]
    fn coupling_requirements_is_built_lazily_and_cached_per_revision() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let first = context.coupling_requirements().unwrap() as *const _;
        let second = context.coupling_requirements().unwrap() as *const _;

        assert_eq!(first, second);
        assert_eq!(
            context
                .coupling_requirements()
                .unwrap()
                .total_two_qubit_ops(),
            0
        );
    }

    #[test]
    fn has_control_flow_detects_control_flow_operations() {
        let mut context = CompilerContext::new(Circuit::new(1));
        assert!(!context.has_control_flow());

        context
            .circuit_mut()
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![crate::circuit::Operation {
                    instruction: Instruction::from(crate::circuit::StandardGate::X),
                    qubits: smallvec::smallvec![Qubit::new(0)],
                    params: smallvec::smallvec![],
                    label: None,
                }],
                None,
            )
            .unwrap();

        assert!(context.has_control_flow());
    }

    #[test]
    fn has_symbolic_parameters_detects_unresolved_params() {
        let mut context = CompilerContext::new(Circuit::new(1));
        assert!(!context.has_symbolic_parameters());

        context
            .circuit_mut()
            .rx(Qubit::new(0), Parameter::symbol("theta"))
            .unwrap();

        assert!(context.has_symbolic_parameters());
    }

    #[test]
    fn ensure_analysis_builds_known_analysis() {
        let mut context = CompilerContext::new(Circuit::new(1));

        context.ensure_analysis(AnalysisKey::CircuitCfg).unwrap();
        context
            .ensure_analysis(AnalysisKey::InstructionStats)
            .unwrap();
        context.ensure_analysis(AnalysisKey::QubitUsage).unwrap();
        context.ensure_analysis(AnalysisKey::BlockSummary).unwrap();
        context
            .ensure_analysis(AnalysisKey::CouplingRequirements)
            .unwrap();
        assert!(context.cfg().is_ok());
        assert_eq!(context.instruction_stats().unwrap().total_ops, 0);
        assert_eq!(context.qubit_usage().unwrap().total_qubits_touched(), 0);
        assert_eq!(context.block_summary().unwrap().num_blocks(), 1);
        assert_eq!(
            context
                .coupling_requirements()
                .unwrap()
                .total_two_qubit_ops(),
            0
        );
    }

    #[test]
    fn ensure_analysis_propagates_build_error() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let err = context
            .ensure_analysis(AnalysisKey::NativeSupportAnalysis)
            .unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));
    }

    #[test]
    fn set_layout_invalidates_cached_native_support_analysis() {
        let mut circuit = Circuit::new(1);
        circuit
            .append(Instruction::Delay, [Qubit::new(0)], [], None)
            .unwrap();
        let device = mock_device("mock-qpu", 2).with_native_gates(vec![Instruction::Delay]);
        let mut context = CompilerContext::with_device(circuit, device);

        assert_eq!(
            context
                .native_support_analysis()
                .unwrap()
                .requires_layout_ops(),
            1
        );

        let layout = Layout::new(
            vec![Qubit::new(0)],
            vec![Qubit::new(0)],
            Some(std::collections::HashMap::from([(
                Qubit::new(0),
                Qubit::new(0),
            )])),
        )
        .unwrap();
        context.set_layout(layout);

        let rebuilt = context.native_support_analysis().unwrap();
        assert_eq!(rebuilt.requires_layout_ops(), 0);
        assert_eq!(rebuilt.supported_ops(), 1);
    }

    #[test]
    fn clear_layout_invalidates_cached_cost_analysis() {
        let mut circuit = Circuit::new(1);
        circuit
            .append(Instruction::Delay, [Qubit::new(0)], [], None)
            .unwrap();
        let device = mock_device("mock-qpu", 1).with_native_gates(vec![Instruction::Delay]);
        let mut context = CompilerContext::with_device(circuit, device);
        let layout = Layout::new(
            vec![Qubit::new(0)],
            vec![Qubit::new(0)],
            Some(std::collections::HashMap::from([(
                Qubit::new(0),
                Qubit::new(0),
            )])),
        )
        .unwrap();
        context.set_layout(layout);

        assert!(context.cost_analysis().unwrap().physical.is_some());

        context.clear_layout();

        let rebuilt = context.cost_analysis().unwrap();
        assert!(rebuilt.physical.is_some());
        assert_eq!(
            context
                .native_support_analysis()
                .unwrap()
                .requires_layout_ops(),
            1
        );
    }

    #[test]
    fn mark_mapping_invalid_invalidates_cached_native_support_analysis() {
        let mut circuit = Circuit::new(1);
        circuit
            .append(Instruction::Delay, [Qubit::new(0)], [], None)
            .unwrap();
        let device = mock_device("mock-qpu", 2).with_native_gates(vec![Instruction::Delay]);
        let mut context = CompilerContext::with_device(circuit, device);
        let layout = Layout::new(
            vec![Qubit::new(0)],
            vec![Qubit::new(0)],
            Some(std::collections::HashMap::from([(
                Qubit::new(0),
                Qubit::new(0),
            )])),
        )
        .unwrap();
        context.set_layout(layout);

        assert_eq!(
            context.native_support_analysis().unwrap().supported_ops(),
            1
        );

        context.mark_mapping_invalid();

        let rebuilt = context.native_support_analysis().unwrap();
        assert_eq!(rebuilt.supported_ops(), 0);
        assert_eq!(rebuilt.requires_layout_ops(), 1);
    }

    #[test]
    fn mark_circuit_changed_invalidates_cached_dag() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let initial_ops = context
            .cfg()
            .unwrap()
            .to_circuit()
            .unwrap()
            .operations()
            .len();
        context.circuit_mut().h(Qubit::new(0)).unwrap();
        context.mark_circuit_changed();
        let rebuilt_ops = context
            .cfg()
            .unwrap()
            .to_circuit()
            .unwrap()
            .operations()
            .len();

        assert_eq!(initial_ops, 0);
        assert_eq!(rebuilt_ops, 1);
        assert_eq!(context.revision(), 1);
    }

    #[test]
    fn mark_circuit_changed_invalidates_cached_instruction_stats() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let initial_ops = context.instruction_stats().unwrap().total_ops;
        context.circuit_mut().h(Qubit::new(0)).unwrap();
        context.mark_circuit_changed();
        let rebuilt_ops = context.instruction_stats().unwrap().total_ops;

        assert_eq!(initial_ops, 0);
        assert_eq!(rebuilt_ops, 1);
        assert_eq!(context.revision(), 1);
    }

    #[test]
    fn mark_circuit_changed_invalidates_cached_qubit_usage() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let initial_touched = context.qubit_usage().unwrap().total_qubits_touched();
        context.circuit_mut().h(Qubit::new(0)).unwrap();
        context.mark_circuit_changed();
        let rebuilt_touched = context.qubit_usage().unwrap().total_qubits_touched();

        assert_eq!(initial_touched, 0);
        assert_eq!(rebuilt_touched, 1);
        assert_eq!(context.revision(), 1);
    }

    #[test]
    fn mark_circuit_changed_invalidates_cached_block_summary() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let initial_op_count = context
            .block_summary()
            .unwrap()
            .entries()
            .next()
            .unwrap()
            .1
            .op_count;
        context.circuit_mut().h(Qubit::new(0)).unwrap();
        context.mark_circuit_changed();
        let rebuilt_op_count = context
            .block_summary()
            .unwrap()
            .entries()
            .next()
            .unwrap()
            .1
            .op_count;

        assert_eq!(initial_op_count, 0);
        assert_eq!(rebuilt_op_count, 1);
        assert_eq!(context.revision(), 1);
    }

    #[test]
    fn mark_circuit_changed_invalidates_cached_coupling_requirements() {
        let mut context = CompilerContext::new(Circuit::new(2));

        let initial_two_qubit_ops = context
            .coupling_requirements()
            .unwrap()
            .total_two_qubit_ops();
        context
            .circuit_mut()
            .cx(Qubit::new(0), Qubit::new(1))
            .unwrap();
        context.mark_circuit_changed();
        let rebuilt_two_qubit_ops = context
            .coupling_requirements()
            .unwrap()
            .total_two_qubit_ops();

        assert_eq!(initial_two_qubit_ops, 0);
        assert_eq!(rebuilt_two_qubit_ops, 1);
        assert_eq!(context.revision(), 1);
    }

    #[test]
    fn set_device_clears_layout_and_invalidates_analysis() {
        let mut context = CompilerContext::with_device(Circuit::new(1), mock_device("qpu-a", 1));
        let logical = vec![Qubit::new(0)];
        let physical = vec![Qubit::new(10)];
        let layout = Layout::new(logical, physical, None).unwrap();

        let cached_ops = context
            .cfg()
            .unwrap()
            .to_circuit()
            .unwrap()
            .operations()
            .len();
        context.set_layout(layout);
        context.set_device(mock_device("qpu-b", 1));
        let rebuilt_ops = context
            .cfg()
            .unwrap()
            .to_circuit()
            .unwrap()
            .operations()
            .len();

        assert!(context.layout().is_none());
        assert_eq!(context.metadata().target_name.as_deref(), Some("qpu-b"));
        assert_eq!(cached_ops, rebuilt_ops);
    }

    #[test]
    fn metadata_changes_do_not_invalidate_analysis() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let first = context.cfg().unwrap() as *const _;
        context.metadata_mut().workflow_name = Some("routing".to_string());
        let second = context.cfg().unwrap() as *const _;

        assert_eq!(first, second);
    }

    #[test]
    fn set_layout_preserves_cached_instruction_stats() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let before = context.instruction_stats().unwrap() as *const _;
        let layout = Layout::new(vec![Qubit::new(0)], vec![Qubit::new(0)], None).unwrap();
        context.set_layout(layout);
        let after = context.instruction_stats().unwrap() as *const _;

        assert_eq!(before, after);
    }

    #[test]
    fn set_device_preserves_cached_cfg() {
        let mut context = CompilerContext::new(Circuit::new(1));

        let before = context.cfg().unwrap() as *const _;
        context.set_device(mock_device("mock-qpu", 1));
        let after = context.cfg().unwrap() as *const _;

        assert_eq!(before, after);
    }
}
