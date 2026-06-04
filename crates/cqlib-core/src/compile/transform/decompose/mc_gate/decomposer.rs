// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Circuit-level multi-controlled-gate decomposition.
//!
//! This module connects the pure synthesis primitives in [`super`] to the
//! compiler ancillary-resource manager. It rebuilds a circuit while replacing
//! every [`Instruction::McGate`] with a deterministic, resource-aware
//! decomposition. Control-flow bodies are rebuilt recursively because each
//! selected primitive restores its leased ancillas before the emitted sequence
//! completes.
//!
//! Candidate selection is intentionally separate from physical routing. This
//! pass operates on logical qubits before layout. The device-aware convenience
//! entry point uses device capacity as a hard logical-qubit limit, but does not
//! claim to satisfy coupling-map constraints or score candidates by topology.

use super::{
    decompose_fsim_n_clean, decompose_fsim_no_aux, decompose_hadamard_n_clean,
    decompose_hadamard_no_aux, decompose_pauli_1_clean_b95, decompose_pauli_1_clean_kg24,
    decompose_pauli_1_dirty, decompose_pauli_2_clean, decompose_pauli_2_dirty,
    decompose_pauli_n_clean, decompose_pauli_n_dirty, decompose_pauli_no_aux,
    decompose_pauli_rotation_n_clean, decompose_pauli_rotation_no_aux, decompose_pauli_small,
    decompose_phase_n_clean, decompose_phase_no_aux, decompose_qcis_n_clean, decompose_qcis_no_aux,
    decompose_rotation_n_clean, decompose_rotation_no_aux, decompose_swap_n_clean,
    decompose_swap_no_aux, decompose_unitary_n_clean, decompose_unitary_no_aux,
};
use crate::circuit::operation::ValueOperation;
use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, MCGate, Operation, ParameterValue,
    Qubit, StandardGate, WhileLoopGate,
};
use crate::compile::CompilerError;
use crate::compile::resource::{
    AncillaRequirement, ResourceError, ResourceLimits, ResourceManager, ResourcePolicy,
    ResourceRequest,
};
use crate::compile::transform::TransformResult;
use crate::device::Device;
use smallvec::SmallVec;
use std::collections::BTreeSet;

const DECOMPOSE_MC_GATES_NAME: &str = "decompose.mc_gates";

/// Configuration for circuit-level multi-controlled-gate decomposition.
///
/// The policy controls whether synthesis may allocate clean logical ancillas
/// before layout or borrow input qubits under the dirty-restoration contract.
/// The limits provide hard logical-qubit bounds, such as target-device
/// capacity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McGateDecomposeConfig {
    /// Ancillary-resource permissions for this decomposition pass.
    pub resource_policy: ResourcePolicy,
    /// Hard logical-qubit limits for this decomposition pass.
    pub resource_limits: ResourceLimits,
}

/// Rewrites every supported [`Instruction::McGate`] in `circuit`.
///
/// Multi-controlled gates inside `if`, `else`, and `while` bodies are rebuilt
/// recursively. Expanded operations do not inherit the source `McGate` label,
/// because one source operation may lower to many operations with different
/// roles. Preserved operations retain their labels.
///
/// Candidate selection is deterministic. Multi-controlled Pauli gates try the
/// available exact MCX algorithms in a fixed two-qubit-cost-oriented order and
/// fall back to ancillary-free synthesis. Other supported gate families try
/// their clean-accumulator primitive first when it would consume ancillas, then
/// fall back to ancillary-free synthesis.
///
/// # Errors
///
/// Returns [`CompilerError`] when the input circuit is inconsistent, a
/// multi-controlled gate family is unsupported, all synthesis candidates fail,
/// or ancillary-resource bookkeeping violates its contract.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Instruction, MCGate, Qubit, StandardGate};
/// use cqlib_core::compile::resource::ResourcePolicy;
/// use cqlib_core::compile::transform::decompose::mc_gate::{
///     McGateDecomposeConfig, decompose_mc_gates,
/// };
///
/// let mut circuit = Circuit::new(3);
/// circuit
///     .append(
///         Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
///         [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
///         [],
///         None,
///     )
///     .unwrap();
///
/// let result = decompose_mc_gates(
///     &circuit,
///     McGateDecomposeConfig {
///         resource_policy: ResourcePolicy::default(),
///         ..McGateDecomposeConfig::default()
///     },
/// )
/// .unwrap();
///
/// assert!(result.changed);
/// assert!(matches!(
///     result.circuit.operations()[0].instruction,
///     Instruction::Standard(StandardGate::CCX),
/// ));
/// ```
pub fn decompose_mc_gates(
    circuit: &Circuit,
    config: McGateDecomposeConfig,
) -> Result<TransformResult, CompilerError> {
    McGateDecomposer::new(circuit, config)?.run()
}

/// Rewrites multi-controlled gates while enforcing target-device capacity.
///
/// This is a pre-layout logical transform. It limits the complete logical
/// circuit to [`Device::num_usable_qubits`], but deliberately does not inspect
/// coupling topology. Physical connectivity belongs to later layout and
/// routing stages where logical qubits have physical mappings.
///
/// # Errors
///
/// Returns the errors documented by [`decompose_mc_gates`], including an input
/// error when the source circuit is already wider than the usable device.
pub fn decompose_mc_gates_for_device(
    circuit: &Circuit,
    device: &Device,
    resource_policy: ResourcePolicy,
) -> Result<TransformResult, CompilerError> {
    decompose_mc_gates(
        circuit,
        McGateDecomposeConfig {
            resource_policy,
            resource_limits: ResourceLimits {
                max_total_qubits: Some(device.num_usable_qubits()),
            },
        },
    )
}

struct McGateDecomposer<'a> {
    source: &'a Circuit,
    target: Circuit,
    resources: ResourceManager,
    changed: bool,
}

impl<'a> McGateDecomposer<'a> {
    fn new(source: &'a Circuit, config: McGateDecomposeConfig) -> Result<Self, CompilerError> {
        let mut target = Circuit::from_qubits(source.qubits())?;
        target.set_global_phase(source.global_phase());
        let resources =
            ResourceManager::from_circuit(&target, config.resource_policy, config.resource_limits)
                .map_err(resource_input_failed)?;
        Ok(Self {
            source,
            target,
            resources,
            changed: false,
        })
    }

    fn run(mut self) -> Result<TransformResult, CompilerError> {
        let source = self.source;
        for operation in source.operations() {
            let operations = self.rebuild_operation(operation)?;
            for operation in operations {
                self.append_top_level(operation)?;
            }
        }
        self.resources
            .verify_idle(&self.target)
            .map_err(resource_invariant_failed)?;
        Ok(TransformResult {
            circuit: self.target,
            changed: self.changed,
        })
    }

    fn rebuild_sequence(
        &mut self,
        source_operations: &[Operation],
    ) -> Result<Vec<Operation>, CompilerError> {
        let mut operations = Vec::with_capacity(source_operations.len());
        for operation in source_operations {
            operations.extend(self.rebuild_operation(operation)?);
        }
        Ok(operations)
    }

    fn rebuild_operation(
        &mut self,
        operation: &Operation,
    ) -> Result<Vec<Operation>, CompilerError> {
        match &operation.instruction {
            Instruction::McGate(gate) => self.decompose_mc_operation(gate, operation),
            Instruction::ControlFlowGate(flow) => {
                Ok(vec![self.rebuild_control_flow(operation, flow)?])
            }
            _ => Ok(vec![Operation {
                instruction: operation.instruction.clone(),
                qubits: operation.qubits.clone(),
                params: self.remap_source_params(&operation.params)?,
                label: operation.label.clone(),
            }]),
        }
    }

    fn rebuild_control_flow(
        &mut self,
        operation: &Operation,
        flow: &ControlFlow,
    ) -> Result<Operation, CompilerError> {
        let instruction = match flow {
            ControlFlow::IfElse(gate) => {
                let true_body = self.rebuild_sequence(gate.true_body())?;
                let false_body = gate
                    .false_body()
                    .map(|body| self.rebuild_sequence(body))
                    .transpose()?;
                Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                    gate.condition(),
                    true_body,
                    false_body,
                )))
            }
            ControlFlow::WhileLoop(gate) => {
                let body = self.rebuild_sequence(gate.body())?;
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                    gate.condition(),
                    body,
                )))
            }
        };
        let qubits = control_flow_qubits(&instruction, &self.target.qubits());
        Ok(Operation {
            instruction,
            qubits,
            params: self.remap_source_params(&operation.params)?,
            label: operation.label.clone(),
        })
    }

    fn decompose_mc_operation(
        &mut self,
        gate: &MCGate,
        operation: &Operation,
    ) -> Result<Vec<Operation>, CompilerError> {
        self.validate_mc_operation(gate, operation)?;
        let params = self.resolve_source_params(&operation.params)?;
        let num_controls = gate.num_ctrl_qubits();
        let controls = &operation.qubits[..num_controls];
        let targets = &operation.qubits[num_controls..];
        let values = self.synthesize_mc_gate(*gate.base_gate(), &params, controls, targets)?;
        self.changed = true;
        values
            .into_iter()
            .map(|operation| self.intern_value_operation(operation))
            .collect()
    }

    fn validate_mc_operation(
        &self,
        gate: &MCGate,
        operation: &Operation,
    ) -> Result<(), CompilerError> {
        if operation.qubits.len() != gate.num_qubits() {
            return Err(CompilerError::InvalidInput(format!(
                "multi-controlled gate {gate} expects {} qubits, got {}",
                gate.num_qubits(),
                operation.qubits.len()
            )));
        }
        if operation.params.len() != gate.num_params() {
            return Err(CompilerError::InvalidInput(format!(
                "multi-controlled gate {gate} expects {} parameters, got {}",
                gate.num_params(),
                operation.params.len()
            )));
        }
        Ok(())
    }

    fn synthesize_mc_gate(
        &mut self,
        gate: StandardGate,
        params: &[ParameterValue],
        controls: &[Qubit],
        targets: &[Qubit],
    ) -> Result<Vec<ValueOperation>, CompilerError> {
        if controls.is_empty() {
            return Ok(vec![ValueOperation {
                instruction: Instruction::Standard(gate),
                qubits: targets.iter().copied().collect(),
                params: params.iter().cloned().collect(),
                label: None,
            }]);
        }

        let excluded = controls
            .iter()
            .chain(targets)
            .copied()
            .collect::<BTreeSet<_>>();
        match gate {
            StandardGate::X
            | StandardGate::CX
            | StandardGate::CCX
            | StandardGate::Y
            | StandardGate::CY
            | StandardGate::Z
            | StandardGate::CZ => {
                self.synthesize_pauli(gate, controls, one_target(gate, targets)?, &excluded)
            }
            StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ => {
                let theta = one_param(gate, params)?;
                let target = one_target(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| decompose_rotation_n_clean(gate, theta, controls, target, ancillas),
                    || decompose_rotation_no_aux(gate, theta, controls, target),
                )
            }
            StandardGate::S
            | StandardGate::SDG
            | StandardGate::T
            | StandardGate::TDG
            | StandardGate::Phase => {
                let theta = phase_param(gate, params)?;
                let target = one_target(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| decompose_phase_n_clean(gate, theta, controls, target, ancillas),
                    || decompose_phase_no_aux(gate, theta, controls, target),
                )
            }
            StandardGate::H => {
                let target = one_target(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| decompose_hadamard_n_clean(controls, target, ancillas),
                    || decompose_hadamard_no_aux(controls, target),
                )
            }
            StandardGate::U => {
                let [theta, phi, lambda] = three_params(gate, params)?;
                let target = one_target(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| {
                        decompose_unitary_n_clean(theta, phi, lambda, controls, target, ancillas)
                    },
                    || decompose_unitary_no_aux(theta, phi, lambda, controls, target),
                )
            }
            StandardGate::RXX | StandardGate::RYY | StandardGate::RZZ | StandardGate::RZX => {
                let theta = one_param(gate, params)?;
                let [first, second] = two_targets(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| {
                        decompose_pauli_rotation_n_clean(
                            gate, theta, controls, first, second, ancillas,
                        )
                    },
                    || decompose_pauli_rotation_no_aux(gate, theta, controls, first, second),
                )
            }
            StandardGate::SWAP => {
                let [first, second] = two_targets(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| decompose_swap_n_clean(controls, first, second, ancillas),
                    || decompose_swap_no_aux(controls, first, second),
                )
            }
            StandardGate::X2P
            | StandardGate::X2M
            | StandardGate::Y2P
            | StandardGate::Y2M
            | StandardGate::XY2P
            | StandardGate::XY2M => {
                let target = one_target(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len().saturating_sub(1),
                    &excluded,
                    |ancillas| decompose_qcis_n_clean(gate, params, controls, target, ancillas),
                    || decompose_qcis_no_aux(gate, params, controls, target),
                )
            }
            StandardGate::FSIM => {
                let [first, second] = two_targets(gate, targets)?;
                self.synthesize_with_optional_clean(
                    controls.len(),
                    &excluded,
                    |ancillas| decompose_fsim_n_clean(params, controls, first, second, ancillas),
                    || decompose_fsim_no_aux(params, controls, first, second),
                )
            }
            _ => Err(CompilerError::TransformFailed {
                name: DECOMPOSE_MC_GATES_NAME,
                reason: format!("multi-controlled {gate} decomposition is not supported"),
            }),
        }
    }

    fn synthesize_pauli(
        &mut self,
        pauli: StandardGate,
        controls: &[Qubit],
        target: Qubit,
        excluded: &BTreeSet<Qubit>,
    ) -> Result<Vec<ValueOperation>, CompilerError> {
        if controls.len() <= 2 {
            return decompose_pauli_small(pauli, controls, target);
        }

        // Try exact Pauli/MCX candidates in a fixed two-qubit-cost-oriented
        // order: 2 clean KG24, 1 clean KG24, n clean V-chain, n dirty V-chain,
        // 2 dirty KG24, 1 dirty KG24, 1 clean B95, then no-auxiliary fallback.
        // This keeps selection deterministic while preferring low-cost
        // ancillary-assisted constructions when the resource policy allows
        // them.
        if let Some(operations) =
            self.try_resource_candidate(excluded, AncillaRequirement::CleanZero, 2, |ancillas| {
                decompose_pauli_2_clean(pauli, controls, target, [ancillas[0], ancillas[1]])
            })?
        {
            return Ok(operations);
        }
        if let Some(operations) =
            self.try_resource_candidate(excluded, AncillaRequirement::CleanZero, 1, |ancillas| {
                decompose_pauli_1_clean_kg24(pauli, controls, target, ancillas[0])
            })?
        {
            return Ok(operations);
        }

        let v_chain_ancillas = controls.len() - 2;
        if let Some(operations) = self.try_resource_candidate(
            excluded,
            AncillaRequirement::CleanZero,
            v_chain_ancillas,
            |ancillas| decompose_pauli_n_clean(pauli, controls, target, ancillas),
        )? {
            return Ok(operations);
        }
        if let Some(operations) = self.try_resource_candidate(
            excluded,
            AncillaRequirement::Dirty,
            v_chain_ancillas,
            |ancillas| decompose_pauli_n_dirty(pauli, controls, target, ancillas),
        )? {
            return Ok(operations);
        }
        if let Some(operations) =
            self.try_resource_candidate(excluded, AncillaRequirement::Dirty, 2, |ancillas| {
                decompose_pauli_2_dirty(pauli, controls, target, [ancillas[0], ancillas[1]])
            })?
        {
            return Ok(operations);
        }
        if let Some(operations) =
            self.try_resource_candidate(excluded, AncillaRequirement::Dirty, 1, |ancillas| {
                decompose_pauli_1_dirty(pauli, controls, target, ancillas[0])
            })?
        {
            return Ok(operations);
        }
        if let Some(operations) =
            self.try_resource_candidate(excluded, AncillaRequirement::CleanZero, 1, |ancillas| {
                decompose_pauli_1_clean_b95(pauli, controls, target, ancillas[0])
            })?
        {
            return Ok(operations);
        }
        decompose_pauli_no_aux(pauli, controls, target)
    }

    fn synthesize_with_optional_clean(
        &mut self,
        required_ancillas: usize,
        excluded: &BTreeSet<Qubit>,
        synthesize_clean: impl FnOnce(&[Qubit]) -> Result<Vec<ValueOperation>, CompilerError>,
        synthesize_no_aux: impl FnOnce() -> Result<Vec<ValueOperation>, CompilerError>,
    ) -> Result<Vec<ValueOperation>, CompilerError> {
        if required_ancillas > 0
            && let Some(operations) = self.try_resource_candidate(
                excluded,
                AncillaRequirement::CleanZero,
                required_ancillas,
                synthesize_clean,
            )?
        {
            return Ok(operations);
        }
        synthesize_no_aux()
    }

    fn try_resource_candidate(
        &mut self,
        excluded: &BTreeSet<Qubit>,
        requirement: AncillaRequirement,
        count: usize,
        synthesize: impl FnOnce(&[Qubit]) -> Result<Vec<ValueOperation>, CompilerError>,
    ) -> Result<Option<Vec<ValueOperation>>, CompilerError> {
        let request = ResourceRequest {
            requirement,
            count,
            excluded: excluded.clone(),
        };
        let plan = match self.resources.preview(&request) {
            Ok(plan) => plan,
            Err(error) if resource_candidate_is_unavailable(&error) => return Ok(None),
            Err(error) => return Err(resource_invariant_failed(error)),
        };

        // Synthesis primitives are pure. Generate from the prospective qubits
        // before committing so a rejected candidate cannot allocate ancillas
        // or invalidate later previews.
        let operations = match synthesize(plan.qubits()) {
            Ok(operations) => operations,
            Err(CompilerError::TransformFailed { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        // Some recursive primitives accept a fixed ancillary signature even
        // when a small instance does not consume every provided qubit. Avoid
        // widening the logical circuit for resources that the emitted circuit
        // does not actually use; the next candidate can synthesize the same
        // instance with a tighter lease.
        if !plan
            .qubits()
            .iter()
            .all(|qubit| value_operations_use_qubit(&operations, *qubit))
        {
            return Ok(None);
        }
        let lease = self
            .resources
            .commit(&mut self.target, plan)
            .map_err(resource_invariant_failed)?;
        self.resources
            .release(&lease)
            .map_err(resource_invariant_failed)?;
        Ok(Some(operations))
    }

    fn resolve_source_params(
        &self,
        params: &[CircuitParam],
    ) -> Result<Vec<ParameterValue>, CompilerError> {
        params
            .iter()
            .map(|param| self.resolve_source_param(param))
            .collect()
    }

    fn resolve_source_param(&self, param: &CircuitParam) -> Result<ParameterValue, CompilerError> {
        match param {
            CircuitParam::Fixed(value) => {
                if !value.is_finite() {
                    return Err(CompilerError::InvalidInput(format!(
                        "non-finite fixed parameter {value}"
                    )));
                }
                Ok(ParameterValue::Fixed(*value))
            }
            CircuitParam::Index(index) => self
                .source
                .parameters()
                .get_index(*index as usize)
                .cloned()
                .map(ParameterValue::Param)
                .ok_or_else(|| {
                    CompilerError::InvalidInput(format!("missing parameter index {index}"))
                }),
        }
    }

    fn remap_source_params(
        &mut self,
        params: &[CircuitParam],
    ) -> Result<SmallVec<[CircuitParam; 1]>, CompilerError> {
        let values = self.resolve_source_params(params)?;
        values
            .into_iter()
            .map(|value| self.intern_value_param(value))
            .collect()
    }

    fn intern_value_operation(
        &mut self,
        operation: ValueOperation,
    ) -> Result<Operation, CompilerError> {
        Ok(Operation {
            instruction: operation.instruction,
            qubits: operation.qubits,
            params: operation
                .params
                .into_iter()
                .map(|value| self.intern_value_param(value))
                .collect::<Result<_, _>>()?,
            label: None,
        })
    }

    fn intern_value_param(&mut self, value: ParameterValue) -> Result<CircuitParam, CompilerError> {
        match value {
            ParameterValue::Fixed(value) => {
                if !value.is_finite() {
                    return Err(CompilerError::InvalidInput(format!(
                        "non-finite parameter value {value}"
                    )));
                }
                Ok(CircuitParam::Fixed(if value == 0.0 { 0.0 } else { value }))
            }
            ParameterValue::Param(parameter) => {
                let (index, _) = self.target.add_parameter(parameter);
                Ok(CircuitParam::Index(index as u32))
            }
        }
    }

    fn append_top_level(&mut self, operation: Operation) -> Result<(), CompilerError> {
        let params = operation
            .params
            .iter()
            .map(|param| target_param_to_value(&self.target, param))
            .collect::<Result<Vec<_>, _>>()?;
        self.target.append(
            operation.instruction,
            operation.qubits,
            params,
            operation.label.as_deref(),
        )?;
        Ok(())
    }
}

fn value_operations_use_qubit(operations: &[ValueOperation], qubit: Qubit) -> bool {
    operations
        .iter()
        .any(|operation| operation.qubits.contains(&qubit))
}

fn one_param(
    gate: StandardGate,
    params: &[ParameterValue],
) -> Result<&ParameterValue, CompilerError> {
    let [param] = params else {
        return Err(invalid_primitive_signature(
            gate,
            "1 parameter",
            params.len(),
        ));
    };
    Ok(param)
}

fn three_params(
    gate: StandardGate,
    params: &[ParameterValue],
) -> Result<[&ParameterValue; 3], CompilerError> {
    let [theta, phi, lambda] = params else {
        return Err(invalid_primitive_signature(
            gate,
            "3 parameters",
            params.len(),
        ));
    };
    Ok([theta, phi, lambda])
}

fn phase_param(
    gate: StandardGate,
    params: &[ParameterValue],
) -> Result<Option<&ParameterValue>, CompilerError> {
    match gate {
        StandardGate::Phase => Ok(Some(one_param(gate, params)?)),
        StandardGate::S | StandardGate::SDG | StandardGate::T | StandardGate::TDG
            if params.is_empty() =>
        {
            Ok(None)
        }
        _ => Err(invalid_primitive_signature(
            gate,
            "0 parameters",
            params.len(),
        )),
    }
}

fn one_target(gate: StandardGate, targets: &[Qubit]) -> Result<Qubit, CompilerError> {
    let [target] = targets else {
        return Err(invalid_primitive_signature(gate, "1 target", targets.len()));
    };
    Ok(*target)
}

fn two_targets(gate: StandardGate, targets: &[Qubit]) -> Result<[Qubit; 2], CompilerError> {
    let [first, second] = targets else {
        return Err(invalid_primitive_signature(
            gate,
            "2 targets",
            targets.len(),
        ));
    };
    Ok([*first, *second])
}

fn invalid_primitive_signature(gate: StandardGate, expected: &str, actual: usize) -> CompilerError {
    CompilerError::InvariantViolation(format!(
        "validated multi-controlled {gate} operation requires {expected}, got {actual}"
    ))
}

fn resource_candidate_is_unavailable(error: &ResourceError) -> bool {
    matches!(
        error,
        ResourceError::InsufficientResources { .. }
            | ResourceError::CapacityExceeded { .. }
            | ResourceError::QubitIdOverflow
    )
}

fn resource_input_failed(error: ResourceError) -> CompilerError {
    CompilerError::InvalidInput(format!("cannot initialize ancillary resources: {error}"))
}

fn resource_invariant_failed(error: ResourceError) -> CompilerError {
    CompilerError::InvariantViolation(format!("ancillary-resource bookkeeping failed: {error}"))
}

fn target_param_to_value(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<ParameterValue, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(*value)),
        CircuitParam::Index(index) => circuit
            .parameters()
            .get_index(*index as usize)
            .cloned()
            .map(ParameterValue::Param)
            .ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "multi-controlled-gate decomposition produced missing target parameter index {index}"
                ))
            }),
    }
}

fn control_flow_qubits(
    instruction: &Instruction,
    circuit_qubits: &[Qubit],
) -> SmallVec<[Qubit; 3]> {
    let mut required = BTreeSet::new();
    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            required.insert(gate.condition().qubit);
            collect_body_qubits(gate.true_body(), &mut required);
            if let Some(false_body) = gate.false_body() {
                collect_body_qubits(false_body, &mut required);
            }
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            required.insert(gate.condition().qubit);
            collect_body_qubits(gate.body(), &mut required);
        }
        _ => {}
    }
    circuit_qubits
        .iter()
        .copied()
        .filter(|qubit| required.contains(qubit))
        .collect()
}

fn collect_body_qubits(operations: &[Operation], out: &mut BTreeSet<Qubit>) {
    for operation in operations {
        out.extend(operation.qubits.iter().copied());
        match &operation.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                out.insert(gate.condition().qubit);
                collect_body_qubits(gate.true_body(), out);
                if let Some(false_body) = gate.false_body() {
                    collect_body_qubits(false_body, out);
                }
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                out.insert(gate.condition().qubit);
                collect_body_qubits(gate.body(), out);
            }
            _ => {}
        }
    }
}
