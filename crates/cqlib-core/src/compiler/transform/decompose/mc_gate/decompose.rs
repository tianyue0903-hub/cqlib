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

use crate::circuit::{CircuitParam, Instruction, MCGate, Operation, Qubit, StandardGate};
use crate::compiler::error::CompilerError;
use std::collections::HashSet;
use std::fmt;

pub(super) const DECOMPOSE_MC_GATE_NAME: &str = "decompose.mc_gate";
const DEFAULT_MAX_EXPANSION_OPS: usize = 10_000;
const DEFAULT_MAX_RECURSION_DEPTH: usize = 64;

/// Explicit ancillary-qubit strategy for multi-controlled gate decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AncillaMode {
    /// Do not use any qubit outside the operation operands.
    NoAncilla,
    /// Use only caller-provided clean work qubits.
    CleanAncilla,
    /// Use only caller-provided borrowed work qubits.
    DirtyAncilla,
}

impl fmt::Display for AncillaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAncilla => write!(f, "no_ancilla"),
            Self::CleanAncilla => write!(f, "clean_ancilla"),
            Self::DirtyAncilla => write!(f, "dirty_ancilla"),
        }
    }
}

/// Local configuration for MCGate decomposition planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McGateDecomposeConfig {
    /// Selected ancillary-qubit strategy.
    pub ancilla_mode: AncillaMode,
    /// Caller-provided clean work qubits.
    pub clean_ancillas: Vec<Qubit>,
    /// Caller-provided borrowed work qubits.
    pub dirty_ancillas: Vec<Qubit>,
    /// Maximum number of operations a decomposition may materialize.
    pub max_expansion_ops: usize,
    /// Maximum recursion depth for control-lifting plans.
    pub max_recursion_depth: usize,
    /// Whether labeled MCGate operations should be protected from lowering.
    pub skip_labeled_ops: bool,
}

impl Default for McGateDecomposeConfig {
    fn default() -> Self {
        Self {
            ancilla_mode: AncillaMode::NoAncilla,
            clean_ancillas: Vec::new(),
            dirty_ancillas: Vec::new(),
            max_expansion_ops: DEFAULT_MAX_EXPANSION_OPS,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            skip_labeled_ops: true,
        }
    }
}

impl McGateDecomposeConfig {
    /// Creates MCGate decomposition config with production defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Controls whether labeled MCGate operations are protected from lowering.
    pub fn skip_labeled_ops(mut self, enabled: bool) -> Self {
        self.skip_labeled_ops = enabled;
        self
    }

    /// Returns whether labeled MCGate operations are protected from lowering.
    pub const fn skips_labeled_ops(&self) -> bool {
        self.skip_labeled_ops
    }
}

/// Result of applying the operation-level MCGate lowering adapter.
#[derive(Debug, Clone)]
pub struct McGateOperationDecomposeResult {
    /// Whether the source operation was replaced by the returned sequence.
    pub changed: bool,
    /// Operations to emit into the rebuilt sequence.
    pub operations: Vec<Operation>,
    /// Integration notes to be forwarded into a transformer outcome.
    pub notes: Vec<String>,
}

impl McGateOperationDecomposeResult {
    fn unchanged(operation: Operation) -> Self {
        Self {
            changed: false,
            operations: vec![operation],
            notes: Vec::new(),
        }
    }

    fn changed(operations: Vec<Operation>) -> Self {
        Self {
            changed: true,
            operations,
            notes: Vec::new(),
        }
    }
}

/// Gate family selected directly from the MCGate base gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McGateFamily {
    /// Pauli-family gates: `X`, `Y`, `Z`, `CX`, `CY`, `CZ`, `CCX`.
    Pauli,
    /// Phase-family gates: `S`, `SDG`, `T`, `TDG`, `Phase`.
    Phase,
    /// Rotation-family gates: `RX`, `RY`, `RZ`, `CRX`, `CRY`, `CRZ`.
    Rotation,
    /// One-qubit non-Pauli and non-rotation gates.
    OneQubit,
    /// Swap-family gate.
    Swap,
    /// Fermionic simulation gate.
    Fsim,
    /// Pauli-interaction gates: `RXX`, `RYY`, `RZZ`, `RZX`.
    PauliInteraction,
    /// Identity gate.
    Identity,
    /// Gate family not supported by this decomposition module.
    Unsupported,
}

impl McGateFamily {
    /// Classifies a multi-controlled gate by its base gate only.
    pub const fn classify(base_gate: StandardGate) -> Self {
        match base_gate {
            StandardGate::I => Self::Identity,
            StandardGate::X
            | StandardGate::Y
            | StandardGate::Z
            | StandardGate::CX
            | StandardGate::CY
            | StandardGate::CZ
            | StandardGate::CCX => Self::Pauli,
            StandardGate::S
            | StandardGate::SDG
            | StandardGate::T
            | StandardGate::TDG
            | StandardGate::Phase => Self::Phase,
            StandardGate::RX
            | StandardGate::RY
            | StandardGate::RZ
            | StandardGate::CRX
            | StandardGate::CRY
            | StandardGate::CRZ => Self::Rotation,
            StandardGate::H
            | StandardGate::U
            | StandardGate::X2P
            | StandardGate::X2M
            | StandardGate::Y2P
            | StandardGate::Y2M
            | StandardGate::RXY
            | StandardGate::XY
            | StandardGate::XY2P
            | StandardGate::XY2M => Self::OneQubit,
            StandardGate::SWAP => Self::Swap,
            StandardGate::FSIM => Self::Fsim,
            StandardGate::RXX | StandardGate::RYY | StandardGate::RZZ | StandardGate::RZX => {
                Self::PauliInteraction
            }
            StandardGate::GPhase => Self::Unsupported,
        }
    }
}

impl fmt::Display for McGateFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pauli => write!(f, "pauli"),
            Self::Phase => write!(f, "phase"),
            Self::Rotation => write!(f, "rotation"),
            Self::OneQubit => write!(f, "one_qubit"),
            Self::Swap => write!(f, "swap"),
            Self::Fsim => write!(f, "fsim"),
            Self::PauliInteraction => write!(f, "pauli_interaction"),
            Self::Identity => write!(f, "identity"),
            Self::Unsupported => write!(f, "unsupported"),
        }
    }
}

/// Read-only partition of MCGate operation operands.
///
/// The view preserves the IR operand contract:
/// `[added_controls..., base_inherent_controls..., base_targets...]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McGateOperandView<'a> {
    base_gate: StandardGate,
    added_controls: &'a [Qubit],
    inherent_controls: &'a [Qubit],
    targets: &'a [Qubit],
    all_controls: Vec<Qubit>,
    has_symbolic_params: bool,
}

impl<'a> McGateOperandView<'a> {
    /// Builds an operand view after local arity and resource validation.
    pub fn new(
        gate: &MCGate,
        qubits: &'a [Qubit],
        params: &[CircuitParam],
        config: &McGateDecomposeConfig,
    ) -> Result<Self, CompilerError> {
        validate_operand_arity(gate, qubits, params, config)?;

        let base_gate = *gate.base_gate();
        let base_control_count = base_gate.num_ctrl_qubits();
        let base_qubit_count = base_gate.num_qubits();
        let target_count =
            base_qubit_count
                .checked_sub(base_control_count)
                .ok_or_else(|| {
                    mc_gate_error(
                        gate,
                        params,
                        config,
                        format!(
                            "invalid base gate metadata: base qubits {base_qubit_count} < base controls {base_control_count}"
                        ),
                    )
                })?;
        let added_control_count =
            qubits
                .len()
                .checked_sub(base_qubit_count)
                .ok_or_else(|| {
                    mc_gate_error(
                        gate,
                        params,
                        config,
                        format!(
                            "operation qubit count {} is smaller than base gate qubit count {base_qubit_count}",
                            qubits.len()
                        ),
                    )
                })?;
        let total_controls = added_control_count + base_control_count;

        if base_gate == StandardGate::GPhase && total_controls > 0 {
            return Err(mc_gate_error(
                gate,
                params,
                config,
                "controlled GPhase is not supported".to_string(),
            ));
        }

        if target_count == 0 && total_controls > 0 {
            return Err(mc_gate_error(
                gate,
                params,
                config,
                "target-less controlled operation is not supported".to_string(),
            ));
        }

        validate_distinct_operands(gate, qubits, params, config)?;
        validate_configured_ancillas(gate, qubits, params, config)?;

        let added_controls = &qubits[..added_control_count];
        let base_operands = &qubits[added_control_count..];
        let inherent_controls = &base_operands[..base_control_count];
        let targets = &base_operands[base_control_count..];

        let mut all_controls = Vec::with_capacity(total_controls);
        all_controls.extend_from_slice(added_controls);
        all_controls.extend_from_slice(inherent_controls);

        Ok(Self {
            base_gate,
            added_controls,
            inherent_controls,
            targets,
            all_controls,
            has_symbolic_params: params
                .iter()
                .any(|param| matches!(param, CircuitParam::Index(_))),
        })
    }

    /// Returns the unchanged base gate.
    pub const fn base_gate(&self) -> StandardGate {
        self.base_gate
    }

    /// Returns the family selected for the unchanged base gate.
    pub fn family(&self) -> McGateFamily {
        McGateFamily::classify(self.base_gate)
    }

    /// Returns controls added by the MCGate wrapper.
    pub const fn added_controls(&self) -> &'a [Qubit] {
        self.added_controls
    }

    /// Returns controls inherent to the base standard gate.
    pub const fn inherent_controls(&self) -> &'a [Qubit] {
        self.inherent_controls
    }

    /// Returns target operands of the base standard gate.
    pub const fn targets(&self) -> &'a [Qubit] {
        self.targets
    }

    /// Returns added controls followed by inherent controls.
    pub fn all_controls(&self) -> &[Qubit] {
        &self.all_controls
    }

    /// Returns the total number of added and inherent controls.
    pub fn total_control_count(&self) -> usize {
        self.all_controls.len()
    }

    fn parameter_mode(&self) -> &'static str {
        if self.has_symbolic_params {
            "symbolic"
        } else {
            "fixed"
        }
    }
}

/// Decomposes a single operation when it contains an MCGate instruction.
///
/// This adapter is intentionally operation-local so the eventual transformer can
/// own circuit traversal, parameter-table interning, control-flow recursion, and
/// `TransformOutcome` aggregation.
pub fn decompose_mc_gate_operation(
    operation: &Operation,
    config: &McGateDecomposeConfig,
) -> Result<McGateOperationDecomposeResult, CompilerError> {
    let Instruction::McGate(gate) = &operation.instruction else {
        return Ok(McGateOperationDecomposeResult::unchanged(operation.clone()));
    };

    if operation.label.is_some() && config.skips_labeled_ops() {
        return Ok(McGateOperationDecomposeResult::unchanged(operation.clone()));
    }

    let operations = decompose_mc_gate(gate, &operation.qubits, &operation.params, config)?;
    Ok(McGateOperationDecomposeResult::changed(operations))
}

/// Decomposes an MCGate into project-supported standard operations.
///
/// This entry validates the operation, builds the operand view, classifies the
/// family, and dispatches supported families to their local lowering modules.
pub fn decompose_mc_gate(
    gate: &MCGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<Vec<Operation>, CompilerError> {
    let view = McGateOperandView::new(gate, qubits, params, config)?;
    let family = view.family();

    match family {
        McGateFamily::Identity => Ok(Vec::new()),
        McGateFamily::Pauli => super::pauli::decompose_pauli_family(&view, config),
        McGateFamily::Phase => super::phase::decompose_phase_family(&view, params, config),
        McGateFamily::Rotation => match view.base_gate() {
            StandardGate::RZ | StandardGate::CRZ => {
                super::rz::decompose_rz_family(&view, params, config)
            }
            StandardGate::RX | StandardGate::RY | StandardGate::CRX | StandardGate::CRY => {
                super::rx_ry::decompose_rx_ry_family(&view, params, config)
            }
            base_gate => Err(mc_gate_error(
                gate,
                params,
                config,
                format!("rotation-family gate {base_gate} is not implemented yet"),
            )),
        },
        McGateFamily::OneQubit => {
            super::one_qubit::decompose_one_qubit_family(&view, params, config)
        }
        McGateFamily::Swap => super::swap::decompose_swap_family(&view, config),
        McGateFamily::Fsim => super::fsim::decompose_fsim_family(&view, params, config),
        McGateFamily::PauliInteraction => {
            super::pauli_interaction::decompose_pauli_interaction_family(&view, params, config)
        }
        _ => Err(mc_gate_error(
            gate,
            params,
            config,
            format!("MCGate family {family} is not implemented yet"),
        )),
    }
}

fn validate_operand_arity(
    gate: &MCGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    let expected_qubits = gate.num_qubits();
    if qubits.len() != expected_qubits {
        return Err(mc_gate_error(
            gate,
            params,
            config,
            format!(
                "operation qubit count mismatch: expected {expected_qubits}, got {}",
                qubits.len()
            ),
        ));
    }

    let expected_params = gate.num_params();
    if params.len() != expected_params {
        return Err(mc_gate_error(
            gate,
            params,
            config,
            format!(
                "operation parameter count mismatch: expected {expected_params}, got {}",
                params.len()
            ),
        ));
    }

    Ok(())
}

fn validate_distinct_operands(
    gate: &MCGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    let mut seen = HashSet::with_capacity(qubits.len());
    for qubit in qubits.iter().copied() {
        if !seen.insert(qubit) {
            return Err(mc_gate_error(
                gate,
                params,
                config,
                format!("operation operands must be distinct; duplicate {qubit}"),
            ));
        }
    }

    Ok(())
}

fn validate_configured_ancillas(
    gate: &MCGate,
    qubits: &[Qubit],
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    match config.ancilla_mode {
        AncillaMode::NoAncilla => Ok(()),
        AncillaMode::CleanAncilla => validate_ancillas(
            "clean ancillas",
            qubits,
            &config.clean_ancillas,
            gate,
            params,
            config,
        ),
        AncillaMode::DirtyAncilla => validate_ancillas(
            "dirty ancillas",
            qubits,
            &config.dirty_ancillas,
            gate,
            params,
            config,
        ),
    }
}

fn validate_ancillas(
    label: &str,
    operands: &[Qubit],
    ancillas: &[Qubit],
    gate: &MCGate,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
) -> Result<(), CompilerError> {
    let operands: HashSet<_> = operands.iter().copied().collect();
    let mut seen = HashSet::with_capacity(ancillas.len());

    for ancilla in ancillas.iter().copied() {
        if operands.contains(&ancilla) {
            return Err(mc_gate_error(
                gate,
                params,
                config,
                format!("{label} must not overlap operation operands; duplicate {ancilla}"),
            ));
        }

        if !seen.insert(ancilla) {
            return Err(mc_gate_error(
                gate,
                params,
                config,
                format!("{label} must be distinct; duplicate {ancilla}"),
            ));
        }
    }

    Ok(())
}

fn mc_gate_error(
    gate: &MCGate,
    params: &[CircuitParam],
    config: &McGateDecomposeConfig,
    detail: String,
) -> CompilerError {
    let base_gate = *gate.base_gate();
    let added_control_count = gate.num_qubits().saturating_sub(base_gate.num_qubits());
    let total_control_count = gate.num_ctrl_qubits();
    let parameter_mode = if params
        .iter()
        .any(|param| matches!(param, CircuitParam::Index(_)))
    {
        "symbolic"
    } else {
        "fixed"
    };

    CompilerError::TransformFailed {
        name: DECOMPOSE_MC_GATE_NAME,
        reason: format!(
            "{detail}; base_gate={base_gate}, added_control_count={added_control_count}, control_count={total_control_count}, parameter_mode={parameter_mode}, strategy={}",
            config.ancilla_mode
        ),
    }
}

/// Builds a transform error after an operand view has already been validated.
pub(super) fn mc_gate_view_error(
    view: &McGateOperandView<'_>,
    config: &McGateDecomposeConfig,
    detail: String,
) -> CompilerError {
    CompilerError::TransformFailed {
        name: DECOMPOSE_MC_GATE_NAME,
        reason: format!(
            "{detail}; base_gate={}, added_control_count={}, control_count={}, parameter_mode={}, strategy={}",
            view.base_gate(),
            view.added_controls().len(),
            view.total_control_count(),
            view.parameter_mode(),
            config.ancilla_mode
        ),
    }
}
