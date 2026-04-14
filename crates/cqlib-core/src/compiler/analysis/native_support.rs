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

//! Target native-support analysis.
//!
//! This module classifies each operation against a target device (and optional
//! layout) to answer whether it is natively executable, requires additional
//! lowering, or is unsupported under the current mapping conditions.

use crate::circuit::{Circuit, CircuitParam, ControlFlow, Instruction, Qubit};
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::device::{Device, Layout};

/// Native support status for one circuit operation on the current target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSupportStatus {
    /// Operation is natively executable under the current target and mapping.
    Supported,
    /// A concrete layout is required before support can be decided.
    RequiresLayout,
    /// Instruction family is not listed as native on the target.
    UnsupportedInstruction,
    /// Instruction is native in principle but not on these mapped qubits.
    UnsupportedOnQubits,
    /// Instruction is available only in the reverse coupling direction.
    DirectionMismatch,
    /// Control-flow operations are not natively supported in this path.
    UnsupportedControlFlow,
    /// Operation uses symbolic parameters not accepted in this native path.
    UnsupportedSymbolicParameters,
    /// Operation is expected to become executable only after lowering.
    NeedsLowering,
}

/// Native support diagnostics for one operation in circuit order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSupportOpEntry {
    /// Operation index in circuit order.
    pub op_index: usize,
    /// Native support classification for this operation.
    pub status: NativeSupportStatus,
    /// Logical-to-physical mapped qubits used for this classification.
    pub mapped_qubits: Option<Vec<Qubit>>,
}

/// Target-native support diagnostics for the current circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeSupportAnalysis {
    entries: Vec<NativeSupportOpEntry>,
    supported_ops: usize,
    unsupported_ops: usize,
    requires_layout_ops: usize,
    direction_mismatch_ops: usize,
    requires_lowering_ops: usize,
}

impl NativeSupportAnalysis {
    /// Builds native-support diagnostics from a compiler context.
    ///
    /// Returns [`CompilerError::MissingDevice`] when the context has no target.
    pub fn from_context(ctx: &CompilerContext) -> Result<Self, CompilerError> {
        let device = ctx.device().ok_or(CompilerError::MissingDevice)?;
        Ok(Self::from_circuit_and_device(
            ctx.circuit(),
            device,
            ctx.layout(),
        ))
    }

    /// Builds native-support diagnostics from explicit circuit/target inputs.
    pub fn from_circuit_and_device(
        circuit: &Circuit,
        device: &Device,
        layout: Option<&Layout>,
    ) -> Self {
        let mut analysis = Self::default();

        for (op_index, operation) in circuit.operations().iter().enumerate() {
            let mapped_qubits = layout.and_then(|layout| {
                operation
                    .qubits
                    .iter()
                    .map(|&logical| layout.get_physical(logical))
                    .collect::<Option<Vec<_>>>()
            });
            let status = classify_operation(
                &operation.instruction,
                &operation.params,
                operation.qubits.len(),
                mapped_qubits.as_deref(),
                device,
            );

            match status {
                NativeSupportStatus::Supported => analysis.supported_ops += 1,
                NativeSupportStatus::RequiresLayout => analysis.requires_layout_ops += 1,
                NativeSupportStatus::DirectionMismatch => {
                    analysis.direction_mismatch_ops += 1;
                    analysis.unsupported_ops += 1;
                }
                NativeSupportStatus::NeedsLowering => {
                    analysis.requires_lowering_ops += 1;
                    analysis.unsupported_ops += 1;
                }
                NativeSupportStatus::UnsupportedInstruction
                | NativeSupportStatus::UnsupportedOnQubits
                | NativeSupportStatus::UnsupportedControlFlow
                | NativeSupportStatus::UnsupportedSymbolicParameters => {
                    analysis.unsupported_ops += 1;
                }
            }

            analysis.entries.push(NativeSupportOpEntry {
                op_index,
                status,
                mapped_qubits,
            });
        }

        analysis
    }

    /// Number of operations currently classified as natively supported.
    pub fn supported_ops(&self) -> usize {
        self.supported_ops
    }

    /// Number of operations currently classified as unsupported.
    pub fn unsupported_ops(&self) -> usize {
        self.unsupported_ops
    }

    /// Number of operations that cannot be checked without layout.
    pub fn requires_layout_ops(&self) -> usize {
        self.requires_layout_ops
    }

    /// Number of operations that fail due to coupling direction mismatch.
    pub fn direction_mismatch_ops(&self) -> usize {
        self.direction_mismatch_ops
    }

    /// Number of operations that are valid only after lowering/decomposition.
    pub fn requires_lowering_ops(&self) -> usize {
        self.requires_lowering_ops
    }

    /// Returns true when every operation is supported with no missing-layout gaps.
    pub fn fully_supported(&self) -> bool {
        self.unsupported_ops == 0 && self.requires_layout_ops == 0
    }

    /// Returns one operation-level diagnostic by operation index.
    pub fn get(&self, op_index: usize) -> Option<&NativeSupportOpEntry> {
        self.entries.get(op_index)
    }

    /// Returns operation-level diagnostics in circuit order.
    pub fn entries(&self) -> impl Iterator<Item = &NativeSupportOpEntry> {
        self.entries.iter()
    }
}

fn classify_operation(
    instruction: &Instruction,
    params: &[CircuitParam],
    arity: usize,
    mapped_qubits: Option<&[Qubit]>,
    device: &Device,
) -> NativeSupportStatus {
    if params
        .iter()
        .any(|param| matches!(param, CircuitParam::Index(_)))
    {
        return NativeSupportStatus::UnsupportedSymbolicParameters;
    }

    match instruction {
        Instruction::ControlFlowGate(_) => NativeSupportStatus::UnsupportedControlFlow,
        Instruction::McGate(_) | Instruction::UnitaryGate(_) | Instruction::CircuitGate(_) => {
            NativeSupportStatus::NeedsLowering
        }
        Instruction::Standard(_) if arity > 2 => NativeSupportStatus::NeedsLowering,
        Instruction::Delay | Instruction::Directive(_) | Instruction::Standard(_) => {
            classify_native_like_operation(instruction, mapped_qubits, device)
        }
    }
}

fn classify_native_like_operation(
    instruction: &Instruction,
    mapped_qubits: Option<&[Qubit]>,
    device: &Device,
) -> NativeSupportStatus {
    let mapped_qubits = match mapped_qubits {
        Some(qubits) => qubits,
        None => return NativeSupportStatus::RequiresLayout,
    };

    match mapped_qubits.len() {
        0 => {
            if device
                .native_gates()
                .iter()
                .any(|native| instruction_matches(native, instruction))
            {
                NativeSupportStatus::Supported
            } else {
                NativeSupportStatus::UnsupportedInstruction
            }
        }
        1 => {
            let physical = mapped_qubits[0];
            if let Some(props) = device.qubit_properties(physical) {
                if props
                    .native_instructions()
                    .iter()
                    .any(|prop| instruction_matches(prop.instruction(), instruction))
                {
                    return NativeSupportStatus::Supported;
                }
            }

            if device
                .native_gates()
                .iter()
                .any(|native| instruction_matches(native, instruction))
            {
                NativeSupportStatus::Supported
            } else {
                NativeSupportStatus::UnsupportedInstruction
            }
        }
        2 => {
            let control = mapped_qubits[0];
            let target = mapped_qubits[1];

            let supported_here = device
                .edge_properties(control, target)
                .map(|props| {
                    props
                        .native_instructions()
                        .iter()
                        .any(|prop| instruction_matches(prop.instruction(), instruction))
                })
                .unwrap_or(false);

            if supported_here {
                return NativeSupportStatus::Supported;
            }

            let supported_by_device = device
                .native_gates()
                .iter()
                .any(|native| instruction_matches(native, instruction));
            if supported_by_device && device.topology().is_connected(control, target) {
                return NativeSupportStatus::Supported;
            }

            let reverse_support = device
                .edge_properties(target, control)
                .map(|props| {
                    props
                        .native_instructions()
                        .iter()
                        .any(|prop| instruction_matches(prop.instruction(), instruction))
                })
                .unwrap_or(false)
                || (supported_by_device && device.topology().is_connected(target, control));

            if reverse_support {
                NativeSupportStatus::DirectionMismatch
            } else if device.topology().is_connected(control, target)
                || device.topology().is_connected(target, control)
            {
                NativeSupportStatus::UnsupportedInstruction
            } else {
                NativeSupportStatus::UnsupportedOnQubits
            }
        }
        _ => NativeSupportStatus::NeedsLowering,
    }
}

pub(crate) fn instruction_matches(lhs: &Instruction, rhs: &Instruction) -> bool {
    match (lhs, rhs) {
        (Instruction::Standard(a), Instruction::Standard(b)) => a == b,
        (Instruction::Directive(a), Instruction::Directive(b)) => a == b,
        (Instruction::Delay, Instruction::Delay) => true,
        (
            Instruction::ControlFlowGate(ControlFlow::IfElse(_)),
            Instruction::ControlFlowGate(ControlFlow::IfElse(_)),
        ) => true,
        (
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(_)),
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(_)),
        ) => true,
        (Instruction::McGate(a), Instruction::McGate(b)) => {
            a.num_ctrl_qubits() == b.num_ctrl_qubits() && a.base_gate() == b.base_gate()
        }
        (Instruction::UnitaryGate(a), Instruction::UnitaryGate(b)) => a.label() == b.label(),
        (Instruction::CircuitGate(a), Instruction::CircuitGate(b)) => a.name == b.name,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeSupportAnalysis, NativeSupportStatus};
    use crate::circuit::{Circuit, Qubit, StandardGate};
    use crate::compiler::CompilerContext;
    use crate::compiler::CompilerError;
    use crate::device::{Device, EdgeProp, InstructionProp, Layout, QubitProp, Topology};
    use std::collections::{HashMap, HashSet};

    fn device_with_native_support() -> Device {
        let physical = vec![Qubit::new(10), Qubit::new(11)];
        let topology = Topology::new(
            physical.clone(),
            vec![(Qubit::new(10), Qubit::new(11), "cx".to_string())],
        )
        .unwrap();
        let mut device = Device::new("mock-qpu", HashSet::from_iter(physical.clone()), topology)
            .unwrap()
            .with_native_gates(vec![StandardGate::H.into(), StandardGate::CX.into()]);
        device
            .add_qubit_properties(
                Qubit::new(10),
                QubitProp::new(0.01)
                    .with_native_instruction(InstructionProp::new(StandardGate::H.into(), 0.01)),
            )
            .unwrap();
        device
            .add_qubit_properties(
                Qubit::new(11),
                QubitProp::new(0.01)
                    .with_native_instruction(InstructionProp::new(StandardGate::H.into(), 0.01)),
            )
            .unwrap();
        device
            .add_edge_properties(
                Qubit::new(10),
                Qubit::new(11),
                EdgeProp::new()
                    .with_native_instruction(InstructionProp::new(StandardGate::CX.into(), 0.02)),
            )
            .unwrap();
        device
    }

    #[test]
    fn native_support_requires_device_in_context() {
        let ctx = CompilerContext::new(Circuit::new(1));
        let err = NativeSupportAnalysis::from_context(&ctx).unwrap_err();
        assert!(matches!(err, CompilerError::MissingDevice));
    }

    #[test]
    fn native_support_marks_two_qubit_ops_as_needing_layout_when_missing() {
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let analysis = NativeSupportAnalysis::from_circuit_and_device(
            &circuit,
            &device_with_native_support(),
            None,
        );

        assert_eq!(
            analysis.get(0).unwrap().status,
            NativeSupportStatus::RequiresLayout
        );
        assert_eq!(analysis.requires_layout_ops(), 1);
    }

    #[test]
    fn native_support_marks_supported_and_direction_mismatch_cases() {
        let mut circuit = Circuit::new(2);
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.cx(Qubit::new(1), Qubit::new(0)).unwrap();

        let layout = Layout::new(
            vec![Qubit::new(0), Qubit::new(1)],
            vec![Qubit::new(10), Qubit::new(11)],
            Some(HashMap::from([
                (Qubit::new(0), Qubit::new(10)),
                (Qubit::new(1), Qubit::new(11)),
            ])),
        )
        .unwrap();

        let analysis = NativeSupportAnalysis::from_circuit_and_device(
            &circuit,
            &device_with_native_support(),
            Some(&layout),
        );

        assert_eq!(
            analysis.get(0).unwrap().status,
            NativeSupportStatus::Supported
        );
        assert_eq!(
            analysis.get(1).unwrap().status,
            NativeSupportStatus::DirectionMismatch
        );
        assert_eq!(analysis.supported_ops(), 1);
        assert_eq!(analysis.direction_mismatch_ops(), 1);
    }

    #[test]
    fn native_support_marks_mc_gate_as_needing_lowering() {
        let mut circuit = Circuit::new(3);
        circuit
            .multi_control(
                StandardGate::X,
                vec![Qubit::new(0), Qubit::new(1)],
                vec![Qubit::new(2)],
                [],
            )
            .unwrap();

        let analysis = NativeSupportAnalysis::from_circuit_and_device(
            &circuit,
            &device_with_native_support(),
            None,
        );

        assert_eq!(
            analysis.get(0).unwrap().status,
            NativeSupportStatus::NeedsLowering
        );
        assert_eq!(analysis.requires_lowering_ops(), 1);
    }
}
