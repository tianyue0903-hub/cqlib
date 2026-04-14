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

//! Unified compiler cost analysis.
//!
//! This module provides a shared cost surface used by workflows and transforms:
//! a device-agnostic logical estimate is always available, while an optional
//! target-aware estimate is produced when device information exists.

use crate::circuit::{Circuit, Instruction, Operation, Qubit};
use crate::compiler::analysis::native_support::{
    NativeSupportAnalysis, NativeSupportStatus, instruction_matches,
};
use crate::compiler::context::CompilerContext;
use crate::compiler::error::CompilerError;
use crate::device::{Device, Layout};
use std::collections::BTreeMap;

/// Device-agnostic logical cost estimates for the current circuit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LogicalCost {
    /// Total number of operations in the circuit.
    pub total_ops: usize,
    /// Number of operations acting on one qubit.
    pub single_qubit_ops: usize,
    /// Number of operations acting on two qubits.
    pub two_qubit_ops: usize,
    /// Number of operations acting on three or more qubits.
    pub multi_qubit_ops: usize,
    /// Number of operations carrying one or more parameters.
    pub parameterized_ops: usize,
    /// Number of explicit control-flow operations.
    pub control_flow_ops: usize,
    /// Conservative logical layer estimate across all qubits.
    pub depth_estimate: usize,
    /// Conservative logical layer estimate considering only 2-qubit operations.
    pub two_qubit_depth_estimate: usize,
}

/// Optional device-aware cost estimates for the current circuit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PhysicalCostEstimate {
    /// Conservative serial duration sum over supported operations.
    pub estimated_duration: Option<f64>,
    /// Aggregated error-rate estimate over supported operations.
    pub estimated_error: Option<f64>,
    /// Number of supported operations with known duration values.
    pub ops_with_duration: usize,
    /// Number of supported operations with known error-rate values.
    pub ops_with_error_rate: usize,
    /// Number of supported operations missing duration metadata.
    pub missing_duration_ops: usize,
    /// Number of supported operations missing error-rate metadata.
    pub missing_error_ops: usize,
}

/// Unified logical and optional target-aware cost estimates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CostAnalysis {
    /// Device-agnostic logical cost view.
    pub logical: LogicalCost,
    /// Optional target-aware cost view when target data is available.
    pub physical: Option<PhysicalCostEstimate>,
}

impl CostAnalysis {
    /// Builds cost estimates from a compiler context.
    ///
    /// Logical estimates are always produced. Physical estimates are present only
    /// when a target device exists in the context.
    pub fn from_context(ctx: &CompilerContext) -> Result<Self, CompilerError> {
        let logical = LogicalCost::from_circuit(ctx.circuit());
        let physical = ctx.device().map(|device| {
            PhysicalCostEstimate::from_circuit_and_target(ctx.circuit(), device, ctx.layout())
        });

        Ok(Self { logical, physical })
    }
}

impl LogicalCost {
    /// Builds logical costs by scanning the circuit once.
    ///
    /// `depth_estimate` and `two_qubit_depth_estimate` are conservative layer
    /// estimates, not schedule-accurate execution depths.
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut cost = Self::default();
        let mut qubit_depths: BTreeMap<Qubit, usize> = BTreeMap::new();
        let mut two_qubit_depths: BTreeMap<Qubit, usize> = BTreeMap::new();

        for operation in circuit.operations() {
            cost.total_ops += 1;
            match operation.qubits.len() {
                0 => {}
                1 => cost.single_qubit_ops += 1,
                2 => cost.two_qubit_ops += 1,
                _ => cost.multi_qubit_ops += 1,
            }
            if !operation.params.is_empty() {
                cost.parameterized_ops += 1;
            }
            if matches!(operation.instruction, Instruction::ControlFlowGate(_)) {
                cost.control_flow_ops += 1;
            }

            let unique_qubits = unique_qubits(operation);
            if !unique_qubits.is_empty() {
                let next_depth = unique_qubits
                    .iter()
                    .filter_map(|qubit| qubit_depths.get(qubit))
                    .max()
                    .copied()
                    .unwrap_or(0)
                    + 1;
                for qubit in &unique_qubits {
                    qubit_depths.insert(*qubit, next_depth);
                }
                cost.depth_estimate = cost.depth_estimate.max(next_depth);
            }

            if operation.qubits.len() == 2
                && !matches!(
                    operation.instruction,
                    Instruction::Directive(_)
                        | Instruction::Delay
                        | Instruction::ControlFlowGate(_)
                )
            {
                let next_two_qubit_depth = unique_qubits
                    .iter()
                    .filter_map(|qubit| two_qubit_depths.get(qubit))
                    .max()
                    .copied()
                    .unwrap_or(0)
                    + 1;
                for qubit in &unique_qubits {
                    two_qubit_depths.insert(*qubit, next_two_qubit_depth);
                }
                cost.two_qubit_depth_estimate =
                    cost.two_qubit_depth_estimate.max(next_two_qubit_depth);
            }
        }

        cost
    }
}

impl PhysicalCostEstimate {
    /// Builds target-aware duration/error estimates using native-support results.
    ///
    /// The duration estimate is a serial sum over supported operations and should
    /// be interpreted as a conservative upper estimate, not a scheduled runtime.
    pub fn from_circuit_and_target(
        circuit: &Circuit,
        device: &Device,
        layout: Option<&Layout>,
    ) -> Self {
        let native = NativeSupportAnalysis::from_circuit_and_device(circuit, device, layout);
        let mut total_duration = 0.0;
        let mut total_error = 0.0;
        let mut saw_duration = false;
        let mut saw_error = false;
        let mut estimate = Self::default();

        for (op_index, operation) in circuit.operations().iter().enumerate() {
            let entry = native
                .get(op_index)
                .expect("native support entry must exist for each operation");
            if entry.status != NativeSupportStatus::Supported {
                continue;
            }

            if let Some(mapped_qubits) = entry.mapped_qubits.as_deref() {
                let duration = estimate_duration(operation, mapped_qubits, device);
                match duration {
                    Some(value) => {
                        total_duration += value;
                        saw_duration = true;
                        estimate.ops_with_duration += 1;
                    }
                    None => estimate.missing_duration_ops += 1,
                }

                let error = estimate_error(operation, mapped_qubits, device);
                match error {
                    Some(value) => {
                        total_error += value;
                        saw_error = true;
                        estimate.ops_with_error_rate += 1;
                    }
                    None => estimate.missing_error_ops += 1,
                }
            }
        }

        estimate.estimated_duration = saw_duration.then_some(total_duration);
        estimate.estimated_error = saw_error.then_some(total_error);
        estimate
    }
}

fn unique_qubits(operation: &Operation) -> Vec<Qubit> {
    let mut unique = Vec::with_capacity(operation.qubits.len());
    for &qubit in &operation.qubits {
        if !unique.contains(&qubit) {
            unique.push(qubit);
        }
    }
    unique
}

fn estimate_duration(
    operation: &Operation,
    mapped_qubits: &[Qubit],
    device: &Device,
) -> Option<f64> {
    match mapped_qubits.len() {
        1 => device
            .qubit_properties(mapped_qubits[0])?
            .native_instructions()
            .iter()
            .find(|prop| instruction_matches(prop.instruction(), &operation.instruction))
            .and_then(|prop| prop.length()),
        2 => device
            .edge_properties(mapped_qubits[0], mapped_qubits[1])?
            .native_instructions()
            .iter()
            .find(|prop| instruction_matches(prop.instruction(), &operation.instruction))
            .and_then(|prop| prop.length()),
        _ => None,
    }
}

fn estimate_error(operation: &Operation, mapped_qubits: &[Qubit], device: &Device) -> Option<f64> {
    match mapped_qubits.len() {
        1 => device
            .qubit_properties(mapped_qubits[0])
            .and_then(|prop| {
                prop.native_instructions()
                    .iter()
                    .find(|instruction| {
                        instruction_matches(instruction.instruction(), &operation.instruction)
                    })
                    .map(|instruction| instruction.error_rate())
            })
            .or(device.default_single_qubit_error()),
        2 => device
            .edge_properties(mapped_qubits[0], mapped_qubits[1])
            .and_then(|prop| {
                prop.native_instructions()
                    .iter()
                    .find(|instruction| {
                        instruction_matches(instruction.instruction(), &operation.instruction)
                    })
                    .map(|instruction| instruction.error_rate())
            })
            .or(device.default_two_qubit_error()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::CostAnalysis;
    use crate::circuit::{Circuit, Qubit, StandardGate};
    use crate::compiler::CompilerContext;
    use crate::device::{Device, EdgeProp, InstructionProp, Layout, QubitProp, Topology};
    use std::collections::{HashMap, HashSet};

    fn device_with_props() -> (Device, Layout) {
        let physical = vec![Qubit::new(10), Qubit::new(11)];
        let topology = Topology::new(
            physical.clone(),
            vec![(Qubit::new(10), Qubit::new(11), "cx".to_string())],
        )
        .unwrap();
        let mut device = Device::new("mock-qpu", HashSet::from_iter(physical.clone()), topology)
            .unwrap()
            .with_native_gates(vec![StandardGate::H.into(), StandardGate::CX.into()])
            .with_default_single_qubit_error(0.01)
            .with_default_two_qubit_error(0.02);
        device
            .add_qubit_properties(
                Qubit::new(10),
                QubitProp::new(0.01).with_native_instruction(
                    InstructionProp::new(StandardGate::H.into(), 0.01).with_length(10.0),
                ),
            )
            .unwrap();
        device
            .add_qubit_properties(
                Qubit::new(11),
                QubitProp::new(0.01).with_native_instruction(
                    InstructionProp::new(StandardGate::H.into(), 0.01).with_length(10.0),
                ),
            )
            .unwrap();
        device
            .add_edge_properties(
                Qubit::new(10),
                Qubit::new(11),
                EdgeProp::new().with_native_instruction(
                    InstructionProp::new(StandardGate::CX.into(), 0.02).with_length(200.0),
                ),
            )
            .unwrap();
        let layout = Layout::new(
            vec![Qubit::new(0), Qubit::new(1)],
            physical,
            Some(HashMap::from([
                (Qubit::new(0), Qubit::new(10)),
                (Qubit::new(1), Qubit::new(11)),
            ])),
        )
        .unwrap();

        (device, layout)
    }

    #[test]
    fn cost_analysis_without_device_only_reports_logical_costs() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let ctx = CompilerContext::new(circuit);
        let cost = CostAnalysis::from_context(&ctx).unwrap();

        assert_eq!(cost.logical.total_ops, 2);
        assert_eq!(cost.logical.single_qubit_ops, 1);
        assert_eq!(cost.logical.two_qubit_ops, 1);
        assert!(cost.physical.is_none());
    }

    #[test]
    fn cost_analysis_with_device_reports_duration_and_error_estimates() {
        let (device, layout) = device_with_props();
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

        let mut ctx = CompilerContext::with_device(circuit, device);
        ctx.set_layout(layout);

        let cost = CostAnalysis::from_context(&ctx).unwrap();
        let physical = cost.physical.unwrap();

        assert_eq!(cost.logical.depth_estimate, 2);
        assert_eq!(cost.logical.two_qubit_depth_estimate, 1);
        assert_eq!(physical.estimated_duration, Some(210.0));
        assert_eq!(physical.estimated_error, Some(0.03));
        assert_eq!(physical.ops_with_duration, 2);
        assert_eq!(physical.ops_with_error_rate, 2);
    }
}
