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

//! Basis usage analysis.
//!
//! This module provides a normalized view of which instruction families appear
//! in the current circuit. It intentionally tracks instruction *shape* and
//! usage metadata (occurrences, arity, parameterization) rather than pass-level
//! transformation policy.

use crate::circuit::{Circuit, ControlFlow, Directive, Instruction, StandardGate};
use crate::compiler::{CompilerContext, CompilerError};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Typed identifier for analyses that can be materialized by a
/// [`crate::compiler::CompilerContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalysisKey {
    /// Control-flow graph view of the current circuit.
    CircuitCfg,
    /// Circuit-wide instruction category counts.
    InstructionStats,
    /// Per-qubit usage spans and participation categories.
    QubitUsage,
    /// Block-level control-flow graph summaries.
    BlockSummary,
    /// Logical coupling requirements derived from CFG blocks.
    CouplingRequirements,
    /// Normalized instruction-family usage over the circuit.
    BasisAnalysis,
    /// Target-native support diagnostics for each operation.
    NativeSupportAnalysis,
    /// Unified logical and optional target-aware cost estimates.
    CostAnalysis,
}

/// Contract implemented by analyses that can be lazily built and cached
/// by [`crate::compiler::CompilerContext`].
///
/// This trait keeps construction policy next to each analysis type while
/// allowing the context to share one generic cache/materialization path.
pub trait ContextAnalysis: Sized + 'static {
    /// Stable typed key used by descriptors and prerequisite checks.
    const KEY: AnalysisKey;

    /// Builds the analysis from the current compiler context.
    fn build(ctx: &mut CompilerContext) -> Result<Self, CompilerError>;
}

/// Stable normalized key for one instruction family used by the circuit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BasisKey {
    /// One specific standard gate.
    Standard(StandardGate),
    /// One specific non-unitary directive.
    Directive(Directive),
    /// Delay instruction family.
    Delay,
    /// `if-else` control-flow family.
    ControlFlowIfElse,
    /// `while-loop` control-flow family.
    ControlFlowWhileLoop,
    /// Multi-controlled gate family.
    McGate,
    /// Custom unitary gate family.
    UnitaryGate,
    /// Circuit-as-gate family.
    CircuitGate,
}

impl BasisKey {
    fn rank(&self) -> u8 {
        match self {
            BasisKey::Standard(_) => 0,
            BasisKey::Directive(_) => 1,
            BasisKey::Delay => 2,
            BasisKey::ControlFlowIfElse => 3,
            BasisKey::ControlFlowWhileLoop => 4,
            BasisKey::McGate => 5,
            BasisKey::UnitaryGate => 6,
            BasisKey::CircuitGate => 7,
        }
    }

    fn stable_name(&self) -> String {
        match self {
            BasisKey::Standard(gate) => gate.to_string(),
            BasisKey::Directive(directive) => directive.to_string(),
            BasisKey::Delay => "delay".to_string(),
            BasisKey::ControlFlowIfElse => "if_else".to_string(),
            BasisKey::ControlFlowWhileLoop => "while_loop".to_string(),
            BasisKey::McGate => "mc_gate".to_string(),
            BasisKey::UnitaryGate => "unitary_gate".to_string(),
            BasisKey::CircuitGate => "circuit_gate".to_string(),
        }
    }

    fn from_instruction(instruction: &Instruction) -> Self {
        match instruction {
            Instruction::Standard(gate) => Self::Standard(*gate),
            Instruction::Directive(directive) => Self::Directive(*directive),
            Instruction::Delay => Self::Delay,
            Instruction::ControlFlowGate(control_flow) => match control_flow {
                ControlFlow::IfElse(_) => Self::ControlFlowIfElse,
                ControlFlow::WhileLoop(_) => Self::ControlFlowWhileLoop,
            },
            Instruction::McGate(_) => Self::McGate,
            Instruction::UnitaryGate(_) => Self::UnitaryGate,
            Instruction::CircuitGate(_) => Self::CircuitGate,
        }
    }
}

impl PartialOrd for BasisKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BasisKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank()
            .cmp(&other.rank())
            .then_with(|| self.stable_name().cmp(&other.stable_name()))
    }
}

/// Aggregated usage of one normalized instruction family.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BasisEntry {
    /// Total number of operations mapped to this basis key.
    pub occurrences: usize,
    /// First operation index where this key appears.
    pub first_op_index: Option<usize>,
    /// Last operation index where this key appears.
    pub last_op_index: Option<usize>,
    /// Distinct arities observed for this key.
    pub arities_seen: BTreeSet<usize>,
    /// Whether any instance under this key carries parameters.
    pub is_parameterized: bool,
    /// Whether this key represents non-unitary behavior.
    pub is_non_unitary: bool,
    /// Whether this key represents control-flow behavior.
    pub is_control_flow: bool,
}

impl BasisEntry {
    fn record(
        &mut self,
        op_index: usize,
        arity: usize,
        instruction: &Instruction,
        is_parameterized: bool,
    ) {
        self.occurrences += 1;
        self.first_op_index = Some(
            self.first_op_index
                .map_or(op_index, |first| first.min(op_index)),
        );
        self.last_op_index = Some(
            self.last_op_index
                .map_or(op_index, |last| last.max(op_index)),
        );
        self.arities_seen.insert(arity);
        self.is_parameterized |= is_parameterized;
        self.is_control_flow |= matches!(instruction, Instruction::ControlFlowGate(_));
        self.is_non_unitary |= matches!(
            instruction,
            Instruction::Directive(_) | Instruction::Delay | Instruction::ControlFlowGate(_)
        );
    }
}

/// Normalized instruction-family usage extracted from the current circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BasisAnalysis {
    entries: BTreeMap<BasisKey, BasisEntry>,
    num_distinct_instructions: usize,
    has_control_flow: bool,
    has_non_unitary_ops: bool,
    has_parameterized_ops: bool,
}

impl BasisAnalysis {
    /// Builds basis usage information by scanning the circuit once.
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut analysis = Self::default();

        for (op_index, operation) in circuit.operations().iter().enumerate() {
            let key = BasisKey::from_instruction(&operation.instruction);
            let entry = analysis.entries.entry(key).or_default();
            let is_parameterized = !operation.params.is_empty();

            entry.record(
                op_index,
                operation.qubits.len(),
                &operation.instruction,
                is_parameterized,
            );

            analysis.has_control_flow |= entry.is_control_flow;
            analysis.has_non_unitary_ops |= entry.is_non_unitary;
            analysis.has_parameterized_ops |= is_parameterized;
        }

        analysis.num_distinct_instructions = analysis.entries.len();
        analysis
    }

    pub fn num_distinct_instructions(&self) -> usize {
        self.num_distinct_instructions
    }

    /// Returns whether any control-flow instruction appears in the circuit.
    pub fn has_control_flow(&self) -> bool {
        self.has_control_flow
    }

    /// Returns whether any non-unitary instruction appears in the circuit.
    pub fn has_non_unitary_ops(&self) -> bool {
        self.has_non_unitary_ops
    }

    /// Returns whether any operation carries parameters.
    pub fn has_parameterized_ops(&self) -> bool {
        self.has_parameterized_ops
    }

    /// Returns aggregated usage for one normalized basis key.
    pub fn get(&self, key: &BasisKey) -> Option<&BasisEntry> {
        self.entries.get(key)
    }

    /// Returns all basis entries in a stable key order.
    pub fn entries(&self) -> impl Iterator<Item = (&BasisKey, &BasisEntry)> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{BasisAnalysis, BasisKey};
    use crate::circuit::{Circuit, ConditionView, Operation, Parameter, Qubit, StandardGate};
    use smallvec::smallvec;

    #[test]
    fn basis_analysis_tracks_distinct_instruction_families() {
        let mut circuit = Circuit::new(2);
        circuit.h(Qubit::new(0)).unwrap();
        circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
        circuit.measure(Qubit::new(1)).unwrap();

        let basis = BasisAnalysis::from_circuit(&circuit);

        assert_eq!(basis.num_distinct_instructions(), 3);
        assert!(!basis.has_control_flow());
        assert!(basis.has_non_unitary_ops());
        assert!(!basis.has_parameterized_ops());

        assert_eq!(
            basis
                .get(&BasisKey::Standard(StandardGate::H))
                .unwrap()
                .occurrences,
            1
        );
        assert_eq!(
            basis
                .get(&BasisKey::Standard(StandardGate::CX))
                .unwrap()
                .occurrences,
            1
        );
        assert_eq!(
            basis
                .get(&BasisKey::Directive(crate::circuit::Directive::Measure))
                .unwrap()
                .occurrences,
            1
        );
    }

    #[test]
    fn basis_analysis_marks_parameterized_and_control_flow_usage() {
        let mut circuit = Circuit::new(2);
        circuit.rx(Qubit::new(0), Parameter::from(0.5)).unwrap();
        circuit
            .if_else(
                ConditionView::new(Qubit::new(0), 1),
                vec![Operation {
                    instruction: StandardGate::X.into(),
                    qubits: smallvec![Qubit::new(1)],
                    params: smallvec![],
                    label: None,
                }],
                None,
            )
            .unwrap();

        let basis = BasisAnalysis::from_circuit(&circuit);
        assert!(basis.has_control_flow());
        assert!(basis.has_parameterized_ops());

        let rx = basis.get(&BasisKey::Standard(StandardGate::RX)).unwrap();
        assert!(rx.is_parameterized);
        let if_else = basis.get(&BasisKey::ControlFlowIfElse).unwrap();
        assert!(if_else.is_control_flow);
        assert!(if_else.is_non_unitary);
    }
}
