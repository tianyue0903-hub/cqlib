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

//! Transformer entry point for multi-controlled standard-gate lowering.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, ParameterValue, Qubit,
    WhileLoopGate,
};
use crate::compiler::context::{CompilerContext, ContextChangeSet};
use crate::compiler::error::CompilerError;
use crate::compiler::transform::{TransformDescriptor, TransformOutcome, Transformer};
use indexmap::IndexSet;
use smallvec::SmallVec;

use super::decompose::{McGateDecomposeConfig, decompose_mc_gate_operation};

/// Compiler transformer that lowers [`Instruction::McGate`] operations.
#[derive(Debug, Clone)]
pub struct McGateDecomposer {
    /// Local MCGate lowering policy.
    config: McGateDecomposeConfig,
    /// Whether nested control-flow bodies should be decomposed recursively.
    recurse_control_flow: bool,
}

impl McGateDecomposer {
    /// Creates a multi-controlled gate decomposer with explicit configuration.
    pub fn new(config: McGateDecomposeConfig) -> Self {
        Self {
            config,
            recurse_control_flow: true,
        }
    }

    /// Returns the active MCGate decomposition configuration.
    pub const fn config(&self) -> &McGateDecomposeConfig {
        &self.config
    }

    /// Controls whether nested control-flow bodies are recursively decomposed.
    pub fn recurse_control_flow(mut self, enabled: bool) -> Self {
        self.recurse_control_flow = enabled;
        self
    }
}

impl Default for McGateDecomposer {
    fn default() -> Self {
        Self::new(McGateDecomposeConfig::default())
    }
}

static MC_GATE_DECOMPOSER_DESCRIPTOR: TransformDescriptor = TransformDescriptor::new(
    "decompose.mc_gate",
    "Decomposes multi-controlled standard-gate wrappers",
)
.supports_control_flow(true)
.supports_symbolic_parameters(true)
.modifies_circuit();

impl Transformer for McGateDecomposer {
    fn descriptor(&self) -> &'static TransformDescriptor {
        &MC_GATE_DECOMPOSER_DESCRIPTOR
    }

    fn transform(&self, ctx: &mut CompilerContext) -> Result<TransformOutcome, CompilerError> {
        let source = ctx.circuit().clone();
        let qubits: IndexSet<_> = source.qubits().into_iter().collect();
        let mut rebuilt = Circuit::from_parts(
            qubits,
            source.symbols().clone(),
            source.parameters().clone(),
            Vec::new(),
            source.global_phase_param().clone(),
        );
        let mut stats = McGateDecomposeStats::default();

        apply_sequence(
            &source,
            source.operations(),
            &mut rebuilt,
            &self.config,
            self.recurse_control_flow,
            SequenceTarget::TopLevel,
            &mut stats,
        )?;

        if stats.decomposed_operations == 0 {
            return Ok(TransformOutcome::unchanged());
        }

        *ctx.circuit_mut() = rebuilt;
        let mut outcome = TransformOutcome::changed()
            .with_changes(
                ContextChangeSet::circuit_changed()
                    .with_cfg_structure_changed(true)
                    .with_parameter_table_changed(true),
            )
            .with_note(format!(
                "decompose.mc_gate: lowered {} MCGate operations into {} standard operations across {} changed sequences",
                stats.decomposed_operations,
                stats.emitted_replacement_operations,
                stats.changed_sequences
            ));
        outcome.notes.extend(stats.notes);

        Ok(outcome)
    }
}

/// Aggregate statistics for one MCGate decomposition run.
#[derive(Debug, Clone, Default)]
struct McGateDecomposeStats {
    decomposed_operations: usize,
    emitted_replacement_operations: usize,
    changed_sequences: usize,
    notes: Vec<String>,
}

/// Destination for emitted operations while rebuilding a sequence.
enum SequenceTarget<'a> {
    /// Output sequence is the rebuilt top-level circuit.
    TopLevel,
    /// Output sequence is a rebuilt control-flow body.
    ControlFlowBody { output: &'a mut Vec<Operation> },
}

/// Applies MCGate lowering to one operation sequence.
fn apply_sequence(
    source: &Circuit,
    operations: &[Operation],
    rebuilt: &mut Circuit,
    config: &McGateDecomposeConfig,
    recurse_control_flow: bool,
    mut target: SequenceTarget<'_>,
    stats: &mut McGateDecomposeStats,
) -> Result<(), CompilerError> {
    let before = stats.decomposed_operations;
    for operation in operations {
        emit_operation(
            source,
            operation,
            rebuilt,
            config,
            recurse_control_flow,
            &mut target,
            stats,
        )?;
    }
    if stats.decomposed_operations > before {
        stats.changed_sequences += 1;
    }
    Ok(())
}

/// Emits one operation, recursively lowering control-flow bodies.
fn emit_operation(
    source: &Circuit,
    operation: &Operation,
    rebuilt: &mut Circuit,
    config: &McGateDecomposeConfig,
    recurse_control_flow: bool,
    target: &mut SequenceTarget<'_>,
    stats: &mut McGateDecomposeStats,
) -> Result<(), CompilerError> {
    let rewritten_instruction = if recurse_control_flow {
        match &operation.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                let mut true_body = Vec::with_capacity(gate.true_body().len());
                apply_sequence(
                    source,
                    gate.true_body(),
                    rebuilt,
                    config,
                    recurse_control_flow,
                    SequenceTarget::ControlFlowBody {
                        output: &mut true_body,
                    },
                    stats,
                )?;

                let false_body = gate
                    .false_body()
                    .map(|body| {
                        let mut rewritten = Vec::with_capacity(body.len());
                        apply_sequence(
                            source,
                            body,
                            rebuilt,
                            config,
                            recurse_control_flow,
                            SequenceTarget::ControlFlowBody {
                                output: &mut rewritten,
                            },
                            stats,
                        )?;
                        Ok::<_, CompilerError>(rewritten)
                    })
                    .transpose()?;

                Some(Instruction::ControlFlowGate(ControlFlow::IfElse(
                    IfElseGate::new(gate.condition(), true_body, false_body),
                )))
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                let mut body = Vec::with_capacity(gate.body().len());
                apply_sequence(
                    source,
                    gate.body(),
                    rebuilt,
                    config,
                    recurse_control_flow,
                    SequenceTarget::ControlFlowBody { output: &mut body },
                    stats,
                )?;

                Some(Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                    WhileLoopGate::new(gate.condition(), body),
                )))
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(instruction) = rewritten_instruction {
        let qubits = control_flow_operation_qubits(&instruction);
        return emit_operation_parts(
            source,
            rebuilt,
            target,
            instruction,
            qubits,
            operation.params.as_slice(),
            operation.label.clone(),
        );
    }

    let result = decompose_mc_gate_operation(operation, config)?;
    if result.changed {
        stats.decomposed_operations += 1;
        stats.emitted_replacement_operations += result.operations.len();
        stats.notes.extend(result.notes);
        for replacement in result.operations {
            emit_operation_parts(
                source,
                rebuilt,
                target,
                replacement.instruction,
                replacement.qubits,
                replacement.params.as_slice(),
                replacement.label,
            )?;
        }
    } else {
        emit_operation_parts(
            source,
            rebuilt,
            target,
            operation.instruction.clone(),
            operation.qubits.clone(),
            operation.params.as_slice(),
            operation.label.clone(),
        )?;
    }

    Ok(())
}

/// Recomputes the qubit list for a rebuilt control-flow operation.
fn control_flow_operation_qubits(instruction: &Instruction) -> SmallVec<[Qubit; 3]> {
    let mut qubits = SmallVec::new();
    match instruction {
        Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
            collect_operation_qubits(gate.true_body(), &mut qubits);
            if let Some(false_body) = gate.false_body() {
                collect_operation_qubits(false_body, &mut qubits);
            }
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
            collect_operation_qubits(gate.body(), &mut qubits);
            push_unique_qubit(&mut qubits, gate.condition().qubit);
        }
        _ => {}
    }
    qubits
}

/// Collects operation qubits without duplicates.
fn collect_operation_qubits(operations: &[Operation], output: &mut SmallVec<[Qubit; 3]>) {
    for operation in operations {
        for &qubit in &operation.qubits {
            push_unique_qubit(output, qubit);
        }
    }
}

/// Pushes one qubit if it is not already present.
fn push_unique_qubit(output: &mut SmallVec<[Qubit; 3]>, qubit: Qubit) {
    if !output.contains(&qubit) {
        output.push(qubit);
    }
}

/// Emits operation parts into either the rebuilt circuit or a control-flow body.
fn emit_operation_parts(
    source: &Circuit,
    rebuilt: &mut Circuit,
    target: &mut SequenceTarget<'_>,
    instruction: Instruction,
    qubits: SmallVec<[Qubit; 3]>,
    params: &[CircuitParam],
    label: Option<Box<str>>,
) -> Result<(), CompilerError> {
    match target {
        SequenceTarget::TopLevel => {
            let param_values = params
                .iter()
                .map(|param| resolve_parameter_value(source, param))
                .collect::<Result<SmallVec<[_; 3]>, _>>()?;
            rebuilt.append(instruction, qubits, param_values, label.as_deref())?;
        }
        SequenceTarget::ControlFlowBody { output } => output.push(Operation {
            instruction,
            qubits,
            params: params.iter().cloned().collect(),
            label,
        }),
    }

    Ok(())
}

/// Resolves a circuit-local parameter reference against the source circuit.
fn resolve_parameter_value(
    circuit: &Circuit,
    param: &CircuitParam,
) -> Result<ParameterValue, CompilerError> {
    match param {
        CircuitParam::Fixed(value) => Ok(ParameterValue::Fixed(*value)),
        CircuitParam::Index(index) => {
            let index = *index as usize;
            let parameter = circuit.parameters().get_index(index).ok_or_else(|| {
                CompilerError::InvariantViolation(format!(
                    "operation references missing parameter index {index}"
                ))
            })?;
            Ok(ParameterValue::Param(parameter.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::McGateDecomposer;
    use crate::circuit::{
        Circuit, ConditionView, ControlFlow, IfElseGate, Instruction, MCGate, Operation, Parameter,
        ParameterValue, Qubit, StandardGate, WhileLoopGate,
    };
    use crate::compiler::CompilerContext;
    use crate::compiler::transform::Transformer;
    use crate::compiler::transform::decompose::{
        McGateDecomposeConfig, McGateDecomposer as PublicMcGateDecomposer,
    };
    use smallvec::smallvec;

    fn mc_x_operation(label: Option<&str>) -> Operation {
        Operation {
            instruction: Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
            qubits: smallvec![Qubit::new(0), Qubit::new(1)],
            params: smallvec![],
            label: label.map(Into::into),
        }
    }

    #[test]
    fn public_export_exposes_mc_gate_decomposer() {
        let _transformer = PublicMcGateDecomposer::default();
    }

    #[test]
    fn mc_gate_decomposer_lowers_top_level_mc_gate() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        let outcome = McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(outcome.changed);
        assert_eq!(ctx.revision(), 1);
        assert!(outcome.notes[0].contains("lowered 1 MCGate"));
        let operations = ctx.circuit().operations();
        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::CX)
        ));
        assert_eq!(
            operations[0].qubits.as_slice(),
            &[Qubit::new(0), Qubit::new(1)]
        );
    }

    #[test]
    fn mc_gate_decomposer_reports_unchanged_when_no_mc_gate_is_present() {
        let mut circuit = Circuit::new(1);
        circuit.h(Qubit::new(0)).unwrap();
        let mut ctx = CompilerContext::new(circuit);

        let outcome = McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(!outcome.changed);
        assert_eq!(ctx.revision(), 0);
        assert_eq!(ctx.circuit().operations().len(), 1);
        assert!(matches!(
            ctx.circuit().operations()[0].instruction,
            Instruction::Standard(StandardGate::H)
        ));
    }

    #[test]
    fn mc_gate_decomposer_skips_labeled_mc_gate_by_default() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                Some("protected"),
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        let outcome = McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(!outcome.changed);
        assert_eq!(ctx.revision(), 0);
        assert!(matches!(
            ctx.circuit().operations()[0].instruction,
            Instruction::McGate(_)
        ));
        assert_eq!(
            ctx.circuit().operations()[0].label.as_deref(),
            Some("protected")
        );
    }

    #[test]
    fn mc_gate_decomposer_lowers_labeled_mc_gate_when_enabled() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                Some("lower"),
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);
        let transformer =
            McGateDecomposer::new(McGateDecomposeConfig::new().skip_labeled_ops(false));

        let outcome = transformer.run(&mut ctx).unwrap();

        assert!(outcome.changed);
        let operations = ctx.circuit().operations();
        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::CX)
        ));
        assert_eq!(operations[0].label, None);
    }

    #[test]
    fn mc_gate_decomposer_recurses_into_if_else_bodies_by_default() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                    ConditionView::new(Qubit::new(0), 1),
                    vec![mc_x_operation(None)],
                    Some(vec![Operation {
                        instruction: Instruction::Standard(StandardGate::H),
                        qubits: smallvec![Qubit::new(1)],
                        params: smallvec![],
                        label: None,
                    }]),
                ))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        let outcome = McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(outcome.changed);
        let Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) =
            &ctx.circuit().operations()[0].instruction
        else {
            panic!("expected if-else operation");
        };
        assert_eq!(gate.true_body().len(), 1);
        assert!(matches!(
            gate.true_body()[0].instruction,
            Instruction::Standard(StandardGate::CX)
        ));
        assert!(matches!(
            gate.false_body().unwrap()[0].instruction,
            Instruction::Standard(StandardGate::H)
        ));
    }

    #[test]
    fn mc_gate_decomposer_can_leave_control_flow_bodies_unmodified() {
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                    ConditionView::new(Qubit::new(0), 1),
                    vec![mc_x_operation(None)],
                ))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);
        let transformer = McGateDecomposer::default().recurse_control_flow(false);

        let outcome = transformer.run(&mut ctx).unwrap();

        assert!(!outcome.changed);
        let Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) =
            &ctx.circuit().operations()[0].instruction
        else {
            panic!("expected while-loop operation");
        };
        assert!(matches!(gate.body()[0].instruction, Instruction::McGate(_)));
    }

    #[test]
    fn mc_gate_decomposer_reinterns_symbolic_parameters_through_append() {
        let mut circuit = Circuit::new(2);
        let theta = Parameter::symbol("theta");
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::RX))),
                [Qubit::new(0), Qubit::new(1)],
                [ParameterValue::from(theta.clone())],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        let outcome = McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(outcome.changed);
        let operations = ctx.circuit().operations();
        assert_eq!(operations.len(), 1);
        assert!(matches!(
            operations[0].instruction,
            Instruction::Standard(StandardGate::CRX)
        ));
        assert_eq!(ctx.circuit().parameters().len(), 1);
        assert_eq!(ctx.circuit().parameters().get_index(0).unwrap(), &theta);
    }

    #[test]
    fn mc_gate_decomposer_preserves_global_phase() {
        let mut circuit = Circuit::new(2);
        circuit.set_global_phase(Parameter::from(0.125));
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        McGateDecomposer::default().run(&mut ctx).unwrap();

        assert_eq!(ctx.circuit().global_phase().evaluate(&None).unwrap(), 0.125);
    }

    #[test]
    fn mc_gate_decomposer_removes_all_top_level_mc_gate_operations() {
        let mut circuit = Circuit::new(3);
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(1, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1)],
                [],
                None,
            )
            .unwrap();
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
                [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
                [],
                None,
            )
            .unwrap();
        let mut ctx = CompilerContext::new(circuit);

        McGateDecomposer::default().run(&mut ctx).unwrap();

        assert!(
            ctx.circuit()
                .operations()
                .iter()
                .all(|operation| !matches!(operation.instruction, Instruction::McGate(_)))
        );
    }
}
