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

//! Numeric matrix-backed `UnitaryGate` synthesis.
//!
//! This is the circuit-facing entry point for [`super`]. It rebuilds a source
//! circuit while replacing remaining matrix-backed one- and two-qubit
//! `UnitaryGate` operations with standard-gate sequences. Definition expansion
//! is expected to run first: a unitary gate without a matrix representation is
//! rejected with an error directing the caller to expand definitions.
//!
//! Preserved operations are copied into a fresh circuit instead of retaining
//! source [`CircuitParam`] indices. Their parameters are resolved against the
//! source parameter table and interned into the target table. This applies
//! recursively to preserved control-flow bodies as well.
//!
//! Synthesized global phases are accumulated into the rebuilt circuit. For a
//! decomposition nested inside a control-flow body, the phase is represented by
//! a leading [`StandardGate::GPhase`] operation in that body, because it is
//! conditional on executing the body and cannot be lifted to the circuit-level
//! phase.
//!
//! This pass deliberately leaves ordinary standard gates such as `RX` and `RZ`
//! unchanged. Target-basis rewrites own that behavior. Its supported input
//! contract requires finite numeric one- or two-qubit unitary matrices. It
//! rejects unresolved or non-finite unitary parameters, missing matrices,
//! invalid matrix shapes, and unitary gates acting on three or more qubits.

use super::unitary_1q::synthesize_numeric_1q_unitary;
use super::unitary_2q::{TwoQubitUnitaryDecomposeBasis, synthesize_numeric_2q_unitary};
use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, IfElseGate, Instruction, Operation, Parameter,
    ParameterValue, StandardGate, UnitaryGate, WhileLoopGate,
};
use crate::compiler::CompilerError;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;

const SYNTHESIS_NAME: &str = "decompose.unitary";
const PHASE_EPS: f64 = 1e-12;

/// Configuration for circuit-level matrix-backed `UnitaryGate` synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitaryDecomposeConfig {
    /// Output basis for synthesized two-qubit numeric unitaries.
    ///
    /// This affects only emitted two-qubit interaction gates. Local factors are
    /// emitted as [`StandardGate::U`] operations in both modes.
    pub two_qubit_basis: TwoQubitUnitaryDecomposeBasis,
    /// Whether matrix-backed `UnitaryGate` operations inside control-flow
    /// bodies should be synthesized recursively.
    pub recurse_control_flow: bool,
}

impl Default for UnitaryDecomposeConfig {
    fn default() -> Self {
        Self {
            two_qubit_basis: TwoQubitUnitaryDecomposeBasis::PauliRotations,
            recurse_control_flow: true,
        }
    }
}

/// Rewrites supported matrix-backed `UnitaryGate` operations in `circuit`.
///
/// The output circuit owns a fresh parameter table. Every preserved operation
/// parameter is resolved against the source circuit and rebuilt through the
/// target circuit instead of copying `CircuitParam::Index` values.
///
/// One-qubit matrices are emitted as [`StandardGate::U`] operations when a
/// non-trivial local gate remains. Two-qubit matrices are emitted according to
/// [`UnitaryDecomposeConfig::two_qubit_basis`]. Synthesized scalar phases are
/// preserved either as circuit global phase or, within recursively processed
/// control-flow bodies, as explicit [`StandardGate::GPhase`] operations.
///
/// # Errors
///
/// Returns [`CompilerError`] if a matrix-backed `UnitaryGate` has non-fixed
/// parameters, no matrix representation, unsupported arity, invalid qubit
/// arity, or if the numeric 1q/2q synthesis primitive rejects the matrix.
pub fn decompose_unitaries(
    circuit: &Circuit,
    config: UnitaryDecomposeConfig,
) -> Result<Circuit, CompilerError> {
    let decomposer = UnitaryDecomposer {
        source: circuit,
        target: Circuit::from_qubits(circuit.qubits())?,
        top_phase: circuit.global_phase(),
        config,
    };
    decomposer.run()
}

struct UnitaryDecomposer<'a> {
    source: &'a Circuit,
    target: Circuit,
    top_phase: Parameter,
    config: UnitaryDecomposeConfig,
}

enum SequenceTarget<'a> {
    TopLevel,
    ControlFlowBody(&'a mut Vec<Operation>),
}

struct Decomposition {
    operations: Vec<Operation>,
    phase_delta: f64,
}

impl<'a> UnitaryDecomposer<'a> {
    fn run(mut self) -> Result<Circuit, CompilerError> {
        let phase_delta =
            self.apply_sequence(self.source.operations(), SequenceTarget::TopLevel)?;
        if phase_delta.abs() > PHASE_EPS {
            self.top_phase = self.top_phase + Parameter::from(phase_delta);
        }
        self.target.set_global_phase(self.top_phase);
        Ok(self.target)
    }

    fn apply_sequence(
        &mut self,
        source_operations: &[Operation],
        mut target: SequenceTarget<'_>,
    ) -> Result<f64, CompilerError> {
        let mut phase_delta = 0.0;

        for operation in source_operations {
            let decomposition = self.decompose_operation(operation)?;
            phase_delta += decomposition.phase_delta;
            match &mut target {
                SequenceTarget::TopLevel => {
                    for operation in decomposition.operations {
                        self.append_top_level(operation)?;
                    }
                }
                SequenceTarget::ControlFlowBody(output) => {
                    output.extend(decomposition.operations);
                }
            }
        }

        Ok(phase_delta)
    }

    fn decompose_operation(
        &mut self,
        operation: &Operation,
    ) -> Result<Decomposition, CompilerError> {
        match &operation.instruction {
            Instruction::UnitaryGate(gate) => self.decompose_unitary_gate(gate, operation),
            Instruction::ControlFlowGate(flow) => self.decompose_control_flow(operation, flow),
            _ => Ok(Decomposition {
                operations: vec![self.remap_preserved_operation(operation)?],
                phase_delta: 0.0,
            }),
        }
    }

    fn decompose_control_flow(
        &mut self,
        operation: &Operation,
        flow: &ControlFlow,
    ) -> Result<Decomposition, CompilerError> {
        let instruction = match flow {
            ControlFlow::IfElse(gate) => {
                let true_body = self.rebuild_body(gate.true_body())?;
                let false_body = gate
                    .false_body()
                    .map(|body| self.rebuild_body(body))
                    .transpose()?;
                Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                    gate.condition(),
                    true_body,
                    false_body,
                )))
            }
            ControlFlow::WhileLoop(gate) => {
                let body = self.rebuild_body(gate.body())?;
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                    gate.condition(),
                    body,
                )))
            }
        };

        Ok(Decomposition {
            operations: vec![Operation {
                instruction,
                qubits: operation.qubits.clone(),
                params: self.remap_params(&operation.params)?,
                label: operation.label.clone(),
            }],
            phase_delta: 0.0,
        })
    }

    fn rebuild_body(&mut self, source_body: &[Operation]) -> Result<Vec<Operation>, CompilerError> {
        let mut body = Vec::with_capacity(source_body.len());
        if self.config.recurse_control_flow {
            let phase_delta =
                self.apply_sequence(source_body, SequenceTarget::ControlFlowBody(&mut body))?;
            if phase_delta.abs() > PHASE_EPS {
                body.insert(
                    0,
                    Operation {
                        instruction: Instruction::Standard(StandardGate::GPhase),
                        qubits: smallvec![],
                        params: smallvec![CircuitParam::Fixed(phase_delta)],
                        label: None,
                    },
                );
            }
        } else {
            for operation in source_body {
                body.push(self.remap_preserved_operation(operation)?);
            }
        }
        Ok(body)
    }

    fn decompose_unitary_gate(
        &mut self,
        gate: &UnitaryGate,
        operation: &Operation,
    ) -> Result<Decomposition, CompilerError> {
        if operation.qubits.len() != gate.num_qubits() as usize {
            return Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!(
                    "operation qubit count mismatch for UnitaryGate '{}': expected {}, got {}",
                    gate.label(),
                    gate.num_qubits(),
                    operation.qubits.len()
                ),
            });
        }
        if gate.matrix_repr().is_none() {
            return Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!(
                    "UnitaryGate '{}' has no matrix representation; run definition expansion before unitary synthesis",
                    gate.label()
                ),
            });
        }

        let matrix = self.numeric_matrix_for_gate(gate, operation)?;
        match gate.num_qubits() {
            1 => {
                let ([theta, phi, lambda], global_phase) =
                    synthesize_numeric_1q_unitary(matrix.as_ref()).map_err(|source| {
                        CompilerError::TransformFailed {
                            name: SYNTHESIS_NAME,
                            reason: format!(
                                "one-qubit synthesis failed for UnitaryGate '{}': {source}",
                                gate.label()
                            ),
                        }
                    })?;
                let mut operations = Vec::new();
                if theta.abs() > PHASE_EPS || phi.abs() > PHASE_EPS || lambda.abs() > PHASE_EPS {
                    operations.push(Operation {
                        instruction: Instruction::Standard(StandardGate::U),
                        qubits: operation.qubits.clone(),
                        params: smallvec![
                            CircuitParam::Fixed(theta),
                            CircuitParam::Fixed(phi),
                            CircuitParam::Fixed(lambda)
                        ],
                        label: None,
                    });
                }
                Ok(Decomposition {
                    operations,
                    phase_delta: global_phase,
                })
            }
            2 => {
                let qubits = [operation.qubits[0], operation.qubits[1]];
                let (operations, phase_delta) = synthesize_numeric_2q_unitary(
                    matrix.as_ref(),
                    qubits,
                    self.config.two_qubit_basis,
                )
                .map_err(|source| CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!(
                        "two-qubit synthesis failed for UnitaryGate '{}': {source}",
                        gate.label(),
                    ),
                })?;
                Ok(Decomposition {
                    operations,
                    phase_delta,
                })
            }
            other => Err(CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!(
                    "3q+ UnitaryGate synthesis is not supported yet; gate '{}' has {other} qubits",
                    gate.label()
                ),
            }),
        }
    }

    fn numeric_matrix_for_gate<'gate>(
        &self,
        gate: &'gate UnitaryGate,
        operation: &Operation,
    ) -> Result<Cow<'gate, Array2<Complex64>>, CompilerError> {
        let mut fixed_params = Vec::with_capacity(operation.params.len());
        for (position, param) in operation.params.iter().enumerate() {
            let parameter = self.resolve_source_param(param)?;
            let value = parameter.evaluate(&None).map_err(|_| {
                let mut symbols = parameter.get_symbols().into_iter().collect::<Vec<_>>();
                symbols.sort();
                let detail = if symbols.is_empty() {
                    parameter.to_string()
                } else {
                    symbols.join(", ")
                };
                CompilerError::TransformFailed {
                    name: SYNTHESIS_NAME,
                    reason: format!(
                        "UnitaryGate '{}' parameter {position} must be fixed numeric before synthesis; unresolved symbols: {detail}",
                        gate.label()
                    ),
                }
            })?;
            if !value.is_finite() {
                return Err(CompilerError::InvalidInput(format!(
                    "non-finite unitary parameter {value} at position {position} for UnitaryGate '{}'",
                    gate.label()
                )));
            }
            fixed_params.push(value);
        }

        gate.matrix_for_params(&fixed_params)
            .map_err(|source| CompilerError::TransformFailed {
                name: SYNTHESIS_NAME,
                reason: format!(
                    "failed to resolve numeric matrix for UnitaryGate '{}': {source}",
                    gate.label()
                ),
            })
    }

    fn remap_preserved_operation(
        &mut self,
        operation: &Operation,
    ) -> Result<Operation, CompilerError> {
        let instruction = match &operation.instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                let mut true_body = Vec::with_capacity(gate.true_body().len());
                for inner in gate.true_body() {
                    true_body.push(self.remap_preserved_operation(inner)?);
                }
                let false_body = gate
                    .false_body()
                    .map(|source_body| {
                        let mut mapped = Vec::with_capacity(source_body.len());
                        for inner in source_body {
                            mapped.push(self.remap_preserved_operation(inner)?);
                        }
                        Ok::<_, CompilerError>(mapped)
                    })
                    .transpose()?;
                Instruction::ControlFlowGate(ControlFlow::IfElse(IfElseGate::new(
                    gate.condition(),
                    true_body,
                    false_body,
                )))
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                let mut body = Vec::with_capacity(gate.body().len());
                for inner in gate.body() {
                    body.push(self.remap_preserved_operation(inner)?);
                }
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(WhileLoopGate::new(
                    gate.condition(),
                    body,
                )))
            }
            _ => operation.instruction.clone(),
        };

        Ok(Operation {
            instruction,
            qubits: operation.qubits.clone(),
            params: self.remap_params(&operation.params)?,
            label: operation.label.clone(),
        })
    }

    fn remap_params(
        &mut self,
        params: &[CircuitParam],
    ) -> Result<SmallVec<[CircuitParam; 1]>, CompilerError> {
        let mut remapped = SmallVec::with_capacity(params.len());
        for param in params {
            let parameter = self.resolve_source_param(param)?;
            remapped.push(self.intern_target_param(parameter)?);
        }
        Ok(remapped)
    }

    fn resolve_source_param(&self, param: &CircuitParam) -> Result<Parameter, CompilerError> {
        match param {
            CircuitParam::Fixed(value) => {
                if !value.is_finite() {
                    return Err(CompilerError::InvalidInput(format!(
                        "non-finite fixed parameter {value}"
                    )));
                }
                Ok(Parameter::from(*value))
            }
            CircuitParam::Index(index) => self
                .source
                .parameters()
                .get_index(*index as usize)
                .cloned()
                .ok_or_else(|| {
                    CompilerError::InvalidInput(format!("missing parameter index {index}"))
                }),
        }
    }

    fn intern_target_param(&mut self, parameter: Parameter) -> Result<CircuitParam, CompilerError> {
        match ParameterValue::from(parameter) {
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
        let mut params = Vec::with_capacity(operation.params.len());
        for param in &operation.params {
            match param {
                CircuitParam::Fixed(value) => params.push(ParameterValue::Fixed(*value)),
                CircuitParam::Index(index) => {
                    let parameter = self
                        .target
                        .parameters()
                        .get_index(*index as usize)
                        .cloned()
                        .ok_or_else(|| {
                            CompilerError::InvariantViolation(format!(
                                "unitary decomposition produced missing target parameter index {index}"
                            ))
                        })?;
                    params.push(ParameterValue::Param(parameter));
                }
            }
        }

        self.target.append(
            operation.instruction,
            operation.qubits,
            params,
            operation.label.as_deref(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Qubit;
    use crate::circuit::gate::gate_matrix;
    use crate::circuit::symbolic_matrix::{SymbolicComplex, SymbolicMatrix};
    use crate::circuit::{ConditionView, circuit_to_matrix};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn operation_parameter(circuit: &Circuit, operation: &Operation, position: usize) -> Parameter {
        match operation.params.get(position) {
            Some(CircuitParam::Fixed(value)) => Parameter::from(*value),
            Some(CircuitParam::Index(index)) => circuit.parameters()[*index as usize].clone(),
            None => panic!("operation has no parameter at position {position}"),
        }
    }

    #[test]
    fn decomposes_numeric_1q_unitary_gate() {
        let gamma = 0.37;
        let matrix = gate_matrix::u_gate(0.3, 0.4, 0.5) * Complex64::from_polar(1.0, gamma);
        let gate = UnitaryGate::new("custom_u", 1, 0)
            .with_matrix(matrix)
            .unwrap();
        let mut circuit = Circuit::new(1);
        circuit.unitary(gate, vec![Qubit::new(0)]).unwrap();

        let before = circuit_to_matrix(&circuit, None).unwrap();
        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let after = circuit_to_matrix(&decomposed, None).unwrap();

        assert!(decomposed.operations().iter().all(|operation| matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::U)
        )));
        assert_abs_diff_eq!(before, after, epsilon = 1e-8);
    }

    #[test]
    fn decomposes_numeric_2q_unitary_gate_with_pauli_backend() {
        let matrix = StandardGate::FSIM
            .matrix(&[0.2, -0.3])
            .unwrap()
            .into_owned();
        let gate = UnitaryGate::new("custom_2q", 2, 0)
            .with_matrix(matrix)
            .unwrap();
        let mut circuit = Circuit::new(2);
        circuit
            .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
            .unwrap();

        let before = circuit_to_matrix(&circuit, None).unwrap();
        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let after = circuit_to_matrix(&decomposed, None).unwrap();

        assert!(decomposed.operations().iter().all(|operation| matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::U)
                | Instruction::Standard(StandardGate::RXX)
                | Instruction::Standard(StandardGate::RYY)
                | Instruction::Standard(StandardGate::RZZ)
        )));
        assert_abs_diff_eq!(before, after, epsilon = 1e-8);
    }

    #[test]
    fn decomposes_numeric_2q_unitary_gate_with_cx_backend() {
        let matrix = StandardGate::SWAP.matrix(&[]).unwrap().into_owned();
        let gate = UnitaryGate::new("custom_2q", 2, 0)
            .with_matrix(matrix)
            .unwrap();
        let mut circuit = Circuit::new(2);
        circuit
            .unitary(gate, vec![Qubit::new(0), Qubit::new(1)])
            .unwrap();
        let config = UnitaryDecomposeConfig {
            two_qubit_basis: TwoQubitUnitaryDecomposeBasis::Cx,
            ..Default::default()
        };

        let before = circuit_to_matrix(&circuit, None).unwrap();
        let decomposed = decompose_unitaries(&circuit, config).unwrap();
        let after = circuit_to_matrix(&decomposed, None).unwrap();

        assert!(decomposed.operations().iter().all(|operation| matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::U) | Instruction::Standard(StandardGate::CX)
        )));
        assert_abs_diff_eq!(before, after, epsilon = 1e-8);
    }

    #[test]
    fn remaps_preserved_top_level_operation_parameters() {
        let mut circuit = Circuit::new(2);
        circuit.add_parameter(Parameter::symbol("unused"));
        let theta = Parameter::symbol("theta");
        circuit
            .append(
                Instruction::Standard(StandardGate::RXX),
                [Qubit::new(0), Qubit::new(1)],
                [ParameterValue::Param(theta)],
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let operation = &decomposed.operations()[0];
        let CircuitParam::Index(index) = operation.params[0] else {
            panic!("expected symbolic parameter index");
        };

        assert_eq!(index, 0);
        assert_eq!(decomposed.parameters().len(), 1);
        assert!(
            operation_parameter(&decomposed, operation, 0)
                .get_symbols()
                .contains("theta")
        );
    }

    #[test]
    fn remaps_preserved_control_flow_body_parameters() {
        let mut circuit = Circuit::new(2);
        circuit.add_parameter(Parameter::symbol("unused"));
        let (theta_index, _) = circuit.add_parameter(Parameter::symbol("theta"));
        let body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![CircuitParam::Index(theta_index as u32)],
            label: None,
        }];
        let flow = IfElseGate::new(ConditionView::new(Qubit::new(0), 1), body, None);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(flow)),
                [Qubit::new(0), Qubit::new(1)],
                std::iter::empty(),
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let Instruction::ControlFlowGate(ControlFlow::IfElse(flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected if-else gate");
        };
        let operation = &flow.true_body()[0];
        let CircuitParam::Index(index) = operation.params[0] else {
            panic!("expected remapped symbolic parameter index");
        };

        assert_eq!(index, 0);
        assert_eq!(decomposed.parameters().len(), 1);
        assert!(
            operation_parameter(&decomposed, operation, 0)
                .get_symbols()
                .contains("theta")
        );
    }

    #[test]
    fn remaps_preserved_false_body_parameters() {
        let mut circuit = Circuit::new(2);
        circuit.add_parameter(Parameter::symbol("unused"));
        let (phi_index, _) = circuit.add_parameter(Parameter::symbol("phi"));
        let true_body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::I),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        }];
        let false_body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![CircuitParam::Index(phi_index as u32)],
            label: None,
        }];
        let flow = IfElseGate::new(
            ConditionView::new(Qubit::new(0), 1),
            true_body,
            Some(false_body),
        );
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(flow)),
                [Qubit::new(0), Qubit::new(1)],
                std::iter::empty(),
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let Instruction::ControlFlowGate(ControlFlow::IfElse(flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected if-else gate");
        };
        let false_body = flow.false_body().expect("expected false body");
        let CircuitParam::Index(index) = false_body[0].params[0] else {
            panic!("expected remapped false-body parameter index");
        };

        assert_eq!(index, 0);
        assert_eq!(decomposed.parameters().len(), 1);
        assert!(
            operation_parameter(&decomposed, &false_body[0], 0)
                .get_symbols()
                .contains("phi")
        );
    }

    #[test]
    fn remaps_preserved_while_body_parameters() {
        let mut circuit = Circuit::new(2);
        circuit.add_parameter(Parameter::symbol("unused"));
        let (beta_index, _) = circuit.add_parameter(Parameter::symbol("beta"));
        let body = vec![Operation {
            instruction: Instruction::Standard(StandardGate::RY),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![CircuitParam::Index(beta_index as u32)],
            label: None,
        }];
        let flow = WhileLoopGate::new(ConditionView::new(Qubit::new(0), 1), body);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::WhileLoop(flow)),
                [Qubit::new(0), Qubit::new(1)],
                std::iter::empty(),
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let Instruction::ControlFlowGate(ControlFlow::WhileLoop(flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected while gate");
        };
        let CircuitParam::Index(index) = flow.body()[0].params[0] else {
            panic!("expected remapped while-body parameter index");
        };

        assert_eq!(index, 0);
        assert_eq!(decomposed.parameters().len(), 1);
        assert!(
            operation_parameter(&decomposed, &flow.body()[0], 0)
                .get_symbols()
                .contains("beta")
        );
    }

    #[test]
    fn keeps_body_local_phase_inside_control_flow() {
        let phase = Complex64::from_polar(1.0, 0.25);
        let gate = UnitaryGate::new("phase_cx", 2, 0)
            .with_matrix(
                StandardGate::CX
                    .matrix(&[])
                    .unwrap()
                    .into_owned()
                    .mapv(|value| phase * value),
            )
            .unwrap();
        let body = vec![Operation {
            instruction: Instruction::UnitaryGate(Box::new(gate)),
            qubits: smallvec![Qubit::new(1), Qubit::new(2)],
            params: smallvec![],
            label: None,
        }];
        let flow = IfElseGate::new(ConditionView::new(Qubit::new(0), 1), body, None);
        let mut circuit = Circuit::new(3);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(flow)),
                [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
                std::iter::empty(),
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        assert!(decomposed.global_phase().is_zero());
        let Instruction::ControlFlowGate(ControlFlow::IfElse(flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected if-else gate");
        };
        assert!(matches!(
            flow.true_body()[0].instruction,
            Instruction::Standard(StandardGate::GPhase)
        ));
        assert!(matches!(
            flow.true_body()[0].params[0],
            CircuitParam::Fixed(_)
        ));
    }

    #[test]
    fn recurses_nested_control_flow_bodies() {
        let phase = Complex64::from_polar(1.0, 0.41);
        let gate = UnitaryGate::new("phase_x", 1, 0)
            .with_matrix(
                StandardGate::X
                    .matrix(&[])
                    .unwrap()
                    .into_owned()
                    .mapv(|value| phase * value),
            )
            .unwrap();
        let inner_body = vec![Operation {
            instruction: Instruction::UnitaryGate(Box::new(gate)),
            qubits: smallvec![Qubit::new(1)],
            params: smallvec![],
            label: None,
        }];
        let inner_flow = WhileLoopGate::new(ConditionView::new(Qubit::new(0), 1), inner_body);
        let outer_body = vec![Operation {
            instruction: Instruction::ControlFlowGate(ControlFlow::WhileLoop(inner_flow)),
            qubits: smallvec![Qubit::new(0), Qubit::new(1)],
            params: smallvec![],
            label: None,
        }];
        let outer_flow = IfElseGate::new(ConditionView::new(Qubit::new(0), 1), outer_body, None);
        let mut circuit = Circuit::new(2);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(outer_flow)),
                [Qubit::new(0), Qubit::new(1)],
                std::iter::empty(),
                None,
            )
            .unwrap();

        let decomposed = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap();
        let Instruction::ControlFlowGate(ControlFlow::IfElse(outer_flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected outer if-else gate");
        };
        let Instruction::ControlFlowGate(ControlFlow::WhileLoop(inner_flow)) =
            &outer_flow.true_body()[0].instruction
        else {
            panic!("expected nested while gate");
        };

        assert!(matches!(
            inner_flow.body()[0].instruction,
            Instruction::Standard(StandardGate::GPhase)
        ));
        assert!(inner_flow.body().iter().any(|operation| matches!(
            operation.instruction,
            Instruction::Standard(StandardGate::U)
        )));
    }

    #[test]
    fn can_preserve_control_flow_body_unitaries_without_recursing() {
        let mut circuit = Circuit::new(2);
        circuit.add_parameter(Parameter::symbol("unused"));
        let (theta_index, _) = circuit.add_parameter(Parameter::symbol("theta"));
        let gate = UnitaryGate::new("x_body", 1, 0)
            .with_matrix(StandardGate::X.matrix(&[]).unwrap().into_owned())
            .unwrap();
        let body = vec![
            Operation {
                instruction: Instruction::UnitaryGate(Box::new(gate)),
                qubits: smallvec![Qubit::new(1)],
                params: smallvec![],
                label: None,
            },
            Operation {
                instruction: Instruction::Standard(StandardGate::RZ),
                qubits: smallvec![Qubit::new(1)],
                params: smallvec![CircuitParam::Index(theta_index as u32)],
                label: None,
            },
        ];
        let flow = IfElseGate::new(ConditionView::new(Qubit::new(0), 1), body, None);
        circuit
            .append(
                Instruction::ControlFlowGate(ControlFlow::IfElse(flow)),
                [Qubit::new(0), Qubit::new(1)],
                std::iter::empty(),
                None,
            )
            .unwrap();
        let config = UnitaryDecomposeConfig {
            recurse_control_flow: false,
            ..Default::default()
        };

        let decomposed = decompose_unitaries(&circuit, config).unwrap();
        let Instruction::ControlFlowGate(ControlFlow::IfElse(flow)) =
            &decomposed.operations()[0].instruction
        else {
            panic!("expected if-else gate");
        };
        let body = flow.true_body();
        let CircuitParam::Index(index) = body[1].params[0] else {
            panic!("expected remapped preserved body parameter index");
        };

        assert!(matches!(body[0].instruction, Instruction::UnitaryGate(_)));
        assert_eq!(index, 0);
        assert_eq!(decomposed.parameters().len(), 1);
    }

    #[test]
    fn rejects_unbound_symbolic_unitary_gate() {
        let theta = Parameter::symbol("theta");
        let matrix: SymbolicMatrix = array![
            [SymbolicComplex::one(), SymbolicComplex::zero()],
            [SymbolicComplex::zero(), SymbolicComplex::exp_i(theta)]
        ];
        let gate = UnitaryGate::new("symbolic_1q", 1, 1)
            .with_symbolic_matrix(["theta"], matrix)
            .unwrap();
        let mut circuit = Circuit::new(1);
        circuit
            .unitary_with_params(
                gate,
                vec![Qubit::new(0)],
                vec![Parameter::symbol("alpha").into()],
            )
            .unwrap();

        let err = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap_err();
        assert!(
            err.to_string()
                .contains("parameter 0 must be fixed numeric")
        );
        assert!(err.to_string().contains("unresolved symbols: alpha"));
    }

    #[test]
    fn rejects_three_qubit_unitary_gate() {
        let gate = UnitaryGate::new("custom_3q", 3, 0)
            .with_matrix(Array2::eye(8))
            .unwrap();
        let mut circuit = Circuit::new(3);
        circuit
            .unitary(gate, vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)])
            .unwrap();

        let err = decompose_unitaries(&circuit, UnitaryDecomposeConfig::default()).unwrap_err();
        assert!(err.to_string().contains("not supported yet"));
    }
}
