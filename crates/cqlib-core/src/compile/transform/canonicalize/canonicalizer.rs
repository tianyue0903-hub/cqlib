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

//! Canonicalizer entry point and round orchestration.

use crate::circuit::{
    Circuit, CircuitParam, ControlFlow, Directive, IfElseGate, Instruction, Operation, Parameter,
    StandardGate, WhileLoopGate,
};
use crate::compile::CompilerError;
use crate::compile::transform::transformer::{TransformResult, Transformer};
use smallvec::{SmallVec, smallvec};

use super::config::CanonicalizeConfig;
use super::equivalence::circuits_equivalent_for_canonicalize;
use super::ops::{
    canonical_control_flow_qubits_for_operation, canonicalize_operation_qubits, is_strict_noop,
    push_operation,
};
use super::params::{
    canonical_parameter, circuit_param_to_value, parameter_is_exact_zero,
    parameter_to_circuit_param, resolve_parameter,
};
use super::verify::{VerifyMode, verify_circuit};

/// Result of a canonicalization run.
#[derive(Debug, Clone)]
pub struct CanonicalizeResult {
    /// Canonicalized circuit.
    pub circuit: Circuit,
    /// Whether the output differs from the input representation.
    pub changed: bool,
    /// Number of canonicalization rounds executed.
    pub rounds: u8,
}

/// Canonicalizer entry point.
#[derive(Debug, Clone)]
pub struct Canonicalizer {
    config: CanonicalizeConfig,
}

impl Default for Canonicalizer {
    fn default() -> Self {
        Self::production()
    }
}

impl Canonicalizer {
    /// Creates a canonicalizer with the supplied configuration.
    pub const fn new(config: CanonicalizeConfig) -> Self {
        Self { config }
    }

    /// Creates a canonicalizer using production defaults.
    pub const fn production() -> Self {
        Self::new(CanonicalizeConfig::production())
    }

    /// Returns the active configuration.
    pub const fn config(&self) -> &CanonicalizeConfig {
        &self.config
    }

    /// Canonicalizes `circuit` and returns the rebuilt circuit plus run metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::{Circuit, Qubit};
    /// use cqlib_core::compile::transform::Canonicalizer;
    ///
    /// let mut circuit = Circuit::new(1);
    /// circuit.h(Qubit::new(0)).unwrap();
    ///
    /// let result = Canonicalizer::production().run(&circuit).unwrap();
    /// assert_eq!(result.circuit.qubits(), circuit.qubits());
    /// assert!(result.rounds >= 1);
    /// let _changed = result.changed;
    /// ```
    pub fn run(&self, circuit: &Circuit) -> Result<CanonicalizeResult, CompilerError> {
        verify_circuit(circuit, VerifyMode::Input)?;

        if self.config.round_limit() == 0 {
            return Err(CompilerError::InvalidInput(
                "canonicalize round_limit must be greater than zero".to_string(),
            ));
        }

        // A single pass can expose new canonicalization opportunities. For
        // example, parameter simplification can turn `theta - theta` into a
        // fixed zero, which then lets the next pass remove a rotation. The loop
        // therefore proves a stable representation before reporting success.
        let mut current = circuit.clone();
        for round in 1..=self.config.round_limit() {
            let next = CanonicalizeRound::new(&current, &self.config).run()?;
            verify_circuit(
                &next,
                VerifyMode::Output {
                    config: &self.config,
                },
            )?;

            if circuits_equivalent_for_canonicalize(&current, &next) {
                return Ok(CanonicalizeResult {
                    circuit: next,
                    changed: !circuits_equivalent_for_canonicalize(circuit, &current),
                    rounds: round,
                });
            }

            current = next;
        }

        Err(CompilerError::InvariantViolation(format!(
            "canonicalization did not reach a fixed point within {} rounds",
            self.config.round_limit()
        )))
    }
}

// Transformer integration keeps only the common circuit/changed shape; callers
// that need canonicalization rounds should use `Canonicalizer::run` directly.
impl Transformer for Canonicalizer {
    fn transform(&self, circuit: &Circuit) -> Result<TransformResult, CompilerError> {
        let result = self.run(circuit)?;
        Ok(TransformResult {
            circuit: result.circuit,
            changed: result.changed,
        })
    }
}

/// Canonicalizes a circuit using production defaults.
pub fn canonicalize_circuit(circuit: &Circuit) -> Result<CanonicalizeResult, CompilerError> {
    Canonicalizer::production().run(circuit)
}

struct CanonicalizeRound<'a> {
    source: &'a Circuit,
    config: &'a CanonicalizeConfig,
    target: Circuit,
    top_phase: Parameter,
}

impl<'a> CanonicalizeRound<'a> {
    fn new(source: &'a Circuit, config: &'a CanonicalizeConfig) -> Self {
        Self {
            source,
            config,
            target: Circuit::from_qubits(source.qubits()).expect("source qubits are unique"),
            top_phase: source.global_phase(),
        }
    }

    fn run(mut self) -> Result<Circuit, CompilerError> {
        self.top_phase = canonical_parameter(self.top_phase)?;
        let mut top_level = Vec::with_capacity(self.source.operations().len());

        // Top-level `GPhase` operations are not retained as operations. Their
        // phase contribution is accumulated here and materialized into
        // `Circuit::global_phase` after all operations have been rebuilt.
        for operation in self.source.operations() {
            let rewritten = self.rewrite_operation(operation, ScopeKind::TopLevel)?;
            self.top_phase = canonical_parameter(self.top_phase + rewritten.phase)?;
            for operation in rewritten.operations {
                push_operation(&mut top_level, operation, self.config);
            }
        }

        for operation in top_level {
            self.append_top_level(operation)?;
        }

        let phase = canonical_parameter(self.top_phase)?;
        self.target.set_global_phase(phase);
        Ok(self.target)
    }

    fn append_top_level(&mut self, operation: Operation) -> Result<(), CompilerError> {
        let params = operation
            .params
            .iter()
            .map(|param| circuit_param_to_value(&self.target, param))
            .collect::<Result<Vec<_>, _>>()?;

        self.target.append(
            operation.instruction,
            operation.qubits,
            params,
            operation.label.as_deref(),
        )?;
        Ok(())
    }

    fn canonicalize_body(&mut self, body: &[Operation]) -> Result<Vec<Operation>, CompilerError> {
        let mut out = Vec::with_capacity(body.len());
        let mut body_phase = Parameter::from(0.0);

        // Body-local phase is conditional on the enclosing control-flow branch
        // or loop iteration, so it cannot be lifted to circuit global phase.
        // The canonical body representation keeps it as one leading `GPhase`.
        for operation in body {
            let rewritten = self.rewrite_operation(operation, ScopeKind::ControlFlowBody)?;
            body_phase = canonical_parameter(body_phase + rewritten.phase)?;
            for operation in rewritten.operations {
                push_operation(&mut out, operation, self.config);
            }
        }

        body_phase = canonical_parameter(body_phase)?;
        if !parameter_is_exact_zero(&body_phase)? {
            let param = self.intern_parameter(body_phase)?;
            out.insert(
                0,
                Operation {
                    instruction: Instruction::Standard(StandardGate::GPhase),
                    qubits: smallvec![],
                    params: smallvec![param],
                    label: None,
                },
            );
        }

        Ok(out)
    }

    fn rewrite_operation(
        &mut self,
        operation: &Operation,
        scope: ScopeKind,
    ) -> Result<RewriteResult, CompilerError> {
        let mut instruction = operation.instruction.clone();
        if self.config.canonicalizes_instruction_form() {
            instruction = instruction.canonicalize_form();
        }

        if self.config.recurses_control_flow() {
            instruction = self.rewrite_control_flow_instruction(instruction)?;
        }

        let qubits = canonicalize_operation_qubits(&instruction, &operation.qubits, self.config);
        let semantic_params = operation
            .params
            .iter()
            .map(|param| resolve_parameter(self.source, param))
            .collect::<Result<Vec<_>, _>>()?;

        if self.config.folds_gphase()
            && matches!(instruction, Instruction::Standard(StandardGate::GPhase))
        {
            let phase = semantic_params
                .first()
                .cloned()
                .unwrap_or_else(|| Parameter::from(0.0));
            return Ok(RewriteResult::phase(phase));
        }

        if self.config.drops_noops() && is_strict_noop(&instruction, &semantic_params, &qubits)? {
            return Ok(RewriteResult::drop());
        }

        let params = semantic_params
            .into_iter()
            .map(|param| self.intern_parameter(param))
            .collect::<Result<SmallVec<[CircuitParam; 1]>, _>>()?;

        let mut label = operation.label.clone();
        if self.config.canonicalizes_barriers()
            && matches!(instruction, Instruction::Directive(Directive::Barrier))
        {
            label = None;
        }

        let mut operation = Operation {
            instruction,
            qubits,
            params,
            label,
        };

        // The operation-level qubit list is derived from the canonicalized body,
        // not preserved from input. This prevents deleted body no-ops from
        // keeping dead qubits visible to later analysis passes.
        if matches!(
            (&operation.instruction, scope),
            (
                Instruction::ControlFlowGate(_),
                ScopeKind::TopLevel | ScopeKind::ControlFlowBody
            )
        ) {
            operation.qubits = canonical_control_flow_qubits_for_operation(
                &operation.instruction,
                &self.target.qubits(),
            );
        }

        Ok(RewriteResult::keep(operation))
    }

    fn rewrite_control_flow_instruction(
        &mut self,
        instruction: Instruction,
    ) -> Result<Instruction, CompilerError> {
        match instruction {
            Instruction::ControlFlowGate(ControlFlow::IfElse(gate)) => {
                let true_body = self.canonicalize_body(gate.true_body())?;
                let false_body = gate
                    .false_body()
                    .map(|body| self.canonicalize_body(body))
                    .transpose()?;
                Ok(Instruction::ControlFlowGate(ControlFlow::IfElse(
                    IfElseGate::new(gate.condition(), true_body, false_body),
                )))
            }
            Instruction::ControlFlowGate(ControlFlow::WhileLoop(gate)) => {
                let body = self.canonicalize_body(gate.body())?;
                Ok(Instruction::ControlFlowGate(ControlFlow::WhileLoop(
                    WhileLoopGate::new(gate.condition(), body),
                )))
            }
            _ => Ok(instruction),
        }
    }

    fn intern_parameter(&mut self, param: Parameter) -> Result<CircuitParam, CompilerError> {
        parameter_to_circuit_param(&mut self.target, param)
    }
}

#[derive(Debug, Clone, Copy)]
enum ScopeKind {
    TopLevel,
    ControlFlowBody,
}

#[derive(Debug, Clone)]
struct RewriteResult {
    operations: Vec<Operation>,
    phase: Parameter,
}

impl RewriteResult {
    fn keep(operation: Operation) -> Self {
        Self {
            operations: vec![operation],
            phase: Parameter::from(0.0),
        }
    }

    fn drop() -> Self {
        Self {
            operations: Vec::new(),
            phase: Parameter::from(0.0),
        }
    }

    fn phase(phase: Parameter) -> Self {
        Self {
            operations: Vec::new(),
            phase,
        }
    }
}
