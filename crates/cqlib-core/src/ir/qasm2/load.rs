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

//! OpenQASM 2.0 Parser Module
//!
//! This module provides functionality to parse OpenQASM 2.0 quantum programs
//! and convert them into the internal `Circuit` representation.
//!
//! # Example
//!
//! ```rust
//! use cqlib_core::ir::qasm2::load::loads;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     qreg q[2];
//!     h q[0];
//!     cx q[0], q[1];
//! "#;
//!
//! let circuit = loads(qasm).unwrap();
//! assert_eq!(circuit.num_qubits(), 2);
//! ```

use crate::circuit::Circuit;
use crate::circuit::bit::Qubit;
use crate::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use crate::circuit::gate::{Directive, Instruction, StandardGate};
use crate::circuit::param::ParameterValue;

use crate::circuit::parameter::Parameter;
use crate::ir::qasm2::ast::{
    Argument, Expression, OpCode, OpenQASMProgram, Statement, UnaryOpCode,
};
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Built-in qelib1.inc content
const QELIB1: &str = include_str!("qelib1.inc");

#[rustfmt::skip]
mod parser {
    include!(concat!(env!("OUT_DIR"), "/ir/qasm2/parser.rs"));
}

/// Parse OpenQASM 2.0 file and convert to Circuit
pub fn load<P: AsRef<Path>>(path: P) -> Result<Circuit, QasmParseError> {
    let path = path.as_ref();
    let content = fs::read_to_string(path).map_err(|e| QasmParseError::IoError(e.to_string()))?;

    // Pass the parent directory as the base path for includes
    let base_path = path.parent().map(|p| p.to_path_buf());

    parse_qasm_with_path(&content, base_path)
}

/// Parse OpenQASM 2.0 string and convert to Circuit
pub fn loads(source: &str) -> Result<Circuit, QasmParseError> {
    parse_qasm_with_path(source, None)
}

fn parse_qasm_with_path(
    source: &str,
    base_path: Option<PathBuf>,
) -> Result<Circuit, QasmParseError> {
    let parser = parser::MainParser::new();
    let program = match parser.parse(source) {
        Ok(program) => program,
        Err(e) => return Err(QasmParseError::ParseError(format!("{:?}", e))),
    };

    let mut converter = AstToCircuit::new(base_path);
    converter.convert(&program)
}

/// Errors that can occur during OpenQASM parsing
#[derive(Debug, Clone, PartialEq)]
pub enum QasmParseError {
    IoError(String),
    ParseError(String),
    ConversionError(String),
    UndefinedQubit(String),
    UndefinedRegister(String),
    UndefinedGate(String),
    InvalidArgument(String),
    MismatchedQubitCount { expected: usize, actual: usize },
    RecursionLimitExceeded(String),
    EvaluationError(String),
}

impl std::fmt::Display for QasmParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QasmParseError::IoError(s) => write!(f, "IO error: {}", s),
            QasmParseError::ParseError(s) => write!(f, "Parse error: {}", s),
            QasmParseError::ConversionError(s) => write!(f, "Conversion error: {}", s),
            QasmParseError::UndefinedQubit(s) => write!(f, "Undefined qubit: {}", s),
            QasmParseError::UndefinedRegister(s) => write!(f, "Undefined register: {}", s),
            QasmParseError::UndefinedGate(s) => write!(f, "Undefined gate: {}", s),
            QasmParseError::InvalidArgument(s) => write!(f, "Invalid argument: {}", s),
            QasmParseError::MismatchedQubitCount { expected, actual } => {
                write!(
                    f,
                    "Mismatched qubit count: expected {}, got {}",
                    expected, actual
                )
            }
            QasmParseError::RecursionLimitExceeded(s) => {
                write!(f, "Recursion limit exceeded: {}", s)
            }
            QasmParseError::EvaluationError(s) => write!(f, "Evaluation error: {}", s),
        }
    }
}

impl std::error::Error for QasmParseError {}

/// Default maximum recursion depth for gate expansion
const DEFAULT_MAX_RECURSION_DEPTH: usize = 100;

/// Converts OpenQASM AST to Circuit
struct AstToCircuit {
    /// Maps register names to their sizes
    qregs: HashMap<String, i64>,
    /// Tracks the order of register declarations to ensure deterministic qubit mapping
    qreg_order: Vec<String>,
    cregs: HashMap<String, i64>,
    /// Custom gate definitions
    custom_gates: HashMap<String, CustomGateDef>,
    /// Base path for resolving includes
    base_path: Option<PathBuf>,
    /// Cache of parsed included files to avoid expensive re-parsing
    file_cache: HashMap<PathBuf, OpenQASMProgram>,
    /// Current recursion depth for gate expansion
    recursion_depth: usize,
    /// Maximum allowed recursion depth
    max_recursion_depth: usize,
}

/// A custom gate definition that stores either AST or compiled CircuitGate
#[derive(Clone, Debug)]
struct CustomGateDef {
    #[allow(dead_code)]
    name: String,
    params: Vec<String>,
    qubits: Vec<String>,
    /// The AST body for lazy compilation
    body: Vec<Statement>,
    /// The compiled circuit gate - None until compiled, None for opaque gates
    circuit_gate: Option<CircuitGate>,
    /// Whether this is an opaque gate
    is_opaque: bool,
}

impl AstToCircuit {
    fn new(base_path: Option<PathBuf>) -> Self {
        Self {
            qregs: HashMap::new(),
            qreg_order: Vec::new(),
            cregs: HashMap::new(),
            custom_gates: HashMap::new(),
            base_path,
            file_cache: HashMap::new(),
            recursion_depth: 0,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
        }
    }

    fn convert(&mut self, program: &OpenQASMProgram) -> Result<Circuit, QasmParseError> {
        // Phase 1: Discovery
        // Recursively traverse includes to find all qregs and gate definitions
        self.discovery_pass(program)?;

        // Phase 1.5: Compile all custom gates
        // This must happen after all gate definitions are discovered
        self.compile_all_gates()?;

        // Calculate total qubits
        let total_qubits: usize = self.qregs.values().sum::<i64>() as usize;
        let mut circuit = Circuit::new(total_qubits);

        // Create the register start map: reg_name -> global_start_index
        let mut reg_start_map: HashMap<String, usize> = HashMap::new();
        let mut global_idx = 0;

        // Use declaration order for qubit mapping
        for name in &self.qreg_order {
            reg_start_map.insert(name.clone(), global_idx);
            let size = self.qregs[name];
            global_idx += size as usize;
        }

        // Phase 2: Generation
        // Process operations
        self.generation_pass(program, &mut circuit, &reg_start_map)?;

        Ok(circuit)
    }

    /// Compile all custom gate definitions
    fn compile_all_gates(&mut self) -> Result<(), QasmParseError> {
        // Get list of gate names to compile
        let gate_names: Vec<String> = self.custom_gates.keys().cloned().collect();

        for name in gate_names {
            self.compile_gate_if_needed(&name)?;
        }

        Ok(())
    }

    /// Compile a single gate if not already compiled
    fn compile_gate_if_needed(
        &mut self,
        name: &str,
    ) -> Result<Option<CircuitGate>, QasmParseError> {
        // Check recursion depth
        if self.recursion_depth >= self.max_recursion_depth {
            return Err(QasmParseError::RecursionLimitExceeded(format!(
                "Gate expansion depth exceeded limit of {} (compiling {})",
                self.max_recursion_depth, name
            )));
        }

        // Check if already compiled
        if let Some(def) = self.custom_gates.get(name) {
            if def.is_opaque {
                return Ok(None);
            }
            if def.circuit_gate.is_some() {
                return Ok(def.circuit_gate.clone());
            }
        } else {
            return Err(QasmParseError::UndefinedGate(name.to_string()));
        }

        // Clone needed data to avoid borrow issues
        let (params, qubits, body) = if let Some(def) = self.custom_gates.get(name) {
            (def.params.clone(), def.qubits.clone(), def.body.clone())
        } else {
            return Err(QasmParseError::UndefinedGate(name.to_string()));
        };

        // Increment recursion depth
        self.recursion_depth += 1;

        // Ensure dependencies are compiled
        for stmt in &body {
            if let Statement::CustomGate(dep_name, _, _) = stmt {
                if self.custom_gates.contains_key(dep_name) {
                    self.compile_gate_if_needed(dep_name)?;
                }
            }
        }

        // Build the circuit gate
        let circuit_gate = self.build_circuit_gate(name, &params, &qubits, &body)?;

        // Decrement recursion depth
        self.recursion_depth -= 1;

        // Store the compiled result
        if let Some(def) = self.custom_gates.get_mut(name) {
            def.circuit_gate = Some(circuit_gate.clone());
        }

        Ok(Some(circuit_gate))
    }

    fn discovery_pass(&mut self, program: &OpenQASMProgram) -> Result<(), QasmParseError> {
        for stmt in &program.statements {
            match stmt {
                Statement::QReg(name, size) => {
                    if !self.qregs.contains_key(name) {
                        self.qregs.insert(name.clone(), *size);
                        self.qreg_order.push(name.clone());
                    }
                }
                Statement::CReg(name, size) => {
                    self.cregs.insert(name.clone(), *size);
                }
                Statement::GateDecl(data) => {
                    // Store AST for lazy compilation
                    let decl = CustomGateDef {
                        name: data.name.clone(),
                        params: data.params.clone(),
                        qubits: data.qubits.clone(),
                        body: data.body.clone(),
                        circuit_gate: None,
                        is_opaque: false,
                    };
                    self.custom_gates.insert(data.name.clone(), decl);
                }
                Statement::Include(filename) => {
                    // Resolve path
                    let target_path = if let Some(base) = &self.base_path {
                        base.join(filename)
                    } else {
                        PathBuf::from(filename)
                    };

                    if !self.file_cache.contains_key(&target_path) {
                        let content_res = if filename == "qelib1.inc" {
                            Ok(QELIB1.to_string())
                        } else if target_path.exists() {
                            fs::read_to_string(&target_path).map_err(|e| {
                                QasmParseError::IoError(format!("Include {}: {}", filename, e))
                            })
                        } else {
                            Err(QasmParseError::IoError(format!(
                                "Include file not found: {:?}",
                                target_path
                            )))
                        };

                        let content = content_res?;
                        // Use ProgramBodyParser for included files (no version header required)
                        let parser = parser::ProgramBodyParser::new();
                        let included_program = parser.parse(&content).map_err(|e| {
                            QasmParseError::ParseError(format!("In {}: {:?}", filename, e))
                        })?;

                        // Cache the parsed AST
                        self.file_cache
                            .insert(target_path.clone(), included_program.clone());

                        // Recurse
                        self.discovery_pass(&included_program)?;
                    }
                }
                Statement::Opaque(name, params, qubits) => {
                    // Opaque gates have no body - they cannot be expanded
                    let decl = CustomGateDef {
                        name: name.clone(),
                        params: params.clone(),
                        qubits: qubits.clone(),
                        body: vec![],
                        circuit_gate: None,
                        is_opaque: true,
                    };
                    self.custom_gates.insert(name.clone(), decl);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Build a CircuitGate from a gate declaration
    fn build_circuit_gate(
        &self,
        name: &str,
        params: &[String],
        qubits: &[String],
        body: &[Statement],
    ) -> Result<CircuitGate, QasmParseError> {
        // Create a circuit with qubits named after the gate's formal parameters
        let num_qubits = qubits.len();
        let mut gate_circuit = Circuit::new(num_qubits);

        // Map formal qubit names to actual qubit indices
        let mut qubit_map: HashMap<String, Qubit> = HashMap::new();
        for (i, qubit_name) in qubits.iter().enumerate() {
            qubit_map.insert(qubit_name.clone(), Qubit::new(i as u32));
        }

        // Build parameter map for symbol resolution
        // In the gate body, parameter symbols should be bound to circuit parameters
        let mut param_map: HashMap<String, Parameter> = HashMap::new();
        for param_name in params.iter() {
            // Create a parameter symbol for this gate parameter
            let param = Parameter::symbol(param_name);
            param_map.insert(param_name.clone(), param);
        }

        // Process each statement in the gate body
        for stmt in body {
            self.build_gate_body_statement(stmt, &mut gate_circuit, &qubit_map, &param_map)?;
        }

        // Convert to CircuitGate
        let frozen = FrozenCircuit {
            circuit: gate_circuit,
        };
        CircuitGate::new(name, frozen).map_err(|e| QasmParseError::ConversionError(e.to_string()))
    }

    /// Build a single statement in a gate body
    fn build_gate_body_statement(
        &self,
        stmt: &Statement,
        circuit: &mut Circuit,
        qubit_map: &HashMap<String, Qubit>,
        param_map: &HashMap<String, Parameter>,
    ) -> Result<(), QasmParseError> {
        match stmt {
            Statement::CustomGate(name, args, qargs) => {
                // Resolve qubits
                let mut resolved_qubits = Vec::new();
                for arg in qargs {
                    match arg {
                        Argument::Id(qname) => {
                            if let Some(&q) = qubit_map.get(qname) {
                                resolved_qubits.push(q);
                            } else {
                                return Err(QasmParseError::UndefinedQubit(qname.clone()));
                            }
                        }
                        Argument::IndexedId(_, _) => {
                            return Err(QasmParseError::InvalidArgument(
                                "Indexed arguments not supported in gate body".to_string(),
                            ));
                        }
                    }
                }

                // Resolve parameters - convert expressions to Parameters
                let mut resolved_params: SmallVec<[ParameterValue; 3]> = smallvec![];
                for expr in args {
                    let param = self.expr_to_parameter(expr, param_map)?;
                    resolved_params.push(ParameterValue::from(param));
                }

                // Try to find the gate definition
                if let Some(gate_def) = self.custom_gates.get(name) {
                    if let Some(ref cg) = gate_def.circuit_gate {
                        // Add the CircuitGate directly
                        circuit
                            .append(
                                Instruction::CircuitGate(Box::new(cg.clone())),
                                resolved_qubits,
                                resolved_params,
                                None,
                            )
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    } else {
                        // Opaque gate - cannot be added to circuit body
                        return Err(QasmParseError::UndefinedGate(format!(
                            "Opaque gate {} cannot be used in gate body",
                            name
                        )));
                    }
                } else {
                    // Try standard gate
                    self.append_standard_gate(circuit, name, &resolved_params, &resolved_qubits)?;
                }
            }
            Statement::Barrier(args) => {
                let mut resolved_qubits = Vec::new();
                for arg in args {
                    match arg {
                        Argument::Id(qname) => {
                            if let Some(&q) = qubit_map.get(qname) {
                                resolved_qubits.push(q);
                            }
                        }
                        _ => {}
                    }
                }
                circuit
                    .append(
                        Instruction::Directive(Directive::Barrier),
                        resolved_qubits,
                        std::iter::empty::<ParameterValue>(),
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            _ => {
                // Other statements not allowed in gate body
            }
        }
        Ok(())
    }

    /// Convert an expression to a Parameter
    fn expr_to_parameter(
        &self,
        expr: &Expression,
        param_map: &HashMap<String, Parameter>,
    ) -> Result<Parameter, QasmParseError> {
        match expr {
            Expression::Real(v) => Ok(Parameter::from(*v)),
            Expression::Integer(v) => Ok(Parameter::from(*v as f64)),
            Expression::Pi => Ok(Parameter::pi()),
            Expression::Id(name) => {
                if let Some(param) = param_map.get(name) {
                    Ok(param.clone())
                } else {
                    // Try to evaluate as a constant or return error
                    Err(QasmParseError::EvaluationError(format!(
                        "Unknown parameter: {}",
                        name
                    )))
                }
            }
            Expression::BinaryOp(left, op, right) => {
                let l = self.expr_to_parameter(left, param_map)?;
                let r = self.expr_to_parameter(right, param_map)?;
                Ok(match op {
                    OpCode::Add => l + r,
                    OpCode::Sub => l - r,
                    OpCode::Mul => l * r,
                    OpCode::Div => {
                        // Check for division by zero in constant case
                        if let Ok(val) = r.evaluate(&None) {
                            if val == 0.0 {
                                return Err(QasmParseError::EvaluationError(
                                    "Division by zero".to_string(),
                                ));
                            }
                        }
                        l / r
                    }
                    OpCode::Pow => l.pow(&r),
                })
            }
            Expression::UnaryOp(op, expr) => {
                let v = self.expr_to_parameter(expr, param_map)?;
                Ok(match op {
                    UnaryOpCode::Sin => v.sin(),
                    UnaryOpCode::Cos => v.cos(),
                    UnaryOpCode::Tan => v.tan(),
                    UnaryOpCode::Exp => v.exp(),
                    UnaryOpCode::Ln => v.ln(),
                    UnaryOpCode::Sqrt => v.sqrt(),
                    UnaryOpCode::Asin => v.asin(),
                    UnaryOpCode::Acos => v.acos(),
                    UnaryOpCode::Atan => v.atan(),
                    UnaryOpCode::Neg => Parameter::from(0.0) - v,
                })
            }
        }
    }

    /// Append a standard gate to the circuit
    fn append_standard_gate(
        &self,
        circuit: &mut Circuit,
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
    ) -> Result<(), QasmParseError> {
        let p = |i: usize| params.get(i).cloned().unwrap_or(ParameterValue::Fixed(0.0));
        let q = |i: usize| {
            qubits
                .get(i)
                .cloned()
                .ok_or(QasmParseError::MismatchedQubitCount {
                    expected: i + 1,
                    actual: qubits.len(),
                })
        };

        match name {
            "h" | "H" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::H), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "x" | "X" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::X), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "y" | "Y" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::Y), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "z" | "Z" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::Z), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "s" | "S" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::S), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sdg" | "SDG" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::SDG), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "t" | "T" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::T), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "tdg" | "TDG" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::TDG), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sx" | "SX" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::X2P), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sxdg" | "SXDG" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::X2M), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "rx" | "RX" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::RX), [q0], [p(0)], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "ry" | "RY" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::RY), [q0], [p(0)], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "rz" | "RZ" | "p" | "P" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::RZ), [q0], [p(0)], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u1" | "U1" => {
                let q0 = q(0)?;
                circuit
                    .append(
                        Instruction::Standard(StandardGate::Phase),
                        [q0],
                        [p(0)],
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u2" | "U2" => {
                let q0 = q(0)?;
                // U2(phi, lambda) = U(pi/2, phi, lambda)
                let pi_2 = ParameterValue::Param(Parameter::pi() / 2.0);
                circuit
                    .append(
                        Instruction::Standard(StandardGate::U),
                        [q0],
                        [pi_2, p(0), p(1)],
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u3" | "U3" | "u" | "U" => {
                // Built-in U gate (primitive)
                let q0 = q(0)?;
                circuit
                    .append(
                        Instruction::Standard(StandardGate::U),
                        [q0],
                        [p(0), p(1), p(2)],
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cx" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .append(Instruction::Standard(StandardGate::CX), [q0, q1], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "CX" => {
                // Built-in CX gate (primitive)
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .append(Instruction::Standard(StandardGate::CX), [q0, q1], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cy" | "CY" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .append(Instruction::Standard(StandardGate::CY), [q0, q1], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cz" | "CZ" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .append(Instruction::Standard(StandardGate::CZ), [q0, q1], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "swap" | "SWAP" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .append(
                        Instruction::Standard(StandardGate::SWAP),
                        [q0, q1],
                        [],
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "ccx" | "CCX" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                let q2 = q(2)?;
                circuit
                    .append(
                        Instruction::Standard(StandardGate::CCX),
                        [q0, q1, q2],
                        [],
                        None,
                    )
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "id" | "ID" => {
                let q0 = q(0)?;
                circuit
                    .append(Instruction::Standard(StandardGate::I), [q0], [], None)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            _ => {
                return Err(QasmParseError::UndefinedGate(name.to_string()));
            }
        }
        Ok(())
    }

    fn generation_pass(
        &mut self,
        program: &OpenQASMProgram,
        circuit: &mut Circuit,
        reg_start_map: &HashMap<String, usize>,
    ) -> Result<(), QasmParseError> {
        for stmt in &program.statements {
            match stmt {
                Statement::Include(filename) => {
                    let target_path = if let Some(base) = &self.base_path {
                        base.join(filename)
                    } else {
                        PathBuf::from(filename)
                    };

                    // Retrieve from cache instead of re-parsing
                    // Clone the program to avoid borrow issues
                    let included_program = self.file_cache.get(&target_path).cloned();
                    if let Some(included) = included_program {
                        self.generation_pass(&included, circuit, reg_start_map)?;
                    } else {
                        // Should not happen if discovery pass worked correctly
                        return Err(QasmParseError::ConversionError(format!(
                            "Included file not found in cache: {:?}",
                            target_path
                        )));
                    }
                }
                Statement::Barrier(args) => {
                    let qubits = self.resolve_global_args(args, reg_start_map)?;
                    circuit
                        .barrier(qubits)
                        .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                }
                Statement::Reset(arg) => {
                    // Reset can be applied to single qubit or entire register
                    let qubits = self.resolve_global_args_single_or_register(arg, reg_start_map)?;
                    for qubit in qubits {
                        circuit
                            .reset(qubit)
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    }
                }
                Statement::Measure(qarg, carg) => {
                    // Resolve quantum argument (can be single qubit or register)
                    let qubits =
                        self.resolve_global_args_single_or_register(qarg, reg_start_map)?;

                    // Validate classical argument matches in size
                    let creg_indices = self.resolve_creg_indices(carg)?;

                    if qubits.len() != creg_indices.len() {
                        return Err(QasmParseError::InvalidArgument(format!(
                            "Measure qubit count ({}) does not match classical register count ({})",
                            qubits.len(),
                            creg_indices.len()
                        )));
                    }

                    // Apply measurement operations
                    for (_i, qubit) in qubits.iter().enumerate() {
                        // Store measurement result with classical register index
                        // For now, we apply the measurement; classical register tracking would need Circuit support
                        circuit
                            .measure(*qubit)
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    }
                }
                Statement::CustomGate(name, params, args) => {
                    // Convert parameters to ParameterValues (symbolic, not evaluated)
                    let mut param_values: SmallVec<[ParameterValue; 3]> = smallvec![];
                    let empty_param_map: HashMap<String, Parameter> = HashMap::new();
                    for e in params {
                        let param = self.expr_to_parameter(e, &empty_param_map)?;
                        param_values.push(ParameterValue::from(param));
                    }

                    let qubits = self.resolve_global_args(args, reg_start_map)?;

                    // Try to find the gate definition
                    if let Some(gate_def) = self.custom_gates.get(name) {
                        if let Some(ref cg) = gate_def.circuit_gate {
                            // Add the CircuitGate directly - preserves the gate structure
                            circuit
                                .append(
                                    Instruction::CircuitGate(Box::new(cg.clone())),
                                    qubits,
                                    param_values,
                                    None,
                                )
                                .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                        } else {
                            // Opaque gate - cannot be added
                            return Err(QasmParseError::UndefinedGate(format!(
                                "Opaque gate {} cannot be used",
                                name
                            )));
                        }
                    } else {
                        // Try standard gate
                        self.append_standard_gate_to_circuit(
                            circuit,
                            name,
                            &param_values,
                            &qubits,
                        )?;
                    }
                }
                Statement::If(creg, value, _stmt) => {
                    return Err(QasmParseError::ConversionError(format!(
                        "if statements are not yet supported (if ({} == {}) ...)",
                        creg, value
                    )));
                }
                _ => {} // Declarations already handled
            }
        }
        Ok(())
    }

    fn resolve_global_args(
        &self,
        args: &[Argument],
        reg_start_map: &HashMap<String, usize>,
    ) -> Result<Vec<Qubit>, QasmParseError> {
        let mut qubits = Vec::new();
        for arg in args {
            match arg {
                Argument::Id(name) => {
                    // Expand whole register
                    if let Some(size) = self.qregs.get(name) {
                        if let Some(&start_idx) = reg_start_map.get(name) {
                            for i in 0..*size {
                                qubits.push(Qubit::new((start_idx as i64 + i) as u32));
                            }
                        } else {
                            return Err(QasmParseError::UndefinedRegister(name.clone()));
                        }
                    } else {
                        return Err(QasmParseError::UndefinedRegister(name.clone()));
                    }
                }
                Argument::IndexedId(name, idx) => {
                    if let Some(&start_idx) = reg_start_map.get(name) {
                        // Validate index
                        if let Some(size) = self.qregs.get(name) {
                            if *idx < 0 || *idx >= *size {
                                return Err(QasmParseError::UndefinedQubit(format!(
                                    "{}[{}] out of bounds",
                                    name, idx
                                )));
                            }
                            qubits.push(Qubit::new((start_idx as i64 + *idx) as u32));
                        } else {
                            return Err(QasmParseError::UndefinedRegister(name.clone()));
                        }
                    } else {
                        return Err(QasmParseError::UndefinedQubit(format!("{}[{}]", name, idx)));
                    }
                }
            }
        }
        Ok(qubits)
    }

    /// Resolve argument to either single qubit or expanded register
    fn resolve_global_args_single_or_register(
        &self,
        arg: &Argument,
        reg_start_map: &HashMap<String, usize>,
    ) -> Result<Vec<Qubit>, QasmParseError> {
        let mut qubits = Vec::new();
        match arg {
            Argument::Id(name) => {
                // Expand entire register
                if let Some(size) = self.qregs.get(name) {
                    if let Some(&start_idx) = reg_start_map.get(name) {
                        for i in 0..*size {
                            qubits.push(Qubit::new((start_idx as i64 + i) as u32));
                        }
                    } else {
                        return Err(QasmParseError::UndefinedRegister(name.clone()));
                    }
                } else {
                    return Err(QasmParseError::UndefinedRegister(name.clone()));
                }
            }
            Argument::IndexedId(name, idx) => {
                // Single indexed qubit
                if let Some(&start_idx) = reg_start_map.get(name) {
                    if let Some(size) = self.qregs.get(name) {
                        if *idx < 0 || *idx >= *size {
                            return Err(QasmParseError::UndefinedQubit(format!(
                                "{}[{}]",
                                name, idx
                            )));
                        }
                        qubits.push(Qubit::new((start_idx as i64 + *idx) as u32));
                    } else {
                        return Err(QasmParseError::UndefinedRegister(name.clone()));
                    }
                } else {
                    return Err(QasmParseError::UndefinedQubit(format!("{}[{}]", name, idx)));
                }
            }
        }
        Ok(qubits)
    }

    /// Resolve classical register argument to indices
    fn resolve_creg_indices(&self, arg: &Argument) -> Result<Vec<usize>, QasmParseError> {
        let mut indices = Vec::new();
        match arg {
            Argument::Id(name) => {
                // Entire classical register
                if let Some(size) = self.cregs.get(name) {
                    for i in 0..*size {
                        indices.push(i as usize);
                    }
                } else {
                    return Err(QasmParseError::UndefinedRegister(format!(
                        "Classical register '{}' not defined",
                        name
                    )));
                }
            }
            Argument::IndexedId(name, idx) => {
                // Single classical bit
                if self.cregs.get(name).is_none() {
                    return Err(QasmParseError::UndefinedRegister(format!(
                        "Classical register '{}' not defined",
                        name
                    )));
                }
                indices.push(*idx as usize);
            }
        }
        Ok(indices)
    }

    /// Append a standard gate to the circuit using ParameterValues (for top-level circuit)
    fn append_standard_gate_to_circuit(
        &self,
        circuit: &mut Circuit,
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
    ) -> Result<(), QasmParseError> {
        let p = |i: usize| params.get(i).cloned().unwrap_or(ParameterValue::Fixed(0.0));
        let q = |i: usize| {
            qubits
                .get(i)
                .cloned()
                .ok_or(QasmParseError::MismatchedQubitCount {
                    expected: i + 1,
                    actual: qubits.len(),
                })
        };

        match name {
            // OpenQASM 2.0 built-in primitive gates
            "U" => {
                let q0 = q(0)?;
                circuit
                    .u(q0, p(0), p(1), p(2))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "CX" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .cx(q0, q1)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            // Standard gates
            "h" | "H" => {
                let q0 = q(0)?;
                circuit
                    .h(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "x" | "X" => {
                let q0 = q(0)?;
                circuit
                    .x(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "y" | "Y" => {
                let q0 = q(0)?;
                circuit
                    .y(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "z" | "Z" => {
                let q0 = q(0)?;
                circuit
                    .z(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "s" | "S" => {
                let q0 = q(0)?;
                circuit
                    .s(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sdg" | "SDG" => {
                let q0 = q(0)?;
                circuit
                    .sdg(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "t" | "T" => {
                let q0 = q(0)?;
                circuit
                    .t(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "tdg" | "TDG" => {
                let q0 = q(0)?;
                circuit
                    .tdg(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sx" | "SX" => {
                let q0 = q(0)?;
                circuit
                    .x2p(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "sxdg" | "SXDG" => {
                let q0 = q(0)?;
                circuit
                    .x2m(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "rx" | "RX" => {
                let q0 = q(0)?;
                circuit
                    .rx(q0, p(0))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "ry" | "RY" => {
                let q0 = q(0)?;
                circuit
                    .ry(q0, p(0))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "rz" | "RZ" => {
                let q0 = q(0)?;
                circuit
                    .rz(q0, p(0))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "p" | "P" => {
                let q0 = q(0)?;
                circuit
                    .phase(q0, p(0))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u1" | "U1" => {
                let q0 = q(0)?;
                circuit
                    .phase(q0, p(0))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u2" | "U2" => {
                let q0 = q(0)?;
                let pi_2 = ParameterValue::Param(Parameter::pi() / 2.0);
                circuit
                    .u(q0, pi_2, p(0), p(1))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "u3" | "U3" | "u" => {
                let q0 = q(0)?;
                circuit
                    .u(q0, p(0), p(1), p(2))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cx" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .cx(q0, q1)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cy" | "CY" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .cy(q0, q1)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "cz" | "CZ" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .cz(q0, q1)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "swap" | "SWAP" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                circuit
                    .swap(q0, q1)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "ccx" | "CCX" => {
                let q0 = q(0)?;
                let q1 = q(1)?;
                let q2 = q(2)?;
                circuit
                    .ccx(q0, q1, q2)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            "id" | "ID" => {
                let q0 = q(0)?;
                circuit
                    .i(q0)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            }
            _ => {
                return Err(QasmParseError::UndefinedGate(name.to_string()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "./load_test.rs"]
mod load_test;
