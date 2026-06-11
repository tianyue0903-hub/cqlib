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
//! ## Features
//!
//! - **Full OpenQASM 2.0 Support**: Parses all standard statements including quantum registers,
//!   classical registers, gates, measurements, barriers, resets, and conditional execution
//! - **Custom Gate Compilation**: Compiles user-defined gates into internal representations
//! - **Include Processing**: Handles `include` statements for standard gate libraries (qelib1.inc)
//! - **Expression Evaluation**: Evaluates mathematical expressions in gate parameters
//! - **Error Handling**: Provides detailed error messages with line numbers and context
//!
//! ## Architecture
//!
//! The parser works in multiple phases:
//! 1. **Discovery Pass**: Collects register declarations, gate definitions, and processes includes
//! 2. **Compilation Phase**: Compiles custom gates into internal `CircuitGate` representations
//! 3. **Generation Pass**: Converts AST statements into circuit operations
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::ir::qasm2::load::loads;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     creg c[2];
//!     h q[0];
//!     cx q[0], q[1];
//!     measure q[0] -> c[0];
//!     measure q[1] -> c[1];
//! "#;
//!
//! let circuit = loads(qasm).unwrap();
//! assert_eq!(circuit.num_qubits(), 2);
//! ```
//!
//! ## Gate Compilation
//!
//! Custom gates are lazily compiled on first use to detect circular dependencies:
//!
//! ```rust
//! use cqlib_core::ir::qasm2::load::loads;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     qreg q[1];
//!     gate my_gate q { h q; }
//!     my_gate q;
//! "#;
//!
//! let circuit = loads(qasm).unwrap();
//! ```

use crate::circuit::bit::Qubit;
use crate::circuit::circuit_param::{CircuitParam, ParameterValue};
use crate::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use crate::circuit::gate::{Directive, Instruction, StandardGate};
use crate::circuit::operation::Operation;
use crate::circuit::{
    Circuit, ClassicalControlOp, ClassicalExpr, ClassicalType, ClassicalVar, ControlBody, IfOp,
};

use crate::circuit::parameter::Parameter;
use crate::ir::qasm2::ast::{
    Argument, Expression, OpCode, OpenQASMProgram, Statement, UnaryOpCode,
};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Trait for abstracting file system access during OpenQASM parsing.
///
/// This allows the parser to work in different environments:
/// - Real file system (use `FileSystemResolver`)
/// - Memory-only mode (use `NullResolver`)
/// - Custom resolvers for WASM or embedded files
pub trait QasmSourceResolver {
    /// Resolve the content of a file given its path.
    ///
    /// # Arguments
    /// * `path` - The file path to resolve
    ///
    /// # Returns
    /// * `Ok(String)` - The file contents
    /// * `Err(String)` - Error message if resolution fails
    fn resolve_source(&self, path: &Path) -> Result<String, String>;
}

/// Default resolver that uses the real file system.
pub struct FileSystemResolver;

impl QasmSourceResolver for FileSystemResolver {
    fn resolve_source(&self, path: &Path) -> Result<String, String> {
        std::fs::read_to_string(path).map_err(|e| e.to_string())
    }
}

/// A null resolver that fails on any include. Used when loading raw strings without context.
pub struct NullResolver;

impl QasmSourceResolver for NullResolver {
    fn resolve_source(&self, path: &Path) -> Result<String, String> {
        Err(format!(
            "Cannot include files in raw string mode (path: {:?})",
            path
        ))
    }
}

/// Built-in qelib1.inc content
const QELIB1: &str = include_str!("qelib1.inc");

/// Standard gates defined in qelib1.inc that should be treated as native StandardGate
/// instead of being compiled into CircuitGate.
/// These gates have direct StandardGate counterparts and don't need definition expansion.
const QELIB1_STANDARD_GATES: &[&str] = &[
    "cx", "CX", // mapped to StandardGate::CX
    "cy", "CY", // mapped to StandardGate::CY
    "cz", "CZ", // mapped to StandardGate::CZ
    "ccx", "CCX", // mapped to StandardGate::CCX
    "swap", "SWAP", // mapped to StandardGate::SWAP
    "id", "ID", // mapped to StandardGate::I
    "x", "X", // mapped to StandardGate::X
    "y", "Y", // mapped to StandardGate::Y
    "z", "Z", // mapped to StandardGate::Z
    "h", "H", // mapped to StandardGate::H
    "s", "S", // mapped to StandardGate::S
    "sdg", "SDG", // mapped to StandardGate::SDG
    "t", "T", // mapped to StandardGate::T
    "tdg", "TDG", // mapped to StandardGate::TDG
    "rx", "RX", // mapped to StandardGate::RX
    "ry", "RY", // mapped to StandardGate::RY
    "rz", "RZ", // mapped to StandardGate::RZ
    "u1", "U1", // mapped to StandardGate::Phase
    "u2", "U2", // decomposed to U gate
    "u3", "U3", // mapped to StandardGate::U
    // Additional standard gates
    "rxx", "RXX", // mapped to StandardGate::RXX
    "ryy", "RYY", // mapped to StandardGate::RYY
    "rzz", "RZZ", // mapped to StandardGate::RZZ
    "fsim", "FSIM", // mapped to StandardGate::FSIM
    "crx", "CRX", // mapped to StandardGate::CRX
    "cry", "CRY", // mapped to StandardGate::CRY
    "crz", "CRZ", // mapped to StandardGate::CRZ
];

/// Gate names declared by the bundled qelib1.inc file.
///
/// User sources may call these gates after including qelib1.inc, but may not
/// redefine them. OpenQASM identifiers are case-sensitive, so this list uses
/// the exact lowercase spellings present in the bundled file.
const QELIB1_RESERVED_GATE_NAMES: &[&str] = &[
    "u3", "u2", "u1", "cx", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz",
    "cz", "cy", "ch", "ccx", "crz", "cu1", "cu3",
];

#[rustfmt::skip]
mod parser {
    include!(concat!(env!("OUT_DIR"), "/ir/qasm2/parser.rs"));
}

/// Parse OpenQASM 2.0 file and convert to Circuit
pub fn load<P: AsRef<Path>>(path: P) -> Result<Circuit, QasmParseError> {
    let path = path.as_ref();

    // Use FileSystemResolver for file-based loading
    let resolver = Box::new(FileSystemResolver);

    // Get content using the resolver
    let content = resolver
        .resolve_source(path)
        .map_err(QasmParseError::IoError)?;

    // Pass the parent directory as the base path for includes
    let base_path = path.parent().map(|p| p.to_path_buf());

    parse_qasm_with_context(&content, base_path, resolver)
}

/// Parse OpenQASM 2.0 string and convert to Circuit
pub fn loads(source: &str) -> Result<Circuit, QasmParseError> {
    // Use NullResolver for string-based loading (no file includes allowed)
    parse_qasm_with_context(source, None, Box::new(NullResolver))
}

fn parse_qasm_with_context(
    source: &str,
    base_path: Option<PathBuf>,
    resolver: Box<dyn QasmSourceResolver>,
) -> Result<Circuit, QasmParseError> {
    let parser = parser::MainParser::new();
    let program = match parser.parse(source) {
        Ok(program) => program,
        Err(e) => return Err(QasmParseError::ParseError(format!("{:?}", e))),
    };

    let mut converter = AstToCircuit::new(base_path, resolver);
    converter.convert(&program)
}

/// Errors that can occur during OpenQASM parsing and conversion.
///
/// Each variant represents a distinct error category that can occur
/// when processing OpenQASM 2.0 source code.
#[derive(Debug, Clone, PartialEq)]
pub enum QasmParseError {
    /// File system or I/O error (file not found, permission denied, etc.)
    IoError(String),
    /// Syntax error during parsing (invalid QASM syntax)
    ParseError(String),
    /// Semantic error during AST to Circuit conversion
    ConversionError(String),
    /// Reference to undefined quantum register or qubit
    UndefinedQubit(String),
    /// Reference to undefined classical register
    UndefinedRegister(String),
    /// Reference to undefined gate
    UndefinedGate(String),
    /// Attempt to redefine a gate declared by the bundled qelib1.inc file
    ReservedGateName(String),
    /// Invalid argument format or usage
    InvalidArgument(String),
    /// Gate applied to wrong number of qubits
    MismatchedQubitCount { expected: usize, actual: usize },
    /// Gate called with wrong number of parameters
    MismatchedParameterCount { expected: usize, actual: usize },
    /// Gate expansion exceeded maximum recursion depth (circular definition likely)
    RecursionLimitExceeded(String),
    /// Failed to evaluate parameter expression (undefined variable, division by zero, etc.)
    EvaluationError(String),
    /// Circular gate dependency detected (gate calls itself directly or indirectly)
    CircularGateDependency { gate: String, dependency: String },
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
            QasmParseError::ReservedGateName(s) => write!(
                f,
                "Gate name '{}' is reserved by qelib1.inc and cannot be redefined",
                s
            ),
            QasmParseError::InvalidArgument(s) => write!(f, "Invalid argument: {}", s),
            QasmParseError::MismatchedQubitCount { expected, actual } => {
                write!(
                    f,
                    "Mismatched qubit count: expected {}, got {}",
                    expected, actual
                )
            }
            QasmParseError::MismatchedParameterCount { expected, actual } => {
                write!(
                    f,
                    "Mismatched parameter count: expected {}, got {}",
                    expected, actual
                )
            }
            QasmParseError::RecursionLimitExceeded(s) => {
                write!(f, "Recursion limit exceeded: {}", s)
            }
            QasmParseError::EvaluationError(s) => write!(f, "Evaluation error: {}", s),
            QasmParseError::CircularGateDependency { gate, dependency } => {
                write!(
                    f,
                    "Circular gate dependency detected: '{}' depends on '{}'",
                    gate, dependency
                )
            }
        }
    }
}

impl std::error::Error for QasmParseError {}

/// Default maximum recursion depth for gate expansion
const DEFAULT_MAX_RECURSION_DEPTH: usize = 100;

/// Converts OpenQASM AST to internal Circuit representation.
///
/// This struct performs multi-phase conversion:
/// 1. **Discovery**: Collects register declarations and gate definitions
/// 2. **Compilation**: Compiles custom gates into CircuitGate
/// 3. **Generation**: Converts statements to circuit operations
struct AstToCircuit {
    /// Quantum register name -> size mapping
    qregs: HashMap<String, i64>,
    /// Declaration order for deterministic qubit indexing
    qreg_order: Vec<String>,
    /// Classical register name -> size mapping
    cregs: HashMap<String, i64>,
    /// Declaration order for deterministic classical-variable allocation.
    creg_order: Vec<String>,
    /// Classical register name -> circuit-local BitVec variable.
    creg_vars: HashMap<String, ClassicalVar>,
    /// Custom gate definitions (name -> definition)
    custom_gates: HashMap<String, CustomGateDef>,
    /// Base path for resolving relative include paths
    base_path: Option<PathBuf>,
    /// Parsed include cache (path -> AST) to avoid re-parsing
    file_cache: HashMap<PathBuf, OpenQASMProgram>,
    /// Current gate expansion depth (for recursion limiting)
    recursion_depth: usize,
    /// Maximum allowed recursion depth (default: 100)
    max_recursion_depth: usize,
    /// Gates currently being compiled (for cycle detection)
    compiling_gates: HashSet<String>,
    /// File system abstraction
    resolver: Box<dyn QasmSourceResolver>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiscoverySource {
    User,
    BuiltinQelib1,
}

impl AstToCircuit {
    fn new(base_path: Option<PathBuf>, resolver: Box<dyn QasmSourceResolver>) -> Self {
        Self {
            qregs: HashMap::new(),
            qreg_order: Vec::new(),
            cregs: HashMap::new(),
            creg_order: Vec::new(),
            creg_vars: HashMap::new(),
            custom_gates: HashMap::new(),
            base_path,
            file_cache: HashMap::new(),
            recursion_depth: 0,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            compiling_gates: HashSet::new(),
            resolver,
        }
    }

    fn convert(&mut self, program: &OpenQASMProgram) -> Result<Circuit, QasmParseError> {
        // Phase 1: Discovery
        // Recursively traverse includes to find all qregs and gate definitions
        self.discovery_pass(program, DiscoverySource::User)?;

        // Phase 1.5: Compile all custom gates
        // This must happen after all gate definitions are discovered
        self.compile_all_gates()?;

        // Calculate total qubits
        let total_qubits = self.qregs.values().try_fold(0usize, |total, size| {
            total.checked_add(*size as usize).ok_or_else(|| {
                QasmParseError::ConversionError(
                    "Total quantum register size exceeds platform limits".to_string(),
                )
            })
        })?;
        let mut circuit = Circuit::new(total_qubits);

        for name in &self.creg_order {
            let width = self.cregs[name] as u32;
            let ty = ClassicalType::bit_vec(width).ok_or_else(|| {
                QasmParseError::ConversionError(format!(
                    "Classical register '{name}' must have non-zero width"
                ))
            })?;
            let var = circuit.var(ty);
            let zero =
                ClassicalExpr::pack_bits((0..width).map(|_| ClassicalExpr::bit_literal(false)))
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            circuit
                .store(var, zero)
                .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
            self.creg_vars.insert(name.clone(), var);
        }

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

        // Check for circular dependency: if the gate is already being compiled, we have a cycle
        if self.compiling_gates.contains(name) {
            return Err(QasmParseError::CircularGateDependency {
                gate: name.to_string(),
                dependency: name.to_string(),
            });
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

        // Mark this gate as being compiled (for cycle detection)
        self.compiling_gates.insert(name.to_string());

        // Increment recursion depth
        self.recursion_depth += 1;

        let result = (|| {
            // Ensure dependencies are compiled
            for stmt in &body {
                if let Statement::CustomGate(dep_name, _, _) = stmt {
                    if self.custom_gates.contains_key(dep_name) {
                        // Check if this dependency creates a cycle
                        if self.compiling_gates.contains(dep_name) {
                            return Err(QasmParseError::CircularGateDependency {
                                gate: name.to_string(),
                                dependency: dep_name.to_string(),
                            });
                        }
                        self.compile_gate_if_needed(dep_name)?;
                    }
                }
            }

            self.build_circuit_gate(name, &params, &qubits, &body)
        })();

        // Restore compilation state before propagating any error.
        self.recursion_depth -= 1;
        self.compiling_gates.remove(name);

        let circuit_gate = result?;

        // Store the compiled result
        if let Some(def) = self.custom_gates.get_mut(name) {
            def.circuit_gate = Some(circuit_gate.clone());
        }

        Ok(Some(circuit_gate))
    }

    fn discovery_pass(
        &mut self,
        program: &OpenQASMProgram,
        source: DiscoverySource,
    ) -> Result<(), QasmParseError> {
        for stmt in &program.statements {
            match stmt {
                Statement::QReg(name, size) => {
                    if *size <= 0 {
                        return Err(QasmParseError::InvalidArgument(format!(
                            "Quantum register '{name}' must have positive width, got {size}"
                        )));
                    }
                    if !self.qregs.contains_key(name) {
                        self.qregs.insert(name.clone(), *size);
                        self.qreg_order.push(name.clone());
                    }
                }
                Statement::CReg(name, size) => {
                    if *size <= 0 {
                        return Err(QasmParseError::InvalidArgument(format!(
                            "Classical register '{name}' must have positive width, got {size}"
                        )));
                    }
                    if !self.cregs.contains_key(name) {
                        self.cregs.insert(name.clone(), *size);
                        self.creg_order.push(name.clone());
                    }
                }
                Statement::GateDecl(data) => {
                    if source == DiscoverySource::User
                        && QELIB1_RESERVED_GATE_NAMES.contains(&data.name.as_str())
                    {
                        return Err(QasmParseError::ReservedGateName(data.name.clone()));
                    }

                    // Skip standard gates from qelib1.inc - they have direct StandardGate mappings
                    if source == DiscoverySource::BuiltinQelib1
                        && QELIB1_STANDARD_GATES.contains(&data.name.as_str())
                    {
                        continue;
                    }

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
                            // Built-in qelib1.inc content - always available
                            Ok(QELIB1.to_string())
                        } else {
                            // Use resolver for external files
                            self.resolver.resolve_source(&target_path).map_err(|e| {
                                QasmParseError::IoError(format!("Include {}: {}", filename, e))
                            })
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
                        let included_source = if filename == "qelib1.inc" {
                            DiscoverySource::BuiltinQelib1
                        } else {
                            DiscoverySource::User
                        };
                        self.discovery_pass(&included_program, included_source)?;
                    }
                }
                Statement::Opaque(name, params, qubits) => {
                    if source == DiscoverySource::User
                        && QELIB1_RESERVED_GATE_NAMES.contains(&name.as_str())
                    {
                        return Err(QasmParseError::ReservedGateName(name.clone()));
                    }

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
        let frozen = FrozenCircuit::new(gate_circuit);
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
                    let param = Self::expr_to_parameter(expr, param_map)?;
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
                    if let Argument::Id(qname) = arg {
                        if let Some(&q) = qubit_map.get(qname) {
                            resolved_qubits.push(q);
                        }
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
                let l = Self::expr_to_parameter(left, param_map)?;
                let r = Self::expr_to_parameter(right, param_map)?;
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
                    OpCode::Pow => l.pow(r),
                })
            }
            Expression::UnaryOp(op, expr) => {
                let v = Self::expr_to_parameter(expr, param_map)?;
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

    /// Build a standard gate instruction with validated parameters.
    ///
    /// This centralized function handles all standard gate name matching, qubit/parameter count
    /// validation, and special parameter transformations (like U2).
    ///
    /// Returns the Instruction and the validated ParameterValues to use.
    fn build_standard_gate_instruction(
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
    ) -> Result<(Instruction, Vec<ParameterValue>), QasmParseError> {
        // Helper closure for checking qubit and parameter counts
        let check_counts = |req_q: usize, req_p: usize| -> Result<(), QasmParseError> {
            if qubits.len() != req_q {
                return Err(QasmParseError::MismatchedQubitCount {
                    expected: req_q,
                    actual: qubits.len(),
                });
            }
            if params.len() != req_p {
                return Err(QasmParseError::MismatchedParameterCount {
                    expected: req_p,
                    actual: params.len(),
                });
            }
            Ok(())
        };

        match name {
            "u2" | "U2" => {
                check_counts(1, 2)?;
                // U2(phi, lambda) = U(pi/2, phi, lambda)
                let pi_2 = ParameterValue::Param(Parameter::pi() / 2.0);
                let new_params = vec![pi_2, params[0].clone(), params[1].clone()];
                Ok((Instruction::Standard(StandardGate::U), new_params))
            }

            "rx" | "RX" => {
                check_counts(1, 1)?;
                Ok((Instruction::Standard(StandardGate::RX), params.to_vec()))
            }
            "ry" | "RY" => {
                check_counts(1, 1)?;
                Ok((Instruction::Standard(StandardGate::RY), params.to_vec()))
            }
            "rz" | "RZ" => {
                check_counts(1, 1)?;
                Ok((Instruction::Standard(StandardGate::RZ), params.to_vec()))
            }
            "p" | "P" => {
                check_counts(1, 1)?;
                Ok((Instruction::Standard(StandardGate::Phase), params.to_vec()))
            }
            "u1" | "U1" => {
                check_counts(1, 1)?;
                Ok((Instruction::Standard(StandardGate::Phase), params.to_vec()))
            }

            // --- Standard Parametrized Gates (1 qubit, 3 params) ---
            "u3" | "U3" | "u" | "U" => {
                check_counts(1, 3)?;
                Ok((Instruction::Standard(StandardGate::U), params.to_vec()))
            }

            "rxx" | "RXX" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::RXX), params.to_vec()))
            }
            "ryy" | "RYY" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::RYY), params.to_vec()))
            }
            "rzz" | "RZZ" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::RZZ), params.to_vec()))
            }
            "fsim" | "FSIM" => {
                check_counts(2, 2)?;
                Ok((Instruction::Standard(StandardGate::FSIM), params.to_vec()))
            }

            // --- Controlled rotation gates (2 qubits, 1 param) ---
            "crx" | "CRX" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::CRX), params.to_vec()))
            }
            "cry" | "CRY" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::CRY), params.to_vec()))
            }
            "crz" | "CRZ" => {
                check_counts(2, 1)?;
                Ok((Instruction::Standard(StandardGate::CRZ), params.to_vec()))
            }

            "h" | "H" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::H), vec![]))
            }
            "x" | "X" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::X), vec![]))
            }
            "y" | "Y" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::Y), vec![]))
            }
            "z" | "Z" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::Z), vec![]))
            }
            "s" | "S" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::S), vec![]))
            }
            "sdg" | "SDG" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::SDG), vec![]))
            }
            "t" | "T" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::T), vec![]))
            }
            "tdg" | "TDG" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::TDG), vec![]))
            }
            "sx" | "SX" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::X2P), vec![]))
            }
            "sxdg" | "SXDG" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::X2M), vec![]))
            }
            "id" | "ID" | "i" | "I" => {
                check_counts(1, 0)?;
                Ok((Instruction::Standard(StandardGate::I), vec![]))
            }
            "cx" | "CX" => {
                check_counts(2, 0)?;
                Ok((Instruction::Standard(StandardGate::CX), vec![]))
            }
            "cy" | "CY" => {
                check_counts(2, 0)?;
                Ok((Instruction::Standard(StandardGate::CY), vec![]))
            }
            "cz" | "CZ" => {
                check_counts(2, 0)?;
                Ok((Instruction::Standard(StandardGate::CZ), vec![]))
            }
            "swap" | "SWAP" => {
                check_counts(2, 0)?;
                Ok((Instruction::Standard(StandardGate::SWAP), vec![]))
            }
            "ccx" | "CCX" | "toffoli" | "TOFFOLI" => {
                check_counts(3, 0)?;
                Ok((Instruction::Standard(StandardGate::CCX), vec![]))
            }

            _ => Err(QasmParseError::UndefinedGate(name.to_string())),
        }
    }

    /// Append a standard gate to the circuit (used in gate declarations)
    fn append_standard_gate(
        &self,
        circuit: &mut Circuit,
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
    ) -> Result<(), QasmParseError> {
        let (instruction, valid_params) =
            Self::build_standard_gate_instruction(name, params, qubits)?;
        circuit
            .append(instruction, qubits.to_vec(), valid_params, None)
            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
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

                    let creg_name = match carg {
                        Argument::Id(name) | Argument::IndexedId(name, _) => name,
                    };
                    let target = *self
                        .creg_vars
                        .get(creg_name)
                        .ok_or_else(|| QasmParseError::UndefinedRegister(creg_name.clone()))?;

                    if matches!(carg, Argument::Id(_)) {
                        let result = circuit
                            .measure_bits(qubits)
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                        circuit
                            .store(target, result.expr())
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    } else {
                        let result = circuit
                            .measure(qubits[0])
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                        let width = target.ty().width();
                        let measured_index = creg_indices[0] as u32;
                        let mut bits = Vec::with_capacity(width as usize);
                        for index in 0..width {
                            let bit = if index == measured_index {
                                result.expr()
                            } else {
                                ClassicalExpr::extract_bit(target.expr(), index)
                                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?
                            };
                            bits.push(bit);
                        }
                        let packed = ClassicalExpr::pack_bits(bits)
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                        circuit
                            .store(target, packed)
                            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    }
                }
                Statement::CustomGate(name, params, args) => {
                    // Convert parameters to ParameterValues (symbolic, not evaluated)
                    let mut param_values: SmallVec<[ParameterValue; 3]> = smallvec![];
                    let empty_param_map: HashMap<String, Parameter> = HashMap::new();
                    for e in params {
                        let param = Self::expr_to_parameter(e, &empty_param_map)?;
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
                Statement::If(creg, value, stmt) => {
                    let true_body = self.statement_to_operations(stmt, reg_start_map)?;
                    let condition = self.build_condition(creg, *value)?;
                    let if_op = IfOp::new(condition, ControlBody::new(true_body), None)
                        .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
                    circuit
                        .append_control(ClassicalControlOp::If(if_op))
                        .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
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
                let size = self.cregs.get(name).ok_or_else(|| {
                    QasmParseError::UndefinedRegister(format!(
                        "Classical register '{}' not defined",
                        name
                    ))
                })?;
                if *idx < 0 || *idx >= *size {
                    return Err(QasmParseError::InvalidArgument(format!(
                        "Classical register index {name}[{idx}] is out of bounds for width {size}"
                    )));
                }
                indices.push(*idx as usize);
            }
        }
        Ok(indices)
    }

    fn build_condition(
        &self,
        reference: &Argument,
        value: i64,
    ) -> Result<ClassicalExpr, QasmParseError> {
        let (name, index) = match reference {
            Argument::Id(name) => (name, None),
            Argument::IndexedId(name, index) => (name, Some(*index)),
        };
        let var = *self
            .creg_vars
            .get(name)
            .ok_or_else(|| QasmParseError::UndefinedRegister(name.clone()))?;
        let width = var.ty().width();

        if value < 0 {
            return Err(QasmParseError::InvalidArgument(format!(
                "Classical condition value must be non-negative, got {value}"
            )));
        }

        let (lhs, rhs) = if let Some(index) = index {
            if index < 0 || index as u32 >= width {
                return Err(QasmParseError::InvalidArgument(format!(
                    "Classical register index {name}[{index}] is out of bounds for width {width}"
                )));
            }
            if value > 1 {
                return Err(QasmParseError::InvalidArgument(format!(
                    "Single-bit condition value must be 0 or 1, got {value}"
                )));
            }
            (
                ClassicalExpr::extract_bit(var.expr(), index as u32)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?,
                ClassicalExpr::bit_literal(value == 1),
            )
        } else {
            let value = value as u128;
            if width < 128 && value >= (1u128 << width) {
                return Err(QasmParseError::InvalidArgument(format!(
                    "Condition value {value} does not fit classical register '{name}' width {width}"
                )));
            }
            (
                ClassicalExpr::bit_vec_to_uint(var.expr())
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?,
                ClassicalExpr::uint_literal(width, value)
                    .map_err(|e| QasmParseError::ConversionError(e.to_string()))?,
            )
        };

        ClassicalExpr::eq(lhs, rhs).map_err(|e| QasmParseError::ConversionError(e.to_string()))
    }

    /// Convert a single Statement to Operations for use in if-else body.
    /// Note: Symbolic parameters are NOT allowed in conditional bodies per OpenQASM 2.0 spec.
    fn statement_to_operations(
        &mut self,
        stmt: &Statement,
        reg_start_map: &HashMap<String, usize>,
    ) -> Result<Vec<Operation>, QasmParseError> {
        let mut operations = Vec::new();

        match stmt {
            Statement::CustomGate(name, params, args) => {
                // Convert parameters - symbolic parameters are NOT allowed in if bodies
                let mut param_values: SmallVec<[ParameterValue; 3]> = smallvec![];
                let empty_param_map: HashMap<String, Parameter> = HashMap::new();
                for e in params {
                    let param = Self::expr_to_parameter(e, &empty_param_map)?;
                    param_values.push(ParameterValue::from(param));
                }

                let qubits = self.resolve_global_args(args, reg_start_map)?;

                // Try to find the gate definition
                if let Some(gate_def) = self.custom_gates.get(name) {
                    if let Some(ref cg) = gate_def.circuit_gate {
                        // Strict conversion: symbolic parameters are not allowed
                        let mut circuit_params = SmallVec::new();
                        for pv in &param_values {
                            match pv {
                                ParameterValue::Fixed(v) => {
                                    circuit_params.push(CircuitParam::Fixed(*v))
                                }
                                ParameterValue::Param(p) => {
                                    return Err(QasmParseError::ConversionError(format!(
                                        "Symbolic parameters (like '{}') are not allowed in conditional 'if' statements. Only constants are permitted.",
                                        p
                                    )));
                                }
                            }
                        }
                        operations.push(Operation {
                            instruction: Instruction::CircuitGate(Box::new(cg.clone())),
                            qubits: qubits.into(),
                            params: circuit_params,
                            label: None,
                        });
                        return Ok(operations);
                    }
                }

                self.append_standard_gate_to_operation(
                    name,
                    &param_values,
                    &qubits,
                    &mut operations,
                )?;
            }
            Statement::Barrier(args) => {
                let qubits = self.resolve_global_args(args, reg_start_map)?;
                operations.push(Operation {
                    instruction: Instruction::Directive(Directive::Barrier),
                    qubits: qubits.into(),
                    params: smallvec![],
                    label: None,
                });
            }
            Statement::Reset(arg) => {
                let qubits = self.resolve_global_args_single_or_register(arg, reg_start_map)?;
                for qubit in qubits {
                    operations.push(Operation {
                        instruction: Instruction::Directive(Directive::Reset),
                        qubits: smallvec![qubit],
                        params: smallvec![],
                        label: None,
                    });
                }
            }
            Statement::Measure(_, _) => {
                return Err(QasmParseError::ConversionError(
                    "Measurement in an OpenQASM 2.0 conditional body is not supported".to_string(),
                ));
            }
            _ => {
                return Err(QasmParseError::ConversionError(format!(
                    "Unsupported statement type in if-body: {:?}",
                    stmt
                )));
            }
        }

        Ok(operations)
    }

    /// Converts a ParameterValue to a CircuitParam for if-statement bodies.
    /// Returns an error for symbolic parameters (OpenQASM 2.0 strict compliance).
    fn param_value_to_circuit_param(pv: ParameterValue) -> Result<CircuitParam, QasmParseError> {
        match pv {
            ParameterValue::Fixed(v) => Ok(CircuitParam::Fixed(v)),
            ParameterValue::Param(p) => Err(QasmParseError::ConversionError(format!(
                "Symbolic parameters (like '{}') are not allowed in conditional 'if' statements. Only constants are permitted.",
                p
            ))),
        }
    }

    /// Append a standard gate to operations (helper for if-statement)
    /// Strictly enforces that only constant parameters are allowed.
    fn append_standard_gate_to_operation(
        &self,
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
        operations: &mut Vec<Operation>,
    ) -> Result<(), QasmParseError> {
        let (instruction, valid_params) =
            Self::build_standard_gate_instruction(name, params, qubits)?;

        // Strict conversion: symbolic parameters are not allowed in if bodies
        let circuit_params: SmallVec<[CircuitParam; 1]> = valid_params
            .into_iter()
            .map(Self::param_value_to_circuit_param)
            .collect::<Result<_, _>>()?;

        operations.push(Operation {
            instruction,
            qubits: qubits.into(),
            params: circuit_params,
            label: None,
        });
        Ok(())
    }

    /// Append a standard gate to the circuit using ParameterValues (for top-level circuit)
    fn append_standard_gate_to_circuit(
        &self,
        circuit: &mut Circuit,
        name: &str,
        params: &[ParameterValue],
        qubits: &[Qubit],
    ) -> Result<(), QasmParseError> {
        let (instruction, valid_params) =
            Self::build_standard_gate_instruction(name, params, qubits)?;
        circuit
            .append(instruction, qubits.to_vec(), valid_params, None)
            .map_err(|e| QasmParseError::ConversionError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "./load_test.rs"]
mod load_test;
