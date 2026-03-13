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

//! Template rule loading and registration.

use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::ParameterValue;
use crate::circuit::{Circuit, Qubit};
use crate::compile::error::CompileError;
use crate::compile::graph::{CommutationView, GateGraph};
use crate::compile::prepared::{PreparedCircuit, preprocess_circuit};
use crate::ir::qcis_loads;
use serde::Deserialize;
use std::fs;

/// Embedded default 1q/2q templates.
const DEFAULT_TEMPLATES_JSON: &str = include_str!("default_templates.json");

/// Validated template plus cached graph data for repeated optimization runs.
#[derive(Debug, Clone)]
pub(crate) struct CompiledTemplate {
    pub(crate) prepared: PreparedCircuit,
    pub(crate) graph: GateGraph,
    pub(crate) view: CommutationView,
    pub(crate) node_subsets: Vec<Vec<usize>>,
}

/// Reusable template storage and loading entrypoint.
#[derive(Debug, Clone, Default)]
pub struct TemplateLibrary {
    templates: Vec<Circuit>,
    compiled_templates: Vec<CompiledTemplate>,
}

impl TemplateLibrary {
    /// Creates an empty template library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a library from embedded default template rules.
    pub fn with_default_rules() -> Result<Self, CompileError> {
        Self::from_json_str(DEFAULT_TEMPLATES_JSON)
    }

    /// Creates a library from one JSON template string.
    pub fn from_json_str(content: &str) -> Result<Self, CompileError> {
        let templates = load_templates_from_json_string(content)?;
        let mut library = Self::new();
        library.register_rules(templates)?;
        Ok(library)
    }

    /// Creates a library from one QCIS template string.
    pub fn from_qcis_str(content: &str) -> Result<Self, CompileError> {
        let templates = load_qcis_templates_from_string(content)?;
        let mut library = Self::new();
        library.register_rules(templates)?;
        Ok(library)
    }

    /// Creates a library from a `.json` or `.qcis` template file.
    pub fn from_template_file(template_file: &str) -> Result<Self, CompileError> {
        let content = fs::read_to_string(template_file).map_err(|err| {
            CompileError::Internal(format!(
                "failed to read template file {}: {}",
                template_file, err
            ))
        })?;

        let lower = template_file.to_ascii_lowercase();
        if lower.ends_with(".json") {
            Self::from_json_str(&content)
        } else if lower.ends_with(".qcis") {
            Self::from_qcis_str(&content)
        } else {
            Err(CompileError::Internal(format!(
                "unsupported template file extension for {}",
                template_file
            )))
        }
    }

    /// Registers one template rule after compile-layer validation.
    pub fn register_rule(&mut self, template: Circuit) -> Result<(), CompileError> {
        let compiled = compile_template(&template)?;
        self.templates.push(template);
        self.compiled_templates.push(compiled);
        Ok(())
    }

    /// Registers multiple template rules transactionally.
    pub fn register_rules<I>(&mut self, templates: I) -> Result<(), CompileError>
    where
        I: IntoIterator<Item = Circuit>,
    {
        let mut new_templates = Vec::new();
        let mut compiled_templates = Vec::new();
        for template in templates {
            compiled_templates.push(compile_template(&template)?);
            new_templates.push(template);
        }

        self.templates.extend(new_templates);
        self.compiled_templates.extend(compiled_templates);
        Ok(())
    }

    /// Returns the number of registered template rules.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Returns whether this library is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Returns immutable template circuits in registration order.
    pub fn templates(&self) -> &[Circuit] {
        &self.templates
    }

    pub(crate) fn compiled_templates(&self) -> &[CompiledTemplate] {
        &self.compiled_templates
    }
}

pub(crate) fn compile_template(template: &Circuit) -> Result<CompiledTemplate, CompileError> {
    let prepared = preprocess_circuit(template)?;
    let graph = GateGraph::from_prepared(&prepared)?;
    let view = graph.commutation_view()?;
    let node_subsets = enumerate_template_node_subsets(graph.size());

    Ok(CompiledTemplate {
        prepared,
        graph,
        view,
        node_subsets,
    })
}

/// JSON template file wrapper.
#[derive(Debug, Clone, Deserialize)]
struct TemplateFile {
    /// Optional version tag.
    version: Option<u32>,
    /// Template definitions.
    templates: Vec<TemplateDefinition>,
}

/// One JSON template definition.
#[derive(Debug, Clone, Deserialize)]
struct TemplateDefinition {
    /// Optional human-readable template name.
    #[allow(dead_code)]
    name: Option<String>,
    /// Gate sequence.
    gates: Vec<GateDefinition>,
}

/// One gate in JSON template definition.
#[derive(Debug, Clone, Deserialize)]
struct GateDefinition {
    /// Gate mnemonic.
    gate: String,
    /// Gate qubit list.
    qubits: Vec<usize>,
    /// Optional gate parameters.
    params: Option<Vec<f64>>,
}

/// Enumerates template node subsets used for substitution matching.
fn enumerate_template_node_subsets(template_size: usize) -> Vec<Vec<usize>> {
    const MAX_EXHAUSTIVE_TEMPLATE_SIZE: usize = 10;

    if template_size == 0 {
        return Vec::new();
    }
    if template_size > MAX_EXHAUSTIVE_TEMPLATE_SIZE {
        return vec![(0..template_size).collect()];
    }

    let mut subsets = Vec::<Vec<usize>>::new();
    let total = 1usize << template_size;
    for mask in 1..total {
        let mut subset = Vec::new();
        for bit in 0..template_size {
            if (mask >> bit) & 1 == 1 {
                subset.push(bit);
            }
        }
        subsets.push(subset);
    }
    subsets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    subsets
}

/// Loads templates from one JSON content string.
fn load_templates_from_json_string(content: &str) -> Result<Vec<Circuit>, CompileError> {
    let file: TemplateFile = serde_json::from_str(content)
        .map_err(|err| CompileError::Internal(format!("invalid template JSON: {}", err)))?;
    if let Some(version) = file.version
        && version != 1
    {
        return Err(CompileError::Internal(format!(
            "unsupported template JSON version {}",
            version
        )));
    }

    let mut templates = Vec::with_capacity(file.templates.len());
    for (idx, template) in file.templates.iter().enumerate() {
        if template.gates.is_empty() {
            return Err(CompileError::Internal(format!(
                "template at index {} has no gates",
                idx
            )));
        }
        templates.push(build_template_circuit_from_defs(&template.gates)?);
    }
    Ok(templates)
}

/// Loads templates from one QCIS template string.
fn load_qcis_templates_from_string(content: &str) -> Result<Vec<Circuit>, CompileError> {
    let mut templates = Vec::new();
    for chunk in split_qcis_templates(content) {
        let circuit = qcis_loads(&chunk)
            .map_err(|err| CompileError::Internal(format!("invalid QCIS template: {}", err)))?;
        templates.push(circuit);
    }
    Ok(templates)
}

/// Splits a QCIS text into per-template chunks using `---` separators.
fn split_qcis_templates(content: &str) -> Vec<String> {
    let mut chunks = Vec::<String>::new();
    let mut cur = Vec::<String>::new();
    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with("---") {
            if !cur.is_empty() {
                chunks.push(cur.join("\n"));
                cur.clear();
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        cur.push(line.to_string());
    }
    if !cur.is_empty() {
        chunks.push(cur.join("\n"));
    }
    chunks
}

/// Builds one template circuit from JSON gate definitions.
fn build_template_circuit_from_defs(gates: &[GateDefinition]) -> Result<Circuit, CompileError> {
    let mut max_q = None::<usize>;
    for gate in gates {
        for &q in &gate.qubits {
            max_q = Some(max_q.map_or(q, |m| m.max(q)));
        }
    }
    let width = max_q.map_or(0usize, |v| v + 1);
    let mut circuit = Circuit::new(width);

    for gate in gates {
        let std_gate = parse_standard_gate(&gate.gate)?;
        if std_gate.num_qubits() > 2 {
            return Err(CompileError::Internal(format!(
                "template gate {} exceeds phase-1 2q scope",
                gate.gate
            )));
        }
        if gate.qubits.len() != std_gate.num_qubits() {
            return Err(CompileError::Internal(format!(
                "gate {} expects {} qubits, got {}",
                gate.gate,
                std_gate.num_qubits(),
                gate.qubits.len()
            )));
        }

        let params = gate.params.clone().unwrap_or_default();
        if params.len() != std_gate.num_params() {
            return Err(CompileError::Internal(format!(
                "gate {} expects {} params, got {}",
                gate.gate,
                std_gate.num_params(),
                params.len()
            )));
        }

        let qubits: Vec<Qubit> = gate.qubits.iter().map(|&q| Qubit::new(q as u32)).collect();
        let pvals: Vec<ParameterValue> = params.into_iter().map(ParameterValue::Fixed).collect();
        circuit.append(Instruction::Standard(std_gate), qubits, pvals, None)?;
    }

    Ok(circuit)
}

/// Parses one standard gate name from JSON.
fn parse_standard_gate(name: &str) -> Result<StandardGate, CompileError> {
    let upper = name.trim().to_ascii_uppercase();
    let gate = match upper.as_str() {
        "I" => StandardGate::I,
        "H" => StandardGate::H,
        "RX" => StandardGate::RX,
        "RXX" => StandardGate::RXX,
        "RXY" => StandardGate::RXY,
        "RY" => StandardGate::RY,
        "RYY" => StandardGate::RYY,
        "RZ" => StandardGate::RZ,
        "RZX" => StandardGate::RZX,
        "RZZ" => StandardGate::RZZ,
        "S" => StandardGate::S,
        "SD" | "SDG" => StandardGate::SDG,
        "SWAP" => StandardGate::SWAP,
        "T" => StandardGate::T,
        "TD" | "TDG" => StandardGate::TDG,
        "U" => StandardGate::U,
        "X" => StandardGate::X,
        "XY" => StandardGate::XY,
        "X2P" => StandardGate::X2P,
        "X2M" => StandardGate::X2M,
        "XY2P" => StandardGate::XY2P,
        "XY2M" => StandardGate::XY2M,
        "Y" => StandardGate::Y,
        "Y2P" => StandardGate::Y2P,
        "Y2M" => StandardGate::Y2M,
        "Z" => StandardGate::Z,
        "P" | "PHASE" => StandardGate::Phase,
        "CX" => StandardGate::CX,
        "CCX" => StandardGate::CCX,
        "CY" => StandardGate::CY,
        "CZ" => StandardGate::CZ,
        "CRX" => StandardGate::CRX,
        "CRY" => StandardGate::CRY,
        "CRZ" => StandardGate::CRZ,
        "FSIM" => StandardGate::FSIM,
        _ => {
            return Err(CompileError::Internal(format!(
                "unsupported template gate {}",
                name
            )));
        }
    };
    Ok(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn simple_template() -> Circuit {
        let mut template = Circuit::new(1);
        template.h(Qubit::new(0)).unwrap();
        template.h(Qubit::new(0)).unwrap();
        template
    }

    fn invalid_three_qubit_template() -> Circuit {
        let mut template = Circuit::new(3);
        template
            .append(
                Instruction::Standard(StandardGate::CCX),
                [Qubit::new(0), Qubit::new(1), Qubit::new(2)],
                [],
                None,
            )
            .unwrap();
        template
    }

    fn temp_template_path(extension: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("cqlib_template_{unique}.{extension}"))
    }

    #[test]
    fn test_template_library_default_rule_loading() {
        let library = TemplateLibrary::with_default_rules().unwrap();
        assert!(library.template_count() >= 1);
    }

    #[test]
    fn test_template_library_from_qcis_string() {
        let library = TemplateLibrary::from_qcis_str("H Q0\n---\nCZ Q0 Q1\n").unwrap();
        assert_eq!(library.template_count(), 2);
    }

    #[test]
    fn test_template_library_register_rules_is_transactional() {
        let mut library = TemplateLibrary::new();
        library.register_rule(simple_template()).unwrap();

        let valid = simple_template();
        let invalid = invalid_three_qubit_template();
        assert!(library.register_rules(vec![valid, invalid]).is_err());
        assert_eq!(library.template_count(), 1);
    }

    #[test]
    fn test_template_library_register_rule_rejects_invalid_template() {
        let mut library = TemplateLibrary::new();
        assert!(
            library
                .register_rule(invalid_three_qubit_template())
                .is_err()
        );
        assert!(library.is_empty());
    }

    #[test]
    fn test_template_library_from_template_file_json() {
        let path = temp_template_path("json");
        fs::write(
            &path,
            r#"{"version":1,"templates":[{"gates":[{"gate":"H","qubits":[0]}]}]}"#,
        )
        .unwrap();

        let library = TemplateLibrary::from_template_file(path.to_str().unwrap()).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(library.template_count(), 1);
    }
}
