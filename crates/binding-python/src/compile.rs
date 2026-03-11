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

//! Python bindings for compile-time mapping and routing APIs.
//!
//! This module exposes hardware topology modeling and mapper entry points to the
//! `cqlib.compiler` Python namespace. It provides:
//! - `Topology` construction and connectivity queries.
//! - `SabreConfig` configuration for hybrid VF2/SABRE flow.
//! - Standalone VF2 helpers and the full `map_with_vf2_sabre` API.
//!
//! The implementation intentionally keeps data conversion explicit so errors can
//! be mapped to Python `ValueError` with actionable messages.

use crate::circuit::PyCircuit;
use crate::device::topology::PyTopology;
use cqlib_core::compile::{
    map_with_vf2_sabre, FidelityMap, SabreConfig, TemplateMatching as CoreTemplateMatching,
    TemplateOptimization as CoreTemplateOptimization, Vf2CandidateOptions, Vf2Mapping, Vf2Policy,
    Vf2ScoreWeights,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Python wrapper for SABRE configuration.
#[pyclass(name = "SabreConfig", module = "cqlib.compiler")]
#[derive(Clone, Debug)]
pub struct PySabreConfig {
    /// Internal core configuration object.
    pub(crate) inner: SabreConfig,
}

#[pymethods]
impl PySabreConfig {
    /// Creates a SABRE configuration object.
    ///
    /// Args:
    ///     vf2_policy (str): `direct_then_sabre`, `initial_only`, or `disabled`.
    ///     field_mode (bool): Enables field-aware swap ranking.
    ///     size_e (int): SABRE look-ahead window size.
    ///     w (float): Look-ahead weight.
    ///     decay_coff (float): Repeated-swap decay coefficient.
    ///     decay_reset_time (int): Steps before decay reset.
    ///     greedy_strategy (int): Internal greedy strategy id.
    ///     initial_iterations (int): Initial layout sampling iterations.
    ///     repeat_iterations (int): Alternating refinement iterations.
    ///     swap_iterations (int): Swap-sampling iterations per stage.
    ///     seed (int): RNG seed; `-1` means random seed.
    ///     vf2_seed_top_k (int): Number of ranked VF2 seed layouts to use before random fill.
    ///     vf2_seed_weight_fidelity (float): VF2 seed scoring weight for fidelity fit.
    ///     vf2_seed_weight_topology (float): VF2 seed scoring weight for topology fit.
    ///     vf2_seed_weight_gate_distribution (float): VF2 seed scoring weight for gate distribution fit.
    ///     swap_fidelity_weight (float): Weight of SWAP-edge fidelity penalty in local scoring.
    ///     gate_cost_weight (float): Weight of gate-count cost in global objective.
    ///     predicted_fidelity_weight (float): Weight of predicted fidelity loss in global objective.
    ///
    /// Raises:
    ///     ValueError: If `vf2_policy` is not recognized.
    #[new]
    #[pyo3(signature = (
        vf2_policy = "direct_then_sabre".to_string(),
        field_mode = true,
        size_e = 20,
        w = 0.5,
        decay_coff = 0.001,
        decay_reset_time = 5,
        greedy_strategy = 3,
        initial_iterations = 1,
        repeat_iterations = 1,
        swap_iterations = 1,
        seed = -1,
        vf2_seed_top_k = 8,
        vf2_seed_weight_fidelity = 0.5,
        vf2_seed_weight_topology = 0.3,
        vf2_seed_weight_gate_distribution = 0.2,
        swap_fidelity_weight = 0.25,
        gate_cost_weight = 1.0,
        predicted_fidelity_weight = 0.1,
    ))]
    fn new(
        vf2_policy: String,
        field_mode: bool,
        size_e: usize,
        w: f64,
        decay_coff: f64,
        decay_reset_time: usize,
        greedy_strategy: usize,
        initial_iterations: usize,
        repeat_iterations: usize,
        swap_iterations: usize,
        seed: i64,
        vf2_seed_top_k: usize,
        vf2_seed_weight_fidelity: f64,
        vf2_seed_weight_topology: f64,
        vf2_seed_weight_gate_distribution: f64,
        swap_fidelity_weight: f64,
        gate_cost_weight: f64,
        predicted_fidelity_weight: f64,
    ) -> PyResult<Self> {
        let policy = parse_vf2_policy(&vf2_policy)?;
        Ok(Self {
            inner: SabreConfig {
                vf2_policy: policy,
                vf2_seed_top_k,
                vf2_seed_weights: Vf2ScoreWeights {
                    fidelity: vf2_seed_weight_fidelity,
                    topology: vf2_seed_weight_topology,
                    gate_distribution: vf2_seed_weight_gate_distribution,
                },
                field_mode,
                size_e,
                w,
                decay_coff,
                decay_reset_time,
                greedy_strategy,
                initial_iterations,
                repeat_iterations,
                swap_iterations,
                swap_fidelity_weight,
                gate_cost_weight,
                predicted_fidelity_weight,
                seed,
            },
        })
    }

    /// Returns a compact debug representation.
    fn __repr__(&self) -> String {
        let policy = match self.inner.vf2_policy {
            Vf2Policy::DirectThenSabre => "direct_then_sabre",
            Vf2Policy::InitialOnly => "initial_only",
            Vf2Policy::Disabled => "disabled",
        };
        format!(
            "SabreConfig(vf2_policy='{}', vf2_seed_top_k={}, vf2_seed_weight_fidelity={}, vf2_seed_weight_topology={}, vf2_seed_weight_gate_distribution={}, field_mode={}, size_e={}, w={}, decay_coff={}, decay_reset_time={}, greedy_strategy={}, initial_iterations={}, repeat_iterations={}, swap_iterations={}, swap_fidelity_weight={}, gate_cost_weight={}, predicted_fidelity_weight={}, seed={})",
            policy,
            self.inner.vf2_seed_top_k,
            self.inner.vf2_seed_weights.fidelity,
            self.inner.vf2_seed_weights.topology,
            self.inner.vf2_seed_weights.gate_distribution,
            self.inner.field_mode,
            self.inner.size_e,
            self.inner.w,
            self.inner.decay_coff,
            self.inner.decay_reset_time,
            self.inner.greedy_strategy,
            self.inner.initial_iterations,
            self.inner.repeat_iterations,
            self.inner.swap_iterations,
            self.inner.swap_fidelity_weight,
            self.inner.gate_cost_weight,
            self.inner.predicted_fidelity_weight,
            self.inner.seed,
        )
    }
}

/// Python wrapper for template-matching API.
#[pyclass(name = "TemplateMatching", module = "cqlib.compiler")]
#[derive(Clone, Debug, Default)]
pub struct PyTemplateMatching;

#[pymethods]
impl PyTemplateMatching {
    /// Creates a new template matcher.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Executes template matching and returns match pairs and qubit mapping.
    ///
    /// Args:
    ///     circuit (Circuit): Circuit to match.
    ///     template (Circuit): Template pattern circuit.
    ///     qubit_fixing_cnt (Optional[int]): Optional matching heuristic knob.
    ///     prune_depth (Optional[int]): Optional prune depth.
    ///     prune_width (Optional[int]): Optional prune width.
    ///
    /// Returns:
    ///     List[Tuple[List[Tuple[int, int]], List[int]]]: Match list.
    ///
    /// Raises:
    ///     ValueError: If compile constraints or matching fails.
    #[pyo3(signature = (circuit, template, qubit_fixing_cnt = None, prune_depth = None, prune_width = None))]
    fn run(
        &self,
        circuit: &PyCircuit,
        template: &PyCircuit,
        qubit_fixing_cnt: Option<usize>,
        prune_depth: Option<usize>,
        prune_width: Option<usize>,
    ) -> PyResult<Vec<(Vec<(usize, usize)>, Vec<usize>)>> {
        let prune_param = parse_prune_params(prune_depth, prune_width);
        let matches = CoreTemplateMatching::run(
            &circuit.inner,
            &template.inner,
            qubit_fixing_cnt,
            prune_param,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(matches
            .into_iter()
            .map(|m| (m.match_pairs, m.qubit_mapping))
            .collect())
    }

    /// Returns a compact debug representation.
    fn __repr__(&self) -> String {
        "TemplateMatching()".to_string()
    }
}

/// Python wrapper for template optimization API.
#[pyclass(name = "TemplateOptimization", module = "cqlib.compiler")]
#[derive(Clone, Debug)]
pub struct PyTemplateOptimization {
    /// Internal optimizer object.
    pub(crate) inner: CoreTemplateOptimization,
}

#[pymethods]
impl PyTemplateOptimization {
    /// Creates a template optimizer.
    ///
    /// Args:
    ///     template_list (Optional[List[Circuit]]): Explicit template circuits.
    ///     qubit_fixing_cnt (Optional[int]): Optional matching heuristic knob.
    ///     prune_depth (Optional[int]): Optional prune depth.
    ///     prune_width (Optional[int]): Optional prune width.
    ///     template_file (Optional[str]): Optional template file path (.json or .qcis).
    ///
    /// Raises:
    ///     ValueError: If both template_list and template_file are set, or template loading fails.
    #[new]
    #[pyo3(signature = (template_list = None, qubit_fixing_cnt = None, prune_depth = None, prune_width = None, template_file = None))]
    fn new(
        py: Python<'_>,
        template_list: Option<&Bound<'_, PyList>>,
        qubit_fixing_cnt: Option<usize>,
        prune_depth: Option<usize>,
        prune_width: Option<usize>,
        template_file: Option<String>,
    ) -> PyResult<Self> {
        if template_list.is_some() && template_file.is_some() {
            return Err(PyValueError::new_err(
                "provide either template_list or template_file, not both",
            ));
        }

        let prune_param = parse_prune_params(prune_depth, prune_width);
        let inner = match (template_list, template_file) {
            (Some(list), None) => {
                let mut templates = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let circuit_obj: Py<PyCircuit> = item.extract()?;
                    let circuit = circuit_obj.bind(py).borrow().inner.clone();
                    templates.push(circuit);
                }
                CoreTemplateOptimization::new(templates, qubit_fixing_cnt, prune_param)
            }
            (None, Some(path)) => {
                CoreTemplateOptimization::from_template_file(&path, qubit_fixing_cnt, prune_param)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
            (None, None) => {
                CoreTemplateOptimization::with_default_templates(qubit_fixing_cnt, prune_param)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
            (Some(_), Some(_)) => unreachable!(),
        };

        Ok(Self { inner })
    }

    /// Executes one optimization pass.
    ///
    /// Args:
    ///     circuit (Circuit): Circuit to optimize.
    ///
    /// Returns:
    ///     Circuit: Optimized circuit.
    ///
    /// Raises:
    ///     ValueError: If optimization fails.
    fn execute(&self, circuit: &PyCircuit) -> PyResult<PyCircuit> {
        self.inner
            .execute(&circuit.inner)
            .map(PyCircuit::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Executes optimization iteratively.
    ///
    /// Args:
    ///     circuit (Circuit): Circuit to optimize.
    ///     max_iterations (Optional[int]): Max iteration count.
    ///
    /// Returns:
    ///     Circuit: Optimized circuit.
    ///
    /// Raises:
    ///     ValueError: If optimization fails.
    #[pyo3(signature = (circuit, max_iterations = None))]
    fn execute_iterative(
        &self,
        circuit: &PyCircuit,
        max_iterations: Option<usize>,
    ) -> PyResult<PyCircuit> {
        self.inner
            .execute_iterative(&circuit.inner, max_iterations)
            .map(PyCircuit::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns the number of templates in this optimizer.
    fn template_count(&self) -> usize {
        self.inner.template_count()
    }

    /// Returns a compact debug representation.
    fn __repr__(&self) -> String {
        format!(
            "TemplateOptimization(templates={})",
            self.inner.template_count()
        )
    }
}

/// Returns whether strict VF2 subgraph mapping exists for the circuit.
///
/// Args:
///     circuit (Circuit): Logical circuit to check.
///     topology (Topology): Target hardware topology.
///     fidelity_map (Optional[Dict[Tuple[int, int], float]]): Optional edge fidelity map.
///
/// Returns:
///     bool: `True` if strict VF2 embedding exists.
///
/// Raises:
///     ValueError: If validation fails in VF2 or topology conversion.
#[pyfunction(name = "vf2_is_subgraph_isomorphic")]
#[pyo3(signature = (circuit, topology, fidelity_map = None))]
pub fn py_vf2_is_subgraph_isomorphic(
    circuit: &PyCircuit,
    topology: &PyTopology,
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
) -> PyResult<bool> {
    let fidelity = py_fidelity_to_core(fidelity_map)?;
    let vf2 = Vf2Mapping::new(topology.inner.clone(), fidelity)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    vf2.is_subgraph_isomorphic(&circuit.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Finds one logical-to-physical initial layout candidate.
///
/// Args:
///     circuit (Circuit): Logical circuit to map.
///     topology (Topology): Target hardware topology.
///     fidelity_map (Optional[Dict[Tuple[int, int], float]]): Optional edge fidelity map.
///
/// Returns:
///     Optional[List[int]]: Logical-index -> physical-id mapping, or `None`.
///
/// Raises:
///     ValueError: If validation fails in VF2 or topology conversion.
#[pyfunction(name = "vf2_find_initial_layout")]
#[pyo3(signature = (circuit, topology, fidelity_map = None))]
pub fn py_vf2_find_initial_layout(
    circuit: &PyCircuit,
    topology: &PyTopology,
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
) -> PyResult<Option<Vec<usize>>> {
    let fidelity = py_fidelity_to_core(fidelity_map)?;
    let vf2 = Vf2Mapping::new(topology.inner.clone(), fidelity)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let layout = vf2
        .find_initial_layout(&circuit.inner)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(layout.map(|qubits| qubits.into_iter().map(|q| q.id() as usize).collect()))
}

/// Returns scored VF2 initial-layout candidates.
///
/// Keyword arguments control candidate search:
/// `top_k`, `w_fidelity`, `w_topology`, `w_gate_distribution`,
/// `max_seed_subgraphs`, `max_matches_per_subgraph`,
/// `region_beam_width`, `region_oversample_factor`.
///
/// Raises:
///     ValueError: If validation fails in VF2 or topology conversion.
#[pyfunction(name = "vf2_find_initial_layout_candidates")]
#[pyo3(signature = (
    circuit,
    topology,
    fidelity_map = None,
    top_k = 10,
    w_fidelity = 0.5,
    w_topology = 0.3,
    w_gate_distribution = 0.2,
    max_seed_subgraphs = 2000,
    max_matches_per_subgraph = 128,
    region_beam_width = 32,
    region_oversample_factor = 3,
))]
pub fn py_vf2_find_initial_layout_candidates(
    py: Python<'_>,
    circuit: &PyCircuit,
    topology: &PyTopology,
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
    top_k: usize,
    w_fidelity: f64,
    w_topology: f64,
    w_gate_distribution: f64,
    max_seed_subgraphs: usize,
    max_matches_per_subgraph: usize,
    region_beam_width: usize,
    region_oversample_factor: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let fidelity = py_fidelity_to_core(fidelity_map)?;
    let vf2 = Vf2Mapping::new(topology.inner.clone(), fidelity)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let options = Vf2CandidateOptions {
        top_k,
        weights: Vf2ScoreWeights {
            fidelity: w_fidelity,
            topology: w_topology,
            gate_distribution: w_gate_distribution,
        },
        max_seed_subgraphs,
        max_matches_per_subgraph,
        region_beam_width,
        region_oversample_factor,
    };
    let candidates = vf2
        .find_initial_layout_candidates(&circuit.inner, Some(options))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut out = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let region: Vec<usize> = candidate.region.iter().map(|q| q.id() as usize).collect();
        let layout: Vec<usize> = candidate
            .logic2phy
            .iter()
            .map(|q| q.id() as usize)
            .collect();

        let score_dict = PyDict::new(py);
        score_dict.set_item("total", candidate.score.total)?;
        score_dict.set_item("fidelity", candidate.score.fidelity)?;
        score_dict.set_item("topology_fit", candidate.score.topology_fit)?;
        score_dict.set_item("gate_distribution", candidate.score.gate_distribution)?;

        let item = PyDict::new(py);
        item.set_item("region", region)?;
        item.set_item("layout", layout)?;
        item.set_item("score", score_dict)?;
        out.push(item.into_any().unbind());
    }

    Ok(out)
}

/// Runs strict VF2 mapping and returns a mapped circuit.
///
/// Raises:
///     ValueError: If strict mapping fails or validation fails.
#[pyfunction(name = "vf2_map")]
#[pyo3(signature = (circuit, topology, fidelity_map = None))]
pub fn py_vf2_map(
    circuit: &PyCircuit,
    topology: &PyTopology,
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
) -> PyResult<PyCircuit> {
    let fidelity = py_fidelity_to_core(fidelity_map)?;
    let mut vf2 = Vf2Mapping::new(topology.inner.clone(), fidelity)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    vf2.execute(&circuit.inner)
        .map(PyCircuit::from)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Runs the hybrid VF2 + SABRE mapping flow.
///
/// Raises:
///     ValueError: If mapping/routing fails.
#[pyfunction(name = "map_with_vf2_sabre")]
#[pyo3(signature = (circuit, topology, fidelity_map = None, config = None))]
pub fn py_map_with_vf2_sabre(
    circuit: &PyCircuit,
    topology: &PyTopology,
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
    config: Option<PySabreConfig>,
) -> PyResult<PyCircuit> {
    let fidelity = py_fidelity_to_core(fidelity_map)?;
    let cfg = config.map(|c| c.inner).unwrap_or_default();

    map_with_vf2_sabre(&circuit.inner, &topology.inner, fidelity.as_ref(), &cfg)
        .map(PyCircuit::from)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Converts Python fidelity map keys to core `Qubit` keys.
///
/// The conversion performs index-range validation through `py_id_to_qubit`.
fn py_fidelity_to_core(
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
) -> PyResult<Option<FidelityMap>> {
    let Some(fidelity_map) = fidelity_map else {
        return Ok(None);
    };

    let mut core = FidelityMap::with_capacity(fidelity_map.len());
    for ((u, v), value) in fidelity_map {
        core.insert(((u as u32).into(), (v as u32).into()), value);
    }
    Ok(Some(core))
}

/// Parses user-provided `vf2_policy` strings into enum policy values.
fn parse_vf2_policy(policy: &str) -> PyResult<Vf2Policy> {
    match policy.to_ascii_lowercase().as_str() {
        "direct_then_sabre" | "direct" | "vf2_then_sabre" => Ok(Vf2Policy::DirectThenSabre),
        "initial_only" | "vf2_initial_only" => Ok(Vf2Policy::InitialOnly),
        "disabled" | "off" | "none" => Ok(Vf2Policy::Disabled),
        _ => Err(PyValueError::new_err(format!(
            "unknown vf2_policy '{}'; expected one of: direct_then_sabre, initial_only, disabled",
            policy
        ))),
    }
}

/// Converts optional prune depth/width into compile prune parameters.
fn parse_prune_params(depth: Option<usize>, width: Option<usize>) -> Option<(usize, usize)> {
    match (depth, width) {
        (Some(d), Some(w)) => Some((d, w)),
        _ => None,
    }
}
