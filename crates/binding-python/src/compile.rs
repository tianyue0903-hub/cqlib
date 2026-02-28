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

use crate::circuit::PyCircuit;
use cqlib_core::circuit::Qubit;
use cqlib_core::compile::{
    FidelityMap, SabreConfig, Vf2CandidateOptions, Vf2Mapping, Vf2Policy, Vf2ScoreWeights,
    map_with_vf2_sabre,
};
use cqlib_core::device::Topology;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[derive(FromPyObject)]
enum PyCouplingSpec {
    Bare((usize, usize)),
    Named((usize, usize, String)),
}

#[pyclass(name = "Topology", module = "cqlib.compiler")]
#[derive(Clone, Debug)]
pub struct PyTopology {
    pub(crate) inner: Topology,
}

#[pymethods]
impl PyTopology {
    #[new]
    #[pyo3(signature = (qubits, couplings))]
    fn new(qubits: Vec<usize>, couplings: Vec<PyCouplingSpec>) -> PyResult<Self> {
        let qubits = qubits
            .into_iter()
            .map(py_id_to_qubit)
            .collect::<PyResult<Vec<_>>>()?;
        let couplings = py_couplings_to_core(couplings)?;
        Ok(Self {
            inner: Topology::new(qubits, couplings),
        })
    }

    #[staticmethod]
    fn line(qubits: Vec<usize>) -> PyResult<Self> {
        let core_qubits = qubits
            .iter()
            .copied()
            .map(py_id_to_qubit)
            .collect::<PyResult<Vec<_>>>()?;

        let mut couplings = Vec::new();
        for pair in qubits.windows(2) {
            let u = py_id_to_qubit(pair[0])?;
            let v = py_id_to_qubit(pair[1])?;
            couplings.push((u, v, "CX".to_string()));
        }

        Ok(Self {
            inner: Topology::new(core_qubits, couplings),
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_couplings(&self) -> usize {
        self.inner.num_couplings()
    }

    fn is_connected(&self, u: usize, v: usize) -> PyResult<bool> {
        Ok(self
            .inner
            .is_connected(py_id_to_qubit(u)?, py_id_to_qubit(v)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "Topology(num_qubits={}, num_couplings={})",
            self.inner.num_qubits(),
            self.inner.num_couplings()
        )
    }
}

#[pyclass(name = "SabreConfig", module = "cqlib.compiler")]
#[derive(Clone, Debug)]
pub struct PySabreConfig {
    pub(crate) inner: SabreConfig,
}

#[pymethods]
impl PySabreConfig {
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
    ) -> PyResult<Self> {
        let policy = parse_vf2_policy(&vf2_policy)?;
        Ok(Self {
            inner: SabreConfig {
                vf2_policy: policy,
                field_mode,
                size_e,
                w,
                decay_coff,
                decay_reset_time,
                greedy_strategy,
                initial_iterations,
                repeat_iterations,
                swap_iterations,
                seed,
            },
        })
    }

    fn __repr__(&self) -> String {
        let policy = match self.inner.vf2_policy {
            Vf2Policy::DirectThenSabre => "direct_then_sabre",
            Vf2Policy::InitialOnly => "initial_only",
            Vf2Policy::Disabled => "disabled",
        };
        format!(
            "SabreConfig(vf2_policy='{}', field_mode={}, size_e={}, w={}, decay_coff={}, decay_reset_time={}, greedy_strategy={}, initial_iterations={}, repeat_iterations={}, swap_iterations={}, seed={})",
            policy,
            self.inner.field_mode,
            self.inner.size_e,
            self.inner.w,
            self.inner.decay_coff,
            self.inner.decay_reset_time,
            self.inner.greedy_strategy,
            self.inner.initial_iterations,
            self.inner.repeat_iterations,
            self.inner.swap_iterations,
            self.inner.seed,
        )
    }
}

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

#[pyfunction(name = "vf2_find_initial_layout_candidates")]
/// Returns scored VF2 initial-layout candidates.
///
/// Keyword arguments control candidate search:
/// `top_k`, `w_fidelity`, `w_topology`, `w_gate_distribution`,
/// `max_seed_subgraphs`, `max_matches_per_subgraph`,
/// `region_beam_width`, `region_oversample_factor`.
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

fn py_couplings_to_core(couplings: Vec<PyCouplingSpec>) -> PyResult<Vec<(Qubit, Qubit, String)>> {
    let mut core = Vec::with_capacity(couplings.len());
    for coupling in couplings {
        match coupling {
            PyCouplingSpec::Bare((u, v)) => {
                core.push((py_id_to_qubit(u)?, py_id_to_qubit(v)?, "CX".to_string()));
            }
            PyCouplingSpec::Named((u, v, name)) => {
                core.push((py_id_to_qubit(u)?, py_id_to_qubit(v)?, name));
            }
        }
    }
    Ok(core)
}

fn py_fidelity_to_core(
    fidelity_map: Option<HashMap<(usize, usize), f64>>,
) -> PyResult<Option<FidelityMap>> {
    let Some(fidelity_map) = fidelity_map else {
        return Ok(None);
    };

    let mut core = FidelityMap::with_capacity(fidelity_map.len());
    for ((u, v), value) in fidelity_map {
        core.insert((py_id_to_qubit(u)?, py_id_to_qubit(v)?), value);
    }
    Ok(Some(core))
}

fn py_id_to_qubit(idx: usize) -> PyResult<Qubit> {
    let id = u32::try_from(idx)
        .map_err(|_| PyValueError::new_err(format!("qubit id {} overflows u32", idx)))?;
    Ok(Qubit::new(id))
}

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
