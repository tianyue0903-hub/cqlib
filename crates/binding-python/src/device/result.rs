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

use super::common::{py_id_to_qubit, qubit_to_py_id};
use cqlib_core::device::{ExecutionResult, Outcome, Status};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[pyclass(name = "Outcome", module = "cqlib.device")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyOutcome {
    pub(crate) inner: Outcome,
}

impl From<Outcome> for PyOutcome {
    fn from(inner: Outcome) -> Self {
        Self { inner }
    }
}

impl From<PyOutcome> for Outcome {
    fn from(value: PyOutcome) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyOutcome {
    #[new]
    fn new(bitstring: String) -> PyResult<Self> {
        Outcome::from_bitstring(&bitstring)
            .map(Self::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_bitstring(bitstring: String) -> PyResult<Self> {
        Self::new(bitstring)
    }

    fn is_one(&self, index: usize) -> bool {
        self.inner.is_one(index)
    }

    fn to_bitstring(&self, num_qubits: usize) -> String {
        self.inner.to_string(num_qubits)
    }

    #[getter]
    fn chunks(&self) -> Vec<u64> {
        self.inner.0.to_vec()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !other.is_instance_of::<PyOutcome>() {
            return Ok(false);
        }
        let other = other.extract::<PyOutcome>()?;
        Ok(self.inner == other.inner)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("Outcome(chunks={:?})", self.chunks())
    }
}

#[pyclass(name = "Status", module = "cqlib.device")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyStatus {
    pub(crate) inner: Status,
}

impl From<Status> for PyStatus {
    fn from(inner: Status) -> Self {
        Self { inner }
    }
}

impl From<PyStatus> for Status {
    fn from(value: PyStatus) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyStatus {
    #[staticmethod]
    fn queued() -> Self {
        Self {
            inner: Status::Queued,
        }
    }

    #[staticmethod]
    fn running() -> Self {
        Self {
            inner: Status::Running,
        }
    }

    #[staticmethod]
    fn completed() -> Self {
        Self {
            inner: Status::Completed,
        }
    }

    #[staticmethod]
    fn failed(error_msg: String, error_code: i32) -> Self {
        Self {
            inner: Status::Failed {
                error_msg,
                error_code,
            },
        }
    }

    #[staticmethod]
    fn cancelled() -> Self {
        Self {
            inner: Status::Cancelled,
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        status_kind(&self.inner)
    }

    #[getter]
    fn error_msg(&self) -> Option<String> {
        match &self.inner {
            Status::Failed { error_msg, .. } => Some(error_msg.clone()),
            _ => None,
        }
    }

    #[getter]
    fn error_code(&self) -> Option<i32> {
        match self.inner {
            Status::Failed { error_code, .. } => Some(error_code),
            _ => None,
        }
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    fn is_success(&self) -> bool {
        self.inner.is_success()
    }

    fn __repr__(&self) -> String {
        format!("Status({})", self.kind())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

#[pyclass(name = "ExecutionResult", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyExecutionResult {
    pub(crate) inner: ExecutionResult,
}

impl From<ExecutionResult> for PyExecutionResult {
    fn from(inner: ExecutionResult) -> Self {
        Self { inner }
    }
}

impl From<PyExecutionResult> for ExecutionResult {
    fn from(value: PyExecutionResult) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyExecutionResult {
    #[new]
    #[pyo3(signature = (task_id, qubits, shots, num_qubits, backend=None))]
    fn new(
        task_id: String,
        qubits: Vec<usize>,
        shots: usize,
        num_qubits: usize,
        backend: Option<String>,
    ) -> PyResult<Self> {
        let qubits = qubits
            .into_iter()
            .map(py_id_to_qubit)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: ExecutionResult::new(task_id, qubits, shots, num_qubits, backend, None),
        })
    }

    fn start(&mut self) {
        self.inner.start(None);
    }

    fn finish(&mut self, counts: HashMap<String, usize>) -> PyResult<()> {
        let counts = counts
            .into_iter()
            .map(|(bitstring, count)| {
                Outcome::from_bitstring(&bitstring)
                    .map(|outcome| (outcome, count))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .collect::<PyResult<HashMap<_, _>>>()?;
        self.inner.finish(counts, None);
        Ok(())
    }

    fn fail(&mut self, msg: String, code: i32) {
        self.inner.fail(msg, code);
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }

    fn calc_probabilities(&mut self) {
        self.inner.calc_probabilities();
    }

    #[getter]
    fn task_id(&self) -> String {
        self.inner.task_id().to_string()
    }

    #[getter]
    fn shots(&self) -> usize {
        self.inner.shots()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.inner
            .qubits()
            .iter()
            .copied()
            .map(qubit_to_py_id)
            .collect()
    }

    #[getter]
    fn status(&self) -> PyStatus {
        PyStatus {
            inner: self.inner.status().clone(),
        }
    }

    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at().to_string()
    }

    #[getter]
    fn started_at(&self) -> Option<String> {
        self.inner.started_at().as_ref().map(ToString::to_string)
    }

    #[getter]
    fn finished_at(&self) -> Option<String> {
        self.inner.finished_at().as_ref().map(ToString::to_string)
    }

    #[getter]
    fn backend(&self) -> Option<String> {
        self.inner.backend().cloned()
    }

    #[getter]
    fn counts(&self) -> HashMap<String, usize> {
        counts_to_bitstring_map(self.inner.counts(), self.num_qubits())
    }

    #[getter]
    fn probabilities(&self) -> Option<HashMap<String, f64>> {
        self.inner
            .probabilities()
            .as_ref()
            .map(|probs| probabilities_to_bitstring_map(probs, self.num_qubits()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionResult(task_id='{}', status='{}', shots={}, num_qubits={})",
            self.task_id(),
            self.status().kind(),
            self.shots(),
            self.num_qubits()
        )
    }
}

fn status_kind(status: &Status) -> &'static str {
    match status {
        Status::Queued => "queued",
        Status::Running => "running",
        Status::Completed => "completed",
        Status::Failed { .. } => "failed",
        Status::Cancelled => "cancelled",
    }
}

fn counts_to_bitstring_map(
    counts: &HashMap<Outcome, usize>,
    num_qubits: usize,
) -> HashMap<String, usize> {
    counts
        .iter()
        .map(|(outcome, count)| (outcome.to_string(num_qubits), *count))
        .collect()
}

fn probabilities_to_bitstring_map(
    probabilities: &HashMap<Outcome, f64>,
    num_qubits: usize,
) -> HashMap<String, f64> {
    probabilities
        .iter()
        .map(|(outcome, prob)| (outcome.to_string(num_qubits), *prob))
        .collect()
}
