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

use crate::circuit::{PyCircuit, PyInstruction};
use crate::device::device_impl::PyDevice;
use crate::device::layout::PyLayout;
use cqlib_core::compile::resource::ResourcePolicy;
use cqlib_core::compile::{CompileConfig, CompileMode, CompileResult, WorkflowStepReport, compile};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Optimization effort selected for the compiler workflow.
#[pyclass(name = "CompileMode", module = "cqlib.compile")]
#[derive(Clone, Copy, Debug)]
pub struct PyCompileMode {
    pub(crate) inner: CompileMode,
}

impl From<CompileMode> for PyCompileMode {
    fn from(inner: CompileMode) -> Self {
        Self { inner }
    }
}

impl From<PyCompileMode> for CompileMode {
    fn from(value: PyCompileMode) -> Self {
        value.inner
    }
}

impl PyCompileMode {
    pub(crate) fn repr_label(&self) -> &'static str {
        match self.inner {
            CompileMode::Normal => "CompileMode.Normal",
            CompileMode::Enhanced => "CompileMode.Enhanced",
        }
    }
}

#[pymethods]
impl PyCompileMode {
    /// Returns the normal production compiler mode.
    #[staticmethod]
    fn normal() -> Self {
        Self {
            inner: CompileMode::Normal,
        }
    }

    /// Returns the enhanced compiler mode.
    #[staticmethod]
    fn enhanced() -> Self {
        Self {
            inner: CompileMode::Enhanced,
        }
    }

    fn __repr__(&self) -> &'static str {
        self.repr_label()
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            CompileMode::Normal => "normal",
            CompileMode::Enhanced => "enhanced",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        match self.inner {
            CompileMode::Normal => 0_u8,
            CompileMode::Enhanced => 1_u8,
        }
        .hash(&mut hasher);
        hasher.finish()
    }

    fn __copy__(&self) -> Self {
        *self
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        *self
    }
}

/// Per-step execution record produced by a compiler workflow run.
#[pyclass(name = "WorkflowStepReport", module = "cqlib.compile")]
#[derive(Clone, Debug)]
pub struct PyWorkflowStepReport {
    pub(crate) inner: WorkflowStepReport,
}

impl From<WorkflowStepReport> for PyWorkflowStepReport {
    fn from(inner: WorkflowStepReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyWorkflowStepReport {
    #[getter]
    fn stage(&self) -> &'static str {
        self.inner.stage
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name
    }

    #[getter]
    fn changed(&self) -> bool {
        self.inner.changed
    }

    #[getter]
    fn skipped(&self) -> bool {
        self.inner.skipped
    }

    #[getter]
    fn reason(&self) -> Option<String> {
        self.inner.reason.clone()
    }

    fn __repr__(&self) -> String {
        match &self.inner.reason {
            Some(reason) => format!(
                "WorkflowStepReport(stage={:?}, name={:?}, changed={}, skipped={}, reason={:?})",
                self.inner.stage, self.inner.name, self.inner.changed, self.inner.skipped, reason
            ),
            None => format!(
                "WorkflowStepReport(stage={:?}, name={:?}, changed={}, skipped={}, reason=None)",
                self.inner.stage, self.inner.name, self.inner.changed, self.inner.skipped
            ),
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Result returned by `cqlib.compile.compile`.
#[pyclass(name = "CompileResult", module = "cqlib.compile")]
#[derive(Clone, Debug)]
pub struct PyCompileResult {
    pub(crate) inner: CompileResult,
}

impl From<CompileResult> for PyCompileResult {
    fn from(inner: CompileResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCompileResult {
    #[getter]
    fn circuit(&self) -> PyCircuit {
        PyCircuit::from(self.inner.circuit.clone())
    }

    #[getter]
    fn changed(&self) -> bool {
        self.inner.changed
    }

    #[getter]
    fn mode(&self) -> PyCompileMode {
        self.inner.mode.into()
    }

    #[getter]
    fn steps(&self) -> Vec<PyWorkflowStepReport> {
        self.inner
            .steps
            .iter()
            .cloned()
            .map(PyWorkflowStepReport::from)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompileResult(changed={}, mode={}, steps={})",
            self.inner.changed,
            PyCompileMode::from(self.inner.mode).repr_label(),
            self.inner.steps.len()
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Compiles a circuit with the configured compiler workflow.
#[pyfunction(name = "compile")]
#[pyo3(signature = (circuit, *, mode=None, target_basis=None, device=None, initial_layout=None, seed=None))]
pub fn py_compile(
    circuit: PyRef<'_, PyCircuit>,
    mode: Option<PyCompileMode>,
    target_basis: Option<Vec<PyInstruction>>,
    device: Option<PyRef<'_, PyDevice>>,
    initial_layout: Option<PyRef<'_, PyLayout>>,
    seed: Option<u32>,
) -> PyResult<PyCompileResult> {
    let config = CompileConfig {
        mode: mode.map_or(CompileMode::Normal, |mode| mode.inner),
        target_basis: target_basis.map(|basis| {
            basis
                .into_iter()
                .map(|instruction| instruction.inner)
                .collect()
        }),
        device: device.map(|device| device.inner.clone()),
        initial_layout: initial_layout.map(|layout| layout.inner.clone()),
        resource_policy: ResourcePolicy::default(),
        seed,
    };

    compile(&circuit.inner, config)
        .map(PyCompileResult::from)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
