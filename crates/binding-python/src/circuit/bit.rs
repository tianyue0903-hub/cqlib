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

use cqlib_core::circuit::Qubit;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[pyclass(name = "Qubit", module = "cqlib.circuit")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyQubit {
    pub inner: Qubit,
}

#[pymethods]
impl PyQubit {
    #[new]
    fn new(index: u32) -> Self {
        PyQubit {
            inner: Qubit::new(index),
        }
    }

    #[getter]
    fn index(&self) -> usize {
        self.inner.index()
    }

    fn __repr__(&self) -> String {
        format!("Qubit({})", self.inner.index())
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !other.is_instance_of::<PyQubit>() {
            return Ok(false);
        }
        let other_qubit = other.extract::<PyQubit>()?;
        Ok(self.inner == other_qubit.inner)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __lt__(&self, other: &PyQubit) -> bool {
        self.inner < other.inner
    }

    fn __le__(&self, other: &PyQubit) -> bool {
        self.inner <= other.inner
    }

    fn __gt__(&self, other: &PyQubit) -> bool {
        self.inner > other.inner
    }

    fn __ge__(&self, other: &PyQubit) -> bool {
        self.inner >= other.inner
    }
}

impl From<Qubit> for PyQubit {
    fn from(inner: Qubit) -> Self {
        PyQubit { inner }
    }
}

impl From<PyQubit> for Qubit {
    fn from(py_qubit: PyQubit) -> Self {
        py_qubit.inner
    }
}
