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

//! Python bindings for cqlib-core Pauli module.

use cqlib_core::qis::pauli::{Pauli, PauliString, Phase};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use std::fmt;

/// Phase factor in the Pauli group, isomorphic to Z4 (the cyclic group of order 4).
///
/// Represents powers of the imaginary unit: $i^n$ where $n \\in \\{0, 1, 2, 3\\}$.
///
/// Variants:
///     Plus (1): $i^0 = 1$
///     I (1): $i^1 = i$
///     Minus (-1): $i^2 = -1$
///     MinusI (-i): $i^3 = -i$
#[pyclass(name = "Phase", module = "cqlib.qis")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PyPhase {
    pub(crate) inner: Phase,
}

impl From<Phase> for PyPhase {
    fn from(inner: Phase) -> Self {
        Self { inner }
    }
}

impl From<PyPhase> for Phase {
    fn from(value: PyPhase) -> Self {
        value.inner
    }
}

impl fmt::Display for PyPhase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[pymethods]
impl PyPhase {
    /// Creates a Phase from an integer (mod 4).
    #[new]
    fn new(val: u8) -> Self {
        Self {
            inner: Phase::from(val),
        }
    }

    /// Returns the +1 phase.
    #[staticmethod]
    fn plus() -> Self {
        Self { inner: Phase::Plus }
    }

    /// Returns the +i phase.
    #[staticmethod]
    fn i() -> Self {
        Self { inner: Phase::I }
    }

    /// Returns the -1 phase.
    #[staticmethod]
    fn minus() -> Self {
        Self {
            inner: Phase::Minus,
        }
    }

    /// Returns the -i phase.
    #[staticmethod]
    fn minus_i() -> Self {
        Self {
            inner: Phase::MinusI,
        }
    }

    /// Converts the phase to a Python complex number.
    fn to_complex<'py>(&self, py: Python<'py>) -> Bound<'py, PyComplex> {
        let c = self.inner.to_complex();
        PyComplex::from_doubles(py, c.re, c.im)
    }

    /// Returns the phase as an integer exponent (0-3).
    #[getter]
    fn exponent(&self) -> u8 {
        self.inner as u8
    }

    /// Adds two phases (multiplication in the group).
    fn __add__(&self, other: &PyPhase) -> Self {
        Self {
            inner: self.inner + other.inner,
        }
    }

    /// Multiplies two phases (same as addition in Z4).
    fn __mul__(&self, other: &PyPhase) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }

    fn __eq__(&self, other: &PyPhase) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("Phase({})", self.inner as u8)
    }

    fn __str__(&self) -> String {
        match self.inner {
            Phase::Plus => "1".to_string(),
            Phase::I => "i".to_string(),
            Phase::Minus => "-1".to_string(),
            Phase::MinusI => "-i".to_string(),
        }
    }
}

/// Single-qubit Pauli operators.
///
/// The four Pauli matrices form the basis of single-qubit quantum operations.
///
/// Variants:
///     X: Pauli-X (bit-flip) operator
///     Y: Pauli-Y operator
///     Z: Pauli-Z (phase-flip) operator
///     I: Identity operator
#[pyclass(name = "Pauli", module = "cqlib.qis")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PyPauli {
    pub(crate) inner: Pauli,
}

impl From<Pauli> for PyPauli {
    fn from(inner: Pauli) -> Self {
        Self { inner }
    }
}

impl From<PyPauli> for Pauli {
    fn from(value: PyPauli) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyPauli {
    /// Returns the X Pauli operator.
    #[staticmethod]
    fn x() -> Self {
        Self { inner: Pauli::X }
    }

    /// Returns the Y Pauli operator.
    #[staticmethod]
    fn y() -> Self {
        Self { inner: Pauli::Y }
    }

    /// Returns the Z Pauli operator.
    #[staticmethod]
    fn z() -> Self {
        Self { inner: Pauli::Z }
    }

    /// Returns the Identity operator.
    #[staticmethod]
    fn i() -> Self {
        Self { inner: Pauli::I }
    }

    /// Returns the symplectic representation (x, z) as a tuple.
    ///
    /// The symplectic encoding maps Pauli operators to binary pairs:
    /// - I = (0, 0)
    /// - X = (1, 0)
    /// - Y = (1, 1)
    /// - Z = (0, 1)
    fn to_symplectic(&self) -> (u8, u8) {
        self.inner.to_symplectic()
    }

    /// Returns the 2x2 complex matrix representation as a NumPy array.
    fn to_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<num_complex::Complex64>> {
        let mat = self.inner.to_matrix();
        let data: Vec<Vec<num_complex::Complex64>> = vec![
            vec![mat[[0, 0]], mat[[0, 1]]],
            vec![mat[[1, 0]], mat[[1, 1]]],
        ];
        PyArray2::from_vec2(py, &data).unwrap()
    }

    /// Multiplies two Pauli operators, returning the result and phase factor.
    ///
    /// Returns:
    ///     tuple: (result_pauli, phase) where phase is a Phase object
    fn mul_with_phase(&self, other: &PyPauli) -> (Self, PyPhase) {
        let (p, ph) = self.inner.mul_with_phase(other.inner);
        (Self { inner: p }, PyPhase { inner: ph })
    }

    /// Multiplication operator (without explicit phase tracking).
    fn __mul__(&self, other: &PyPauli) -> Self {
        let (p, _) = self.inner.mul_with_phase(other.inner);
        Self { inner: p }
    }

    fn __eq__(&self, other: &PyPauli) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("Pauli.{}", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// Multi-qubit Pauli string operator in symplectic representation.
///
/// A Pauli string is a tensor product of single-qubit Pauli operators across
/// multiple qubits: $P = \\bigotimes_{i=0}^{N-1} P_i$ where $P_i \\in \\{I, X, Y, Z\\}$.
#[pyclass(name = "PauliString", module = "cqlib.qis")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPauliString {
    pub(crate) inner: PauliString,
}

impl From<PauliString> for PyPauliString {
    fn from(inner: PauliString) -> Self {
        Self { inner }
    }
}

impl From<PyPauliString> for PauliString {
    fn from(value: PyPauliString) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyPauliString {
    /// Creates a new identity Pauli string with the specified number of qubits.
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: PauliString::new(num_qubits),
        }
    }

    /// Sets the Pauli operator at the specified qubit index.
    ///
    /// Args:
    ///     idx: Qubit index (0 to num_qubits-1)
    ///     pauli: The Pauli operator to set
    ///
    /// Raises:
    ///     IndexError: If idx >= num_qubits
    fn set_pauli(&mut self, idx: usize, pauli: &PyPauli) -> PyResult<()> {
        if idx >= self.inner.num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for {} qubits",
                idx, self.inner.num_qubits
            )));
        }
        self.inner.set_pauli(idx, pauli.inner);
        Ok(())
    }

    /// Gets the Pauli operator at the specified qubit index.
    ///
    /// Args:
    ///     idx: Qubit index (0 to num_qubits-1)
    ///
    /// Returns:
    ///     The Pauli operator at the specified index
    fn get_pauli(&self, idx: usize) -> PyResult<PyPauli> {
        if idx >= self.inner.num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} out of bounds for {} qubits",
                idx, self.inner.num_qubits
            )));
        }
        let x = self.inner.x[idx];
        let z = self.inner.z[idx];
        let pauli = match (x, z) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (true, true) => Pauli::Y,
            (false, true) => Pauli::Z,
        };
        Ok(PyPauli { inner: pauli })
    }

    /// Returns the number of qubits in the Pauli string.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits
    }

    /// Returns the global phase factor.
    #[getter]
    fn phase(&self) -> PyPhase {
        PyPhase {
            inner: self.inner.phase,
        }
    }

    /// Sets the global phase factor.
    #[setter]
    fn set_phase(&mut self, phase: &PyPhase) {
        self.inner.phase = phase.inner;
    }

    /// Returns the X-component bit vector as a list of booleans.
    #[getter]
    fn x_bits(&self) -> Vec<bool> {
        self.inner.x.iter().map(|b| *b).collect()
    }

    /// Returns the Z-component bit vector as a list of booleans.
    #[getter]
    fn z_bits(&self) -> Vec<bool> {
        self.inner.z.iter().map(|b| *b).collect()
    }

    /// Checks if this Pauli string commutes with another.
    ///
    /// Two Pauli strings commute if their symplectic inner product is 0 (mod 2).
    fn commutes_with(&self, other: &PyPauliString) -> PyResult<bool> {
        if self.inner.num_qubits != other.inner.num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pauli strings must have the same number of qubits",
            ));
        }
        Ok(self.inner.commutes_with(&other.inner))
    }

    /// Returns a new Pauli string that is the product of this and another.
    fn __mul__(&self, other: &PyPauliString) -> PyResult<Self> {
        if self.inner.num_qubits != other.inner.num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pauli strings must have the same number of qubits",
            ));
        }
        let result = &self.inner * &other.inner;
        Ok(Self { inner: result })
    }

    /// In-place multiplication with another Pauli string.
    fn __imul__(&mut self, other: &PyPauliString) -> PyResult<()> {
        if self.inner.num_qubits != other.inner.num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Pauli strings must have the same number of qubits",
            ));
        }
        self.inner *= &other.inner;
        Ok(())
    }

    fn __eq__(&self, other: &PyPauliString) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliString(num_qubits={}, phase={}, x_bits={:?}, z_bits={:?})",
            self.inner.num_qubits,
            self.inner.phase,
            self.x_bits(),
            self.z_bits()
        )
    }

    /// Returns a copy of this Pauli string.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
