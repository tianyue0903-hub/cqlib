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

//! Python Bindings for Quantum Circuit
//!
//! This module provides Python bindings for the quantum [`Circuit`] from cqlib-core.
//!
//! # Key Components
//!
//! - [`PyCircuit`]: The main class representing a quantum circuit in Python.
//! - [`PyParamLike`]: Helper type for accepting either `float` or `Parameter` objects.
//! - [`PyIntQubitList`]: Helper type for flexible qubit initialization.
//!
//! # Design Notes
//!
//! The Python bindings wrap the core Rust [`Circuit`] type, providing a ergonomic API
//! for quantum circuit construction. All gate methods follow the pattern of accepting
//! Python integers (which are converted to [`Qubit`] internally) and parameters that
//! can be either fixed values (`float`) or symbolic (`Parameter`).
//!
//! # Example
//!
//! ```python
//! from cqlib import Circuit, Parameter
//!
//! # Create a 2-qubit circuit
//! circuit = Circuit(2)
//!
//! # Apply gates
//! circuit.h(0)
//! circuit.cx(0, 1)
//!
//! # Use symbolic parameters
//! theta = Parameter.symbol("theta")
//! circuit.rx(0, theta)
//! ```

use super::bit::PyQubit;
use super::gate::{PyCircuitGate, PyConditionView, PyMcGate, PyStandardGate, PyUnitaryGate};
use super::operation::PyOperation;
use super::parameter::PyParameter;
use crate::circuit::operation::PyOperationIter;
use cqlib_core::circuit::gate::Instruction;
use cqlib_core::circuit::param::CircuitParam;
use cqlib_core::circuit::param::ParameterValue;
use cqlib_core::circuit::{Circuit, Operation, Qubit};
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::*;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Parameter type that accepts either a fixed value or a symbolic parameter.
///
/// This enum enables Python methods to accept both:
/// - `float` values: Fixed numeric parameters (e.g., `3.14159`)
/// - `Parameter` objects: Symbolic parameters for variational algorithms
///
/// # Example
///
/// ```python
/// from cqlib import Parameter
///
/// # Fixed parameter
/// circuit.rx(0, 1.57)  # OK
///
/// # Symbolic parameter
/// theta = Parameter.symbol("theta")
/// circuit.rx(0, theta)  # OK
/// ```
#[derive(FromPyObject)]
pub enum PyParamLike {
    /// A fixed numeric value.
    Float(f64),
    /// A symbolic parameter for parameterization.
    Param(PyParameter),
}

impl From<PyParamLike> for ParameterValue {
    fn from(val: PyParamLike) -> Self {
        match val {
            PyParamLike::Float(f) => ParameterValue::Fixed(f),
            PyParamLike::Param(p) => ParameterValue::Param(p.into_inner()),
        }
    }
}

/// Python wrapper for the quantum [`Circuit`].
///
/// This is the main class for constructing and manipulating quantum circuits in Python.
/// It wraps the Rust [`cqlib_core::circuit::Circuit`] type, providing an ergonomic API
/// for building quantum programs.
///
/// # Creating a Circuit
///
/// A circuit can be created in three ways:
///
/// 1. **By number of qubits**: `Circuit(5)` creates a circuit with qubits 0-4.
/// 2. **By list of integers**: `Circuit([0, 2, 4])` creates a circuit with specific qubits.
/// 3. **By list of Qubit objects**: `Circuit([Qubit(0), Qubit(1)])`.
///
/// # Example
///
/// ```python
/// from cqlib import Circuit
///
/// # Create a 2-qubit circuit
/// circuit = Circuit(2)
///
/// # Apply gates
/// circuit.h(0)        # Hadamard on qubit 0
/// circuit.cx(0, 1)    # CNOT from qubit 0 to 1
/// circuit.measure(0)  # Measure qubit 0
///
/// print(circuit.num_qubits)  # 2
/// ```
#[pyclass(name = "Circuit", module = "cqlib.circuit")]
#[derive(Debug, Clone)]
pub struct PyCircuit {
    /// The underlying Rust Circuit instance.
    pub inner: Circuit,
}

impl From<Circuit> for PyCircuit {
    fn from(circuit: Circuit) -> Self {
        PyCircuit { inner: circuit }
    }
}

/// Flexible qubit specification for circuit initialization.
///
/// This enum allows the [`PyCircuit::new`] constructor to accept three different
/// input formats for specifying which qubits the circuit should manage.
///
/// # Variants
///
/// - `NumQubits(usize)`: Create a circuit with N sequential qubits (0 to N-1).
/// - `IndexList(Vec<usize>)`: Create a circuit with specific qubit indices.
/// - `QubitList(Vec<PyQubit>)`: Create a circuit from existing Qubit objects.
///
/// # Example
///
/// ```python
/// from cqlib import Circuit, Qubit
///
/// # 5 qubits numbered 0-4
/// c1 = Circuit(5)
///
/// # Specific indices: 0, 2, 4
/// c2 = Circuit([0, 2, 4])
///
/// # Using Qubit objects
/// c3 = Circuit([Qubit(0), Qubit(1)])
/// ```
#[derive(FromPyObject)]
pub enum PyIntQubitList {
    /// Create a circuit with N sequential qubits (0 to N-1).
    NumQubits(usize),
    /// Create a circuit with specific qubit indices.
    IndexList(Vec<usize>),
    /// Create a circuit from existing Qubit objects.
    QubitList(Vec<PyQubit>),
}

/// Accepts a single qubit identifier: either an integer or a Qubit object.
///
/// This type enables methods to accept both:
/// - `int`: Direct qubit index (e.g., `0`)
/// - `Qubit`: A Qubit object (e.g., `Qubit(0)`)
///
/// # Example
///
/// ```python
/// from cqlib import Qubit
///
/// # Both are equivalent
/// circuit.h(0)
/// circuit.h(Qubit(0))
/// ```
#[derive(FromPyObject)]
pub enum PyIntOrQubit {
    /// Integer qubit index.
    Int(usize),
    /// Qubit object.
    Qubit(PyQubit),
}

impl PyIntOrQubit {
    /// Converts to the underlying Qubit type.
    pub fn into_qubit(self) -> Qubit {
        match self {
            PyIntOrQubit::Int(i) => Qubit::new(i as u32),
            PyIntOrQubit::Qubit(q) => q.inner,
        }
    }
}

/// Accepts a list of qubits: either integers or Qubit objects.
///
/// This type enables methods to accept both:
/// - `List[int]`: List of qubit indices (e.g., `[0, 1, 2]`)
/// - `List[Qubit]`: List of Qubit objects (e.g., `[Qubit(0), Qubit(1)]`)
///
/// # Example
///
/// ```python
/// from cqlib import Qubit
///
/// # Both are equivalent
/// circuit.barrier([0, 1, 2])
/// circuit.barrier([Qubit(0), Qubit(1), Qubit(2)])
/// ```
#[derive(FromPyObject)]
pub enum PyIntListOrQubitList {
    /// List of integer qubit indices.
    IntList(Vec<usize>),
    /// List of Qubit objects.
    QubitList(Vec<PyQubit>),
}

impl PyIntListOrQubitList {
    /// Converts to a vector of Qubits.
    pub fn into_qubits(self) -> Vec<Qubit> {
        match self {
            PyIntListOrQubitList::IntList(indices) => {
                indices.into_iter().map(|i| Qubit::new(i as u32)).collect()
            }
            PyIntListOrQubitList::QubitList(qubits) => {
                qubits.into_iter().map(|q| q.inner).collect()
            }
        }
    }
}

#[pymethods]
impl PyCircuit {
    /// Creates a new quantum circuit.
    ///
    /// The circuit can be initialized in three ways:
    /// 1. By number of qubits: `Circuit(5)` creates a circuit with qubits 0-4.
    /// 2. By list of integers: `Circuit([0, 2, 4])` creates a circuit with specific qubits.
    /// 3. By list of Qubit objects: `Circuit([Qubit(0), Qubit(1)])`.
    ///
    /// Args:
    ///     qubits (Union[int, List[int], List[Qubit]]): The qubits to include in the circuit.
    #[new]
    fn new(qubits: PyIntQubitList) -> PyResult<Self> {
        match qubits {
            PyIntQubitList::NumQubits(num) => Ok(PyCircuit {
                inner: Circuit::new(num),
            }),
            PyIntQubitList::IndexList(indices) => {
                let core_qubits: Vec<Qubit> = indices
                    .into_iter()
                    .map(|idx| Qubit::new(idx as u32))
                    .collect();
                let inner = Circuit::from_qubits(core_qubits)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyCircuit { inner })
            }
            PyIntQubitList::QubitList(qubits) => {
                let core_qubits: Vec<Qubit> = qubits.into_iter().map(|q| q.inner).collect();
                let inner = Circuit::from_qubits(core_qubits)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyCircuit { inner })
            }
        }
    }

    /// Returns the number of qubits in the circuit.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns a tuple of all qubits in the circuit.
    #[getter]
    fn qubits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.qubits().iter().map(|&q| PyQubit::from(q)))
    }

    /// Returns a tuple of all symbolic parameters used in the circuit.
    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner
                .parameters()
                .iter()
                .map(|p| PyParameter::from(p.clone())),
        )
    }

    /// Returns the width (number of qubits) of the circuit.
    ///
    /// This is an alias for `num_qubits`.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Returns a tuple of all symbolic variable names used in the circuit.
    #[getter]
    fn symbols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.symbols().iter().map(|s| s.as_str()))
    }

    /// Returns the global phase of the circuit as a Parameter.
    ///
    /// The global phase represents a scalar factor e^(i*theta).
    /// While unobservable in isolated systems, it is critical for
    /// controlled operations and sub-circuit composition.
    #[getter]
    fn global_phase(&self) -> PyParameter {
        PyParameter::from(self.inner.global_phase())
    }

    /// Sets the global phase of the circuit.
    ///
    /// Args:
    ///     phase: The phase value (can be float or Parameter)
    fn set_global_phase(&mut self, phase: PyParamLike) -> PyResult<()> {
        use cqlib_core::circuit::parameter::Parameter;
        let param: Parameter = match phase {
            PyParamLike::Float(f) => Parameter::from(f),
            PyParamLike::Param(p) => p.into_inner(),
        };
        self.inner.set_global_phase(param);
        Ok(())
    }

    /// Applies an Identity (I) gate to the specified qubit.
    ///
    /// Identity gate is a no-operation that leaves the qubit state unchanged.
    /// Often used for alignment or waiting periods.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn i(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .i(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Hadamard (H) gate to the specified qubit.
    ///
    /// Creates superposition: H|0⟩ = (|0⟩ + |1⟩) / √2
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn h(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .h(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Pauli-X (X) gate (bit flip) to the specified qubit.
    ///
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn x(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .x(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Pauli-Y (Y) gate to the specified qubit.
    ///
    /// Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩ (bit and phase flip)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn y(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .y(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Pauli-Z (Z) gate (phase flip) to the specified qubit.
    ///
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn z(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .z(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an S (Phase) gate to the specified qubit.
    ///
    /// Applies a phase of i to the |1⟩ state (Z^½)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn s(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .s(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an S-dagger (S†) gate to the specified qubit.
    ///
    /// Inverse of the S gate, applies a phase of -i
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn sdg(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .sdg(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a T gate to the specified qubit.
    ///
    /// Applies a phase of e^(iπ/4) to the |1⟩ state (Z^¼)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn t(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .t(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a T-dagger (T†) gate to the specified qubit.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn tdg(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .tdg(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √X (SX) gate to the specified qubit.
    ///
    /// A 90-degree rotation around the X-axis. SX² = X
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn x2p(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .x2p(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √X† (SXdg) gate to the specified qubit.
    ///
    /// Inverse of the SX gate
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn x2m(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .x2m(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √Y gate to the specified qubit.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn y2p(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .y2p(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √Y† gate to the specified qubit.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn y2m(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .y2m(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a rotation around the X-axis by angle theta.
    ///
    /// RX(θ) = e^(-iθX/2)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn rx(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rx(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a rotation around the Y-axis by angle theta.
    ///
    /// RY(θ) = e^(-iθY/2)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn ry(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .ry(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a rotation around the Z-axis by angle theta.
    ///
    /// RZ(θ) = e^(-iθZ/2)
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn rz(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rz(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Phase (P) gate to the specified qubit.
    ///
    /// Applies a phase of e^(iλ) to the |1⟩ state.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     lambda: Phase angle (can be float or Parameter)
    fn phase(&mut self, qubit: PyIntOrQubit, lambda: PyParamLike) -> PyResult<()> {
        self.inner
            .phase(qubit.into_qubit(), lambda)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an XY gate to the specified qubit.
    ///
    /// Rotation in the XY plane between |01⟩ and |10⟩ subspace.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn xy(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √XY gate (positive phase) to the specified qubit.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn xy2p(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy2p(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an √XY† gate (negative phase) to the specified qubit.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn xy2m(&mut self, qubit: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .xy2m(qubit.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a generic single-qubit rotation gate U(θ, φ, λ).
    ///
    /// The most general single-qubit unitary gate.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta, phi, lambda: Rotation angles (can be float or Parameter)
    #[pyo3(signature = (qubit, theta, phi, lambda))]
    fn u(
        &mut self,
        qubit: PyIntOrQubit,
        theta: PyParamLike,
        phi: PyParamLike,
        lambda: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .u(qubit.into_qubit(), theta, phi, lambda)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a rotation in the XY plane.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     theta, phi: Rotation angles (can be float or Parameter)
    fn rxy(&mut self, qubit: PyIntOrQubit, theta: PyParamLike, phi: PyParamLike) -> PyResult<()> {
        self.inner
            .rxy(qubit.into_qubit(), theta, phi)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-NOT (CX/CNOT) gate.
    ///
    /// Flips the target qubit if and only if the control qubit is |1⟩.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    fn cx(&mut self, control: PyIntOrQubit, target: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .cx(control.into_qubit(), target.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-Y (CY) gate.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    fn cy(&mut self, control: PyIntOrQubit, target: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .cy(control.into_qubit(), target.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-Z (CZ) gate.
    ///
    /// Adds a phase of -1 only if both qubits are |1⟩.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    fn cz(&mut self, control: PyIntOrQubit, target: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .cz(control.into_qubit(), target.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a SWAP gate.
    ///
    /// Exchanges the states of two qubits.
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    fn swap(&mut self, a: PyIntOrQubit, b: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .swap(a.into_qubit(), b.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an Ising XX coupling gate RXX(θ).
    ///
    /// RXX(θ) = e^(-iθ X⊗X / 2)
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    ///     theta: Rotation angle (can be float or Parameter)
    fn rxx(&mut self, a: PyIntOrQubit, b: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rxx(a.into_qubit(), b.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an Ising YY coupling gate RYY(θ).
    ///
    /// RYY(θ) = e^(-iθ Y⊗Y / 2)
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    ///     theta: Rotation angle (can be float or Parameter)
    fn ryy(&mut self, a: PyIntOrQubit, b: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .ryy(a.into_qubit(), b.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an Ising ZZ coupling gate RZZ(θ).
    ///
    /// RZZ(θ) = e^(-iθ Z⊗Z / 2)
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    ///     theta: Rotation angle (can be float or Parameter)
    fn rzz(&mut self, a: PyIntOrQubit, b: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rzz(a.into_qubit(), b.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies an Ising ZX coupling gate RZX(θ).
    ///
    /// RZX(θ) = e^(-iθ Z⊗X / 2)
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    ///     theta: Rotation angle (can be float or Parameter)
    fn rzx(&mut self, a: PyIntOrQubit, b: PyIntOrQubit, theta: PyParamLike) -> PyResult<()> {
        self.inner
            .rzx(a.into_qubit(), b.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Fermionic Simulation gate (fSim).
    ///
    /// Useful in quantum chemistry simulations.
    /// fSim(θ, φ) = [[1, 0, 0, 0],
    ///                [0, cos(θ), -i sin(θ), 0],
    ///                [0, -i sin(θ), cos(θ), 0],
    ///                [0, 0, 0, e^(-iφ)]]
    ///
    /// Args:
    ///     a, b: Qubit indices (int) or Qubit objects
    ///     theta, phi: Parameters (can be float or Parameter)
    fn fsim(
        &mut self,
        a: PyIntOrQubit,
        b: PyIntOrQubit,
        theta: PyParamLike,
        phi: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .fsim(a.into_qubit(), b.into_qubit(), theta, phi)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-RX (CRX) gate.
    ///
    /// Performs an X-rotation on the target if the control is |1⟩.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn crx(
        &mut self,
        control: PyIntOrQubit,
        target: PyIntOrQubit,
        theta: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .crx(control.into_qubit(), target.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-RY (CRY) gate.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn cry(
        &mut self,
        control: PyIntOrQubit,
        target: PyIntOrQubit,
        theta: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .cry(control.into_qubit(), target.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Controlled-RZ (CRZ) gate.
    ///
    /// Args:
    ///     control: Control qubit index (int) or Qubit object
    ///     target: Target qubit index (int) or Qubit object
    ///     theta: Rotation angle (can be float or Parameter)
    fn crz(
        &mut self,
        control: PyIntOrQubit,
        target: PyIntOrQubit,
        theta: PyParamLike,
    ) -> PyResult<()> {
        self.inner
            .crz(control.into_qubit(), target.into_qubit(), theta)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a Toffoli gate (CCX / Toffoli).
    ///
    /// A 3-qubit gate where the target flips if and only if both controls are |1⟩.
    ///
    /// Args:
    ///     control1, control2: Control qubit indices (int) or Qubit objects
    ///     target: Target qubit index (int) or Qubit object
    pub fn ccx(
        &mut self,
        control1: PyIntOrQubit,
        control2: PyIntOrQubit,
        target: PyIntOrQubit,
    ) -> PyResult<()> {
        self.inner
            .ccx(
                control1.into_qubit(),
                control2.into_qubit(),
                target.into_qubit(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a multi-controlled version of a standard gate.
    ///
    /// Automatically handles gate promotion: X with 1 control becomes CX,
    /// with 2 controls becomes CCX, etc. For higher controls, creates an MCGate.
    ///
    /// Args:
    ///     instruction: The base standard gate (e.g., StandardGate.X)
    ///     controls: List of control qubit indices
    ///     targets: List of target qubit indices
    ///     params: Optional parameters for the base gate
    ///
    /// # Example
    ///
    /// ```python
    /// circuit.multi_control(StandardGate.X, [0, 1], [2], None)  # Equivalent to CCX
    /// ```
    #[pyo3(signature = (instruction, controls, targets, params=None))]
    pub fn multi_control(
        &mut self,
        instruction: PyStandardGate,
        controls: Vec<usize>,
        targets: Vec<usize>,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let mut ps = vec![];
        if let Some(params) = params {
            for p in params {
                match p {
                    PyParamLike::Float(f) => ps.push(ParameterValue::Fixed(f)),
                    PyParamLike::Param(p) => ps.push(ParameterValue::Param(p.into_inner())),
                }
            }
        }
        let control_qubits: Vec<Qubit> = controls
            .into_iter()
            .map(|q| Qubit::try_from(q).map_err(|e| PyValueError::new_err(e.to_string())))
            .collect::<PyResult<_>>()?;
        let target_qubits: Vec<Qubit> = targets
            .into_iter()
            .map(|q| Qubit::try_from(q).map_err(|e| PyValueError::new_err(e.to_string())))
            .collect::<PyResult<_>>()?;
        self.inner
            .multi_control(instruction.inner, control_qubits, target_qubits, ps)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Applies a multi-controlled (MC) gate.
    ///
    /// Similar to [`multi_control`][PyCircuit::multi_control] but accepts an
    /// existing [`McGate`][PyMcGate] instead of a StandardGate.
    ///
    /// Args:
    ///     instruction: The multi-controlled gate to apply
    ///     qubits: List of qubit indices (first N-1 are controls, last is target)
    ///     params: Optional parameters for the gate
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib.circuit.gate import McGate, StandardGate
    ///
    /// # Create a 3-control toffoli-like gate
    /// mc_gate = McGate(StandardGate.X, 3)
    /// circuit.multi_control_gate(mc_gate, [0, 1, 2], None)
    /// ```
    #[pyo3(signature = (instruction, qubits, params=None))]
    pub fn multi_control_gate(
        &mut self,
        instruction: PyMcGate,
        qubits: Vec<usize>,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits
            .into_iter()
            .map(|q| Qubit::try_from(q).map_err(|e| PyValueError::new_err(e.to_string())))
            .collect::<PyResult<_>>()?;
        let inst = Instruction::McGate(Box::new(instruction.inner));
        let params_core: Vec<ParameterValue> = params
            .unwrap_or_default()
            .into_iter()
            .map(ParameterValue::from)
            .collect();

        self.inner
            .append(inst, qubits_core, params_core, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a custom unitary gate to the circuit.
    ///
    /// Allows inserting user-defined gates specified by a unitary matrix.
    ///
    /// Args:
    ///     gate: The custom unitary gate definition (UnitaryGate)
    ///     qubits: The list of qubit indices to apply the gate to
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib.circuit.gate import UnitaryGate
    /// import numpy as np
    ///
    /// # Define a custom gate (e.g., a rotation)
    /// mat = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X matrix
    /// u_gate = UnitaryGate("MyGate", 1).with_matrix(mat)
    ///
    /// circuit.unitary(u_gate, [0])
    /// ```
    fn unitary(&mut self, gate: PyUnitaryGate, qubits: PyIntListOrQubitList) -> PyResult<()> {
        self.inner
            .unitary(gate.into(), qubits.into_qubits())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Measures the specified qubit.
    ///
    /// This is a non-unitary operation that collapses the qubit's state to |0⟩ or |1⟩.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn measure(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .measure(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Resets the specified qubit to the |0⟩ state.
    ///
    /// This is a non-unitary operation.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    fn reset(&mut self, qubit: PyIntOrQubit) -> PyResult<()> {
        self.inner
            .reset(qubit.into_qubit())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Inserts a barrier operation.
    ///
    /// A barrier forbids the compiler from optimizing across this boundary.
    /// It has no physical effect but is useful for debugging and manual optimization.
    ///
    /// Args:
    ///     qubits: List of qubit indices (int) or Qubit objects
    fn barrier(&mut self, qubits: PyIntListOrQubitList) -> PyResult<()> {
        self.inner
            .barrier(qubits.into_qubits())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a pre-compiled CircuitGate to this circuit.
    ///
    /// Allows nesting circuits within circuits (subroutines).
    ///
    /// Args:
    ///     instruction: The CircuitGate to append
    ///     qubits: List of qubit indices (int) or Qubit objects
    ///     params: Optional parameter values to bind to the sub-circuit
    #[pyo3(signature = (instruction, qubits, params=None))]
    fn circuit_gate(
        &mut self,
        instruction: PyCircuitGate,
        qubits: PyIntListOrQubitList,
        params: Option<Vec<PyParamLike>>,
    ) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits.into_qubits();

        let inst = Instruction::CircuitGate(Box::new(instruction.inner));
        let params_core: Vec<ParameterValue> = params
            .unwrap_or_default()
            .into_iter()
            .map(|p| p.into())
            .collect();
        self.inner
            .append(inst, qubits_core, params_core, None)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    /// Applies a Delay instruction to the specified qubit.
    ///
    /// Represents an idle period, often used for timing control in pulse-level scheduling.
    ///
    /// Args:
    ///     qubit: Qubit index (int) or Qubit object
    ///     param: The duration of the delay (can be float or Parameter)
    fn delay(&mut self, qubit: PyIntOrQubit, param: PyParamLike) -> PyResult<()> {
        self.inner
            .delay(qubit.into_qubit(), param.into())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Returns the inverse (adjoint) of the circuit.
    ///
    /// Creates a new circuit representing U† such that U†U = I.
    ///
    /// Returns:
    ///     A new circuit that is the inverse of this circuit.
    ///
    /// Raises:
    ///     ValueError: If the circuit contains non-unitary operations (Measure, Reset)
    ///                  or gates that cannot be symbolically inverted.
    fn inverse(&self) -> PyResult<Self> {
        let new_inner = self
            .inner
            .inverse()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyCircuit { inner: new_inner })
    }

    /// Assign parameters to the circuit and return a new circuit with the assigned values.
    ///
    /// Args:
    ///     bindings: A dictionary mapping parameter names to values.
    ///         If None, all symbolic parameters remain symbolic.
    ///
    /// Returns:
    ///     Circuit: A new circuit with parameters assigned.
    ///
    /// Example:
    ///     >>> circuit = Circuit(1)
    ///     >>> theta = Parameter.symbol("theta")
    ///     >>> circuit.rx(0, theta)
    ///     >>> assigned = circuit.assign_parameters({"theta": 3.14159})
    #[pyo3(signature = (bindings=None))]
    fn assign_parameters(&self, bindings: Option<HashMap<String, f64>>) -> PyResult<Self> {
        let new_inner = self
            .inner
            .assign_parameters(&bindings)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyCircuit { inner: new_inner })
    }

    /// Returns an iterator over all operations in the circuit.
    #[getter]
    fn operations(&self) -> PyOperationIter {
        PyOperationIter::new(self.inner.operations().to_vec(), 0)
    }

    /// Converts the circuit to a reusable gate (CircuitGate).
    ///
    /// "Freezes" the current circuit and wraps it into an instruction that can be
    /// appended to another circuit.
    ///
    /// Args:
    ///     name: A name for the new gate
    ///
    /// Returns:
    ///     A CircuitGate that can be applied to qubits
    ///
    /// # Example
    ///
    /// ```python
    /// sub_circuit = Circuit(1)
    /// sub_circuit.h(0)
    /// sub_circuit.rz(0, 0.5)
    ///
    /// # Convert to a reusable gate
    /// my_gate = sub_circuit.to_gate("MyH")
    ///
    /// # Use in another circuit
    /// main_circuit = Circuit(2)
    /// main_circuit.circuit_gate(my_gate, [0])
    /// ```
    fn to_gate(&self, name: String) -> PyResult<PyCircuitGate> {
        let instruction = self
            .inner
            .clone()
            .to_gate(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Instruction::CircuitGate(gate) = instruction {
            Ok(PyCircuitGate { inner: *gate })
        } else {
            Err(PyValueError::new_err(
                "Unexpected instruction type returned from to_gate",
            ))
        }
    }

    /// Converts the circuit to a unitary matrix.
    ///
    /// Args:
    ///     qubits_order: Optional order of qubits for the matrix (default: qubit order in circuit)
    ///
    /// Returns:
    ///     A 2D NumPy array representing the unitary matrix of the circuit.
    ///
    /// # Example
    ///
    /// ```python
    /// circuit = Circuit(2)
    /// circuit.h(0)
    /// circuit.cx(0, 1)
    /// matrix = circuit.to_matrix()
    /// ```
    #[pyo3(signature = (qubits_order=None))]
    fn to_matrix<'py>(
        &self,
        py: Python<'py>,
        qubits_order: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        Ok(self.inner.to_matrix(qubits_order.as_ref()).to_pyarray(py))
    }

    /// Decomposes the circuit by expanding all sub-circuit gates.
    ///
    /// Recursively unpacks any CircuitGate instructions into their fundamental operations.
    /// Handles parameter substitution and qubit mapping from parent circuits.
    ///
    /// Returns:
    ///     A new flattened circuit with only base instructions.
    fn decompose(&self) -> Self {
        Self {
            inner: self.inner.decompose(),
        }
    }

    /// Accesses operations by index (supports negative indexing and slicing).
    ///
    /// Args:
    ///     idx: Integer index or slice
    ///
    /// Returns:
    ///     Operation at the given index, or a list of operations for a slice
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        idx: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ops = self.inner.operations();
        let len = ops.len();

        // 处理单个整数索引
        if let Ok(index) = idx.extract::<isize>() {
            let idx = if index < 0 {
                let neg = len as isize + index;
                if neg < 0 {
                    return Err(PyIndexError::new_err(format!(
                        "Index {} out of range for circuit with {} operations",
                        index, len
                    )));
                }
                neg as usize
            } else {
                if index as usize >= len {
                    return Err(PyIndexError::new_err(format!(
                        "Index {} out of range for circuit with {} operations",
                        index, len
                    )));
                }
                index as usize
            };

            let op = PyOperation::from(ops[idx].clone());
            return op.into_bound_py_any(py);
        }

        if let Ok(slice) = idx.cast_into::<PySlice>() {
            let indices = slice.indices(len as isize)?;

            // indices 是 PySliceIndices 结构体，不是元组
            let mut result = Vec::with_capacity(indices.slicelength);
            let mut i = indices.start;

            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                result.push(PyOperation::from(ops[i as usize].clone()));
                i += indices.step;
            }

            return Ok(PyList::new(py, result)?.into_any());
        }

        Err(PyTypeError::new_err("Index must be integer or slice"))
    }

    /// Returns the number of operations in the circuit.
    fn __len__(&self) -> usize {
        self.inner.operations().len()
    }

    /// Adds new qubits to the circuit.
    ///
    /// Args:
    ///     qubits: A list of qubit indices to add.
    ///
    /// Raises:
    ///     ValueError: If any qubit already exists in the circuit.
    ///
    /// # Example
    ///
    /// ```python
    /// circuit = Circuit(2)  # Qubits 0, 1
    /// circuit.add_qubits([2, 3])  # Now has qubits 0, 1, 2, 3
    /// ```
    fn add_qubits(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        let qubits_core: Vec<Qubit> = qubits
            .into_iter()
            .map(|idx| Qubit::new(idx as u32))
            .collect();
        self.inner
            .add_qubits(qubits_core)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a conditional (if-else) operation to the circuit.
    ///
    /// Executes different quantum operations based on a classical condition
    /// (typically from a previous measurement).
    ///
    /// Args:
    ///     condition: The classical condition to evaluate (ConditionView)
    ///     true_body: List of operation tuples for the true branch.
    ///         Each tuple is (gate, qubits) or (gate, qubits, params).
    ///         - gate: The gate to apply (StandardGate, McGate, UnitaryGate)
    ///         - qubits: List of qubit indices
    ///         - params: Optional list of float parameters
    ///     false_body: Optional list of operation tuples for the false branch.
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib import Circuit, StandardGate
    /// from cqlib.circuit.gate import ConditionView
    ///
    /// circuit = Circuit(2)
    /// circuit.x(0)
    /// circuit.measure(0)
    ///
    /// # If qubit 0 is 1, apply X to qubit 1; otherwise apply Z
    /// condition = ConditionView(Qubit(0), 1)
    /// circuit.if_else(
    ///     condition,
    ///     [(StandardGate.X, [1])],      # true body
    ///     [(StandardGate.Z, [1])]       # false body
    /// )
    /// ```
    #[pyo3(signature = (condition, true_body, false_body=None))]
    fn if_else(
        &mut self,
        py: Python<'_>,
        condition: PyConditionView,
        true_body: Vec<PyOpTuple>,
        false_body: Option<Vec<PyOpTuple>>,
    ) -> PyResult<()> {
        let true_body_core = convert_op_tuples(py, self, true_body)?;
        let false_body_core = false_body
            .map(|ops| convert_op_tuples(py, self, ops))
            .transpose()?;

        self.inner
            .if_else(condition.inner, true_body_core, false_body_core)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Appends a while-loop operation to the circuit.
    ///
    /// Repeatedly executes quantum operations while a classical condition is true.
    ///
    /// Args:
    ///     condition: The classical condition to evaluate before each iteration
    ///     body: List of operation tuples for the loop body.
    ///         Each tuple is (gate, qubits) or (gate, qubits, params).
    ///         - gate: The gate to apply (StandardGate, McGate, UnitaryGate)
    ///         - qubits: List of qubit indices
    ///         - params: Optional list of float parameters
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib import Circuit, StandardGate
    /// from cqlib.circuit.gate import ConditionView
    ///
    /// circuit = Circuit(2)
    /// circuit.x(0)
    /// circuit.measure(0)
    ///
    /// # While qubit 0 equals 1, apply H to qubit 1
    /// condition = ConditionView(Qubit(0), 1)
    /// circuit.while_loop(condition, [(StandardGate.H, [1])])
    /// ```
    fn while_loop(
        &mut self,
        py: Python<'_>,
        condition: PyConditionView,
        body: Vec<PyOpTuple>,
    ) -> PyResult<()> {
        let body_core = convert_op_tuples(py, self, body)?;

        self.inner
            .while_loop(condition.inner, body_core)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// A tuple representing an operation for control flow bodies.
/// Format: (gate, qubits) or (gate, qubits, params)
/// - gate: StandardGate, McGate, or UnitaryGate
/// - qubits: List of qubit indices or Qubit objects
/// - params: Optional list of parameters (float or Parameter objects)
pub struct PyOpTuple {
    gate: Py<PyAny>,
    qubits: PyIntListOrQubitList,
    params: Option<Vec<PyParamLike>>,
}

impl<'py> FromPyObject<'_, 'py> for PyOpTuple {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok((gate, qubits)) = obj.extract::<(Py<PyAny>, PyIntListOrQubitList)>() {
            return Ok(PyOpTuple {
                gate,
                qubits,
                params: None,
            });
        }
        if let Ok((gate, qubits, params)) =
            obj.extract::<(Py<PyAny>, PyIntListOrQubitList, Option<Vec<PyParamLike>>)>()
        {
            return Ok(PyOpTuple {
                gate,
                qubits,
                params,
            });
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a tuple of (gate, qubits) or (gate, qubits, params)",
        ))
    }
}

/// Converts a list of operation tuples to core Operations.
fn convert_op_tuples(
    py: Python,
    circuit: &mut PyCircuit,
    ops: Vec<PyOpTuple>,
) -> PyResult<Vec<Operation>> {
    ops.into_iter()
        .map(|op_tuple| {
            let PyOpTuple {
                gate: gate_obj,
                qubits,
                params,
            } = op_tuple;
            // Extract instruction from gate object
            let instruction: Instruction;

            // Check the type of gate and extract the inner instruction
            if let Ok(std_gate) = gate_obj.cast_bound::<PyStandardGate>(py) {
                let py_gate = std_gate.extract::<PyStandardGate>()?;
                instruction = py_gate.inner.into();
            } else if let Ok(mc_gate) = gate_obj.cast_bound::<PyMcGate>(py) {
                let py_gate = mc_gate.extract::<PyMcGate>()?;
                instruction = Instruction::McGate(Box::new(py_gate.inner));
            } else if let Ok(u_gate) = gate_obj.cast_bound::<PyUnitaryGate>(py) {
                let py_gate = u_gate.extract::<PyUnitaryGate>()?;
                instruction = Instruction::UnitaryGate(Box::new(py_gate.into()));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Gate must be StandardGate, McGate, or UnitaryGate",
                ));
            }

            // Convert qubits
            // let qubits: Vec<Qubit> = qubits.iter().map(|&q| Qubit::new(q as u32)).collect();
            let qubits: SmallVec<[Qubit; 3]> = qubits.into_qubits().into_iter().collect();
            // Convert params - handle both fixed values and symbolic parameters
            let mut circuit_params = SmallVec::new();
            if let Some(params) = params {
                for p in params {
                    match p {
                        PyParamLike::Float(f) => {
                            circuit_params.push(CircuitParam::Fixed(f));
                        }
                        PyParamLike::Param(py_param) => {
                            let param = py_param.into_inner();
                            // Add parameter to circuit's parameter table
                            let (index, _) = circuit.inner.add_parameter(param);
                            circuit_params.push(CircuitParam::Index(index as u32));
                        }
                    }
                }
            }
            Ok(Operation {
                instruction,
                qubits,
                params: circuit_params,
                label: None,
            })
        })
        .collect()
}
