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

//! Python bindings for quantum device topology.
//!
//! This module provides [`PyTopology`], a Python wrapper around the core [`Topology`] type
//! that represents the coupling/connectivity structure of a quantum device.
//!
//! # Directed Couplings
//!
//! **Important:** Couplings in this topology are **directed** (asymmetric). A coupling from
//! qubit `a` to qubit `b` does **not** imply a coupling from `b` to `a`. This accurately
//! models real quantum hardware where CNOT gates may only work in specific directions.
//!
//! Use [`PyTopology::is_connected`] to check for a coupling in a specific direction, or
//! check both directions for bidirectional connectivity.
//!
//! # Example
//!
//! ```python
//! from cqlib.device import Topology
//!
//! # Create a topology with directed couplings
//! topology = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])
//!
//! # Check directed connectivity
//! topology.is_connected(0, 1)  # True: 0 -> 1 coupling exists
//! topology.is_connected(1, 0)  # False: 1 -> 0 coupling does NOT exist
//!
//! # Get neighbors (qubits reachable via outgoing couplings)
//! topology.neighbors(0)  # [1]
//! ```

use crate::circuit::PyQubit;
use crate::circuit::bit::{PyIntListOrQubitList, PyIntOrQubit};
use cqlib_core::circuit::Qubit;
use cqlib_core::device::topology::Topology;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyclass, pymethods};

/// A directed coupling graph representing quantum hardware connectivity.
///
/// Each node represents a physical qubit, and each directed edge represents a
/// coupling between qubits (e.g., for two-qubit gates like CNOT).
///
/// # Directed vs Undirected
///
/// This topology is **directed**. A coupling `a -> b` does not imply `b -> a`.
/// This models hardware where gates only work in specific directions.
///
/// # Python Example
///
/// ```python
/// from cqlib import Topology
///
/// # Create topology with explicit qubit list and directed couplings
/// topology = Topology(
///     qubits=[0, 1, 2, 3],
///     couplings=[(0, 1, "CX"), (1, 2, "CX"), (2, 3, "CX")]
/// )
///
/// # Query properties
/// topology.num_qubits      # 4
/// topology.num_couplings   # 3
/// topology.qubits          # [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]
///
/// # Check connectivity
/// topology.is_connected(0, 1)  # True (0 -> 1)
/// topology.is_connected(1, 0)  # False (no 1 -> 0 coupling)
/// ```
#[pyclass(name = "Topology", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyTopology {
    /// The underlying core topology.
    pub(crate) inner: Topology,
}

#[pymethods]
impl PyTopology {
    /// Creates a new topology with specified qubits and directed couplings.
    ///
    /// Args:
    ///     qubits: List of qubit identifiers (integers or Qubit objects).
    ///         Duplicate qubits are silently deduplicated.
    ///     couplings: List of directed couplings as tuples `(control, target, name)`.
    ///         Each coupling is directed from `control` to `target`.
    ///         Couplings referencing non-existent qubits are silently ignored.
    ///
    /// Returns:
    ///     Topology: A new topology instance.
    ///
    /// Example:
    ///     ```python
    ///     from cqlib import Topology
    ///
    ///     # Directed line: 0 -> 1 -> 2
    ///     topology = Topology(
    ///         qubits=[0, 1, 2],
    ///         couplings=[(0, 1, "CX"), (1, 2, "CX")]
    ///     )
    ///     ```
    #[new]
    #[pyo3(signature = (qubits, couplings))]
    fn new(
        qubits: PyIntListOrQubitList,
        couplings: Vec<(PyIntOrQubit, PyIntOrQubit, String)>,
    ) -> PyResult<Self> {
        let qubits: Vec<Qubit> = qubits.into();
        let couplings: Vec<(Qubit, Qubit, String)> = couplings
            .into_iter()
            .map(|(q0, q1, s)| (q0.into(), q1.into(), s))
            .collect();
        let inner =
            Topology::new(qubits, couplings).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Creates a line topology with directed couplings between adjacent qubits.
    ///
    /// Constructs a linear chain where each qubit has a directed coupling to
    /// its next neighbor: `q[0] -> q[1] -> q[2] -> ...`
    ///
    /// Args:
    ///     qubits: List of qubit identifiers in line order.
    ///
    /// Returns:
    ///     Topology: A new line topology with `len(qubits) - 1` couplings.
    ///
    /// Raises:
    ///     ValueError: If fewer than 2 qubits are provided.
    ///
    /// Example:
    ///     ```python
    ///     from cqlib import Topology
    ///
    ///     # Creates couplings: 0 -> 1, 1 -> 2, 2 -> 3
    ///     topology = Topology.line([0, 1, 2, 3])
    ///     ```
    #[staticmethod]
    fn line(qubits: PyIntListOrQubitList) -> PyResult<Self> {
        let qubits: Vec<Qubit> = qubits.into();

        if qubits.len() < 2 {
            return Err(PyValueError::new_err(
                "Line topology requires at least 2 qubits",
            ));
        }

        let couplings: Vec<(Qubit, Qubit, String)> = qubits
            .windows(2)
            .map(|qs| (qs[0], qs[1], String::new()))
            .collect();

        let inner =
            Topology::new(qubits, couplings).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the number of physical qubits in the topology.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the number of directed coupling edges.
    #[getter]
    fn num_couplings(&self) -> usize {
        self.inner.num_couplings()
    }

    /// Returns all physical qubits in the topology.
    #[getter]
    fn qubits(&self) -> Vec<PyQubit> {
        self.inner.qubits().into_iter().map(|q| q.into()).collect()
    }

    /// Adds physical qubits to the topology.
    ///
    /// Args:
    ///     qubits: List of qubit identifiers to add.
    ///
    /// Raises:
    ///     ValueError: If any qubit already exists in the topology.
    fn add_qubits(&mut self, qubits: PyIntListOrQubitList) -> PyResult<()> {
        self.inner
            .add_qubits(<PyIntListOrQubitList as Into<Vec<Qubit>>>::into(qubits))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Adds directed couplings to the topology.
    ///
    /// Each coupling is directed from the first qubit to the second.
    ///
    /// Args:
    ///     couplings: List of tuples `(control, target, name)` where:
    ///         - `control`: Source qubit of the directed coupling
    ///         - `target`: Destination qubit of the directed coupling
    ///         - `name`: String identifier for the coupling (e.g., "CX", "CZ")
    ///
    /// Raises:
    ///     ValueError: If either endpoint qubit does not exist in the topology.
    fn add_couplings(
        &mut self,
        couplings: Vec<(PyIntOrQubit, PyIntOrQubit, String)>,
    ) -> PyResult<()> {
        self.inner
            .add_couplings(
                couplings
                    .into_iter()
                    .map(|(q0, q1, s)| (q0.into(), q1.into(), s)),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Removes physical qubits and all their incident couplings from the topology.
    ///
    /// When a qubit is removed, all directed couplings where it is either the
    /// source or target are also removed.
    ///
    /// Args:
    ///     qubits: List of qubit identifiers to remove.
    ///
    /// Raises:
    ///     ValueError: If any qubit does not exist in the topology.
    fn remove_qubits(&mut self, qubits: PyIntListOrQubitList) -> PyResult<()> {
        self.inner
            .remove_qubits(<PyIntListOrQubitList as Into<Vec<Qubit>>>::into(qubits))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Removes directed couplings from the topology.
    ///
    /// Only removes the specific directed coupling. The reverse coupling
    /// (if present) is not affected.
    ///
    /// Args:
    ///     couplings: List of `(control, target)` tuples specifying directed
    ///         couplings to remove.
    ///
    /// Raises:
    ///     ValueError: If a coupling does not exist or endpoint qubits are missing.
    fn remove_couplings(&mut self, couplings: Vec<(PyIntOrQubit, PyIntOrQubit)>) -> PyResult<()> {
        self.inner
            .remove_couplings(couplings.into_iter().map(|(q0, q1)| (q0.into(), q1.into())))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Checks for a directed coupling from `u` to `v`.
    ///
    /// Returns `True` if there is a directed coupling edge from qubit `u`
    /// to qubit `v`. This does **not** check the reverse direction.
    ///
    /// Args:
    ///     u: Source qubit (control).
    ///     v: Target qubit (target).
    ///
    /// Returns:
    ///     bool: `True` if `u -> v` coupling exists, `False` otherwise.
    ///
    /// Example:
    ///     ```python
    ///     topology = Topology([0, 1], [(0, 1, "CX")])
    ///     topology.is_connected(0, 1)  # True: 0 -> 1 exists
    ///     topology.is_connected(1, 0)  # False: 1 -> 0 does not exist
    ///     ```
    fn is_connected(&self, u: PyIntOrQubit, v: PyIntOrQubit) -> bool {
        self.inner.is_connected(u.into(), v.into())
    }

    /// Returns neighbors reachable via outgoing couplings from a qubit.
    ///
    /// Returns all qubits `v` such that a directed coupling `qubit -> v` exists.
    ///
    /// Args:
    ///     qubit: The source qubit.
    ///
    /// Returns:
    ///     List[Qubit]: List of qubits reachable via outgoing couplings.
    ///
    /// Example:
    ///     ```python
    ///     topology = Topology([0, 1, 2], [(0, 1, "CX"), (0, 2, "CX")])
    ///     topology.neighbors(0)  # [Qubit(1), Qubit(2)]
    ///     ```
    fn neighbors(&self, qubit: PyIntOrQubit) -> Vec<PyQubit> {
        self.inner
            .neighbors(qubit.into())
            .map(|q| q.into())
            .collect()
    }

    /// Returns the name of the directed coupling from `u` to `v`, if it exists.
    ///
    /// Args:
    ///     u: Source qubit.
    ///     v: Target qubit.
    ///
    /// Returns:
    ///     Optional[str]: The coupling name if `u -> v` exists, `None` otherwise.
    fn get_coupling_name(&self, u: PyIntOrQubit, v: PyIntOrQubit) -> Option<String> {
        self.inner.get_coupling_name(u.into(), v.into())
    }

    /// Checks if a qubit exists in the topology.
    ///
    /// Args:
    ///     qubit: Qubit to check.
    ///
    /// Returns:
    ///     bool: `True` if the qubit exists in the topology.
    fn contains_qubit(&self, qubit: PyIntOrQubit) -> bool {
        self.inner.contains_qubit(&qubit.into())
    }

    /// Returns the out-degree of a qubit (number of outgoing couplings).
    ///
    /// Returns the count of directed couplings where this qubit is the source.
    ///
    /// Args:
    ///     qubit: The qubit to query.
    ///
    /// Returns:
    ///     int: Number of outgoing couplings. Returns 0 if qubit does not exist.
    fn degree(&self, qubit: PyIntOrQubit) -> usize {
        self.inner.degree(&qubit.into())
    }

    /// Returns a string representation for debugging.
    ///
    /// Example: `Topology(num_qubits=4, num_couplings=3)`
    fn __repr__(&self) -> String {
        format!(
            "Topology(num_qubits={}, num_couplings={})",
            self.inner.num_qubits(),
            self.inner.num_couplings()
        )
    }
}
