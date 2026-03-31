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

//! Python bindings for qubit layout management.
//!
//! This module provides [`PyLayout`], a Python wrapper for mapping logical (virtual)
//! qubits to physical qubits on a quantum device. Layouts are essential for circuit
//! routing algorithms like SABRE that track how virtual qubits move across hardware.
//!
//! # Concepts
//!
//! - **Logical qubits**: Virtual qubits used in the quantum circuit (Q0, Q1, ...)
//! - **Physical qubits**: Actual hardware qubits on the device (P100, P101, ...)
//! - **Ancilla qubits**: Automatically generated auxiliary qubits to fill unused physical qubits
//!
//! # Example
//!
//! ```python
//! from cqlib.device import Layout
//!
//! # Create a layout with 2 logical qubits mapped to 4 physical qubits
//! layout = Layout(
//!     logical=[0, 1],
//!     physical=[100, 101, 102, 103]
//! )
//!
//! # Query mappings
//! physical = layout.get_physical(0)  # Qubit(100)
//! virtual = layout.get_virtual(101)   # Qubit(1)
//!
//! # Perform SWAP operation (used in routing)
//! layout.swap_physical(100, 101)
//! ```

use crate::circuit::PyQubit;
use crate::circuit::bit::{PyIntListOrQubitList, PyIntOrQubit};
use cqlib_core::circuit::Qubit;
use cqlib_core::device::Layout;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyclass, pymethods};
use std::collections::HashMap;

/// Maps logical (virtual) qubits to physical qubits on a quantum device.
///
/// A layout represents the current assignment of virtual qubits to physical hardware.
/// It maintains bidirectional mappings and is used by routing algorithms to track
/// qubit placement and update mappings when SWAP gates are inserted.
///
/// # Ancilla Qubits
///
/// When the number of logical qubits is less than physical qubits, ancilla qubits
/// are automatically created to fill the gap. These are auxiliary qubits used
/// during circuit execution.
///
/// # Example
///
/// ```python
/// from cqlib.device import Layout
///
/// # Create layout with initial mapping
/// init_map = {0: 100, 1: 101}  # logical 0 -> physical 100
/// layout = Layout(
///     logical=[0, 1],
///     physical=[100, 101, 102],
///     init_map=init_map
/// )
///
/// # Check ancilla count
/// print(layout.num_ancilla)  # 1 (physical qubit 102)
///
/// # Get all mappings
/// print(layout.v2p_map)  # {Qubit(0): Qubit(100), Qubit(1): Qubit(101), ...}
/// ```
#[pyclass(name = "Layout", module = "cqlib.device")]
#[derive(Clone, Debug)]
pub struct PyLayout {
    pub(crate) inner: Layout,
}

impl From<Layout> for PyLayout {
    fn from(inner: Layout) -> Self {
        Self { inner }
    }
}

impl From<PyLayout> for Layout {
    fn from(value: PyLayout) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyLayout {
    /// Creates a new layout mapping logical qubits to physical qubits.
    ///
    /// # Arguments
    ///
    /// * `logical`: List of logical (virtual) qubit identifiers.
    /// * `physical`: List of physical qubit identifiers available on the device.
    /// * `init_map`: Optional initial mapping from logical to physical qubits.
    ///     If not provided, logical qubits are mapped sequentially to physical qubits.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if:
    /// - The number of logical qubits exceeds physical qubits
    /// - `init_map` contains invalid virtual or physical qubits
    /// - `init_map` maps multiple virtual qubits to the same physical qubit
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib.device import Layout
    ///
    /// # Automatic sequential mapping
    /// layout = Layout(logical=[0, 1], physical=[100, 101, 102])
    ///
    /// # Custom initial mapping
    /// layout = Layout(
    ///     logical=[0, 1],
    ///     physical=[100, 101, 102],
    ///     init_map={0: 101, 1: 100}  # Swap mapping
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (logical, physical, init_map=None))]
    fn new(
        logical: PyIntListOrQubitList,
        physical: PyIntListOrQubitList,
        init_map: Option<HashMap<PyQubit, PyQubit>>,
    ) -> PyResult<Self> {
        let logical = <PyIntListOrQubitList as Into<Vec<Qubit>>>::into(logical)
            .into_iter()
            .collect();
        let physical = <PyIntListOrQubitList as Into<Vec<Qubit>>>::into(physical)
            .into_iter()
            .collect();
        let init_map = init_map
            .map(|m| {
                m.into_iter()
                    .map(|(v, p)| Ok((v.inner, p.inner)))
                    .collect::<PyResult<HashMap<_, _>>>()
            })
            .transpose()?;

        let inner = Layout::new(logical, physical, init_map)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Returns the number of logical qubits.
    #[getter]
    fn num_logical(&self) -> usize {
        self.inner.num_logical()
    }

    /// Returns the number of ancilla qubits.
    ///
    /// Ancilla qubits are automatically generated to fill unused physical qubits
    /// when `len(logical) < len(physical)`.
    #[getter]
    fn num_ancilla(&self) -> usize {
        self.inner.num_ancilla()
    }

    /// Returns the number of physical qubits.
    #[getter]
    fn num_physical(&self) -> usize {
        self.inner.num_physical()
    }

    /// Returns the physical qubit mapped to a virtual qubit.
    ///
    /// # Arguments
    ///
    /// * `virtual_id`: The logical qubit identifier.
    ///
    /// # Returns
    ///
    /// `Qubit` if the logical qubit is mapped, `None` otherwise.
    fn get_physical(&self, virtual_id: PyIntOrQubit) -> PyResult<Option<PyQubit>> {
        Ok(self
            .inner
            .get_physical(virtual_id.into())
            .map(PyQubit::from))
    }

    /// Returns the virtual qubit mapped to a physical qubit.
    ///
    /// # Arguments
    ///
    /// * `physical_id`: The physical qubit identifier.
    ///
    /// # Returns
    ///
    /// `Qubit` if a virtual qubit is mapped to this physical qubit, `None` otherwise
    /// (e.g., if it's an unmapped ancilla).
    fn get_virtual(&self, physical_id: PyIntOrQubit) -> PyResult<Option<PyQubit>> {
        Ok(self
            .inner
            .get_virtual(physical_id.into())
            .map(PyQubit::from))
    }

    /// Returns all logical qubits in the layout.
    #[getter]
    fn logical_qubits(&self) -> Vec<PyQubit> {
        self.inner.logical_qubits().map(PyQubit::from).collect()
    }

    /// Returns all ancilla qubits in the layout.
    #[getter]
    fn ancilla_qubits(&self) -> Vec<PyQubit> {
        self.inner.ancilla_qubits().map(PyQubit::from).collect()
    }

    /// Returns all physical qubits in the layout.
    #[getter]
    fn physical_qubits(&self) -> Vec<PyQubit> {
        self.inner.physical_qubits().map(PyQubit::from).collect()
    }

    /// Returns the virtual-to-physical qubit mapping.
    ///
    /// Returns a dictionary mapping each virtual qubit (including ancillas)
    /// to its assigned physical qubit.
    #[getter]
    fn v2p_map(&self) -> HashMap<PyQubit, PyQubit> {
        self.inner
            .v2p_map()
            .iter()
            .map(|(k, v)| (PyQubit::from(*k), PyQubit::from(*v)))
            .collect()
    }

    /// Returns the physical-to-virtual qubit mapping.
    ///
    /// Returns a dictionary mapping each physical qubit to its assigned
    /// virtual qubit (if any). Unmapped physical qubits are not included.
    #[getter]
    fn p2v_map(&self) -> HashMap<PyQubit, PyQubit> {
        self.inner
            .p2v_map()
            .iter()
            .map(|(k, v)| (PyQubit::from(*k), PyQubit::from(*v)))
            .collect()
    }

    /// Swaps the virtual qubits mapped to two physical qubits.
    ///
    /// This is the core operation used by routing algorithms (e.g., SABRE) when
    /// inserting SWAP gates. After a SWAP gate is applied on the hardware,
    /// the virtual qubits on those physical qubits are exchanged.
    ///
    /// # Arguments
    ///
    /// * `phys_a`: First physical qubit.
    /// * `phys_b`: Second physical qubit.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if either physical qubit is not in the layout.
    ///
    /// # Example
    ///
    /// ```python
    /// from cqlib.device import Layout
    ///
    /// layout = Layout(logical=[0], physical=[100, 101])
    ///
    /// # Before swap: Qubit 0 is on some physical qubit
    /// phys_before = layout.get_physical(0)
    ///
    /// # Perform SWAP on physical qubits
    /// layout.swap_physical(100, 101)
    ///
    /// # After swap, Qubit 0 is on the other physical qubit
    /// phys_after = layout.get_physical(0)
    /// assert phys_before != phys_after
    /// ```
    fn swap_physical(&mut self, phys_a: PyIntOrQubit, phys_b: PyIntOrQubit) -> PyResult<()> {
        let phys_a = phys_a.into();
        let phys_b = phys_b.into();
        self.inner
            .swap_physical(phys_a, phys_b)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Layout(num_logical={}, num_ancilla={}, num_physical={})",
            self.num_logical(),
            self.num_ancilla(),
            self.num_physical()
        )
    }
}
