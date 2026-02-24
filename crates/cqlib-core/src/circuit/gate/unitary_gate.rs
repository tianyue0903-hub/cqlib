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

//! Custom Unitary Gate Definitions
//!
//! This module provides [`UnitaryGate`], a type for defining custom quantum gates
//! via their unitary matrix representation. Unlike [`StandardGate`](crate::circuit::gate::StandardGate),
//! which represents predefined gates, `UnitaryGate` allows users to specify arbitrary
//! unitary operations.

use crate::circuit::gate::circuit_gate::FrozenCircuit;
use ndarray::Array2;
use num_complex::Complex;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

/// A user-defined unitary quantum gate.
///
/// `UnitaryGate` represents a custom quantum gate defined by its unitary matrix
/// or by an internal circuit representation. Each gate has a unique identifier
/// for equality comparisons and hashing.
///
/// # Examples
///
/// ```
/// use cqlib_core::circuit::gate::UnitaryGate;
/// use ndarray::array;
/// use num_complex::Complex;
///
/// // Create a custom 1-qubit gate
/// let mut gate = UnitaryGate::new("MyGate", 1);
///
/// // Define the unitary matrix (Pauli-X as example)
/// let matrix = array![
///     [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
///     [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
/// ];
///
/// // Attach the matrix
/// let gate = gate.with_matrix(matrix).unwrap();
///
/// assert_eq!(gate.label(), "MyGate");
/// assert_eq!(gate.num_qubits(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct UnitaryGate
where
    Self: Send + Sync,
{
    /// Unique identifier for this gate definition.
    ///
    /// Used for equality comparisons and hashing. Each `UnitaryGate::new`
    /// call generates a fresh UUID.
    id: Uuid,
    /// A human-readable label for the gate (e.g., "QFT", "Oracle").
    label: Arc<String>,
    /// The matrix representation of the gate, wrapped in `Arc` for cheap cloning.
    ///
    /// Can be `None` if the gate is purely symbolic (defined by circuit only).
    matrix: Option<Arc<Array2<Complex<f64>>>>,
    /// The number of qubits this gate acts on.
    num_qubits: u16,
    /// Optional internal circuit representation.
    circuit: Option<Arc<FrozenCircuit>>,
}

impl UnitaryGate {
    /// Creates a new unitary gate definition without a matrix.
    ///
    /// The gate is assigned a unique ID and can later be configured with
    /// a matrix using [`with_matrix`](Self::with_matrix) or with a circuit
    /// using [`with_circuit`](Self::with_circuit).
    ///
    /// # Arguments
    ///
    /// * `label` - A descriptive name for the gate.
    /// * `num_qubits` - The number of qubits the gate operates on.
    ///
    /// # Returns
    ///
    /// A new `UnitaryGate` with no matrix attached.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::gate::UnitaryGate;
    ///
    /// let gate = UnitaryGate::new("QFT_3", 3);
    /// assert_eq!(gate.label(), "QFT_3");
    /// assert_eq!(gate.num_qubits(), 3);
    /// assert!(gate.matrix().is_none());
    /// ```
    pub fn new(label: &str, num_qubits: u16) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: Arc::new(label.to_string()),
            matrix: None,
            num_qubits,
            circuit: None,
        }
    }

    /// Returns the label of the gate.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::gate::UnitaryGate;
    ///
    /// let gate = UnitaryGate::new("Oracle", 2);
    /// assert_eq!(gate.label(), "Oracle");
    /// ```
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the number of qubits this gate acts on.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::gate::UnitaryGate;
    ///
    /// let gate = UnitaryGate::new("TwoQubitGate", 2);
    /// assert_eq!(gate.num_qubits(), 2);
    /// ```
    pub fn num_qubits(&self) -> u16 {
        self.num_qubits
    }

    /// Returns the matrix representation if available.
    ///
    /// # Returns
    ///
    /// - `Some(&Array2)`: The unitary matrix if it has been attached.
    /// - `None`: If no matrix was provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::gate::UnitaryGate;
    ///
    /// let gate = UnitaryGate::new("SymbolicGate", 1);
    /// assert!(gate.matrix().is_none());
    /// ```
    pub fn matrix(&self) -> Option<&Array2<Complex<f64>>> {
        self.matrix.as_deref()
    }

    /// Returns the internal circuit representation if available.
    ///
    /// Some unitary gates are defined by their circuit decomposition
    /// rather than an explicit matrix.
    pub fn circuit(&self) -> &Option<Arc<FrozenCircuit>> {
        &self.circuit
    }

    /// Attaches a matrix to the unitary definition.
    ///
    /// Consumes the gate and returns a new one with the matrix attached.
    /// The matrix dimensions must match the expected size for the gate's
    /// qubit count: $2^n \times 2^n$ where $n$ is `num_qubits`.
    ///
    /// # Arguments
    ///
    /// * `mat` - A square matrix of size $2^N \times 2^N$.
    ///
    /// # Returns
    ///
    /// - `Ok(Self)`: The gate with matrix attached.
    /// - `Err(String)`: Error message if dimensions are incorrect.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::gate::UnitaryGate;
    /// use ndarray::array;
    /// use num_complex::Complex;
    ///
    /// let gate = UnitaryGate::new("Hadamard", 1);
    ///
    /// // Correct 2x2 matrix for 1 qubit
    /// let h = Complex::new(1.0 / f64::sqrt(2.0), 0.0);
    /// let matrix = array![
    ///     [h, h],
    ///     [h, -h],
    /// ];
    ///
    /// let gate = gate.with_matrix(matrix).unwrap();
    /// assert!(gate.matrix().is_some());
    /// ```
    pub fn with_matrix(mut self, mat: Array2<Complex<f64>>) -> Result<Self, String> {
        let expected_dim = 1 << self.num_qubits;
        if mat.shape() != [expected_dim, expected_dim] {
            return Err(format!(
                "Matrix dimension mismatch. Expected {}x{}, got {}x{}",
                expected_dim,
                expected_dim,
                mat.nrows(),
                mat.ncols()
            ));
        }

        self.matrix = Some(Arc::new(mat));
        Ok(self)
    }

    /// Attaches a circuit representation to the unitary definition.
    ///
    /// This allows the gate to be defined by its circuit decomposition,
    /// which is useful for inverse operations and optimization.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The frozen circuit representing this gate.
    pub fn with_circuit(mut self, circuit: Arc<FrozenCircuit>) -> Self {
        self.circuit = Some(circuit);
        self
    }
}

impl Eq for UnitaryGate {}

impl PartialEq for UnitaryGate {
    /// Equality is based solely on the unique ID.
    ///
    /// Two `UnitaryGate` instances are considered equal only if they
    /// were created by the same constructor call (share the same UUID).
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for UnitaryGate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl fmt::Display for UnitaryGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.label().fmt(f)
    }
}
