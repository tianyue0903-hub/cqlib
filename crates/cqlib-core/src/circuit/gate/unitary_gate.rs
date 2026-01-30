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

use crate::circuit::gate::circuit_gate::FrozenCircuit;
use ndarray::Array2;
use num_complex::Complex;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct UnitaryGate
where
    Self: Send + Sync,
{
    /// Unique identifier for this gate definition.
    id: Uuid,
    /// A human-readable label for the gate (e.g., "QFT", "Oracle").
    label: Arc<String>,
    /// The matrix representation of the gate. wrapped in `Arc` for cheap cloning.
    /// Can be `None` if the gate is purely symbolic.
    matrix: Option<Arc<Array2<Complex<f64>>>>,
    /// The number of qubits this gate acts on.
    num_qubits: u16,
    circuit: Option<Arc<FrozenCircuit>>,
}

impl UnitaryGate {
    /// Creates a new unitary gate definition without a matrix.
    ///
    /// # Arguments
    ///
    /// * `label` - A name for the gate.
    /// * `num_qubits` - The number of qubits the gate operates on.
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
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the number of qubits.
    pub fn num_qubits(&self) -> u16 {
        self.num_qubits
    }

    /// Returns the matrix representation if available.
    pub fn matrix(&self) -> Option<&Array2<Complex<f64>>> {
        self.matrix.as_deref()
    }

    /// Returns the matrix representation if available.
    pub fn circuit(&self) -> &Option<Arc<FrozenCircuit>> {
        &self.circuit
    }

    /// Attaches a matrix to the unitary definition.
    ///
    /// # Arguments
    ///
    /// * `mat` - A square matrix of size $2^N \times 2^N$.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if the matrix dimensions match `num_qubits`.
    /// Returns `Err(String)` if the dimensions are incorrect.
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

    pub fn with_circuit(mut self, circuit: Arc<FrozenCircuit>) -> Self {
        self.circuit = Some(circuit);
        self
    }
}

impl Eq for UnitaryGate {}
impl PartialEq for UnitaryGate {
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
