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

//! Extended Quantum Gate Definitions
//!
//! This module defines gate structures that go beyond the fixed set of [`StandardGate`](crate::circuit::gate::standard_gate::StandardGate)s.
//! It primarily supports two advanced use cases:
//! 1. **Multi-Controlled Gates (`MCGate`)**: Generalizing any standard gate to have $N$ control qubits.
//! 2. **Custom Unitaries (`Unitary`)**: Allowing users to define arbitrary gates via matrices.

use crate::circuit::Parameter;
use crate::circuit::gate::gate_matrix;
use crate::circuit::gate::standard_gate::StandardGate;
use ndarray::Array2;
use num_complex::Complex;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

/// A definition for a custom Unitary gate.
///
/// `UnitaryDef` acts as a blueprint for user-defined gates. It contains metadata (label, qubit count)
/// and optionally the actual matrix representation.
///
/// # Identity and Equality
///
/// Each `UnitaryDef` is assigned a unique UUID upon creation. Equality checks (`Eq`, `PartialEq`)
/// and Hashing (`Hash`) are performed **solely based on this UUID**, not the matrix content or label.
/// This means two identical matrices created separately will be treated as distinct gate definitions.
#[derive(Debug, Clone)]
pub struct UnitaryDef {
    /// Unique identifier for this gate definition.
    pub(crate) id: Uuid,
    /// A human-readable label for the gate (e.g., "QFT", "Oracle").
    pub(crate) label: Arc<String>,
    /// The matrix representation of the gate. wrapped in `Arc` for cheap cloning.
    /// Can be `None` if the gate is purely symbolic.
    pub(crate) matrix: Option<Arc<Array2<Complex<f64>>>>,
    /// The number of qubits this gate acts on.
    pub(crate) num_qubits: u16,
}

impl UnitaryDef {
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
        }
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
}

impl Eq for UnitaryDef {}
impl PartialEq for UnitaryDef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for UnitaryDef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Represents extended quantum gates that are not part of the standard set.
///
/// This enum covers:
/// - **Generalized Controls**: Applying arbitrary controls to a standard gate (e.g., $C^3-H$).
/// - **Custom Operations**: User-defined unitary matrices.
#[derive(Eq, Hash, PartialEq, Debug, Clone)]
pub enum ExtendedGate {
    /// A wrapper around a [`StandardGate`] that adds extra control qubits.
    ///
    /// Structure: `MCGate(num_extra_controls, base_gate)`
    MCGate(u8, StandardGate),

    /// A custom unitary gate with optional controls.
    ///
    /// Structure: `Unitary(num_controls, num_targets, definition)`
    Unitary(u8, u8, UnitaryDef),
}

impl ExtendedGate {
    /// Computes the unitary matrix for the gate.
    ///
    /// If the gate has control qubits, this returns the full controlled matrix $C^k(U)$,
    /// which is a block diagonal matrix:
    /// $$
    /// \begin{pmatrix}
    /// I & 0 \\
    /// 0 & U
    /// \end{pmatrix}
    /// $$
    /// (Note: The exact structure depends on the basis ordering).
    ///
    /// # Arguments
    ///
    /// * `params` - Parameters for the underlying gate (if applicable).
    ///
    /// # Returns
    ///
    /// A `Cow<Array2>` containing the matrix. It may borrow static data or own a newly computed matrix.
    ///
    /// # Panics
    ///
    /// Panics if a `Unitary` gate is used but no matrix was provided in its definition (`UnitaryDef`).
    pub fn matrix(&self, params: &[f64]) -> Cow<'_, Array2<Complex<f64>>> {
        match self {
            Self::MCGate(ctrls, gate) => {
                let base_matrix = gate.matrix(params);
                if *ctrls == 0 {
                    return base_matrix;
                }
                // Construct controlled matrix
                let controlled = gate_matrix::control_matrix(&base_matrix, *ctrls as usize);
                Cow::Owned(controlled)
            }
            Self::Unitary(ctrls, _, def) => {
                // UnitaryDef currently stores a static matrix (Option<Arc<...>>)
                let base_matrix = def
                    .matrix
                    .as_ref()
                    .expect("Unitary definition must have a matrix for simulation");

                if *ctrls == 0 {
                    // Clone the Arc content to Cow.
                    return Cow::Owned((**base_matrix).clone());
                }

                let controlled = gate_matrix::control_matrix(base_matrix, *ctrls as usize);
                Cow::Owned(controlled)
            }
        }
    }

    /// Computes the inverse of the extended gate.
    ///
    /// # Returns
    ///
    /// - For `MCGate`: Returns a new `MCGate` wrapping the inverse of the standard gate.
    ///   The control qubits remain unchanged (i.e., $(CU)^\dagger = C(U^\dagger)$).
    /// - For `Unitary`: Currently returns `None` as symbolic inversion of arbitrary matrices is not supported.
    pub fn inverse(
        &self,
        params: &[Parameter],
    ) -> Option<(ExtendedGate, SmallVec<[Parameter; 3]>)> {
        match self {
            Self::MCGate(ctrls, gate) => {
                // The inverse of a controlled gate C(U) is C(U†).
                let (inv_gate, inv_params) = gate.inverse(params)?;
                Some((Self::MCGate(*ctrls, inv_gate), inv_params))
            }
            Self::Unitary(_, _, _) => {
                // Cannot automatically invert a custom unitary without more context.
                None
            }
        }
    }

    /// Returns the number of control qubits.
    pub fn num_ctrl_qubits(&self) -> usize {
        match self {
            Self::MCGate(c, g) => *c as usize + g.num_ctrl_qubits(),
            Self::Unitary(c, _, _) => *c as usize,
        }
    }

    /// Returns the total number of qubits (controls + targets).
    pub fn num_qubits(&self) -> usize {
        match self {
            Self::MCGate(c, g) => *c as usize + g.num_qubits(),
            Self::Unitary(c, t, _) => *c as usize + *t as usize,
        }
    }

    /// Returns the number of parameters required by the gate.
    pub fn num_params(&self) -> usize {
        match self {
            Self::MCGate(_, g) => g.num_params(),
            Self::Unitary(_, _, _) => 0,
        }
    }
}

impl fmt::Display for ExtendedGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtendedGate::MCGate(ctrls, gate) => {
                if *ctrls == 0 {
                    write!(f, "{}", gate)
                } else {
                    write!(f, "C{}-{}", ctrls, gate)
                }
            }
            ExtendedGate::Unitary(ctrls, _targets, def) => {
                if *ctrls == 0 {
                    write!(f, "Unitary({})", def.label)
                } else {
                    write!(f, "C{}-Unitary({})", ctrls, def.label)
                }
            }
        }
    }
}
