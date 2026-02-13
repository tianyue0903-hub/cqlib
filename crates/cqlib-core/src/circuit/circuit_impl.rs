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

//! # Quantum Circuit Module
//!
//! This module defines the [`Circuit`] struct, which is the primary container for quantum programs
//! in the `Cqlib` ecosystem. It acts as an intermediate representation (IR) that captures the sequence
//! of quantum operations, qubit management, and symbolic parameters.
//!
//! ## Core Features
//!
//! - **Instruction Scheduling**: Stores a sequence of operations ([`Operation`]) including gates, measurements, and barriers.
//! - **Qubit Management**: Efficiently handles qubit allocation using topological ordering.
//! - **Parametric Circuits**: Native support for variational quantum algorithms (VQA) via symbolic parameters.
//!   Parameters are "interned" to minimize memory usage and accelerate bulk evaluation.
//! - **Extensibility**: Supports standard gates, custom unitary matrices, and arbitrary control structures.
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::circuit::circuit_impl::Circuit;
//! use cqlib_core::circuit::Qubit;
//!
//! // Create a circuit with 2 qubits
//! let mut circuit = Circuit::new(2);
//!
//! let q0 = Qubit::new(0);
//! let q1 = Qubit::new(1);
//!
//! // Apply Hadamard gate to q0
//! circuit.h(q0);
//!
//! // Apply Controlled-NOT gate (q0 controls q1)
//! circuit.cx(q0, q1);
//!
//! // Measure q0
//! circuit.measure(q0);
//! ```

use crate::circuit::bit::Qubit;
use crate::circuit::circuit_to_matrix;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::circuit_gate::{CircuitGate, FrozenCircuit};
use crate::circuit::gate::{Directive, Instruction, StandardGate, UnitaryGate};
use crate::circuit::operation::Operation;
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::parameter::Parameter;
use indexmap::IndexSet;
use ndarray::Array2;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

/// A quantum circuit representation serving as the core IR for quantum programs.
///
/// The `Circuit` struct is designed to be a high-performance, memory-efficient container for quantum
/// operations. It supports both static circuits (fixed angles) and parameterized circuits (symbolic angles),
/// making it suitable for a wide range of applications from error correction to variational quantum algorithms.
///
/// # Internal Architecture
///
/// - **Qubit Storage**: Uses `IndexSet<Qubit>` to maintain deterministic ordering of qubits while allowing $O(1)$ lookups.
/// - **Parameter Interning**: Symbolic parameters are stored in a centralized `IndexSet`. Instructions reference these parameters
///   by index rather than owning them. This "interning" strategy significantly reduces memory footprint for deep parameterized
///   circuits and enables vectorized parameter updates.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// The set of quantum bits (qubits) managed by this circuit.
    ///
    /// # Implementation Note
    /// Used `IndexSet` to maintain the strict insertion order of qubits (which defines the logical
    /// qubit indices 0, 1, 2...) while allowing $O(1)$ membership testing (`contains`).
    qubits: IndexSet<Qubit>,
    /// A registry of all unique symbolic variables (e.g., "theta", "phi") used within the circuit.
    /// This field serves as a cache to quickly identify which free parameters need to be bound
    /// before simulation, avoiding the need to traverse the entire instruction list.
    symbols: IndexSet<String>,
    /// The centralized storage for symbolic parameters.
    ///
    /// This table implements the **Interning** pattern. Instructions in the `data` vector do not
    /// own their `Parameter` objects; instead, they store lightweight indices pointing to this set.
    /// This design allows for:
    /// 1. **Deduplication**: Identical expressions are stored only once.
    /// 2. **Batch Evaluation**: All parameters can be resolved to `f64` values in a single linear pass.
    parameters: IndexSet<Parameter>,
    /// The ordered sequence of operations (quantum gates, measurements, etc.) in the circuit.
    ///
    /// This vector represents the circuit schedule.
    data: Vec<Operation>,
    ///  The global phase of the circuit, representing a scalar factor $e^{i\theta}$.
    ///
    /// While the global phase is unobservable in isolated systems, it is critical for:
    /// - **Controlled Operations**: When this circuit is controlled by another qubit.
    /// - **Sub-circuit Composition**: Correctly merging phases when combining circuits.
    global_phase: CircuitParam,
}

impl From<usize> for Circuit {
    fn from(num_qubits: usize) -> Self {
        Circuit::new(num_qubits)
    }
}

impl Circuit {
    /// Creates a new, empty quantum circuit with a specified number of qubits.
    ///
    /// The qubits will be automatically indexed from `0` to `num_qubits - 1`.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits to initialize in the circuit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cqlib_core::circuit::circuit_impl::Circuit;
    ///
    /// let circuit = Circuit::new(5);
    /// assert_eq!(circuit.num_qubits(), 5);
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        let qubits = (0..num_qubits).map(|i| Qubit::new(i as u32)).collect();

        Self {
            qubits,
            data: vec![],
            symbols: IndexSet::default(),
            parameters: IndexSet::default(),
            global_phase: CircuitParam::Fixed(0.0),
        }
    }

    /// Creates a circuit from a specific list of qubits.
    ///
    /// This is useful when you want to define a sub-circuit or use non-contiguous qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of `Qubit` identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::DuplicateQubits`] if the input vector contains duplicate qubits.
    pub fn from_qubits(qubits: Vec<Qubit>) -> Result<Circuit, CircuitError> {
        if !Self::check_qubits_unique(&qubits) {
            return Err(CircuitError::DuplicateQubits);
        }

        Ok(Self {
            symbols: IndexSet::new(),
            qubits: qubits.into_iter().collect(),
            data: vec![],
            parameters: IndexSet::default(),
            global_phase: CircuitParam::Fixed(0.0),
        })
    }

    /// Adds new qubits to the existing circuit.
    ///
    /// # Arguments
    ///
    /// * `new_qubits` - A vector of new `Qubit` identifiers to add.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::DuplicateQubits`] if any of the new qubits already exist in the circuit
    /// or if `new_qubits` contains duplicates.
    pub fn add_qubits(&mut self, new_qubits: Vec<Qubit>) -> Result<(), CircuitError> {
        let mut seen_new = HashSet::with_capacity(new_qubits.len());

        for q in &new_qubits {
            if self.qubits.contains(q) {
                return Err(CircuitError::DuplicateQubits);
            }

            if !seen_new.insert(*q) {
                return Err(CircuitError::DuplicateQubits);
            }
        }
        self.qubits.extend(new_qubits);
        Ok(())
    }

    /// Returns the number of qubits in the circuit.
    ///
    /// Alias for `num_qubits()`.
    pub fn width(&self) -> usize {
        self.qubits.len()
    }

    /// Returns the number of qubits in the circuit.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Returns the parameters of the circuit.
    pub fn parameters(&self) -> &IndexSet<Parameter> {
        &self.parameters
    }

    pub fn symbols(&self) -> &IndexSet<String> {
        &self.symbols
    }
    /// Returns a vector of all qubits in the circuit, preserving their insertion order.
    pub fn qubits(&self) -> Vec<Qubit> {
        self.qubits.iter().cloned().collect()
    }

    /// Returns the global phase of the circuit as a `Parameter`.
    pub fn global_phase(&self) -> Parameter {
        match self.global_phase {
            CircuitParam::Index(index) => self.parameters[index as usize].clone(),
            CircuitParam::Fixed(value) => Parameter::from(value),
        }
    }

    pub fn operations(&self) -> &[Operation] {
        &self.data
    }

    /// Appends a generic instruction to the circuit.
    ///
    /// This is the low-level method used by all specific gate methods (e.g., `h`, `cx`).
    /// It handles parameter interning and qubit validation.
    ///
    /// # Arguments
    ///
    /// * `instruction` - The instruction to append (Standard, Extended, or Directive).
    /// * `qubits` - The qubits this instruction acts upon.
    /// * `params` - The parameters for the instruction (if any).
    /// * `label` - An optional label for the operation.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::QubitNotFound`] if any of the specified qubits are not present in the circuit.
    pub fn append<Q, P>(
        &mut self,
        instruction: Instruction,
        qubits: Q,
        params: P,
        label: Option<&str>,
    ) -> Result<(), CircuitError>
    where
        Q: IntoIterator,
        Q::Item: Into<Qubit>,
        P: IntoIterator<Item = ParameterValue>,
    {
        let qubits_sv: SmallVec<[Qubit; 3]> = qubits.into_iter().map(|q| q.into()).collect();
        for qubit in &qubits_sv {
            if !self.qubits.contains(qubit) {
                return Err(CircuitError::QubitNotFound(qubit.id()));
            }
        }

        let mut circuit_params = smallvec![];
        for p in params {
            match p {
                ParameterValue::Param(param) => {
                    let (index, is_new) = self.parameters.insert_full(param.clone());
                    if is_new {
                        for sym in param.get_symbols() {
                            self.symbols.insert(sym);
                        }
                    }
                    circuit_params.push(CircuitParam::Index(index as u32));
                }
                ParameterValue::Fixed(value) => circuit_params.push(CircuitParam::Fixed(value)),
            }
        }

        self.data.push(Operation {
            instruction,
            qubits: qubits_sv,
            params: circuit_params,
            label: label.map(Into::into),
        });

        Ok(())
    }

    /// Appends a Hadamard (H) gate.
    ///
    /// The H gate creates a superposition state: $H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$.
    pub fn h(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::H),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    // --- Pauli Gates ---

    /// Appends an Identity (I) gate.
    ///
    /// This is a no-op gate, often used for alignment or waiting.
    pub fn i(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::I),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Pauli-X (NOT) gate.
    ///
    /// Performs a bit flip: $X|0\rangle = |1\rangle, X|1\rangle = |0\rangle$.
    pub fn x(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Pauli-Y gate.
    ///
    /// Performs a bit and phase flip: $Y|0\rangle = i|1\rangle, Y|1\rangle = -i|0\rangle$.
    pub fn y(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Pauli-Z gate.
    ///
    /// Performs a phase flip: $Z|0\rangle = |0\rangle, Z|1\rangle = -|1\rangle$.
    pub fn z(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Z),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a $\sqrt{X}$ (SX) gate.
    ///
    /// A 90-degree rotation around the X-axis. $SX^2 = X$.
    pub fn x2p(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X2P),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a $\sqrt{X}^\dagger$ (SXdg) gate.
    ///
    /// The inverse of the SX gate.
    pub fn x2m(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X2M),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a $\sqrt{Y}$ gate.
    pub fn y2p(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y2P),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a $\sqrt{Y}^\dagger$ gate.
    pub fn y2m(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y2M),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends an XY gate.
    ///
    /// Rotation between the $|01\rangle$ and $|10\rangle$ subspace.
    pub fn xy(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a $\sqrt{XY}$ gate (positive phase).
    pub fn xy2p(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY2P),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a $\sqrt{XY}^\dagger$ gate (negative phase).
    pub fn xy2m(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY2M),
            [qubit],
            params,
            None,
        )
    }

    // --- Clifford & Phase Gates ---

    /// Appends an S (Phase) gate.
    ///
    /// Applies a phase of $i$ to the $|1\rangle$ state ($Z^{1/2}$).
    pub fn s(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::S),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends an S-dagger ($S^\dagger$) gate.
    ///
    /// Applies a phase of $-i$ to the $|1\rangle$ state.
    pub fn sdg(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::SDG),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a T gate.
    ///
    /// Applies a phase of $e^{i\pi/4}$ to the $|1\rangle$ state ($Z^{1/4}$).
    pub fn t(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::T),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a T-dagger ($T^\dagger$) gate.
    pub fn tdg(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::TDG),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    // --- Parametric Rotations ---

    /// Appends a rotation around the X-axis by angle `theta`.
    ///
    /// $RX(\theta) = e^{-i\theta X/2}$
    pub fn rx(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::RX),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a rotation around the Y-axis by angle `theta`.
    ///
    /// $RY(\theta) = e^{-i\theta Y/2}$
    pub fn ry(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RY),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a rotation around the Z-axis by angle `theta`.
    ///
    /// $RZ(\theta) = e^{-i\theta Z/2}$
    pub fn rz(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZ),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a Phase gate (P gate).
    ///
    /// Applies a phase of $e^{i\lambda}$ to the $|1\rangle$ state.
    /// Equivalent to $RZ(\lambda)$ up to a global phase.
    pub fn phase(
        &mut self,
        qubit: Qubit,
        lambda: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![lambda.into()];

        self.append(
            Instruction::Standard(StandardGate::Phase),
            [qubit],
            params,
            None,
        )
    }

    /// Appends a generic single-qubit rotation gate $U(\theta, \phi, \lambda)$.
    ///
    /// This is the most general single-qubit unitary gate.
    /// $$
    /// U(\theta, \phi, \lambda) = \begin{pmatrix}
    /// \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
    /// e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
    /// \end{pmatrix}
    /// $$
    pub fn u(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
        phi: impl Into<ParameterValue>,
        lambda: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> =
            smallvec![theta.into(), phi.into(), lambda.into()];

        self.append(
            Instruction::Standard(StandardGate::U),
            [qubit],
            params,
            None,
        )
    }

    // --- Two-Qubit Gates ---

    /// Appends a Controlled-NOT (CX or CNOT) gate.
    ///
    /// Flips the `target` qubit if and only if the `control` qubit is $|1\rangle$.
    pub fn cx(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CX),
            [control, target],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Controlled-Y (CY) gate.
    pub fn cy(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CY),
            [control, target],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Controlled-Z (CZ) gate.
    ///
    /// Adds a phase of -1 only if both qubits are $|1\rangle$.
    pub fn cz(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CZ),
            [control, target],
            std::iter::empty(),
            None,
        )
    }

    /// Appends a SWAP gate.
    ///
    /// Exchange the states of two qubits.
    pub fn swap(&mut self, a: Qubit, b: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::SWAP),
            [a, b],
            std::iter::empty(),
            None,
        )
    }

    /// Appends an Ising XX coupling gate ($R_{XX}(\theta)$).
    ///
    /// $R_{XX}(\theta) = e^{-i\theta X \otimes X / 2}$
    pub fn rxx(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RXX),
            [a, b],
            params,
            None,
        )
    }

    /// Appends an Ising YY coupling gate ($R_{YY}(\theta)$).
    ///
    /// $R_{YY}(\theta) = e^{-i\theta Y \otimes Y / 2}$
    pub fn ryy(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RYY),
            [a, b],
            params,
            None,
        )
    }

    /// Appends an Ising ZZ coupling gate ($R_{ZZ}(\theta)$).
    ///
    /// $R_{ZZ}(\theta) = e^{-i\theta Z \otimes Z / 2}$
    pub fn rzz(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZZ),
            [a, b],
            params,
            None,
        )
    }

    /// Appends an Ising ZX coupling gate ($R_{ZX}(\theta)$).
    ///
    /// $R_{ZX}(\theta) = e^{-i\theta Z \otimes X / 2}$
    pub fn rzx(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZX),
            [a, b],
            params,
            None,
        )
    }

    // --- Controlled Rotations ---

    /// Appends a Controlled-RX gate (CRX).
    ///
    /// Performs an X-rotation on the target if the control is $|1\rangle$.
    pub fn crx(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRX),
            [control, target],
            params,
            None,
        )
    }

    /// Appends a Controlled-RY gate (CRY).
    pub fn cry(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRY),
            [control, target],
            params,
            None,
        )
    }

    /// Appends a Controlled-RZ gate (CRZ).
    pub fn crz(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRZ),
            [control, target],
            params,
            None,
        )
    }

    // --- Multi-Controlled Gates ---

    /// Appends a Toffoli gate (CCX).
    ///
    /// A 3-qubit gate where the target flips if and only if both controls are $|1\rangle$.
    pub fn ccx(
        &mut self,
        control1: Qubit,
        control2: Qubit,
        target: Qubit,
    ) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CCX),
            [control1, control2, target],
            std::iter::empty(),
            None,
        )
    }

    // --- Advanced / Other Gates ---

    /// Appends a Fermionic Simulation gate (fSim).
    ///
    /// Useful in quantum chemistry simulations.
    ///
    /// $$
    /// \text{fSim}(\theta, \phi) = \begin{pmatrix}
    /// 1 & 0 & 0 & 0 \\
    /// 0 & \cos\theta & -i\sin\theta & 0 \\
    /// 0 & -i\sin\theta & \cos\theta & 0 \\
    /// 0 & 0 & 0 & e^{-i\phi}
    /// \end{pmatrix}
    /// $$
    pub fn fsim(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
        phi: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into(), phi.into()];

        self.append(
            Instruction::Standard(StandardGate::FSIM),
            [a, b],
            params,
            None,
        )
    }

    /// Appends a rotation in the XY plane.
    pub fn rxy(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
        phi: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 2]> = smallvec![theta.into(), phi.into()];

        self.append(
            Instruction::Standard(StandardGate::RXY),
            [qubit],
            params,
            None,
        )
    }

    /// Measures a qubit.
    ///
    /// This is a non-unitary operation that collapses the qubit's state to $|0\rangle$ or $|1\rangle$.
    pub fn measure(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Measure),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Inserts a Barrier.
    ///
    /// A barrier forbids the compiler from optimizing across this line. It has no physical effect
    /// on the qubits but is crucial for debugging and manual optimization control.
    pub fn barrier(&mut self, qubits: Vec<Qubit>) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Barrier),
            qubits,
            std::iter::empty(),
            None,
        )
    }

    /// Resets a qubit to the $|0\rangle$ state.
    ///
    /// This is a non-unitary operation.
    pub fn reset(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Reset),
            [qubit],
            std::iter::empty(),
            None,
        )
    }

    /// Applies a multi-controlled version of a standard gate.
    ///
    /// This method automatically handles gate promotion. For example, applying `X` with 1 control
    /// becomes `CX`, and with 2 controls becomes `CCX`. For higher numbers of controls, it creates
    /// an [`ExtendedGate::MCGate`].
    ///
    /// # Arguments
    ///
    /// * `gate` - The base standard gate to apply (e.g., `X`, `Y`, `RX`).
    /// * `controls` - A list of control qubits.
    /// * `targets` - A list of target qubits.
    /// * `params` - Parameters for the base gate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cqlib_core::circuit::circuit_impl::Circuit;
    /// use cqlib_core::circuit::Qubit;
    /// use cqlib_core::circuit::gate::StandardGate;
    ///
    /// let mut circuit = Circuit::new(4);
    /// let q0 = Qubit::new(0);
    /// let q1 = Qubit::new(1);
    /// let q2 = Qubit::new(2);
    ///
    /// // Equivalent to CCX(q0, q1, q2)
    /// circuit.multi_control(StandardGate::X, [q0, q1], vec![q2], []).unwrap();
    /// ```
    pub fn multi_control<I, C, T, P>(
        &mut self,
        instruction: I,
        controls: C,
        targets: T,
        params: P,
    ) -> Result<(), CircuitError>
    where
        I: Into<Instruction>,
        C: IntoIterator,
        C::Item: Into<Qubit>,
        T: IntoIterator,
        T::Item: Into<Qubit>,
        P: IntoIterator<Item = ParameterValue>,
    {
        let controls_sv: SmallVec<[Qubit; 3]> = controls.into_iter().map(|q| q.into()).collect();
        let targets_sv: SmallVec<[Qubit; 1]> = targets.into_iter().map(|q| q.into()).collect();
        let num_controls = controls_sv.len();

        let inst: Instruction = instruction.into();

        let controlled_inst = inst
            .control(num_controls)
            .ok_or_else(|| CircuitError::InvalidControlOperation(inst.to_string()))?;

        let mut all_qubits = controls_sv;
        all_qubits.extend(targets_sv);
        self.append(controlled_inst, all_qubits, params, None)
    }

    /// Appends a custom unitary gate to the circuit.
    ///
    /// This allows inserting user-defined gates defined by a specific matrix.
    ///
    /// # Arguments
    /// * `definition` - The definition of the custom gate (matrix, label, etc.).
    /// * `qubits` - The list of qubits to apply the gate to.
    ///
    /// # Example
    /// ```rust
    /// use ndarray::Array2;
    /// use num_complex::Complex64;
    /// use cqlib_core::circuit::circuit_impl::Circuit;
    /// use cqlib_core::circuit::gate::UnitaryGate;
    /// use cqlib_core::circuit::Qubit;
    ///
    /// // Define a custom gate (e.g., Identity)
    /// let mat = Array2::eye(2).mapv(|x| Complex64::new(x, 0.0));
    /// let u_gate = UnitaryGate::new("MyGate", 1)
    ///      .with_matrix(mat)
    ///      .unwrap();
    ///
    /// let mut circuit = Circuit::new(4);
    /// let q0 = Qubit::new(0);
    ///
    /// // Apply the custom gate
    /// circuit.unitary(u_gate, vec![q0]).unwrap();
    /// ```
    pub fn unitary(&mut self, gate: UnitaryGate, qubits: Vec<Qubit>) -> Result<(), CircuitError> {
        let qubits_sv: SmallVec<[Qubit; 3]> = qubits.into();

        // Check if qubit count matches definition.num_qubits
        if qubits_sv.len() != gate.num_qubits() as usize {
            return Err(CircuitError::QubitCountMismatch {
                expected: gate.num_qubits() as usize,
                actual: qubits_sv.len(),
            });
        }

        self.append(
            Instruction::UnitaryGate(Box::new(gate)),
            qubits_sv,
            std::iter::empty(),
            None,
        )
    }

    /// Appends a Delay instruction to the circuit.
    ///
    /// This instruction represents an idle period on a specific qubit, often used for
    /// dynamical decoupling or timing control in pulse-level scheduling.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The qubit to apply the delay to.
    /// * `delay` - The duration of the delay. The unit depends on the target backend (e.g., seconds, samples, or dt).
    pub fn delay(
        &mut self,
        qubit: impl Into<Qubit>,
        delay: ParameterValue,
    ) -> Result<(), CircuitError> {
        self.append(Instruction::Delay, vec![qubit], vec![delay], None)
    }

    /// Appends a pre-compiled `CircuitGate` to this circuit.
    ///
    /// This allows nesting circuits within circuits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The `CircuitGate` instance to append.
    /// * `qubits` - The qubits in this circuit that the sub-circuit acts upon.
    /// * `params` - The parameter values to bind to the sub-circuit's parameters.
    pub fn circuit_gate(
        &mut self,
        gate: CircuitGate,
        qubits: Vec<Qubit>,
        params: impl IntoIterator<Item = ParameterValue>,
    ) -> Result<(), CircuitError> {
        self.append(
            Instruction::CircuitGate(Box::new(gate)),
            qubits,
            params,
            None,
        )
    }

    /// Creates the inverse (adjoint) of the circuit.
    ///
    /// The inverse circuit represents the unitary $U^\dagger$ such that $U^\dagger U = I$.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::IrreversibleOperation`] if the circuit contains non-unitary
    /// operations (e.g., `Measure`, `Reset`) or gates that cannot be symbolically inverted.
    pub fn inverse(&self) -> Result<Circuit, CircuitError> {
        let mut new_circuit = Circuit::from_qubits(self.qubits())?;
        new_circuit.data.reserve(self.data.len());
        // 1. Invert Global Phase
        let current_phase_param = self.global_phase();
        // New phase = -1.0 * old_phase
        let new_phase_param = Parameter::from(-1.0) * current_phase_param;

        // Try to simplify/evaluate to keep it clean (e.g. Fixed(-0.5))
        if let Ok(val) = new_phase_param.evaluate(&None) {
            new_circuit.global_phase = CircuitParam::Fixed(val);
        } else {
            let (index, is_new) = new_circuit.parameters.insert_full(new_phase_param.clone());
            if is_new {
                for sym in new_phase_param.get_symbols() {
                    new_circuit.symbols.insert(sym);
                }
            }
            new_circuit.global_phase = CircuitParam::Index(index as u32);
        }

        // 2. Iterate backwards
        for op in self.data.iter().rev() {
            // Special handling for Directives
            match &op.instruction {
                Instruction::Directive(directive) => match directive {
                    Directive::Barrier => {
                        new_circuit.append(
                            Instruction::Directive(Directive::Barrier),
                            op.qubits.clone(),
                            std::iter::empty(),
                            op.label.as_deref(),
                        )?;
                        continue;
                    }
                    _ => return Err(CircuitError::IrreversibleOperation),
                },
                _ => {
                    // Resolve parameters
                    let params: SmallVec<[Parameter; 3]> = op
                        .params
                        .iter()
                        .map(|p| match p {
                            CircuitParam::Fixed(val) => Parameter::from(*val),
                            CircuitParam::Index(idx) => self.parameters[*idx as usize].clone(),
                        })
                        .collect();

                    // Invert instruction
                    if let Some((inv_inst, inv_params)) = op.instruction.inverse(&params) {
                        // Convert back to CircuitParam/ParameterValue
                        let param_values: SmallVec<[ParameterValue; 3]> =
                            inv_params.into_iter().map(ParameterValue::from).collect();

                        new_circuit.append(
                            inv_inst,
                            op.qubits.clone(),
                            param_values,
                            op.label.as_deref(),
                        )?;
                    } else {
                        return Err(CircuitError::IrreversibleOperation);
                    }
                }
            }
        }

        Ok(new_circuit)
    }

    /// Converts the circuit into a `CircuitGate` instruction.
    ///
    /// This method "freezes" the current circuit and wraps it into an instruction that can be
    /// appended to another circuit. The provided `params` are bound to the circuit's free symbols
    /// in the order they were defined.
    ///
    /// # Arguments
    ///
    /// * `name` - A name for the new gate.
    pub fn to_gate(self, name: impl Into<String>) -> Result<Instruction, CircuitError> {
        let frozen = FrozenCircuit { circuit: self };
        let gate = CircuitGate::new(name, frozen)?;
        Ok(Instruction::CircuitGate(Box::new(gate)))
    }

    fn check_qubits_unique(qubits: &[Qubit]) -> bool {
        let mut seen = HashSet::with_capacity(qubits.len());
        for q in qubits {
            if !seen.insert(q) {
                return false;
            }
        }
        true
    }

    /// Decomposes the circuit by resolving sub-circuit gates into their fundamental operations.
    ///
    /// This method recursively unpacks any [`Instruction::CircuitGate`] (hierarchical instructions)
    /// found in the circuit. It handles:
    ///
    /// 1. **Parameter Substitution**: Parameters in the sub-circuit are replaced by the arguments
    ///    passed from the parent circuit.
    ///    - Example: If sub-circuit has `Rx(theta+1)` and is called with `theta = beta`,
    ///      the result is `Rx(beta+1)`.
    /// 2. **Qubit Mapping**: Virtual qubits in the sub-circuit definition are mapped to the
    ///    physical qubits in the parent circuit.
    ///
    /// # Returns
    ///
    /// A new flattened `Circuit` containing only base instructions (Standard, Unitary, Directives).
    pub fn decompose(&self) -> Self {
        let mut new_circuit = Circuit::from_qubits(self.qubits()).unwrap();
        // Preserve the order of symbols from the original circuit.
        new_circuit.symbols = self.symbols.clone();

        // Copy global phase
        match &self.global_phase {
            CircuitParam::Fixed(f) => new_circuit.global_phase = CircuitParam::Fixed(*f),
            CircuitParam::Index(i) => {
                let p = self.parameters[*i as usize].clone();
                let (idx, is_new) = new_circuit.parameters.insert_full(p.clone());
                if is_new {
                    for sym in p.get_symbols() {
                        new_circuit.symbols.insert(sym);
                    }
                }
                new_circuit.global_phase = CircuitParam::Index(idx as u32);
            }
        }

        let initial_qubit_map: HashMap<Qubit, Qubit> =
            self.qubits.iter().map(|q| (*q, *q)).collect();
        let initial_param_map: HashMap<String, Parameter> = HashMap::new();

        for op in &self.data {
            Self::decompose_recursive(
                op,
                self,
                &initial_qubit_map,
                &initial_param_map,
                &mut new_circuit,
            );
        }

        new_circuit
    }

    fn decompose_recursive(
        op: &Operation,
        context_circuit: &Circuit,
        qubit_map: &HashMap<Qubit, Qubit>,
        param_map: &HashMap<String, Parameter>,
        target_circuit: &mut Circuit,
    ) {
        match &op.instruction {
            Instruction::CircuitGate(cg) => {
                // 1. Resolve Parameters in current context
                let mut resolved_params = Vec::with_capacity(op.params.len());
                for p in &op.params {
                    let mut param = match p {
                        CircuitParam::Fixed(v) => Parameter::from(*v),
                        CircuitParam::Index(idx) => {
                            context_circuit.parameters[*idx as usize].clone()
                        }
                    };

                    // Apply substitution from the *parent* scope (if we are deep in recursion)
                    // We need simultaneous substitution here too
                    param = Self::apply_param_map(param, param_map);
                    resolved_params.push(param);
                }

                // 2. Build maps for the next level
                // Param Map: Inner Symbol -> Resolved Value
                let mut next_param_map = HashMap::new();
                for (i, sym) in cg.symbols().iter().enumerate() {
                    if i < resolved_params.len() {
                        next_param_map.insert(sym.clone(), resolved_params[i].clone());
                    }
                }

                // Qubit Map: Inner Qubit -> Outer Qubit
                // op.qubits are the qubits in 'context_circuit' that the gate acts on.
                // We need to map them through 'qubit_map' to get 'target_circuit' qubits.
                let mut next_qubit_map = HashMap::new();
                for (i, inner_q) in cg.circuit.circuit.qubits().iter().enumerate() {
                    if i < op.qubits.len() {
                        let local_q = op.qubits[i];
                        let global_q = qubit_map.get(&local_q).unwrap_or(&local_q);
                        next_qubit_map.insert(*inner_q, *global_q);
                    }
                }

                // 3. Recurse
                for sub_op in &cg.circuit.circuit.data {
                    Self::decompose_recursive(
                        sub_op,
                        &cg.circuit.circuit,
                        &next_qubit_map,
                        &next_param_map,
                        target_circuit,
                    );
                }
            }
            _ => {
                // Base case: Standard/Unitary/Directive
                // Map Qubits
                let mapped_qubits: SmallVec<[Qubit; 3]> = op
                    .qubits
                    .iter()
                    .map(|q| *qubit_map.get(q).unwrap_or(q))
                    .collect();

                // Map Parameters
                let mut mapped_params: SmallVec<[ParameterValue; 3]> = smallvec![];
                for p in &op.params {
                    let mut param = match p {
                        CircuitParam::Fixed(v) => Parameter::from(*v),
                        CircuitParam::Index(idx) => {
                            context_circuit.parameters[*idx as usize].clone()
                        }
                    };

                    param = Self::apply_param_map(param, param_map);

                    mapped_params.push(ParameterValue::from(param));
                }

                target_circuit
                    .append(
                        op.instruction.clone(),
                        mapped_qubits,
                        mapped_params,
                        op.label.as_deref(),
                    )
                    .unwrap();
            }
        }
    }

    fn apply_param_map(mut param: Parameter, map: &HashMap<String, Parameter>) -> Parameter {
        if map.is_empty() {
            return param;
        }

        // Simultaneous substitution strategy using temporary placeholders
        // 1. Replace all target symbols with unique temp symbols
        let mut temp_map = HashMap::new();
        for (key, val) in map {
            // Use a specific internal prefix to avoid collisions during the two-step replacement.
            // This acts as a simultaneous substitution.
            let temp_key = format!("__INTERNAL_SUB_{}", key);
            param = param.replace(key, &Parameter::try_from(temp_key.as_str()).unwrap());
            temp_map.insert(temp_key, val);
        }

        // 2. Replace temp symbols with actual values
        for (temp_key, val) in temp_map {
            param = param.replace(&temp_key, val);
        }

        param
    }

    pub fn to_matrix(&self, qubits_order: Option<&Vec<usize>>) -> Array2<Complex64> {
        circuit_to_matrix(self, qubits_order).unwrap()
    }

    pub fn assign_parameters(
        &self,
        bindings: &Option<HashMap<String, f64>>,
    ) -> Result<Circuit, CircuitError> {
        use crate::circuit::parameter::expr_node::ExprNode;

        let mut new_circuit = Circuit::from_qubits(self.qubits())?;

        // Map from old parameter index to new CircuitParam (either Fixed or Index)
        let mut index_map: Vec<CircuitParam> = Vec::with_capacity(self.parameters.len());

        let empty_bindings = HashMap::new();
        let bind_map = bindings.as_ref().unwrap_or(&empty_bindings);

        for param in self.parameters.iter() {
            if let Ok(val) = param.evaluate(bindings) {
                index_map.push(CircuitParam::Fixed(val));
            } else {
                // Otherwise perform partial evaluation/simplification
                let expr = param.node.evaluate_partial(bind_map)?;
                match expr {
                    ExprNode::Integer(i) => index_map.push(CircuitParam::Fixed(i as f64)),
                    ExprNode::Float(f) => index_map.push(CircuitParam::Fixed(f)),
                    ExprNode::Pi => index_map.push(CircuitParam::Fixed(std::f64::consts::PI)),
                    ExprNode::E => index_map.push(CircuitParam::Fixed(std::f64::consts::E)),
                    _ => {
                        // Still symbolic
                        let new_param = Parameter::new(expr);
                        // Intern the new parameter (deduplicates automatically)
                        let (idx, is_new) = new_circuit.parameters.insert_full(new_param.clone());

                        // If it's a new symbolic parameter, track its symbols
                        if is_new {
                            for sym in new_param.get_symbols() {
                                new_circuit.symbols.insert(sym);
                            }
                        }
                        index_map.push(CircuitParam::Index(idx as u32));
                    }
                }
            }
        }

        // Remap operations to use new parameter indices or fixed values
        new_circuit.data.reserve(self.data.len());
        for op in &self.data {
            let mut new_op = op.clone();
            for p in &mut new_op.params {
                if let CircuitParam::Index(old_idx) = p {
                    *p = index_map[*old_idx as usize].clone();
                }
            }
            new_circuit.data.push(new_op);
        }

        // Remap global phase
        match self.global_phase {
            CircuitParam::Index(old_idx) => {
                new_circuit.global_phase = index_map[old_idx as usize].clone();
            }
            CircuitParam::Fixed(val) => {
                new_circuit.global_phase = CircuitParam::Fixed(val);
            }
        }

        Ok(new_circuit)
    }
}

#[cfg(test)]
#[path = "./circuit_test.rs"]
mod circuit_test;
