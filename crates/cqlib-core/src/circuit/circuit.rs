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

use crate::circuit::bit::Qubit;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::{Directive, Instruction, StandardGate, instruction};
use crate::circuit::operation::Operation;
use crate::circuit::param::{CircuitParam, ParameterValue};
use crate::circuit::parameter::Parameter;
use indexmap::IndexSet;
use smallvec::{SmallVec, smallvec};
use std::collections::HashSet;

/// A quantum circuit representation
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

    pub fn width(&self) -> usize {
        self.qubits.len()
    }

    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    pub fn qubits(&self) -> Vec<Qubit> {
        self.qubits.iter().cloned().collect()
    }

    pub fn global_phase(&self) -> Parameter {
        match self.global_phase {
            CircuitParam::Index(index) => self.parameters[index as usize].clone(),
            CircuitParam::Fixed(value) => Parameter::from(value),
        }
    }

    pub fn append<Q, P>(
        &mut self,
        instruction: Instruction,
        qubits: Q,
        params: P,
        label: Option<&str>,
    ) -> Result<(), CircuitError>
    where
        Q: Into<SmallVec<[Qubit; 3]>>,
        P: IntoIterator<Item = ParameterValue>,
    {
        let qubits_sv = qubits.into();
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

    pub fn h(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::H),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    // --- Pauli Gates ---

    pub fn i(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::I),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn x(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn y(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn z(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Z),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn x2p(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X2P),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn x2m(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::X2M),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn y2p(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y2P),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn y2m(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::Y2M),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn xy(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn xy2p(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY2P),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn xy2m(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::XY2M),
            smallvec![qubit],
            params,
            None,
        )
    }

    // --- Clifford & Phase Gates ---

    pub fn s(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::S),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn sdg(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::SDG),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn t(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::T),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn tdg(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::TDG),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    // --- Parametric Rotations ---

    pub fn rx(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];
        self.append(
            Instruction::Standard(StandardGate::RX),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn ry(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RY),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn rz(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZ),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn phase(
        &mut self,
        qubit: Qubit,
        lambda: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![lambda.into()];

        self.append(
            Instruction::Standard(StandardGate::Phase),
            smallvec![qubit],
            params,
            None,
        )
    }

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
            smallvec![qubit],
            params,
            None,
        )
    }

    // --- Two-Qubit Gates ---

    pub fn cx(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CX),
            smallvec![control, target],
            std::iter::empty(),
            None,
        )
    }

    pub fn cy(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CY),
            smallvec![control, target],
            std::iter::empty(),
            None,
        )
    }

    pub fn cz(&mut self, control: Qubit, target: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CZ),
            smallvec![control, target],
            std::iter::empty(),
            None,
        )
    }

    pub fn swap(&mut self, a: Qubit, b: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::SWAP),
            smallvec![a, b],
            std::iter::empty(),
            None,
        )
    }

    pub fn rxx(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RXX),
            smallvec![a, b],
            params,
            None,
        )
    }

    pub fn ryy(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RYY),
            smallvec![a, b],
            params,
            None,
        )
    }

    pub fn rzz(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZZ),
            smallvec![a, b],
            params,
            None,
        )
    }

    pub fn rzx(
        &mut self,
        a: Qubit,
        b: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::RZX),
            smallvec![a, b],
            params,
            None,
        )
    }

    // --- Controlled Rotations ---

    pub fn crx(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRX),
            smallvec![control, target],
            params,
            None,
        )
    }

    pub fn cry(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRY),
            smallvec![control, target],
            params,
            None,
        )
    }

    pub fn crz(
        &mut self,
        control: Qubit,
        target: Qubit,
        theta: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 1]> = smallvec![theta.into()];

        self.append(
            Instruction::Standard(StandardGate::CRZ),
            smallvec![control, target],
            params,
            None,
        )
    }

    // --- Multi-Controlled Gates ---

    pub fn ccx(
        &mut self,
        control1: Qubit,
        control2: Qubit,
        target: Qubit,
    ) -> Result<(), CircuitError> {
        self.append(
            Instruction::Standard(StandardGate::CCX),
            smallvec![control1, control2, target],
            std::iter::empty(),
            None,
        )
    }

    // --- Advanced / Other Gates ---

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
            smallvec![a, b],
            params,
            None,
        )
    }

    pub fn rxy(
        &mut self,
        qubit: Qubit,
        theta: impl Into<ParameterValue>,
        phi: impl Into<ParameterValue>,
    ) -> Result<(), CircuitError> {
        let params: SmallVec<[ParameterValue; 2]> = smallvec![theta.into(), phi.into()];

        self.append(
            Instruction::Standard(StandardGate::RXY),
            smallvec![qubit],
            params,
            None,
        )
    }

    pub fn measure(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Measure),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
    }

    pub fn barrier(&mut self, qubits: Vec<Qubit>) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Barrier),
            SmallVec::from_vec(qubits),
            std::iter::empty(),
            None,
        )
    }

    pub fn reset(&mut self, qubit: Qubit) -> Result<(), CircuitError> {
        self.append(
            Instruction::Directive(Directive::Reset),
            smallvec![qubit],
            std::iter::empty(),
            None,
        )
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
}
