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

//! Quantum Device Hardware Characteristics and Topology.
//!
//! This module defines the core data structures used to represent a target quantum backend
//! within the CQLib compiler. These structures (`Device`, `QubitProp`, `EdgeProp`, `InstructionProp`)
//! encapsulate all the physical constraints and fidelity data necessary for noise-aware compilation,
//! mapping, routing, and circuit scheduling.

use crate::circuit::{Instruction, Qubit};
use crate::device::DeviceError;
use crate::device::topology::Topology;
use std::collections::{HashMap, HashSet};
use time::OffsetDateTime;

/// Represents the physical properties and execution characteristics of a quantum instruction (gate)
/// when applied to specific qubit(s).
///
/// This structure stores crucial calibration data such as the error rate (fidelity) and the
/// execution duration of the gate. This information is vital for noise-aware compilation,
/// gate scheduling, and duration-based dynamical decoupling.
#[derive(Debug, Clone)]
pub struct InstructionProp {
    /// The instruction (gate) being characterized.
    instruction: Instruction,
    /// Error rate for this instruction on the specific qubit(s), range [0.0, 1.0].
    error_rate: f64,
    /// Gate duration in nanoseconds.
    length: Option<f64>,
}

impl InstructionProp {
    /// Creates a new `InstructionProp`.
    pub fn new(instruction: Instruction, error_rate: f64) -> Self {
        Self {
            instruction,
            error_rate,
            length: None,
        }
    }

    /// Sets the gate duration (in nanoseconds) using the builder pattern.
    pub fn with_length(mut self, length: f64) -> Self {
        self.length = Some(length);
        self
    }

    pub fn set_length(&mut self, length: f64) {
        self.length = Some(length);
    }
    pub fn with_instruction(mut self, instruction: Instruction) -> Self {
        self.instruction = instruction;
        self
    }

    pub fn set_instruction(&mut self, instruction: Instruction) {
        self.instruction = instruction;
    }

    pub fn with_error_rate(mut self, error_rate: f64) -> Self {
        self.error_rate = error_rate;
        self
    }

    pub fn set_error_rate(&mut self, error_rate: f64) {
        self.error_rate = error_rate;
    }

    /// Gets a reference to the underlying instruction.
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Gets the error rate of this instruction.
    pub fn error_rate(&self) -> f64 {
        self.error_rate
    }

    /// Gets the duration of this instruction in nanoseconds, if defined.
    pub fn length(&self) -> Option<f64> {
        self.length
    }
}

/// Represents the physical and operational properties of a single quantum qubit.
///
/// This includes coherence metrics (T1 relaxation time, T2 dephasing time), operational frequency,
/// and measurement error rates. It also maintains a list of `InstructionProp` which defines
/// the specific native single-qubit instructions supported by this qubit along with their
/// calibrated fidelities and durations.
#[derive(Debug, Clone)]
pub struct QubitProp {
    /// Readout error rate, range [0.0, 1.0].
    readout_error: f64,
    /// Prob of measuring 0 given state was prepared in 1 (p0|1)
    prob_meas0_prep1: Option<f64>,
    /// Prob of measuring 1 given state was prepared in 0 (p1|0)
    prob_meas1_prep0: Option<f64>,

    /// T1 relaxation time in microseconds.
    t1: Option<f64>,
    /// T2 dephasing time in microseconds.
    t2: Option<f64>,
    /// Qubit frequency in GHz.
    frequency: Option<f64>,
    /// Native instructions supported on this qubit.
    native_instructions: Vec<InstructionProp>,
}

impl QubitProp {
    /// Creates a new `QubitProp` with the specified readout error rate.
    pub fn new(readout_error: f64) -> Self {
        Self {
            readout_error,
            prob_meas0_prep1: None,
            prob_meas1_prep0: None,
            t1: None,
            t2: None,
            frequency: None,
            native_instructions: Vec::new(),
        }
    }

    /// Sets the probability of measuring 0 given the state was prepared in 1.
    pub fn with_prob_meas0_prep1(mut self, prob: f64) -> Self {
        self.prob_meas0_prep1 = Some(prob);
        self
    }
    pub fn set_prob_meas0_prep1(&mut self, prob: f64) {
        self.prob_meas0_prep1 = Some(prob);
    }

    /// Sets the probability of measuring 1 given the state was prepared in 0.
    pub fn with_prob_meas1_prep0(mut self, prob: f64) -> Self {
        self.prob_meas1_prep0 = Some(prob);
        self
    }
    pub fn set_prob_meas1_prep0(&mut self, prob: f64) {
        self.prob_meas1_prep0 = Some(prob);
    }

    /// Sets the T1 relaxation time in microseconds.
    pub fn with_t1(mut self, t1: f64) -> Self {
        self.t1 = Some(t1);
        self
    }

    pub fn set_t1(&mut self, t1: f64) {
        self.t1 = Some(t1);
    }

    /// Sets the T2 dephasing time in microseconds.
    pub fn with_t2(mut self, t2: f64) -> Self {
        self.t2 = Some(t2);
        self
    }
    pub fn set_t2(&mut self, t2: f64) {
        self.t2 = Some(t2);
    }

    /// Sets the qubit frequency in GHz.
    pub fn with_frequency(mut self, frequency: f64) -> Self {
        self.frequency = Some(frequency);
        self
    }

    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Adds a native instruction to this qubit's supported instructions.
    pub fn with_native_instruction(mut self, prop: InstructionProp) -> Self {
        self.native_instructions.push(prop);
        self
    }
    pub fn set_native_instruction(&mut self, prop: InstructionProp) {
        self.native_instructions.push(prop);
    }

    /// Gets the readout error rate.
    pub fn readout_error(&self) -> f64 {
        self.readout_error
    }

    /// Gets the probability of measuring 0 given the state was prepared in 1 (p0|1).
    pub fn prob_meas0_prep1(&self) -> Option<f64> {
        self.prob_meas0_prep1
    }

    /// Gets the probability of measuring 1 given the state was prepared in 0 (p1|0).
    pub fn prob_meas1_prep0(&self) -> Option<f64> {
        self.prob_meas1_prep0
    }

    /// Gets the T1 relaxation time in microseconds, if defined.
    pub fn t1(&self) -> Option<f64> {
        self.t1
    }

    /// Gets the T2 dephasing time in microseconds, if defined.
    pub fn t2(&self) -> Option<f64> {
        self.t2
    }

    /// Gets the qubit frequency in GHz, if defined.
    pub fn frequency(&self) -> Option<f64> {
        self.frequency
    }

    /// Gets a slice of the native instructions supported on this qubit.
    pub fn native_instructions(&self) -> &[InstructionProp] {
        &self.native_instructions
    }
}

/// Represents the physical properties of a coupling edge between two qubits in the device topology.
///
/// This structure primarily tracks the native multi-qubit instructions (e.g., CX, CZ)
/// supported across this specific physical connection, including their directional
/// error rates and execution times.
#[derive(Debug, Clone)]
pub struct EdgeProp {
    /// Native instructions supported on this edge (typically 2-qubit gates).
    native_instructions: Vec<InstructionProp>,
}

impl EdgeProp {
    /// Creates a new empty edge property.
    pub fn new() -> Self {
        Self {
            native_instructions: Vec::new(),
        }
    }

    /// Adds a native instruction to this edge.
    pub fn with_native_instruction(mut self, prop: InstructionProp) -> Self {
        self.native_instructions.push(prop);
        self
    }

    pub fn set_native_instruction(&mut self, prop: InstructionProp) {
        self.native_instructions.push(prop);
    }

    /// Gets a slice of the native instructions supported on this edge.
    pub fn native_instructions(&self) -> &[InstructionProp] {
        &self.native_instructions
    }
}

impl Default for EdgeProp {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a quantum device's hardware characteristics and topology.
///
/// The `Device` struct is a fundamental component for compiler optimization, mapping,
/// routing, and noise-aware scheduling. It encapsulates:
/// - The physical connectivity (`Topology`) between qubits.
/// - Available and faulty qubits.
/// - Device-wide default parameters (e.g., T1, T2, gate error rates).
/// - Specific physical properties and supported instructions for individual qubits
///   and coupling edges.
///
/// This structure provides the necessary physical constraints and fidelity data
/// required to simulate noise models or compile quantum circuits onto realistic
/// backend hardware.
///
/// # Example
///
/// ```rust
/// use std::collections::HashSet;
/// use cqlib_core::circuit::Qubit;
/// use cqlib_core::device::{Device, Topology, QubitProp};
///
/// // Create a 2-qubit topology
/// let q0 = Qubit::new(0);
/// let q1 = Qubit::new(1);
/// let topo = Topology::new(vec![q0, q1], vec![(q0, q1, "G1".to_string())]).unwrap();
///
/// // Initialize a device with defaults
/// let mut device = Device::new("mock_device", HashSet::from_iter([q0, q1]), topo).unwrap()
///     .with_default_t1(50.0)
///     .with_default_t2(25.0)
///     .with_default_readout_error(0.01);
///
/// // Add specific properties for Qubit 0
/// let q0_prop = QubitProp::new(0.05).with_t1(40.0);
/// device.add_qubit_properties(q0, q0_prop).unwrap();
///
/// // Query T1, using specific properties if available, else fallback to defaults
/// assert_eq!(device.get_t1(q0), Some(40.0));
/// assert_eq!(device.get_t1(q1), Some(50.0));
/// ```
#[derive(Debug, Clone)]
pub struct Device {
    name: String,
    /// Available (online) qubits.
    qubits: HashSet<Qubit>,
    /// Offline or faulty qubits.
    invalid_qubits: HashSet<Qubit>,
    /// Connectivity topology.
    topology: Topology,
    /// Device-wide native gates (fallback when per-qubit gates not specified).
    native_gates: Vec<Instruction>,

    /// System calibration timestamp.
    calibration_time: Option<OffsetDateTime>,
    /// Default T1 time (μs) for qubits without specific data.
    default_t1: Option<f64>,
    /// Default T2 time (μs) for qubits without specific data.
    default_t2: Option<f64>,
    /// Default readout error for qubits without specific data.
    default_readout_error: Option<f64>,
    /// Default single-qubit gate error.
    default_single_qubit_error: Option<f64>,
    /// Default two-qubit gate error.
    default_two_qubit_error: Option<f64>,

    /// Per-qubit properties (T1, T2, readout error, native gates).
    qubit_properties: HashMap<Qubit, QubitProp>,
    /// Per-edge properties (gate fidelities on specific couplings).
    edge_properties: HashMap<(Qubit, Qubit), EdgeProp>,
}

impl Device {
    /// Creates a new `Device` with the specified name and topology.
    pub fn new(
        name: impl Into<String>,
        qubits: HashSet<Qubit>,
        topology: Topology,
    ) -> Result<Self, DeviceError> {
        for q in topology.qubits() {
            if !qubits.contains(&q) {
                return Err(DeviceError::InvalidOnlineQubit(q));
            }
        }

        Ok(Self {
            name: name.into(),
            qubits,
            invalid_qubits: HashSet::new(),
            topology,
            native_gates: Vec::new(),
            calibration_time: None,
            default_t1: None,
            default_t2: None,
            default_readout_error: None,
            default_single_qubit_error: None,
            default_two_qubit_error: None,
            qubit_properties: HashMap::new(),
            edge_properties: HashMap::new(),
        })
    }

    pub fn with_invalid_qubits(mut self, invalid_qubits: HashSet<Qubit>) -> Self {
        self.invalid_qubits = invalid_qubits;
        self
    }

    pub fn set_invalid_qubits(&mut self, invalid_qubits: HashSet<Qubit>) {
        self.invalid_qubits = invalid_qubits;
    }

    /// Sets the device-wide native gates.
    pub fn with_native_gates(mut self, gates: Vec<Instruction>) -> Self {
        self.native_gates = gates;
        self
    }

    pub fn set_native_gates(&mut self, gates: Vec<Instruction>) {
        self.native_gates = gates;
    }

    /// Sets the system calibration timestamp.
    pub fn with_calibration_time(mut self, time: OffsetDateTime) -> Self {
        self.calibration_time = Some(time);
        self
    }

    pub fn set_calibration_time(&mut self, time: OffsetDateTime) {
        self.calibration_time = Some(time);
    }

    /// Sets the default T1 time (μs).
    pub fn with_default_t1(mut self, t1: f64) -> Self {
        self.default_t1 = Some(t1);
        self
    }

    pub fn set_default_t1(&mut self, t1: f64) {
        self.default_t1 = Some(t1);
    }

    /// Sets the default T2 time (μs).
    pub fn with_default_t2(mut self, t2: f64) -> Self {
        self.default_t2 = Some(t2);
        self
    }

    pub fn set_default_t2(&mut self, t2: f64) {
        self.default_t2 = Some(t2);
    }

    /// Sets the default readout error rate.
    pub fn with_default_readout_error(mut self, error: f64) -> Self {
        self.default_readout_error = Some(error);
        self
    }

    pub fn set_default_readout_error(&mut self, error: f64) {
        self.default_readout_error = Some(error);
    }

    /// Sets the default single-qubit gate error rate.
    pub fn with_default_single_qubit_error(mut self, error: f64) -> Self {
        self.default_single_qubit_error = Some(error);
        self
    }

    pub fn set_default_single_qubit_error(&mut self, error: f64) {
        self.default_single_qubit_error = Some(error);
    }

    /// Sets the default two-qubit gate error rate.
    pub fn with_default_two_qubit_error(mut self, error: f64) -> Self {
        self.default_two_qubit_error = Some(error);
        self
    }

    pub fn set_default_two_qubit_error(&mut self, error: f64) {
        self.default_two_qubit_error = Some(error);
    }

    /// Adds properties for a specific qubit.
    ///
    /// # Errors
    ///
    /// Returns `DeviceError::QubitNotInTopology` if the qubit is not in the device's topology.
    pub fn add_qubit_properties(
        &mut self,
        qubit: Qubit,
        props: QubitProp,
    ) -> Result<(), DeviceError> {
        if !self.qubits.contains(&qubit) || self.invalid_qubits.contains(&qubit) {
            return Err(DeviceError::QubitNotInTopology(qubit));
        }
        self.qubit_properties.insert(qubit, props);
        Ok(())
    }

    /// Adds properties for a specific coupling edge.
    ///
    /// # Errors
    ///
    /// Returns `DeviceError::EdgeNotInTopology` if the edge is not in the device's topology.
    pub fn add_edge_properties(
        &mut self,
        control: Qubit,
        target: Qubit,
        props: EdgeProp,
    ) -> Result<(), DeviceError> {
        if !self.topology.is_connected(control, target) {
            return Err(DeviceError::EdgeNotInTopology(control, target));
        }
        self.edge_properties.insert((control, target), props);
        Ok(())
    }

    /// Gets the name of the device.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets an iterator over the available (online) qubits.
    pub fn qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.qubits.iter().copied()
    }

    /// Gets an iterator over the invalid (offline/faulty) qubits.
    pub fn invalid_qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.invalid_qubits.iter().copied()
    }

    /// Gets a reference to the device's connectivity topology.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Gets the default native gates supported by the device.
    pub fn native_gates(&self) -> &[Instruction] {
        &self.native_gates
    }

    /// Gets the properties of a specific qubit.
    pub fn qubit_properties(&self, qubit: Qubit) -> Option<&QubitProp> {
        self.qubit_properties.get(&qubit)
    }

    /// Gets the properties of a specific coupling edge.
    pub fn edge_properties(&self, control: Qubit, target: Qubit) -> Option<&EdgeProp> {
        self.edge_properties.get(&(control, target))
    }

    /// Gets the T1 relaxation time for a qubit (μs).
    ///
    /// Falls back to the default T1 time if not specified for the qubit.
    pub fn get_t1(&self, qubit: Qubit) -> Option<f64> {
        self.qubit_properties
            .get(&qubit)
            .and_then(|p| p.t1)
            .or(self.default_t1)
    }

    /// Gets the T2 dephasing time for a qubit (μs).
    ///
    /// Falls back to the default T2 time if not specified for the qubit.
    pub fn get_t2(&self, qubit: Qubit) -> Option<f64> {
        self.qubit_properties
            .get(&qubit)
            .and_then(|p| p.t2)
            .or(self.default_t2)
    }

    /// Gets the readout error rate for a qubit.
    ///
    /// Falls back to the default readout error if not specified for the qubit.
    pub fn get_readout_error(&self, qubit: Qubit) -> Option<f64> {
        self.qubit_properties
            .get(&qubit)
            .map(|p| p.readout_error)
            .or(self.default_readout_error)
    }

    /// Gets the default single-qubit gate error rate.
    pub fn default_single_qubit_error(&self) -> Option<f64> {
        self.default_single_qubit_error
    }

    /// Gets the default two-qubit gate error rate.
    pub fn default_two_qubit_error(&self) -> Option<f64> {
        self.default_two_qubit_error
    }

    /// Gets the system calibration timestamp.
    pub fn calibration_time(&self) -> Option<OffsetDateTime> {
        self.calibration_time
    }
}

#[cfg(test)]
#[path = "./device_test.rs"]
mod device_test;
