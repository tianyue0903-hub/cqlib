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

use crate::circuit::{Qubit, StandardGate};
use crate::device::noise::{NoiseModel, OperationKey};
use crate::qis::error::QisError;
use crate::qis::state::density_matrix::DensityMatrix;
use ndarray::Array2;
use num_complex::Complex64;

/// A density matrix quantum simulator with noise modeling capabilities.
///
/// This simulator wraps the [`DensityMatrix`] kernel and automatically applies
/// Kraus operator noise after each quantum gate based on a configurable
/// [`NoiseModel`]. It supports both interactive gate-by-gate simulation and
/// batch circuit execution.
///
/// # Examples
///
/// Basic usage with bit-flip noise on X gates:
///
/// ```
/// use cqlib_core::device::{NoiseModel, noise::SingleQubitNoise};
/// use cqlib_core::circuit::StandardGate;
/// use cqlib_core::qis::DensityMatrixNoise;
/// use cqlib_core::circuit::Qubit;
///
/// let mut noise_model = NoiseModel::new();
/// noise_model.add_single_qubit_error(
///     StandardGate::X,
///     Qubit::new(0),
///     SingleQubitNoise::BitFlip(0.01),
/// ).unwrap();
///
/// let mut sim = DensityMatrixNoise::new(1, Some(noise_model));
/// sim.apply_x(0);
///
/// let probs = sim.probabilities();
/// // P(|1>) ≈ 0.99 due to 1% bit-flip noise
/// ```
///
/// Circuit-based simulation:
///
/// ```
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::qis::DensityMatrixNoise;
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(Qubit::new(0)).unwrap();
/// circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
///
/// let sim = DensityMatrixNoise::from_circuit(&circuit, None).unwrap();
/// let probs = sim.probabilities();
/// ```
#[derive(Debug, Clone)]
pub struct DensityMatrixNoise {
    /// The underlying density matrix state.
    pub state: DensityMatrix,
    /// Optional noise model applied to gate operations.
    pub noise_model: Option<NoiseModel>,
}

impl DensityMatrixNoise {
    /// Creates a new simulator with the specified number of qubits and optional noise model.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the quantum system.
    /// * `noise_model` - Optional [`NoiseModel`] defining gate and readout errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::qis::DensityMatrixNoise;
    ///
    /// // Simulator without noise (ideal simulation)
    /// let sim = DensityMatrixNoise::new(3, None);
    ///
    /// // Simulator with empty noise model
    /// let sim = DensityMatrixNoise::new(2, Some(cqlib_core::device::NoiseModel::new()));
    /// ```
    pub fn new(num_qubits: usize, noise_model: Option<NoiseModel>) -> Self {
        Self {
            state: DensityMatrix::new(num_qubits),
            noise_model,
        }
    }

    /// Simulates a circuit, applying noise after each operation.
    ///
    /// The circuit is decomposed into basis gates before execution. Noise is
    /// applied according to the noise model immediately following each gate.
    ///
    /// # Arguments
    ///
    /// * `circuit` - The quantum circuit to simulate.
    /// * `noise_model` - Optional [`NoiseModel`] for noise injection.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError`] if the circuit contains unsupported operations
    /// (e.g., control flow gates) or invalid parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::circuit::{Circuit, Qubit, StandardGate};
    /// use cqlib_core::device::{NoiseModel, noise::SingleQubitNoise};
    /// use cqlib_core::qis::DensityMatrixNoise;
    ///
    /// let mut circuit = Circuit::new(2);
    /// circuit.h(Qubit::new(0)).unwrap();
    /// circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
    ///
    /// let mut noise_model = NoiseModel::new();
    /// noise_model.add_single_qubit_error(
    ///     StandardGate::H,
    ///     Qubit::new(0),
    ///     SingleQubitNoise::Depolarizing(0.001),
    /// ).unwrap();
    ///
    /// let sim = DensityMatrixNoise::from_circuit(&circuit, Some(noise_model)).unwrap();
    /// ```
    pub fn from_circuit(
        circuit: &crate::circuit::Circuit,
        noise_model: Option<NoiseModel>,
    ) -> Result<Self, QisError> {
        use crate::circuit::{CircuitParam, Instruction};
        let circuit = circuit.decompose()?;
        let mut sim = Self::new(circuit.num_qubits(), noise_model);

        let qubits = circuit.qubits();
        let qubit_map: std::collections::HashMap<_, _> = qubits
            .iter()
            .enumerate()
            .map(|(idx, q)| (*q, idx))
            .collect();

        let parameter_values: Vec<Option<f64>> = circuit
            .parameters()
            .iter()
            .map(|p| p.evaluate(&None).ok())
            .collect();

        for op in circuit.operations() {
            let params: Vec<f64> = op
                .params
                .iter()
                .map(|p| match p {
                    CircuitParam::Fixed(v) => Ok(*v),
                    CircuitParam::Index(idx) => parameter_values
                        .get(*idx as usize)
                        .copied()
                        .flatten()
                        .ok_or(QisError::CircuitError(
                            crate::circuit::CircuitError::SymbolicParameterError,
                        )),
                })
                .collect::<Result<Vec<_>, QisError>>()?;

            let qs: Result<Vec<usize>, QisError> = op
                .qubits
                .iter()
                .map(|q| {
                    qubit_map.get(q).copied().ok_or_else(|| {
                        QisError::CircuitError(crate::circuit::CircuitError::QubitNotFound(q.id()))
                    })
                })
                .collect();
            let qs = qs?;

            match &op.instruction {
                Instruction::Standard(gate) => {
                    sim.apply_standard_gate_noise(*gate, &qs, &params)?;
                }
                Instruction::McGate(mc_gate) => {
                    let num_controls = mc_gate.num_ctrl_qubits();
                    let base_gate = mc_gate.base_gate();

                    if num_controls == 1 {
                        let c = qs[0];
                        let t = qs[1];
                        match base_gate {
                            StandardGate::X => sim.apply_cx(c, t)?,
                            StandardGate::Y => sim.apply_cy(c, t)?,
                            StandardGate::Z => sim.apply_cz(c, t)?,
                            StandardGate::RX => sim.apply_crx(c, t, params[0])?,
                            StandardGate::RY => sim.apply_cry(c, t, params[0])?,
                            StandardGate::RZ => sim.apply_crz(c, t, params[0])?,
                            _ => {
                                let matrix = mc_gate.matrix(&params).map_err(|_| {
                                    QisError::CircuitError(
                                        crate::circuit::CircuitError::NoMatrixRepresentation,
                                    )
                                })?;
                                sim.apply_unitary_gate(&qs, &matrix)?;
                            }
                        }
                    } else if num_controls == 2 && *base_gate == StandardGate::X {
                        sim.apply_ccx(qs[0], qs[1], qs[2])?;
                    } else {
                        let matrix = mc_gate.matrix(&params).map_err(|_| {
                            QisError::CircuitError(
                                crate::circuit::CircuitError::NoMatrixRepresentation,
                            )
                        })?;
                        sim.apply_unitary_gate(&qs, &matrix)?;
                    }
                }
                Instruction::UnitaryGate(u_gate) => {
                    if let Some(matrix) = u_gate.matrix() {
                        sim.apply_unitary_gate(&qs, matrix)?;
                    } else {
                        return Err(QisError::CircuitError(
                            crate::circuit::CircuitError::NoMatrixRepresentation,
                        ));
                    }
                }
                Instruction::CircuitGate(_) => {
                    return Err(QisError::CircuitError(
                        crate::circuit::CircuitError::InvalidOperation(
                            "CircuitGate should have been decomposed".to_string(),
                        ),
                    ));
                }
                Instruction::Directive(_) | Instruction::Delay => continue,
                Instruction::ControlFlowGate(_) => {
                    return Err(QisError::UnsupportedOperation(
                        "Control flow gates not supported in density matrix simulation".to_string(),
                    ));
                }
            }
        }
        Ok(sim)
    }

    fn apply_standard_gate_noise(
        &mut self,
        gate: StandardGate,
        qs: &[usize],
        params: &[f64],
    ) -> Result<(), QisError> {
        match gate {
            StandardGate::I => {}
            StandardGate::X => self.apply_x(qs[0])?,
            StandardGate::Y => self.apply_y(qs[0])?,
            StandardGate::Z => self.apply_z(qs[0])?,
            StandardGate::H => self.apply_h(qs[0])?,
            StandardGate::S => self.apply_s(qs[0])?,
            StandardGate::SDG => self.apply_sdg(qs[0])?,
            StandardGate::T => self.apply_t(qs[0])?,
            StandardGate::TDG => self.apply_tdg(qs[0])?,
            StandardGate::RX => self.apply_rx(qs[0], params[0])?,
            StandardGate::RY => self.apply_ry(qs[0], params[0])?,
            StandardGate::RZ => self.apply_rz(qs[0], params[0])?,
            StandardGate::Phase => self.apply_p(qs[0], params[0])?,
            StandardGate::X2P => self.apply_x2p(qs[0])?,
            StandardGate::X2M => self.apply_x2m(qs[0])?,
            StandardGate::Y2P => self.apply_y2p(qs[0])?,
            StandardGate::Y2M => self.apply_y2m(qs[0])?,
            StandardGate::RXY => self.apply_rxy(qs[0], params[0], params[1])?,
            StandardGate::XY => self.apply_xy(qs[0], params[0])?,
            StandardGate::XY2P => self.apply_xy2p(qs[0], params[0])?,
            StandardGate::XY2M => self.apply_xy2m(qs[0], params[0])?,
            StandardGate::U => self.apply_u(qs[0], params[0], params[1], params[2])?,
            StandardGate::GPhase => self.apply_gphase(params[0])?,

            StandardGate::CX => self.apply_cx(qs[0], qs[1])?,
            StandardGate::CY => self.apply_cy(qs[0], qs[1])?,
            StandardGate::CZ => self.apply_cz(qs[0], qs[1])?,
            StandardGate::SWAP => self.apply_swap(qs[0], qs[1])?,
            StandardGate::RXX => self.apply_rxx(qs[0], qs[1], params[0])?,
            StandardGate::RYY => self.apply_ryy(qs[0], qs[1], params[0])?,
            StandardGate::RZZ => self.apply_rzz(qs[0], qs[1], params[0])?,
            StandardGate::RZX => self.apply_rzx(qs[0], qs[1], params[0])?,

            StandardGate::CRX => self.apply_crx(qs[0], qs[1], params[0])?,
            StandardGate::CRY => self.apply_cry(qs[0], qs[1], params[0])?,
            StandardGate::CRZ => self.apply_crz(qs[0], qs[1], params[0])?,

            StandardGate::CCX => self.apply_ccx(qs[0], qs[1], qs[2])?,

            StandardGate::FSIM => self.apply_fsim(qs[0], qs[1], params[0], params[1])?,
        }
        Ok(())
    }

    /// Converts 2D Kraus operators to flat vectors for the density matrix kernel.
    fn convert_kraus_ops(&self, ops: &[Array2<Complex64>]) -> Vec<Vec<Complex64>> {
        ops.iter()
            .map(|op| op.iter().copied().collect::<Vec<Complex64>>())
            .collect()
    }

    /// Applies noise channels associated with a gate operation.
    ///
    /// Looks up the noise model for errors associated with `gate` on `qubits`,
    /// converts them to Kraus operators, and applies them to the state.
    /// Supports single-qubit, two-qubit, and three-qubit gates.
    fn apply_noise(&mut self, gate: StandardGate, qubits: &[usize]) -> Result<(), QisError> {
        if let Some(noise_model) = &self.noise_model {
            if qubits.len() == 1 {
                let q0 = Qubit::new(qubits[0] as u32);
                let key = OperationKey::new_single(gate, q0);
                if let Some(errors) = noise_model.get_single_qubit_errors(&key) {
                    for error in errors {
                        let kraus_ops = error.to_kraus();
                        let flat_ops = self.convert_kraus_ops(&kraus_ops);
                        self.state.apply_kraus(&flat_ops, qubits)?;
                    }
                }
            } else if qubits.len() == 2 {
                let q0 = Qubit::new(qubits[0] as u32);
                let q1 = Qubit::new(qubits[1] as u32);
                if let Ok(key) = OperationKey::new_double(gate, q0, q1) {
                    if let Some(errors) = noise_model.get_two_qubit_errors(&key) {
                        for error in errors {
                            let kraus_ops = error.to_kraus();
                            let flat_ops = self.convert_kraus_ops(&kraus_ops);
                            self.state.apply_kraus(&flat_ops, qubits)?;
                        }
                    }
                }
            } else if qubits.len() == 3 {
                let q0 = Qubit::new(qubits[0] as u32);
                let q1 = Qubit::new(qubits[1] as u32);
                let q2 = Qubit::new(qubits[2] as u32);
                if let Ok(_key) = OperationKey::new_triple(gate, q0, q1, q2) {
                    // Current noise model struct doesn't have `get_three_qubit_errors`.
                    // Ready for future extension without silently failing or panicking.
                }
            }
        }
        Ok(())
    }

    /// Applies the Pauli-X gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_x(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_x(q)?;
        self.apply_noise(StandardGate::X, &[q])?;
        Ok(())
    }

    /// Applies the Pauli-Y gate with optional noise.
    pub fn apply_y(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_y(q)?;
        self.apply_noise(StandardGate::Y, &[q])?;
        Ok(())
    }

    /// Applies the Pauli-Z gate with optional noise.
    pub fn apply_z(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_z(q)?;
        self.apply_noise(StandardGate::Z, &[q])?;
        Ok(())
    }

    /// Applies the Hadamard gate with optional noise.
    pub fn apply_h(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_h(q)?;
        self.apply_noise(StandardGate::H, &[q])?;
        Ok(())
    }

    /// Applies the general single-qubit unitary gate with optional noise.
    ///
    /// The unitary is defined as:
    /// ```text
    /// U(θ, φ, λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)],
    ///              [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle around the Bloch sphere.
    /// * `phi` - Azimuthal angle.
    /// * `lam` - Additional phase parameter.
    pub fn apply_u(&mut self, q: usize, theta: f64, phi: f64, lam: f64) -> Result<(), QisError> {
        self.state.apply_u(q, theta, phi, lam)?;
        self.apply_noise(StandardGate::U, &[q])?;
        Ok(())
    }

    /// Applies the phase gate P(θ) with optional noise.
    ///
    /// Adds a phase factor e^(iθ) to the |1⟩ state. Equivalent to RZ up to a global phase.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Phase angle in radians.
    pub fn apply_p(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_p(q, theta)?;
        self.apply_noise(StandardGate::Phase, &[q])?;
        Ok(())
    }

    /// Applies the S gate (√Z) with optional noise.
    ///
    /// A π/2 rotation around the Z-axis. Square root of the Pauli-Z gate.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_s(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_s(q)?;
        self.apply_noise(StandardGate::S, &[q])?;
        Ok(())
    }

    /// Applies the S† (S dagger) gate with optional noise.
    ///
    /// The Hermitian conjugate of the S gate (-π/2 Z rotation).
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_sdg(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_sdg(q)?;
        self.apply_noise(StandardGate::SDG, &[q])?;
        Ok(())
    }

    /// Applies the T gate (√S) with optional noise.
    ///
    /// A π/4 rotation around the Z-axis. Square root of the S gate.
    /// Essential for universal quantum computation.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_t(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_t(q)?;
        self.apply_noise(StandardGate::T, &[q])?;
        Ok(())
    }

    /// Applies the T† (T dagger) gate with optional noise.
    ///
    /// The Hermitian conjugate of the T gate (-π/4 Z rotation).
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_tdg(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_tdg(q)?;
        self.apply_noise(StandardGate::TDG, &[q])?;
        Ok(())
    }

    /// Applies a rotation around the X-axis with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_rx(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_rx(q, theta)?;
        self.apply_noise(StandardGate::RX, &[q])?;
        Ok(())
    }

    /// Applies a rotation around the Y-axis with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_ry(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_ry(q, theta)?;
        self.apply_noise(StandardGate::RY, &[q])?;
        Ok(())
    }

    /// Applies a rotation around the Z-axis with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_rz(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_rz(q, theta)?;
        self.apply_noise(StandardGate::RZ, &[q])?;
        Ok(())
    }

    /// Applies the global phase gate with optional noise.
    ///
    /// Multiplies the entire state by a phase factor e^(iθ).
    /// Note: Does not affect ideal measurement probabilities.
    ///
    /// # Arguments
    ///
    /// * `theta` - Phase angle in radians.
    pub fn apply_gphase(&mut self, theta: f64) -> Result<(), QisError> {
        self.state.apply_gphase(theta);
        self.apply_noise(StandardGate::GPhase, &[])?;
        Ok(())
    }

    /// Applies the X2P (SX, √X+) gate with optional noise.
    ///
    /// Rotates around the X-axis by +π/2. Equivalent to √X or SX gate.
    /// Two consecutive X2P gates equal a full X gate.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_x2p(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_x2p(q)?;
        self.apply_noise(StandardGate::X2P, &[q])?;
        Ok(())
    }

    /// Applies the X2M (SX†, √X-) gate with optional noise.
    ///
    /// Rotates around the X-axis by -π/2. The Hermitian conjugate of X2P.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_x2m(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_x2m(q)?;
        self.apply_noise(StandardGate::X2M, &[q])?;
        Ok(())
    }

    /// Applies the Y2P (√Y+) gate with optional noise.
    ///
    /// Rotates around the Y-axis by +π/2.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_y2p(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_y2p(q)?;
        self.apply_noise(StandardGate::Y2P, &[q])?;
        Ok(())
    }

    /// Applies the Y2M (√Y-) gate with optional noise.
    ///
    /// Rotates around the Y-axis by -π/2. The Hermitian conjugate of Y2P.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    pub fn apply_y2m(&mut self, q: usize) -> Result<(), QisError> {
        self.state.apply_y2m(q)?;
        self.apply_noise(StandardGate::Y2M, &[q])?;
        Ok(())
    }

    /// Applies an arbitrary rotation on the Bloch sphere with optional noise.
    ///
    /// Rotates by angle θ around the axis defined by angle φ in the X-Y plane.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    /// * `phi` - Azimuthal angle defining the rotation axis.
    pub fn apply_rxy(&mut self, q: usize, theta: f64, phi: f64) -> Result<(), QisError> {
        self.state.apply_rxy(q, theta, phi)?;
        self.apply_noise(StandardGate::RXY, &[q])?;
        Ok(())
    }

    /// Applies the XY2P gate (√XY+) with optional noise.
    ///
    /// A native gate for certain superconducting platforms representing
    /// a partial iSWAP-like rotation in the XY plane.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Phase angle parameter in radians.
    pub fn apply_xy2p(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_xy2p(q, theta)?;
        self.apply_noise(StandardGate::XY2P, &[q])?;
        Ok(())
    }

    /// Applies the XY2M gate (√XY-) with optional noise.
    ///
    /// The Hermitian conjugate of XY2P.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Phase angle parameter in radians.
    pub fn apply_xy2m(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_xy2m(q, theta)?;
        self.apply_noise(StandardGate::XY2M, &[q])?;
        Ok(())
    }

    /// Applies the Controlled-Y (CY) gate with optional noise.
    ///
    /// Applies Y to the target qubit when the control qubit is |1⟩.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit index.
    /// * `target` - Target qubit index.
    pub fn apply_cy(&mut self, control: usize, target: usize) -> Result<(), QisError> {
        self.state.apply_cy(control, target)?;
        self.apply_noise(StandardGate::CY, &[control, target])?;
        Ok(())
    }

    /// Applies the Controlled-X (CX/CNOT) gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit index.
    /// * `target` - Target qubit index.
    pub fn apply_cx(&mut self, control: usize, target: usize) -> Result<(), QisError> {
        self.state.apply_cx(control, target)?;
        self.apply_noise(StandardGate::CX, &[control, target])?;
        Ok(())
    }

    /// Applies the Controlled-Z (CZ) gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index (acts symmetrically).
    /// * `q1` - Second qubit index (acts symmetrically).
    pub fn apply_cz(&mut self, q0: usize, q1: usize) -> Result<(), QisError> {
        self.state.apply_cz(q0, q1)?;
        self.apply_noise(StandardGate::CZ, &[q0, q1])?;
        Ok(())
    }

    /// Applies the RXX gate (XX rotation) with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_rxx(&mut self, q0: usize, q1: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_rxx(q0, q1, theta)?;
        self.apply_noise(StandardGate::RXX, &[q0, q1])?;
        Ok(())
    }

    /// Applies the RYY gate (YY rotation) with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_ryy(&mut self, q0: usize, q1: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_ryy(q0, q1, theta)?;
        self.apply_noise(StandardGate::RYY, &[q0, q1])?;
        Ok(())
    }

    /// Applies the RZZ gate (ZZ rotation) with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_rzz(&mut self, q0: usize, q1: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_rzz(q0, q1, theta)?;
        self.apply_noise(StandardGate::RZZ, &[q0, q1])?;
        Ok(())
    }

    /// Applies the RZX gate (ZX rotation) with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_rzx(&mut self, q0: usize, q1: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_rzx(q0, q1, theta)?;
        self.apply_noise(StandardGate::RZX, &[q0, q1])?;
        Ok(())
    }

    /// Applies the XY gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_xy(&mut self, q: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_xy(q, theta)?;
        self.apply_noise(StandardGate::XY, &[q])?;
        Ok(())
    }

    /// Applies the Controlled-RX gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit index.
    /// * `target` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_crx(&mut self, control: usize, target: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_crx(control, target, theta)?;
        self.apply_noise(StandardGate::CRX, &[control, target])?;
        Ok(())
    }

    /// Applies the Controlled-RY gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit index.
    /// * `target` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_cry(&mut self, control: usize, target: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_cry(control, target, theta)?;
        self.apply_noise(StandardGate::CRY, &[control, target])?;
        Ok(())
    }

    /// Applies the Controlled-RZ gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `control` - Control qubit index.
    /// * `target` - Target qubit index.
    /// * `theta` - Rotation angle in radians.
    pub fn apply_crz(&mut self, control: usize, target: usize, theta: f64) -> Result<(), QisError> {
        self.state.apply_crz(control, target, theta)?;
        self.apply_noise(StandardGate::CRZ, &[control, target])?;
        Ok(())
    }

    /// Applies the fSim gate with optional noise.
    ///
    /// The fSim gate is a native two-qubit gate used in superconducting qubits.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    /// * `theta` - Swap angle in radians.
    /// * `phi` - Controlled-phase angle in radians.
    pub fn apply_fsim(
        &mut self,
        q0: usize,
        q1: usize,
        theta: f64,
        phi: f64,
    ) -> Result<(), QisError> {
        self.state.apply_fsim(q0, q1, theta, phi)?;
        self.apply_noise(StandardGate::FSIM, &[q0, q1])?;
        Ok(())
    }

    /// Applies an arbitrary unitary gate to the state.
    ///
    /// Note: No noise is applied for generic unitary gates as there is no
    /// associated [`StandardGate`] type. Use specific gate methods if noise
    /// modeling is required.
    ///
    /// # Arguments
    ///
    /// * `qs` - Qubit indices the gate acts on.
    /// * `mat` - Unitary matrix as a 2D array.
    pub fn apply_unitary_gate(
        &mut self,
        qs: &[usize],
        mat: &ndarray::Array2<Complex64>,
    ) -> Result<(), QisError> {
        self.state.apply_unitary_gate(qs, mat)?;
        Ok(())
    }

    /// Applies the SWAP gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `q0` - First qubit index.
    /// * `q1` - Second qubit index.
    pub fn apply_swap(&mut self, q0: usize, q1: usize) -> Result<(), QisError> {
        self.state.apply_swap(q0, q1)?;
        self.apply_noise(StandardGate::SWAP, &[q0, q1])?;
        Ok(())
    }

    /// Applies the Toffoli (CCX) gate with optional noise.
    ///
    /// # Arguments
    ///
    /// * `c1` - First control qubit index.
    /// * `c2` - Second control qubit index.
    /// * `t` - Target qubit index.
    pub fn apply_ccx(&mut self, c1: usize, c2: usize, t: usize) -> Result<(), QisError> {
        self.state.apply_ccx(c1, c2, t)?;
        self.apply_noise(StandardGate::CCX, &[c1, c2, t])?;
        Ok(())
    }

    /// Returns the ideal measurement probabilities without readout noise.
    ///
    /// Returns the diagonal elements of the density matrix (probabilities for each
    /// computational basis state), computed directly from the underlying density matrix.
    /// This does NOT include any readout error modeling.
    ///
    /// # Returns
    ///
    /// A vector of probabilities for all 2^n computational basis states, where n
    /// is the total number of qubits in the simulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::qis::DensityMatrixNoise;
    ///
    /// let mut sim = DensityMatrixNoise::new(2, None);
    /// sim.apply_h(0);
    /// sim.apply_cx(0, 1);
    ///
    /// let probs = sim.probabilities();
    /// // probs[0] = P(|00⟩), probs[1] = P(|01⟩), probs[2] = P(|10⟩), probs[3] = P(|11⟩)
    /// // For Bell state |Φ+⟩, P(|00⟩) ≈ 0.5 and P(|11⟩) ≈ 0.5
    /// ```
    pub fn probabilities(&self) -> Vec<f64> {
        self.state.probabilities()
    }

    /// Applies readout noise to the given probabilities.
    ///
    /// Modifies the provided probability distribution by applying the readout errors
    /// configured in the noise model for the specified qubits. If no readout errors
    /// are configured for a qubit, it is skipped.
    ///
    /// # Arguments
    ///
    /// * `probs` - Mutable reference to the probability vector to modify.
    /// * `qubits` - Indices of qubits to apply readout noise for.
    fn apply_readout_noise(&self, probs: &mut Vec<f64>, qubits: &[usize]) {
        let Some(noise_model) = &self.noise_model else {
            return;
        };

        let mut next_probs = vec![0.0; probs.len()];
        for &q in qubits {
            let q_obj = Qubit::new(q as u32);
            let Some(err) = noise_model.get_readout_error(&q_obj) else {
                continue;
            };

            let p_0_given_1 = err.p_0_given_1;
            let p_1_given_0 = err.p_1_given_0;
            let p_0_given_0 = 1.0 - p_1_given_0;
            let p_1_given_1 = 1.0 - p_0_given_1;

            next_probs.fill(0.0);
            for (state, &p) in probs.iter().enumerate() {
                let bit = (state >> q) & 1;
                if bit == 0 {
                    next_probs[state] += p * p_0_given_0;
                    next_probs[state | (1 << q)] += p * p_1_given_0;
                } else {
                    next_probs[state] += p * p_1_given_1;
                    next_probs[state & !(1 << q)] += p * p_0_given_1;
                }
            }
            std::mem::swap(probs, &mut next_probs);
        }
    }

    /// Computes measurement probabilities with readout error modeling.
    ///
    /// Returns the diagonal elements of the density matrix (probabilities for each
    /// computational basis state), modified by readout errors configured in the
    /// noise model for the specified qubits.
    ///
    /// # Arguments
    ///
    /// * `qubits` - Indices of qubits to measure.
    ///
    /// # Returns
    ///
    /// A vector of probabilities for all 2^n computational basis states, where n
    /// is the total number of qubits in the simulator.
    ///
    /// # Examples
    ///
    /// ```
    /// use cqlib_core::qis::DensityMatrixNoise;
    ///
    /// let mut sim = DensityMatrixNoise::new(2, None);
    /// sim.apply_h(0);
    /// sim.apply_cx(0, 1);
    ///
    /// let probs = sim.probabilities_with_readout(&[0, 1]);
    /// // probs[0] = P(|00⟩), probs[1] = P(|01⟩), probs[2] = P(|10⟩), probs[3] = P(|11⟩)
    /// // For Bell state |Φ+⟩, P(|00⟩) ≈ 0.5 and P(|11⟩) ≈ 0.5
    /// ```
    pub fn probabilities_with_readout(&self, qubits: &[usize]) -> Vec<f64> {
        let mut probs = self.probabilities();
        self.apply_readout_noise(&mut probs, qubits);
        probs
    }

    /// Computes the expectation value of a Hamiltonian observable.
    ///
    /// Calculates Tr(ρ * H) for the current noisy density matrix ρ and a given
    /// Hamiltonian H. This delegates to the underlying [`DensityMatrix::expectation`]
    /// method.
    ///
    /// Note: This computes the expectation value of the noisy state, which includes
    /// the effects of gate noise but NOT readout noise (since readout noise only
    /// affects measurement probabilities, not the quantum state itself).
    ///
    /// # Arguments
    /// * `h` - The Hamiltonian observable.
    ///
    /// # Returns
    /// The expectation value as a real number (f64), or a `CircuitError` if the
    /// qubit counts do not match.
    ///
    /// # Example
    /// ```rust
    /// use cqlib_core::qis::DensityMatrixNoise;
    /// use cqlib_core::qis::hamiltonian::Hamiltonian;
    /// use cqlib_core::qis::pauli::{Pauli, PauliString};
    ///
    /// let mut sim = DensityMatrixNoise::new(1, None);
    /// sim.apply_x(0);
    ///
    /// // Create Hamiltonian H = Z
    /// let mut ps = PauliString::new(1);
    /// ps.set_pauli(0, Pauli::Z);
    /// let h = Hamiltonian::from_pauli(ps);
    ///
    /// let exp = sim.expectation(&h).unwrap();
    /// // For |1⟩ state, ⟨Z⟩ = -1
    /// ```
    pub fn expectation(&self, h: &crate::qis::hamiltonian::Hamiltonian) -> Result<f64, QisError> {
        self.state.expectation(h)
    }
}

#[cfg(test)]
#[path = "density_matrix_noise_test.rs"]
mod density_matrix_noise_test;
