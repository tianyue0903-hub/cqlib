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

//! Pauli evolution gates and operations.
//!
//! This module provides functionality to implement Pauli rotation gates of the form
//! $e^{-i\theta/2 \cdot P}$ where $P$ is a multi-qubit Pauli string operator.
//!
//! # Algorithm
//!
//! The Pauli rotation is implemented using basis transformation and CNOT ladder:
//!
//! 1. **Basis Transformation**: For each non-identity Pauli operator $P_i \in \{X, Y, Z\}$,
//!    apply a basis transformation to convert it to the $Z$ basis:
//!    - $X \to Z$: Apply $H$ gate
//!    - $Y \to Z$: Apply $H \cdot S^\dagger$ (or equivalently $R_X(\pi/2)$)
//!    - $Z \to Z$: No transformation needed
//!
//! 2. **CNOT Chain (Cascade)**: Use CNOT gates to accumulate parity along a chain
//!    to the last non-identity qubit:
//!    - For non-identity qubits at indices $[i_0, i_1, i_2, \ldots, i_{n-1}]$,
//!      apply $CNOT(i_0 \to i_1), CNOT(i_1 \to i_2), \ldots, CNOT(i_{n-2} \to i_{n-1})$
//!    - This computes the product $P_{i_0} \cdot P_{i_1} \cdot \ldots \cdot P_{i_{n-1}}$
//!      in the computational basis
//!
//! 3. **Core Rotation**: Apply $RZ(\theta)$ on the last non-identity qubit.
//!    The multi-qubit Pauli rotation reduces to a single-qubit RZ after the
//!    parity accumulation.
//!
//! 4. **Reverse CNOT Ladder**: Apply the CNOT gates in reverse order to uncompute
//!    the parity.
//!
//! 5. **Inverse Basis Transformation**: Apply the inverse basis transformations in
//!    reverse order to restore the original basis.
//!
//! # Mathematical Background
//!
//! For a Pauli string $P = \bigotimes_{i} P_i$, the rotation is:
//! $$e^{-i\theta/2 \cdot P} = \cos(\theta/2) I - i\sin(\theta/2) P$$
//!
//! Since all Pauli operators can be simultaneously diagonalized in the $Z$ basis,
//! they share eigenstates. The CNOT ladder effectively performs a basis change
//! from the individual qubit basis to the joint parity basis, allowing the
//! multi-qubit rotation to be performed as a single-qubit RZ gate.

use crate::circuit::bit::Qubit;
use crate::circuit::circuit_impl::Circuit;
use crate::circuit::error::CircuitError;
use crate::circuit::param::ParameterValue;
use crate::qis::error::QisError;
use crate::qis::hamiltonian::Hamiltonian;
use crate::qis::pauli::{Pauli, PauliString};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Trotter-Suzuki decomposition modes for Hamiltonian time evolution.
///
/// These modes determine how the time evolution operator $U(t) = e^{-iHt}$ is
/// approximated as a product of Pauli rotations.
#[derive(Debug, Clone, Copy)]
pub enum TrotterMode {
    /// First-order Lie-Trotter decomposition.
    ///
    /// $$U(t) \approx \left[ \prod_k e^{-i c_k t/n \cdot P_k} \right]^n$$
    ///
    /// Error scales as $O(t^2/n)$.
    FirstOrder,

    /// Second-order Strange splitting (symmetric decomposition).
    ///
    /// $$U(t) \approx \left[ e^{-i c_1 t/2n \cdot P_1} \cdots e^{-i c_m t/2n \cdot P_m}
    /// \cdot e^{-i c_m t/2n \cdot P_m} \cdots e^{-i c_1 t/2n \cdot P_1} \right]^n$$
    ///
    /// Error scales as $O(t^3/n^2)$.
    SecondOrder,

    /// Randomized first-order Trotter with specified random seed.
    ///
    /// In each Trotter step, the order of Pauli terms is randomly shuffled.
    /// This can help reduce systematic errors and improve convergence in some cases.
    ///
    /// # Arguments
    /// * `seed` - The random seed for reproducibility
    Randomized(u64),
}

/// Extension trait for Circuit to add Pauli evolution functionality.
///
/// This trait provides methods to append Pauli rotation gates to a circuit.
pub trait PauliEvolution {
    /// Appends a Pauli evolution gate $e^{-i\theta/2 \cdot P}$ to the circuit.
    ///
    /// This method implements the Pauli rotation using the basis transformation
    /// and CNOT ladder algorithm.
    ///
    /// # Arguments
    ///
    /// * `pauli` - The Pauli string operator $P$ to exponentiate. Must be Hermitian
    ///   (i.e., its phase must be $\pm 1$) for the evolution to be unitary.
    /// * `angle` - The rotation angle $\theta$ (can be symbolic or fixed)
    /// * `qubits` - The qubits to apply the operation on (must match `pauli.num_qubits`)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the operation was successfully appended
    /// * `Err(CircuitError)` - If qubit count mismatch, PauliString has non-Hermitian
    ///   phase ($\pm i$), or other error occurs
    ///
    /// # Algorithm Details
    ///
    /// The rotation $e^{-i\theta/2 \cdot P}$ is implemented as follows:
    ///
    /// 1. **Phase Validation**: The PauliString's internal phase must be $\pm 1$
    ///    (Hermitian). If the phase is $\pm i$, an error is returned because
    ///    such operators cannot generate unitary evolution.
    ///
    /// 2. **Phase Absorption**: The PauliString's phase ($\pm 1$) is absorbed into
    ///    the rotation angle: $\theta_{eff} = \theta \cdot \text{phase}$
    ///
    /// 3. **Basis Transformation**: For each qubit $i$ with non-identity Pauli $P_i$:
    ///    - $P_i = X$: Apply $H$ gate (transforms $X$ to $Z$ basis)
    ///    - $P_i = Y$: Apply $H \cdot S^\dagger$ (transforms $Y$ to $Z$ basis)
    ///    - $P_i = Z$: No transformation needed
    ///
    /// 4. **CNOT Chain**: For non-identity qubits $[i_0, i_1, \ldots, i_{n-1}]$,
    ///    apply $CNOT(i_0 \to i_1), CNOT(i_1 \to i_2), \ldots$ to accumulate parity.
    ///
    /// 5. **Core Rotation**: Apply $RZ(\theta_{eff})$ on the last non-identity qubit.
    ///
    /// 6. **Reverse CNOT Ladder**: Apply CNOT gates in reverse order.
    ///
    /// 7. **Inverse Transformation**: Apply inverse basis transformations.
    ///
    /// # Global Phase
    ///
    /// If the Pauli string consists entirely of identity operators (e.g., "III"),
    /// the operation reduces to a global phase $e^{-i\theta/2}$, which is added
    /// to the circuit's global phase.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::qis::pauli::PauliString;
    /// use cqlib_core::qis::evolution::PauliEvolution;
    ///
    /// let mut circuit = Circuit::new(3);
    /// let qubits = circuit.qubits();
    ///
    /// // Create a Pauli string X ⊗ Z ⊗ I
    /// let pauli: PauliString = "XZI".parse().unwrap();
    ///
    /// // Apply e^(-i * π/4 * XZI)
    /// circuit.pauli_evolution(&pauli, std::f64::consts::PI / 2.0, &qubits).unwrap();
    /// ```
    fn pauli_evolution(
        &mut self,
        pauli: &PauliString,
        angle: impl Into<ParameterValue>,
        qubits: &[Qubit],
    ) -> Result<(), CircuitError>;
}

impl PauliEvolution for Circuit {
    fn pauli_evolution(
        &mut self,
        pauli: &PauliString,
        angle: impl Into<ParameterValue>,
        qubits: &[Qubit],
    ) -> Result<(), CircuitError> {
        // Validate qubit count
        if qubits.len() != pauli.num_qubits {
            return Err(CircuitError::QubitCountMismatch {
                expected: pauli.num_qubits,
                actual: qubits.len(),
            });
        }

        // Check that the PauliString phase is Hermitian (±1)
        // Only Hermitian operators can generate unitary evolution
        let phase_complex = pauli.phase.to_complex();

        // For Hermitian operators, phase must be real (±1)
        // If phase is ±i, the operator is not Hermitian
        if phase_complex.im.abs() > 1e-10 {
            return Err(CircuitError::InvalidOperation(
                "Pauli string phase must be Hermitian (±1)".to_string(),
            ));
        }

        // Absorb PauliString's phase into the rotation angle
        // phase_complex.re is either 1.0 or -1.0
        let angle_param: ParameterValue = angle.into();
        let adjusted_angle = if phase_complex.re < 0.0 {
            multiply_angle_by_factor(angle_param, -1.0)
        } else {
            angle_param
        };

        // Collect non-identity Pauli operators with their positions
        let mut non_identity_positions: Vec<(usize, Pauli)> = Vec::new();
        for i in 0..pauli.num_qubits {
            let x_bit = pauli.x[i];
            let z_bit = pauli.z[i];

            let pauli_op = match (x_bit, z_bit) {
                (false, false) => Pauli::I,
                (true, false) => Pauli::X,
                (true, true) => Pauli::Y,
                (false, true) => Pauli::Z,
            };

            if pauli_op != Pauli::I {
                non_identity_positions.push((i, pauli_op));
            }
        }

        // Handle the case where all Pauli operators are identity
        if non_identity_positions.is_empty() {
            // The operation is e^(-iθ/2 * I) = e^(-iθ/2) * I
            // This is just a global phase
            let global_phase_param = multiply_angle_by_factor(adjusted_angle, -0.5);

            // Add to circuit's global phase
            let current_phase = self.global_phase();
            let global_phase_param = parameter_value_to_parameter(global_phase_param)?;
            let new_phase = current_phase + global_phase_param;
            self.set_global_phase(new_phase);

            return Ok(());
        }

        // Apply basis transformations to convert non-Z Paulis to Z basis
        // Store the transformations for later inversion
        let mut transformations: Vec<(Qubit, Pauli)> =
            Vec::with_capacity(non_identity_positions.len());

        for (idx, pauli_op) in &non_identity_positions {
            let qubit = qubits[*idx];
            transformations.push((qubit, *pauli_op));

            match pauli_op {
                Pauli::X => {
                    // X -> Z: Apply H
                    self.h(qubit)?;
                }
                Pauli::Y => {
                    // Y -> Z: Apply H · S†
                    // S† converts Y to X, then H converts X to Z
                    self.sdg(qubit)?;
                    self.h(qubit)?;
                }
                Pauli::Z => {
                    // Already in Z basis, no transformation needed
                }
                Pauli::I => unreachable!(),
            }
        }

        // Chain CNOT ladder: accumulate parity along the chain to the last qubit
        // For non-identity qubits at positions [i0, i1, i2, ...], apply:
        // CNOT(i0 -> i1), CNOT(i1 -> i2), ...
        let num_non_identity = non_identity_positions.len();
        let last_qubit = qubits[non_identity_positions[num_non_identity - 1].0];

        // Forward chain: propagate parity from first to last
        for i in 0..num_non_identity - 1 {
            let curr_qubit = qubits[non_identity_positions[i].0];
            let next_qubit = qubits[non_identity_positions[i + 1].0];
            self.cx(curr_qubit, next_qubit)?;
        }

        // Apply RZ(θ) on the last qubit
        // After the CNOT chain, this effectively rotates the joint parity eigenstate
        self.rz(last_qubit, adjusted_angle)?;

        // Reverse chain: uncompute parity from last-1 down to 0
        for i in (0..num_non_identity - 1).rev() {
            let curr_qubit = qubits[non_identity_positions[i].0];
            let next_qubit = qubits[non_identity_positions[i + 1].0];
            self.cx(curr_qubit, next_qubit)?;
        }

        // Apply inverse basis transformations in reverse order
        for (qubit, pauli_op) in transformations.iter().rev() {
            match pauli_op {
                Pauli::X => {
                    // H† = H
                    self.h(*qubit)?;
                }
                Pauli::Y => {
                    // (H · S†)† = S · H
                    self.h(*qubit)?;
                    self.s(*qubit)?;
                }
                Pauli::Z => {
                    // No transformation needed
                }
                Pauli::I => unreachable!(),
            }
        }

        Ok(())
    }
}

/// Helper function to multiply an angle parameter by a factor
fn multiply_angle_by_factor(angle: ParameterValue, factor: f64) -> ParameterValue {
    match angle {
        ParameterValue::Fixed(val) => ParameterValue::Fixed(val * factor),
        ParameterValue::Param(param) => {
            ParameterValue::Param(crate::circuit::Parameter::from(factor) * param)
        }
    }
}

/// Helper function to convert ParameterValue to Parameter
fn parameter_value_to_parameter(
    pv: ParameterValue,
) -> Result<crate::circuit::Parameter, CircuitError> {
    match pv {
        ParameterValue::Fixed(val) => Ok(crate::circuit::Parameter::from(val)),
        ParameterValue::Param(param) => Ok(param),
    }
}

impl Hamiltonian {
    /// Converts the Hamiltonian to a Trotterized time evolution circuit.
    ///
    /// Implements Trotter-Suzuki decomposition to approximate the time evolution
    /// operator $U(t) = e^{-iHt}$ as a sequence of Pauli rotations.
    ///
    /// # Arguments
    ///
    /// * `time` - The total evolution time $t$
    /// * `steps` - The number of Trotter steps $n$ (must be > 0)
    /// * `mode` - The Trotter decomposition mode (FirstOrder, SecondOrder, Randomized)
    ///
    /// # Returns
    ///
    /// * `Ok(Circuit)` - The approximated time evolution circuit
    /// * `Err(QisError)` - If steps is 0, Hamiltonian is empty, or other error occurs
    ///
    /// # Mathematical Formulation
    ///
    /// For $H = \sum_k c_k P_k$, the decomposition approximates:
    ///
    /// **First Order:**
    /// $$U(t) \approx \prod_{s=1}^{n} \prod_k e^{-i c_k \Delta t \cdot P_k}, \quad \Delta t = t/n$$
    ///
    /// **Second Order:**
    /// $$U(t) \approx \prod_{s=1}^{n} \left[ \prod_k e^{-i c_k \Delta t/2 \cdot P_k}
    /// \cdot \prod_k e^{-i c_k \Delta t/2 \cdot P_k} \right]$$
    /// (with reverse order in second product)
    ///
    /// **Randomized:**
    /// Like first order, but with randomized term order in each step.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::qis::{Hamiltonian, PauliString};
    /// use cqlib_core::qis::evolution::{PauliEvolution, TrotterMode};
    ///
    /// // Create H = 0.5 * ZZ + 0.3 * XX
    /// let mut h = Hamiltonian::new(2);
    /// h.add_term("ZZ".into(), 0.5.into()).unwrap();
    /// h.add_term("XX".into(), 0.3.into()).unwrap();
    ///
    /// // Generate Trotter circuit for t=1.0 with 10 steps
    /// let circuit = h.to_trotter_circuit(1.0, 10, TrotterMode::FirstOrder).unwrap();
    ///
    /// // Circuit contains Pauli evolution gates
    /// assert!(circuit.num_qubits() >= 2);
    /// ```
    pub fn to_trotter_circuit(
        &self,
        time: f64,
        steps: usize,
        mode: TrotterMode,
    ) -> Result<Circuit, QisError> {
        // Validate inputs
        if steps == 0 {
            return Err(QisError::InvalidParameterValue(
                "Trotter steps must be greater than 0".to_string(),
            ));
        }

        if self.terms.is_empty() {
            return Err(QisError::InvalidParameterValue(
                "Cannot create Trotter circuit from empty Hamiltonian".to_string(),
            ));
        }

        // Simplify the Hamiltonian to combine identical terms
        let mut simplified_h = self.clone();
        simplified_h.simplify();

        for (_, coeff) in &simplified_h.terms {
            if coeff.im.abs() > 1e-10 {
                return Err(QisError::NotHermitian);
            }
        }

        // Create the circuit
        let mut circuit = Circuit::new(simplified_h.num_qubits);
        let qubits: Vec<Qubit> = circuit.qubits();

        // Calculate time per step
        let dt = time / steps as f64;

        // Setup RNG for randomized mode
        let mut rng = match mode {
            TrotterMode::Randomized(seed) => Some(StdRng::seed_from_u64(seed)),
            _ => None,
        };

        // Build the circuit for each Trotter step
        for _step in 0..steps {
            match mode {
                TrotterMode::FirstOrder | TrotterMode::Randomized(_) => {
                    // Get term indices
                    let mut indices: Vec<usize> = (0..simplified_h.terms.len()).collect();

                    // Shuffle if randomized mode
                    if let Some(ref mut r) = rng {
                        indices.shuffle(r);
                    }

                    // Apply each term's evolution
                    for idx in indices {
                        let (pauli_str, coeff) = &simplified_h.terms[idx];
                        // Angle = 2 * coeff.re * dt
                        //
                        // Mathematical derivation:
                        // Physical evolution: U = e^{-i H t} = e^{-i c P t}
                        // pauli_evolution implements: R_P(θ) = e^{-i θ/2 P}
                        // Therefore: θ/2 = c*t => θ = 2*c*t
                        let angle = 2.0 * coeff.re * dt;
                        circuit
                            .pauli_evolution(pauli_str, angle, &qubits)
                            .map_err(|e| {
                                QisError::UnsupportedOperation(format!(
                                    "Failed to add Pauli evolution: {}",
                                    e
                                ))
                            })?;
                    }
                }
                TrotterMode::SecondOrder => {
                    // Second-order Strange splitting
                    // Forward half-step
                    for (pauli_str, coeff) in &simplified_h.terms {
                        let angle = 2.0 * coeff.re * dt / 2.0; // half time
                        circuit
                            .pauli_evolution(pauli_str, angle, &qubits)
                            .map_err(|e| {
                                QisError::UnsupportedOperation(format!(
                                    "Failed to add Pauli evolution: {}",
                                    e
                                ))
                            })?;
                    }

                    // Backward half-step (reverse order)
                    for (pauli_str, coeff) in simplified_h.terms.iter().rev() {
                        let angle = 2.0 * coeff.re * dt / 2.0; // half time
                        circuit
                            .pauli_evolution(pauli_str, angle, &qubits)
                            .map_err(|e| {
                                QisError::UnsupportedOperation(format!(
                                    "Failed to add Pauli evolution: {}",
                                    e
                                ))
                            })?;
                    }
                }
            }
        }

        Ok(circuit)
    }
}

#[cfg(test)]
#[path = "evolution_test.rs"]
mod evolution_test;
