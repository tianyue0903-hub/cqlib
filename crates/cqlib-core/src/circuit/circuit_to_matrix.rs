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

//! # Circuit to Matrix Conversion
//!
//! This module provides functionality to simulate a quantum circuit and compute its final unitary matrix representation.
//!
//! ## Core Logic
//!
//! The simulation assumes a state vector simulation model where the full unitary matrix $U$ of the circuit is computed
//! by sequentially applying gate matrices to an initial Identity matrix.
//!
//! $$ U = U_n \cdot U_{n-1} \cdots U_1 \cdot I $$
//!
//! ## Memory Layout & Convention
//!
//! - **State Vector Ordering**: Little-Endian (similar to Qiskit). Qubit 0 corresponds to the Least Significant Bit (LSB).
//!   State $|q_{n-1} \dots q_1 q_0\rangle$.
//! - **Parallelization**: Large matrix multiplications (large state spaces) are automatically parallelized using `rayon`.

use crate::circuit::Circuit;
use crate::circuit::error::CompileError;
use crate::circuit::gate::Instruction;
use crate::circuit::param::CircuitParam;
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

// Threshold for parallelizing the loop over row pairs.
// If the number of pairs (loops) * row size is large enough.
// For small circuits (e.g. 4 qubits), we want purely sequential.
const PARALLEL_THRESHOLD_OPS: usize = 10000;

/// Computes the unitary matrix representation of a quantum circuit.
///
/// This function simulates the circuit by applying each gate's unitary matrix to the full system state.
/// The result is a $2^N \times 2^N$ matrix, where $N$ is the number of qubits.
///
/// # Arguments
///
/// * `circuit` - The quantum circuit to simulate.
/// * `qubits_order` - Optional custom ordering of qubits for the output matrix.
///   If `None`, defaults to sorting qubit indices ascendingly (Little-Endian: q0=LSB).
///
/// # Returns
///
/// * `Ok(Array2<Complex64>)` - The resulting unitary matrix.
/// * `Err(CompileError)` - If the circuit contains unresolved symbolic parameters or non-unitary operations (not yet fully enforced).
///
/// # Panics
///
/// Panics if:
/// - `qubits_order` contains duplicates or does not match the circuit's qubits.
/// - The circuit contains symbolic parameters (currently unsupported).
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::circuit_impl::Circuit;
/// use cqlib_core::circuit::Qubit;
/// use cqlib_core::circuit::circuit_to_matrix;
///
/// let mut circuit = Circuit::new(2);
/// circuit.h(Qubit::new(0));
/// circuit.cx(Qubit::new(0), Qubit::new(1));
///
/// let matrix = circuit_to_matrix(&circuit, None).unwrap();
/// // matrix is now the 4x4 unitary of the Bell state preparation.
/// ```
pub fn circuit_to_matrix(
    circuit: &Circuit,
    qubits_order: Option<&Vec<usize>>,
) -> Result<Array2<Complex64>, CompileError> {
    if !circuit.parameters().is_empty() {
        return Err(CompileError::Error);
    }
    let circuit_qubits: Vec<usize> = circuit.qubits().iter().map(|q| q.index()).collect();
    let num_qubits = circuit_qubits.len();
    let dim = 1usize << num_qubits;

    let target_order: Vec<usize> = match qubits_order {
        Some(order) => {
            let circuit_set: HashSet<usize> = circuit_qubits.iter().cloned().collect();
            let order_set: HashSet<usize> = order.iter().cloned().collect();
            if circuit_set != order_set || circuit_set.len() != order.len() {
                panic!(
                    "qubits_order mismatch! Circuit has {:?}, but order provided is {:?}",
                    circuit_qubits, order
                );
            }
            order.clone()
        }
        None => {
            let mut sorted = circuit_qubits.clone();
            sorted.sort();
            sorted
        }
    };

    // Map logical qubit index to physical bit position (0 to N-1).
    // Little-Endian mapping: first qubit in target_order corresponds to LSB (bit 0)
    // q0 -> bit 0, q1 -> bit 1, ...
    // This matches Qiskit's convention.
    let qubit_bit_map: HashMap<usize, usize> = target_order
        .iter()
        .enumerate()
        .map(|(i, &q_id)| (q_id, i))
        .collect();

    // Start with Identity
    let mut matrix: Array2<Complex64> = Array2::eye(dim);

    // Iterate over operations and update the matrix in-place
    for op in circuit.operations().iter() {
        // Map operation qubits to physical bit positions
        let bits: Vec<usize> = op
            .qubits
            .iter()
            .map(|q| qubit_bit_map[&q.index()])
            .collect();

        let params: Vec<f64> = op
            .params
            .iter()
            .map(|p| match p {
                CircuitParam::Fixed(value) => *value,
                CircuitParam::Index(_) => {
                    panic!("Symbolic parameters not supported in circuit_to_matrix")
                }
            })
            .collect();

        match &op.instruction {
            Instruction::Standard(std_gate) => {
                let gate_matrix = std_gate.matrix(params.as_slice());
                // StandardGate matrices are Big-Endian (Controls/First Args are MSB).
                // System is Little-Endian. Reverse bits to align.
                let reversed_bits: Vec<usize> = bits.iter().cloned().rev().collect();
                apply_gate_to_matrix(&mut matrix, gate_matrix.as_ref(), &reversed_bits);
            }
            Instruction::McGate(mc_gate) => {
                let gate_matrix = mc_gate.matrix(params.as_slice());
                // McGate matrices are Big-Endian (Controls MSB).
                let reversed_bits: Vec<usize> = bits.iter().cloned().rev().collect();
                apply_gate_to_matrix(&mut matrix, gate_matrix.as_ref(), &reversed_bits);
            }
            Instruction::UnitaryGate(u_gate) => {
                if let Some(gate_matrix) = u_gate.matrix() {
                    // UnitaryGate matrices are assumed to follow standard Big-Endian convention.
                    // System is Little-Endian. Reverse bits to align.
                    let reversed_bits: Vec<usize> = bits.iter().cloned().rev().collect();
                    apply_gate_to_matrix(&mut matrix, gate_matrix, &reversed_bits);
                } else {
                    panic!("UnitaryGate matrix missing")
                }
            }
            Instruction::CircuitGate(circuit_gate) => {
                let symbols = circuit_gate.symbols();
                let mut bindings = HashMap::new();
                for (sym, val) in symbols.iter().zip(params.iter()) {
                    bindings.insert(sym.clone(), *val);
                }
                let sub_circuit = circuit_gate
                    .circuit
                    .circuit
                    .assign_parameters(&Some(bindings))
                    .map_err(|_| CompileError::Error)?;
                let sub_matrix = circuit_to_matrix(&sub_circuit, None).unwrap();
                apply_gate_to_matrix(&mut matrix, &sub_matrix, &bits);
            }
            Instruction::ControlFlowGate(_) => continue,
            Instruction::Directive(_) => continue,
            Instruction::Delay => continue,
        }
    }

    Ok(matrix)
}

/// Applies a gate (given as a matrix) to the full state matrix.
fn apply_gate_to_matrix(
    matrix: &mut Array2<Complex64>,
    gate_matrix: &Array2<Complex64>,
    bits: &[usize],
) {
    match bits.len() {
        1 => apply_single_qubit_gate(matrix, gate_matrix, bits[0]),
        2 => apply_two_qubit_gate(matrix, gate_matrix, bits[0], bits[1]),
        _ => apply_general_gate(matrix, gate_matrix, bits),
    }
}

// Helper struct for raw pointer access to split mutable borrow
struct UnsafeSlice<'a> {
    ptr: *mut Complex64,
    _marker: PhantomData<&'a mut [Complex64]>,
}
unsafe impl<'a> Sync for UnsafeSlice<'a> {}
unsafe impl<'a> Send for UnsafeSlice<'a> {}

impl<'a> UnsafeSlice<'a> {
    fn new(slice: &'a mut [Complex64]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            _marker: PhantomData,
        }
    }

    // SAFETY: Caller must ensure that concurrent accesses do not alias.
    unsafe fn row_ptr(&self, row_idx: usize, cols: usize) -> *mut Complex64 {
        unsafe { self.ptr.add(row_idx * cols) }
    }
}

fn apply_single_qubit_gate(matrix: &mut Array2<Complex64>, gate: &Array2<Complex64>, bit: usize) {
    let dim = matrix.nrows(); // 2^N
    let cols = matrix.ncols(); // 2^N, same as dim

    let u00 = gate[[0, 0]];
    let u01 = gate[[0, 1]];
    let u10 = gate[[1, 0]];
    let u11 = gate[[1, 1]];

    let step = 1 << bit;
    let total_ops = dim * cols;
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let slice = matrix
        .as_slice_mut()
        .expect("Matrix must be contiguous (standard layout)");
    let unsafe_slice = UnsafeSlice::new(slice);

    let process_block = |i: usize| unsafe {
        for j in 0..step {
            let r0_idx = i + j;
            let r1_idx = r0_idx + step;

            let r0_ptr = unsafe_slice.row_ptr(r0_idx, cols);
            let r1_ptr = unsafe_slice.row_ptr(r1_idx, cols);

            for k in 0..cols {
                let v0 = *r0_ptr.add(k);
                let v1 = *r1_ptr.add(k);

                *r0_ptr.add(k) = u00 * v0 + u01 * v1;
                *r1_ptr.add(k) = u10 * v0 + u11 * v1;
            }
        }
    };

    if parallel {
        (0..dim)
            .into_par_iter()
            .step_by(step * 2)
            .for_each(process_block);
    } else {
        (0..dim).step_by(step * 2).for_each(process_block);
    }
}

fn apply_two_qubit_gate(
    matrix: &mut Array2<Complex64>,
    gate: &Array2<Complex64>,
    b0: usize,
    b1: usize,
) {
    let dim = matrix.nrows();
    let cols = matrix.ncols();

    let (low, high) = if b0 < b1 { (b0, b1) } else { (b1, b0) };

    let slice = matrix.as_slice_mut().expect("Matrix must be contiguous");
    let unsafe_slice = UnsafeSlice::new(slice);

    let loop_limit = dim >> 2;
    let total_ops = dim * cols;
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let process_idx = |i: usize| {
        let mask_low = (1 << low) - 1;
        let left_part = (i & !mask_low) << 1;
        let right_part = i & mask_low;
        let tmp = left_part | right_part;

        let mask_high = (1 << high) - 1;
        let left_final = (tmp & !mask_high) << 1;
        let right_final = tmp & mask_high;

        let base = left_final | right_final;

        // Correct Little-Endian Mapping:
        // bits[0] (b0) corresponds to LSB (Gate Index Bit 0) -> off0
        // bits[1] (b1) corresponds to MSB (Gate Index Bit 1) -> off1
        let off0 = 1 << b0;
        let off1 = 1 << b1;

        // Gate Indices:
        // 00 -> base
        // 01 -> base | off0
        // 10 -> base | off1
        // 11 -> base | off0 | off1

        let r0_idx = base;
        let r1_idx = base | off0;
        let r2_idx = base | off1;
        let r3_idx = base | off0 | off1;

        unsafe {
            let p0 = unsafe_slice.row_ptr(r0_idx, cols);
            let p1 = unsafe_slice.row_ptr(r1_idx, cols);
            let p2 = unsafe_slice.row_ptr(r2_idx, cols);
            let p3 = unsafe_slice.row_ptr(r3_idx, cols);

            for k in 0..cols {
                let v0 = *p0.add(k);
                let v1 = *p1.add(k);
                let v2 = *p2.add(k);
                let v3 = *p3.add(k);

                let mut sum0 = Complex64::default();
                let mut sum1 = Complex64::default();
                let mut sum2 = Complex64::default();
                let mut sum3 = Complex64::default();

                sum0 +=
                    gate[[0, 0]] * v0 + gate[[0, 1]] * v1 + gate[[0, 2]] * v2 + gate[[0, 3]] * v3;
                sum1 +=
                    gate[[1, 0]] * v0 + gate[[1, 1]] * v1 + gate[[1, 2]] * v2 + gate[[1, 3]] * v3;
                sum2 +=
                    gate[[2, 0]] * v0 + gate[[2, 1]] * v1 + gate[[2, 2]] * v2 + gate[[2, 3]] * v3;
                sum3 +=
                    gate[[3, 0]] * v0 + gate[[3, 1]] * v1 + gate[[3, 2]] * v2 + gate[[3, 3]] * v3;

                *p0.add(k) = sum0;
                *p1.add(k) = sum1;
                *p2.add(k) = sum2;
                *p3.add(k) = sum3;
            }
        }
    };

    if parallel {
        (0..loop_limit).into_par_iter().for_each(process_idx);
    } else {
        (0..loop_limit).for_each(process_idx);
    }
}

fn apply_general_gate(matrix: &mut Array2<Complex64>, gate: &Array2<Complex64>, bits: &[usize]) {
    let dim = matrix.nrows();
    let cols = matrix.ncols();
    let num_targets = bits.len();
    let gate_dim = 1 << num_targets;

    let mut sorted_bits = bits.to_vec();
    sorted_bits.sort();

    let mut gate_offsets = vec![0usize; gate_dim];
    for (k, gate_offset_ref) in gate_offsets.iter_mut().enumerate().take(gate_dim) {
        let mut offset = 0;
        for (j, &phy_bit) in bits.iter().enumerate() {
            // Correct Little-Endian Mapping:
            // bits[j] corresponds to the j-th bit of the gate index k.
            if (k >> j) & 1 == 1 {
                offset |= 1 << phy_bit;
            }
        }
        *gate_offset_ref = offset;
    }

    let slice = matrix.as_slice_mut().expect("Matrix must be contiguous");
    let unsafe_slice = UnsafeSlice::new(slice);

    let loop_limit = dim >> num_targets;
    let total_ops = dim * cols;
    let parallel = total_ops >= PARALLEL_THRESHOLD_OPS;

    let process_idx = |i: usize| {
        let mut base = i;
        for &q in &sorted_bits {
            let mask = (1 << q) - 1;
            let left = (base & !mask) << 1;
            let right = base & mask;
            base = left | right;
        }

        unsafe {
            let mut row_ptrs = Vec::with_capacity(gate_dim);
            for item in gate_offsets.iter().take(gate_dim) {
                row_ptrs.push(unsafe_slice.row_ptr(base | item, cols));
            }

            let mut input_vec = vec![Complex64::default(); gate_dim];

            for k_col in 0..cols {
                for g in 0..gate_dim {
                    input_vec[g] = *row_ptrs[g].add(k_col);
                }

                for r in 0..gate_dim {
                    let mut sum = Complex64::default();
                    for c in 0..gate_dim {
                        sum += gate[[r, c]] * input_vec[c];
                    }
                    *row_ptrs[r].add(k_col) = sum;
                }
            }
        }
    };

    if parallel {
        (0..loop_limit).into_par_iter().for_each(process_idx);
    } else {
        (0..loop_limit).for_each(process_idx);
    }
}

#[cfg(test)]
#[path = "./circuit_to_matrix_test.rs"]
mod circuit_to_matrix_test;
