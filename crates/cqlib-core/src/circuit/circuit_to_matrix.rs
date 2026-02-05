// This code is part of Cqlib.

// (C) Copyright China Telecom Quantum Group 2026

// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::circuit::Circuit;
use crate::circuit::error::CompileError;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::circuit::param::CircuitParam;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

pub fn circuit_to_matrix(
    circuit: &Circuit,
    qubits_order: Option<&Vec<usize>>,
) -> Result<Array2<Complex64>, CompileError> {
    if circuit.parameters().len() > 0 {
        return Err(CompileError::Error);
    }

    let circuit_qubits: Vec<usize> = circuit.qubits().iter().map(|q| q.index()).collect();
    let num_qubits = circuit_qubits.len();

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
    // Big-Endian mapping: first qubit in target_order corresponds to MSB (highest bit position)
    // This matches the standard quantum computing convention where |q0 q1 ... qn-1> means
    // q0 is the most significant bit (highest weight in the state vector index)
    let qubit_idx_map: HashMap<usize, usize> = target_order
        .iter()
        .enumerate()
        .map(|(i, &q_id)| (q_id, num_qubits - 1 - i))
        .collect();

    let mut matrix: Array2<Complex64> = Array2::eye(1usize << num_qubits);
    if circuit.operations().is_empty() {
        return Ok(matrix);
    }

    matrix
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let mut state_vec = col.to_owned();
            for op in circuit.operations().iter() {
                let qs = op
                    .qubits
                    .iter()
                    .map(|q| qubit_idx_map[&q.index()])
                    .collect();
                let params: Vec<f64> = op
                    .params
                    .iter()
                    .map(|p| match p {
                        CircuitParam::Fixed(value) => *value,
                        CircuitParam::Index(_) => {
                            panic!("error")
                        }
                    })
                    .collect();
                match &op.instruction {
                    Instruction::Standard(std_gate) => {
                        apply_standard_gate(&mut state_vec, &std_gate, params, qs);
                    }
                    Instruction::McGate(mc_gate) => {
                        let matrix = mc_gate.matrix(params.as_slice());
                        apply_general_gate(&mut state_vec, matrix.as_ref(), &qs);
                    }
                    Instruction::UnitaryGate(u_gate) => {
                        if let Some(matrix) = u_gate.matrix() {
                            apply_general_gate(&mut state_vec, matrix, &qs);
                        } else {
                            panic!("error")
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
                            .map_err(|_| CompileError::Error)
                            .unwrap();
                        let matrix = circuit_to_matrix(&sub_circuit, None).unwrap();
                        apply_general_gate(&mut state_vec, &matrix, &qs);
                    }
                    Instruction::Directive(_) => continue,
                }
            }
            col.assign(&state_vec);
        });

    Ok(matrix)
}

fn apply_standard_gate(
    state: &mut Array1<Complex64>,
    std_gate: &StandardGate,
    params: Vec<f64>,
    qubits: Vec<usize>,
) {
    let matrix_cow = std_gate.matrix(params.as_slice());
    let matrix = matrix_cow.as_ref();
    match qubits.len() {
        1 => apply_single_qubit(state, matrix, qubits[0]),
        2 => apply_two_qubits(state, matrix, qubits[0], qubits[1]),
        _ => apply_general_gate(state, matrix, &qubits),
    }
}

fn apply_single_qubit(state: &mut Array1<Complex64>, matrix: &Array2<Complex64>, target: usize) {
    let step = 1 << target;
    let m00 = matrix[[0, 0]];
    let m01 = matrix[[0, 1]];
    let m10 = matrix[[1, 0]];
    let m11 = matrix[[1, 1]];
    for i in (0..state.len()).step_by(step * 2) {
        for j in i..(i + step) {
            let k = j + step;

            let v0 = state[j];
            let v1 = state[k];

            state[j] = m00 * v0 + m01 * v1;
            state[k] = m10 * v0 + m11 * v1;
        }
    }
}

use std::cell::UnsafeCell;

struct UnsafeSlice<'a> {
    ptr: *const UnsafeCell<Complex64>,
    _marker: PhantomData<&'a mut [Complex64]>,
}
unsafe impl<'a> Sync for UnsafeSlice<'a> {}
unsafe impl<'a> Send for UnsafeSlice<'a> {}

impl<'a> UnsafeSlice<'a> {
    fn new(slice: &'a mut [Complex64]) -> Self {
        let ptr = slice.as_mut_ptr() as *const UnsafeCell<Complex64>;
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    unsafe fn get_mut(&self, idx: usize) -> &mut Complex64 {
        unsafe { &mut *(*self.ptr.add(idx)).get() }
    }
}

pub fn apply_two_qubits(
    state: &mut Array1<Complex64>,
    matrix: &Array2<Complex64>,
    q0: usize,
    q1: usize,
) {
    let len = state.len();
    assert!(len >= 4, "State vector too small");

    let (low_q, high_q) = if q0 < q1 { (q0, q1) } else { (q1, q0) };

    // 提取矩阵元素到栈变量 (Copy)
    let m00 = matrix[[0, 0]];
    let m01 = matrix[[0, 1]];
    let m02 = matrix[[0, 2]];
    let m03 = matrix[[0, 3]];
    let m10 = matrix[[1, 0]];
    let m11 = matrix[[1, 1]];
    let m12 = matrix[[1, 2]];
    let m13 = matrix[[1, 3]];
    let m20 = matrix[[2, 0]];
    let m21 = matrix[[2, 1]];
    let m22 = matrix[[2, 2]];
    let m23 = matrix[[2, 3]];
    let m30 = matrix[[3, 0]];
    let m31 = matrix[[3, 1]];
    let m32 = matrix[[3, 2]];
    let m33 = matrix[[3, 3]];

    // 创建 UnsafeSlice
    // 注意：state.as_slice_mut() 可能会 panic 如果内存不连续，建议加上 handle
    let slice = state.as_slice_mut().expect("Array must be contiguous");
    let unsafe_slice = UnsafeSlice::new(slice);

    let loop_limit = len >> 2;

    // Rayon 并行循环
    (0..loop_limit).into_par_iter().for_each(|i| {
        // --- 这里的算法逻辑保持不变 ---
        let mask_low = (1 << low_q) - 1;
        let left_part = (i & !mask_low) << 1;
        let right_part = i & mask_low;
        let temp_idx = left_part | right_part;

        let mask_high = (1 << high_q) - 1;
        let left_final = (temp_idx & !mask_high) << 1;
        let right_final = temp_idx & mask_high;

        let idx_00 = left_final | right_final;

        // 构造四个物理索引
        let idx_v00 = idx_00;
        let idx_v01 = idx_00 | (1 << q1);
        let idx_v10 = idx_00 | (1 << q0);
        let idx_v11 = idx_00 | (1 << q0) | (1 << q1);

        // 使用 UnsafeSlice 获取可变引用
        // 因为我们逻辑上保证了每个线程处理的 i 生成的 idx 组是完全互斥的，所以这是安全的
        unsafe {
            // Read
            let v00 = *unsafe_slice.get_mut(idx_v00);
            let v01 = *unsafe_slice.get_mut(idx_v01);
            let v10 = *unsafe_slice.get_mut(idx_v10);
            let v11 = *unsafe_slice.get_mut(idx_v11);

            // Write back (Matrix Vector Multiplication)
            *unsafe_slice.get_mut(idx_v00) = m00 * v00 + m01 * v01 + m02 * v10 + m03 * v11;
            *unsafe_slice.get_mut(idx_v01) = m10 * v00 + m11 * v01 + m12 * v10 + m13 * v11;
            *unsafe_slice.get_mut(idx_v10) = m20 * v00 + m21 * v01 + m22 * v10 + m23 * v11;
            *unsafe_slice.get_mut(idx_v11) = m30 * v00 + m31 * v01 + m32 * v10 + m33 * v11;
        }
    });
}

pub fn apply_general_gate(
    state: &mut Array1<Complex64>,
    matrix: &Array2<Complex64>,
    qubits: &Vec<usize>,
) {
    let num_qubits = qubits.len();
    let dim = 1 << num_qubits;
    let len = state.len();
    let mut sorted_qubits = qubits.clone();
    sorted_qubits.sort();

    // 预计算：物理偏移量映射
    // 我们需要知道矩阵的第 k 列 (0..2^N) 对应到状态向量的哪个物理偏移
    // 比如 qubits=[5, 2] (5是control, 2是target)
    // 矩阵列 k=2 (二进制10) -> 对应 q[0]=0, q[1]=1 -> 也就是 qubit 5=1, qubit 2=0
    // 所以偏移量是 1 << 5
    let mut bit_offsets = vec![0usize; dim];
    for k in 0..dim {
        let mut offset = 0;
        for (bit_idx, &q_target) in qubits.iter().rev().enumerate() {
            // 检查 k 的第 bit_idx 位是否为 1
            if (k >> bit_idx) & 1 == 1 {
                offset |= 1 << q_target;
            }
        }
        bit_offsets[k] = offset;
    }
    let slice = state.as_slice_mut().expect("Array must be contiguous");
    let unsafe_slice = UnsafeSlice::new(slice);

    (0..len >> num_qubits).into_par_iter().for_each(|i| {
        let mut base_idx = i;
        for &q in &sorted_qubits {
            // 在 q 位置插入 0
            let mask = (1 << q) - 1;
            let left_part = (base_idx & !mask) << 1;
            let right_part = base_idx & mask;
            base_idx = left_part | right_part;
        }
        let mut input_vec = vec![Complex64::default(); dim];
        unsafe {
            // Step A: 读取 (Read)
            for k in 0..dim {
                let physical_idx = base_idx | bit_offsets[k];
                input_vec[k] = *unsafe_slice.get_mut(physical_idx);
            }
            // Step B: 矩阵乘法 + 写回 (Write)
            // 结果 = Matrix * Input
            // 外层循环 row (对应 matrix 的行)
            for r in 0..dim {
                let mut sum = Complex64::default();
                // 内层循环 col (对应 matrix 的列)
                for c in 0..dim {
                    // matrix[[row, col]] * input[col]
                    // 使用 get_unchecked 进一步压榨性能 (前提是 shape 检查过了)
                    sum += *matrix.uget((r, c)) * input_vec[c];
                }

                let write_idx = base_idx | bit_offsets[r];
                *unsafe_slice.get_mut(write_idx) = sum;
            }
        }
    });
}

#[cfg(test)]
#[path = "./circuit_to_matrix_test.rs"]
mod circuit_to_matrix_test;
