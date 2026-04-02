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

#![allow(unsafe_code)]

//! QIS (Quantum Information Science) module for C binding.
//!
//! Provides C-compatible APIs for quantum information operations including
//! state vectors, density matrices, Pauli operators, Hamiltonians, and observables.

// Allow clippy warnings for FFI functions that dereference raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::manual_unwrap_or)]

use cqlib_core::qis::{
    DensityMatrix, DensityMatrixNoise, Hamiltonian, Observable, Pauli, PauliString, Statevector,
};
use num_complex;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double};

pub struct StatevectorWrapper {
    pub inner: Statevector,
}

/// Create a new statevector with specified number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_new(num_qubits: usize) -> *mut StatevectorWrapper {
    let sv = Statevector::new(num_qubits);
    Box::into_raw(Box::new(StatevectorWrapper { inner: sv }))
}

/// Free a statevector.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_free(ptr: *mut StatevectorWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits in the statevector.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_num_qubits(ptr: *const StatevectorWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits }
}

/// Apply H gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_h(ptr: *mut StatevectorWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_h(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply X gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_x(ptr: *mut StatevectorWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_x(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Y gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_y(ptr: *mut StatevectorWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_y(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Z gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_z(ptr: *mut StatevectorWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_z(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CNOT gate.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_cx(ptr: *mut StatevectorWrapper, control: u32, target: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if control as usize >= wrapper.inner.num_qubits || target as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_cx(control as usize, target as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Get probabilities.
#[unsafe(no_mangle)]
pub extern "C" fn statevector_probabilities(
    ptr: *const StatevectorWrapper,
    out_probs: *mut c_double,
    len: usize,
) -> i32 {
    if ptr.is_null() || out_probs.is_null() {
        return -1;
    }
    let wrapper = unsafe { &*ptr };
    let probs = wrapper.inner.probabilities();
    if len != probs.len() {
        return -2;
    }
    unsafe {
        for (i, &prob) in probs.iter().enumerate() {
            *out_probs.add(i) = prob;
        }
    }
    0
}

pub struct DensityMatrixWrapper {
    pub inner: DensityMatrix,
}

/// Create a new density matrix with specified number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_new(num_qubits: usize) -> *mut DensityMatrixWrapper {
    let dm = DensityMatrix::new(num_qubits);
    Box::into_raw(Box::new(DensityMatrixWrapper { inner: dm }))
}

/// Free a density matrix.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_free(ptr: *mut DensityMatrixWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits in the density matrix.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_num_qubits(ptr: *const DensityMatrixWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits }
}

/// Apply H gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_h(ptr: *mut DensityMatrixWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_h(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply X gate.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_x(ptr: *mut DensityMatrixWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_x(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CNOT gate.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_cx(
    ptr: *mut DensityMatrixWrapper,
    control: u32,
    target: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if control as usize >= wrapper.inner.num_qubits || target as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_cx(control as usize, target as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Get probabilities.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_probabilities(
    ptr: *const DensityMatrixWrapper,
    out_probs: *mut c_double,
    len: usize,
) -> i32 {
    if ptr.is_null() || out_probs.is_null() {
        return -1;
    }
    let wrapper = unsafe { &*ptr };
    let probs = wrapper.inner.probabilities();
    if len != probs.len() {
        return -2;
    }
    unsafe {
        for (i, &prob) in probs.iter().enumerate() {
            *out_probs.add(i) = prob;
        }
    }
    0
}

pub struct DensityMatrixNoiseWrapper {
    pub inner: DensityMatrixNoise,
}

/// Create a new density matrix noise simulator.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_new(num_qubits: usize) -> *mut DensityMatrixNoiseWrapper {
    let dm = DensityMatrixNoise::new(num_qubits, None);
    Box::into_raw(Box::new(DensityMatrixNoiseWrapper { inner: dm }))
}

/// Free a density matrix noise simulator.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_free(ptr: *mut DensityMatrixNoiseWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_num_qubits(ptr: *const DensityMatrixNoiseWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.state.num_qubits }
}

/// Apply H gate with noise.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_h(
    ptr: *mut DensityMatrixNoiseWrapper,
    qubit_idx: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.state.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_h(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply X gate with noise.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_x(
    ptr: *mut DensityMatrixNoiseWrapper,
    qubit_idx: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.state.num_qubits {
        return -2;
    }
    match wrapper.inner.apply_x(qubit_idx as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CNOT gate with noise.
#[unsafe(no_mangle)]
pub extern "C" fn density_matrix_noise_cx(
    ptr: *mut DensityMatrixNoiseWrapper,
    control: u32,
    target: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if control as usize >= wrapper.inner.state.num_qubits
        || target as usize >= wrapper.inner.state.num_qubits
    {
        return -2;
    }
    match wrapper.inner.apply_cx(control as usize, target as usize) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

pub struct PauliStringWrapper {
    pub inner: PauliString,
}

/// Create a new Pauli string with specified number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_new(num_qubits: usize) -> *mut PauliStringWrapper {
    let ps = PauliString::new(num_qubits);
    Box::into_raw(Box::new(PauliStringWrapper { inner: ps }))
}

/// Free a Pauli string.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_free(ptr: *mut PauliStringWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_num_qubits(ptr: *const PauliStringWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits }
}

/// Set Pauli operator at a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_set_pauli(
    ptr: *mut PauliStringWrapper,
    qubit_idx: u32,
    pauli: u8,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return -2;
    }
    let p = match pauli {
        0 => Pauli::I,
        1 => Pauli::X,
        2 => Pauli::Y,
        3 => Pauli::Z,
        _ => return -3,
    };
    wrapper.inner.set_pauli(qubit_idx as usize, p);
    0
}

/// Get Pauli operator at a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_get_pauli(ptr: *const PauliStringWrapper, qubit_idx: u32) -> u8 {
    if ptr.is_null() {
        return 255;
    }
    let wrapper = unsafe { &*ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits {
        return 255;
    }
    let idx = qubit_idx as usize;
    let x = wrapper.inner.x[idx];
    let z = wrapper.inner.z[idx];
    match (x, z) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// Convert to string.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_to_string(ptr: *const PauliStringWrapper) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let wrapper = unsafe { &*ptr };
    // Build string in forward order (index 0 to num_qubits-1)
    // instead of reverse order
    let mut s = String::new();
    for i in 0..wrapper.inner.num_qubits {
        let char_code = match (wrapper.inner.x[i], wrapper.inner.z[i]) {
            (false, false) => 'I',
            (true, false) => 'X',
            (false, true) => 'Z',
            (true, true) => 'Y',
        };
        s.push(char_code);
    }
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free string returned by pauli_string_to_string.
#[unsafe(no_mangle)]
pub extern "C" fn pauli_string_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

pub struct HamiltonianWrapper {
    pub inner: Hamiltonian,
}

/// Create a new Hamiltonian with specified number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn hamiltonian_new(num_qubits: usize) -> *mut HamiltonianWrapper {
    let h = Hamiltonian::new(num_qubits);
    Box::into_raw(Box::new(HamiltonianWrapper { inner: h }))
}

/// Free a Hamiltonian.
#[unsafe(no_mangle)]
pub extern "C" fn hamiltonian_free(ptr: *mut HamiltonianWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn hamiltonian_num_qubits(ptr: *const HamiltonianWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits }
}

/// Add a term to the Hamiltonian.
#[unsafe(no_mangle)]
pub extern "C" fn hamiltonian_add_term(
    ptr: *mut HamiltonianWrapper,
    pauli_str: *const c_char,
    real: c_double,
    imag: c_double,
) -> i32 {
    if ptr.is_null() || pauli_str.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let cstr = unsafe { CStr::from_ptr(pauli_str) };
    let pauli_string = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    let pauli = match pauli_string.parse::<PauliString>() {
        Ok(p) => p,
        Err(_) => return -3,
    };
    let coeff = num_complex::Complex64::new(real, imag);
    match wrapper.inner.add_term(pauli, coeff) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Get the number of terms.
#[unsafe(no_mangle)]
pub extern "C" fn hamiltonian_num_terms(ptr: *const HamiltonianWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.terms.len() }
}

/// Compute expectation value of Hamiltonian on statevector.
#[unsafe(no_mangle)]
pub extern "C" fn observable_expectation_sv(
    h_ptr: *const HamiltonianWrapper,
    sv_ptr: *const StatevectorWrapper,
    out_real: *mut c_double,
    out_imag: *mut c_double,
) -> i32 {
    if h_ptr.is_null() || sv_ptr.is_null() || out_real.is_null() || out_imag.is_null() {
        return -1;
    }
    let h_wrapper = unsafe { &*h_ptr };
    let sv_wrapper = unsafe { &*sv_ptr };
    match h_wrapper.inner.expectation_statevector(&sv_wrapper.inner) {
        Ok(val) => {
            unsafe {
                *out_real = val;
                *out_imag = 0.0;
            }
            0
        }
        Err(_) => -2,
    }
}

/// Compute expectation value of Hamiltonian on density matrix.
#[unsafe(no_mangle)]
pub extern "C" fn observable_expectation_dm(
    h_ptr: *const HamiltonianWrapper,
    dm_ptr: *const DensityMatrixWrapper,
    out_real: *mut c_double,
    out_imag: *mut c_double,
) -> i32 {
    if h_ptr.is_null() || dm_ptr.is_null() || out_real.is_null() || out_imag.is_null() {
        return -1;
    }
    let h_wrapper = unsafe { &*h_ptr };
    let dm_wrapper = unsafe { &*dm_ptr };
    match h_wrapper
        .inner
        .expectation_density_matrix(&dm_wrapper.inner)
    {
        Ok(val) => {
            unsafe {
                *out_real = val;
                *out_imag = 0.0;
            }
            0
        }
        Err(_) => -2,
    }
}
