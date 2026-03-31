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

//! Circuit module for C binding.
//!
//! Provides C-compatible APIs for quantum circuit operations.

// Allow clippy warnings for FFI functions that dereference raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::manual_unwrap_or)]

use cqlib_core::circuit::circuit_param::ParameterValue;
use cqlib_core::circuit::parameter::Parameter;
use cqlib_core::circuit::{Circuit, Qubit};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

// =============================================================================
// Circuit Wrapper
// =============================================================================

pub struct CircuitWrapper {
    pub inner: Circuit,
}

/// Create a new quantum circuit with specified number of qubits.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_new(num_qubits: usize) -> *mut CircuitWrapper {
    let circuit = Circuit::new(num_qubits);
    Box::into_raw(Box::new(CircuitWrapper { inner: circuit }))
}

/// Free a quantum circuit.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_free(ptr: *mut CircuitWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Get the number of qubits in the circuit.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_num_qubits(ptr: *const CircuitWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits() }
}

// =============================================================================
// Single-Qubit Gates
// =============================================================================

/// Apply H (Hadamard) gate to a qubit.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_h(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.h(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply X gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_x(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.x(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Y gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_y(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.y(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Z gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_z(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.z(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply S gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_s(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.s(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply T gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_t(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.t(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply SX (Square Root of X) gate.
/// Note: SX is approximated using X2P (X/2 rotation) applied twice.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_sx(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    let q = Qubit::new(qubit_idx);
    match wrapper.inner.x2p(q) {
        Ok(_) => match wrapper.inner.x2p(q) {
            Ok(_) => 0,
            Err(_) => -3,
        },
        Err(_) => -3,
    }
}

/// Apply X2P gate (X/2 rotation).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_x2p(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.x2p(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply X2M gate (-X/2 rotation).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_x2m(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.x2m(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Y2P gate (Y/2 rotation).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_y2p(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.y2p(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply Y2M gate (-Y/2 rotation).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_y2m(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.y2m(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// =============================================================================
// Two-Qubit Gates
// =============================================================================

/// Apply CX (CNOT) gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_cx(ptr: *mut CircuitWrapper, ctrl_idx: u32, target_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .cx(Qubit::new(ctrl_idx), Qubit::new(target_idx))
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CY gate (controlled-Y).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_cy(ptr: *mut CircuitWrapper, ctrl_idx: u32, target_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .cy(Qubit::new(ctrl_idx), Qubit::new(target_idx))
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CZ gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_cz(ptr: *mut CircuitWrapper, ctrl_idx: u32, target_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .cz(Qubit::new(ctrl_idx), Qubit::new(target_idx))
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply SWAP gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_swap(ptr: *mut CircuitWrapper, idx1: u32, idx2: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if idx1 as usize >= n || idx2 as usize >= n {
        return -2;
    }

    match wrapper.inner.swap(Qubit::new(idx1), Qubit::new(idx2)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// =============================================================================
// Parameterized Gates (with concrete float values)
// =============================================================================

/// Apply RX gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rx(ptr: *mut CircuitWrapper, qubit_idx: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.rx(Qubit::new(qubit_idx), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RY gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_ry(ptr: *mut CircuitWrapper, qubit_idx: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.ry(Qubit::new(qubit_idx), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZ gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rz(ptr: *mut CircuitWrapper, qubit_idx: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.rz(Qubit::new(qubit_idx), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RXY gate with concrete float values.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rxy(
    ptr: *mut CircuitWrapper,
    qubit_idx: u32,
    theta: f64,
    phi: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.rxy(Qubit::new(qubit_idx), theta, phi) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RXX gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rxx(ptr: *mut CircuitWrapper, a: u32, b: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }

    match wrapper.inner.rxx(Qubit::new(a), Qubit::new(b), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RYY gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_ryy(ptr: *mut CircuitWrapper, a: u32, b: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }

    match wrapper.inner.ryy(Qubit::new(a), Qubit::new(b), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZZ gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rzz(ptr: *mut CircuitWrapper, a: u32, b: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }

    match wrapper.inner.rzz(Qubit::new(a), Qubit::new(b), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZX gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rzx(ptr: *mut CircuitWrapper, a: u32, b: u32, theta: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }

    match wrapper.inner.rzx(Qubit::new(a), Qubit::new(b), theta) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRX gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_crx(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .crx(Qubit::new(ctrl_idx), Qubit::new(target_idx), theta)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRY gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_cry(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .cry(Qubit::new(ctrl_idx), Qubit::new(target_idx), theta)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRZ gate with concrete float value.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_crz(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .crz(Qubit::new(ctrl_idx), Qubit::new(target_idx), theta)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply fSim gate with concrete floats.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_fsim(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta: f64,
    phi: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }

    match wrapper.inner.fsim(Qubit::new(a), Qubit::new(b), theta, phi) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CCX (Toffoli) gate.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_ccx(
    ptr: *mut CircuitWrapper,
    ctrl1: u32,
    ctrl2: u32,
    target: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl1 as usize >= n || ctrl2 as usize >= n || target as usize >= n {
        return -2;
    }

    match wrapper
        .inner
        .ccx(Qubit::new(ctrl1), Qubit::new(ctrl2), Qubit::new(target))
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Measure a qubit (directive).
#[unsafe(no_mangle)]
pub extern "C" fn circuit_measure(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.measure(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Reset a qubit to |0> state.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_reset(ptr: *mut CircuitWrapper, qubit_idx: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    match wrapper.inner.reset(Qubit::new(qubit_idx)) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// =============================================================================
// Other Operations
// =============================================================================

/// Apply barrier to qubits.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_barrier(ptr: *mut CircuitWrapper, num_qubits: u32) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if num_qubits as usize > n {
        return -2;
    }

    let qubits: Vec<Qubit> = (0..num_qubits).map(Qubit::new).collect();
    match wrapper.inner.barrier(qubits) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// =============================================================================
// Parameter Wrapper (for symbolic parameters)
// =============================================================================

pub struct ParameterWrapper {
    pub inner: Parameter,
}

/// Parse a parameter expression string.
/// Returns a pointer to a ParameterWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn param_parse(expr: *const c_char) -> *mut ParameterWrapper {
    if expr.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { std::ffi::CStr::from_ptr(expr) };
    let expr_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match Parameter::try_from(expr_str) {
        Ok(param) => Box::into_raw(Box::new(ParameterWrapper { inner: param })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a parameter.
#[unsafe(no_mangle)]
pub extern "C" fn param_free(ptr: *mut ParameterWrapper) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ptr);
    }
}

/// Evaluate a parameter with given variable bindings.
/// The bindings format is "var1:value1,var2:value2,..."
#[unsafe(no_mangle)]
pub extern "C" fn param_evaluate(ptr: *const ParameterWrapper, bindings: *const c_char) -> f64 {
    if ptr.is_null() {
        return 0.0;
    }

    let param = unsafe { &(*ptr).inner };
    let mut map: HashMap<&str, f64> = HashMap::new();

    if !bindings.is_null() {
        if let Ok(c_str) = unsafe { std::ffi::CStr::from_ptr(bindings).to_str() } {
            for pair in c_str.split(',') {
                let parts: Vec<&str> = pair.splitn(2, ':').collect();
                if parts.len() == 2 {
                    if let Ok(val) = parts[1].parse::<f64>() {
                        map.insert(parts[0], val);
                    }
                }
            }
        }
    }

    let bindings_opt = if map.is_empty() { None } else { Some(map) };
    match param.evaluate(&bindings_opt) {
        Ok(val) => val,
        Err(_) => 0.0,
    }
}

/// Apply RX gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rx_param(
    ptr: *mut CircuitWrapper,
    qubit_idx: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || param_ptr.is_null() {
        return -1;
    }

    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    let param = unsafe { &(*param_ptr).inner };
    let param_value = ParameterValue::Param(param.clone());

    match wrapper.inner.rx(Qubit::new(qubit_idx), param_value) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RY gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_ry_param(
    ptr: *mut CircuitWrapper,
    qubit_idx: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || param_ptr.is_null() {
        return -1;
    }

    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    let param = unsafe { &(*param_ptr).inner };
    let param_value = ParameterValue::Param(param.clone());

    match wrapper.inner.ry(Qubit::new(qubit_idx), param_value) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZ gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rz_param(
    ptr: *mut CircuitWrapper,
    qubit_idx: u32,
    param_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || param_ptr.is_null() {
        return -1;
    }

    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }

    let param = unsafe { &(*param_ptr).inner };
    let param_value = ParameterValue::Param(param.clone());

    match wrapper.inner.rz(Qubit::new(qubit_idx), param_value) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// Additional parameterized gate wrappers

/// Apply RXY gate with symbolic parameters.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rxy_param(
    ptr: *mut CircuitWrapper,
    qubit_idx: u32,
    theta_ptr: *const ParameterWrapper,
    phi_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() || phi_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    if qubit_idx as usize >= wrapper.inner.num_qubits() {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let phi = unsafe { &(*phi_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    let pval = ParameterValue::Param(phi.clone());
    match wrapper.inner.rxy(Qubit::new(qubit_idx), tval, pval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RXX gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rxx_param(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper.inner.rxx(Qubit::new(a), Qubit::new(b), tval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RYY gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_ryy_param(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper.inner.ryy(Qubit::new(a), Qubit::new(b), tval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZZ gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rzz_param(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper.inner.rzz(Qubit::new(a), Qubit::new(b), tval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply RZX gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_rzx_param(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper.inner.rzx(Qubit::new(a), Qubit::new(b), tval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRX gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_crx_param(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper
        .inner
        .crx(Qubit::new(ctrl_idx), Qubit::new(target_idx), tval)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRY gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_cry_param(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper
        .inner
        .cry(Qubit::new(ctrl_idx), Qubit::new(target_idx), tval)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply CRZ gate with symbolic parameter.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_crz_param(
    ptr: *mut CircuitWrapper,
    ctrl_idx: u32,
    target_idx: u32,
    theta_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if ctrl_idx as usize >= n || target_idx as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    match wrapper
        .inner
        .crz(Qubit::new(ctrl_idx), Qubit::new(target_idx), tval)
    {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Apply fSim gate with symbolic parameters.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_fsim_param(
    ptr: *mut CircuitWrapper,
    a: u32,
    b: u32,
    theta_ptr: *const ParameterWrapper,
    phi_ptr: *const ParameterWrapper,
) -> i32 {
    if ptr.is_null() || theta_ptr.is_null() || phi_ptr.is_null() {
        return -1;
    }
    let wrapper = unsafe { &mut *ptr };
    let n = wrapper.inner.num_qubits();
    if a as usize >= n || b as usize >= n {
        return -2;
    }
    let theta = unsafe { &(*theta_ptr).inner };
    let phi = unsafe { &(*phi_ptr).inner };
    let tval = ParameterValue::Param(theta.clone());
    let pval = ParameterValue::Param(phi.clone());
    match wrapper.inner.fsim(Qubit::new(a), Qubit::new(b), tval, pval) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

/// Assign parameters to a circuit and return a new circuit.
/// The bindings format is "var1:value1,var2:value2,..."
/// Returns a pointer to a new CircuitWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn circuit_assign_params(
    circuit: *const CircuitWrapper,
    bindings: *const c_char,
) -> *mut CircuitWrapper {
    if circuit.is_null() {
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &(*circuit).inner };

    // Parse bindings
    let mut map: HashMap<&str, f64> = HashMap::new();
    if !bindings.is_null() {
        if let Ok(c_str) = unsafe { CStr::from_ptr(bindings).to_str() } {
            for pair in c_str.split(',') {
                let parts: Vec<&str> = pair.splitn(2, ':').collect();
                if parts.len() == 2 {
                    if let Ok(val) = parts[1].parse::<f64>() {
                        map.insert(parts[0], val);
                    }
                }
            }
        }
    }

    // Assign parameters and create new circuit
    let bindings_opt = if map.is_empty() { None } else { Some(map) };
    match wrapper.assign_parameters(&bindings_opt) {
        Ok(new_circuit) => Box::into_raw(Box::new(CircuitWrapper { inner: new_circuit })),
        Err(_) => std::ptr::null_mut(),
    }
}
