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

//! IR (Intermediate Representation) module for C binding.
//!
//! Provides C-compatible APIs for parsing QCIS and OpenQASM 2.0 formats.

// Allow clippy warnings for FFI functions that dereference raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::circuit::CircuitWrapper;
use cqlib_core::ir::qasm2_dumps as core_qasm2_dumps;
use cqlib_core::ir::qcis_dumps as core_qcis_dumps;
use cqlib_core::ir::{qasm2_loads, qcis_loads};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Parse a QCIS string and create a circuit.
/// Returns a pointer to a CircuitWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn qcis_load(qcis_str: *const c_char) -> *mut CircuitWrapper {
    if qcis_str.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(qcis_str) };
    let qcis = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match qcis_loads(qcis) {
        Ok(circuit) => Box::into_raw(Box::new(CircuitWrapper { inner: circuit })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get the last error message.
#[unsafe(no_mangle)]
pub extern "C" fn qcis_error() -> *mut c_char {
    CString::new("QCIS parse error").unwrap().into_raw()
}

/// Dump a circuit to QCIS string.
/// Returns a pointer to a C string, or NULL on error.
/// The caller must free the returned string with cstring_free.
#[unsafe(no_mangle)]
pub extern "C" fn qcis_dumps(circuit: *const CircuitWrapper) -> *mut c_char {
    if circuit.is_null() {
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &(*circuit).inner };
    match core_qcis_dumps(wrapper) {
        Ok(s) => CString::new(s).unwrap().into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Parse an OpenQASM 2.0 string and create a circuit.
/// Returns a pointer to a CircuitWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn qasm2_load(qasm_str: *const c_char) -> *mut CircuitWrapper {
    if qasm_str.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(qasm_str) };
    let qasm = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match qasm2_loads(qasm) {
        Ok(circuit) => Box::into_raw(Box::new(CircuitWrapper { inner: circuit })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get the last error message for QASM2.
#[unsafe(no_mangle)]
pub extern "C" fn qasm2_error() -> *mut c_char {
    CString::new("QASM2 parse error").unwrap().into_raw()
}

/// Dump a circuit to OpenQASM 2.0 string.
/// Returns a pointer to a C string, or NULL on error.
/// The caller must free the returned string with cstring_free.
#[unsafe(no_mangle)]
pub extern "C" fn qasm2_dumps(circuit: *const CircuitWrapper) -> *mut c_char {
    if circuit.is_null() {
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &(*circuit).inner };
    match core_qasm2_dumps(wrapper) {
        Ok(s) => CString::new(s).unwrap().into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a C string allocated by the library.
#[unsafe(no_mangle)]
pub extern "C" fn cstring_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}
