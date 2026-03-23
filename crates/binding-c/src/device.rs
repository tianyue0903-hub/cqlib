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

//! Device module for C binding.
//!
//! Provides C-compatible APIs for quantum device topology and properties.

// Allow clippy warnings for FFI functions that dereference raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use cqlib_core::circuit::Qubit;
use cqlib_core::device::{Device, QubitProp, Topology};
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_char;

// =====================================================================
// Wrapper Types for C FFI
// =====================================================================

/// Wrapper for Topology to be used from C
pub struct TopologyWrapper {
    pub(crate) inner: Topology,
}

/// Wrapper for Device to be used from C
pub struct DeviceWrapper {
    pub(crate) inner: Device,
}

/// Wrapper for QubitProp to be used from C
pub struct QubitPropWrapper {
    pub(crate) inner: QubitProp,
}

// =====================================================================
// Topology C Interface
// =====================================================================

/// Create a new topology from arrays of qubits and couplings.
/// qubits: array of qubit indices
/// num_qubits: length of qubits array
/// Returns a pointer to TopologyWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn topology_new(
    qubits: *const u32,
    num_qubits: usize,
    couplings: *const u64,
    num_couplings: usize,
) -> *mut TopologyWrapper {
    if qubits.is_null() || num_qubits == 0 {
        return std::ptr::null_mut();
    }

    let qubit_vec: Vec<Qubit> = unsafe {
        std::slice::from_raw_parts(qubits, num_qubits)
            .iter()
            .copied()
            .map(Qubit::new)
            .collect()
    };

    // Parse couplings: each coupling is represented as (control << 32) | target
    let coupling_vec: Vec<(Qubit, Qubit, String)> = if !couplings.is_null() && num_couplings > 0 {
        unsafe {
            std::slice::from_raw_parts(couplings, num_couplings)
                .iter()
                .map(|&coupling| {
                    let target = (coupling & 0xFFFFFFFF) as u32;
                    let control = (coupling >> 32) as u32;
                    (
                        Qubit::new(control),
                        Qubit::new(target),
                        "".to_string(),
                    )
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    match Topology::new(qubit_vec, coupling_vec) {
        Ok(topo) => Box::into_raw(Box::new(TopologyWrapper { inner: topo })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Create a line topology (chain of connected qubits).
/// qubits: array of qubit indices
/// num_qubits: length of array
/// Returns pointer to TopologyWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn topology_new_line(qubits: *const u32, num_qubits: usize) -> *mut TopologyWrapper {
    if qubits.is_null() || num_qubits == 0 {
        return std::ptr::null_mut();
    }

    let qubit_vec: Vec<Qubit> = unsafe {
        std::slice::from_raw_parts(qubits, num_qubits)
            .iter()
            .copied()
            .map(Qubit::new)
            .collect()
    };

    Box::into_raw(Box::new(TopologyWrapper {
        inner: Topology::line(qubit_vec),
    }))
}

/// Free a topology.
#[unsafe(no_mangle)]
pub extern "C" fn topology_free(ptr: *mut TopologyWrapper) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Get the number of qubits in the topology.
#[unsafe(no_mangle)]
pub extern "C" fn topology_num_qubits(ptr: *const TopologyWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_qubits() }
}

/// Get the number of couplings (edges) in the topology.
#[unsafe(no_mangle)]
pub extern "C" fn topology_num_couplings(ptr: *const TopologyWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.num_couplings() }
}

/// Check if two qubits are connected in the topology.
/// Returns 1 if connected, 0 if not connected or error.
#[unsafe(no_mangle)]
pub extern "C" fn topology_is_connected(
    ptr: *const TopologyWrapper,
    control: u32,
    target: u32,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let ctrl_qubit = Qubit::new(control);
    let tgt_qubit = Qubit::new(target);

    let graph = unsafe { (*ptr).inner.graph() };
    let ctrl_idx = unsafe { (*ptr).inner.graph() }
        .node_indices()
        .find(|&idx| graph[idx] == ctrl_qubit);
    let tgt_idx = unsafe { (*ptr).inner.graph() }
        .node_indices()
        .find(|&idx| graph[idx] == tgt_qubit);

    if let (Some(ctrl), Some(tgt)) = (ctrl_idx, tgt_idx) {
        if unsafe { (*ptr).inner.graph() }.contains_edge(ctrl, tgt) {
            return 1;
        }
    }

    0
}

// =====================================================================
// QubitProp C Interface
// =====================================================================

/// Create a new QubitProp with readout error.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_new(readout_error: f64) -> *mut QubitPropWrapper {
    Box::into_raw(Box::new(QubitPropWrapper {
        inner: QubitProp::new(readout_error),
    }))
}

/// Free a QubitProp.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_free(ptr: *mut QubitPropWrapper) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Set T1 relaxation time (in microseconds) for a QubitProp.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_set_t1(ptr: *mut QubitPropWrapper, t1: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_t1(t1) };
    0
}

/// Set T2 dephasing time (in microseconds) for a QubitProp.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_set_t2(ptr: *mut QubitPropWrapper, t2: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_t2(t2) };
    0
}

/// Set frequency (in GHz) for a QubitProp.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_set_frequency(ptr: *mut QubitPropWrapper, frequency: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_frequency(frequency) };
    0
}

/// Set probability of measuring 0 given state was prepared in 1.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_set_prob_meas0_prep1(ptr: *mut QubitPropWrapper, prob: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_prob_meas0_prep1(prob) };
    0
}

/// Set probability of measuring 1 given state was prepared in 0.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_set_prob_meas1_prep0(ptr: *mut QubitPropWrapper, prob: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_prob_meas1_prep0(prob) };
    0
}

/// Get readout error from QubitProp.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_get_readout_error(ptr: *const QubitPropWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.readout_error() }
}

/// Get T1 from QubitProp (returns -1.0 if not set).
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_get_t1(ptr: *const QubitPropWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.t1().unwrap_or(-1.0) }
}

/// Get T2 from QubitProp (returns -1.0 if not set).
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_get_t2(ptr: *const QubitPropWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.t2().unwrap_or(-1.0) }
}

/// Get frequency from QubitProp (returns -1.0 if not set).
#[unsafe(no_mangle)]
pub extern "C" fn qubit_prop_get_frequency(ptr: *const QubitPropWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.frequency().unwrap_or(-1.0) }
}

// =====================================================================
// Device C Interface
// =====================================================================

/// Create a new Device with a name and topology.
/// Returns pointer to DeviceWrapper, or NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn device_new(
    name: *const c_char,
    topology: *mut TopologyWrapper,
) -> *mut DeviceWrapper {
    if name.is_null() || topology.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let topo = unsafe { (*topology).inner.clone() };
    let qubits: HashSet<Qubit> = topo.qubits().collect();

    match Device::new(c_str, qubits, topo) {
        Ok(device) => Box::into_raw(Box::new(DeviceWrapper { inner: device })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a Device.
#[unsafe(no_mangle)]
pub extern "C" fn device_free(ptr: *mut DeviceWrapper) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Get device name as a C string.
#[unsafe(no_mangle)]
pub extern "C" fn device_get_name(ptr: *const DeviceWrapper) -> *const c_char {
    if ptr.is_null() {
        return std::ptr::null();
    }
    // Note: This approach is problematic for long-lived access
    // In production, should use device_get_name_into_buffer
    unsafe { (*ptr).inner.name().as_ptr() as *const c_char }
}

/// Get number of qubits in the device.
#[unsafe(no_mangle)]
pub extern "C" fn device_num_qubits(ptr: *const DeviceWrapper) -> usize {
    if ptr.is_null() {
        return 0;
    }
    unsafe { (*ptr).inner.qubits().count() }
}

/// Set default T1 time (in microseconds).
#[unsafe(no_mangle)]
pub extern "C" fn device_set_default_t1(ptr: *mut DeviceWrapper, t1: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_default_t1(t1) };
    0
}

/// Get default T1 time (returns -1.0 - not directly accessible).
#[unsafe(no_mangle)]
pub extern "C" fn device_get_default_t1(ptr: *const DeviceWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    // Note: Device doesn't expose default T1 getter, use get_t1() for specific qubits
    -1.0
}

/// Set default T2 time (in microseconds).
#[unsafe(no_mangle)]
pub extern "C" fn device_set_default_t2(ptr: *mut DeviceWrapper, t2: f64) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_default_t2(t2) };
    0
}

/// Get default T2 time (returns -1.0 - not directly accessible).
#[unsafe(no_mangle)]
pub extern "C" fn device_get_default_t2(ptr: *const DeviceWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    // Note: Device doesn't expose default T2 getter, use get_t2() for specific qubits
    -1.0
}

/// Set default readout error.
#[unsafe(no_mangle)]
pub extern "C" fn device_set_default_readout_error(
    ptr: *mut DeviceWrapper,
    error: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_default_readout_error(error) };
    0
}

/// Get default readout error (returns -1.0 - not directly accessible).
#[unsafe(no_mangle)]
pub extern "C" fn device_get_default_readout_error(ptr: *const DeviceWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    // Note: Device doesn't expose default readout error getter, use get_readout_error() for specific qubits
    -1.0
}

/// Set default single-qubit gate error.
#[unsafe(no_mangle)]
pub extern "C" fn device_set_default_single_qubit_error(
    ptr: *mut DeviceWrapper,
    error: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_default_single_qubit_error(error) };
    0
}

/// Get default single-qubit gate error (returns -1.0 if not set).
#[unsafe(no_mangle)]
pub extern "C" fn device_get_default_single_qubit_error(ptr: *const DeviceWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.default_single_qubit_error().unwrap_or(-1.0) }
}

/// Set default two-qubit gate error.
#[unsafe(no_mangle)]
pub extern "C" fn device_set_default_two_qubit_error(
    ptr: *mut DeviceWrapper,
    error: f64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }
    unsafe { (*ptr).inner.set_default_two_qubit_error(error) };
    0
}

/// Get default two-qubit gate error (returns -1.0 if not set).
#[unsafe(no_mangle)]
pub extern "C" fn device_get_default_two_qubit_error(ptr: *const DeviceWrapper) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }
    unsafe { (*ptr).inner.default_two_qubit_error().unwrap_or(-1.0) }
}

/// Add qubit properties to a device.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn device_add_qubit_properties(
    ptr: *mut DeviceWrapper,
    qubit_idx: u32,
    prop: *mut QubitPropWrapper,
) -> i32 {
    if ptr.is_null() || prop.is_null() {
        return -1;
    }

    let qubit = Qubit::new(qubit_idx);
    let prop_inner = unsafe { (*prop).inner.clone() };

    match unsafe { (*ptr).inner.add_qubit_properties(qubit, prop_inner) } {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get T1 time for a specific qubit (returns -1.0 if not set).
/// Uses qubit-specific properties if available, else falls back to default.
#[unsafe(no_mangle)]
pub extern "C" fn device_get_t1(ptr: *const DeviceWrapper, qubit_idx: u32) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }

    let qubit = Qubit::new(qubit_idx);
    unsafe { (*ptr).inner.get_t1(qubit).unwrap_or(-1.0) }
}

/// Get T2 time for a specific qubit (returns -1.0 if not set).
/// Uses qubit-specific properties if available, else falls back to default.
#[unsafe(no_mangle)]
pub extern "C" fn device_get_t2(ptr: *const DeviceWrapper, qubit_idx: u32) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }

    let qubit = Qubit::new(qubit_idx);
    unsafe { (*ptr).inner.get_t2(qubit).unwrap_or(-1.0) }
}

/// Get readout error for a specific qubit.
/// Uses qubit-specific properties if available, else falls back to default.
#[unsafe(no_mangle)]
pub extern "C" fn device_get_readout_error(ptr: *const DeviceWrapper, qubit_idx: u32) -> f64 {
    if ptr.is_null() {
        return -1.0;
    }

    let qubit = Qubit::new(qubit_idx);
    unsafe { (*ptr).inner.get_readout_error(qubit).unwrap_or(-1.0) }
}

/// Get the topology of the device.
#[unsafe(no_mangle)]
pub extern "C" fn device_get_topology(ptr: *const DeviceWrapper) -> *mut TopologyWrapper {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let topo = unsafe { (*ptr).inner.topology().clone() };
    Box::into_raw(Box::new(TopologyWrapper { inner: topo }))
}
