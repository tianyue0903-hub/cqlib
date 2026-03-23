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

use binding_c::circuit::{
    circuit_ccx, circuit_crx, circuit_crx_param, circuit_crz, circuit_crz_param,
    circuit_cry, circuit_cry_param, circuit_cx, circuit_cy, circuit_cz,
    circuit_fsim, circuit_fsim_param, circuit_free, circuit_h, circuit_measure,
    circuit_new, circuit_num_qubits, circuit_rxy, circuit_rxy_param, circuit_rxx,
    circuit_rxx_param, circuit_rzx, circuit_rzx_param, circuit_rzz, circuit_rzz_param,
    circuit_ryy, circuit_ryy_param, circuit_rx, circuit_ry,
    circuit_rz, circuit_s, circuit_swap, circuit_sx,
    circuit_t, circuit_x, circuit_x2m, circuit_x2p, circuit_y, circuit_y2m, circuit_y2p,
    circuit_z, circuit_reset, param_evaluate, param_free, param_parse,
};
use binding_c::device::{
    device_new, device_free, device_num_qubits, device_add_qubit_properties,
    device_set_default_single_qubit_error, device_get_default_single_qubit_error,
    device_set_default_two_qubit_error, device_get_default_two_qubit_error,
    device_get_t1, topology_new_line, topology_free, topology_num_qubits,
    qubit_prop_new, qubit_prop_free, qubit_prop_set_t1, qubit_prop_get_t1,
};
use binding_c::ir::{qasm2_load, qcis_load};

#[test]
fn test_circuit_lifecycle() {
    // 1. Create Circuit
    let num_qubits = 3;
    let ptr = circuit_new(num_qubits);
    assert!(!ptr.is_null(), "Circuit pointer should not be null");

    // 2. Check Property
    let count = circuit_num_qubits(ptr);
    assert_eq!(count, num_qubits, "Qubit count mismatch");

    // 3. Apply Gates (Success cases)
    let res_h = circuit_h(ptr, 0);
    assert_eq!(res_h, 0, "H gate should return 0 on success");

    let res_rx = circuit_rx(ptr, 1, 3.14159);
    assert_eq!(res_rx, 0, "RX gate should return 0 on success");

    let res_cx = circuit_cx(ptr, 0, 1);
    assert_eq!(res_cx, 0, "CX gate should return 0 on success");

    // 4. Apply Gates (Error cases)
    let res_err_bounds = circuit_h(ptr, 99); // Invalid index
    assert_eq!(
        res_err_bounds, -2,
        "Should return -2 for index out of bounds"
    );

    // 5. Free Memory
    circuit_free(ptr);
}

#[test]
fn test_null_pointer_safety() {
    // Functions should handle null pointers gracefully without crashing
    circuit_free(std::ptr::null_mut());

    let count = circuit_num_qubits(std::ptr::null());
    assert_eq!(count, 0);

    let res = circuit_h(std::ptr::null_mut(), 0);
    assert_eq!(res, -1);
}

#[test]
fn test_circuit_gates() {
    let ptr = circuit_new(4);
    assert!(!ptr.is_null());

    // Test various gates
    assert_eq!(circuit_x(ptr, 0), 0);
    assert_eq!(circuit_y(ptr, 0), 0);
    assert_eq!(circuit_z(ptr, 0), 0);
    assert_eq!(circuit_s(ptr, 0), 0);
    assert_eq!(circuit_t(ptr, 0), 0);
    assert_eq!(circuit_sx(ptr, 0), 0);

    // Test two-qubit gates
    assert_eq!(circuit_cz(ptr, 0, 1), 0);
    assert_eq!(circuit_cy(ptr, 0, 1), 0);
    assert_eq!(circuit_swap(ptr, 0, 1), 0);

    // Test parameterized gates
    assert_eq!(circuit_ry(ptr, 0, 1.0), 0);
    assert_eq!(circuit_rz(ptr, 0, 2.0), 0);
    assert_eq!(circuit_rx(ptr, 0, 0.5), 0);

    // Test rotation gates
    assert_eq!(circuit_x2p(ptr, 0), 0);
    assert_eq!(circuit_x2m(ptr, 0), 0);
    assert_eq!(circuit_y2p(ptr, 0), 0);
    assert_eq!(circuit_y2m(ptr, 0), 0);

    // Test advanced parameterized gates
    assert_eq!(circuit_rxy(ptr, 0, 0.1, 0.2), 0);
    assert_eq!(circuit_rxx(ptr, 0, 1, 0.3), 0);
    assert_eq!(circuit_ryy(ptr, 0, 1, 0.4), 0);
    assert_eq!(circuit_rzz(ptr, 0, 1, 0.5), 0);
    assert_eq!(circuit_rzx(ptr, 0, 1, 0.6), 0);
    assert_eq!(circuit_crx(ptr, 0, 1, 0.7), 0);
    assert_eq!(circuit_cry(ptr, 0, 1, 0.8), 0);
    assert_eq!(circuit_crz(ptr, 0, 1, 0.9), 0);
    // toffoli gate with valid target
    assert_eq!(circuit_ccx(ptr, 0, 1, 2), 0);
    // out-of-bounds target should return error code
    assert_eq!(circuit_ccx(ptr, 0, 1, 4), -2);
    assert_eq!(circuit_measure(ptr, 0), 0);
    assert_eq!(circuit_reset(ptr, 0), 0);
    assert_eq!(circuit_fsim(ptr, 0, 1, 0.2, 0.3), 0);

    circuit_free(ptr);
}

#[test]
fn test_parameter_parsing() {
    // Test parameter parsing
    let expr = std::ffi::CString::new("theta").unwrap();
    let param_ptr = param_parse(expr.as_ptr());
    assert!(!param_ptr.is_null(), "Parameter parsing should succeed");

    // Evaluate with bindings
    let bindings = std::ffi::CString::new("theta:3.14159").unwrap();
    let result = param_evaluate(param_ptr, bindings.as_ptr());
    assert!(
        (result - 3.14159).abs() < 1e-6,
        "Parameter evaluation failed"
    );

    param_free(param_ptr);
}

#[test]
fn test_symbolic_gates() {
    let ptr = circuit_new(3);
    assert!(!ptr.is_null());

    let theta = std::ffi::CString::new("theta").unwrap();
    let phi = std::ffi::CString::new("phi").unwrap();
    let th_ptr = param_parse(theta.as_ptr());
    let ph_ptr = param_parse(phi.as_ptr());
    assert!(!th_ptr.is_null() && !ph_ptr.is_null());

    assert_eq!(circuit_rxy_param(ptr, 0, th_ptr, ph_ptr), 0);
    assert_eq!(circuit_rxx_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_ryy_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_rzz_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_rzx_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_crx_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_cry_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_crz_param(ptr, 0, 1, th_ptr), 0);
    assert_eq!(circuit_fsim_param(ptr, 0, 1, th_ptr, ph_ptr), 0);

    param_free(th_ptr);
    param_free(ph_ptr);
    circuit_free(ptr);
}

#[test]
fn test_qcis_parsing() {
    // QCIS uses CZ, not CX. Also uses H for Hadamard.
    let qcis = std::ffi::CString::new("H Q0\nCZ Q0 Q1\n").unwrap();
    let ptr = qcis_load(qcis.as_ptr());
    assert!(!ptr.is_null(), "QCIS parsing should succeed");

    let num = circuit_num_qubits(ptr);
    assert_eq!(num, 2, "QCIS should create circuit with 2 qubits");

    circuit_free(ptr);
}

#[test]
fn test_qasm2_parsing() {
    let qasm =
        std::ffi::CString::new("OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n").unwrap();
    let ptr = qasm2_load(qasm.as_ptr());
    assert!(!ptr.is_null(), "QASM2 parsing should succeed");

    let num = circuit_num_qubits(ptr);
    assert_eq!(num, 2, "QASM2 should create circuit with 2 qubits");

    circuit_free(ptr);
}

// Device FFI functionality tests
#[test]
fn test_device_and_topology() {
    // Test topology creation
    let qubits = [0u32, 1, 2];
    let topo = topology_new_line(qubits.as_ptr(), 3);
    assert!(!topo.is_null(), "Topology should not be null");
    assert_eq!(topology_num_qubits(topo), 3, "Topology should have 3 qubits");
    
    // Test device creation
    let device = device_new(std::ffi::CStr::from_bytes_with_nul(b"TestDevice\0").unwrap().as_ptr(), topo);
    assert!(!device.is_null(), "Device should not be null");
    assert_eq!(device_num_qubits(device), 3, "Device should have 3 qubits");
    
    // Test gate error setters/getters
    assert_eq!(device_set_default_single_qubit_error(device, 0.01), 0, "Should set single-qubit error");
    assert_eq!(device_get_default_single_qubit_error(device), 0.01, "Should get single-qubit error");
    
    assert_eq!(device_set_default_two_qubit_error(device, 0.02), 0, "Should set two-qubit error");
    assert_eq!(device_get_default_two_qubit_error(device), 0.02, "Should get two-qubit error");
    
    // Test qubit properties
    let qubit_prop = qubit_prop_new(0.001);
    assert!(!qubit_prop.is_null(), "QubitProp should not be null");
    assert_eq!(qubit_prop_set_t1(qubit_prop, 50.0), 0, "Should set T1");
    assert_eq!(qubit_prop_get_t1(qubit_prop), 50.0, "Should get T1");
    
    assert_eq!(device_add_qubit_properties(device, 0, qubit_prop), 0, "Should add qubit properties");
    assert_eq!(device_get_t1(device, 0), 50.0, "Should retrieve qubit T1");
    
    // Cleanup
    qubit_prop_free(qubit_prop);
    device_free(device);
    topology_free(topo);
}

#[test]
fn test_device_null_pointer_safety() {
    // Verify Device FFI handles null pointers gracefully
    assert_eq!(device_num_qubits(std::ptr::null()), 0, "NULL device should return 0 qubits");
    device_free(std::ptr::null_mut()); // Should not crash
    
    // Verify getters on NULL return default/error values
    assert_eq!(device_get_default_single_qubit_error(std::ptr::null()), -1.0, "NULL device should return -1.0");
}
