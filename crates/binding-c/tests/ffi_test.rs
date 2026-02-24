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
    circuit_cx, circuit_cz, circuit_free, circuit_h, circuit_new, circuit_num_qubits, circuit_rx,
    circuit_ry, circuit_rz, circuit_s, circuit_swap, circuit_sx, circuit_t, circuit_x, circuit_x2m,
    circuit_x2p, circuit_y, circuit_y2m, circuit_y2p, circuit_z, param_evaluate, param_free,
    param_parse,
};
use binding_c::ir::{qasm2_load, qcis_load};

#[test]
fn test_circuit_lifecycle() {
    unsafe {
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
}

#[test]
fn test_null_pointer_safety() {
    unsafe {
        // Functions should handle null pointers gracefully without crashing
        circuit_free(std::ptr::null_mut());

        let count = circuit_num_qubits(std::ptr::null());
        assert_eq!(count, 0);

        let res = circuit_h(std::ptr::null_mut(), 0);
        assert_eq!(res, -1);
    }
}

#[test]
fn test_circuit_gates() {
    unsafe {
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
        assert_eq!(circuit_swap(ptr, 0, 1), 0);

        // Test parameterized gates
        assert_eq!(circuit_ry(ptr, 0, 1.0), 0);
        assert_eq!(circuit_rz(ptr, 0, 2.0), 0);

        // Test rotation gates
        assert_eq!(circuit_x2p(ptr, 0), 0);
        assert_eq!(circuit_x2m(ptr, 0), 0);
        assert_eq!(circuit_y2p(ptr, 0), 0);
        assert_eq!(circuit_y2m(ptr, 0), 0);

        circuit_free(ptr);
    }
}

#[test]
fn test_parameter_parsing() {
    unsafe {
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
}

#[test]
fn test_qcis_parsing() {
    unsafe {
        // QCIS uses CZ, not CX. Also uses H for Hadamard.
        let qcis = std::ffi::CString::new("H Q0\nCZ Q0 Q1\n").unwrap();
        let ptr = qcis_load(qcis.as_ptr());
        assert!(!ptr.is_null(), "QCIS parsing should succeed");

        let num = circuit_num_qubits(ptr);
        assert_eq!(num, 2, "QCIS should create circuit with 2 qubits");

        circuit_free(ptr);
    }
}

#[test]
fn test_qasm2_parsing() {
    unsafe {
        let qasm =
            std::ffi::CString::new("OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n").unwrap();
        let ptr = qasm2_load(qasm.as_ptr());
        assert!(!ptr.is_null(), "QASM2 parsing should succeed");

        let num = circuit_num_qubits(ptr);
        assert_eq!(num, 2, "QASM2 should create circuit with 2 qubits");

        circuit_free(ptr);
    }
}
