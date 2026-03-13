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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "cqlib_c.h"

// Define math constants for GCC compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// Test circuit creation and basic operations
void test_circuit_lifecycle() {
    printf("Running test_circuit_lifecycle...\n");

    CircuitWrapper* c = circuit_new(5);
    assert(c != NULL);
    assert(circuit_num_qubits(c) == 5);

    // Test Gate Success
    assert(circuit_h(c, 0) == 0);
    assert(circuit_cx(c, 0, 1) == 0);
    assert(circuit_rx(c, 2, 1.23) == 0);

    // Test Gate Failure (Out of bounds)
    assert(circuit_h(c, 10) == -2);

    circuit_free(c);
    printf("test_circuit_lifecycle PASSED\n");
}

// Test all single-qubit gates
void test_single_qubit_gates() {
    printf("Running test_single_qubit_gates...\n");

    CircuitWrapper* c = circuit_new(4);
    assert(c != NULL);

    // Test all single-qubit gates
    assert(circuit_x(c, 0) == 0);
    assert(circuit_y(c, 1) == 0);
    assert(circuit_z(c, 2) == 0);
    assert(circuit_h(c, 3) == 0);
    assert(circuit_s(c, 0) == 0);
    assert(circuit_t(c, 1) == 0);
    assert(circuit_sx(c, 2) == 0);

    // Test rotation gates
    assert(circuit_x2p(c, 0) == 0);
    assert(circuit_x2m(c, 1) == 0);
    assert(circuit_y2p(c, 2) == 0);
    assert(circuit_y2m(c, 3) == 0);

    circuit_free(c);
    printf("test_single_qubit_gates PASSED\n");
}

// Test two-qubit gates
void test_two_qubit_gates() {
    printf("Running test_two_qubit_gates...\n");

    CircuitWrapper* c = circuit_new(4);
    assert(c != NULL);

    // Test CX (CNOT)
    assert(circuit_cx(c, 0, 1) == 0);

    // Test CY
    assert(circuit_cy(c, 0, 1) == 0);

    // Test CZ
    assert(circuit_cz(c, 2, 3) == 0);

    // Test SWAP
    assert(circuit_swap(c, 0, 3) == 0);

    circuit_free(c);
    printf("test_two_qubit_gates PASSED\n");
}

// Test parameterized gates with float values
void test_parameterized_gates() {
    printf("Running test_parameterized_gates...\n");

    CircuitWrapper* c = circuit_new(3);
    assert(c != NULL);

    // Test RX, RY, RZ
    assert(circuit_rx(c, 0, 1.57) == 0);
    assert(circuit_ry(c, 1, 2.0) == 0);
    assert(circuit_rz(c, 2, 0.5) == 0);

    // Test with different values
    assert(circuit_rx(c, 0, -1.57) == 0);
    assert(circuit_ry(c, 1, M_PI) == 0);
    assert(circuit_rz(c, 2, M_PI_2) == 0);

    // Test advanced parameterized gates
    assert(circuit_rxy(c, 0, 0.1, 0.2) == 0);
    assert(circuit_rxx(c, 0, 1, 0.3) == 0);
    assert(circuit_ryy(c, 0, 1, 0.4) == 0);
    assert(circuit_rzz(c, 0, 1, 0.5) == 0);
    assert(circuit_rzx(c, 0, 1, 0.6) == 0);
    assert(circuit_crx(c, 0, 1, 0.7) == 0);
    assert(circuit_cry(c, 0, 1, 0.8) == 0);
    assert(circuit_crz(c, 0, 1, 0.9) == 0);
    assert(circuit_fsim(c, 0, 1, 0.2, 0.3) == 0);

    circuit_free(c);
    printf("test_parameterized_gates PASSED\n");
}

// Test advanced gates and operations
void test_advanced_gates() {
    printf("Running test_advanced_gates...\n");

    CircuitWrapper* c = circuit_new(4);
    assert(c != NULL);

    // Test CCX (Toffoli gate)
    assert(circuit_ccx(c, 0, 1, 2) == 0);

    // Test measure
    assert(circuit_measure(c, 0) == 0);
    assert(circuit_measure(c, 1) == 0);

    // Test reset
    assert(circuit_reset(c, 2) == 0);
    assert(circuit_reset(c, 3) == 0);

    // Test error cases
    assert(circuit_ccx(c, 0, 1, 10) == -2); // out of bounds
    assert(circuit_measure(c, 10) == -2);   // out of bounds
    assert(circuit_reset(c, 10) == -2);     // out of bounds

    circuit_free(c);
    printf("test_advanced_gates PASSED\n");
}

// Test QCIS parsing
void test_qcis_parsing() {
    printf("Running test_qcis_parsing...\n");

    const char* qcis_code =
        "H Q0\n"
        "X Q1\n"
        "Y Q2\n"
        "CZ Q0 Q1\n"
        "CZ Q1 Q2\n";

    CircuitWrapper* c = qcis_load(qcis_code);
    assert(c != NULL);
    assert(circuit_num_qubits(c) == 3);

    circuit_free(c);
    printf("test_qcis_parsing PASSED\n");
}

// Test OpenQASM 2.0 parsing
void test_qasm2_parsing() {
    printf("Running test_qasm2_parsing...\n");

    const char* qasm_code =
        "OPENQASM 2.0;\n"
        "qreg q[3];\n"
        "h q[0];\n"
        "x q[1];\n"
        "y q[2];\n"
        "cx q[0], q[1];\n"
        "cz q[1], q[2];\n";

    CircuitWrapper* c = qasm2_load(qasm_code);
    assert(c != NULL);
    assert(circuit_num_qubits(c) == 3);

    circuit_free(c);
    printf("test_qasm2_parsing PASSED\n");
}

// Test parameter parsing and evaluation
void test_parameter() {
    printf("Running test_parameter...\n");

    // Test parsing a simple parameter
    ParameterWrapper* param = param_parse("theta");
    assert(param != NULL);

    // Evaluate with theta = 0
    double result = param_evaluate(param, "theta:0");
    assert(fabs(result - 0.0) < 1e-6);

    // Evaluate with theta = pi
    result = param_evaluate(param, "theta:3.14159");
    assert(fabs(result - 3.14159) < 1e-4);

    // additional symbol parameters test
    ParameterWrapper* phi = param_parse("phi");
    assert(phi != NULL);
    CircuitWrapper* c_sym = circuit_new(3);
    assert(c_sym != NULL);
    assert(circuit_rxy_param(c_sym, 0, param, phi) == 0);
    assert(circuit_rxx_param(c_sym, 0, 1, param) == 0);
    assert(circuit_ryy_param(c_sym, 0, 1, param) == 0);
    assert(circuit_rzz_param(c_sym, 0, 1, param) == 0);
    assert(circuit_rzx_param(c_sym, 0, 1, param) == 0);
    assert(circuit_crx_param(c_sym, 0, 1, param) == 0);
    assert(circuit_cry_param(c_sym, 0, 1, param) == 0);
    assert(circuit_crz_param(c_sym, 0, 1, param) == 0);
    assert(circuit_fsim_param(c_sym, 0, 1, param, phi) == 0);
    circuit_free(c_sym);
    param_free(phi);

    param_free(param);

    // Test parameter expression
    param = param_parse("pi/2");
    assert(param != NULL);
    result = param_evaluate(param, "");
    assert(fabs(result - M_PI_2) < 1e-6);
    param_free(param);

    printf("test_parameter PASSED\n");
}

// Test parameterized gates with symbolic parameters
void test_parameterized_gates_with_symbols() {
    printf("Running test_parameterized_gates_with_symbols...\n");

    CircuitWrapper* c = circuit_new(2);
    assert(c != NULL);

    ParameterWrapper* theta = param_parse("theta");
    assert(theta != NULL);

    // Apply RX with symbolic parameter
    assert(circuit_rx_param(c, 0, theta) == 0);

    // Apply RY with symbolic parameter
    assert(circuit_ry_param(c, 1, theta) == 0);

    // additional symbol gates
    assert(circuit_rxy_param(c, 0, theta, theta) == 0);
    assert(circuit_rxx_param(c, 0, 1, theta) == 0);
    assert(circuit_ryy_param(c, 0, 1, theta) == 0);
    assert(circuit_rzz_param(c, 0, 1, theta) == 0);
    assert(circuit_rzx_param(c, 0, 1, theta) == 0);
    assert(circuit_crx_param(c, 0, 1, theta) == 0);
    assert(circuit_cry_param(c, 0, 1, theta) == 0);
    assert(circuit_crz_param(c, 0, 1, theta) == 0);
    assert(circuit_fsim_param(c, 0, 1, theta, theta) == 0);

    param_free(theta);
    circuit_free(c);
    printf("test_parameterized_gates_with_symbols PASSED\n");
}

// Test null pointer safety
void test_null_safety() {
    printf("Running test_null_safety...\n");

    // These should not crash
    circuit_free(NULL);
    param_free(NULL);
    circuit_num_qubits(NULL);
    circuit_h(NULL, 0);
    circuit_cx(NULL, 0, 1);
    circuit_rx(NULL, 0, 1.0);
    param_evaluate(NULL, "");
    qcis_load(NULL);
    qasm2_load(NULL);

    printf("test_null_safety PASSED\n");
}

// Test memory stress and resource management
void test_memory_stress() {
    printf("Running test_memory_stress...\n");

    // Create multiple circuits to test memory management
    CircuitWrapper* circuits[10];
    ParameterWrapper* params[10];

    for (int i = 0; i < 10; i++) {
        circuits[i] = circuit_new(5);
        assert(circuits[i] != NULL);

        // Add some gates
        circuit_h(circuits[i], 0);
        circuit_cx(circuits[i], 0, 1);
        circuit_rx(circuits[i], 2, 1.0);

        // Create parameters
        params[i] = param_parse("theta");
        assert(params[i] != NULL);
        circuit_rx_param(circuits[i], 3, params[i]);
    }

    // Free all resources
    for (int i = 0; i < 10; i++) {
        circuit_free(circuits[i]);
        param_free(params[i]);
    }

    printf("test_memory_stress PASSED\n");
}

int main() {
    printf("=== Starting C ABI Tests for Circuit Module ===\n\n");

    test_circuit_lifecycle();
    test_single_qubit_gates();
    test_two_qubit_gates();
    test_parameterized_gates();
    test_advanced_gates();
    test_qcis_parsing();
    test_qasm2_parsing();
    test_parameter();
    test_parameterized_gates_with_symbols();
    test_null_safety();
    test_memory_stress();

    printf("\n=== All C ABI Tests for Circuit Module Passed ===\n");
    return 0;
}
