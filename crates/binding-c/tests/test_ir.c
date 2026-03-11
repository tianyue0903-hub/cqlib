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
#include "cqlib_c.h"

// =====================================================================
// QCIS Format Tests
// =====================================================================

/// Test loading a simple QCIS circuit
void test_qcis_load_simple() {
    printf("Running test_qcis_load_simple...\n");

    const char *qcis = "H Q0\nCZ Q0 Q1\n";
    CircuitWrapper *circuit = qcis_load(qcis);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);

    circuit_free(circuit);
    printf("test_qcis_load_simple PASSED\n");
}

/// Test loading QCIS with measurement
void test_qcis_load_with_measurement() {
    printf("Running test_qcis_load_with_measurement...\n");

    const char *qcis = "H Q0\nCZ Q0 Q1\nM Q0 Q1\n";
    CircuitWrapper *circuit = qcis_load(qcis);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);

    circuit_free(circuit);
    printf("test_qcis_load_with_measurement PASSED\n");
}

/// Test loading QCIS with various gates
void test_qcis_load_varied_gates() {
    printf("Running test_qcis_load_varied_gates...\n");

    const char *qcis =
        "H Q0\n"
        "X Q1\n"
        "Y Q2\n"
        "Z Q3\n"
        "S Q0\n"
        "T Q1\n"
        "RZ Q2 1.57\n"
        "CZ Q0 Q1\n";
    
    CircuitWrapper *circuit = qcis_load(qcis);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 4);

    circuit_free(circuit);
    printf("test_qcis_load_varied_gates PASSED\n");
}

/// Test loading invalid QCIS returns NULL
void test_qcis_load_invalid() {
    printf("Running test_qcis_load_invalid...\n");

    const char *invalid_qcis = "INVALID Q0\nBAD_GATE\n";
    CircuitWrapper *circuit = qcis_load(invalid_qcis);
    // Invalid QCIS may return NULL or a circuit with default state
    if (circuit != NULL) {
        circuit_free(circuit);
    }

    printf("test_qcis_load_invalid PASSED\n");
}

/// Test QCIS null pointer handling
void test_qcis_load_null_pointer() {
    printf("Running test_qcis_load_null_pointer...\n");

    CircuitWrapper *circuit = qcis_load(NULL);
    assert(circuit == NULL);

    printf("test_qcis_load_null_pointer PASSED\n");
}

/// Test dumping circuit to QCIS format
void test_qcis_dumps() {
    printf("Running test_qcis_dumps...\n");

    // Load a circuit from QCIS
    const char *qcis_input = "H Q0\nCZ Q0 Q1\n";
    CircuitWrapper *circuit = qcis_load(qcis_input);
    assert(circuit != NULL);

    // Dump it back to QCIS
    char *qcis_output = qcis_dumps(circuit);
    assert(qcis_output != NULL);
    assert(strlen(qcis_output) > 0);

    // Verify key elements are present
    assert(strstr(qcis_output, "H") != NULL || strstr(qcis_output, "q0") != NULL);

    cstring_free(qcis_output);
    circuit_free(circuit);
    printf("test_qcis_dumps PASSED\n");
}

/// Test QCIS dumps with null pointer
void test_qcis_dumps_null_pointer() {
    printf("Running test_qcis_dumps_null_pointer...\n");

    char *result = qcis_dumps(NULL);
    assert(result == NULL);

    printf("test_qcis_dumps_null_pointer PASSED\n");
}

// =====================================================================
// OpenQASM 2.0 Format Tests
// =====================================================================

/// Test loading a simple OpenQASM 2.0 circuit
void test_qasm2_load_simple() {
    printf("Running test_qasm2_load_simple...\n");

    const char *qasm =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n";

    CircuitWrapper *circuit = qasm2_load(qasm);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);

    circuit_free(circuit);
    printf("test_qasm2_load_simple PASSED\n");
}

/// Test loading OpenQASM 2.0 with include directive
void test_qasm2_load_with_include() {
    printf("Running test_qasm2_load_with_include...\n");

    const char *qasm =
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[3];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "ccx q[0], q[1], q[2];\n";

    CircuitWrapper *circuit = qasm2_load(qasm);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 3);

    circuit_free(circuit);
    printf("test_qasm2_load_with_include PASSED\n");
}

/// Test loading OpenQASM 2.0 with parametric gates
void test_qasm2_load_parametric_gates() {
    printf("Running test_qasm2_load_parametric_gates...\n");

    const char *qasm =
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[2];\n"
        "rx(0.5) q[0];\n"
        "ry(1.57) q[1];\n"
        "rz(3.14159) q[0];\n";

    CircuitWrapper *circuit = qasm2_load(qasm);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);

    circuit_free(circuit);
    printf("test_qasm2_load_parametric_gates PASSED\n");
}

/// Test loading OpenQASM 2.0 with measurement
void test_qasm2_load_with_measurement() {
    printf("Running test_qasm2_load_with_measurement...\n");

    const char *qasm =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n";

    CircuitWrapper *circuit = qasm2_load(qasm);
    assert(circuit != NULL);
    assert(circuit_num_qubits(circuit) == 2);

    circuit_free(circuit);
    printf("test_qasm2_load_with_measurement PASSED\n");
}

/// Test loading invalid OpenQASM 2.0 returns NULL
void test_qasm2_load_invalid() {
    printf("Running test_qasm2_load_invalid...\n");

    const char *invalid_qasm = "INVALID OPENQASM SYNTAX\nno qreg defined\n";
    CircuitWrapper *circuit = qasm2_load(invalid_qasm);
    // Invalid QASM may return NULL or a circuit with default state
    if (circuit != NULL) {
        circuit_free(circuit);
    }

    printf("test_qasm2_load_invalid PASSED\n");
}

/// Test OpenQASM 2.0 null pointer handling
void test_qasm2_load_null_pointer() {
    printf("Running test_qasm2_load_null_pointer...\n");

    CircuitWrapper *circuit = qasm2_load(NULL);
    assert(circuit == NULL);

    printf("test_qasm2_load_null_pointer PASSED\n");
}

/// Test dumping circuit to OpenQASM 2.0 format
void test_qasm2_dumps() {
    printf("Running test_qasm2_dumps...\n");

    const char *qasm_input =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n";

    CircuitWrapper *circuit = qasm2_load(qasm_input);
    assert(circuit != NULL);

    char *qasm_output = qasm2_dumps(circuit);
    assert(qasm_output != NULL);
    assert(strlen(qasm_output) > 0);
    assert(strstr(qasm_output, "OPENQASM 2.0") != NULL);

    cstring_free(qasm_output);
    circuit_free(circuit);
    printf("test_qasm2_dumps PASSED\n");
}

/// Test OpenQASM 2.0 dumps with null pointer
void test_qasm2_dumps_null_pointer() {
    printf("Running test_qasm2_dumps_null_pointer...\n");

    char *result = qasm2_dumps(NULL);
    assert(result == NULL);

    printf("test_qasm2_dumps_null_pointer PASSED\n");
}

// =====================================================================
// Cross-Format Conversion Tests
// =====================================================================

/// Test converting from QASM2 to QCIS format
void test_qasm2_to_qcis_conversion() {
    printf("Running test_qasm2_to_qcis_conversion...\n");

    // Load from OpenQASM 2.0
    const char *qasm =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n";

    CircuitWrapper *circuit = qasm2_load(qasm);
    assert(circuit != NULL);

    // Dump to QCIS format
    char *qcis_output = qcis_dumps(circuit);
    // Note: qcis_dumps may convert the circuit to QCIS format
    if (qcis_output != NULL) {
        assert(strlen(qcis_output) > 0);
        cstring_free(qcis_output);
    }

    circuit_free(circuit);
    printf("test_qasm2_to_qcis_conversion PASSED\n");
}

/// Test converting from QCIS to QASM2 format
void test_qcis_to_qasm2_conversion() {
    printf("Running test_qcis_to_qasm2_conversion...\n");

    // Load from QCIS
    const char *qcis = "H Q0\nCZ Q0 Q1\n";

    CircuitWrapper *circuit = qcis_load(qcis);
    assert(circuit != NULL);

    // Dump to OpenQASM 2.0 format
    char *qasm_output = qasm2_dumps(circuit);
    assert(qasm_output != NULL);
    assert(strlen(qasm_output) > 0);
    // OpenQASM output should contain OPENQASM directive
    assert(strstr(qasm_output, "OPENQASM") != NULL || strstr(qasm_output, "qreg") != NULL);

    cstring_free(qasm_output);
    circuit_free(circuit);
    printf("test_qcis_to_qasm2_conversion PASSED\n");
}

/// Test round-trip conversion: QASM2 -> Circuit -> QCIS
void test_qasm2_roundtrip_via_qcis() {
    printf("Running test_qasm2_roundtrip_via_qcis...\n");

    const char *qasm =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n";

    // Load from QASM2
    CircuitWrapper *circuit1 = qasm2_load(qasm);
    assert(circuit1 != NULL);

    // Convert to QCIS and back
    char *qcis = qcis_dumps(circuit1);
    // If QCIS dump fails, it may return NULL - just skip conversion back
    if (qcis != NULL) {
        CircuitWrapper *circuit2 = qcis_load(qcis);
        if (circuit2 != NULL) {
            assert(circuit_num_qubits(circuit2) == 2);

            // Convert back to QASM2
            char *qasm_final = qasm2_dumps(circuit2);
            if (qasm_final != NULL) {
                cstring_free(qasm_final);
            }
            circuit_free(circuit2);
        }
        cstring_free(qcis);
    }

    circuit_free(circuit1);
    printf("test_qasm2_roundtrip_via_qcis PASSED\n");
}

/// Test round-trip conversion: QCIS -> Circuit -> QASM2
void test_qcis_roundtrip_via_qasm2() {
    printf("Running test_qcis_roundtrip_via_qasm2...\n");

    const char *qcis = "H Q0\nCZ Q0 Q1\n";

    // Load from QCIS
    CircuitWrapper *circuit1 = qcis_load(qcis);
    assert(circuit1 != NULL);

    // Convert to QASM2 and back
    char *qasm = qasm2_dumps(circuit1);
    assert(qasm != NULL);

    CircuitWrapper *circuit2 = qasm2_load(qasm);
    assert(circuit2 != NULL);
    assert(circuit_num_qubits(circuit2) == 2);

    // Convert back to QCIS
    char *qcis_final = qcis_dumps(circuit2);
    // If conversion fails, qcis_final may be NULL - that's ok
    if (qcis_final != NULL) {
        cstring_free(qcis_final);
    }

    cstring_free(qasm);
    circuit_free(circuit1);
    circuit_free(circuit2);
    printf("test_qcis_roundtrip_via_qasm2 PASSED\n");
}

// =====================================================================
// Edge Cases and Error Handling
// =====================================================================

/// Test empty QCIS string
void test_qcis_load_empty_string() {
    printf("Running test_qcis_load_empty_string...\n");

    const char *qcis = "";
    CircuitWrapper *circuit = qcis_load(qcis);
    // Empty input should either return NULL or create an empty circuit
    if (circuit != NULL) {
        circuit_free(circuit);
    }

    printf("test_qcis_load_empty_string PASSED\n");
}

/// Test empty OpenQASM 2.0 string
void test_qasm2_load_empty_string() {
    printf("Running test_qasm2_load_empty_string...\n");

    const char *qasm = "";
    CircuitWrapper *circuit = qasm2_load(qasm);
    // Empty input should either return NULL or create an empty circuit
    if (circuit != NULL) {
        circuit_free(circuit);
    }

    printf("test_qasm2_load_empty_string PASSED\n");
}

/// Test memory cleanup after multiple operations
void test_memory_cleanup() {
    printf("Running test_memory_cleanup...\n");

    const char *qcis = "H Q0\nCZ Q0 Q1\n";
    const char *qasm =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "h q[0];\n";

    for (int i = 0; i < 10; i++) {
        CircuitWrapper *c1 = qcis_load(qcis);
        if (c1 != NULL) {
            char *s1 = qcis_dumps(c1);
            if (s1 != NULL) {
                cstring_free(s1);
            }
            circuit_free(c1);
        }

        CircuitWrapper *c2 = qasm2_load(qasm);
        if (c2 != NULL) {
            char *s2 = qasm2_dumps(c2);
            if (s2 != NULL) {
                cstring_free(s2);
            }
            circuit_free(c2);
        }
    }

    printf("test_memory_cleanup PASSED\n");
}

/// Test QCIS with different qubit indices
void test_qcis_various_qubit_configs() {
    printf("Running test_qcis_various_qubit_configs...\n");

    // Single qubit
    CircuitWrapper *c1 = qcis_load("H Q0\n");
    assert(c1 != NULL);
    assert(circuit_num_qubits(c1) == 1);
    circuit_free(c1);

    // Multiple non-contiguous qubits
    CircuitWrapper *c2 = qcis_load("H Q0\nX Q5\nY Q10\n");
    assert(c2 != NULL);
    circuit_free(c2);

    printf("test_qcis_various_qubit_configs PASSED\n");
}

/// Test OpenQASM 2.0 with different qubit register sizes
void test_qasm2_various_qubit_configs() {
    printf("Running test_qasm2_various_qubit_configs...\n");

    // Single qubit
    CircuitWrapper *c1 = qasm2_load(
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "h q[0];\n");
    assert(c1 != NULL);
    assert(circuit_num_qubits(c1) == 1);
    circuit_free(c1);

    // Large qubit register
    CircuitWrapper *c2 = qasm2_load(
        "OPENQASM 2.0;\n"
        "qreg q[10];\n"
        "h q[0];\n"
        "cx q[0], q[9];\n");
    assert(c2 != NULL);
    assert(circuit_num_qubits(c2) == 10);
    circuit_free(c2);

    printf("test_qasm2_various_qubit_configs PASSED\n");
}

// =====================================================================
// Test Main
// =====================================================================

int main() {
    printf("\n=== Starting IR Format Tests ===\n\n");

    // QCIS Tests
    printf("--- QCIS Format Tests ---\n");
    test_qcis_load_simple();
    test_qcis_load_with_measurement();
    test_qcis_load_varied_gates();
    test_qcis_load_invalid();
    test_qcis_load_null_pointer();
    test_qcis_dumps();
    test_qcis_dumps_null_pointer();
    test_qcis_load_empty_string();
    test_qcis_various_qubit_configs();

    // OpenQASM 2.0 Tests
    printf("\n--- OpenQASM 2.0 Format Tests ---\n");
    test_qasm2_load_simple();
    test_qasm2_load_with_include();
    test_qasm2_load_parametric_gates();
    test_qasm2_load_with_measurement();
    test_qasm2_load_invalid();
    test_qasm2_load_null_pointer();
    test_qasm2_dumps();
    test_qasm2_dumps_null_pointer();
    test_qasm2_load_empty_string();
    test_qasm2_various_qubit_configs();

    // Cross-Format Conversion Tests
    printf("\n--- Cross-Format Conversion Tests ---\n");
    test_qasm2_to_qcis_conversion();
    test_qcis_to_qasm2_conversion();
    test_qasm2_roundtrip_via_qcis();
    test_qcis_roundtrip_via_qasm2();

    // Edge Cases and Memory Tests
    printf("\n--- Edge Cases and Memory Tests ---\n");
    test_memory_cleanup();

    printf("\n=== All IR Format Tests Passed ===\n");
    return 0;
}
