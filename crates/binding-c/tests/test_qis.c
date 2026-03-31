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

// Test statevector lifecycle
void test_statevector_lifecycle() {
    printf("Running test_statevector_lifecycle...\n");

    StatevectorWrapper* sv = statevector_new(2);
    assert(sv != NULL);
    assert(statevector_num_qubits(sv) == 2);

    // Apply gates
    assert(statevector_h(sv, 0) == 0);
    assert(statevector_cx(sv, 0, 1) == 0);

    // Test probabilities
    double probs[4];
    assert(statevector_probabilities(sv, probs, 4) == 0);
    // Bell state: |00> + |11> -> probs [0.5, 0, 0, 0.5]
    assert(fabs(probs[0] - 0.5) < 1e-10);
    assert(fabs(probs[3] - 0.5) < 1e-10);

    statevector_free(sv);
    printf("test_statevector_lifecycle PASSED\n");
}

// Test density matrix lifecycle
void test_density_matrix_lifecycle() {
    printf("Running test_density_matrix_lifecycle...\n");

    DensityMatrixWrapper* dm = density_matrix_new(2);
    assert(dm != NULL);
    assert(density_matrix_num_qubits(dm) == 2);

    // Apply gates
    assert(density_matrix_h(dm, 0) == 0);
    assert(density_matrix_cx(dm, 0, 1) == 0);

    // Test probabilities
    double probs[4];
    assert(density_matrix_probabilities(dm, probs, 4) == 0);
    assert(fabs(probs[0] - 0.5) < 1e-10);
    assert(fabs(probs[3] - 0.5) < 1e-10);

    density_matrix_free(dm);
    printf("test_density_matrix_lifecycle PASSED\n");
}

// Test density matrix noise
void test_density_matrix_noise_lifecycle() {
    printf("Running test_density_matrix_noise_lifecycle...\n");

    DensityMatrixNoiseWrapper* dmn = density_matrix_noise_new(2);
    assert(dmn != NULL);
    assert(density_matrix_noise_num_qubits(dmn) == 2);

    // Apply gates with noise
    assert(density_matrix_noise_h(dmn, 0) == 0);
    assert(density_matrix_noise_cx(dmn, 0, 1) == 0);

    density_matrix_noise_free(dmn);
    printf("test_density_matrix_noise_lifecycle PASSED\n");
}

// Test Pauli string operations
void test_pauli_string_operations() {
    printf("Running test_pauli_string_operations...\n");

    PauliStringWrapper* ps = pauli_string_new(3);
    assert(ps != NULL);
    assert(pauli_string_num_qubits(ps) == 3);

    // Set Pauli operators
    assert(pauli_string_set_pauli(ps, 0, 1) == 0); // X
    assert(pauli_string_set_pauli(ps, 1, 3) == 0); // Z
    assert(pauli_string_set_pauli(ps, 2, 0) == 0); // I

    // Get Pauli operators
    assert(pauli_string_get_pauli(ps, 0) == 1); // X
    assert(pauli_string_get_pauli(ps, 1) == 3); // Z
    assert(pauli_string_get_pauli(ps, 2) == 0); // I

    // Test string representation
    char* str = pauli_string_to_string(ps);
    assert(str != NULL);
    assert(strcmp(str, "XZI") == 0);
    pauli_string_free_string(str);

    pauli_string_free(ps);
    printf("test_pauli_string_operations PASSED\n");
}

// Test Hamiltonian operations
void test_hamiltonian_operations() {
    printf("Running test_hamiltonian_operations...\n");

    HamiltonianWrapper* h = hamiltonian_new(2);
    assert(h != NULL);
    assert(hamiltonian_num_qubits(h) == 2);

    // Add terms
    assert(hamiltonian_add_term(h, "XX", 1.0, 0.0) == 0);
    assert(hamiltonian_add_term(h, "ZZ", 0.5, 0.0) == 0);
    assert(hamiltonian_num_terms(h) == 2);

    hamiltonian_free(h);
    printf("test_hamiltonian_operations PASSED\n");
}

// Test observable expectation values
void test_observable_expectation() {
    printf("Running test_observable_expectation...\n");

    // Create Hamiltonian H = Z_0
    HamiltonianWrapper* h = hamiltonian_new(1);
    assert(hamiltonian_add_term(h, "Z", 1.0, 0.0) == 0);

    // Create statevector |1>
    StatevectorWrapper* sv = statevector_new(1);
    assert(statevector_x(sv, 0) == 0); // |0> -> |1>

    // Compute <1|Z|1>
    double real, imag;
    assert(observable_expectation_sv(h, sv, &real, &imag) == 0);
    assert(fabs(real - (-1.0)) < 1e-10);
    assert(fabs(imag) < 1e-10);

    // Test with density matrix
    DensityMatrixWrapper* dm = density_matrix_new(1);
    assert(density_matrix_x(dm, 0) == 0);
    assert(observable_expectation_dm(h, dm, &real, &imag) == 0);
    assert(fabs(real - (-1.0)) < 1e-10);
    assert(fabs(imag) < 1e-10);

    hamiltonian_free(h);
    statevector_free(sv);
    density_matrix_free(dm);
    printf("test_observable_expectation PASSED\n");
}

// Test error handling
void test_error_handling() {
    printf("Running test_error_handling...\n");

    // Test null pointers
    assert(statevector_h(NULL, 0) == -1);
    assert(density_matrix_h(NULL, 0) == -1);
    assert(pauli_string_set_pauli(NULL, 0, 1) == -1);

    // Test out of bounds
    StatevectorWrapper* sv = statevector_new(2);
    assert(statevector_h(sv, 5) == -2);
    statevector_free(sv);

    PauliStringWrapper* ps = pauli_string_new(2);
    assert(pauli_string_set_pauli(ps, 5, 1) == -2);
    pauli_string_free(ps);

    printf("test_error_handling PASSED\n");
}

int main() {
    printf("Running QIS C binding tests...\n");

    test_statevector_lifecycle();
    test_density_matrix_lifecycle();
    test_density_matrix_noise_lifecycle();
    test_pauli_string_operations();
    test_hamiltonian_operations();
    test_observable_expectation();
    test_error_handling();

    printf("All QIS tests PASSED!\n");
    return 0;
}