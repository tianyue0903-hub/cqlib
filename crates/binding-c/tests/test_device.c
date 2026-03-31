/*
 * This code is part of Cqlib.
 *
 * (C) Copyright China Telecom Quantum Group 2026
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

/*
 * Comprehensive Device Module Tests
 * Tests topology creation, device properties, and qubit-specific configurations.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

// Forward declarations of device functions
typedef struct TopologyWrapper TopologyWrapper;
typedef struct DeviceWrapper DeviceWrapper;
typedef struct QubitPropWrapper QubitPropWrapper;

// Topology functions
TopologyWrapper *topology_new(const uint32_t *qubits, size_t num_qubits,
                              const uint64_t *couplings, size_t num_couplings);
TopologyWrapper *topology_new_line(const uint32_t *qubits, size_t num_qubits);
void topology_free(TopologyWrapper *ptr);
size_t topology_num_qubits(const TopologyWrapper *ptr);
size_t topology_num_couplings(const TopologyWrapper *ptr);
int topology_is_connected(const TopologyWrapper *ptr, uint32_t control,
                          uint32_t target);

// QubitProp functions
QubitPropWrapper *qubit_prop_new(double readout_error);
void qubit_prop_free(QubitPropWrapper *ptr);
int qubit_prop_set_t1(QubitPropWrapper *ptr, double t1);
int qubit_prop_set_t2(QubitPropWrapper *ptr, double t2);
int qubit_prop_set_frequency(QubitPropWrapper *ptr, double frequency);
int qubit_prop_set_prob_meas0_prep1(QubitPropWrapper *ptr, double prob);
int qubit_prop_set_prob_meas1_prep0(QubitPropWrapper *ptr, double prob);
double qubit_prop_get_readout_error(const QubitPropWrapper *ptr);
double qubit_prop_get_t1(const QubitPropWrapper *ptr);
double qubit_prop_get_t2(const QubitPropWrapper *ptr);
double qubit_prop_get_frequency(const QubitPropWrapper *ptr);

// Device functions
DeviceWrapper *device_new(const char *name, TopologyWrapper *topology);
void device_free(DeviceWrapper *ptr);
const char *device_get_name(const DeviceWrapper *ptr);
size_t device_num_qubits(const DeviceWrapper *ptr);
int device_set_default_t1(DeviceWrapper *ptr, double t1);
int device_set_default_t2(DeviceWrapper *ptr, double t2);
int device_set_default_readout_error(DeviceWrapper *ptr, double error);
int device_set_default_single_qubit_error(DeviceWrapper *ptr, double error);
double device_get_default_single_qubit_error(const DeviceWrapper *ptr);
int device_set_default_two_qubit_error(DeviceWrapper *ptr, double error);
double device_get_default_two_qubit_error(const DeviceWrapper *ptr);
int device_add_qubit_properties(DeviceWrapper *ptr, uint32_t qubit_idx,
                                QubitPropWrapper *prop);
double device_get_t1(const DeviceWrapper *ptr, uint32_t qubit_idx);
double device_get_t2(const DeviceWrapper *ptr, uint32_t qubit_idx);
double device_get_readout_error(const DeviceWrapper *ptr, uint32_t qubit_idx);
TopologyWrapper *device_get_topology(const DeviceWrapper *ptr);

// =====================================================================
// Test Utility Functions
// =====================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_EQ(actual, expected, msg)                                       \
  do {                                                                          \
    if ((actual) != (expected)) {                                              \
      printf("FAIL: %s (expected %ld, got %ld)\n", (msg), (long)(expected),   \
             (long)(actual));                                                   \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

#define ASSERT_TRUE(condition, msg)                                            \
  do {                                                                          \
    if (!(condition)) {                                                         \
      printf("FAIL: %s\n", (msg));                                             \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

#define ASSERT_FALSE(condition, msg)                                           \
  do {                                                                          \
    if ((condition)) {                                                          \
      printf("FAIL: %s\n", (msg));                                             \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

#define ASSERT_DOUBLE_EQ(actual, expected, msg)                                \
  do {                                                                          \
    if (fabs((actual) - (expected)) > 1e-9) {                                  \
      printf("FAIL: %s (expected %f, got %f)\n", (msg), (expected),            \
             (actual));                                                         \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

#define ASSERT_NULL(ptr, msg)                                                  \
  do {                                                                          \
    if ((ptr) != NULL) {                                                        \
      printf("FAIL: %s (expected NULL, got non-NULL)\n", (msg));               \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

#define ASSERT_NOT_NULL(ptr, msg)                                              \
  do {                                                                          \
    if ((ptr) == NULL) {                                                        \
      printf("FAIL: %s (expected non-NULL, got NULL)\n", (msg));               \
      tests_failed++;                                                           \
    } else {                                                                    \
      printf("PASS: %s\n", (msg));                                             \
      tests_passed++;                                                           \
    }                                                                            \
  } while (0)

// =====================================================================
// Topology Tests
// =====================================================================

void test_topology_creation() {
  printf("\n=== Topology Creation Tests ===\n");

  // Test 1: Create topology with explicit qubits and couplings
  uint32_t qubits[] = {0, 1, 2};
  size_t num_qubits = 3;

  // Couplings: (0->1), (1->2)
  uint64_t couplings[] = {
      (0ULL << 32) | 1,  // Control: 0, Target: 1
      (1ULL << 32) | 2   // Control: 1, Target: 2
  };
  size_t num_couplings = 2;

  TopologyWrapper *topo =
      topology_new(qubits, num_qubits, couplings, num_couplings);
  ASSERT_NOT_NULL(topo, "test_topology_creation: Create topology");

  ASSERT_EQ(topology_num_qubits(topo), 3, "test_topology_creation: 3 qubits");
  ASSERT_EQ(topology_num_couplings(topo), 2,
            "test_topology_creation: 2 couplings");

  // Verify connectivity
  ASSERT_EQ(topology_is_connected(topo, 0, 1), 1,
            "test_topology_creation: 0-1 connected");
  ASSERT_EQ(topology_is_connected(topo, 1, 2), 1,
            "test_topology_creation: 1-2 connected");
  ASSERT_EQ(topology_is_connected(topo, 2, 1), 0,
            "test_topology_creation: 2-1 not connected");

  topology_free(topo);
}

void test_topology_line() {
  printf("\n=== Topology Line Tests ===\n");

  uint32_t qubits[] = {0, 1, 2, 3, 4};
  size_t num_qubits = 5;

  TopologyWrapper *topo = topology_new_line(qubits, num_qubits);
  ASSERT_NOT_NULL(topo, "test_topology_line: Create line topology");

  ASSERT_EQ(topology_num_qubits(topo), 5, "test_topology_line: 5 qubits");
  // Line topology of 5 qubits has 4 couplings: 0-1, 1-2, 2-3, 3-4
  ASSERT_EQ(topology_num_couplings(topo), 4, "test_topology_line: 4 couplings");

  // Verify line connectivity
  ASSERT_EQ(topology_is_connected(topo, 0, 1), 1,
            "test_topology_line: 0-1 connected");
  ASSERT_EQ(topology_is_connected(topo, 1, 2), 1,
            "test_topology_line: 1-2 connected");
  ASSERT_EQ(topology_is_connected(topo, 0, 2), 0,
            "test_topology_line: 0-2 not connected");

  topology_free(topo);
}

void test_topology_edge_cases() {
  printf("\n=== Topology Edge Cases ===\n");

  // Test with NULL pointer
  ASSERT_EQ(topology_num_qubits(NULL), 0,
            "test_topology_edge_cases: NULL qubits count");
  ASSERT_EQ(topology_num_couplings(NULL), 0,
            "test_topology_edge_cases: NULL couplings count");
  ASSERT_EQ(topology_is_connected(NULL, 0, 1), -1,
            "test_topology_edge_cases: NULL connectivity check");

  // Test with single qubit
  uint32_t single_qubit[] = {0};
  TopologyWrapper *topo = topology_new_line(single_qubit, 1);
  ASSERT_NOT_NULL(topo, "test_topology_edge_cases: Single qubit topology");
  ASSERT_EQ(topology_num_qubits(topo), 1,
            "test_topology_edge_cases: Single qubit count");
  topology_free(topo);
}

// =====================================================================
// QubitProp Tests
// =====================================================================

void test_qubit_prop_creation() {
  printf("\n=== QubitProp Creation Tests ===\n");

  // Test 1: Create QubitProp with readout error
  double readout_error = 0.01;
  QubitPropWrapper *prop = qubit_prop_new(readout_error);
  ASSERT_NOT_NULL(prop, "test_qubit_prop_creation: Create QubitProp");

  ASSERT_DOUBLE_EQ(qubit_prop_get_readout_error(prop), readout_error,
                   "test_qubit_prop_creation: Readout error");

  qubit_prop_free(prop);
}

void test_qubit_prop_t1_t2() {
  printf("\n=== QubitProp T1/T2 Tests ===\n");

  QubitPropWrapper *prop = qubit_prop_new(0.001);
  ASSERT_NOT_NULL(prop, "test_qubit_prop_t1_t2: Create QubitProp");

  // Test T1 setting
  double t1 = 25.5;
  ASSERT_EQ(qubit_prop_set_t1(prop, t1), 0, "test_qubit_prop_t1_t2: Set T1");
  ASSERT_DOUBLE_EQ(qubit_prop_get_t1(prop), t1, "test_qubit_prop_t1_t2: Get T1");

  // Test T2 setting
  double t2 = 50.0;
  ASSERT_EQ(qubit_prop_set_t2(prop, t2), 0, "test_qubit_prop_t1_t2: Set T2");
  ASSERT_DOUBLE_EQ(qubit_prop_get_t2(prop), t2, "test_qubit_prop_t1_t2: Get T2");

  qubit_prop_free(prop);
}

void test_qubit_prop_frequency() {
  printf("\n=== QubitProp Frequency Tests ===\n");

  QubitPropWrapper *prop = qubit_prop_new(0.001);
  ASSERT_NOT_NULL(prop, "test_qubit_prop_frequency: Create QubitProp");

  double frequency = 5.25;
  ASSERT_EQ(qubit_prop_set_frequency(prop, frequency), 0,
            "test_qubit_prop_frequency: Set frequency");
  ASSERT_DOUBLE_EQ(qubit_prop_get_frequency(prop), frequency,
                   "test_qubit_prop_frequency: Get frequency");

  qubit_prop_free(prop);
}

void test_qubit_prop_measurement_errors() {
  printf("\n=== QubitProp Measurement Error Tests ===\n");

  QubitPropWrapper *prop = qubit_prop_new(0.001);
  ASSERT_NOT_NULL(prop, "test_qubit_prop_measurement_errors: Create QubitProp");

  // Set measurement errors
  double prob_0_1 = 0.02;
  double prob_1_0 = 0.03;

  ASSERT_EQ(qubit_prop_set_prob_meas0_prep1(prop, prob_0_1), 0,
            "test_qubit_prop_measurement_errors: Set P(0|1)");
  ASSERT_EQ(qubit_prop_set_prob_meas1_prep0(prop, prob_1_0), 0,
            "test_qubit_prop_measurement_errors: Set P(1|0)");

  qubit_prop_free(prop);
}

void test_qubit_prop_unset_values() {
  printf("\n=== QubitProp Unset Values Tests ===\n");

  QubitPropWrapper *prop = qubit_prop_new(0.05);
  ASSERT_NOT_NULL(prop, "test_qubit_prop_unset_values: Create QubitProp");

  // Initially, T1, T2, and frequency are not set
  ASSERT_DOUBLE_EQ(qubit_prop_get_t1(prop), -1.0,
                   "test_qubit_prop_unset_values: Unset T1");
  ASSERT_DOUBLE_EQ(qubit_prop_get_t2(prop), -1.0,
                   "test_qubit_prop_unset_values: Unset T2");
  ASSERT_DOUBLE_EQ(qubit_prop_get_frequency(prop), -1.0,
                   "test_qubit_prop_unset_values: Unset frequency");

  qubit_prop_free(prop);
}

void test_qubit_prop_null_pointer() {
  printf("\n=== QubitProp Null Pointer Tests ===\n");

  ASSERT_EQ(qubit_prop_set_t1(NULL, 10.0), -1,
            "test_qubit_prop_null_pointer: Set T1 on NULL");
  ASSERT_EQ(qubit_prop_set_t2(NULL, 20.0), -1,
            "test_qubit_prop_null_pointer: Set T2 on NULL");
  ASSERT_DOUBLE_EQ(qubit_prop_get_t1(NULL), -1.0,
                   "test_qubit_prop_null_pointer: Get T1 on NULL");
  ASSERT_DOUBLE_EQ(qubit_prop_get_readout_error(NULL), -1.0,
                   "test_qubit_prop_null_pointer: Get readout error on NULL");
}

// =====================================================================
// Device Tests
// =====================================================================

void test_device_creation() {
  printf("\n=== Device Creation Tests ===\n");

  // Create topology
  uint32_t qubits[] = {0, 1, 2};
  TopologyWrapper *topo = topology_new_line(qubits, 3);
  ASSERT_NOT_NULL(topo, "test_device_creation: Create topology");

  // Create device
  DeviceWrapper *device = device_new("TestDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_creation: Create device");

  ASSERT_EQ(device_num_qubits(device), 3, "test_device_creation: 3 qubits");

  device_free(device);
  topology_free(topo);
}

void test_device_default_properties() {
  printf("\n=== Device Default Properties Tests ===\n");

  uint32_t qubits[] = {0, 1, 2};
  TopologyWrapper *topo = topology_new_line(qubits, 3);
  DeviceWrapper *device = device_new("PropDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_default_properties: Create device");

  // Test default T1 (setter only - getter not implemented)
  ASSERT_EQ(device_set_default_t1(device, 30.0), 0,
            "test_device_default_properties: Set T1");

  // Test default T2 (setter only - getter not implemented)
  ASSERT_EQ(device_set_default_t2(device, 60.0), 0,
            "test_device_default_properties: Set T2");

  device_free(device);
  topology_free(topo);
}

void test_device_readout_error() {
  printf("\n=== Device Readout Error Tests ===\n");

  uint32_t qubits[] = {0, 1};
  TopologyWrapper *topo = topology_new_line(qubits, 2);
  DeviceWrapper *device = device_new("ReadoutDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_readout_error: Create device");

  // Test readout error (setter only - getter not implemented)
  ASSERT_EQ(device_set_default_readout_error(device, 0.005), 0,
            "test_device_readout_error: Set readout error");

  device_free(device);
  topology_free(topo);
}

void test_device_gate_errors() {
  printf("\n=== Device Gate Error Tests ===\n");

  uint32_t qubits[] = {0, 1};
  TopologyWrapper *topo = topology_new_line(qubits, 2);
  DeviceWrapper *device = device_new("GateErrorDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_gate_errors: Create device");

  // Test single-qubit gate error
  ASSERT_DOUBLE_EQ(device_get_default_single_qubit_error(device), -1.0,
                   "test_device_gate_errors: Initial single-qubit error unset");
  ASSERT_EQ(device_set_default_single_qubit_error(device, 0.001), 0,
            "test_device_gate_errors: Set single-qubit error");
  ASSERT_DOUBLE_EQ(device_get_default_single_qubit_error(device), 0.001,
                   "test_device_gate_errors: Get single-qubit error");

  // Test two-qubit gate error
  ASSERT_DOUBLE_EQ(device_get_default_two_qubit_error(device), -1.0,
                   "test_device_gate_errors: Initial two-qubit error unset");
  ASSERT_EQ(device_set_default_two_qubit_error(device, 0.01), 0,
            "test_device_gate_errors: Set two-qubit error");
  ASSERT_DOUBLE_EQ(device_get_default_two_qubit_error(device), 0.01,
                   "test_device_gate_errors: Get two-qubit error");

  device_free(device);
  topology_free(topo);
}

void test_device_qubit_properties() {
  printf("\n=== Device Qubit Properties Tests ===\n");

  uint32_t qubits[] = {0, 1, 2};
  TopologyWrapper *topo = topology_new_line(qubits, 3);
  DeviceWrapper *device = device_new("QubitPropDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_qubit_properties: Create device");

  // Set default properties
  device_set_default_t1(device, 25.0);
  device_set_default_t2(device, 50.0);
  device_set_default_readout_error(device, 0.01);

  // Create qubit-specific properties
  QubitPropWrapper *qprop = qubit_prop_new(0.002);
  qubit_prop_set_t1(qprop, 30.0);
  qubit_prop_set_t2(qprop, 60.0);

  // Add properties to device
  ASSERT_EQ(device_add_qubit_properties(device, 0, qprop), 0,
            "test_device_qubit_properties: Add qubit properties");

  // Query properties
  ASSERT_DOUBLE_EQ(device_get_t1(device, 0), 30.0,
                   "test_device_qubit_properties: Qubit 0 T1");
  ASSERT_DOUBLE_EQ(device_get_t2(device, 0), 60.0,
                   "test_device_qubit_properties: Qubit 0 T2");
  ASSERT_DOUBLE_EQ(device_get_readout_error(device, 0), 0.002,
                   "test_device_qubit_properties: Qubit 0 readout error");

  // Qubit 1 should use defaults
  ASSERT_DOUBLE_EQ(device_get_t1(device, 1), 25.0,
                   "test_device_qubit_properties: Qubit 1 T1 (default)");
  ASSERT_DOUBLE_EQ(device_get_t2(device, 1), 50.0,
                   "test_device_qubit_properties: Qubit 1 T2 (default)");
  ASSERT_DOUBLE_EQ(device_get_readout_error(device, 1), 0.01,
                   "test_device_qubit_properties: Qubit 1 readout error (default)");

  qubit_prop_free(qprop);
  device_free(device);
  topology_free(topo);
}

void test_device_multiple_qubits() {
  printf("\n=== Device Multiple Qubits Tests ===\n");

  uint32_t qubits[] = {0, 1, 2, 3};
  TopologyWrapper *topo = topology_new_line(qubits, 4);
  DeviceWrapper *device = device_new("MultiQubitDevice", topo);
  ASSERT_NOT_NULL(device, "test_device_multiple_qubits: Create device");

  device_set_default_t1(device, 20.0);
  device_set_default_t2(device, 40.0);

  // Add specific properties for qubits 0, 2
  QubitPropWrapper *qprop0 = qubit_prop_new(0.001);
  qubit_prop_set_t1(qprop0, 35.0);
  device_add_qubit_properties(device, 0, qprop0);

  QubitPropWrapper *qprop2 = qubit_prop_new(0.002);
  qubit_prop_set_t1(qprop2, 40.0);
  device_add_qubit_properties(device, 2, qprop2);

  // Verify specific and default properties
  ASSERT_DOUBLE_EQ(device_get_t1(device, 0), 35.0,
                   "test_device_multiple_qubits: Qubit 0 T1");
  ASSERT_DOUBLE_EQ(device_get_t1(device, 1), 20.0,
                   "test_device_multiple_qubits: Qubit 1 T1 (default)");
  ASSERT_DOUBLE_EQ(device_get_t1(device, 2), 40.0,
                   "test_device_multiple_qubits: Qubit 2 T1");
  ASSERT_DOUBLE_EQ(device_get_t1(device, 3), 20.0,
                   "test_device_multiple_qubits: Qubit 3 T1 (default)");

  qubit_prop_free(qprop0);
  qubit_prop_free(qprop2);
  device_free(device);
  topology_free(topo);
}

void test_device_null_pointer() {
  printf("\n=== Device Null Pointer Tests ===\n");

  ASSERT_EQ(device_num_qubits(NULL), 0,
            "test_device_null_pointer: NULL num_qubits");
  ASSERT_EQ(device_set_default_t1(NULL, 10.0), -1,
            "test_device_null_pointer: NULL set_default_t1");
  ASSERT_NULL(device_get_name(NULL), "test_device_null_pointer: NULL get_name");
}

void test_device_topology_retrieval() {
  printf("\n=== Device Topology Retrieval Tests ===\n");

  uint32_t qubits[] = {0, 1, 2};
  TopologyWrapper *original_topo = topology_new_line(qubits, 3);
  DeviceWrapper *device = device_new("TopoDevice", original_topo);
  ASSERT_NOT_NULL(device, "test_device_topology_retrieval: Create device");

  // Retrieve topology from device
  TopologyWrapper *retrieved_topo = device_get_topology(device);
  ASSERT_NOT_NULL(retrieved_topo,
                  "test_device_topology_retrieval: Retrieve topology");

  ASSERT_EQ(topology_num_qubits(retrieved_topo), 3,
            "test_device_topology_retrieval: Retrieved topology qubits");
  ASSERT_EQ(topology_num_couplings(retrieved_topo), 2,
            "test_device_topology_retrieval: Retrieved topology couplings");

  device_free(device);
  topology_free(original_topo);
  topology_free(retrieved_topo);
}

void test_device_name() {
  printf("\n=== Device Name Tests ===\n");

  uint32_t qubits[] = {0, 1};
  TopologyWrapper *topo = topology_new_line(qubits, 2);

  const char *device_name = "MyQuantumDevice";
  DeviceWrapper *device = device_new(device_name, topo);
  ASSERT_NOT_NULL(device, "test_device_name: Create device with name");

  const char *retrieved_name = device_get_name(device);
  ASSERT_NOT_NULL(retrieved_name, "test_device_name: Retrieved name");

  device_free(device);
  topology_free(topo);
}

// =====================================================================
// Integration Tests
// =====================================================================

void test_complex_topology_device() {
  printf("\n=== Complex Topology Device Integration Test ===\n");

  // Create complex topology (grid-like structure)
  uint32_t qubits[] = {0, 1, 2, 3, 4, 5};
  uint64_t couplings[] = {
      (0ULL << 32) | 1,  // 0-1
      (1ULL << 32) | 2,  // 1-2
      (0ULL << 32) | 3,  // 0-3
      (1ULL << 32) | 4,  // 1-4
      (2ULL << 32) | 5   // 2-5
  };

  TopologyWrapper *topo = topology_new(qubits, 6, couplings, 5);
  ASSERT_NOT_NULL(topo,
                  "test_complex_topology_device: Create complex topology");

  DeviceWrapper *device = device_new("ComplexDevice", topo);
  ASSERT_NOT_NULL(device, "test_complex_topology_device: Create device");

  // Configure defaults
  device_set_default_t1(device, 30.0);
  device_set_default_t2(device, 60.0);
  device_set_default_readout_error(device, 0.005);
  device_set_default_single_qubit_error(device, 0.001);
  device_set_default_two_qubit_error(device, 0.01);

  // Set specific properties for some qubits
  QubitPropWrapper *qprop_high_noise = qubit_prop_new(0.01);
  qubit_prop_set_t1(qprop_high_noise, 20.0);
  device_add_qubit_properties(device, 0, qprop_high_noise);

  QubitPropWrapper *qprop_good = qubit_prop_new(0.001);
  qubit_prop_set_t1(qprop_good, 40.0);
  device_add_qubit_properties(device, 5, qprop_good);

  // Verify the configuration
  ASSERT_EQ(device_num_qubits(device), 6,
            "test_complex_topology_device: Total qubits");
  ASSERT_DOUBLE_EQ(device_get_default_single_qubit_error(device), 0.001,
                   "test_complex_topology_device: Default single-qubit error");
  ASSERT_DOUBLE_EQ(device_get_default_two_qubit_error(device), 0.01,
                   "test_complex_topology_device: Default two-qubit error");

  ASSERT_DOUBLE_EQ(device_get_t1(device, 0), 20.0,
                   "test_complex_topology_device: Qubit 0 T1");
  ASSERT_DOUBLE_EQ(device_get_readout_error(device, 0), 0.01,
                   "test_complex_topology_device: Qubit 0 readout error");

  ASSERT_DOUBLE_EQ(device_get_t1(device, 5), 40.0,
                   "test_complex_topology_device: Qubit 5 T1");
  ASSERT_DOUBLE_EQ(device_get_readout_error(device, 5), 0.001,
                   "test_complex_topology_device: Qubit 5 readout error");

  qubit_prop_free(qprop_high_noise);
  qubit_prop_free(qprop_good);
  device_free(device);
  topology_free(topo);
}

// =====================================================================
// Main Test Runner
// =====================================================================

int main() {
  printf("================================\n");
  printf("Device Module C Binding Tests\n");
  printf("================================\n");

  // Topology tests
  test_topology_creation();
  test_topology_line();
  test_topology_edge_cases();

  // QubitProp tests
  test_qubit_prop_creation();
  test_qubit_prop_t1_t2();
  test_qubit_prop_frequency();
  test_qubit_prop_measurement_errors();
  test_qubit_prop_unset_values();
  test_qubit_prop_null_pointer();

  // Device tests
  test_device_creation();
  test_device_default_properties();
  test_device_readout_error();
  test_device_gate_errors();
  test_device_qubit_properties();
  test_device_multiple_qubits();
  test_device_null_pointer();
  test_device_topology_retrieval();
  test_device_name();

  // Integration tests
  test_complex_topology_device();

  // Print summary
  printf("\n================================\n");
  printf("Test Summary\n");
  printf("================================\n");
  printf("Tests passed: %d\n", tests_passed);
  printf("Tests failed: %d\n", tests_failed);
  printf("Total tests: %d\n", tests_passed + tests_failed);
  printf("================================\n");

  return tests_failed == 0 ? 0 : 1;
}
