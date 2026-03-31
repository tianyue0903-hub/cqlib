# C Binding for Cqlib

This crate provides C bindings for `cqlib-core`, allowing you to use the quantum circuit library from C/C++ applications.

## Features

- **Quantum Circuit Operations**: Create and manipulate quantum circuits with various quantum gates
- **Parameterized Circuits**: Support for symbolic parameters with expression parsing and binding
- **IR Format Support**: Parse and dump QCIS and OpenQASM 2.0 formats
- **C ABI Compatible**: Pure C API with no C++ dependencies

## Build

```bash
# Build the Rust library (release mode recommended for smaller binary)
cargo build -p binding-c --release

# The header file is automatically generated at include/cqlib_c.h
ls crates/binding-c/include/cqlib_c.h
```

## Running the Example

A simple C example is provided in `examples/main.c`.

```bash
# 1. Ensure the Rust library is built
cargo build -p binding-c --release

# 2. Compile the C example
gcc crates/binding-c/examples/main.c \
    -I crates/binding-c/include \
    -L target/release \
    -lbinding_c \
    -lm \
    -o target/release/example_main

# 3. Run it
./target/release/example_main
```

Output:
```
QCIS (symbolic):
RX Q0 theta
RY Q1 phi
CZ Q0 Q1
H Q0

QCIS (assigned):
RX Q0 0.5
RY Q1 1.57
CZ Q0 Q1
H Q0
```

## Testing

### Rust Integration Tests

```bash
cargo test -p binding-c
```

### C ABI Tests

```bash
# 1. Compile the C test
gcc crates/binding-c/tests/test_c_abi.c \
    -I crates/binding-c/include \
    -L target/release \
    -lbinding_c \
    -lm \
    -o target/release/test_c_abi

# 2. Run it
./target/release/test_c_abi
```

## API Reference

### Circuit Module

| Function | Description |
|----------|-------------|
| `circuit_new(size_t num_qubits)` | Create a new circuit |
| `circuit_free(CircuitWrapper* ptr)` | Free circuit memory |
| `circuit_num_qubits(const CircuitWrapper* ptr)` | Get qubit count |

### Quantum Gates

| Function | Description |
|----------|-------------|
| `circuit_h(CircuitWrapper*, uint32_t qubit)` | H gate |
| `circuit_x(CircuitWrapper*, uint32_t qubit)` | X gate |
| `circuit_y(CircuitWrapper*, uint32_t qubit)` | Y gate |
| `circuit_z(CircuitWrapper*, uint32_t qubit)` | Z gate |
| `circuit_s(CircuitWrapper*, uint32_t qubit)` | S gate |
| `circuit_t(CircuitWrapper*, uint32_t qubit)` | T gate |
| `circuit_sx(CircuitWrapper*, uint32_t qubit)` | SX gate |
| `circuit_x2p(CircuitWrapper*, uint32_t qubit)` | X/2 rotation |
| `circuit_x2m(CircuitWrapper*, uint32_t qubit)` | -X/2 rotation |
| `circuit_y2p(CircuitWrapper*, uint32_t qubit)` | Y/2 rotation |
| `circuit_y2m(CircuitWrapper*, uint32_t qubit)` | -Y/2 rotation |
| `circuit_cx(CircuitWrapper*, uint32_t ctrl, uint32_t target)` | CNOT gate |
| `circuit_cz(CircuitWrapper*, uint32_t ctrl, uint32_t target)` | CZ gate |
| `circuit_swap(CircuitWrapper*, uint32_t idx1, uint32_t idx2)` | SWAP gate |
| `circuit_cy(CircuitWrapper*, uint32_t ctrl, uint32_t target)` | CY gate |
| `circuit_rx(CircuitWrapper*, uint32_t qubit, double theta)` | RX gate (float) |
| `circuit_ry(CircuitWrapper*, uint32_t qubit, double theta)` | RY gate (float) |
| `circuit_rz(CircuitWrapper*, uint32_t qubit, double theta)` | RZ gate (float) |
| `circuit_rxy(CircuitWrapper*, uint32_t qubit, double theta, double phi)` | RXY rotation |
| `circuit_rxx(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RXX two-qubit rotation |
| `circuit_ryy(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RYY two-qubit rotation |
| `circuit_rzz(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RZZ two-qubit rotation |
| `circuit_rzx(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RZX two-qubit rotation |
| `circuit_crx(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRX gate |
| `circuit_cry(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRY gate |
| `circuit_crz(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRZ gate |
| `circuit_ccx(CircuitWrapper*, uint32_t ctrl1, uint32_t ctrl2, uint32_t target)` | Toffoli (CCX) gate |
| `circuit_measure(CircuitWrapper*, uint32_t qubit)` | Measure qubit |
| `circuit_reset(CircuitWrapper*, uint32_t qubit)` | Reset qubit |
| `circuit_fsim(CircuitWrapper*, uint32_t a, uint32_t b, double theta, double phi)` | fSim gate |

### Parameterized Gates

| Function | Description |
|----------|-------------|
| `circuit_rx(CircuitWrapper*, uint32_t qubit, double theta)` | RX gate (float) |
| `circuit_ry(CircuitWrapper*, uint32_t qubit, double theta)` | RY gate (float) |
| `circuit_rz(CircuitWrapper*, uint32_t qubit, double theta)` | RZ gate (float) |
| `circuit_rxy(CircuitWrapper*, uint32_t qubit, double theta, double phi)` | RXY rotation (float) |
| `circuit_rxx(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RXX rotation (float) |
| `circuit_ryy(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RYY rotation (float) |
| `circuit_rzz(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RZZ rotation (float) |
| `circuit_rzx(CircuitWrapper*, uint32_t a, uint32_t b, double theta)` | RZX rotation (float) |
| `circuit_crx(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRX gate (float) |
| `circuit_cry(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRY gate (float) |
| `circuit_crz(CircuitWrapper*, uint32_t ctrl, uint32_t target, double theta)` | CRZ gate (float) |
| `circuit_fsim(CircuitWrapper*, uint32_t a, uint32_t b, double theta, double phi)` | fSim gate (float) |
| `circuit_rx_param(CircuitWrapper*, uint32_t qubit, ParameterWrapper*)` | RX gate (symbolic) |
| `circuit_ry_param(CircuitWrapper*, uint32_t qubit, ParameterWrapper*)` | RY gate (symbolic) |
| `circuit_rz_param(CircuitWrapper*, uint32_t qubit, ParameterWrapper*)` | RZ gate (symbolic) |
| `circuit_rxy_param(CircuitWrapper*, uint32_t qubit, ParameterWrapper*, ParameterWrapper*)` | RXY (symbolic) |
| `circuit_rxx_param(CircuitWrapper*, uint32_t a, uint32_t b, ParameterWrapper*)` | RXX (symbolic) |
| `circuit_ryy_param(CircuitWrapper*, uint32_t a, uint32_t b, ParameterWrapper*)` | RYY (symbolic) |
| `circuit_rzz_param(CircuitWrapper*, uint32_t a, uint32_t b, ParameterWrapper*)` | RZZ (symbolic) |
| `circuit_rzx_param(CircuitWrapper*, uint32_t a, uint32_t b, ParameterWrapper*)` | RZX (symbolic) |
| `circuit_crx_param(CircuitWrapper*, uint32_t ctrl, uint32_t target, ParameterWrapper*)` | CRX (symbolic) |
| `circuit_cry_param(CircuitWrapper*, uint32_t ctrl, uint32_t target, ParameterWrapper*)` | CRY (symbolic) |
| `circuit_crz_param(CircuitWrapper*, uint32_t ctrl, uint32_t target, ParameterWrapper*)` | CRZ (symbolic) |
| `circuit_fsim_param(CircuitWrapper*, uint32_t a, uint32_t b, ParameterWrapper*, ParameterWrapper*)` | fSim (symbolic) |

### Parameter Module

| Function | Description |
|----------|-------------|
| `param_parse(const char* expr)` | Parse parameter expression |
| `param_free(ParameterWrapper* ptr)` | Free parameter memory |
| `param_evaluate(ParameterWrapper*, const char* bindings)` | Evaluate with bindings |
| `circuit_assign_params(CircuitWrapper*, const char* bindings)` | Assign params and return new circuit |

### IR Module (QCIS)

| Function | Description |
|----------|-------------|
| `qcis_load(const char* qcis_str)` | Parse QCIS string to circuit |
| `qcis_dumps(CircuitWrapper*)` | Dump circuit to QCIS string |

### IR Module (OpenQASM 2.0)

| Function | Description |
|----------|-------------|
| `qasm2_load(const char* qasm_str)` | Parse OpenQASM 2.0 string to circuit |
| `qasm2_dumps(CircuitWrapper*)` | Dump circuit to OpenQASM 2.0 string |

### Utilities

| Function | Description |
|----------|-------------|
| `cstring_free(char* ptr)` | Free C string returned by the library |

## Return Values

- Functions returning `int32_t`: `0` on success, negative on error
- Functions returning pointer: `NULL` on error
- Functions returning `double`: `0.0` on error
