# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Type hints for QCIS module.

This module provides type hints for parsing and serializing QCIS programs.
QCIS is Telecom Quantum's native quantum circuit format.

All cqlib standard gates except identity and global phase are supported. QCIS
``I Qn t`` represents delay rather than a standard identity gate.
"""

from ..circuit import Circuit

def loads(qcis: str) -> Circuit:
    """Parse a QCIS string into a Circuit.

    QCIS (Quantum Circuit Intermediate Representation) is Telecom Quantum's
    native quantum circuit format. Each line represents a gate operation:
    `GATE_NAME QUBIT_LIST [PARAMETER_LIST]`

    Args:
        qcis: A string containing QCIS code.

    Returns:
        A Circuit object representing the parsed quantum circuit.

    Raises:
        ValueError: If the QCIS string is invalid or cannot be parsed
            (e.g., invalid qubit format, unknown gate, etc.).

    Example:
        >>> qcis_code = '''
        ... H Q0
        ... CZ Q0 Q1
        ... RZ Q0 3.14159
        ... M Q0 Q1
        ... '''
        >>> circuit = loads(qcis_code)
    """
    ...

def load(path: str) -> Circuit:
    """Load and parse a QCIS file into a Circuit.

    Args:
        path: Path to the QCIS file.

    Returns:
        A Circuit object.

    Raises:
        ValueError: If parsing fails (syntax error, unknown gate, etc.).
        OSError: If file cannot be read.

    Example:
        >>> circuit = load("/path/to/circuit.qcis")
    """
    ...

def dumps(circuit: Circuit) -> str:
    """Serialize a Circuit to a QCIS string.

    All cqlib standard gates except identity and global phase can be serialized.
    Custom, unitary, multi-controlled, and control-flow instructions are not
    represented by QCIS.

    Supported Gates:
        - Single-qubit: H, RX, RXY, RY, RZ, S, SD, T, TD, U, X, XY,
          X2P, X2M, XY2P, XY2M, Y, Y2P, Y2M, Z, PHASE
        - Multi-qubit: RXX, RYY, RZX, RZZ, SWAP, CX, CCX, CY, CZ,
          CRX, CRY, CRZ, FSIM
        - Directives: B (Barrier), M (Measurement)
        - Delay: I Qn t, where t is a non-negative integer tick count

    Args:
        circuit: The Circuit object to serialize.

    Returns:
        A string containing the QCIS representation.

    Raises:
        ValueError: If the circuit contains gates not supported by QCIS
            (e.g., multi-controlled gates, custom unitary gates).

    Example:
        >>> from cqlib import Circuit, Qubit
        >>> circuit = Circuit(2)
        >>> circuit.h(Qubit(0))
        >>> circuit.cz(Qubit(0), Qubit(1))
        >>> qcis_str = dumps(circuit)
        >>> print(qcis_str)
        H Q0
        CZ Q0 Q1
    """
    ...

def dump(circuit: Circuit, path: str) -> None:
    """Serialize a Circuit to a QCIS file.

    Args:
        circuit: The Circuit object to serialize.
        path: Path to the output file.

    Raises:
        ValueError: If the circuit contains unsupported gates.
        OSError: If file cannot be written.

    Example:
        >>> from cqlib import Circuit, Qubit
        >>> circuit = Circuit(2)
        >>> circuit.h(Qubit(0))
        >>> circuit.cz(Qubit(0), Qubit(1))
        >>> dump(circuit, "output.qcis")
    """
    ...
