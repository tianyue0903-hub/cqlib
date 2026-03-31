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

"""Type hints for OpenQASM 2.0 module.

This module provides type hints for parsing and serializing OpenQASM 2.0 programs.
"""

from ..circuit import Circuit

def loads(qasm: str) -> Circuit:
    """Parse an OpenQASM 2.0 string into a Circuit.

    Args:
        qasm: A string containing OpenQASM 2.0 code.

    Returns:
        A Circuit object representing the parsed quantum circuit.

    Raises:
        ValueError: If the QASM string is invalid or cannot be parsed.

    Example:
        >>> qasm_code = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... '''
        >>> circuit = loads(qasm_code)
        >>> circuit.num_qubits
        2
    """
    ...

def load(path: str) -> Circuit:
    """Load and parse an OpenQASM 2.0 file into a Circuit.

    Args:
        path: Path to the QASM file.

    Returns:
        A Circuit object.

    Raises:
        ValueError: If parsing fails (syntax error, unknown gate, etc.).
        IOError: If file cannot be read.

    Example:
        >>> circuit = load("/path/to/circuit.qasm")
    """
    ...

def dumps(circuit: Circuit) -> str:
    """Serialize a Circuit to an OpenQASM 2.0 string.

    Args:
        circuit: The Circuit object to serialize.

    Returns:
        A string containing the OpenQASM 2.0 representation.

    Raises:
        ValueError: If the circuit contains gates that cannot be
            represented in OpenQASM 2.0.

    Example:
        >>> from cqlib import Circuit, Qubit
        >>> circuit = Circuit(2)
        >>> circuit.h(Qubit(0))
        >>> circuit.cx(Qubit(0), Qubit(1))
        >>> qasm_str = dumps(circuit)
        >>> print(qasm_str)
    """
    ...

def dump(circuit: Circuit, path: str) -> None:
    """Serialize a Circuit to an OpenQASM 2.0 file.

    Args:
        circuit: The Circuit object to serialize.
        path: Path to the output file.

    Raises:
        ValueError: If the circuit contains unsupported gates.
        IOError: If file cannot be written.

    Example:
        >>> from cqlib import Circuit, Qubit
        >>> circuit = Circuit(2)
        >>> circuit.h(Qubit(0))
        >>> circuit.cx(Qubit(0), Qubit(1))
        >>> dump(circuit, "output.qasm")
    """
    ...
