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

from ..circuit import Circuit


def loads(qasm: str) -> Circuit:
    """Load a quantum circuit from a QASM 2.0 string.

    Args:
        qasm: A string containing QASM 2.0 code.

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
    """Load a quantum circuit from a QASM 2.0 file.

    Args:
        path: Path to the QASM file.

    Returns:
        A Circuit object.

    Raises:
        ValueError: If parsing fails.
        IOError: If file cannot be read.
    """
    ...


def dumps(circuit: Circuit) -> str:
    """Serialize a quantum circuit to a QASM 2.0 string.

    Args:
        circuit: The Circuit object to serialize.

    Returns:
        A string containing the QASM 2.0 representation.

    Example:
        >>> qasm_str = dumps(circuit)
        >>> print(qasm_str)
    """
    ...


def dump(circuit: Circuit, path: str) -> None:
    """Serialize a quantum circuit to a QASM 2.0 file.

    Args:
        circuit: The Circuit object to serialize.
        path: Path to the output file.

    Raises:
        IOError: If file cannot be written.
    """
    ...
