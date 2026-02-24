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


def loads(qcis: str) -> Circuit:
    """Load a quantum circuit from a QCIS string.

    Args:
        qcis: A string containing QCIS code.

    Returns:
        A Circuit object representing the parsed quantum circuit.

    Raises:
        ValueError: If the QCIS string is invalid or cannot be parsed.
    """
    ...


def load(path: str) -> Circuit:
    """Load a quantum circuit from a QCIS file.

    Args:
        path: Path to the QCIS file.

    Returns:
        A Circuit object.

    Raises:
        ValueError: If parsing fails.
        IOError: If file cannot be read.
    """
    ...


def dumps(circuit: Circuit) -> str:
    """Serialize a quantum circuit to a QCIS string.

    Args:
        circuit: The Circuit object to serialize.

    Returns:
        A string containing the QCIS representation.
    """
    ...


def dump(circuit: Circuit, path: str) -> None:
    """Serialize a quantum circuit to a QCIS file.

    Args:
        circuit: The Circuit object to serialize.
        path: Path to the output file.

    Raises:
        IOError: If file cannot be written.
    """
    ...
