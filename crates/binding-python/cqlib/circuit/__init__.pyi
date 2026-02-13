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

from typing import Optional, List
import numpy as np
from .bit import Qubit
from .circuit import Circuit
from .parameter import Parameter
from .operation import Operation, Instruction
from .gates import StandardGate, UnitaryGate


def circuit_to_matrix(
        circuit: Circuit,
        qubits_order: Optional[List[int]] = None,
) -> np.ndarray:
    """Convert a circuit to its unitary matrix representation.

    This function computes the full unitary matrix corresponding to the
    quantum circuit's operations.

    Args:
        circuit: The quantum circuit to convert.
        qubits_order: Optional list specifying the order of qubits in the output matrix.
            If None, uses the natural qubit order (0, 1, 2, ...).

    Returns:
        A 2D numpy array representing the unitary matrix of the circuit.
        The dtype is complex128.

    Raises:
        ValueError: If the circuit contains unbound symbolic parameters or
            if qubits_order contains invalid qubit indices.

    Example:
        >>> circuit = Circuit(2)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> matrix = circuit_to_matrix(circuit)
        >>> matrix.shape
        (4, 4)
    """
    ...


__all__ = [
    "Qubit",
    "Circuit",
    "Parameter",
    "Operation",
    "Instruction",
    "StandardGate",
    "UnitaryGate",
    "circuit_to_matrix",
]
