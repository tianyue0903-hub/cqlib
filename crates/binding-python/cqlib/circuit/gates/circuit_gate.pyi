# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing_extensions import final
from ..circuit import Circuit

@final
class CircuitGate:
    """A quantum gate defined by a quantum circuit.

    CircuitGate allows you to define custom gates by wrapping a Circuit object.
    The gate can have symbolic parameters that are mapped to the internal circuit's
    parameters when the gate is applied.
    """

    def __init__(self, name: str, circuit: Circuit) -> None:
        """Creates a new circuit-based gate.

        Args:
            name: A descriptive name for the gate.
            circuit: The circuit to use as the gate definition.
        """
        ...

    @property
    def name(self) -> str:
        """Returns the name of this circuit gate."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits this gate acts on."""
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of parameters this gate accepts."""
        ...

    def symbols(self) -> list[str]:
        """Returns the set of symbolic parameter names used in the circuit."""
        ...

    def inverse(self) -> "CircuitGate":
        """Computes the inverse of this circuit gate.

        Creates a new CircuitGate with the circuit inverted and appends "_dg" to the name.
        """
        ...
