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

from typing import final

from ...circuit import Circuit
from ...circuit.gates import StandardGate

@final
class BasicEntanglerLayers:
    """Basic rotation + ring entanglement layer template."""

    def __init__(self, num_qubits: int) -> None:
        """Creates a new BasicEntanglerLayers template."""
        ...

    def reps(self, n: int) -> "BasicEntanglerLayers":
        """Sets the number of repetition layers."""
        ...

    def rotation_gate(self, gate: StandardGate) -> "BasicEntanglerLayers":
        """Sets the single-parameter rotation gate."""
        ...

    def entanglement_gate(self, gate: StandardGate) -> "BasicEntanglerLayers":
        """Sets the two-qubit entanglement gate."""
        ...

    def validate(self) -> None:
        """Validates the configuration."""
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the parameterized circuit."""
        ...

    def num_parameters(self) -> int:
        """Returns the number of symbolic parameters."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> "BasicEntanglerLayers": ...
    def __deepcopy__(self, memo: dict) -> "BasicEntanglerLayers": ...

@final
class StronglyEntanglingLayers:
    """U-rotation + range-based ring entanglement layer template."""

    def __init__(self, num_qubits: int) -> None:
        """Creates a new StronglyEntanglingLayers template."""
        ...

    def reps(self, n: int) -> "StronglyEntanglingLayers":
        """Sets the number of repetition layers."""
        ...

    def entanglement_gate(self, gate: StandardGate) -> "StronglyEntanglingLayers":
        """Sets the two-qubit entanglement gate."""
        ...

    def ranges(self, ranges: list[int]) -> "StronglyEntanglingLayers":
        """Sets explicit entanglement ranges reused cyclically by layer."""
        ...

    def validate(self) -> None:
        """Validates the configuration."""
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the parameterized circuit."""
        ...

    def num_parameters(self) -> int:
        """Returns the number of symbolic parameters."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> "StronglyEntanglingLayers": ...
    def __deepcopy__(self, memo: dict) -> "StronglyEntanglingLayers": ...
