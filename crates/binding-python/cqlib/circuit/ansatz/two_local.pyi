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
from ...qis import PauliString


@final
class EntanglementTopology:
    """Qubit connectivity topology for the entanglement layer of an ansatz.

    Use the factory methods to create instances:

    Examples:
        >>> t = EntanglementTopology.linear()
        >>> t = EntanglementTopology.circular()
        >>> t = EntanglementTopology.full()
        >>> t = EntanglementTopology.custom([(0, 1), (1, 2)])
    """

    @staticmethod
    def linear() -> "EntanglementTopology":
        """Linear nearest-neighbor topology: (0,1), (1,2), ..., (n-2, n-1)."""
        ...

    @staticmethod
    def circular() -> "EntanglementTopology":
        """Circular topology: linear + wrap-around edge (n-1, 0)."""
        ...

    @staticmethod
    def full() -> "EntanglementTopology":
        """Full all-to-all topology: every pair of qubits."""
        ...

    @staticmethod
    def custom(pairs: list[tuple[int, int]]) -> "EntanglementTopology":
        """Custom topology with explicit qubit pairs.

        Args:
            pairs: List of (control, target) qubit index pairs.
        """
        ...

    def generate_pairs(self, num_qubits: int) -> list[tuple[int, int]]:
        """Returns the list of qubit pairs for this topology.

        Args:
            num_qubits: Total number of qubits.

        Returns:
            List of (control, target) qubit index pairs.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


@final
class TwoLocal:
    """Hardware-efficient ansatz with alternating rotation and entanglement layers.

    Consists of:
      1. Rotation layers: single-qubit parameterized gates (e.g. RY, RZ).
      2. Entanglement layers: two-qubit gates (e.g. CX) per the topology.

    Pattern: [Rotation] → [Entanglement] → [Rotation] → ... → [Final Rotation]

    Builder methods return a **new** ``TwoLocal`` (immutable builder pattern).

    Examples:
        >>> from cqlib.circuit.ansatz import TwoLocal, EntanglementTopology
        >>> from cqlib import StandardGate
        >>> ansatz = (TwoLocal(3)
        ...     .reps(2)
        ...     .rotation_gates([StandardGate.RY, StandardGate.RZ])
        ...     .entanglement(EntanglementTopology.linear()))
        >>> circuit = ansatz.build_circuit("theta")
        >>> ansatz.num_parameters()
        18
    """

    def __init__(self, num_qubits: int) -> None:
        """Creates a new TwoLocal ansatz.

        Args:
            num_qubits: Number of qubits (must be ≥ 1).

        Defaults:
            - 1 repetition, RY rotation, CX entanglement, Linear topology.
        """
        ...

    def reps(self, n: int) -> "TwoLocal":
        """Sets the number of repetition layers.

        Args:
            n: Number of [Rotation + Entanglement] repetitions.

        Returns:
            A new TwoLocal with the updated setting.
        """
        ...

    def rotation_gates(self, gates: list[StandardGate]) -> "TwoLocal":
        """Sets the single-qubit rotation gates for each rotation layer.

        Args:
            gates: List of single-qubit parameterized gates.

        Returns:
            A new TwoLocal with the updated setting.
        """
        ...

    def entanglement_gate(self, gate: StandardGate) -> "TwoLocal":
        """Sets the two-qubit entanglement gate.

        Args:
            gate: A two-qubit gate (e.g. StandardGate.CX).

        Returns:
            A new TwoLocal with the updated setting.
        """
        ...

    def entanglement(self, topology: EntanglementTopology) -> "TwoLocal":
        """Sets the entanglement topology.

        Args:
            topology: An EntanglementTopology instance.

        Returns:
            A new TwoLocal with the updated setting.
        """
        ...

    def skip_final_rotation_layer(self, skip: bool) -> "TwoLocal":
        """Controls whether the final rotation layer is included.

        Args:
            skip: If True, omit the last rotation layer.

        Returns:
            A new TwoLocal with the updated setting.
        """
        ...

    def validate(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the parameterized quantum circuit.

        Parameters are named ``{prefix}_0``, ``{prefix}_1``, etc.

        Args:
            prefix: Prefix for parameter names (e.g. ``"theta"``).

        Returns:
            A Circuit with ``num_parameters()`` symbolic parameters.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Returns the total number of parameters."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
