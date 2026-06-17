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
from .two_local import EntanglementTopology

@final
class AngleEncoding:
    """Data encoding circuit using a single rotation gate per qubit.

    Each qubit i receives a rotation ``R(x_i)`` for the i-th input feature.
    This is the simplest encoding strategy.

    Examples:
        >>> from cqlib.circuit.ansatz import AngleEncoding
        >>> from cqlib import StandardGate
        >>> ae = AngleEncoding(4, StandardGate.RX)
        >>> circuit = ae.build_circuit("x")
        >>> ae.num_parameters()
        4
    """

    def __init__(self, num_qubits: int, rotation_gate: StandardGate) -> None:
        """Creates a new AngleEncoding feature map.

        Args:
            num_qubits: Number of qubits (= number of input features, ≥ 1).
            rotation_gate: Single-qubit rotation gate (RX, RY, or RZ).
        """
        ...

    def validate(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the encoding circuit.

        Parameters are named ``{prefix}_0`` ... ``{prefix}_{n-1}``.

        Args:
            prefix: Prefix for feature parameter names (e.g. ``"x"``).

        Returns:
            A Circuit with ``num_qubits`` symbolic parameters.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Returns the number of parameters (= num_qubits)."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> "AngleEncoding": ...
    def __deepcopy__(self, memo: dict) -> "AngleEncoding": ...

@final
class ZZFeatureMap:
    """Second-order Pauli-Z feature map for quantum kernel methods.

    Encodes classical data using:
      1. Hadamard on all qubits.
      2. ``RZ(2·x_i)`` on each qubit.
      3. ``exp(-i·2·(π-x_i)(π-x_j)·ZZ)`` for each entangled pair.

    Builder methods return a **new** ``ZZFeatureMap``.

    Examples:
        >>> from cqlib.circuit.ansatz import ZZFeatureMap, EntanglementTopology
        >>> fm = ZZFeatureMap(3).reps(2).entanglement(EntanglementTopology.full())
        >>> circuit = fm.build_circuit("x")
        >>> fm.num_parameters()
        3
    """

    def __init__(self, num_qubits: int) -> None:
        """Creates a new ZZFeatureMap.

        Args:
            num_qubits: Number of qubits (= number of input features, ≥ 1).

        Defaults:
            - 2 repetition layers, Full topology.
        """
        ...

    def reps(self, n: int) -> "ZZFeatureMap":
        """Sets the number of repetition layers.

        Args:
            n: Number of encoding repetitions.

        Returns:
            A new ZZFeatureMap with the updated setting.
        """
        ...

    def entanglement(self, topology: EntanglementTopology) -> "ZZFeatureMap":
        """Sets the entanglement topology.

        Args:
            topology: An EntanglementTopology instance.

        Returns:
            A new ZZFeatureMap with the updated setting.
        """
        ...

    def validate(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the encoding circuit.

        Args:
            prefix: Prefix for feature parameter names (e.g. ``"x"``).

        Returns:
            A Circuit with ``num_parameters()`` symbolic parameters.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Returns the number of parameters (= num_qubits)."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> "ZZFeatureMap": ...
    def __deepcopy__(self, memo: dict) -> "ZZFeatureMap": ...

@final
class PauliFeatureMap:
    """General Pauli evolution feature map for quantum machine learning.

    Supports arbitrary Pauli strings and entanglement topologies.

    For each repetition:
      1. Hadamard on all qubits.
      2. For each Pauli template P and each matching k-tuple of qubit indices:

         - k=1: ``exp(-i·x_i·P)`` (angle = ``2·x_i``)
         - k≥2: ``exp(-i·2·∏(π-x_j)·P)`` (angle = ``4·∏(π-x_j)``)

    Builder methods return a **new** ``PauliFeatureMap``.

    Examples:
        >>> from cqlib.circuit.ansatz import PauliFeatureMap, EntanglementTopology
        >>> from cqlib import PauliString
        >>> fm = (PauliFeatureMap(3)
        ...     .reps(2)
        ...     .paulis([PauliString.from_str("Z"), PauliString.from_str("ZZ")])
        ...     .entanglement(EntanglementTopology.full()))
        >>> circuit = fm.build_circuit("x")
        >>> fm.num_parameters()
        3
    """

    def __init__(self, num_qubits: int) -> None:
        """Creates a new PauliFeatureMap with default configuration.

        Args:
            num_qubits: Number of qubits (= number of input features, ≥ 1).

        Defaults:
            - 2 repetitions, Paulis=[Z, ZZ], Full topology, prefix="x".
        """
        ...

    def reps(self, n: int) -> "PauliFeatureMap":
        """Sets the number of repetition layers.

        Returns:
            A new PauliFeatureMap with the updated setting.
        """
        ...

    def paulis(self, paulis: list[PauliString]) -> "PauliFeatureMap":
        """Sets the Pauli string templates.

        The number of non-identity operators in each string determines locality k.

        Args:
            paulis: List of PauliString instances.

        Returns:
            A new PauliFeatureMap with the updated setting.
        """
        ...

    def entanglement(self, topology: EntanglementTopology) -> "PauliFeatureMap":
        """Sets the entanglement topology.

        Returns:
            A new PauliFeatureMap with the updated setting.
        """
        ...

    def parameter_prefix(self, prefix: str) -> "PauliFeatureMap":
        """Sets the parameter name prefix (default ``"x"``).

        Returns:
            A new PauliFeatureMap with the updated setting.
        """
        ...

    def validate(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the encoding circuit.

        Args:
            prefix: Prefix for feature parameter names.

        Returns:
            A Circuit with ``num_parameters()`` symbolic parameters.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Returns the number of parameters (= num_qubits)."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> "PauliFeatureMap": ...
    def __deepcopy__(self, memo: dict) -> "PauliFeatureMap": ...
