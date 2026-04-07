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

from cqlib.circuit import Qubit

class Topology:
    """
    A directed coupling graph representing quantum hardware connectivity.

    Each node represents a physical qubit, and each directed edge represents a
    coupling between qubits (e.g., for two-qubit gates like CNOT).

    # Directed vs Undirected

    This topology is **directed**. A coupling `a -> b` does not imply `b -> a`.
    This models hardware where gates only work in specific directions.

    # Example

    ```python
    from cqlib.device import Topology

    # Create topology with explicit qubit list and directed couplings
    topology = Topology(
        qubits=[0, 1, 2, 3],
        couplings=[(0, 1, "CX"), (1, 2, "CX"), (2, 3, "CX")]
    )

    # Query properties
    topology.num_qubits      # 4
    topology.num_couplings   # 3
    topology.qubits          # [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]

    # Check connectivity
    topology.is_connected(0, 1)  # True (0 -> 1)
    topology.is_connected(1, 0)  # False (no 1 -> 0 coupling)
    ```
    """

    def __init__(
        self,
        qubits: list[int] | list[Qubit],
        couplings: list[tuple[int | Qubit, int | Qubit, str]],
    ) -> None:
        """
        Creates a new topology with specified qubits and directed couplings.

        Note: The `qubits` list must be either all `int` or all `Qubit`.

        Args:
            qubits: List of qubit identifiers. Duplicate qubits are silently deduplicated.
            couplings: List of directed couplings as tuples `(control, target, name)`.
                Each coupling is directed from `control` to `target`.
                Couplings referencing non-existent qubits are silently ignored.
        """
        ...

    @staticmethod
    def line(qubits: list[int] | list[Qubit]) -> "Topology":
        """
        Creates a line topology with directed couplings between adjacent qubits.

        Constructs a linear chain where each qubit has a directed coupling to
        its next neighbor: `q[0] -> q[1] -> q[2] -> ...`

        Note: The `qubits` list must be either all `int` or all `Qubit`.

        Args:
            qubits: List of qubit identifiers in line order.

        Returns:
            Topology: A new line topology with `len(qubits) - 1` couplings.

        Raises:
            ValueError: If fewer than 2 qubits are provided.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of physical qubits in the topology."""
        ...

    @property
    def num_couplings(self) -> int:
        """Returns the number of directed coupling edges."""
        ...

    @property
    def qubits(self) -> list[Qubit]:
        """Returns all physical qubits in the topology."""
        ...

    def add_qubits(self, qubits: list[int] | list[Qubit]) -> None:
        """
        Adds physical qubits to the topology.

        Note: The `qubits` list must be either all `int` or all `Qubit`.

        Args:
            qubits: List of qubit identifiers to add.

        Raises:
            ValueError: If any qubit already exists in the topology.
        """
        ...

    def add_couplings(
        self, couplings: list[tuple[int | Qubit, int | Qubit, str]]
    ) -> None:
        """
        Adds directed couplings to the topology.

        Each coupling is directed from the first qubit to the second.

        Args:
            couplings: List of tuples `(control, target, name)` where:
                - `control`: Source qubit of the directed coupling
                - `target`: Destination qubit of the directed coupling
                - `name`: String identifier for the coupling (e.g., "CX", "CZ")

        Raises:
            ValueError: If either endpoint qubit does not exist in the topology.
        """
        ...

    def remove_qubits(self, qubits: list[int] | list[Qubit]) -> None:
        """
        Removes physical qubits and all their incident couplings from the topology.

        When a qubit is removed, all directed couplings where it is either the
        source or target are also removed.

        Note: The `qubits` list must be either all `int` or all `Qubit`.

        Args:
            qubits: List of qubit identifiers to remove.

        Raises:
            ValueError: If any qubit does not exist in the topology.
        """
        ...

    def remove_couplings(
        self, couplings: list[tuple[int | Qubit, int | Qubit]]
    ) -> None:
        """
        Removes directed couplings from the topology.

        Only removes the specific directed coupling. The reverse coupling
        (if present) is not affected.

        Args:
            couplings: List of `(control, target)` tuples specifying directed
                couplings to remove.

        Raises:
            ValueError: If a coupling does not exist or endpoint qubits are missing.
        """
        ...

    def is_connected(self, u: int | Qubit, v: int | Qubit) -> bool:
        """
        Checks for a directed coupling from `u` to `v`.

        Returns `True` if there is a directed coupling edge from qubit `u`
        to qubit `v`. This does **not** check the reverse direction.

        Args:
            u: Source qubit (control).
            v: Target qubit (target).

        Returns:
            bool: `True` if `u -> v` coupling exists, `False` otherwise.
        """
        ...

    def neighbors(self, qubit: int | Qubit) -> list[Qubit]:
        """
        Returns neighbors reachable via outgoing couplings from a qubit.

        Returns all qubits `v` such that a directed coupling `qubit -> v` exists.

        Args:
            qubit: The source qubit.

        Returns:
            list[Qubit]: List of qubits reachable via outgoing couplings.
        """
        ...

    def get_coupling_name(self, u: int | Qubit, v: int | Qubit) -> str | None:
        """
        Returns the name of the directed coupling from `u` to `v`, if it exists.

        Args:
            u: Source qubit.
            v: Target qubit.

        Returns:
            str | None: The coupling name if `u -> v` exists, `None` otherwise.
        """
        ...

    def contains_qubit(self, qubit: int | Qubit) -> bool:
        """
        Checks if a qubit exists in the topology.

        Args:
            qubit: Qubit to check.

        Returns:
            bool: `True` if the qubit exists in the topology.
        """
        ...

    def degree(self, qubit: int | Qubit) -> int:
        """
        Returns the out-degree of a qubit (number of outgoing couplings).

        Returns the count of directed couplings where this qubit is the source.

        Args:
            qubit: The qubit to query.

        Returns:
            int: Number of outgoing couplings. Returns 0 if qubit does not exist.
        """
        ...
