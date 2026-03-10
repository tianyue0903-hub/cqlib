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

class Layout:
    """
    Maps logical (virtual) qubits to physical qubits on a quantum device.

    A layout represents the current assignment of virtual qubits to physical hardware.
    It maintains bidirectional mappings and is used by routing algorithms to track
    qubit placement and update mappings when SWAP gates are inserted.

    # Ancilla Qubits

    When the number of logical qubits is less than physical qubits, ancilla qubits
    are automatically created to fill the gap. These are auxiliary qubits used
    during circuit execution.

    # Example

    ```python
    from cqlib.device import Layout

    # Create layout with initial mapping
    init_map = {0: 100, 1: 101}  # logical 0 -> physical 100
    layout = Layout(
        logical=[0, 1],
        physical=[100, 101, 102],
        init_map=init_map
    )

    # Check ancilla count
    print(layout.num_ancilla)  # 1 (physical qubit 102)

    # Get all mappings
    print(layout.v2p_map)  # {Qubit(0): Qubit(100), Qubit(1): Qubit(101), ...}
    ```
    """

    def __init__(
        self,
        logical: list[int] | list[Qubit],
        physical: list[int] | list[Qubit],
        init_map: dict[Qubit, Qubit] | None = None,
    ) -> None:
        """
        Creates a new layout mapping logical qubits to physical qubits.

        Note: The `logical` and `physical` lists must each be either all `int` or all `Qubit`.

        Args:
            logical: List of logical (virtual) qubit identifiers.
            physical: List of physical qubit identifiers available on the device.
            init_map: Optional initial mapping from logical to physical qubits.
                If not provided, logical qubits are mapped sequentially to physical qubits.

        Raises:
            ValueError: If the number of logical qubits exceeds physical qubits,
                `init_map` contains invalid virtual or physical qubits,
                or `init_map` maps multiple virtual qubits to the same physical qubit.
        """
        ...

    @property
    def num_logical(self) -> int:
        """Returns the number of logical qubits."""
        ...

    @property
    def num_ancilla(self) -> int:
        """
        Returns the number of ancilla qubits.

        Ancilla qubits are automatically generated to fill unused physical qubits
        when `len(logical) < len(physical)`.
        """
        ...

    @property
    def num_physical(self) -> int:
        """Returns the number of physical qubits."""
        ...

    def get_physical(self, virtual_id: int | Qubit) -> Qubit | None:
        """
        Returns the physical qubit mapped to a virtual qubit.

        Args:
            virtual_id: The logical qubit identifier.

        Returns:
            Qubit if the logical qubit is mapped, None otherwise.
        """
        ...

    def get_virtual(self, physical_id: int | Qubit) -> Qubit | None:
        """
        Returns the virtual qubit mapped to a physical qubit.

        Args:
            physical_id: The physical qubit identifier.

        Returns:
            Qubit if a virtual qubit is mapped to this physical qubit, None otherwise
            (e.g., if it's an unmapped ancilla).
        """
        ...

    @property
    def logical_qubits(self) -> list[Qubit]:
        """Returns all logical qubits in the layout."""
        ...

    @property
    def ancilla_qubits(self) -> list[Qubit]:
        """Returns all ancilla qubits in the layout."""
        ...

    @property
    def physical_qubits(self) -> list[Qubit]:
        """Returns all physical qubits in the layout."""
        ...

    @property
    def v2p_map(self) -> dict[Qubit, Qubit]:
        """
        Returns the virtual-to-physical qubit mapping.

        Returns a dictionary mapping each virtual qubit (including ancillas)
        to its assigned physical qubit.
        """
        ...

    @property
    def p2v_map(self) -> dict[Qubit, Qubit]:
        """
        Returns the physical-to-virtual qubit mapping.

        Returns a dictionary mapping each physical qubit to its assigned
        virtual qubit (if any). Unmapped physical qubits are not included.
        """
        ...

    def swap_physical(self, phys_a: int | Qubit, phys_b: int | Qubit) -> None:
        """
        Swaps the virtual qubits mapped to two physical qubits.

        This is the core operation used by routing algorithms (e.g., SABRE) when
        inserting SWAP gates. After a SWAP gate is applied on the hardware,
        the virtual qubits on those physical qubits are exchanged.

        Args:
            phys_a: First physical qubit.
            phys_b: Second physical qubit.

        Raises:
            ValueError: If either physical qubit is not in the layout.
        """
        ...
