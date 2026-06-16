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

"""Quantum bit (qubit) identifier.

Qubits are lightweight, hashable handles that identify positions in a quantum
register.  They do not carry circuit identity, physical-device location, or
state-vector position — those meanings are determined by the owning
:class:`~cqlib.circuit.Circuit`.

Gate methods accept both integer indices and ``Qubit`` objects
interchangeably::

    from cqlib import Circuit, Qubit

    c = Circuit(2)
    c.h(0)             # integer index
    c.cx(Qubit(0), 1)  # mixed Qubit + int
"""

class Qubit:
    """A lightweight handle representing a unique quantum bit (qubit).

    Equality compares only the numeric identifier.  Qubits are hashable and
    can be used as dictionary keys or in sets.

    Example::

        q0 = Qubit(0)
        q1 = Qubit(1)
        assert q0 < q1
        assert q0 == Qubit(0)
    """

    def __init__(self, index: int) -> None:
        """Create a qubit with the given non-negative index.

        Args:
            index: Qubit index in the quantum register (must fit in u32).

        Raises:
            QubitError: If the index is negative or exceeds u32 max.
        """
        ...
    @property
    def index(self) -> int:
        """The qubit index as a Python ``int``."""
        ...

    @property
    def id(self) -> int:
        """The raw u32 identifier (same numeric value as ``index``)."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: Qubit) -> bool: ...
    def __le__(self, other: Qubit) -> bool: ...
    def __gt__(self, other: Qubit) -> bool: ...
    def __ge__(self, other: Qubit) -> bool: ...
