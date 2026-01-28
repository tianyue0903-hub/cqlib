# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http:#www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

class Qubit:
    """
    A lightweight handle representing a unique quantum bit (qubit).
    """

    def __init__(self, index: int) -> None:
        """
        Creates a new Qubit handle.

        Args:
            index (int): The global unique index of the qubit.
        """
        ...

    @property
    def index(self) -> int:
        """
        Returns the identifier of the qubit.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: "Qubit") -> bool: ...
    def __le__(self, other: "Qubit") -> bool: ...
    def __gt__(self, other: "Qubit") -> bool: ...
    def __ge__(self, other: "Qubit") -> bool: ...
