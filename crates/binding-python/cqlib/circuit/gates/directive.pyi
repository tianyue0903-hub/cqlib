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

from typing_extensions import final

@final
class Directive:
    """A non-unitary operation in a quantum circuit.

    Directives represent operations that do not have a unitary matrix representation:
    - Barrier: Prevents gate reordering during optimization
    - Measure: Collapses qubit state to classical bit
    - Reset: Prepares qubit in |0> state
    """

    @staticmethod
    def barrier() -> "Directive":
        """Creates a barrier directive.

        A barrier prevents gate reordering across its boundary during optimization.
        Useful for timing-critical sequences or hardware constraints.

        Example:
            >>> barrier = Directive.barrier()
        """
        ...

    @staticmethod
    def measure() -> "Directive":
        """Creates a measure directive.

        A measurement operation that collapses qubit state to classical bit.
        Measures the qubit in the computational basis and stores the result (0 or 1).

        Example:
            >>> measure = Directive.measure()
        """
        ...

    @staticmethod
    def reset() -> "Directive":
        """Creates a reset directive.

        A reset operation that prepares qubit in |0> state.
        Forces the qubit into the ground state regardless of its current state.

        Example:
            >>> reset = Directive.reset()
        """
        ...

    def name(self) -> str:
        """Returns the name of the directive.

        Returns:
            A string: "Barrier", "Measure", or "Reset".
        """
        ...

    def is_barrier(self) -> bool:
        """Returns True if this is a barrier directive."""
        ...

    def is_measure(self) -> bool:
        """Returns True if this is a measure directive."""
        ...

    def is_reset(self) -> bool:
        """Returns True if this is a reset directive."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
