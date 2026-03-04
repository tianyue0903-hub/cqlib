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
class Delay:
    """A delay operation in a quantum circuit.

    Represents an idle period, often used for timing control in pulse-level scheduling.
    The delay unit is 0.5 nanoseconds (aligned with common quantum control hardware
    timing resolutions).
    """

    def __init__(self) -> None:
        """Creates a new delay operation."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
