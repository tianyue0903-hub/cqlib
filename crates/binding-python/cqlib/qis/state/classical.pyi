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

from typing import Literal, Optional, final

from cqlib.circuit import ClassicalType, ClassicalValue, ClassicalVar
from cqlib.device import Outcome

@final
class RuntimeValue:
    """Typed runtime classical value produced during circuit execution."""

    @property
    def kind(self) -> Literal["bit", "bool", "uint", "bit_vec"]:
        """Runtime value kind."""
        ...

    @property
    def ty(self) -> ClassicalType:
        """Static classical type represented by this runtime value."""
        ...

    def to_bitstring(self) -> Optional[str]:
        """Returns a bit string for Bit and BitVec values, otherwise ``None``."""
        ...

    def as_bit(self) -> bool:
        """Returns this value as a bit.

        Raises:
            TypeError: If this value is not a Bit.
        """
        ...

    def as_bool(self) -> bool:
        """Returns this value as a logical boolean.

        Raises:
            TypeError: If this value is not a Bool.
        """
        ...

    def as_uint(self) -> int:
        """Returns this value as an unsigned integer.

        Raises:
            TypeError: If this value is not a UInt.
        """
        ...

    def as_bitvec_outcome(self) -> Outcome:
        """Returns this value as a bit-vector outcome.

        Raises:
            TypeError: If this value is not a BitVec.
        """
        ...

    def __copy__(self) -> RuntimeValue: ...
    def __deepcopy__(self, memo: dict) -> RuntimeValue: ...
    def __repr__(self) -> str: ...

@final
class ClassicalState:
    """Runtime classical state produced while executing a circuit."""

    def value(self, value: ClassicalValue) -> Optional[RuntimeValue]:
        """Returns the runtime value produced for an immutable circuit value."""
        ...

    def var(self, var: ClassicalVar) -> Optional[RuntimeValue]:
        """Returns the current runtime value of a mutable classical variable."""
        ...

    def __copy__(self) -> ClassicalState: ...
    def __deepcopy__(self, memo: dict) -> ClassicalState: ...
    def __repr__(self) -> str: ...
