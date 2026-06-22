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

"""Conservative proofs for exchanging quantum operations.

The functions in this package prove whether two concrete
:class:`~cqlib.circuit.ValueOperation` objects may be swapped without changing
circuit semantics. A ``None`` result means that the configured proof sources
could not establish commutation; it is not proof that the operations fail to
commute.

Example::

    from cqlib.circuit import Qubit, StandardGate, ValueOperation
    from cqlib.compile.commutation import check_commutation

    lhs = ValueOperation.from_standard_gate(StandardGate.H, [Qubit(0)])
    rhs = ValueOperation.from_standard_gate(StandardGate.X, [Qubit(1)])

    proof = check_commutation(lhs, rhs)
    assert proof is not None and proof.is_exact()
"""

from .checker import Commutation as Commutation
from .checker import CommutationChecker as CommutationChecker
from .checker import CommutationConfig as CommutationConfig
from .checker import algebraic_commutation as algebraic_commutation
from .checker import check_commutation as check_commutation

__all__ = [
    "Commutation",
    "CommutationConfig",
    "CommutationChecker",
    "check_commutation",
    "algebraic_commutation",
]
