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

"""Public bridge to the native commutation checker implementation.

Configure a reusable checker when a compiler pass needs bounded proof costs::

    from cqlib.circuit import Qubit, StandardGate, ValueOperation
    from cqlib.compile.commutation import CommutationChecker, CommutationConfig

    config = CommutationConfig(
        enable_rule_oracle=False,
        enable_matrix_fallback=False,
        max_matrix_qubits=2,
    )
    checker = CommutationChecker.with_config(config)

    lhs = ValueOperation.from_standard_gate(StandardGate.RZ(0.2), [Qubit(0)])
    rhs = ValueOperation.from_standard_gate(StandardGate.RZ(0.7), [Qubit(0)])
    proof = checker.check(lhs, rhs)
    assert proof is not None and proof.is_exact()
"""

from ..._native import compile as _compile_module

_commutation_module = _compile_module.commutation

Commutation = _commutation_module.Commutation
CommutationConfig = _commutation_module.CommutationConfig
CommutationChecker = _commutation_module.CommutationChecker
check_commutation = _commutation_module.check_commutation
algebraic_commutation = _commutation_module.algebraic_commutation

__all__ = [
    "Commutation",
    "CommutationConfig",
    "CommutationChecker",
    "check_commutation",
    "algebraic_commutation",
]
