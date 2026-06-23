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

"""Virtual distillation bindings.

Virtual distillation builds copy-swap circuits from a base circuit and combines
numerator and denominator estimates into a mitigated expectation value.
"""

from collections.abc import Callable

from cqlib.circuit import Circuit
from cqlib.qis import Hamiltonian

from .._native import error_mitigation as _error_mitigation_module

Estimator = Callable[[Circuit, Hamiltonian | None, int | None], tuple[float, float]]
VirtualDistillationConfig = _error_mitigation_module.VirtualDistillationConfig
VirtualDistillation = _error_mitigation_module.VirtualDistillation

__all__ = [
    "Estimator",
    "VirtualDistillationConfig",
    "VirtualDistillation",
]
