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

"""Unified error-mitigation pipeline bindings.

The unified facade follows a sequential workflow: configure a mitigation
method, run the required circuits with a Python estimator, then post-process
the stored raw estimates into a final mitigated result.
"""

from collections.abc import Callable

from cqlib.circuit import Circuit
from cqlib.qis import Hamiltonian

from .._native import error_mitigation as _error_mitigation_module

Estimator = Callable[[Circuit, Hamiltonian | None, int | None], tuple[float, float]]
ErrorMitigationError = _error_mitigation_module.ErrorMitigationError
MitigationMethod = _error_mitigation_module.MitigationMethod
RunArgs = _error_mitigation_module.RunArgs
ProcessArgs = _error_mitigation_module.ProcessArgs
MitigatedResult = _error_mitigation_module.MitigatedResult
ErrorMitigation = _error_mitigation_module.ErrorMitigation

__all__ = [
    "Estimator",
    "ErrorMitigationError",
    "MitigationMethod",
    "RunArgs",
    "ProcessArgs",
    "MitigatedResult",
    "ErrorMitigation",
]
