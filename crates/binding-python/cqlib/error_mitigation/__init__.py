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

"""Error-mitigation tools.

The package is organized by mitigation workflow:

- :mod:`cqlib.error_mitigation.zne` contains zero-noise extrapolation helpers.
- :mod:`cqlib.error_mitigation.virtual_distillation` contains virtual
  distillation helpers.
- :mod:`cqlib.error_mitigation.unified` contains the sequential facade shared
  by supported mitigation methods.

The most commonly used classes are re-exported here for convenient imports.
"""

from .unified import (
    ErrorMitigation,
    ErrorMitigationError,
    Estimator,
    MitigatedResult,
    MitigationMethod,
    ProcessArgs,
    RunArgs,
)
from .virtual_distillation import VirtualDistillation, VirtualDistillationConfig
from .zne import ExtrapolateMethod, ZNEMitigation, ZneConfig

__all__ = [
    "Estimator",
    "ErrorMitigationError",
    "ExtrapolateMethod",
    "ZneConfig",
    "VirtualDistillationConfig",
    "MitigationMethod",
    "RunArgs",
    "ProcessArgs",
    "MitigatedResult",
    "ZNEMitigation",
    "VirtualDistillation",
    "ErrorMitigation",
]
