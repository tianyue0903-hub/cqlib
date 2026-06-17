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

"""Quantum information metrics."""

from .._native import qis as _qis_module

purity_pure = _qis_module.metrics.purity_pure
purity_mixed = _qis_module.metrics.purity_mixed
state_fidelity_pure = _qis_module.metrics.state_fidelity_pure
trace_distance_pure = _qis_module.metrics.trace_distance_pure
state_fidelity_pure_mixed = _qis_module.metrics.state_fidelity_pure_mixed
entropy = _qis_module.metrics.entropy
trace_distance_mixed = _qis_module.metrics.trace_distance_mixed
state_fidelity_mixed = _qis_module.metrics.state_fidelity_mixed
partial_transpose = _qis_module.metrics.partial_transpose
logarithmic_negativity = _qis_module.metrics.logarithmic_negativity

__all__ = [
    "purity_pure",
    "purity_mixed",
    "state_fidelity_pure",
    "trace_distance_pure",
    "state_fidelity_pure_mixed",
    "entropy",
    "trace_distance_mixed",
    "state_fidelity_mixed",
    "partial_transpose",
    "logarithmic_negativity",
]
