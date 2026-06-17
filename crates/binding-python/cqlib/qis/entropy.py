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

"""Entropy and entanglement measures."""

from .._native import qis as _qis_module

linear_entropy = _qis_module.entropy.linear_entropy
renyi_entropy = _qis_module.entropy.renyi_entropy
entanglement_entropy_pure = _qis_module.entropy.entanglement_entropy_pure
negativity = _qis_module.entropy.negativity
concurrence = _qis_module.entropy.concurrence
entanglement_of_formation = _qis_module.entropy.entanglement_of_formation

__all__ = [
    "linear_entropy",
    "renyi_entropy",
    "entanglement_entropy_pure",
    "negativity",
    "concurrence",
    "entanglement_of_formation",
]
