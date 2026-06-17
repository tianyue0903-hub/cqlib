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

"""Pauli operators and Pauli strings."""

from .._native import qis as _qis_module

Phase = _qis_module.Phase
Pauli = _qis_module.Pauli
PauliString = _qis_module.PauliString

__all__ = ["Phase", "Pauli", "PauliString"]
