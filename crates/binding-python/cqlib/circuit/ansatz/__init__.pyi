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

"""Type stubs for ``cqlib.circuit.ansatz``."""

from .facades import efficient_su2, pauli_feature_map, real_amplitudes, zz_feature_map
from .feature_map import AngleEncoding, PauliFeatureMap, ZZFeatureMap
from .qaoa import QAOAAnsatz
from .two_local import EntanglementTopology, TwoLocal

__all__ = [
    "EntanglementTopology",
    "TwoLocal",
    "AngleEncoding",
    "ZZFeatureMap",
    "PauliFeatureMap",
    "QAOAAnsatz",
    "real_amplitudes",
    "efficient_su2",
    "zz_feature_map",
    "pauli_feature_map",
]
