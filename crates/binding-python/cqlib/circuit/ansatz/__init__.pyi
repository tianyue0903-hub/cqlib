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

"""Parameterized circuit ansatz templates."""

from .facades import (
    efficient_su2 as efficient_su2,
    pauli_feature_map as pauli_feature_map,
    real_amplitudes as real_amplitudes,
    zz_feature_map as zz_feature_map,
)
from .feature_map import (
    AngleEncoding as AngleEncoding,
    BasisEncoding as BasisEncoding,
    IQPFeatureMap as IQPFeatureMap,
    PauliFeatureMap as PauliFeatureMap,
    ZFeatureMap as ZFeatureMap,
    ZZFeatureMap as ZZFeatureMap,
)
from .hamiltonian_evolution import (
    EvolutionInfo as EvolutionInfo,
    EvolutionStrategy as EvolutionStrategy,
    PauliEvolutionAnsatz as PauliEvolutionAnsatz,
)
from .layers import (
    BasicEntanglerLayers as BasicEntanglerLayers,
    StronglyEntanglingLayers as StronglyEntanglingLayers,
)
from .qaoa import QAOAAnsatz as QAOAAnsatz
from .two_local import (
    EntanglementTopology as EntanglementTopology,
    TwoLocal as TwoLocal,
)

__all__ = [
    "EntanglementTopology",
    "TwoLocal",
    "AngleEncoding",
    "BasisEncoding",
    "ZFeatureMap",
    "IQPFeatureMap",
    "ZZFeatureMap",
    "PauliFeatureMap",
    "BasicEntanglerLayers",
    "StronglyEntanglingLayers",
    "QAOAAnsatz",
    "EvolutionStrategy",
    "EvolutionInfo",
    "PauliEvolutionAnsatz",
    "real_amplitudes",
    "efficient_su2",
    "zz_feature_map",
    "pauli_feature_map",
]
