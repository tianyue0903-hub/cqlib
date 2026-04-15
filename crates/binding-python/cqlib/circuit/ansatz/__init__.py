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

"""Quantum ansatz library for variational algorithms and quantum machine learning.

This package provides parameterized quantum circuit templates (ansatze):

- :class:`TwoLocal`: Hardware-efficient ansatz with alternating rotation and entanglement layers.
- :class:`AngleEncoding`: Simple feature map using one rotation gate per qubit.
- :class:`ZZFeatureMap`: Second-order Pauli-Z feature map for quantum kernel methods.
- :class:`PauliFeatureMap`: General Pauli evolution feature map.
- :class:`QAOAAnsatz`: Quantum Approximate Optimization Algorithm ansatz.
- :class:`EntanglementTopology`: Qubit connectivity topology descriptor.

Convenience constructors:

- :func:`real_amplitudes`: RealAmplitudes ansatz (RY + CX).
- :func:`efficient_su2`: EfficientSU2 ansatz (RY+RZ + CX).
- :func:`zz_feature_map`: ZZFeatureMap shortcut.
- :func:`pauli_feature_map`: PauliFeatureMap shortcut.
"""

from ..._native.circuit import ansatz as _ansatz

EntanglementTopology = _ansatz.EntanglementTopology
TwoLocal = _ansatz.TwoLocal
AngleEncoding = _ansatz.AngleEncoding
ZZFeatureMap = _ansatz.ZZFeatureMap
PauliFeatureMap = _ansatz.PauliFeatureMap
QAOAAnsatz = _ansatz.QAOAAnsatz
EvolutionStrategy = _ansatz.EvolutionStrategy
EvolutionInfo = _ansatz.EvolutionInfo
PauliEvolutionAnsatz = _ansatz.PauliEvolutionAnsatz

real_amplitudes = _ansatz.real_amplitudes
efficient_su2 = _ansatz.efficient_su2
zz_feature_map = _ansatz.zz_feature_map
pauli_feature_map = _ansatz.pauli_feature_map

__all__ = [
    "EntanglementTopology",
    "TwoLocal",
    "AngleEncoding",
    "ZZFeatureMap",
    "PauliFeatureMap",
    "QAOAAnsatz",
    "EvolutionStrategy",
    "EvolutionInfo",
    "PauliEvolutionAnsatz",
    "real_amplitudes",
    "efficient_su2",
    "zz_feature_map",
    "pauli_feature_map",
]
