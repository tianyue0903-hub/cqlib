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

"""Circuit decomposition transforms and their configurations."""

from ...._native import compile as _compile_module
from . import mc_gate as mc_gate
from . import unitary as unitary

_decompose_module = _compile_module.transform.decompose

TwoQubitUnitaryDecomposeBasis = _decompose_module.TwoQubitUnitaryDecomposeBasis
UnitaryDecomposeConfig = _decompose_module.UnitaryDecomposeConfig
McGateDecomposeConfig = _decompose_module.McGateDecomposeConfig
DecompositionRuleStats = _decompose_module.DecompositionRuleStats
expand_definitions = _decompose_module.expand_definitions
decompose_unitaries = _decompose_module.decompose_unitaries
decompose_unitaries_with_rule_stats = (
    _decompose_module.decompose_unitaries_with_rule_stats
)
decompose_mc_gates = _decompose_module.decompose_mc_gates
decompose_mc_gates_with_rule_stats = (
    _decompose_module.decompose_mc_gates_with_rule_stats
)
decompose_mc_gates_for_device = _decompose_module.decompose_mc_gates_for_device

__all__ = [
    "mc_gate",
    "unitary",
    "TwoQubitUnitaryDecomposeBasis",
    "UnitaryDecomposeConfig",
    "McGateDecomposeConfig",
    "DecompositionRuleStats",
    "expand_definitions",
    "decompose_unitaries",
    "decompose_unitaries_with_rule_stats",
    "decompose_mc_gates",
    "decompose_mc_gates_with_rule_stats",
    "decompose_mc_gates_for_device",
]
