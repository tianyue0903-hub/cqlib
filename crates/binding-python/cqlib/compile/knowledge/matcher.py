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

"""Structural matching and rewrite-target instantiation.

The matcher operates on adjacent self-contained ``ValueOperation`` objects.
It matches instruction identity, enforces one-to-one qubit bindings, binds
symbolic parameters, evaluates conditions, and instantiates replacements. It
does not search a circuit, commute intervening operations, compare costs, or
patch a circuit; those policies belong to compiler transforms.

Example::

    bindings = rule_matches_operations(rule, adjacent_operations)
    if bindings is not None:
        replacements = instantiate_target(rule.target, bindings)
"""

from ..._native import compile as _compile_module

_knowledge_module = _compile_module.knowledge

MatchBindings = _knowledge_module.MatchBindings
match_rule_item = _knowledge_module.match_rule_item
conditions_hold = _knowledge_module.conditions_hold
instantiate_target = _knowledge_module.instantiate_target
rule_matches_operations = _knowledge_module.rule_matches_operations

__all__ = [
    "MatchBindings",
    "match_rule_item",
    "conditions_hold",
    "instantiate_target",
    "rule_matches_operations",
]
