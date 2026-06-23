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

"""Runtime rule-model bindings.

This module contains the immutable building blocks used to describe a compiler
rewrite: gate patterns, symbolic conditions, complete rules, and equivalence
verification results. Rule construction is deliberately separate from rule
library indexing and concrete-operation matching.

Example::

    from cqlib.circuit import StandardGate
    from cqlib.compile.knowledge.rule import Rule, RuleItem

    rule = Rule(
        "cancel_h",
        [RuleItem.standard(StandardGate.H, [0])] * 2,
        [],
    )
    rule.validate()
    assert rule.verify().passed
"""

from ..._native import compile as _compile_module

_knowledge_module = _compile_module.knowledge

RuleItem = _knowledge_module.RuleItem
Condition = _knowledge_module.Condition
Rule = _knowledge_module.Rule
VerifyResult = _knowledge_module.VerifyResult

__all__ = ["RuleItem", "Condition", "Rule", "VerifyResult"]
