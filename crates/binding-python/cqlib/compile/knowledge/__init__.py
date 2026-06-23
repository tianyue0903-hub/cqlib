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

"""Compiler knowledge rules, validated rule libraries, and structural matching.

A knowledge rule describes an adjacent operation pattern, optional symbolic
parameter constraints, and the operations that replace the matched pattern.
Rule-local qubit labels are non-negative integers and must form a dense range
starting at zero. They are placeholders: label ``0`` may bind to any concrete
:class:`~cqlib.circuit.Qubit` during matching.

The module has three intended workflows:

* author rules with :class:`RuleItem`, :class:`Condition`, and :class:`Rule`;
* load, validate, classify, and query rules with :class:`RuleLibrary`;
* match rules against adjacent :class:`~cqlib.circuit.ValueOperation` objects
  and instantiate replacement operations.

Rules may also be stored in Cqlib's ``.rule`` DSL. The public ``load``/``loads``
functions parse directly into runtime :class:`Rule` objects. Parser tokens,
syntax-tree nodes, and lowering internals are intentionally not public.

Rule construction example::

    from cqlib.circuit import StandardGate
    from cqlib.compile.knowledge import Rule, RuleItem

    cancel_h = Rule(
        "cancel_h",
        operations=[
            RuleItem.standard(StandardGate.H, [0]),
            RuleItem.standard(StandardGate.H, [0]),
        ],
        target=[],
    )
    cancel_h.validate()
    assert cancel_h.verify().passed

DSL and matcher example::

    from cqlib.circuit import Qubit, StandardGate, ValueOperation
    from cqlib.compile.knowledge import loads, rule_matches_operations

    rule = loads('''
    rule cancel_x {
        match { X 0, X 0 }
        rewrite {}
    }
    ''')[0]

    operations = [
        ValueOperation.from_standard_gate(StandardGate.X, [Qubit(3)]),
        ValueOperation.from_standard_gate(StandardGate.X, [Qubit(3)]),
    ]
    bindings = rule_matches_operations(rule, operations)
    assert bindings is not None
    assert bindings.qubit(0) == Qubit(3)
"""

from .library import RuleId as RuleId
from .library import RuleKind as RuleKind
from .library import RuleLibrary as RuleLibrary
from .library import RuleMetadata as RuleMetadata
from .library import dump as dump
from .library import dumps as dumps
from .library import load as load
from .library import loads as loads
from .matcher import MatchBindings as MatchBindings
from .matcher import conditions_hold as conditions_hold
from .matcher import instantiate_target as instantiate_target
from .matcher import match_rule_item as match_rule_item
from .matcher import rule_matches_operations as rule_matches_operations
from .rule import Condition as Condition
from .rule import Rule as Rule
from .rule import RuleItem as RuleItem
from .rule import VerifyResult as VerifyResult

__all__ = [
    "RuleItem",
    "Condition",
    "Rule",
    "VerifyResult",
    "RuleId",
    "RuleKind",
    "RuleMetadata",
    "RuleLibrary",
    "MatchBindings",
    "loads",
    "load",
    "dumps",
    "dump",
    "match_rule_item",
    "conditions_hold",
    "instantiate_target",
    "rule_matches_operations",
]
