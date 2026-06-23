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

"""Validated rule libraries and ``.rule`` DSL serialization.

``RuleLibrary`` owns validated rules, assigns library-local stable IDs, and
precomputes indexes used by compiler passes. The ``load``/``loads`` functions
lower DSL input directly to runtime rules; lexer, parser, and AST internals are
not exposed through Python.

Example::

    from cqlib.compile.knowledge.library import RuleKind, RuleLibrary

    library = RuleLibrary.from_dsl(
        "rule cancel_x { match { X 0, X 0 } rewrite {} }",
        RuleKind.cancel(),
    )
    assert library.get_by_name("cancel_x") is not None
"""

from ..._native import compile as _compile_module

_knowledge_module = _compile_module.knowledge

RuleId = _knowledge_module.RuleId
RuleKind = _knowledge_module.RuleKind
RuleMetadata = _knowledge_module.RuleMetadata
RuleLibrary = _knowledge_module.RuleLibrary
loads = _knowledge_module.loads
load = _knowledge_module.load
dumps = _knowledge_module.dumps
dump = _knowledge_module.dump

__all__ = [
    "RuleId",
    "RuleKind",
    "RuleMetadata",
    "RuleLibrary",
    "loads",
    "load",
    "dumps",
    "dump",
]
