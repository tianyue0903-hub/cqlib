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

"""Public bridge to knowledge-based local circuit rewrite."""

from ..._native import compile as _compile_module

_transform_module = _compile_module.transform

RewriteMode = _transform_module.RewriteMode
RewriteConfig = _transform_module.RewriteConfig
KnowledgeRewriter = _transform_module.KnowledgeRewriter
KnowledgeRewriteStats = _transform_module.KnowledgeRewriteStats
KnowledgeRewriteResult = _transform_module.KnowledgeRewriteResult
rewrite_circuit = _transform_module.rewrite_circuit

__all__ = [
    "RewriteMode",
    "RewriteConfig",
    "KnowledgeRewriter",
    "KnowledgeRewriteStats",
    "KnowledgeRewriteResult",
    "rewrite_circuit",
]
