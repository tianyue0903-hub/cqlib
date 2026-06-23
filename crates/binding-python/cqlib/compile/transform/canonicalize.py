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

"""Public bridge to the native circuit canonicalizer."""

from ..._native import compile as _compile_module

_transform_module = _compile_module.transform

CanonicalizeConfig = _transform_module.CanonicalizeConfig
Canonicalizer = _transform_module.Canonicalizer
CanonicalizeResult = _transform_module.CanonicalizeResult
canonicalize_circuit = _transform_module.canonicalize_circuit

__all__ = [
    "CanonicalizeConfig",
    "Canonicalizer",
    "CanonicalizeResult",
    "canonicalize_circuit",
]
