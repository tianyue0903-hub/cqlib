# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Numeric one- and two-qubit unitary synthesis primitives."""

from ...._native import compile as _compile_module

_unitary_module = _compile_module.transform.decompose.unitary

OneQubitUnitaryDecomposition = _unitary_module.OneQubitUnitaryDecomposition
TwoQubitUnitarySynthesisResult = _unitary_module.TwoQubitUnitarySynthesisResult
KakDecomposition = _unitary_module.KakDecomposition
synthesize_numeric_1q_unitary = _unitary_module.synthesize_numeric_1q_unitary
synthesize_numeric_2q_unitary = _unitary_module.synthesize_numeric_2q_unitary
kak_decompose = _unitary_module.kak_decompose

__all__ = [
    "OneQubitUnitaryDecomposition",
    "TwoQubitUnitarySynthesisResult",
    "KakDecomposition",
    "synthesize_numeric_1q_unitary",
    "synthesize_numeric_2q_unitary",
    "kak_decompose",
]
