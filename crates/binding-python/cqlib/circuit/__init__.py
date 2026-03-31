# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http:#www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from .._native import circuit as _circuit_module

from .gates import (
    StandardGate,
    UnitaryGate,
    CircuitGate,
    McGate,
    ConditionView,
    ControlFlow,
    IfElseGate,
    WhileLoopGate,
    Directive,
)

Parameter = _circuit_module.Parameter
Circuit = _circuit_module.Circuit
Qubit = _circuit_module.Qubit
Operation = _circuit_module.Operation
Instruction = _circuit_module.Instruction
circuit_to_matrix = _circuit_module.circuit_to_matrix

__all__ = [
    "Parameter",
    "Circuit",
    "Qubit",
    "Operation",
    "Instruction",
    "UnitaryGate",
    "StandardGate",
    "McGate",
    "CircuitGate",
    "ConditionView",
    "ControlFlow",
    "IfElseGate",
    "WhileLoopGate",
    "Directive",
    "Delay",
    "circuit_to_matrix",
]
