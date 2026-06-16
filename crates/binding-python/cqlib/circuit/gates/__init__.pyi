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

"""Gate definitions and factory constants.

This submodule provides the gate type classes and pre-built singleton
constants for all standard quantum gates.

Classes
-------
- :class:`StandardGate` — native gate instruction set
- :class:`UnitaryGate` — user-defined unitary gate
- :class:`MCGate` — multi-controlled standard gate
- :class:`CircuitGate` — composite gate from a frozen circuit
- :class:`Directive` — non-unitary operations (barrier, measure, reset)
- :class:`FrozenCircuit` — immutable circuit definition for gates

Pre-built gate constants are available as class attributes on
:class:`StandardGate` (e.g. ``StandardGate.H``, ``StandardGate.CX``).
"""

from .standard import StandardGate
from .unitary import UnitaryGate
from .mc_gate import MCGate
from .circuit_gate import CircuitGate, FrozenCircuit
from .directive import Directive

# --- Single Qubit Gates ---
I: StandardGate
H: StandardGate
X: StandardGate
Y: StandardGate
Z: StandardGate
S: StandardGate
SDG: StandardGate
T: StandardGate
TDG: StandardGate

# --- Parametric Rotation Gates ---
RX: StandardGate
RY: StandardGate
RZ: StandardGate
U: StandardGate
Phase: StandardGate
GPhase: StandardGate

# --- Two Qubit Gates ---
CX: StandardGate
CY: StandardGate
CZ: StandardGate
SWAP: StandardGate
RXX: StandardGate
RYY: StandardGate
RZZ: StandardGate
RZX: StandardGate
RXY: StandardGate
FSIM: StandardGate

# --- Multi-Controlled Gates ---
CCX: StandardGate

# --- Controlled Rotation Gates ---
CRX: StandardGate
CRY: StandardGate
CRZ: StandardGate

# --- Other Gates ---
XY: StandardGate
X2P: StandardGate
X2M: StandardGate
XY2P: StandardGate
XY2M: StandardGate
Y2P: StandardGate
Y2M: StandardGate

__all__ = [
    "StandardGate",
    "UnitaryGate",
    "MCGate",
    "CircuitGate",
    "FrozenCircuit",
    "Directive",
    # Single Qubit
    "I",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "SDG",
    "T",
    "TDG",
    # Parametric
    "RX",
    "RY",
    "RZ",
    "U",
    "Phase",
    "GPhase",
    # Two Qubit
    "CX",
    "CY",
    "CZ",
    "SWAP",
    "RXX",
    "RYY",
    "RZZ",
    "RZX",
    "RXY",
    "FSIM",
    # Multi-Controlled
    "CCX",
    # Controlled Rotation
    "CRX",
    "CRY",
    "CRZ",
    # Other
    "XY",
    "X2P",
    "X2M",
    "XY2P",
    "XY2M",
    "Y2P",
    "Y2M",
]
