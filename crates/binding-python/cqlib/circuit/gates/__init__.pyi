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

from .standard import StandardGate
from .unitary import UnitaryGate
from .mc_gate import McGate
from .circuit_gate import CircuitGate

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
    "McGate",
    "CircuitGate",
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
