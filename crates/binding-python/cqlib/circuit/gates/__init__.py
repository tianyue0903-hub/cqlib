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

from ..._native import StandardGate, UnitaryGate, McGate, CircuitGate

# --- Single Qubit Gates ---
I = StandardGate.I
H = StandardGate.H
X = StandardGate.X
Y = StandardGate.Y
Z = StandardGate.Z
S = StandardGate.S
SDG = StandardGate.SDG
T = StandardGate.T
TDG = StandardGate.TDG

# --- Parametric Rotation Gates ---
RX = StandardGate.RX
RY = StandardGate.RY
RZ = StandardGate.RZ
U = StandardGate.U
Phase = StandardGate.Phase
GPhase = StandardGate.GPhase

# --- Two Qubit Gates ---
CX = StandardGate.CX
CY = StandardGate.CY
CZ = StandardGate.CZ
SWAP = StandardGate.SWAP
RXX = StandardGate.RXX
RYY = StandardGate.RYY
RZZ = StandardGate.RZZ
RZX = StandardGate.RZX
RXY = StandardGate.RXY
FSIM = StandardGate.FSIM

# --- Multi-Controlled Gates ---
CCX = StandardGate.CCX

# --- Controlled Rotation Gates ---
CRX = StandardGate.CRX
CRY = StandardGate.CRY
CRZ = StandardGate.CRZ

# --- Other Gates ---
XY = StandardGate.XY
X2P = StandardGate.X2P
X2M = StandardGate.X2M
XY2P = StandardGate.XY2P
XY2M = StandardGate.XY2M
Y2P = StandardGate.Y2P
Y2M = StandardGate.Y2M

__all__ = [
    "StandardGate",
    "UnitaryGate",
    "McGate",
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
