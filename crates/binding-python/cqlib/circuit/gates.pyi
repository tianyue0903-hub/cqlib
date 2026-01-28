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

from typing import List, Optional, Union, Any, final


class Instruction:
    """
    The base class for all quantum instructions (gates, directives, etc).
    
    This class serves as a unified interface for operations in a quantum circuit.
    """

    @property
    def num_qubits(self) -> int:
        """The total number of qubits this instruction acts on."""
        ...

    @property
    def num_ctrl_qubits(self) -> int:
        """The number of control qubits involved in this instruction."""
        ...

    @property
    def num_params(self) -> int:
        """The number of floating-point parameters this instruction uses."""
        ...

    def matrix(self, params: Optional[List[float]] = None) -> List[List[complex]]:
        """
        Returns the unitary matrix representation of the instruction.

        Args:
            params (Optional[List[float]]): Parameters for the instruction (if any).

        Returns:
            List[List[complex]]: The unitary matrix as a nested list.
            
        Raises:
            ValueError: If the instruction is non-unitary or parameters are missing/incorrect.
        """
        ...

    def control(self, num_ctrls: int) -> "Instruction":
        """
        Returns a controlled version of this instruction.

        Args:
            num_ctrls (int): Number of control qubits to add.

        Returns:
            Instruction: The controlled instruction.
        """
        ...

    def inverse(self) -> "Instruction":
        """
        Returns the inverse instruction.

        Returns:
            Instruction: The inverse (Hermitian conjugate) instruction.
        """
        ...

    def __eq__(self, other: object) -> bool: ...

    def __hash__(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


@final
class StandardGate(Instruction):
    """
    Represents the set of standard quantum logic gates supported natively by Cqlib.
    
    Inherits from Instruction.
    """

    # --- Singleton Gates ---
    I: "StandardGate"
    H: "StandardGate"
    X: "StandardGate"
    Y: "StandardGate"
    Z: "StandardGate"
    S: "StandardGate"
    SDG: "StandardGate"
    T: "StandardGate"
    TDG: "StandardGate"

    RX: "StandardGate"
    RY: "StandardGate"
    RZ: "StandardGate"
    U: "StandardGate"
    Phase: "StandardGate"
    GPhase: "StandardGate"

    RXX: "StandardGate"
    RXY: "StandardGate"
    RYY: "StandardGate"
    RZX: "StandardGate"
    RZZ: "StandardGate"

    CX: "StandardGate"
    CY: "StandardGate"
    CZ: "StandardGate"
    CCX: "StandardGate"
    SWAP: "StandardGate"

    CRX: "StandardGate"
    CRY: "StandardGate"
    CRZ: "StandardGate"

    XY: "StandardGate"
    X2P: "StandardGate"
    X2M: "StandardGate"
    XY2P: "StandardGate"
    XY2M: "StandardGate"
    Y2P: "StandardGate"
    Y2M: "StandardGate"

    FSIM: "StandardGate"

    # StandardGate specific methods (if any) or overrides
    # Note: methods like matrix, control, inverse are inherited from Instruction 
    # but defined in Rust StandardGate too.

    def inverse(self) -> "StandardGate":
        """
        Returns the type of the inverse gate.
        For StandardGate, this returns the specific gate type (e.g., S -> SDG).
        """
        ...

# --- Module-level aliases for convenience ---

# Single Qubit
I: StandardGate
H: StandardGate
X: StandardGate
Y: StandardGate
Z: StandardGate
S: StandardGate
SDG: StandardGate
T: StandardGate
TDG: StandardGate

# Parametric
RX: StandardGate
RY: StandardGate
RZ: StandardGate
U: StandardGate
Phase: StandardGate
GPhase: StandardGate

# Two Qubit
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

# Multi-Controlled
CCX: StandardGate

# Controlled Rotation
CRX: StandardGate
CRY: StandardGate
CRZ: StandardGate

# Other
XY: StandardGate
X2P: StandardGate
X2M: StandardGate
XY2P: StandardGate
XY2M: StandardGate
Y2P: StandardGate
Y2M: StandardGate
