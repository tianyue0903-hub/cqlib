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


@final
class Instruction:
    """
    A unified representation of any operation in a quantum circuit.
    
    This class wraps standard gates, extended gates, and directives.
    """

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
class StandardGate:
    """
    Represents the set of standard quantum logic gates supported natively by Cqlib.
    
    This class provides static access to singleton gate instances and methods to 
    query their properties.
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

    # --- Properties ---

    @property
    def num_qubits(self) -> int:
        """The total number of qubits this gate acts on."""
        ...

    @property
    def num_ctrl_qubits(self) -> int:
        """The number of control qubits defined for this gate."""
        ...

    @property
    def num_params(self) -> int:
        """The number of floating-point parameters this gate accepts."""
        ...

    # --- Methods ---

    def matrix(self, params: Optional[List[float]] = None) -> List[List[complex]]:
        """
        Returns the unitary matrix of the gate.

        Args:
            params (Optional[List[float]]): Parameters for the gate. Required for parametric gates.

        Returns:
            List[List[complex]]: The unitary matrix.
        """
        ...

    def control(self, num_ctrls: int) -> Instruction:
        """
        Returns a controlled version of this gate.

        Args:
            num_ctrls (int): Number of control qubits to add.

        Returns:
            Instruction: The controlled instruction.
        """
        ...

    def inverse(self) -> "StandardGate":
        """
        Returns the type of the inverse gate.

        Returns:
            StandardGate: The standard gate type that corresponds to the inverse operation.
        """
        ...

    def __eq__(self, other: object) -> bool: ...

    def __hash__(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...
