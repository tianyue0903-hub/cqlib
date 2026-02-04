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

from typing import List, Union, Optional
import numpy as np
import numpy.typing as npt
from typing_extensions import final
from ..parameter import Parameter


@final
class StandardGate:
    """Represents the set of standard quantum logic gates supported natively by Cqlib.

    Includes Pauli Gates, Clifford Gates, Parametric Rotations, Two-Qubit Gates,
    and Multi-Controlled Gates.
    """

    # Static factory attributes
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

    @property
    def num_qubits(self) -> int:
        """Returns the total number of qubits this gate acts on."""
        ...

    @property
    def num_ctrl_qubits(self) -> int:
        """Returns the number of control qubits defined for this gate."""
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of floating-point parameters this gate accepts."""
        ...

    @property
    def params(self) -> List[Parameter]:
        """Returns the parameters bound to this gate instance."""
        ...

    def __call__(self, *args: Union[float, Parameter]) -> "StandardGate":
        """Bind parameters to a parametric gate.

        Example:
            RX(0.5)  # Returns a StandardGate with bound parameter
        """
        ...

    def matrix(
            self, params: Optional[List[float]] = None
    ) -> npt.NDArray[np.complex128]:
        """Returns the unitary matrix representation of the gate.

        Args:
            params: Optional list of parameter values for parametric gates.
        """
        ...

    def control(self, num_ctrls: int) -> "StandardGate":
        """Returns a new gate with additional control qubits.

        Args:
            num_ctrls: Number of additional control qubits to add.
        """
        ...

    def inverse(self) -> "StandardGate":
        """Returns the inverse (Hermitian conjugate) of the gate."""
        ...
