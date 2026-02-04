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

from typing import Union, List
import numpy as np
import numpy.typing as npt
from typing_extensions import final
from ..circuit import Circuit


@final
class UnitaryGate:
    """A definition for a custom Unitary gate.

    Acts as a blueprint for user-defined gates. It contains metadata (label, qubit count)
    and optionally the actual matrix representation.
    """

    def __init__(self, label: str, num_qubits: int) -> None:
        """Creates a new unitary gate definition without a matrix.

        Args:
            label: A name for the gate.
            num_qubits: The number of qubits the gate operates on.
        """
        ...

    def with_matrix(
            self, mat: Union[npt.NDArray[np.complex128], List[List[complex]]]
    ) -> "UnitaryGate":
        """Attaches a matrix to the unitary definition.

        Args:
            mat: A square matrix of size 2^N x 2^N.

        Returns:
            A new UnitaryGate with the matrix attached.

        Raises:
            ValueError: If the matrix dimensions do not match num_qubits.
        """
        ...

    def with_circuit(self, circuit: Circuit) -> "UnitaryGate":
        """Attaches a circuit to the unitary definition.
        
        Args:
            circuit: A quantum circuit definition.
            
        Returns:
            A new UnitaryGate with the circuit attached.
        """
        ...

    @property
    def label(self) -> str:
        """Returns the label of the gate."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits this gate acts on."""
        ...

    @property
    def matrix(self) -> npt.NDArray[np.complex128]:
        """Returns the matrix representation if available.

        Raises:
            ValueError: If no matrix has been defined for this gate.
        """
        ...

    def __array__(self) -> npt.NDArray[np.complex128]:
        """Returns the matrix as a numpy array (for numpy interoperability)."""
        ...
