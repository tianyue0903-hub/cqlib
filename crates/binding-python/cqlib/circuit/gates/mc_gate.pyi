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

from typing import Optional, Tuple, List
import numpy as np
import numpy.typing as npt
from typing_extensions import final
from ..parameter import Parameter
from .standard import StandardGate

@final
class McGate:
    """A multi-controlled quantum gate.
    
    Represents a gate with additional control qubits applied to a base gate.
    For example, McGate(2, StandardGate.X) represents a CCX (Toffoli) gate.
    """
    
    def __init__(self, num_controls: int, gate: StandardGate) -> None:
        """Create a multi-controlled gate.
        
        Args:
            num_controls: Number of additional control qubits.
            gate: The base gate to control.
        """
        ...
    
    def matrix(self, params: List[float]) -> npt.NDArray[np.complex128]:
        """Returns the unitary matrix representation of the gate.
        
        Args:
            params: A list of floating-point parameters for parametric gates.
            
        Returns:
            The unitary matrix as a numpy array.
        """
        ...
    
    def inverse(
        self, 
        params: Optional[List[Parameter]] = None
    ) -> Optional[Tuple["McGate", List[Parameter]]]:
        """Computes the inverse (Hermitian conjugate) of the gate.
        
        Args:
            params: A list of Parameter objects for parametric gates.
            
        Returns:
            A tuple of (inverse McGate, inverse parameters) or None if not invertible.
        """
        ...
    
    @property
    def num_ctrl_qubits(self) -> int:
        """Returns the number of control qubits."""
        ...
    
    @property
    def num_qubits(self) -> int:
        """Returns the total number of qubits (controls + targets)."""
        ...
    
    @property
    def num_params(self) -> int:
        """Returns the number of parameters required by the gate."""
        ...
    
    @property
    def base_gate(self) -> StandardGate:
        """Returns the base gate (without controls)."""
        ...
