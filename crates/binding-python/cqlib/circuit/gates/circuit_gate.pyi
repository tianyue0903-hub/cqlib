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

from typing import Set
from typing_extensions import final
from ..circuit import Circuit

@final
class CircuitGate:
    """A quantum gate defined by a quantum circuit.
    
    CircuitGate allows you to define custom gates by wrapping a Circuit object.
    The gate can have symbolic parameters that are mapped to the internal circuit's
    parameters when the gate is applied.
    
    Example:
        # Create a circuit with a symbolic parameter
        circ = Circuit(1)
        theta = Parameter("theta")
        circ.rx(0, theta)
        
        # Create a CircuitGate from the circuit
        my_gate = CircuitGate("my_rx", circ.to_frozen())
    """
    
    def __init__(self, name: str, circuit: "Circuit") -> None:
        """Create a new circuit-defined gate.
        
        Args:
            name: A name for the gate.
            circuit: The frozen circuit that defines the gate's operation.
            
        Raises:
            ValueError: If the circuit is invalid for gate creation.
        """
        ...
    
    @property
    def name(self) -> str:
        """Returns the name of the gate."""
        ...
    
    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits this gate acts on."""
        ...
    
    @property
    def num_params(self) -> int:
        """Returns the number of parameters required by this gate."""
        ...
    
    def symbols(self) -> Set[str]:
        """Returns the set of symbolic parameter names used in this gate.
        
        Returns:
            A set of parameter name strings.
        """
        ...
    
    def circuit(self) -> "Circuit":
        """Returns the underlying circuit that defines this gate.
        
        Returns:
            The frozen circuit wrapped by this gate.
        """
        ...
    
    def inverse(self) -> "CircuitGate":
        """Returns the inverse (Hermitian conjugate) of this gate.
        
        Returns:
            A new CircuitGate representing the inverse operation.
            
        Raises:
            ValueError: If the circuit cannot be inverted.
        """
        ...
