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
from .bit import Qubit
from .parameter import Parameter
from .gates.standard import StandardGate
from .gates.unitary import UnitaryGate

class Circuit:
    """A quantum circuit representation serving as the core IR for quantum programs.

    The Circuit struct is designed to be a high-performance, memory-efficient container
    for quantum operations. It supports both static circuits (fixed angles) and
    parameterized circuits (symbolic angles).
    """

    def __init__(self, qubits: Union[int, List[int], List[Qubit]]) -> None:
        """Creates a new quantum circuit.

        Args:
            qubits: Number of qubits (int), list of indices (List[int]),
                   or list of Qubit objects.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the circuit."""
        ...

    @property
    def qubits(self) -> List[Qubit]:
        """Returns a list of all qubits in the circuit."""
        ...

    # --- Generic Instruction ---
    def append(self, instruction: StandardGate, qubits: List[int]) -> None:
        """Appends an instruction to the circuit."""
        ...

    # --- Single Qubit Gates ---
    def i(self, qubit: int) -> None:
        """Appends an Identity (I) gate."""
        ...

    def h(self, qubit: int) -> None:
        """Appends a Hadamard (H) gate."""
        ...

    def x(self, qubit: int) -> None:
        """Appends a Pauli-X (NOT) gate."""
        ...

    def y(self, qubit: int) -> None:
        """Appends a Pauli-Y gate."""
        ...

    def z(self, qubit: int) -> None:
        """Appends a Pauli-Z gate."""
        ...

    def s(self, qubit: int) -> None:
        """Appends an S (Phase) gate."""
        ...

    def sdg(self, qubit: int) -> None:
        """Appends an S-dagger (S†) gate."""
        ...

    def t(self, qubit: int) -> None:
        """Appends a T gate."""
        ...

    def tdg(self, qubit: int) -> None:
        """Appends a T-dagger (T†) gate."""
        ...

    def x2p(self, qubit: int) -> None:
        """Appends a √X (SX) gate."""
        ...

    def x2m(self, qubit: int) -> None:
        """Appends a √X† (SXdg) gate."""
        ...

    def y2p(self, qubit: int) -> None:
        """Appends a √Y gate."""
        ...

    def y2m(self, qubit: int) -> None:
        """Appends a √Y† gate."""
        ...

    # --- Parametric Single Qubit Gates ---
    def rx(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends a rotation around the X-axis by angle theta."""
        ...

    def ry(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends a rotation around the Y-axis by angle theta."""
        ...

    def rz(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends a rotation around the Z-axis by angle theta."""
        ...

    def phase(self, qubit: int, lambda_: Union[float, Parameter]) -> None:
        """Appends a Phase gate (P gate)."""
        ...

    def xy(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends an XY gate."""
        ...

    def xy2p(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends a √XY gate (positive phase)."""
        ...

    def xy2m(self, qubit: int, theta: Union[float, Parameter]) -> None:
        """Appends a √XY† gate (negative phase)."""
        ...

    def u(
        self,
        qubit: int,
        theta: Union[float, Parameter],
        phi: Union[float, Parameter],
        lambda_: Union[float, Parameter],
    ) -> None:
        """Appends a generic single-qubit rotation gate U(theta, phi, lambda)."""
        ...

    def rxy(
        self, qubit: int, theta: Union[float, Parameter], phi: Union[float, Parameter]
    ) -> None:
        """Appends a rotation in the XY plane."""
        ...

    # --- Two Qubit Gates ---
    def cx(self, control: int, target: int) -> None:
        """Appends a Controlled-NOT (CNOT) gate."""
        ...

    def cy(self, control: int, target: int) -> None:
        """Appends a Controlled-Y gate."""
        ...

    def cz(self, control: int, target: int) -> None:
        """Appends a Controlled-Z gate."""
        ...

    def swap(self, a: int, b: int) -> None:
        """Appends a SWAP gate."""
        ...

    def rxx(self, a: int, b: int, theta: Union[float, Parameter]) -> None:
        """Appends an Ising XX coupling gate."""
        ...

    def ryy(self, a: int, b: int, theta: Union[float, Parameter]) -> None:
        """Appends an Ising YY coupling gate."""
        ...

    def rzz(self, a: int, b: int, theta: Union[float, Parameter]) -> None:
        """Appends an Ising ZZ coupling gate."""
        ...

    def rzx(self, a: int, b: int, theta: Union[float, Parameter]) -> None:
        """Appends an Ising ZX coupling gate."""
        ...

    def fsim(
        self,
        a: int,
        b: int,
        theta: Union[float, Parameter],
        phi: Union[float, Parameter],
    ) -> None:
        """Appends a Fermionic Simulation gate (fSim)."""
        ...

    # --- Controlled Rotations ---
    def crx(self, control: int, target: int, theta: Union[float, Parameter]) -> None:
        """Appends a Controlled-RX gate."""
        ...

    def cry(self, control: int, target: int, theta: Union[float, Parameter]) -> None:
        """Appends a Controlled-RY gate."""
        ...

    def crz(self, control: int, target: int, theta: Union[float, Parameter]) -> None:
        """Appends a Controlled-RZ gate."""
        ...

    # --- Multi-Controlled ---
    def ccx(self, control1: int, control2: int, target: int) -> None:
        """Appends a Toffoli gate (CCX)."""
        ...

    def multi_control(
        self,
        instruction: StandardGate,
        controls: List[int],
        targets: List[int],
        params: Optional[List[Union[float, Parameter]]] = None,
    ) -> None:
        """Appends a multi-controlled version of a standard gate."""
        ...

    # --- Custom Unitary ---
    def unitary(self, gate: UnitaryGate, qubits: List[int]) -> None:
        """Appends a custom unitary gate to the circuit.

        Args:
            gate: The custom unitary gate definition.
            qubits: The list of qubit indices to apply the gate to.

        Raises:
            ValueError: If the number of qubits does not match the gate's num_qubits.
        """
        ...

    # --- Directives ---
    def measure(self, qubit: int) -> None:
        """Measures a qubit."""
        ...

    def reset(self, qubit: int) -> None:
        """Resets a qubit to the |0⟩ state."""
        ...

    def barrier(self, qubits: List[int]) -> None:
        """Inserts a barrier."""
        ...

    # --- Advanced ---
    def inverse(self) -> "Circuit":
        """Creates the inverse (adjoint) of the circuit."""
        ...
