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

from typing import Optional
from typing_extensions import final
import numpy as np
import numpy.typing as npt
from .bit import Qubit
from .gates.standard import StandardGate
from .gates.mc_gate import McGate
from .gates.unitary import UnitaryGate
from .gates.control_flow import ControlFlow
from .gates.directive import Directive
from .gates.delay import Delay

@final
class Instruction:
    """A unified representation of any operation in a quantum circuit.

    This class acts as a sum type for all possible instructions including:
    - Standard gates (e.g., H, CX)
    - Multi-controlled gates (MCGate)
    - Unitary gates
    - Circuit gates (sub-circuits)
    - Directives (Measure, Barrier, Reset)
    """

    @staticmethod
    def from_standard_gate(gate: StandardGate) -> "Instruction":
        """Creates an instruction from a standard gate.

        Args:
            gate: The standard gate.
        """
        ...

    @staticmethod
    def from_mc_gate(gate: McGate) -> "Instruction":
        """Creates an instruction from a multi-controlled gate.

        Args:
            gate: The multi-controlled gate.
        """
        ...

    @staticmethod
    def from_unitary_gate(gate: UnitaryGate) -> "Instruction":
        """Creates an instruction from a unitary gate.

        Args:
            gate: The unitary gate.
        """
        ...

    @staticmethod
    def from_directive(directive: Directive) -> "Instruction":
        """Creates a directive instruction (barrier, measure, reset).

        Args:
            directive: The directive.
        """
        ...

    @staticmethod
    def from_delay(delay: "Delay") -> "Instruction":
        """Creates a delay instruction.

        Args:
            delay: The delay operation.
        """
        ...

    @staticmethod
    def from_control_flow(control_flow: ControlFlow) -> "Instruction":
        """Creates a control flow instruction.

        Args:
            control_flow: The control flow operation.
        """
        ...

    @property
    def instruction_type(self) -> str:
        """Returns the type of instruction: 'standard', 'mcgate', 'unitary', 'circuit', or 'directive'."""
        ...

    @property
    def is_standard(self) -> bool:
        """Returns True if this is a standard gate instruction."""
        ...

    @property
    def is_mcgate(self) -> bool:
        """Returns True if this is a multi-controlled gate instruction."""
        ...

    @property
    def is_unitary(self) -> bool:
        """Returns True if this is a unitary gate instruction."""
        ...

    @property
    def is_circuit(self) -> bool:
        """Returns True if this is a circuit gate instruction."""
        ...

    @property
    def is_directive(self) -> bool:
        """Returns True if this is a directive (measure, barrier, reset)."""
        ...

    @property
    def is_delay(self) -> bool:
        """Returns True if this is a delay instruction."""
        ...

    @property
    def is_control_flow(self) -> bool:
        """Returns True if this is a control flow instruction (if_else, while_loop)."""
        ...

    @property
    def standard_gate(self) -> Optional[StandardGate]:
        """Returns the standard gate if this is a standard instruction, None otherwise."""
        ...

    @property
    def mc_gate(self) -> Optional[McGate]:
        """Returns the multi-controlled gate if this is an mc instruction, None otherwise."""
        ...

    @property
    def unitary_gate(self) -> Optional[UnitaryGate]:
        """Returns the unitary gate if this is a unitary instruction, None otherwise."""
        ...

    @property
    def directive(self) -> Optional[Directive]:
        """Returns the directive if this is a directive instruction, None otherwise."""
        ...

    @property
    def control_flow(self) -> Optional[ControlFlow]:
        """Returns the control flow if this is a control flow instruction, None otherwise."""
        ...

    @property
    def name(self) -> str:
        """Returns the name of the instruction."""
        ...

@final
class Operation:
    """A fully resolved operation in a quantum circuit.

    An Operation combines a gate (instruction) with the specific qubits it acts upon and its
    parameters. It serves as the fundamental node in the circuit's execution list.

    Attributes:
        instruction: The type of operation (gate or directive).
        qubits: The ordered list of qubits involved in this operation.
        params: The parameters for the operation (float values or symbolic indices).
        label: An optional human-readable label for this operation.
    """

    @property
    def instruction(self) -> Instruction:
        """Returns the instruction (gate type) of this operation."""
        ...

    @property
    def qubits(self) -> list[Qubit]:
        """Returns the qubits this operation acts on."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits this operation acts on."""
        ...

    @property
    def params(self) -> list[float | tuple[str, int]]:
        """Returns the parameters of this operation.

        Parameters can be either:
        - Fixed float values
        - Tuples ("param", index) for symbolic parameters
        """
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of parameters."""
        ...

    @property
    def label(self) -> Optional[str]:
        """Returns the label of this operation, if any."""
        ...

    @property
    def name(self) -> str:
        """Returns the name of the instruction."""
        ...

    def matrix(self) -> npt.NDArray[np.complex128]:
        """Returns the unitary matrix representation of this operation.

        Returns:
            A 2D numpy array (dtype=complex128) representing the unitary matrix.

        Raises:
            RuntimeError: If the operation is non-unitary (e.g., Measure, Barrier).
        """
        ...
