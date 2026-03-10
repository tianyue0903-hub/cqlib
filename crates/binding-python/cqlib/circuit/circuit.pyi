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

from typing import Optional, Union, Tuple, List
import numpy as np
from .bit import Qubit
from .parameter import Parameter
from .operation import Operation
from .gates.standard import StandardGate
from .gates.unitary import UnitaryGate
from .gates.circuit_gate import CircuitGate
from .gates.mc_gate import McGate
from .gates.control_flow import ConditionView, ControlFlow
from .gates.directive import Directive

# Type alias for qubit list in control flow operations
# Supports list[int] or list[Qubit]
QubitList = List[Union[int, Qubit]]

# Type alias for parameters in control flow operations
ParamList = List[Union[float, Parameter]]

# Type alias for control flow operation tuples
# Format: (gate, qubits) or (gate, qubits, params)
# - gate: The gate to apply (StandardGate, McGate, UnitaryGate, Directive, or ControlFlow)
# - qubits: List of qubit indices (list[int]) or list of Qubit objects
# - params: Optional list of parameters (float or Parameter objects)
OpTuple2 = Tuple[Union[StandardGate, McGate, UnitaryGate, Directive, ControlFlow], QubitList]
OpTuple3 = Tuple[Union[StandardGate, McGate, UnitaryGate, Directive, ControlFlow], QubitList, ParamList]
OpTuple = Union[OpTuple2, OpTuple3]

class Circuit:
    """A quantum circuit representation serving as the core IR for quantum programs.

    The Circuit struct is designed to be a high-performance, memory-efficient container
    for quantum operations. It supports both static circuits (fixed angles) and
    parameterized circuits (symbolic angles).

    ## Flexible Qubit Arguments

    Most methods accept flexible qubit arguments:
    - Single qubit: `int` (qubit index) or `Qubit` object
    - Multiple qubits: `list[int]` or `list[Qubit]`

    Note: For list arguments, mixing types (e.g., `[0, Qubit(1)]`) is NOT supported.
    Use either all integers or all Qubit objects.

    Example:
        >>> circuit = Circuit(3)
        >>> circuit.h(0)  # int
        >>> circuit.h(Qubit(1))  # Qubit object
        >>> circuit.cx(0, Qubit(2))  # mixed for separate args
        >>> circuit.barrier([0, 1])  # list of ints
        >>> circuit.barrier([Qubit(0), Qubit(1)])  # list of Qubits
    """

    def __init__(self, qubits: int | list[int] | list[Qubit]) -> None:
        """Creates a new quantum circuit.

        Args:
            qubits: Number of qubits (int), list of indices (list[int]),
                   or list of Qubit objects.
                   Note: Mixed list is NOT supported.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the circuit."""
        ...

    @property
    def width(self) -> int:
        """Returns the width (number of qubits) of the circuit.

        This is an alias for `num_qubits`.
        """
        ...

    @property
    def qubits(self) -> list[Qubit]:
        """Returns a list of all qubits in the circuit."""
        ...

    @property
    def parameters(self) -> list[Parameter]:
        """Returns a list of all symbolic parameters used in the circuit."""
        ...

    @property
    def symbols(self) -> list[str]:
        """Returns a list of all symbolic variable names used in the circuit."""
        ...

    @property
    def global_phase(self) -> Parameter:
        """Returns the global phase of the circuit as a Parameter.

        The global phase represents a scalar factor e^(i*theta).
        While unobservable in isolated systems, it is critical for
        controlled operations and sub-circuit composition.
        """
        ...

    def set_global_phase(self, phase: float | Parameter) -> None:
        """Sets the global phase of the circuit.

        Args:
            phase: The phase value (can be float or Parameter).
        """
        ...

    @property
    def operations(self) -> "OperationIterator":
        """Returns an iterator over all operations in the circuit."""
        ...

    def circuit_gate(
        self,
        instruction: CircuitGate,
        qubits: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
    ) -> None:
        """Appends a circuit gate to the circuit.

        Args:
            instruction: The circuit gate to append.
            qubits: List of qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list (e.g., [0, Qubit(1)]) is NOT supported.
            params: Optional parameters for the circuit gate.
        """
        ...

    def multi_control_gate(
        self,
        instruction: McGate,
        qubits: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
    ) -> None:
        """Appends a multi-controlled gate (McGate) to the circuit.

        Args:
            instruction: The multi-controlled gate to append.
            qubits: List of qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list (e.g., [0, Qubit(1)]) is NOT supported.
            params: Optional parameters for the gate.
        """
        ...

    def to_gate(self, name: str) -> CircuitGate:
        """Converts the circuit to a gate.

        Args:
            name: A name for the new gate.

        Returns:
            A new CircuitGate object wrapping this circuit.
        """
        ...

    def i(self, qubit: int | Qubit) -> None:
        """Appends an Identity (I) gate."""
        ...

    def h(self, qubit: int | Qubit) -> None:
        """Appends a Hadamard (H) gate."""
        ...

    def x(self, qubit: int | Qubit) -> None:
        """Appends a Pauli-X (NOT) gate."""
        ...

    def y(self, qubit: int | Qubit) -> None:
        """Appends a Pauli-Y gate."""
        ...

    def z(self, qubit: int | Qubit) -> None:
        """Appends a Pauli-Z gate."""
        ...

    def s(self, qubit: int | Qubit) -> None:
        """Appends an S (Phase) gate."""
        ...

    def sdg(self, qubit: int | Qubit) -> None:
        """Appends an S-dagger (S†) gate."""
        ...

    def t(self, qubit: int | Qubit) -> None:
        """Appends a T gate."""
        ...

    def tdg(self, qubit: int | Qubit) -> None:
        """Appends a T-dagger (T†) gate."""
        ...

    def x2p(self, qubit: int | Qubit) -> None:
        """Appends a √X (SX) gate."""
        ...

    def x2m(self, qubit: int | Qubit) -> None:
        """Appends a √X† (SXdg) gate."""
        ...

    def y2p(self, qubit: int | Qubit) -> None:
        """Appends a √Y gate."""
        ...

    def y2m(self, qubit: int | Qubit) -> None:
        """Appends a √Y† gate."""
        ...

    def rx(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends a rotation around the X-axis by angle theta."""
        ...

    def ry(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends a rotation around the Y-axis by angle theta."""
        ...

    def rz(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends a rotation around the Z-axis by angle theta."""
        ...

    def phase(self, qubit: int | Qubit, lambda_: float | Parameter) -> None:
        """Appends a Phase gate (P gate)."""
        ...

    def xy(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends an XY gate."""
        ...

    def xy2p(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends a √XY gate (positive phase)."""
        ...

    def xy2m(self, qubit: int | Qubit, theta: float | Parameter) -> None:
        """Appends a √XY† gate (negative phase)."""
        ...

    def u(
        self,
        qubit: int | Qubit,
        theta: float | Parameter,
        phi: float | Parameter,
        lambda_: float | Parameter,
    ) -> None:
        """Appends a generic single-qubit rotation gate U(theta, phi, lambda)."""
        ...

    def rxy(
        self, qubit: int | Qubit, theta: float | Parameter, phi: float | Parameter
    ) -> None:
        """Appends a rotation in the XY plane."""
        ...

    def cx(self, control: int | Qubit, target: int | Qubit) -> None:
        """Appends a Controlled-NOT (CNOT) gate."""
        ...

    def cy(self, control: int | Qubit, target: int | Qubit) -> None:
        """Appends a Controlled-Y gate."""
        ...

    def cz(self, control: int | Qubit, target: int | Qubit) -> None:
        """Appends a Controlled-Z gate."""
        ...

    def swap(self, a: int | Qubit, b: int | Qubit) -> None:
        """Appends a SWAP gate."""
        ...

    def rxx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Appends an Ising XX coupling gate."""
        ...

    def ryy(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Appends an Ising YY coupling gate."""
        ...

    def rzz(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Appends an Ising ZZ coupling gate."""
        ...

    def rzx(self, a: int | Qubit, b: int | Qubit, theta: float | Parameter) -> None:
        """Appends an Ising ZX coupling gate."""
        ...

    def fsim(
        self,
        a: int | Qubit,
        b: int | Qubit,
        theta: float | Parameter,
        phi: float | Parameter,
    ) -> None:
        """Appends a Fermionic Simulation gate (fSim)."""
        ...

    def crx(
        self, control: int | Qubit, target: int | Qubit, theta: float | Parameter
    ) -> None:
        """Appends a Controlled-RX gate."""
        ...

    def cry(
        self, control: int | Qubit, target: int | Qubit, theta: float | Parameter
    ) -> None:
        """Appends a Controlled-RY gate."""
        ...

    def crz(
        self, control: int | Qubit, target: int | Qubit, theta: float | Parameter
    ) -> None:
        """Appends a Controlled-RZ gate."""
        ...

    def ccx(
        self, control1: int | Qubit, control2: int | Qubit, target: int | Qubit
    ) -> None:
        """Appends a Toffoli gate (CCX)."""
        ...

    def multi_control(
        self,
        instruction: StandardGate,
        controls: list[int] | list[Qubit],
        targets: list[int] | list[Qubit],
        params: Optional[list[float | Parameter]] = None,
    ) -> None:
        """Appends a multi-controlled version of a standard gate.

        Args:
            instruction: The standard gate to apply.
            controls: List of control qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list is NOT supported.
            targets: List of target qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list is NOT supported.
            params: Optional parameters for the gate.
        """
        ...

    def unitary(self, gate: UnitaryGate, qubits: list[int] | list[Qubit]) -> None:
        """Appends a custom unitary gate to the circuit.

        Args:
            gate: The custom unitary gate definition.
            qubits: List of qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list (e.g., [0, Qubit(1)]) is NOT supported.

        Raises:
            ValueError: If the number of qubits does not match the gate's num_qubits.

        Example:
            >>> circuit = Circuit(2)
            >>> gate = UnitaryGate("my_gate", 2).with_matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
            >>> circuit.unitary(gate, [0, 1])  # Using list of ints
            >>> circuit.unitary(gate, [Qubit(0), Qubit(1)])  # Using list of Qubits
        """
        ...

    def measure(self, qubit: int | Qubit) -> None:
        """Measures a qubit."""
        ...

    def reset(self, qubit: int | Qubit) -> None:
        """Resets a qubit to the |0⟩ state."""
        ...

    def barrier(self, qubits: list[int] | list[Qubit]) -> None:
        """Inserts a barrier.

        Args:
            qubits: List of qubit indices (list[int]) or list of Qubit objects.
                Note: Mixed list (e.g., [0, Qubit(1)]) is NOT supported.

        Example:
            >>> circuit = Circuit(3)
            >>> circuit.barrier([0, 1])  # Using list of ints
            >>> circuit.barrier([Qubit(0), Qubit(1), Qubit(2)])  # Using list of Qubits
        """
        ...

    def delay(self, qubit: int | Qubit, param: float | Parameter) -> None:
        """Applies a Delay instruction to the specified qubit.

        Args:
            qubit: Qubit index (int) or Qubit object.
            param: The duration of the delay (can be float or Parameter).
        """
        ...

    def inverse(self) -> "Circuit":
        """Creates the inverse (adjoint) of the circuit."""
        ...

    def decompose(self) -> "Circuit":
        """Decomposes the circuit into a new circuit with simpler operations."""
        ...

    def assign_parameters(
        self, bindings: Optional[dict[str, float]] = None
    ) -> "Circuit":
        """Assign values to symbolic parameters and return a new circuit.

        Args:
            bindings: A dictionary mapping parameter names to float values.
                If None, all symbolic parameters remain symbolic.

        Returns:
            A new Circuit with parameters assigned. The original circuit is not modified.

        Example:
            >>> circuit = Circuit(1)
            >>> theta = Parameter("theta")
            >>> circuit.rx(0, theta)
            >>> assigned = circuit.assign_parameters({"theta": 3.14159})
            >>> # assigned is a new circuit with theta = 3.14159
        """
        ...

    def to_matrix(self, qubits_order: Optional[list[int]] = None) -> np.ndarray:
        """Convert the circuit to its unitary matrix representation.

        Args:
            qubits_order: Optional list specifying the order of qubits in the output matrix.
                If None, uses the natural qubit order (0, 1, 2, ...).

        Returns:
            A 2D numpy array representing the unitary matrix of the circuit.

        Raises:
            ValueError: If the circuit contains unbound symbolic parameters or
                if qubits_order contains invalid qubit indices.

        Example:
            >>> circuit = Circuit(1)
            >>> circuit.h(0)
            >>> matrix = circuit.to_matrix()
            >>> matrix.shape
            (2, 2)
        """
        ...

    def __len__(self) -> int:
        """Returns the number of operations in the circuit."""
        ...

    def __getitem__(self, idx: int | slice) -> Operation | list[Operation]:
        """Accesses operations by index (supports negative indexing and slicing).

        Args:
            idx: Integer index or slice.

        Returns:
            Operation at the given index, or a list of operations for a slice.

        Raises:
            IndexError: If the index is out of range.
        """
        ...

    def add_qubits(self, qubits: list[int]) -> None:
        """Adds additional qubits to the circuit.

        Args:
            qubits: List of qubit indices to add.

        Example:
            >>> circuit = Circuit(2)  # Qubits 0, 1
            >>> circuit.add_qubits([2, 3])  # Now has qubits 0, 1, 2, 3
        """
        ...

    def if_else(
        self,
        condition: ConditionView,
        true_body: list[OpTuple],
        false_body: Optional[list[OpTuple]] = None,
    ) -> None:
        """Appends a conditional (if-else) operation to the circuit.

        Executes different quantum operations based on a classical condition
        (typically from a previous measurement).

        Args:
            condition: The classical condition to evaluate.
            true_body: List of operation tuples for the true branch.
                Each tuple can be either:
                - (gate, qubits): 2-tuple without parameters
                - (gate, qubits, params): 3-tuple with parameters

                Where:
                - gate: The gate to apply (StandardGate, McGate, UnitaryGate, Directive, or ControlFlow)
                - qubits: List of qubit indices (list[int]) or list of Qubit objects
                - params: List of parameters (float or Parameter objects)
            false_body: Optional list of operation tuples for the false branch.
                Same format as true_body.

        Example:
            >>> from cqlib import Circuit, StandardGate
            >>> from cqlib.circuit import ConditionView, Qubit
            >>> circuit = Circuit(2)
            >>> circuit.x(0)
            >>> circuit.measure(0)
            >>> condition = ConditionView(Qubit(0), 1)
            >>> # If qubit 0 is 1, apply X to qubit 1; otherwise apply Z
            >>> circuit.if_else(
            ...     condition,
            ...     [(StandardGate.X, [1])],           # true body: 2-tuple
            ...     [(StandardGate.Z, [1])]            # false body: 2-tuple
            ... )
            >>>
            >>> # With parameters (3-tuple)
            >>> import numpy as np
            >>> theta = Parameter("theta")
            >>> circuit.if_else(
            ...     condition,
            ...     [(StandardGate.RX, [1], [np.pi/2])],     # true body: 3-tuple with fixed param
            ...     [(StandardGate.RY, [1], [theta])]        # false body: 3-tuple with symbolic param
            ... )
        """
        ...

    def while_loop(
        self,
        condition: ConditionView,
        body: list[OpTuple],
    ) -> None:
        """Appends a while loop operation to the circuit.

        Executes quantum operations repeatedly while a condition is true.

        Args:
            condition: The classical condition to evaluate.
            body: List of operation tuples for the loop body.
                Each tuple can be either:
                - (gate, qubits): 2-tuple without parameters
                - (gate, qubits, params): 3-tuple with parameters

                Where:
                - gate: The gate to apply (StandardGate, McGate, UnitaryGate, Directive, or ControlFlow)
                - qubits: List of qubit indices (list[int]) or list of Qubit objects
                - params: List of parameters (float or Parameter objects)

        Example:
            >>> from cqlib import Circuit, StandardGate
            >>> from cqlib.circuit import ConditionView, Qubit
            >>> circuit = Circuit(2)
            >>> circuit.x(0)
            >>> circuit.measure(0)
            >>> condition = ConditionView(Qubit(0), 1)
            >>> # While qubit 0 equals 1, apply H to qubit 1 (2-tuple format)
            >>> circuit.while_loop(condition, [(StandardGate.H, [1])])
            >>>
            >>> # Multiple operations with parameters (3-tuple format)
            >>> import numpy as np
            >>> theta = Parameter("theta")
            >>> circuit.while_loop(
            ...     condition,
            ...     [
            ...         (StandardGate.RX, [1], [np.pi/4]),   # 3-tuple with fixed param
            ...         (StandardGate.RY, [1], [theta]),     # 3-tuple with symbolic param
            ...         (StandardGate.H, [1]),               # 2-tuple without params
            ...     ]
            ... )
        """
        ...

class OperationIterator:
    """An iterator over the operations in a quantum circuit."""

    def __iter__(self) -> "OperationIterator": ...
    def __next__(self) -> Operation: ...
    def __len__(self) -> int:
        """Returns the number of remaining operations."""
        ...
