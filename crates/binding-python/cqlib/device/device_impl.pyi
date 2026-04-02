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

from datetime import datetime
from cqlib.circuit import Instruction, Qubit
from .topology import Topology

class InstructionProp:
    """
    Represents calibration data for a quantum gate executed on specific qubits,
    including the gate's error rate (infidelity) and optionally its execution duration.

    # Example

    ```python
    from cqlib.device import InstructionProp
    from cqlib.circuit import StandardGate

    # Create properties for an H gate with 0.1% error rate
    prop = InstructionProp(StandardGate.H, error_rate=0.001)

    # Optionally set gate duration in nanoseconds
    prop.length = 35.0  # 35 ns
    ```
    """

    def __init__(self, instruction: Instruction, error_rate: float) -> None:
        """
        Creates calibration data for a given instruction.

        Args:
            instruction: The quantum instruction.
            error_rate: The error rate (infidelity) of the instruction.
        """
        ...

    @property
    def instruction(self) -> Instruction:
        """Returns the quantum instruction associated with these properties."""
        ...

    @instruction.setter
    def instruction(self, instruction: Instruction) -> None:
        """Sets the quantum instruction."""
        ...

    @property
    def error_rate(self) -> float:
        """Returns the error rate of the instruction."""
        ...

    @error_rate.setter
    def error_rate(self, error_rate: float) -> None:
        """Sets the error rate of the instruction."""
        ...

    @property
    def length(self) -> float | None:
        """Returns the duration of the instruction if configured, None otherwise."""
        ...

    @length.setter
    def length(self, length: float) -> None:
        """Sets the duration of the instruction (typically in nanoseconds)."""
        ...

class QubitProp:
    """
    Physical properties of individual qubits (T1, T2, readout errors).
    """

    def __init__(self, readout_error: float) -> None:
        """
        Creates qubit properties with a specified basic readout error.

        Args:
            readout_error: The base readout error rate.
        """
        ...

    @property
    def readout_error(self) -> float:
        """Returns the base readout error of the qubit."""
        ...

    @property
    def prob_meas0_prep1(self) -> float | None:
        """Returns the probability of measuring 0 given state was prepared in 1."""
        ...

    @prob_meas0_prep1.setter
    def prob_meas0_prep1(self, prob: float) -> None:
        """Sets the probability of measuring 0 given state was prepared in 1."""
        ...

    @property
    def prob_meas1_prep0(self) -> float | None:
        """Returns the probability of measuring 1 given state was prepared in 0."""
        ...

    @prob_meas1_prep0.setter
    def prob_meas1_prep0(self, prob: float) -> None:
        """Sets the probability of measuring 1 given state was prepared in 0."""
        ...

    @property
    def t1(self) -> float | None:
        """Returns the T1 relaxation time."""
        ...

    @t1.setter
    def t1(self, t1: float) -> None:
        """Sets the T1 relaxation time (typically in microseconds)."""
        ...

    @property
    def t2(self) -> float | None:
        """Returns the T2 dephasing time."""
        ...

    @t2.setter
    def t2(self, t2: float) -> None:
        """Sets the T2 dephasing time (typically in microseconds)."""
        ...

    @property
    def frequency(self) -> float | None:
        """Returns the qubit transition frequency."""
        ...

    @frequency.setter
    def frequency(self, frequency: float) -> None:
        """Sets the qubit transition frequency."""
        ...

    @property
    def native_instructions(self) -> list[InstructionProp]:
        """Returns the specific calibration properties for instructions on this qubit."""
        ...

    @native_instructions.setter
    def native_instructions(self, prop: InstructionProp) -> None:
        """Adds or updates a native instruction property on this qubit."""
        ...

class EdgeProp:
    """
    Properties of coupling edges between qubits.
    """

    def __init__(self) -> None:
        """Creates empty edge properties."""
        ...

    @property
    def native_instructions(self) -> list[InstructionProp]:
        """Returns the specific calibration properties for multi-qubit instructions on this edge."""
        ...

    @native_instructions.setter
    def native_instructions(self, prop: InstructionProp) -> None:
        """Adds or updates a native instruction property on this edge."""
        ...

class Device:
    """
    Complete hardware description including topology and calibration data.

    # Example

    ```python
    from cqlib.device import Device, Topology, QubitProp
    from datetime import datetime, timezone

    # Create device topology
    topology = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])

    # Initialize device
    device = Device("superconducting_qpu", [0, 1, 2], topology)

    # Set calibration timestamp
    device.calibration_time = datetime.now(timezone.utc)

    # Set default coherence times
    device.default_t1 = 100.0
    device.default_t2 = 50.0

    # Add qubit-specific properties
    prop = QubitProp(readout_error=0.01)
    prop.t1 = 120.0
    device.add_qubit_properties(0, prop)
    ```
    """

    def __init__(
        self, name: str, qubits: list[int] | list[Qubit], topology: Topology
    ) -> None:
        """
        Initializes a Device.

        Note: The `qubits` list must be either all `int` or all `Qubit`. Do not mix them.

        Args:
            name: The name of the quantum device.
            qubits: A list of available qubits.
            topology: The connectivity graph of the device.

        Raises:
            ValueError: If the initialization parameters are invalid.
        """
        ...

    @property
    def name(self) -> str:
        """Returns the name of the device."""
        ...

    @property
    def qubits(self) -> list[Qubit]:
        """Returns the list of physical qubits on the device."""
        ...

    @property
    def invalid_qubits(self) -> list[Qubit]:
        """Returns the list of invalid (broken or uncalibrated) qubits."""
        ...

    @invalid_qubits.setter
    def invalid_qubits(self, qubits: list[int] | list[Qubit]) -> None:
        """
        Sets the list of invalid qubits.
        Note: The `qubits` list must be either all `int` or all `Qubit`.
        """
        ...

    @property
    def topology(self) -> Topology:
        """Returns the device topology (connectivity)."""
        ...

    @property
    def native_gates(self) -> list[Instruction]:
        """Returns the natively supported gates on this device."""
        ...

    @native_gates.setter
    def native_gates(self, gates: list[Instruction]) -> None:
        """Sets the natively supported gates."""
        ...

    @property
    def calibration_time(self) -> datetime | None:
        """Returns the timestamp of the device's last calibration."""
        ...

    @calibration_time.setter
    def calibration_time(self, datetime_: datetime) -> None:
        """Sets the timestamp of the device's last calibration."""
        ...

    @property
    def default_t1(self) -> float | None:
        """Returns the default T1 relaxation time for qubits without specific values."""
        ...

    @default_t1.setter
    def default_t1(self, t1: float) -> None:
        """Sets the default T1 relaxation time."""
        ...

    @property
    def default_t2(self) -> float | None:
        """Returns the default T2 dephasing time for qubits without specific values."""
        ...

    @default_t2.setter
    def default_t2(self, t2: float) -> None:
        """Sets the default T2 dephasing time."""
        ...

    @property
    def default_readout_error(self) -> float | None:
        """Returns the default readout error rate for qubits without specific values."""
        ...

    @default_readout_error.setter
    def default_readout_error(self, error: float) -> None:
        """Sets the default readout error rate."""
        ...

    @property
    def default_single_qubit_error(self) -> float | None:
        """Returns the default error rate for single-qubit gates."""
        ...

    @default_single_qubit_error.setter
    def default_single_qubit_error(self, error: float) -> None:
        """Sets the default error rate for single-qubit gates."""
        ...

    @property
    def default_two_qubit_error(self) -> float | None:
        """Returns the default error rate for two-qubit gates."""
        ...

    @default_two_qubit_error.setter
    def default_two_qubit_error(self, error: float) -> None:
        """Sets the default error rate for two-qubit gates."""
        ...

    def add_qubit_properties(self, qubit: int | Qubit, props: QubitProp) -> None:
        """
        Adds or updates properties for a specific qubit.

        Args:
            qubit: The target qubit.
            props: The properties to assign.

        Raises:
            ValueError: If the operation fails.
        """
        ...

    def add_edge_properties(
        self, control: int | Qubit, target: int | Qubit, props: EdgeProp
    ) -> None:
        """
        Adds or updates properties for a specific coupling edge.

        Args:
            control: The source qubit of the coupling.
            target: The destination qubit of the coupling.
            props: The properties to assign.

        Raises:
            ValueError: If the operation fails.
        """
        ...

    def qubit_properties(self, qubit: int | Qubit) -> QubitProp | None:
        """
        Returns the properties of a specific qubit, if defined.

        Args:
            qubit: The qubit to query.
        """
        ...

    def edge_properties(
        self, control: int | Qubit, target: int | Qubit
    ) -> EdgeProp | None:
        """
        Returns the properties of a specific coupling edge, if defined.

        Args:
            control: Source qubit.
            target: Destination qubit.
        """
        ...

    def get_t1(self, qubit: int | Qubit) -> float | None:
        """Returns the T1 time for a specific qubit (falls back to default if not defined)."""
        ...

    def get_t2(self, qubit: int | Qubit) -> float | None:
        """Returns the T2 time for a specific qubit (falls back to default if not defined)."""
        ...

    def get_readout_error(self, qubit: int | Qubit) -> float | None:
        """Returns the readout error for a specific qubit (falls back to default if not defined)."""
        ...
