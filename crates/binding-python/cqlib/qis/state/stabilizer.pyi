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

from typing import List, Optional, final

from cqlib.circuit import Circuit
from cqlib.device import Outcome
from cqlib.qis import PauliString

@final
class StabilizerCircuitResult:
    """Result of executing a Clifford circuit with a stabilizer simulator."""

    @property
    def state(self) -> "StabilizerState":
        """Final stabilizer state after circuit execution."""
        ...

    @property
    def measurements(self) -> List[Optional[bool]]:
        """Per-qubit last mid-circuit measurement result, or None if not measured."""
        ...

    def __repr__(self) -> str:
        """Returns a string representation of the circuit execution result."""
        ...

@final
class StabilizerState:
    """Stabilizer state simulator for Clifford circuits."""

    def __new__(cls, num_qubits: int) -> "StabilizerState":
        """Creates a new stabilizer state initialized to |0...0>."""
        ...

    @staticmethod
    def from_circuit(circuit: Circuit) -> "StabilizerState":
        """Creates a stabilizer state by simulating a Clifford circuit."""
        ...

    @staticmethod
    def apply_circuit(circuit: Circuit) -> StabilizerCircuitResult:
        """Executes a Clifford circuit and returns final state plus mid-circuit measurements."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the state."""
        ...

    def apply_h(self, qubit: int) -> None:
        """Applies a Hadamard gate."""
        ...

    def apply_s(self, qubit: int) -> None:
        """Applies an S gate."""
        ...

    def apply_sdg(self, qubit: int) -> None:
        """Applies an S-dagger gate."""
        ...

    def apply_x(self, qubit: int) -> None:
        """Applies an X gate."""
        ...

    def apply_y(self, qubit: int) -> None:
        """Applies a Y gate."""
        ...

    def apply_z(self, qubit: int) -> None:
        """Applies a Z gate."""
        ...

    def apply_x2p(self, qubit: int) -> None:
        """Applies an X/2 plus Clifford gate."""
        ...

    def apply_x2m(self, qubit: int) -> None:
        """Applies an X/2 minus Clifford gate."""
        ...

    def apply_y2p(self, qubit: int) -> None:
        """Applies a Y/2 plus Clifford gate."""
        ...

    def apply_y2m(self, qubit: int) -> None:
        """Applies a Y/2 minus Clifford gate."""
        ...

    def apply_cx(self, control: int, target: int) -> None:
        """Applies a controlled-X gate."""
        ...

    def apply_cy(self, control: int, target: int) -> None:
        """Applies a controlled-Y gate."""
        ...

    def apply_cz(self, q0: int, q1: int) -> None:
        """Applies a controlled-Z gate."""
        ...

    def apply_swap(self, q0: int, q1: int) -> None:
        """Applies a SWAP gate."""
        ...

    def measure(self, qubit: int) -> bool:
        """Measures one qubit and collapses the state."""
        ...

    def measure_all(self) -> Outcome:
        """Measures all qubits and collapses the state."""
        ...

    def reset(self, qubit: int) -> None:
        """Resets one qubit to |0>."""
        ...

    def probability_of(self, bits: List[bool]) -> float:
        """Returns the probability of a computational basis bitstring."""
        ...

    def probabilities(self) -> List[float]:
        """Returns the full computational-basis probability distribution."""
        ...

    def sample_shots(self, shots: int) -> List[Outcome]:
        """Samples measurement outcomes without mutating this state."""
        ...

    def get_stabilizers(self) -> List[PauliString]:
        """Returns the stabilizer generators."""
        ...

    def get_destabilizers(self) -> List[PauliString]:
        """Returns the destabilizer generators."""
        ...

    def pauli_expectation(self, pauli: PauliString) -> int:
        """Returns the expectation value of a Pauli string: -1, 0, or 1."""
        ...

    def to_stim_format(self) -> str:
        """Returns a Stim-like tableau representation."""
        ...

    def copy(self) -> "StabilizerState":
        """Returns a copy of this stabilizer state."""
        ...

    def __repr__(self) -> str:
        """Returns a string representation of the stabilizer state."""
        ...
