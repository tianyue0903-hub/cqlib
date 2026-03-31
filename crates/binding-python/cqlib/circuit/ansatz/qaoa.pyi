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

from typing import final

from ...circuit import Circuit
from ...qis import Hamiltonian


@final
class QAOAAnsatz:
    """Quantum Approximate Optimization Algorithm (QAOA) ansatz.

    Alternates between cost and mixer Hamiltonian evolutions:

    .. math::

        U(\\beta, \\gamma) = \\prod_{l=1}^{p}
            e^{-i \\beta_l H_M} \\cdot e^{-i \\gamma_l H_C}

    where :math:`H_C` is the cost Hamiltonian and :math:`H_M` is the mixer
    (default: :math:`\\sum_i X_i`).

    Builder methods return a **new** ``QAOAAnsatz``.

    Examples:
        >>> from cqlib.circuit.ansatz import QAOAAnsatz
        >>> from cqlib import Hamiltonian, PauliString
        >>> h_c = Hamiltonian(2)
        >>> h_c.add_term(PauliString("ZZ"), 0.5)
        >>> ansatz = QAOAAnsatz(h_c).reps(3)
        >>> circuit = ansatz.build_circuit("p")
        >>> ansatz.num_parameters()
        6
    """

    def __init__(self, cost_operator: Hamiltonian) -> None:
        """Creates a new QAOAAnsatz.

        The default mixer is :math:`H_M = \\sum_i X_i` (X on each qubit).

        Args:
            cost_operator: The cost Hamiltonian :math:`H_C` (must have ≥ 1 qubit).

        Raises:
            ValueError: If the cost operator is invalid.
        """
        ...

    def reps(self, n: int) -> "QAOAAnsatz":
        """Sets the number of QAOA layers p.

        Total parameters = 2 * reps (one gamma and one beta per layer).

        Args:
            n: Number of QAOA layers p ≥ 1.

        Returns:
            A new QAOAAnsatz with the updated setting.
        """
        ...

    def mixer(self, mixer_operator: Hamiltonian) -> "QAOAAnsatz":
        """Overrides the default X-mixer with a custom mixer Hamiltonian.

        Args:
            mixer_operator: A Hamiltonian acting on the same number of qubits.

        Returns:
            A new QAOAAnsatz with the updated mixer.

        Raises:
            ValueError: If the mixer has a different qubit count.
        """
        ...

    def initial_state(self, circuit: Circuit) -> "QAOAAnsatz":
        """Sets the initial state circuit (default: uniform superposition via H).

        Args:
            circuit: A Circuit acting on the same number of qubits.

        Returns:
            A new QAOAAnsatz with the updated initial state.

        Raises:
            ValueError: If the circuit has a different qubit count.
        """
        ...

    def validate(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def build_circuit(self, prefix: str) -> Circuit:
        """Builds the QAOA circuit.

        Parameters alternate: ``{prefix}_0`` (γ₁), ``{prefix}_1`` (β₁),
        ``{prefix}_2`` (γ₂), ``{prefix}_3`` (β₂), ...

        Args:
            prefix: Prefix for parameter names (e.g. ``"p"``).

        Returns:
            A Circuit with ``2 * reps`` symbolic parameters.

        Raises:
            ValueError: If the configuration is invalid.
        """
        ...

    def num_parameters(self) -> int:
        """Returns the total number of parameters (= 2 * reps)."""
        ...

    def num_qubits(self) -> int:
        """Returns the number of qubits."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
