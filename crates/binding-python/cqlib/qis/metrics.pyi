# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

from typing import List
from .state import DensityMatrix, Statevector

def purity_pure(sv: Statevector) -> float:
    """Calculates the purity of a pure quantum state.

    For a valid, normalized Statevector, the purity is theoretically 1.0.

    Args:
        sv (Statevector): The statevector representing the pure state.

    Returns:
        float: The purity value (theoretically 1.0).

    Raises:
        ValueError: If the calculation fails.
    """
    ...

def purity_mixed(dm: DensityMatrix) -> float:
    """Calculates the purity of a mixed quantum state.

    Purity is defined as :math:`\\text{Tr}(\\rho^2)`.

    Args:
        dm (DensityMatrix): The density matrix representing the mixed state.

    Returns:
        float: The purity value, ranging from :math:`1/2^N` (maximally mixed) to 1.0 (pure state).

    Raises:
        ValueError: If the calculation fails.
    """
    ...

def state_fidelity_pure(sv1: Statevector, sv2: Statevector) -> float:
    """Calculates the state fidelity between two pure quantum states.

    Fidelity :math:`F(\\psi, \\phi) = |\\langle\\psi|\\phi\\rangle|^2`.

    Args:
        sv1 (Statevector): The first statevector.
        sv2 (Statevector): The second statevector.

    Returns:
        float: The fidelity value, ranging from 0.0 (orthogonal) to 1.0 (identical).

    Raises:
        ValueError: If the number of qubits in the two states do not match.
    """
    ...

def trace_distance_pure(sv1: Statevector, sv2: Statevector) -> float:
    """Calculates the trace distance between two pure quantum states.

    For pure states, trace distance :math:`D(\\psi, \\phi) = \\sqrt{1 - |\\langle\\psi|\\phi\\rangle|^2}`.

    Args:
        sv1 (Statevector): The first statevector.
        sv2 (Statevector): The second statevector.

    Returns:
        float: The trace distance, ranging from 0.0 (identical) to 1.0 (orthogonal).

    Raises:
        ValueError: If the number of qubits in the two states do not match.
    """
    ...

def state_fidelity_pure_mixed(sv: Statevector, dm: DensityMatrix) -> float:
    """Calculates the state fidelity between a pure state and a mixed state.

    Fidelity :math:`F(\\psi, \\rho) = \\langle\\psi|\\rho|\\psi\\rangle`.

    Args:
        sv (Statevector): The pure state.
        dm (DensityMatrix): The mixed state.

    Returns:
        float: The fidelity value in [0.0, 1.0].

    Raises:
        ValueError: If the number of qubits do not match.
    """
    ...

def entropy(dm: DensityMatrix) -> float:
    """Calculates the von Neumann entropy of a mixed state.

    Entropy :math:`S(\\rho) = -\\text{Tr}(\\rho \\log_2 \\rho)`.

    Args:
        dm (DensityMatrix): The density matrix representing the quantum state.

    Returns:
        float: The von Neumann entropy in units of bits (base-2 logarithm).

    Raises:
        ValueError: If eigendecomposition fails.
    """
    ...

def trace_distance_mixed(dm1: DensityMatrix, dm2: DensityMatrix) -> float:
    """Calculates the trace distance between two mixed quantum states.

    Trace distance :math:`D(\\rho, \\sigma) = \\frac{1}{2} \\text{Tr}|\\rho - \\sigma|`.

    Args:
        dm1 (DensityMatrix): The first density matrix.
        dm2 (DensityMatrix): The second density matrix.

    Returns:
        float: The trace distance.

    Raises:
        ValueError: If the number of qubits do not match.
    """
    ...

def state_fidelity_mixed(dm1: DensityMatrix, dm2: DensityMatrix) -> float:
    """Calculates the state fidelity between two mixed states.

    Fidelity :math:`F(\\rho, \\sigma) = (\\text{Tr}\\sqrt{\\sqrt{\\rho} \\sigma \\sqrt{\\rho}})^2`.

    Args:
        dm1 (DensityMatrix): The first density matrix.
        dm2 (DensityMatrix): The second density matrix.

    Returns:
        float: The fidelity value in [0.0, 1.0].

    Raises:
        ValueError: If the number of qubits do not match or eigendecomposition fails.
    """
    ...

def partial_transpose(dm: DensityMatrix, target_qubits: List[int]) -> DensityMatrix:
    """Performs the partial transpose operation on a density matrix.

    Transposes only the indices corresponding to the specified subsystem (target_qubits),
    leaving other qubits unchanged. This is fundamental for entanglement detection.

    Args:
        dm (DensityMatrix): The input density matrix.
        target_qubits (List[int]): The qubit indices specifying the subsystem to be transposed.

    Returns:
        DensityMatrix: A new density matrix resulting from the partial transpose.

    Raises:
        ValueError: If any target qubit index is out of bounds.
    """
    ...

def logarithmic_negativity(dm: DensityMatrix, sys_a: List[int]) -> float:
    """Calculates the logarithmic negativity of a bipartite quantum state.

    The logarithmic negativity is defined as :math:`E_N(\\rho) = \\log_2 ||\\rho^{T_A}||_1`,
    where :math:`\\rho^{T_A}` is the partial transpose of the density matrix with respect to subsystem A.

    Args:
        dm (DensityMatrix): The density matrix of the bipartite quantum state.
        sys_a (List[int]): The qubit indices comprising subsystem A.

    Returns:
        float: The logarithmic negativity value (>= 0). 0 indicates a separable state.

    Raises:
        ValueError: If subsystem indices are invalid or eigendecomposition fails.
    """
    ...
