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

from ...qis import PauliString
from .feature_map import PauliFeatureMap, ZZFeatureMap
from .two_local import EntanglementTopology, TwoLocal

def real_amplitudes(
    num_qubits: int,
    reps: int,
    entanglement: EntanglementTopology,
) -> TwoLocal:
    """Creates a RealAmplitudes ansatz.

    A hardware-efficient ansatz with a single RY rotation layer and CX entanglement.

    Structure: ``[RY layer] → [CX entanglement] × reps → [final RY layer]``

    Args:
        num_qubits: Number of qubits (≥ 1).
        reps: Number of [Rotation + Entanglement] layers.
        entanglement: Connectivity topology.

    Returns:
        A TwoLocal configured as RealAmplitudes.
        Total parameters = ``(reps + 1) * num_qubits``.

    Examples:
        >>> from cqlib.circuit.ansatz import real_amplitudes, EntanglementTopology
        >>> a = real_amplitudes(3, 2, EntanglementTopology.linear())
        >>> a.num_parameters()
        9
    """
    ...

def efficient_su2(
    num_qubits: int,
    reps: int,
    entanglement: EntanglementTopology,
) -> TwoLocal:
    """Creates an EfficientSU2 ansatz.

    A hardware-efficient ansatz spanning SU(2) via [RY, RZ] rotations and CX gates.
    Widely used in VQE and quantum machine learning.

    Structure: ``[RY+RZ layer] → [CX entanglement] × reps → [final RY+RZ layer]``

    Args:
        num_qubits: Number of qubits (≥ 1).
        reps: Number of [Rotation + Entanglement] layers.
        entanglement: Connectivity topology.

    Returns:
        A TwoLocal configured as EfficientSU2.
        Total parameters = ``(reps + 1) * num_qubits * 2``.

    Examples:
        >>> from cqlib.circuit.ansatz import efficient_su2, EntanglementTopology
        >>> a = efficient_su2(2, 1, EntanglementTopology.full())
        >>> a.num_parameters()
        8
    """
    ...

def zz_feature_map(
    num_qubits: int,
    reps: int,
    entanglement: EntanglementTopology,
) -> ZZFeatureMap:
    """Creates a ZZFeatureMap.

    A second-order Pauli-Z feature map (Z + ZZ interactions). Widely used
    for quantum kernel estimation in quantum machine learning.

    Args:
        num_qubits: Number of qubits (= number of input features, ≥ 1).
        reps: Number of repetition layers.
        entanglement: Connectivity for ZZ interactions.

    Returns:
        A ZZFeatureMap with ``num_qubits`` parameters.

    Examples:
        >>> from cqlib.circuit.ansatz import zz_feature_map, EntanglementTopology
        >>> fm = zz_feature_map(3, 2, EntanglementTopology.full())
        >>> fm.num_parameters()
        3
    """
    ...

def pauli_feature_map(
    num_qubits: int,
    reps: int,
    paulis: list[PauliString],
    entanglement: EntanglementTopology,
) -> PauliFeatureMap:
    """Creates a PauliFeatureMap with custom Pauli strings.

    Args:
        num_qubits: Number of qubits (= number of input features, ≥ 1).
        reps: Number of repetition layers.
        paulis: List of PauliString templates. The number of non-identity
                operators in each string determines its locality k.
        entanglement: Connectivity topology for multi-qubit interactions.

    Returns:
        A PauliFeatureMap with ``num_qubits`` parameters.

    Examples:
        >>> from cqlib.circuit.ansatz import pauli_feature_map, EntanglementTopology
        >>> from cqlib import PauliString
        >>> fm = pauli_feature_map(
        ...     3, 2,
        ...     [PauliString("Z"), PauliString("ZZ")],
        ...     EntanglementTopology.full(),
        ... )
        >>> fm.num_parameters()
        3
    """
    ...
