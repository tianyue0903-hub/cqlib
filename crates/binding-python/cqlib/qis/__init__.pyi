# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Quantum Information Science (QIS) module.

This module provides quantum information tools including:
- Pauli operators and strings
- Hamiltonian construction
- Statevector simulation
- Density matrix simulation
- Quantum metrics and entanglement measures
"""

from typing import Protocol, List, Tuple, Dict

# Pauli module
from .pauli import Phase as Phase
from .pauli import Pauli as Pauli
from .pauli import PauliString as PauliString

# Hamiltonian module
from .hamiltonian import Hamiltonian as Hamiltonian

# Evolution module
from .evolution import TrotterMode as TrotterMode

# State simulation module
from .state import DensityMatrix as DensityMatrix
from .state import Statevector as Statevector

# Entropy and metrics modules
from . import entropy as entropy
from . import metrics as metrics

class Observable(Protocol):
    """Protocol for quantum observables."""
    def expectation_statevector(self, sv: Statevector) -> float: ...
    def expectation_density_matrix(self, dm: DensityMatrix) -> float: ...
    def expectation_probs(
        self, measurements: List[Tuple[PauliString, Dict[str, float]]]
    ) -> float: ...
    @property
    def num_qubits(self) -> int: ...

__all__ = [
    "Phase",
    "Pauli",
    "PauliString",
    "Hamiltonian",
    "TrotterMode",
    "Statevector",
    "DensityMatrix",
    "Observable",
    "entropy",
    "metrics",
]
