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

"""Quantum state simulation module.

This module provides quantum state representations including:
- Statevector: Pure quantum state simulation
- DensityMatrix: Mixed quantum state simulation
- DensityMatrixNoise: Noisy density matrix simulation
- StabilizerState: Clifford stabilizer simulation

# Examples

```python
from cqlib.qis.state import Statevector, DensityMatrix, DensityMatrixNoise

# Create a Bell state with Statevector
sv = Statevector(2)
sv.apply_h(0)
sv.apply_cx(0, 1)

# Get measurement probabilities
probs = sv.probabilities()
print(probs)  # [0.5, 0.0, 0.0, 0.5]

# Create a mixed state with DensityMatrix
dm = DensityMatrix(1)
dm.apply_h(0)
print(dm.trace())  # 1.0

# Create a noisy simulator
from cqlib.device import NoiseModel
sim = DensityMatrixNoise(2, NoiseModel())
```
"""

from .statevector import Statevector as Statevector
from .density_matrix import DensityMatrix as DensityMatrix
from .density_matrix_noise import DensityMatrixNoise as DensityMatrixNoise
from .classical import RuntimeValue as RuntimeValue
from .classical import ClassicalState as ClassicalState
from .stabilizer import StabilizerState as StabilizerState
from .stabilizer import StabilizerCircuitResult as StabilizerCircuitResult

__all__ = [
    "Statevector",
    "DensityMatrix",
    "DensityMatrixNoise",
    "RuntimeValue",
    "ClassicalState",
    "StabilizerState",
    "StabilizerCircuitResult",
]
