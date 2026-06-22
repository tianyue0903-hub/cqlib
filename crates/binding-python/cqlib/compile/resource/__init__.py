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

"""Ancillary-qubit planning and leasing for compiler transformations.

Clean and dirty resources are restoration contracts, not state-simulation
results. A clean resource must enter and leave a transform in ``|0>``. A dirty
resource may contain an unknown or entangled state, so the transform must
restore its complete input state before releasing the lease.

The manager separates side-effect-free planning from mutation. Keep it
synchronized with the same circuit as both evolve::

    from cqlib.circuit import Circuit
    from cqlib.compile.resource import (
        AncillaRequirement,
        ResourceManager,
        ResourcePolicy,
        ResourceRequest,
    )

    circuit = Circuit(2)
    manager = ResourceManager.from_circuit(
        circuit,
        policy=ResourcePolicy(max_pre_layout_clean_ancillas=1),
    )
    request = ResourceRequest(AncillaRequirement.clean_zero(), 1)

    # Preview does not reserve resources or change the circuit.
    plan = manager.preview(request)
    lease = manager.commit(circuit, plan)

    # A consuming transform may use lease.qubits here. It must restore |0>.
    manager.release(lease)
    manager.verify_idle(circuit)

For normal compilation, pass only a policy to :func:`cqlib.compile.compile`;
the compiler owns the manager lifecycle internally.
"""

from ..._native import compile as _compile_module
from .manager import ResourceManager as ResourceManager
from .model import AncillaRequirement as AncillaRequirement
from .model import ResourceLease as ResourceLease
from .model import ResourcePlan as ResourcePlan
from .model import ResourceRequest as ResourceRequest
from .policy import ResourceLimits as ResourceLimits
from .policy import ResourcePolicy as ResourcePolicy

ResourceError = _compile_module.resource.ResourceError
ResourceUnavailableError = _compile_module.resource.ResourceUnavailableError

__all__ = [
    "AncillaRequirement",
    "ResourcePolicy",
    "ResourceLimits",
    "ResourceRequest",
    "ResourcePlan",
    "ResourceLease",
    "ResourceManager",
    "ResourceError",
    "ResourceUnavailableError",
]
