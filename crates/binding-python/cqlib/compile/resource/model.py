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

"""Resource request, preview, and lease value types.

``ResourcePlan`` and ``ResourceLease`` are credentials tied to the manager
that created them. Construct requests directly, but obtain plans from
``ResourceManager.preview`` and leases from ``ResourceManager.commit``.
"""

from ..._native import compile as _compile_module

_resource_module = _compile_module.resource

AncillaRequirement = _resource_module.AncillaRequirement
ResourceRequest = _resource_module.ResourceRequest
ResourcePlan = _resource_module.ResourcePlan
ResourceLease = _resource_module.ResourceLease

__all__ = [
    "AncillaRequirement",
    "ResourceRequest",
    "ResourcePlan",
    "ResourceLease",
]
