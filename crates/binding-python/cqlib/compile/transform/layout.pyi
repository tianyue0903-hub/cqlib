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

"""Initial logical-to-physical qubit layout selection."""

from __future__ import annotations

from cqlib.circuit import Circuit
from cqlib.compile.sabre import SabreConfig
from cqlib.device import Device, Layout

class LayoutObjective:
    """Weighted objective used to rank candidate initial layouts.

    Lower scores are better. Weight validation occurs when an algorithm scores
    a layout; every weight must then be finite and non-negative.
    """

    def __init__(
        self,
        *,
        distance_weight: float = 1.0,
        direction_weight: float = 1.0,
        two_qubit_error_weight: float = 0.0,
        readout_error_weight: float = 0.0,
    ) -> None:
        """Create an objective from explicit component weights."""
        ...
    @staticmethod
    def topology_only() -> LayoutObjective:
        """Return the topology-only objective."""
        ...
    @staticmethod
    def fidelity_aware() -> LayoutObjective:
        """Return the default fidelity-aware objective.

        Missing calibration entries contribute zero rather than failing.
        """
        ...
    @staticmethod
    def auto_from_device(device: Device) -> LayoutObjective:
        """Use fidelity scoring when ``device`` has usable calibration data.

        Otherwise this returns :meth:`topology_only`.

        Raises:
            ValueError: If the device cannot be converted into a usable
                physical layout graph.
        """
        ...
    @staticmethod
    def fidelity_required(device: Device) -> LayoutObjective:
        """Return a fidelity-aware objective and require calibration data.

        Raises:
            ValueError: If the device is invalid or has no usable fidelity
                data.
        """
        ...
    @property
    def distance_weight(self) -> float:
        """Weight for logical-interaction distance."""
        ...
    @property
    def direction_weight(self) -> float:
        """Weight for directed-coupling mismatch."""
        ...
    @property
    def two_qubit_error_weight(self) -> float:
        """Weight for known two-qubit error rates."""
        ...
    @property
    def readout_error_weight(self) -> float:
        """Weight for known readout error rates."""
        ...
    @property
    def uses_fidelity(self) -> bool:
        """Whether either fidelity component can affect scoring."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __copy__(self) -> LayoutObjective: ...
    def __deepcopy__(self, memo: dict[int, object]) -> LayoutObjective: ...

class LayoutScore:
    """Breakdown of a candidate layout score."""

    @property
    def total(self) -> float:
        """Weighted sum of all score components."""
        ...
    @property
    def distance(self) -> float:
        """Raw weighted logical-interaction distance."""
        ...
    @property
    def direction(self) -> float:
        """Raw direction-mismatch component."""
        ...
    @property
    def two_qubit_error(self) -> float:
        """Raw two-qubit error component."""
        ...
    @property
    def readout_error(self) -> float:
        """Raw readout error component."""
        ...
    @property
    def used_fidelity(self) -> bool:
        """Whether the objective was configured to use fidelity terms."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __copy__(self) -> LayoutScore: ...
    def __deepcopy__(self, memo: dict[int, object]) -> LayoutScore: ...

class LayoutDiagnostics:
    """Search and scoring diagnostics from a layout algorithm."""

    @property
    def is_perfect(self) -> bool:
        """Whether all positive interactions map to adjacent qubits."""
        ...
    @property
    def candidates_evaluated(self) -> int:
        """Number of candidates considered using the algorithm's search unit."""
        ...
    @property
    def used_fidelity(self) -> bool:
        """Whether fidelity data contributed to the selected score."""
        ...
    @property
    def notes(self) -> list[str]:
        """Copy of human-readable diagnostic notes."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __copy__(self) -> LayoutDiagnostics: ...
    def __deepcopy__(self, memo: dict[int, object]) -> LayoutDiagnostics: ...

class LayoutResult:
    """Selected initial layout, optional score, and diagnostics."""

    @property
    def layout(self) -> Layout:
        """Selected logical-to-physical mapping."""
        ...
    @property
    def score(self) -> LayoutScore | None:
        """Score used to rank this layout, when available."""
        ...
    @property
    def diagnostics(self) -> LayoutDiagnostics:
        """Diagnostics emitted by the layout algorithm."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __copy__(self) -> LayoutResult: ...
    def __deepcopy__(self, memo: dict[int, object]) -> LayoutResult: ...

class Vf2EdgeRequirement:
    """Select which logical interactions are hard VF2 constraints."""

    @staticmethod
    def positive_interactions() -> Vf2EdgeRequirement:
        """Require interactions with positive accumulated weight."""
        ...
    @staticmethod
    def all_interactions() -> Vf2EdgeRequirement:
        """Require every stored interaction."""
        ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __copy__(self) -> Vf2EdgeRequirement: ...
    def __deepcopy__(self, memo: dict[int, object]) -> Vf2EdgeRequirement: ...

class Vf2LayoutConfig:
    """Configuration for VF2 perfect-layout search.

    Validation occurs when :func:`vf2_perfect_layout` runs.
    """

    def __init__(
        self,
        *,
        candidate_limit: int = 10,
        call_limit: int | None = None,
        edge_requirement: Vf2EdgeRequirement | None = None,
    ) -> None:
        """Create a VF2 configuration using Core defaults when omitted."""
        ...
    @property
    def candidate_limit(self) -> int:
        """Maximum number of complete candidates to score."""
        ...
    @property
    def call_limit(self) -> int | None:
        """Maximum partial mapping extensions, or ``None`` for no limit."""
        ...
    @property
    def edge_requirement(self) -> Vf2EdgeRequirement:
        """Hard interaction constraint used by VF2."""
        ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __copy__(self) -> Vf2LayoutConfig: ...
    def __deepcopy__(self, memo: dict[int, object]) -> Vf2LayoutConfig: ...

def trivial_layout(
    circuit: Circuit,
    device: Device,
    objective: LayoutObjective | None = None,
) -> LayoutResult:
    """Map logical and usable physical qubits in their existing order.

    The input objects are not modified. ``None`` selects topology-only scoring.

    Raises:
        ValueError: If capacity is insufficient or layout scoring fails.
    """
    ...

def greedy_layout(
    circuit: Circuit,
    device: Device,
    objective: LayoutObjective | None = None,
) -> LayoutResult:
    """Build a deterministic greedy initial layout.

    The input objects are not modified. ``None`` selects topology-only scoring.

    Raises:
        ValueError: If capacity, topology, circuit, or scoring is invalid.
    """
    ...

def vf2_perfect_layout(
    circuit: Circuit,
    device: Device,
    objective: LayoutObjective | None = None,
    config: Vf2LayoutConfig | None = None,
) -> LayoutResult:
    """Search for a non-induced topology-perfect initial layout.

    ``None`` selects topology-only scoring and the default VF2 configuration.

    Raises:
        ValueError: If configuration or capacity is invalid, no perfect
            mapping exists, or scoring fails.
    """
    ...

def sabre_layout(
    circuit: Circuit,
    device: Device,
    objective: LayoutObjective | None = None,
    config: SabreConfig | None = None,
) -> LayoutResult:
    """Select an initial layout with SABRE forward/backward refinement.

    This function does not insert SWAPs or return a routed circuit. ``None``
    selects topology-only scoring and the default SABRE configuration.

    Raises:
        ValueError: If configuration, capacity, topology, circuit, or scoring
            is invalid.
    """
    ...

__all__: list[str]
