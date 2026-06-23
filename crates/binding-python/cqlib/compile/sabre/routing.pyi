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

"""SABRE configuration, routing results, and routing entry point."""

from __future__ import annotations

from collections.abc import Sequence

from cqlib.circuit import Circuit
from cqlib.device import Device, Layout

class SabreTrialObjective:
    """Objective used to select the best independent routing trial.

    The objective affects only final trial selection. Candidate SWAPs within a
    trial are still scored by :class:`SabreHeuristicConfig`.
    """

    @staticmethod
    def swap_count() -> SabreTrialObjective:
        """Minimize inserted SWAP count only."""
        ...
    @staticmethod
    def depth() -> SabreTrialObjective:
        """Minimize routed two-qubit depth only."""
        ...
    @staticmethod
    def swap_then_depth() -> SabreTrialObjective:
        """Minimize SWAP count, then use two-qubit depth to break ties.

        This is the default production objective.
        """
        ...
    @staticmethod
    def depth_then_swap() -> SabreTrialObjective:
        """Minimize two-qubit depth, then use SWAP count to break ties."""
        ...
    def __copy__(self) -> SabreTrialObjective:
        """Return this immutable objective value."""
        ...
    def __deepcopy__(self, memo: dict[int, object]) -> SabreTrialObjective:
        """Return this immutable objective value."""
        ...
    def __eq__(self, other: object) -> bool:
        """Return whether two objectives select trials identically."""
        ...
    def __hash__(self) -> int:
        """Return a stable hash for this objective during the process."""
        ...

class SabreHeuristicConfig:
    """Weights and limits used to choose candidate SWAPs within one trial.

    Lower candidate scores are preferred. ``basic_weight`` weights currently
    executable two-qubit interactions. ``lookahead_weights`` applies one
    weight per future interaction layer. When ``decay_increment`` is not
    ``None``, recently swapped physical qubits receive a temporary penalty.

    Validation occurs when :func:`sabre_route` is called. Weights and
    ``best_epsilon`` must be finite and non-negative. ``decay_reset`` must be
    positive when decay is enabled.

    Args:
        basic_weight: Weight of the current front-layer distance.
        lookahead_weights: Per-layer lookahead weights. ``None`` selects the
            core default ``[0.5]``; an empty sequence disables lookahead.
        decay_increment: Penalty added after a heuristic SWAP. ``None``
            disables decay.
        decay_reset: Number of heuristic SWAP attempts between decay resets.
        attempt_limit: SWAP attempts without progress before shortest-path
            fallback is used.
        best_epsilon: Absolute tolerance used to treat candidate scores as tied.

    Example::

        heuristic = SabreHeuristicConfig(
            lookahead_weights=[0.5, 0.25],
            decay_increment=0.01,
            attempt_limit=100,
        )
    """

    def __init__(
        self,
        *,
        basic_weight: float = 1.0,
        lookahead_weights: Sequence[float] | None = None,
        decay_increment: float | None = 0.001,
        decay_reset: int = 5,
        attempt_limit: int = 1000,
        best_epsilon: float = 1e-10,
    ) -> None: ...
    @property
    def basic_weight(self) -> float:
        """Weight of current front-layer interaction distances."""
        ...
    @property
    def lookahead_weights(self) -> list[float]:
        """Copy of the per-lookahead-layer distance weights."""
        ...
    @property
    def decay_increment(self) -> float | None:
        """Decay penalty increment, or ``None`` when decay is disabled."""
        ...
    @property
    def decay_reset(self) -> int:
        """Number of heuristic SWAP attempts between decay resets."""
        ...
    @property
    def attempt_limit(self) -> int:
        """No-progress limit before shortest-path fallback."""
        ...
    @property
    def best_epsilon(self) -> float:
        """Tolerance for candidate-score ties."""
        ...
    def __copy__(self) -> SabreHeuristicConfig:
        """Return an independent shallow copy."""
        ...
    def __deepcopy__(self, memo: dict[int, object]) -> SabreHeuristicConfig:
        """Return an independent deep copy."""
        ...
    def __eq__(self, other: object) -> bool:
        """Return whether every heuristic option is equal."""
        ...

class SabreConfig:
    """Configuration shared by SABRE layout refinement and routing.

    Direct :func:`sabre_route` calls start from a concrete layout, so
    ``layout_trials``, ``refinement_iterations``, and
    ``layout_scoring_trials`` do not affect that function. They are retained
    because the same configuration type is used by compiler layout passes.

    Args:
        layout_trials: Starting-layout trials used during layout refinement.
        refinement_iterations: Forward/backward iterations per layout trial.
        layout_scoring_trials: Routing trials used to score refined layouts.
        routing_trials: Independent trials used by :func:`sabre_route`; must
            be greater than zero.
        trial_objective: Trial-selection objective. ``None`` selects
            :meth:`SabreTrialObjective.swap_then_depth`.
        seed: Optional unsigned 64-bit seed. Equal seeds and inputs produce
            equal cqlib routing results.
        heuristic: Swap-selection settings. ``None`` uses core defaults.

    Example::

        config = SabreConfig(
            routing_trials=4,
            seed=23,
            trial_objective=SabreTrialObjective.depth_then_swap(),
            heuristic=SabreHeuristicConfig(attempt_limit=200),
        )
    """

    def __init__(
        self,
        *,
        layout_trials: int = 10,
        refinement_iterations: int = 1,
        layout_scoring_trials: int = 1,
        routing_trials: int = 5,
        trial_objective: SabreTrialObjective | None = None,
        seed: int | None = None,
        heuristic: SabreHeuristicConfig | None = None,
    ) -> None: ...
    @staticmethod
    def deterministic_seeded(seed: int) -> SabreConfig:
        """Return a compact deterministic configuration for tests/examples."""
        ...
    @property
    def layout_trials(self) -> int:
        """Number of starting-layout trials used during refinement."""
        ...
    @property
    def refinement_iterations(self) -> int:
        """Forward/backward refinement iterations per layout trial."""
        ...
    @property
    def layout_scoring_trials(self) -> int:
        """Routing trials used to score a refined layout."""
        ...
    @property
    def routing_trials(self) -> int:
        """Independent trials evaluated by direct routing."""
        ...
    @property
    def trial_objective(self) -> SabreTrialObjective:
        """Objective used to select the winning routing trial."""
        ...
    @property
    def seed(self) -> int | None:
        """Deterministic seed, or ``None`` for entropy-based seeding."""
        ...
    @property
    def heuristic(self) -> SabreHeuristicConfig:
        """Independent copy of the active swap-selection configuration."""
        ...
    def __copy__(self) -> SabreConfig:
        """Return an independent shallow copy."""
        ...
    def __deepcopy__(self, memo: dict[int, object]) -> SabreConfig:
        """Return an independent deep copy."""
        ...
    def __eq__(self, other: object) -> bool:
        """Return whether every SABRE option is equal."""
        ...

class SabreRoutingDiagnostics:
    """Read-only search diagnostics for the selected routed circuit."""

    @property
    def trials_evaluated(self) -> int:
        """Number of independent routing trials evaluated."""
        ...
    @property
    def selected_trial_index(self) -> int:
        """Zero-based index of the selected trial."""
        ...
    @property
    def fallback_count(self) -> int:
        """Number of shortest-path fallback invocations."""
        ...
    @property
    def control_flow_blocks_routed(self) -> int:
        """Number of recursively routed control-flow bodies."""
        ...
    @property
    def two_qubit_depth(self) -> int:
        """ASAP two-qubit depth of the selected routed operation stream."""
        ...
    @property
    def operation_count(self) -> int:
        """Total operation count of the selected routed stream."""
        ...
    def __copy__(self) -> SabreRoutingDiagnostics:
        """Return an independent shallow copy."""
        ...
    def __deepcopy__(self, memo: dict[int, object]) -> SabreRoutingDiagnostics:
        """Return an independent deep copy."""
        ...
    def __eq__(self, other: object) -> bool:
        """Return whether every diagnostic counter is equal."""
        ...

class SabreRoutingResult:
    """Circuit and layout metadata produced by :func:`sabre_route`.

    All object-valued properties return independent wrappers. The routed
    circuit uses physical qubit identifiers and includes inserted SWAP gates.
    """

    @property
    def circuit(self) -> Circuit:
        """Independent copy of the physical routed circuit."""
        ...
    @property
    def initial_layout(self) -> Layout:
        """Normalized initial layout used by the selected trial."""
        ...
    @property
    def final_layout(self) -> Layout:
        """Logical-to-physical layout after all routed operations."""
        ...
    @property
    def swap_count(self) -> int:
        """Number of inserted SWAPs, including control-flow epilogues."""
        ...
    @property
    def diagnostics(self) -> SabreRoutingDiagnostics:
        """Independent copy of routing search diagnostics."""
        ...
    def __copy__(self) -> SabreRoutingResult:
        """Return an independent shallow copy of the complete result."""
        ...
    def __deepcopy__(self, memo: dict[int, object]) -> SabreRoutingResult:
        """Return an independent deep copy of the complete result."""
        ...

def sabre_route(
    circuit: Circuit,
    device: Device,
    initial_layout: Layout,
    config: SabreConfig | None = None,
) -> SabreRoutingResult:
    """Route ``circuit`` from ``initial_layout`` onto ``device``.

    The router inserts SWAPs until every two-qubit operation acts on adjacent
    usable physical qubits. Control-flow bodies are routed recursively and
    restored to their entry layout before leaving the body. The original
    circuit, device, layout, and configuration are not modified.

    Args:
        circuit: Logical input circuit.
        device: Target device whose usable topology constrains interactions.
        initial_layout: Mapping for every logical qubit in ``circuit``.
        config: Routing configuration. ``None`` uses :class:`SabreConfig`
            defaults.

    Returns:
        Routed physical circuit, selected layouts, SWAP count, and diagnostics.

    Raises:
        ValueError: If configuration values are invalid, the layout does not
            map every circuit qubit to a usable physical qubit, an interaction
            crosses disconnected topology components, or the circuit contains
            an operation unsupported by the routing DAG.

    Example::

        circuit = Circuit(2)
        circuit.cx(0, 1)
        device = Device.line("line3", 3)
        layout = Layout.from_pairs([(0, 0), (1, 2)], physical_count=3)
        result = sabre_route(
            circuit,
            device,
            layout,
            SabreConfig(routing_trials=1, seed=7),
        )
        assert result.swap_count == 1
    """
    ...

__all__: list[str]
