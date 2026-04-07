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

from typing import Dict, List, Optional, Tuple, TypedDict
from cqlib.circuit import Circuit, Operation

class CommutativeOptimization:
    """Optimize the given Circuit by merging the adjacent gates with
    the commutative relation between gates in consideration.

    During the process, several parameterization and deparameterization process could be included, as listed
        `'x'`: Rx <--> X, SX, SX_dagger
        `'y'`: Ry <--> Y, SY, SY_dagger
        `'z'`: Rz <--> Z, S, T, S_dagger, T_dagger
    Whether to parameterize or deparameterize certain kinds of gates could be specified
    by listing them in `para` and `depara`.
    """
    def __init__(
        self,
        para: Optional[List[str]],
        depara: Optional[List[str]],
        keep_phase: bool,
        keep_order: bool,
    ) -> None:
        """Initializes the CommutativeOptimization instance.

        Args:
            para: A list of parameters to be parameterized.
            depara: A list of parameters to be deparameterized.
            keep_phase: Whether to keep the global phase of the circuit.
            keep_order: Force to ignore parameterization and deparameterization.
        """
        ...

    @staticmethod
    def is_commutative(a: Operation, b: Operation) -> bool:
        """Checks whether two operations are commutative.

        Args:
            a: The first operation.
            b: The second operation.

        Returns:
            bool: `True` if the operations commute, `False` otherwise.
        """
        ...

    def execute(self, circuit: Circuit) -> Circuit:
        """Executes the commutative optimization on the given circuit.

        Args:
            circuit: The input circuit to be optimized.

        Returns:
            Circuit: The optimized circuit.
        """
        ...

class Vf2CandidateScoreDict(TypedDict):
    """Scoring breakdown for one VF2 initial-layout candidate.

    Attributes:
        total: Weighted sum of all score components.
        fidelity: Edge-fidelity component in [0, 1].
        topology_fit: Topology-distance component in [0, 1].
        gate_distribution: Gate-distribution component in [0, 1].
    """

    total: float
    fidelity: float
    topology_fit: float
    gate_distribution: float

class Vf2LayoutCandidateDict(TypedDict):
    """A candidate logical-to-physical layout produced by VF2 search.

    Attributes:
        region: Physical qubit ids selected as the candidate region.
        layout: Logical-index -> physical-id layout.
        score: Scoring breakdown for ranking.
    """

    region: List[int]
    layout: List[int]
    score: Vf2CandidateScoreDict

class Topology:
    """Hardware topology used by VF2 and SABRE mapping.

    The topology is treated as an undirected coupling graph for connectivity
    checks and route planning.
    """

    def __init__(
        self, qubits: List[int], couplings: List[Tuple[int, int] | Tuple[int, int, str]]
    ) -> None:
        """Creates a topology from physical qubits and couplings.

        Args:
            qubits: Physical qubit ids.
            couplings: Edge list of `(u, v)` or `(u, v, gate_name)`.

        Raises:
            ValueError: If qubit ids overflow internal representation.
        """
        ...

    @staticmethod
    def line(qubits: List[int]) -> "Topology":
        """Builds a line topology where adjacent ids are connected by CX.

        Args:
            qubits: Physical qubit ids in line order.

        Returns:
            Topology: A line-coupled topology instance.

        Raises:
            ValueError: If qubit ids overflow internal representation.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of physical qubits in the topology."""
        ...

    @property
    def num_couplings(self) -> int:
        """Returns the number of coupling edges in the topology."""
        ...

    @property
    def qubits(self) -> List[int]:
        """Returns all physical qubit ids in the topology."""
        ...

    def add_qubits(self, qubits: List[int]) -> None:
        """Adds physical qubits to topology."""
        ...

    def add_couplings(
        self, couplings: List[Tuple[int, int] | Tuple[int, int, str]]
    ) -> None:
        """Adds coupling edges to topology."""
        ...

    def remove_qubits(self, qubits: List[int]) -> None:
        """Removes physical qubits from topology."""
        ...

    def remove_couplings(self, couplings: List[Tuple[int, int]]) -> None:
        """Removes coupling edges from topology."""
        ...

    def is_connected(self, u: int, v: int) -> bool:
        """Checks whether two physical qubits are directly connected.

        Args:
            u: First physical qubit id.
            v: Second physical qubit id.

        Returns:
            bool: `True` when an edge exists between `u` and `v`.

        Raises:
            ValueError: If a qubit id overflows internal representation.
        """
        ...

    def neighbors(self, qubit: int) -> List[int]:
        """Returns neighbor qubits directly connected to `qubit`."""
        ...

    def get_coupling_name(self, u: int, v: int) -> Optional[str]:
        """Returns the coupling name between `u` and `v`, if it exists."""
        ...

    def contains_qubit(self, qubit: int) -> bool:
        """Returns whether `qubit` exists in the topology."""
        ...

    def degree(self, qubit: int) -> int:
        """Returns the degree (number of couplings) of `qubit`."""
        ...

class SabreConfig:
    """Configuration object for `map_with_vf2_sabre`."""

    def __init__(
        self,
        vf2_policy: str = "direct_then_sabre",
        field_mode: bool = True,
        size_e: int = 20,
        w: float = 0.5,
        decay_coff: float = 0.001,
        decay_reset_time: int = 5,
        greedy_strategy: int = 3,
        initial_iterations: int = 1,
        repeat_iterations: int = 1,
        swap_iterations: int = 1,
        seed: int = -1,
    ) -> None:
        """Creates a SABRE configuration.

        Args:
            vf2_policy: One of `direct_then_sabre`, `initial_only`, `disabled`.
            field_mode: Enables field-aware swap scoring.
            size_e: Window size used by SABRE look-ahead heuristic.
            w: Weight parameter for look-ahead term.
            decay_coff: Decay coefficient for repeated swap penalties.
            decay_reset_time: Steps before decay reset.
            greedy_strategy: Internal SABRE greedy strategy id.
            initial_iterations: Number of initial-layout sampling iterations.
            repeat_iterations: Number of alternating refinement iterations.
            swap_iterations: Number of swap-sampling iterations per stage.
            seed: RNG seed (`-1` means random seed).

        Raises:
            ValueError: If `vf2_policy` is not recognized.
        """
        ...

class TemplateMatching:
    """Template-matching engine for compile-ready circuits."""

    def __init__(self) -> None:
        """Creates a new matcher instance."""
        ...

    def run(
        self,
        circuit: Circuit,
        template: Circuit,
        qubit_fixing_cnt: Optional[int] = None,
        prune_depth: Optional[int] = None,
        prune_width: Optional[int] = None,
    ) -> List[Tuple[List[Tuple[int, int]], List[int]]]:
        """Runs template matching.

        Args:
            circuit: Circuit to match.
            template: Template circuit.
            qubit_fixing_cnt: Optional matching heuristic parameter.
            prune_depth: Optional prune depth.
            prune_width: Optional prune width.

        Returns:
            List[Tuple[List[Tuple[int, int]], List[int]]]: Match pairs and qubit mapping.

        Raises:
            ValueError: If preprocessing or matching fails.
        """
        ...


class TemplateOptimization:
    """Template-based optimizer for compile-ready circuits."""

    def __init__(
        self,
        template_list: Optional[List[Circuit]] = None,
        qubit_fixing_cnt: Optional[int] = None,
        prune_depth: Optional[int] = None,
        prune_width: Optional[int] = None,
        template_file: Optional[str] = None,
    ) -> None:
        """Creates an optimizer.

        Args:
            template_list: Optional explicit template list.
            qubit_fixing_cnt: Optional matching heuristic parameter.
            prune_depth: Optional prune depth.
            prune_width: Optional prune width.
            template_file: Optional `.json` or `.qcis` template file path.

        Raises:
            ValueError: If template inputs are invalid or loading fails.
        """
        ...

    def execute(self, circuit: Circuit) -> Circuit:
        """Runs one optimization pass.

        Args:
            circuit: Input circuit.

        Returns:
            Circuit: Optimized circuit.

        Raises:
            ValueError: If optimization fails.
        """
        ...

    def execute_iterative(
        self,
        circuit: Circuit,
        max_iterations: Optional[int] = None,
    ) -> Circuit:
        """Runs optimization iteratively.

        Args:
            circuit: Input circuit.
            max_iterations: Optional max iteration count.

        Returns:
            Circuit: Optimized circuit.

        Raises:
            ValueError: If optimization fails.
        """
        ...

    def template_count(self) -> int:
        """Returns loaded template count."""
        ...


class CliffordRzOptimization:
    """Numeric Clifford+Rz optimizer for linear segments and control-flow bodies."""

    def __init__(
        self,
        level: str = "light",
        numeric_tol: float = 1e-10,
        strategies: Optional[List[str]] = None,
    ) -> None:
        """Creates a Clifford+Rz optimizer.

        Args:
            level: Optimization level, one of `light`, `heavy`, or `custom`.
            numeric_tol: Numeric tolerance for floating-point comparisons.
            strategies: Ordered custom strategy list when `level='custom'`.

        Raises:
            ValueError: If inputs are invalid.
        """
        ...

    @property
    def level(self) -> str:
        """Returns the configured optimization level."""
        ...

    @property
    def numeric_tol(self) -> float:
        """Returns the configured numeric tolerance."""
        ...

    @property
    def strategies(self) -> List[str]:
        """Returns the effective ordered strategy list."""
        ...

    def execute(self, circuit: Circuit) -> Circuit:
        """Runs one Clifford+Rz optimization pass.

        Args:
            circuit: Input circuit.

        Returns:
            Circuit: Optimized circuit.

        Raises:
            ValueError: If optimization fails.
        """
        ...
def vf2_is_subgraph_isomorphic(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> bool:
    """Checks whether strict VF2 subgraph mapping exists.

    Args:
        circuit: Logical circuit to map.
        topology: Target hardware topology.
        fidelity_map: Optional edge-fidelity overrides in [0, 1].

    Returns:
        bool: `True` if VF2 can embed the circuit without routing.

    Raises:
        ValueError: If topology/fidelity/circuit validation fails.

    Example:
        >>> ok = vf2_is_subgraph_isomorphic(circuit, topology)
    """
    ...

def vf2_find_initial_layout(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[List[int]]:
    """Finds a logical-to-physical initial layout.

    The function first attempts strict full-graph monomorphism and then falls
    back to candidate search if needed.

    Args:
        circuit: Logical circuit to map.
        topology: Target hardware topology.
        fidelity_map: Optional edge-fidelity overrides in [0, 1].

    Returns:
        Optional[List[int]]: Logical-index -> physical-id layout, or `None`.

    Raises:
        ValueError: If topology/fidelity/circuit validation fails.

    Example:
        >>> layout = vf2_find_initial_layout(circuit, topology)
    """
    ...

def vf2_find_initial_layout_candidates(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
    top_k: int = 10,
    w_fidelity: float = 0.5,
    w_topology: float = 0.3,
    w_gate_distribution: float = 0.2,
    max_seed_subgraphs: int = 2000,
    max_matches_per_subgraph: int = 128,
    region_beam_width: int = 32,
    region_oversample_factor: int = 3,
) -> List[Vf2LayoutCandidateDict]:
    """Returns scored VF2 initial-layout candidates.

    Args:
        circuit: Logical circuit to map.
        topology: Target hardware topology.
        fidelity_map: Optional edge-fidelity overrides in [0, 1].
        top_k: Maximum number of returned candidates.
        w_fidelity: Candidate score weight for fidelity fit.
        w_topology: Candidate score weight for topology-distance fit.
        w_gate_distribution: Candidate score weight for gate-distribution fit.
        max_seed_subgraphs: Max connected logical subgraphs explored.
        max_matches_per_subgraph: Max matches collected per explored subgraph.
        region_beam_width: Beam width for physical-region expansion.
        region_oversample_factor: Oversampling before top-k filtering.

    Returns:
        List[Vf2LayoutCandidateDict]: Ranked candidate layouts with scores.

    Raises:
        ValueError: If topology/fidelity/circuit/options validation fails.

    Example:
        >>> candidates = vf2_find_initial_layout_candidates(
        ...     circuit,
        ...     topology,
        ...     top_k=5,
        ... )
    """
    ...

def vf2_map(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Circuit:
    """Runs strict VF2 mapping without inserting routing gates.

    Args:
        circuit: Logical circuit to map.
        topology: Target hardware topology.
        fidelity_map: Optional edge-fidelity overrides in [0, 1].

    Returns:
        Circuit: Mapped circuit that fits topology edges directly.

    Raises:
        ValueError: If no strict mapping exists or validation fails.

    Example:
        >>> mapped = vf2_map(circuit, topology)
    """
    ...

def map_with_vf2_sabre(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
    config: Optional[SabreConfig] = None,
) -> Circuit:
    """Runs VF2 + SABRE hybrid mapping and routing.

    Depending on `config.vf2_policy`, the function may:
    1. Use strict VF2 mapping directly when possible.
    2. Use VF2 only to seed SABRE initial layout.
    3. Skip VF2 and route with SABRE only.

    Args:
        circuit: Logical circuit to map.
        topology: Target hardware topology.
        fidelity_map: Optional edge-fidelity overrides in [0, 1].
        config: Optional SABRE configuration. Defaults are used when omitted.

    Returns:
        Circuit: A topology-compliant mapped circuit.

    Raises:
        ValueError: If validation or mapping/routing fails.

    Example:
        >>> mapped = map_with_vf2_sabre(circuit, topology, config=SabreConfig())
    """
    ...
