from typing import Dict, List, Optional, Tuple, TypedDict
from cqlib.circuit import Circuit

class Vf2CandidateScoreDict(TypedDict):
    total: float
    fidelity: float
    topology_fit: float
    gate_distribution: float

class Vf2LayoutCandidateDict(TypedDict):
    # Selected physical region (qubit ids).
    region: List[int]
    # Logical-index -> physical-id layout.
    layout: List[int]
    # Scoring breakdown.
    score: Vf2CandidateScoreDict

class Topology:
    def __init__(self, qubits: List[int], couplings: List[Tuple[int, int] | Tuple[int, int, str]]) -> None: ...
    @staticmethod
    def line(qubits: List[int]) -> "Topology": ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_couplings(self) -> int: ...
    def is_connected(self, u: int, v: int) -> bool: ...

class SabreConfig:
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
    ) -> None: ...

def vf2_is_subgraph_isomorphic(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> bool: ...

def vf2_find_initial_layout(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[List[int]]: ...

def vf2_find_initial_layout_candidates(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
    # Maximum number of returned candidates.
    top_k: int = 10,
    # Candidate scoring weights (normalized internally).
    w_fidelity: float = 0.5,
    w_topology: float = 0.3,
    w_gate_distribution: float = 0.2,
    # Max connected logical subgraphs explored for seed generation.
    max_seed_subgraphs: int = 2000,
    # Max VF2 matches collected per explored subgraph.
    max_matches_per_subgraph: int = 128,
    # Beam width used by physical-region expansion.
    region_beam_width: int = 32,
    # Oversampling multiplier before final top-k filtering.
    region_oversample_factor: int = 3,
) -> List[Vf2LayoutCandidateDict]: ...

def vf2_map(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
) -> Circuit: ...

def map_with_vf2_sabre(
    circuit: Circuit,
    topology: Topology,
    fidelity_map: Optional[Dict[Tuple[int, int], float]] = None,
    config: Optional[SabreConfig] = None,
) -> Circuit: ...
