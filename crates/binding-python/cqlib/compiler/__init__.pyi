from typing import Dict, List, Optional, Tuple
from cqlib.circuit import Circuit

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
