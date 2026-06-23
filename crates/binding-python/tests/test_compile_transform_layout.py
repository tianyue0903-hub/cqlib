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

import copy
import sys

import pytest

from cqlib.circuit import Circuit
from cqlib.compile import CompilerConfigError
from cqlib.compile.sabre import SabreConfig
from cqlib.compile.transform.layout import (
    LayoutDiagnostics,
    LayoutObjective,
    LayoutResult,
    LayoutScore,
    Vf2EdgeRequirement,
    Vf2LayoutConfig,
    greedy_layout,
    sabre_layout,
    trivial_layout,
    vf2_perfect_layout,
)
from cqlib.device import Device, Layout


def test_layout_module_and_public_types_are_registered() -> None:
    assert "cqlib._native.compile.transform.layout" in sys.modules
    for public_type in (
        LayoutObjective,
        LayoutScore,
        LayoutDiagnostics,
        LayoutResult,
        Vf2EdgeRequirement,
        Vf2LayoutConfig,
    ):
        assert public_type.__module__ == "cqlib.compile.transform.layout"


def test_layout_configuration_is_immutable_and_copyable() -> None:
    objective = LayoutObjective(
        distance_weight=2.0,
        direction_weight=3.0,
        two_qubit_error_weight=4.0,
        readout_error_weight=5.0,
    )
    requirement = Vf2EdgeRequirement.all_interactions()
    config = Vf2LayoutConfig(
        candidate_limit=4,
        call_limit=20,
        edge_requirement=requirement,
    )

    assert objective.distance_weight == 2.0
    assert objective.uses_fidelity is True
    assert copy.copy(objective) == objective
    assert copy.deepcopy(config) == config
    assert config.edge_requirement == requirement
    assert hash(copy.copy(requirement)) == hash(requirement)

    with pytest.raises(AttributeError):
        config.candidate_limit = 5


def test_device_aware_objective_factories() -> None:
    device = Device.line("line-2", 2)

    automatic = LayoutObjective.auto_from_device(device)
    assert automatic == LayoutObjective.topology_only()

    with pytest.raises(CompilerConfigError, match="no usable fidelity data"):
        LayoutObjective.fidelity_required(device)

    device.default_readout_error = 0.01
    required = LayoutObjective.fidelity_required(device)
    assert required == LayoutObjective.fidelity_aware()


@pytest.mark.parametrize(
    "algorithm",
    [trivial_layout, greedy_layout, vf2_perfect_layout],
)
def test_deterministic_layout_algorithms_return_complete_results(algorithm) -> None:
    circuit = Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    device = Device.line("line-3", 3)

    result = algorithm(circuit, device)

    assert isinstance(result, LayoutResult)
    assert isinstance(result.layout, Layout)
    assert result.layout.num_logical == 3
    assert result.score is not None
    assert result.diagnostics.is_perfect is True
    assert result.diagnostics.candidates_evaluated >= 1


def test_sabre_layout_is_reproducible_with_a_fixed_seed() -> None:
    circuit = Circuit(3)
    circuit.cx(0, 2)
    device = Device.line("line-3", 3)
    config = SabreConfig.deterministic_seeded(7)

    first = sabre_layout(circuit, device, config=config)
    second = sabre_layout(circuit, device, config=config)

    assert first.layout.l2p_map == second.layout.l2p_map
    assert first.score == second.score
    assert first.diagnostics == second.diagnostics
    assert first == second
    assert first.__eq__(object()) is NotImplemented
    notes = first.diagnostics.notes
    notes.append("caller mutation")
    assert "caller mutation" not in first.diagnostics.notes


def test_invalid_objective_and_vf2_config_are_rejected_when_run() -> None:
    circuit = Circuit(2)
    circuit.cx(0, 1)
    device = Device.line("line-2", 2)

    with pytest.raises(CompilerConfigError, match="distance_weight must be finite and non-negative"):
        trivial_layout(circuit, device, LayoutObjective(distance_weight=-1.0))

    with pytest.raises(CompilerConfigError, match="candidate_limit must be greater than zero"):
        vf2_perfect_layout(circuit, device, config=Vf2LayoutConfig(candidate_limit=0))


def test_layout_rejects_insufficient_physical_capacity() -> None:
    with pytest.raises(CompilerConfigError, match="at least as many usable physical qubits"):
        greedy_layout(Circuit(3), Device.line("line-2", 2))


def test_layout_rejects_undecomposed_three_qubit_operation() -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)

    with pytest.raises(CompilerConfigError, match="more than two qubits to be decomposed"):
        trivial_layout(circuit, Device.line("line-3", 3))


def test_vf2_reports_when_no_perfect_embedding_exists() -> None:
    circuit = Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 2)

    with pytest.raises(CompilerConfigError, match="could not find a perfect mapping"):
        vf2_perfect_layout(circuit, Device.line("line-3", 3))
