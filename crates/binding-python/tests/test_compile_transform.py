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
from cqlib.compile.transform import (
    CanonicalizeConfig,
    CanonicalizeResult,
    Canonicalizer,
    canonicalize_circuit,
)


def test_transform_module_and_public_types_are_registered() -> None:
    assert "cqlib._native.compile.transform" in sys.modules
    assert CanonicalizeConfig.__module__ == "cqlib.compile.transform"
    assert Canonicalizer.__module__ == "cqlib.compile.transform"
    assert CanonicalizeResult.__module__ == "cqlib.compile.transform"


def test_canonicalize_config_exposes_immutable_options() -> None:
    config = CanonicalizeConfig(
        round_limit=3,
        recurse_control_flow=False,
        fold_gphase=False,
        canonicalize_instruction_form=False,
        drop_noops=False,
        canonicalize_barriers=False,
    )

    assert config.round_limit == 3
    assert config.recurse_control_flow is False
    assert config.fold_gphase is False
    assert config.canonicalize_instruction_form is False
    assert config.drop_noops is False
    assert config.canonicalize_barriers is False
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert repr(config).startswith("CanonicalizeConfig(round_limit=3,")

    with pytest.raises(AttributeError):
        config.round_limit = 4


def test_production_canonicalization_does_not_mutate_input() -> None:
    circuit = Circuit(1)
    circuit.i(0)

    result = canonicalize_circuit(circuit)

    assert len(circuit.operations) == 1
    assert len(result.circuit.operations) == 0
    assert result.changed is True
    assert result.rounds >= 1


def test_configured_canonicalizer_can_preserve_noops() -> None:
    circuit = Circuit(1)
    circuit.i(0)
    config = CanonicalizeConfig(drop_noops=False)
    canonicalizer = Canonicalizer(config)

    result = canonicalizer.run(circuit)

    assert canonicalizer.config == config
    assert len(result.circuit.operations) == 1
    assert result.changed is False


def test_canonicalization_is_idempotent() -> None:
    circuit = Circuit(1)
    circuit.i(0)
    circuit.h(0)

    first = canonicalize_circuit(circuit)
    second = canonicalize_circuit(first.circuit)

    assert first.changed is True
    assert second.changed is False
    assert len(second.circuit.operations) == 1


def test_zero_round_limit_is_rejected_when_run() -> None:
    canonicalizer = Canonicalizer(CanonicalizeConfig(round_limit=0))

    with pytest.raises(ValueError, match="round_limit must be greater than zero"):
        canonicalizer.run(Circuit(1))
