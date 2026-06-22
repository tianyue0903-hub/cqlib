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
import math
import sys

from cqlib.circuit import (
    Instruction,
    Parameter,
    Qubit,
    StandardGate,
    ValueInstruction,
    ValueOperation,
)
from cqlib.compile import commutation
from cqlib.compile.commutation import (
    Commutation,
    CommutationChecker,
    CommutationConfig,
    algebraic_commutation,
    check_commutation,
)


def operation(gate: StandardGate, qubits: list[int]) -> ValueOperation:
    return ValueOperation.from_standard_gate(gate, [Qubit(index) for index in qubits])


def test_commutation_modules_and_public_exports_are_registered():
    assert commutation.check_commutation is check_commutation
    assert "cqlib._native.compile.commutation" in sys.modules
    assert Commutation.__module__ == "cqlib.compile.commutation"
    assert CommutationConfig.__module__ == "cqlib.compile.commutation"
    assert CommutationChecker.__module__ == "cqlib.compile.commutation"


def test_builtin_checker_proves_exact_and_global_phase_commutation():
    disjoint = check_commutation(
        operation(StandardGate.H, [0]), operation(StandardGate.X, [1])
    )
    assert disjoint == Commutation.exact()
    assert disjoint.is_exact()
    assert disjoint.phase.evaluate() == 0.0

    global_phase = check_commutation(
        operation(StandardGate.X, [0]), operation(StandardGate.Z, [0])
    )
    assert global_phase is not None
    assert not global_phase.is_exact()
    assert math.isclose(global_phase.phase.evaluate(), math.pi, abs_tol=1e-10)
    assert "Parameter(" in repr(global_phase)


def test_symbolic_parameters_are_preserved_for_algebraic_proofs():
    lhs = operation(StandardGate.RZ(Parameter("a")), [0])
    rhs = operation(StandardGate.RZ(Parameter("b")), [0])

    proof = algebraic_commutation(lhs, rhs)

    assert proof == Commutation.exact()


def test_unproven_and_malformed_applications_return_none():
    assert check_commutation(
        operation(StandardGate.H, [0]), operation(StandardGate.X, [0])
    ) is None
    assert check_commutation(
        operation(StandardGate.CX, [0]), operation(StandardGate.X, [0])
    ) is None

    delay = ValueOperation(ValueInstruction.from_instruction(Instruction.delay()), [])
    assert check_commutation(delay, delay) is None


def test_checker_configuration_and_copy_protocols():
    config = CommutationConfig(
        enable_rule_oracle=False,
        enable_matrix_fallback=False,
        max_matrix_qubits=2,
    )
    checker = CommutationChecker.with_config(config)

    assert checker.config == config
    assert checker.config is not config
    assert copy.copy(config) == config
    assert copy.deepcopy(config) == config
    assert copy.copy(checker).config == config
    assert copy.deepcopy(checker).config == config
    assert "max_matrix_qubits=2" in repr(checker)

    proof = checker.check(
        operation(StandardGate.X, [0]), operation(StandardGate.Z, [0])
    )
    assert proof is not None
    assert copy.copy(proof) == proof
    assert copy.deepcopy(proof) == proof
