# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Exact multi-controlled gate synthesis primitives."""

from ...._native import compile as _compile_module

_mc_gate_module = _compile_module.transform.decompose.mc_gate

Su2RotationAxis = _mc_gate_module.Su2RotationAxis

_FUNCTIONS = [
    "decompose_mcx_small",
    "decompose_mcx_no_aux",
    "decompose_mcx_n_clean",
    "decompose_mcx_n_dirty",
    "decompose_mcx_1_clean_b95",
    "decompose_mcx_1_clean_kg24",
    "decompose_mcx_1_dirty",
    "decompose_mcx_2_clean",
    "decompose_mcx_2_dirty",
    "decompose_mc_su2_no_aux",
    "decompose_mc_su2_n_clean",
    "decompose_rotation_no_aux",
    "decompose_rotation_n_clean",
    "decompose_pauli_small",
    "decompose_pauli_no_aux",
    "decompose_pauli_n_clean",
    "decompose_pauli_n_dirty",
    "decompose_pauli_1_clean_b95",
    "decompose_pauli_1_clean_kg24",
    "decompose_pauli_1_dirty",
    "decompose_pauli_2_clean",
    "decompose_pauli_2_dirty",
    "decompose_mc_rzz_no_aux",
    "decompose_mc_rzz_n_clean",
    "decompose_pauli_rotation_no_aux",
    "decompose_pauli_rotation_n_clean",
    "decompose_phase_no_aux",
    "decompose_phase_n_clean",
    "decompose_qcis_no_aux",
    "decompose_qcis_n_clean",
    "decompose_hadamard_no_aux",
    "decompose_hadamard_n_clean",
    "decompose_swap_no_aux",
    "decompose_swap_n_clean",
    "decompose_fsim_no_aux",
    "decompose_fsim_n_clean",
    "decompose_unitary_no_aux",
    "decompose_unitary_n_clean",
]

globals().update({name: getattr(_mc_gate_module, name) for name in _FUNCTIONS})

__all__ = ["Su2RotationAxis", *_FUNCTIONS]
