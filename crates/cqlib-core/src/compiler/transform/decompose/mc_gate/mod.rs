// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Multi-controlled gate decomposition.
//!
//! This module provides explicit synthesis primitives for lowering
//! [`Instruction::McGate`](crate::circuit::Instruction::McGate) operations.
//! The primitives do not choose an algorithm, allocate ancillary qubits, or
//! rewrite a circuit automatically. Those responsibilities belong to the
//! future multi-controlled-gate decomposition planner.
//!
//! # Gate categories and decomposition flow
//!
//! The modules form a layered hierarchy. Each layer reduces its gate family
//! to simpler primitives in the layer below:
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                    unitary                       │  U(θ,φ,λ) = phase + Z-Y-Z Euler
//! ├─────────────────────────────────────────────────┤
//! │       phase                pauli_rotation       │  Phase  = recursive projector
//! │   (S/SDG/T/TDG/Phase)   (RXX/RYY/RZZ/RZX)       │  R{XX,YY,ZX} = basis-change · RZZ
//! ├──────────────────┬──────────────────────────────┤
//! │    rotation      │            rzz               │  R{X,Y,Z} → mc_su2
//! │  (RX/RY/RZ/      │  CX · MC-RZ(controls,        │  RZZ      → CX · MC-RZ · CX
//! │   CRX/CRY/CRZ)   │      second) · CX            │
//! ├──────────────────┴──────────────────────────────┤
//! │                    mc_su2                        │  MC-SU(2) = Vale et al. 2024
//! ├─────────────────────────────────────────────────┤
//! │              pauli (X/Y/Z/CX/CY/CZ/CCX)          │  Y = SDG·MCX·S
//! │                                                 │  Z =  H ·MCX·H
//! ├─────────────────────────────────────────────────┤
//! │                     mcx                          │  Multi-controlled X (Toffoli)
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## 1. MCX — Multi-controlled X (Toffoli)
//!
//! The foundational primitive. All other decompositions eventually bottom out
//! in MCX gates. Available algorithms:
//!
//! | Function | Algorithm | Reference |
//! |---|---|---|
//! | `decompose_mcx_small` | Trivial (≤2 controls — X, CX, CCX) | |
//! | `decompose_mcx_no_aux` | No-ancilla exact decomposition | |
//! | `decompose_mcx_n_clean` | Clean V-chain (≥3 controls, `n−2` ancillas in \|0⟩) | Saeedi & Pedram 2013 |
//! | `decompose_mcx_n_dirty` | Dirty V-chain (≥3 controls, `n−2` borrowed ancillas) | Saeedi & Pedram 2013 |
//! | `decompose_mcx_1_clean_b95` | One clean ancilla, recursive | Barenco et al. 1995 |
//! | `decompose_mcx_1_clean_kg24` | One conditionally-clean ancilla | Khattar & Gidney 2024 §5.1 |
//! | `decompose_mcx_1_dirty` | One conditionally-dirty ancilla | Khattar & Gidney 2024 §5.2 |
//! | `decompose_mcx_2_clean` | Two conditionally-clean ancillas | Khattar & Gidney 2024 §5.3 |
//! | `decompose_mcx_2_dirty` | Two conditionally-dirty ancillas | Khattar & Gidney 2024 §5.4 |
//!
//! ## 2. MC-SU(2) — Multi-controlled SU(2) rotations
//!
//! Single-qubit rotations around X, Y, or Z with multiple controls,
//! decomposed via the linear construction of Vale et al., *Circuit
//! Decomposition of Multicontrolled Special Unitary Single-Qubit Gates*,
//! IEEE Trans. Quantum Eng. 5 (2024), [arXiv:2302.06377].
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_mc_su2_no_aux` | No ancilla (linear in control count) |
//! | `decompose_mc_su2_n_clean` | Clean accumulator (≥2 controls, `n−1` ancillas) |
//!
//! Type `Su2RotationAxis` selects the rotation axis (X, Y, or Z).
//!
//! ## 3. Rotation — Multi-controlled standard rotations
//!
//! Accepts `RX`, `RY`, `RZ` and their intrinsically-controlled forms
//! `CRX`, `CRY`, `CRZ`. Maps the gate to an `Su2RotationAxis` and delegates
//! to [`mc_su2`]. Controls must already be flattened (i.e. CRZ with 1
//! additional control → 2 controls total).
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_rotation_no_aux` | No ancilla |
//! | `decompose_rotation_n_clean` | Clean ancilla |
//!
//! ## 4. Pauli — Multi-controlled Pauli gates
//!
//! Synthesizes `X`, `Y`, `Z` and their controlled forms `CX`, `CY`, `CZ`,
//! `CCX` by conjugating exact MCX decompositions with single-qubit basis
//! changes:
//!
//! ```text
//! MCY = SDG(target) · MCX · S(target)
//! MCZ =  H(target)  · MCX · H(target)
//! ```
//!
//! Exposes one function for each MCX algorithm variant (e.g.
//! `decompose_pauli_no_aux`, `decompose_pauli_n_clean`,
//! `decompose_pauli_1_clean_b95`, etc.).
//!
//! ## 5. RZZ — Multi-controlled RZZ
//!
//! The canonical building block for two-qubit Pauli interaction rotations:
//!
//! ```text
//! MC-RZZ(θ, controls, a, b) = CX(a, b) · MC-RZ(θ, controls, b) · CX(a, b)
//! ```
//!
//! The flanking `CX` gates are unconditional — they compute the parity bit
//! regardless of control states. Only the central `RZ` rotation is controlled.
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_mc_rzz_no_aux` | No ancilla |
//! | `decompose_mc_rzz_n_clean` | Clean ancilla (delegated to MCRZ) |
//!
//! ## 6. Pauli Rotation — Multi-controlled two-qubit Pauli rotations
//!
//! Reduces `RXX`, `RYY`, `RZZ`, and `RZX` to [`rzz`] via basis changes:
//!
//! ```text
//! RXX(a,b) = H(a)·H(b) · RZZ(a,b) · H(a)·H(b)
//! RYY(a,b) = RX(π/2)(a)·RX(π/2)(b) · RZZ(a,b) · RX(−π/2)(a)·RX(−π/2)(b)
//! RZX(a,b) = H(b) · RZZ(a,b) · H(b)          (a = Z-axis, b = X-axis)
//! RZZ      = (identity — delegates directly)
//! ```
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_pauli_rotation_no_aux` | No ancilla |
//! | `decompose_pauli_rotation_n_clean` | Clean ancilla |
//!
//! ## 7. Phase — Multi-controlled phase gates
//!
//! Synthesizes `S`, `SDG`, `T`, `TDG`, and parameterized `Phase` using a
//! recursive projector decomposition. The key identity:
//!
//! ```text
//! Phase(θ) ≠ RZ(θ)   (they differ by a scalar exp(iθ/2))
//! ```
//!
//! That scalar becomes an observable conditional phase on the controls.
//! The decomposition recursively emits `Phase(θ/2)` on `n−1` controls and
//! `MCRZ(θ)` on all `n` controls, bottoming out in [`rotation`].
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_phase_no_aux` | No ancilla |
//! | `decompose_phase_n_clean` | Clean ancilla |
//!
//! ## 8. Unitary — Multi-controlled U(θ, φ, λ)
//!
//! The most general single-qubit unitary:
//!
//! ```text
//! U(θ, φ, λ) = exp(i(φ+λ)/2) · RZ(φ) · RY(θ) · RZ(λ)
//! ```
//!
//! With controls, the scalar factor is emitted as an observable conditional
//! phase via [`phase`], and the three Euler rotations are decomposed via
//! [`rotation`].
//!
//! | Function | Variant |
//! |---|---|
//! | `decompose_unitary_no_aux` | No ancilla |
//! | `decompose_unitary_n_clean` | Clean ancilla |
//!
//! # Ancilla contracts
//!
//! Every decomposition that consumes ancillary qubits declares its contract
//! explicitly:
//!
//! - **Clean ancilla**: must enter in `|0⟩` and is guaranteed restored to
//!   `|0⟩`. Extra ancillas beyond the consumed prefix are ignored.
//! - **Dirty ancilla**: may enter in any unknown state and is guaranteed
//!   restored exactly. Extra ancillas beyond the consumed prefix are ignored.
//!
//! Callers must provide ancillas that are distinct from all controls and
//! targets. Duplicate-qubit errors are returned as
//! [`CompilerError::TransformFailed`].

pub mod mc_su2;
pub mod mcx;
pub mod pauli;
pub mod pauli_rotation;
pub mod phase;
pub mod rotation;
pub mod rzz;
pub mod unitary;

// ── mc_su2 ──
pub use mc_su2::{Su2RotationAxis, decompose_mc_su2_n_clean, decompose_mc_su2_no_aux};

// ── mcx ──
pub use mcx::{
    decompose_mcx_1_clean_b95, decompose_mcx_1_clean_kg24, decompose_mcx_1_dirty,
    decompose_mcx_2_clean, decompose_mcx_2_dirty, decompose_mcx_n_clean, decompose_mcx_n_dirty,
    decompose_mcx_no_aux, decompose_mcx_small,
};

// ── rotation ──
pub use rotation::{decompose_rotation_n_clean, decompose_rotation_no_aux};

// ── pauli ──
pub use pauli::{
    decompose_pauli_1_clean_b95, decompose_pauli_1_clean_kg24, decompose_pauli_1_dirty,
    decompose_pauli_2_clean, decompose_pauli_2_dirty, decompose_pauli_n_clean,
    decompose_pauli_n_dirty, decompose_pauli_no_aux, decompose_pauli_small,
};

// ── pauli_rotation ──
pub use pauli_rotation::{decompose_pauli_rotation_n_clean, decompose_pauli_rotation_no_aux};

// ── phase ──
pub use phase::{decompose_phase_n_clean, decompose_phase_no_aux};

// ── rzz ──
pub use rzz::{decompose_mc_rzz_n_clean, decompose_mc_rzz_no_aux};

// ── unitary ──
pub use unitary::{decompose_unitary_n_clean, decompose_unitary_no_aux};

#[cfg(test)]
mod pauli_rotation_test;
#[cfg(test)]
mod pauli_test;
#[cfg(test)]
mod phase_test;
#[cfg(test)]
mod rotation_test;
#[cfg(test)]
mod rzz_test;
#[cfg(test)]
mod unitary_test;
