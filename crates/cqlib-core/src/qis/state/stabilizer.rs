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

//! Stabilizer state simulator using the Aaronson-Gottesman tableau algorithm.
//!
//! Implements efficient simulation of Clifford circuits using the symplectic tableau
//! representation from:
//!
//! > Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits",
//! > Phys. Rev. A 70, 052328 (2004).
//!
//! Non-Clifford gates (T, TDG, RX, RZ, U, …) are rejected immediately with
//! [`QisError::NonCliffordGate`] — use [`crate::qis::Statevector`] for universal circuits.
//!
//! # Supported Gates
//!
//! | Gate | Symbol | Kind |
//! |------|--------|------|
//! | Hadamard | H | single-qubit |
//! | Phase | S | single-qubit |
//! | Phase-dagger | S† / SDG | single-qubit |
//! | Pauli-X | X | single-qubit |
//! | Pauli-Y | Y | single-qubit |
//! | Pauli-Z | Z | single-qubit |
//! | √X | X2P / SX | single-qubit |
//! | √X† | X2M / SXdg | single-qubit |
//! | √Y | Y2P / SY | single-qubit |
//! | √Y† | Y2M / SYdg | single-qubit |
//! | Controlled-NOT | CX / CNOT | two-qubit |
//! | Controlled-Y | CY | two-qubit |
//! | Controlled-Z | CZ | two-qubit |
//! | SWAP | SWAP | two-qubit |
//!
//! # Performance
//!
//! | Operation | Complexity | Notes |
//! |-----------|-----------|-------|
//! | Single-qubit gate | O(n) | one pass over 2n rows |
//! | Two-qubit gate | O(n) | one pass over 2n rows |
//! | Measurement (random) | O(n²/w) | SIMD-accelerated rowsum, w = 64 |
//! | Measurement (deterministic) | O(n²/w) | scratch-row accumulation via rowsum |
//! | Memory | O(n²/w) | ~50 MB for n = 10 000 |
//!
//! Multi-shot sampling uses Rayon for parallel execution; each shot gets an
//! independent clone of the state.
//!
//! # Memory Layout
//!
//! The tableau uses a single flat `Box<[u64]>` for cache efficiency:
//! - `2n + 1` rows total: rows `0..n` = destabilizers, rows `n..2n` = stabilizers,
//!   row `2n` = pre-allocated scratch for deterministic measurement.
//! - Each row: `row_len` u64 words for the X-block, then `row_len` words for the Z-block:
//!   `[x₀..xₙ | z₀..zₙ]` packed into `row_len` words each.
//! - `row_len` is padded to a multiple of 8 (512 bits) for AVX-512 alignment.
//! - Phases stored separately in `phases: Box<[u8]>` as `0` (= +1) or `2` (= −1).
//!
//! # Bit Addressing
//!
//! Qubit `q` maps to word `q / 64` and bit `q % 64` within both the X and Z blocks
//! of every row.
//!
//! # Examples
//!
//! **Bell state preparation and measurement:**
//! ```rust
//! use cqlib_core::qis::StabilizerState;
//!
//! let mut s = StabilizerState::new(2);
//! s.apply_h(0).unwrap();
//! s.apply_cx(0, 1).unwrap(); // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
//!
//! // Outcomes are always correlated
//! let shots = s.sample_shots(500);
//! assert!(shots.iter().all(|v| v[0] == v[1]));
//! ```
//!
//! **GHZ state — 1000 qubits:**
//! ```rust
//! use cqlib_core::qis::StabilizerState;
//!
//! let n = 1000;
//! let mut s = StabilizerState::new(n);
//! s.apply_h(0).unwrap();
//! for q in 0..n - 1 {
//!     s.apply_cx(q, q + 1).unwrap();
//! }
//! // All qubits measure identically
//! let mut copy = s.clone();
//! let result = copy.measure_all();
//! assert!(result.iter().all(|&b| b == result[0]));
//! ```
//!
//! **Deterministic measurement on |0⟩:**
//! ```rust
//! use cqlib_core::qis::StabilizerState;
//!
//! let mut s = StabilizerState::new(3);
//! // |000⟩ state — all measurements deterministically 0
//! assert_eq!(s.measure(0).unwrap(), false);
//! assert_eq!(s.measure(1).unwrap(), false);
//! assert_eq!(s.measure(2).unwrap(), false);
//! ```
//!
//! **From a Clifford circuit:**
//! ```rust
//! use cqlib_core::circuit::circuit_impl::Circuit;
//! use cqlib_core::qis::StabilizerState;
//!
//! let mut c = Circuit::new(2);
//! c.h(0.into());
//! c.cx(0.into(), 1.into());
//! let stab = StabilizerState::from_circuit(&c).unwrap();
//! let stabilizers = stab.get_stabilizers();
//! // Bell state is stabilized by +XX and +ZZ
//! assert_eq!(stabilizers.len(), 2);
//! ```
//!
//! **Non-Clifford gate returns an error:**
//! ```rust
//! use cqlib_core::circuit::circuit_impl::Circuit;
//! use cqlib_core::qis::{StabilizerState, QisError};
//!
//! let mut c = Circuit::new(1);
//! c.t(0.into()); // T gate is not Clifford
//! let result = StabilizerState::from_circuit(&c);
//! assert!(matches!(result, Err(QisError::NonCliffordGate(_))));
//! ```

use crate::circuit::circuit_impl::Circuit;
use crate::circuit::circuit_param::CircuitParam;
use crate::circuit::error::CircuitError;
use crate::circuit::gate::{Instruction, StandardGate};
use crate::qis::error::QisError;
use crate::qis::pauli::{Pauli, PauliString, Phase};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ptr::NonNull;

// ── Aligned memory buffer ─────────────────────────────────────────────────────

/// A heap-allocated buffer of `u64` words with 64-byte alignment.
///
/// The standard `Vec<u64>` only guarantees 8-byte alignment, which forces SIMD
/// code to use unaligned load/store instructions (`loadu`/`storeu`) and risks
/// cache-line split penalties. This wrapper allocates with a 64-byte layout so
/// that AVX2/SSE2 aligned loads (`load`/`store`) can be used safely.
struct AlignedBuffer {
    ptr: NonNull<u64>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Allocates `len` zero-initialised `u64` words at 64-byte alignment.
    fn new_zeroed(len: usize) -> Self {
        let size = len * size_of::<u64>();
        // SAFETY: size > 0 (caller ensures len > 0), align is a power of two.
        let layout = Layout::from_size_align(size, 64).expect("valid layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut u64).unwrap_or_else(|| handle_alloc_error(layout));
        AlignedBuffer { ptr, len, layout }
    }

    fn as_slice(&self) -> &[u64] {
        // SAFETY: ptr is valid for `len` u64 words, allocated by `new_zeroed`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u64] {
        // SAFETY: ptr is valid, unique (we hold &mut self).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated with this layout; not yet freed.
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, self.layout) }
    }
}

// SAFETY: AlignedBuffer owns its allocation uniquely; no shared mutable state.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl std::ops::Deref for AlignedBuffer {
    type Target = [u64];
    #[inline(always)]
    fn deref(&self) -> &[u64] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [u64] {
        self.as_mut_slice()
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuffer(len={})", self.len)
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let new_buf = AlignedBuffer::new_zeroed(self.len);
        // SAFETY: both allocations are valid for `self.len` u64 words and non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_buf.ptr.as_ptr(), self.len);
        }
        new_buf
    }
}

/// Stabilizer state simulator based on the Aaronson-Gottesman symplectic tableau.
///
/// Represents an n-qubit stabilizer state using 2n generator rows, each composed
/// of an X-block and Z-block of packed `u64` words, plus a phase per row.
#[derive(Debug)]
pub struct StabilizerState {
    /// Number of qubits.
    pub n: usize,
    /// Number of `u64` words per block (X-block or Z-block) per row.
    /// Padded to a multiple of 8 for 512-bit SIMD alignment.
    row_len: usize,
    /// Flat tableau storage: `(2n+1)` rows × `2 * row_len` words, 64-byte aligned.
    /// Row `i` occupies `tableau[i * 2 * row_len .. (i+1) * 2 * row_len]`.
    /// Within a row: first `row_len` words = X-block, next `row_len` words = Z-block.
    /// Row `2n` is the pre-allocated scratch row for deterministic measurement.
    tableau: AlignedBuffer,
    /// Phase for each of the `2n+1` rows: `0` means `+1`, `2` means `-1`.
    phases: Box<[u8]>,
    /// RNG for random measurement outcomes.
    rng: SmallRng,
}

// ── SIMD XOR helpers ─────────────────────────────────────────────────────────
//
// `xor_rows_dispatch` XORs `count` u64 words from `src_base` into `dst_base`
// within `tableau`, dispatching to the best available SIMD tier at runtime.
// `count` is always a multiple of 16 (= 2 × row_len, row_len ≥ 8), so no tail
// handling is required for any SIMD width up to 512 bits.
//
// SAFETY contract shared by all variants:
//   • `dst_base + count ≤ tableau.len()` and `src_base + count ≤ tableau.len()`
//   • `[dst_base, dst_base+count)` and `[src_base, src_base+count)` do not overlap
//     (guaranteed by the caller via distinct row indices h ≠ i).

/// AVX2 path: 256-bit XOR, processing 4 u64 words per iteration.
///
/// Requires 32-byte alignment. Our `AlignedBuffer` provides 64-byte alignment,
/// so aligned loads (`_mm256_load_si256`) are safe for every row start.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_rows_avx2(tableau: &mut [u64], dst_base: usize, src_base: usize, count: usize) {
    use std::arch::x86_64::{__m256i, _mm256_load_si256, _mm256_store_si256, _mm256_xor_si256};
    // SAFETY: avx2 feature enabled; AlignedBuffer guarantees 64-byte alignment
    // (≥ required 32-byte AVX2 alignment); caller guarantees bounds + non-overlap.
    unsafe {
        let ptr = tableau.as_mut_ptr();
        let dst = ptr.add(dst_base) as *mut __m256i;
        let src = ptr.add(src_base) as *const __m256i;
        let chunks = count / 4; // 4 u64 per __m256i (256 bits)
        for i in 0..chunks {
            let a = _mm256_load_si256(dst.add(i));
            let b = _mm256_load_si256(src.add(i));
            _mm256_store_si256(dst.add(i), _mm256_xor_si256(a, b));
        }
    }
}

/// SSE2 path: 128-bit XOR, processing 2 u64 words per iteration.
///
/// Requires 16-byte alignment. Our `AlignedBuffer` provides 64-byte alignment.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn xor_rows_sse2(tableau: &mut [u64], dst_base: usize, src_base: usize, count: usize) {
    use std::arch::x86_64::{__m128i, _mm_load_si128, _mm_store_si128, _mm_xor_si128};
    // SAFETY: sse2 feature enabled; AlignedBuffer guarantees 64-byte alignment
    // (≥ required 16-byte SSE2 alignment); caller guarantees bounds + non-overlap.
    unsafe {
        let ptr = tableau.as_mut_ptr();
        let dst = ptr.add(dst_base) as *mut __m128i;
        let src = ptr.add(src_base) as *const __m128i;
        let chunks = count / 2; // 2 u64 per __m128i (128 bits)
        for i in 0..chunks {
            let a = _mm_load_si128(dst.add(i));
            let b = _mm_load_si128(src.add(i));
            _mm_store_si128(dst.add(i), _mm_xor_si128(a, b));
        }
    }
}

/// NEON path: 128-bit XOR, processing 2 u64 words per iteration.
/// NEON is mandatory on aarch64; no runtime feature detection required.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_rows_neon(tableau: &mut [u64], dst_base: usize, src_base: usize, count: usize) {
    use std::arch::aarch64::{veorq_u64, vld1q_u64, vst1q_u64};
    // SAFETY: NEON is mandatory on aarch64; caller guarantees bounds and non-overlap.
    unsafe {
        let ptr = tableau.as_mut_ptr();
        let dst = ptr.add(dst_base);
        let src = ptr.add(src_base) as *const u64;
        let chunks = count / 2; // 2 u64 per uint64x2_t (128 bits)
        for i in 0..chunks {
            let a = vld1q_u64(dst.add(i * 2) as *const u64);
            let b = vld1q_u64(src.add(i * 2));
            vst1q_u64(dst.add(i * 2), veorq_u64(a, b));
        }
    }
}

/// Scalar fallback: word loop that the compiler auto-vectorizes at -O2.
/// Used on unsupported architectures or when SIMD features are unavailable.
#[inline]
fn xor_rows_scalar(tableau: &mut [u64], dst_base: usize, src_base: usize, count: usize) {
    // SAFETY: use raw pointer to read src while dst is mutably borrowed from the
    // same slice.  Non-overlap is guaranteed by the caller.
    let ptr = tableau.as_mut_ptr();
    for w in 0..count {
        // SAFETY: dst_base+w < tableau.len() and src_base+w < tableau.len().
        unsafe { *ptr.add(dst_base + w) ^= *ptr.add(src_base + w) };
    }
}

/// Runtime-dispatched XOR of `count` u64 words from `src_base` into `dst_base`.
///
/// # Safety
/// * `dst_base + count ≤ tableau.len()` and `src_base + count ≤ tableau.len()`
/// * `[dst_base, dst_base+count)` and `[src_base, src_base+count)` must not overlap.
#[inline]
#[allow(unreachable_code)] // scalar tail unreachable on aarch64 (always takes NEON path)
unsafe fn xor_rows_dispatch(tableau: &mut [u64], dst_base: usize, src_base: usize, count: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed; bounds + non-overlap guaranteed by caller.
            return unsafe { xor_rows_avx2(tableau, dst_base, src_base, count) };
        }
        if is_x86_feature_detected!("sse2") {
            // SAFETY: sse2 confirmed; bounds + non-overlap guaranteed by caller.
            return unsafe { xor_rows_sse2(tableau, dst_base, src_base, count) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON mandatory on aarch64; bounds + non-overlap guaranteed by caller.
        return unsafe { xor_rows_neon(tableau, dst_base, src_base, count) };
    }
    xor_rows_scalar(tableau, dst_base, src_base, count);
}

impl StabilizerState {
    /// Creates a new stabilizer state initialized to |0...0⟩.
    ///
    /// In the Aaronson-Gottesman convention the initial generators are:
    /// - Destabilizer `i` (row `i`):   X on qubit `i`, identity elsewhere, phase +1.
    /// - Stabilizer  `i` (row `n+i`):  Z on qubit `i`, identity elsewhere, phase +1.
    ///
    /// # Arguments
    /// * `n` — number of qubits
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "StabilizerState requires at least 1 qubit");

        // Pad row_len to the next multiple of 8 u64 words (= 512 bits).
        let words_needed = n.div_ceil(64);
        let row_len = ((words_needed + 7) & !7).max(8);

        // Allocate 2n+1 rows: rows 0..n = destabilizers, n..2n = stabilizers,
        // row 2n = persistent scratch (used by deterministic measurement).
        let total_rows = 2 * n + 1;
        let total_words = total_rows * 2 * row_len;
        let mut tableau = AlignedBuffer::new_zeroed(total_words);
        let phases = vec![0u8; total_rows].into_boxed_slice();

        // Initialise destabilizers: row i has X bit set at column i.
        for i in 0..n {
            let word = i / 64;
            let bit = i % 64;
            tableau[Self::x_offset(i, i, row_len, word)] |= 1u64 << bit;
        }

        // Initialise stabilizers: row n+i has Z bit set at column i.
        for i in 0..n {
            let word = i / 64;
            let bit = i % 64;
            tableau[Self::z_offset(n + i, i, row_len, word)] |= 1u64 << bit;
        }

        StabilizerState {
            n,
            row_len,
            tableau,
            phases,
            rng: SmallRng::from_os_rng(),
        }
    }

    /// Index of the persistent scratch row in the tableau (= `2n`).
    #[inline(always)]
    fn scratch_row(&self) -> usize {
        2 * self.n
    }

    /// Zeroes the scratch row's bit-columns and phase, ready for a new accumulation.
    fn clear_scratch(&mut self) {
        let s = self.scratch_row();
        let base = Self::row_base(s, self.row_len);
        for w in 0..2 * self.row_len {
            self.tableau[base + w] = 0;
        }
        self.phases[s] = 0;
    }

    // ── Row access helpers ────────────────────────────────────────────────────

    /// Base byte offset (in u64 words) of row `row` in the tableau.
    #[inline(always)]
    fn row_base(row: usize, row_len: usize) -> usize {
        row * 2 * row_len
    }

    /// Flat index into `tableau` for the `word`-th u64 in the X-block of `row`,
    /// where `col_qubit` determines which X-block column we're accessing.
    /// (The `word` parameter is `col_qubit / 64`.)
    #[inline(always)]
    fn x_offset(row: usize, _col_qubit: usize, row_len: usize, word: usize) -> usize {
        Self::row_base(row, row_len) + word
    }

    /// Flat index into `tableau` for the `word`-th u64 in the Z-block of `row`.
    #[inline(always)]
    fn z_offset(row: usize, _col_qubit: usize, row_len: usize, word: usize) -> usize {
        Self::row_base(row, row_len) + row_len + word
    }

    // ── Per-qubit bit access ──────────────────────────────────────────────────

    /// Returns the X-bit of `row` at qubit column `q`.
    #[inline(always)]
    pub(crate) fn x_bit(&self, row: usize, q: usize) -> bool {
        let base = Self::row_base(row, self.row_len);
        let word = q / 64;
        let bit = q % 64;
        (self.tableau[base + word] >> bit) & 1 == 1
    }

    /// Returns the Z-bit of `row` at qubit column `q`.
    #[inline(always)]
    pub(crate) fn z_bit(&self, row: usize, q: usize) -> bool {
        let base = Self::row_base(row, self.row_len);
        let word = q / 64;
        let bit = q % 64;
        (self.tableau[base + self.row_len + word] >> bit) & 1 == 1
    }

    /// Sets or clears the X-bit of `row` at qubit column `q`.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn set_x_bit(&mut self, row: usize, q: usize, val: bool) {
        let base = Self::row_base(row, self.row_len);
        let word = q / 64;
        let bit = q % 64;
        if val {
            self.tableau[base + word] |= 1u64 << bit;
        } else {
            self.tableau[base + word] &= !(1u64 << bit);
        }
    }

    /// Sets or clears the Z-bit of `row` at qubit column `q`.
    #[inline(always)]
    pub(crate) fn set_z_bit(&mut self, row: usize, q: usize, val: bool) {
        let base = Self::row_base(row, self.row_len);
        let word = q / 64;
        let bit = q % 64;
        if val {
            self.tableau[base + self.row_len + word] |= 1u64 << bit;
        } else {
            self.tableau[base + self.row_len + word] &= !(1u64 << bit);
        }
    }

    /// Returns the phase of row `row` as 0 (+1) or 2 (-1).
    #[inline(always)]
    pub(crate) fn phase(&self, row: usize) -> u8 {
        self.phases[row]
    }

    /// Sets the phase of row `row` to `val` (0 or 2).
    #[inline(always)]
    pub(crate) fn set_phase(&mut self, row: usize, val: u8) {
        self.phases[row] = val & 2;
    }

    // ── Validation ────────────────────────────────────────────────────────────

    pub(crate) fn validate_qubit(&self, q: usize) -> Result<(), QisError> {
        if q >= self.n {
            return Err(QisError::IndexOutOfBounds {
                index: q,
                max: self.n.saturating_sub(1),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_two_qubits(&self, q0: usize, q1: usize) -> Result<(), QisError> {
        self.validate_qubit(q0)?;
        self.validate_qubit(q1)?;
        if q0 == q1 {
            return Err(QisError::InvalidParameterValue(
                "Two-qubit gate requires distinct qubit indices".to_string(),
            ));
        }
        Ok(())
    }

    // ── Two-qubit Clifford gates ──────────────────────────────────────────────

    /// Applies the CNOT (CX) gate with `control` as control and `target` as target.
    ///
    /// Aaronson-Gottesman rule (Eq. 11) for each row `i`:
    /// - `phase[i] ^= 2 * x[i][c] * z[i][t] * (x[i][t] ^ z[i][c] ^ 1)`
    /// - `x[i][t] ^= x[i][c]`
    /// - `z[i][c] ^= z[i][t]`
    pub fn apply_cx(&mut self, control: usize, target: usize) -> Result<(), QisError> {
        self.validate_two_qubits(control, target)?;
        let c_word = control / 64;
        let c_mask = 1u64 << (control % 64);
        let t_word = target / 64;
        let t_mask = 1u64 << (target % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let xc = (self.tableau[base + c_word] & c_mask) != 0;
            let xt = (self.tableau[base + t_word] & t_mask) != 0;
            let zc = (self.tableau[base + rl + c_word] & c_mask) != 0;
            let zt = (self.tableau[base + rl + t_word] & t_mask) != 0;
            if xc && zt && (xt ^ zc ^ true) {
                self.phases[row] ^= 2;
            }
            if xc {
                self.tableau[base + t_word] ^= t_mask;
            }
            if zt {
                self.tableau[base + rl + c_word] ^= c_mask;
            }
        }
        Ok(())
    }

    /// Applies the SWAP gate between qubits `q0` and `q1`.
    ///
    /// Native rule: for each row `i`, swap (x[i][q0], z[i][q0]) with (x[i][q1], z[i][q1]).
    /// No phase change — SWAP maps Paulis without sign.
    pub fn apply_swap(&mut self, q0: usize, q1: usize) -> Result<(), QisError> {
        self.validate_two_qubits(q0, q1)?;
        let w0 = q0 / 64;
        let m0 = 1u64 << (q0 % 64);
        let w1 = q1 / 64;
        let m1 = 1u64 << (q1 % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            // Read bits
            let x0 = (self.tableau[base + w0] & m0) != 0;
            let z0 = (self.tableau[base + rl + w0] & m0) != 0;
            let x1 = (self.tableau[base + w1] & m1) != 0;
            let z1 = (self.tableau[base + rl + w1] & m1) != 0;
            // Write swapped bits
            if x1 {
                self.tableau[base + w0] |= m0;
            } else {
                self.tableau[base + w0] &= !m0;
            }
            if z1 {
                self.tableau[base + rl + w0] |= m0;
            } else {
                self.tableau[base + rl + w0] &= !m0;
            }
            if x0 {
                self.tableau[base + w1] |= m1;
            } else {
                self.tableau[base + w1] &= !m1;
            }
            if z0 {
                self.tableau[base + rl + w1] |= m1;
            } else {
                self.tableau[base + rl + w1] &= !m1;
            }
        }
        Ok(())
    }

    /// Applies the CZ gate (controlled-Z) between `q0` and `q1`.
    ///
    /// Implemented as (I⊗H)·CNOT·(I⊗H) inlined per-row — no heap allocations.
    pub fn apply_cz(&mut self, q0: usize, q1: usize) -> Result<(), QisError> {
        self.validate_two_qubits(q0, q1)?;
        let w0 = q0 / 64;
        let m0 = 1u64 << (q0 % 64);
        let w1 = q1 / 64;
        let m1 = 1u64 << (q1 % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            // H on q1: swap x1 ↔ z1, phase if both set
            let x1 = (self.tableau[base + w1] & m1) != 0;
            let z1 = (self.tableau[base + rl + w1] & m1) != 0;
            if x1 && z1 {
                self.phases[row] ^= 2;
            }
            if z1 {
                self.tableau[base + w1] |= m1;
            } else {
                self.tableau[base + w1] &= !m1;
            }
            if x1 {
                self.tableau[base + rl + w1] |= m1;
            } else {
                self.tableau[base + rl + w1] &= !m1;
            }

            // CNOT(q0, q1)
            let xc = (self.tableau[base + w0] & m0) != 0;
            let xt = (self.tableau[base + w1] & m1) != 0;
            let zc = (self.tableau[base + rl + w0] & m0) != 0;
            let zt = (self.tableau[base + rl + w1] & m1) != 0;
            if xc && zt && (xt ^ zc ^ true) {
                self.phases[row] ^= 2;
            }
            if xc {
                self.tableau[base + w1] ^= m1;
            }
            if zt {
                self.tableau[base + rl + w0] ^= m0;
            }

            // H on q1 again
            let x1 = (self.tableau[base + w1] & m1) != 0;
            let z1 = (self.tableau[base + rl + w1] & m1) != 0;
            if x1 && z1 {
                self.phases[row] ^= 2;
            }
            if z1 {
                self.tableau[base + w1] |= m1;
            } else {
                self.tableau[base + w1] &= !m1;
            }
            if x1 {
                self.tableau[base + rl + w1] |= m1;
            } else {
                self.tableau[base + rl + w1] &= !m1;
            }
        }
        Ok(())
    }

    /// Applies the CY gate (controlled-Y) between `control` and `target`.
    ///
    /// Implemented as (I⊗S†)·CNOT·(I⊗S) inlined per-row.
    pub fn apply_cy(&mut self, control: usize, target: usize) -> Result<(), QisError> {
        self.validate_two_qubits(control, target)?;
        let cw = control / 64;
        let cm = 1u64 << (control % 64);
        let tw = target / 64;
        let tm = 1u64 << (target % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            // S on target: phase if x&z, then z ^= x
            let x = (self.tableau[base + tw] & tm) != 0;
            let z = (self.tableau[base + rl + tw] & tm) != 0;
            if x && z {
                self.phases[row] ^= 2;
            }
            if x {
                self.tableau[base + rl + tw] ^= tm;
            }

            // CNOT(control, target)
            let xc = (self.tableau[base + cw] & cm) != 0;
            let xt = (self.tableau[base + tw] & tm) != 0;
            let zc = (self.tableau[base + rl + cw] & cm) != 0;
            let zt = (self.tableau[base + rl + tw] & tm) != 0;
            if xc && zt && (xt ^ zc ^ true) {
                self.phases[row] ^= 2;
            }
            if xc {
                self.tableau[base + tw] ^= tm;
            }
            if zt {
                self.tableau[base + rl + cw] ^= cm;
            }

            // S† on target: phase if x&!z, then z ^= x
            let x = (self.tableau[base + tw] & tm) != 0;
            let z = (self.tableau[base + rl + tw] & tm) != 0;
            if x && !z {
                self.phases[row] ^= 2;
            }
            if x {
                self.tableau[base + rl + tw] ^= tm;
            }
        }
        Ok(())
    }

    /// Returns the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    // ── Single-qubit Clifford gates ───────────────────────────────────────────

    /// Applies the Hadamard gate to qubit `qubit`.
    ///
    /// Aaronson-Gottesman update rule for each row `i` (0..2n):
    /// - `phase[i] ^= 2 * x[i][qubit] & z[i][qubit]`  (adds 2 mod 4 when Y, i.e. sign flip)
    /// - swap `x[i][qubit]` and `z[i][qubit]`
    pub fn apply_h(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x && z {
                self.phases[row] ^= 2;
            }
            // Swap x and z bits.
            if x {
                self.tableau[base + rl + word] |= mask;
            } else {
                self.tableau[base + rl + word] &= !mask;
            }
            if z {
                self.tableau[base + word] |= mask;
            } else {
                self.tableau[base + word] &= !mask;
            }
        }
        Ok(())
    }

    /// Applies the S gate (phase gate, √Z) to qubit `qubit`.
    ///
    /// Aaronson-Gottesman rule for each row `i`:
    /// - `phase[i] ^= 2 * x[i][qubit] * z[i][qubit]`
    /// - `z[i][qubit] ^= x[i][qubit]`
    ///
    /// Effect: X→Y, Y→-X, Z→Z.
    pub fn apply_s(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x && z {
                self.phases[row] ^= 2;
            }
            if x {
                // z ^= x (x=1): flip z bit
                self.tableau[base + rl + word] ^= mask;
            }
        }
        Ok(())
    }

    /// Applies the S† gate (S-dagger, inverse phase gate) to qubit `qubit`.
    ///
    /// Native rule for each row `i` (derived from S†: X→-Y, Y→X, Z→Z):
    /// - `phase[i] ^= 2 * x[i][qubit] * (1 - z[i][qubit])`  (flip sign when X but not yet Y)
    /// - `z[i][qubit] ^= x[i][qubit]`
    pub fn apply_sdg(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x && !z {
                self.phases[row] ^= 2;
            }
            if x {
                self.tableau[base + rl + word] ^= mask;
            }
        }
        Ok(())
    }

    /// Applies the Pauli-X (bit-flip) gate to qubit `qubit`.
    ///
    /// Effect: X→X, Y→-Y, Z→-Z.
    /// Rule per row `i`: `phase[i] ^= 2 * z[i][qubit]`. No bit changes.
    pub fn apply_x(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            if (self.tableau[base + rl + word] & mask) != 0 {
                self.phases[row] ^= 2;
            }
        }
        Ok(())
    }

    /// Applies the Pauli-Z (phase-flip) gate to qubit `qubit`.
    ///
    /// Effect: X→-X, Y→-Y, Z→Z.
    /// Rule per row `i`: `phase[i] ^= 2 * x[i][qubit]`. No bit changes.
    pub fn apply_z(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            if (self.tableau[base + word] & mask) != 0 {
                self.phases[row] ^= 2;
            }
        }
        Ok(())
    }

    /// Applies the Pauli-Y gate to qubit `qubit`.
    ///
    /// Effect: X→-X, Y→Y, Z→-Z.
    /// Rule per row `i`: `phase[i] ^= 2 * (x[i][qubit] XOR z[i][qubit])`. No bit changes.
    pub fn apply_y(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x ^ z {
                self.phases[row] ^= 2;
            }
        }
        Ok(())
    }

    /// Applies the √X gate (X2P, SX, Rx(π/2)) to qubit `qubit`.
    ///
    /// Conjugation: X→X, Y→Z, Z→−Y.
    /// Rule per row `i`:
    /// - `phase[i] ^= 2 * (!x[i][qubit] & z[i][qubit])` (flip for Z→−Y)
    /// - `x[i][qubit] ^= z[i][qubit]`
    pub fn apply_x2p(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if !x && z {
                self.phases[row] ^= 2;
            }
            // x ^= z
            if z {
                self.tableau[base + word] ^= mask;
            }
        }
        Ok(())
    }

    /// Applies the √X† gate (X2M, SXdg, Rx(−π/2)) to qubit `qubit`.
    ///
    /// Conjugation: X→X, Y→−Z, Z→Y.
    /// Rule per row `i`:
    /// - `phase[i] ^= 2 * (x[i][qubit] & z[i][qubit])` (flip for Y→−Z)
    /// - `x[i][qubit] ^= z[i][qubit]`
    pub fn apply_x2m(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x && z {
                self.phases[row] ^= 2;
            }
            if z {
                self.tableau[base + word] ^= mask;
            }
        }
        Ok(())
    }

    /// Applies the √Y gate (Y2P, SY, Ry(π/2)) to qubit `qubit`.
    ///
    /// Conjugation: X→Z, Y→Y, Z→−X.
    /// Rule per row `i`:
    /// - `phase[i] ^= 2 * (!x[i][qubit] & z[i][qubit])` (flip for Z→−X)
    /// - swap `x[i][qubit]` ↔ `z[i][qubit]`
    pub fn apply_y2p(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if !x && z {
                self.phases[row] ^= 2;
            }
            // swap x ↔ z
            if x {
                self.tableau[base + rl + word] |= mask;
            } else {
                self.tableau[base + rl + word] &= !mask;
            }
            if z {
                self.tableau[base + word] |= mask;
            } else {
                self.tableau[base + word] &= !mask;
            }
        }
        Ok(())
    }

    /// Applies the √Y† gate (Y2M, SYdg, Ry(−π/2)) to qubit `qubit`.
    ///
    /// Conjugation: X→−Z, Y→Y, Z→X.
    /// Rule per row `i`:
    /// - `phase[i] ^= 2 * (x[i][qubit] & !z[i][qubit])` (flip for X→−Z)
    /// - swap `x[i][qubit]` ↔ `z[i][qubit]`
    pub fn apply_y2m(&mut self, qubit: usize) -> Result<(), QisError> {
        self.validate_qubit(qubit)?;
        let word = qubit / 64;
        let mask = 1u64 << (qubit % 64);
        let rl = self.row_len;
        for row in 0..(2 * self.n) {
            let base = row * 2 * rl;
            let x = (self.tableau[base + word] & mask) != 0;
            let z = (self.tableau[base + rl + word] & mask) != 0;
            if x && !z {
                self.phases[row] ^= 2;
            }
            // swap x ↔ z
            if x {
                self.tableau[base + rl + word] |= mask;
            } else {
                self.tableau[base + rl + word] &= !mask;
            }
            if z {
                self.tableau[base + word] |= mask;
            } else {
                self.tableau[base + word] &= !mask;
            }
        }
        Ok(())
    }

    // ── Measurement ──────────────────────────────────────────────────────────

    /// Phase contribution from multiplying Pauli `P_h` (from h-row) by `P_i` (from i-row).
    ///
    /// Returns an integer exponent `e` such that `P_h * P_i = i^e * P_result`.
    /// Kept for reference/testing; the hot path uses `g_phase_word` instead.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn g_phase(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => z2 as i32 - x2 as i32,
            (true, false) => z2 as i32 * (2 * x2 as i32 - 1),
            (false, true) => x2 as i32 * (1 - 2 * z2 as i32),
        }
    }

    /// Accumulates the g_phase contribution of a single 64-bit word pair using
    /// word-level popcount — **2 `count_ones()` calls** per 64-bit word.
    ///
    /// Algebraic derivation:
    /// - `anti_comm = (xh & zi) XOR (xi & zh)` — positions where the two Paulis
    ///   anti-commute (anti-commuting ↔ g ≠ 0).
    /// - `pos = (xh & zi & (zh XOR xi)) | (!xh & zh & xi & !zi)` — positions
    ///   where g = +1 (the "forward cyclic" Pauli pairs X→Y, Y→Z, Z→X).
    /// - `sum = 2 · popcount(pos) − popcount(anti_comm)`
    ///   (since popcount(neg) = popcount(anti_comm) − popcount(pos),
    ///   and popcount(pos) − popcount(neg) = 2·popcount(pos) − popcount(anti_comm)).
    ///
    /// Verified against the scalar `g_phase` for all 9 single-qubit Pauli pairs.
    #[inline(always)]
    pub(crate) fn g_phase_word(xh: u64, zh: u64, xi: u64, zi: u64) -> i32 {
        // Positions where the Paulis anti-commute (g ≠ 0).
        let anti_comm = (xh & zi) ^ (xi & zh);
        // Positions where g = +1 (cyclic order X→Y, Y→Z, Z→X).
        let pos = (xh & zi & (zh ^ xi)) | (!xh & zh & xi & !zi);
        2 * pos.count_ones() as i32 - anti_comm.count_ones() as i32
    }

    /// Multiplies row `h` by row `i` in-place: `row_h := row_h * row_i`.
    ///
    /// Phase is computed via `g_phase_word` (word-level popcount, O(n/64) vs O(n)).
    /// Bit XOR is dispatched to the best available SIMD instruction set at runtime.
    fn rowsum(&mut self, h: usize, i: usize) {
        debug_assert_ne!(h, i, "rowsum requires distinct rows");

        let h_base = Self::row_base(h, self.row_len);
        let i_base = Self::row_base(i, self.row_len);
        let rl = self.row_len;

        // ── Phase accumulation (word-level popcount) ─────────────────────────
        let mut sum = self.phases[h] as i32 + self.phases[i] as i32;
        // Only the active words (0..n_words) contain qubit data; the padding words
        // in the range n_words..row_len are always zero and contribute 0 to the sum.
        let n_words = self.n.div_ceil(64);
        for w in 0..n_words {
            let xh = self.tableau[h_base + w];
            let zh = self.tableau[h_base + rl + w];
            let xi = self.tableau[i_base + w];
            let zi = self.tableau[i_base + rl + w];
            sum += Self::g_phase_word(xh, zh, xi, zi);
        }
        // New phase is sum mod 4, normalised to {0,1,2,3}.
        self.phases[h] = (((sum % 4) + 4) % 4) as u8;

        // ── XOR bits (SIMD-dispatched) ────────────────────────────────────────
        // SAFETY: h ≠ i guarantees [h_base, h_base+2*rl) and [i_base, i_base+2*rl)
        // are non-overlapping regions within the same 64-byte aligned allocation.
        unsafe {
            xor_rows_dispatch(self.tableau.as_mut_slice(), h_base, i_base, 2 * rl);
        }
    }

    /// Measures qubit `qubit` in the Z basis, collapsing the state.
    ///
    /// Returns `false` for outcome `0` (+1 eigenvalue) and `true` for outcome `1` (-1 eigenvalue).
    ///
    /// # Algorithm (Aaronson-Gottesman §IV)
    ///
    /// **Random case** — some stabilizer generator has X or Y on qubit `qubit`:
    /// 1. Find any stabilizer row `p` (index `n..2n`) with `x[p][qubit] = 1`.
    /// 2. XOR row `p` into every other row that also has `x[.][qubit] = 1`.
    /// 3. Move row `p` to its paired destabilizer row `p - n`.
    /// 4. Replace stabilizer row `p` with `±Z_qubit`, where the sign is random.
    ///
    /// **Deterministic case** — all stabilizers commute with `Z_qubit`:
    /// 1. Compute the product of destabilizer rows `i` (for which the paired
    ///    stabilizer `n+i` has `x[n+i][qubit] = 1`) into a scratch accumulator.
    /// 2. The measurement outcome is the phase of the scratch row.
    pub fn measure(&mut self, qubit: usize) -> Result<bool, QisError> {
        self.validate_qubit(qubit)?;

        // Find first stabilizer row with x[.][qubit] = 1.
        let maybe_p = (self.n..2 * self.n).find(|&row| self.x_bit(row, qubit));

        match maybe_p {
            Some(p) => {
                // ── Random case ──────────────────────────────────────────────
                // Make p the unique row with x[.][qubit] = 1 by XOR-ing p into all others.
                for i in 0..2 * self.n {
                    if i != p && self.x_bit(i, qubit) {
                        self.rowsum(i, p);
                    }
                }

                // Copy stabilizer p → destabilizer p-n.
                let dest = p - self.n;
                let p_base = Self::row_base(p, self.row_len);
                let d_base = Self::row_base(dest, self.row_len);
                // SAFETY: dest ≠ p (since dest < n ≤ p < 2n).
                unsafe {
                    let src = self.tableau.as_slice()[p_base..p_base + 2 * self.row_len].as_ptr();
                    let dst =
                        self.tableau.as_mut_slice()[d_base..d_base + 2 * self.row_len].as_mut_ptr();
                    std::ptr::copy_nonoverlapping(src, dst, 2 * self.row_len);
                }
                self.phases[dest] = self.phases[p];

                // Reset stabilizer p to Z_qubit with random phase.
                let p_base = Self::row_base(p, self.row_len);
                for w in 0..2 * self.row_len {
                    self.tableau[p_base + w] = 0;
                }
                self.set_z_bit(p, qubit, true);
                let b: bool = self.rng.random();
                self.set_phase(p, if b { 2 } else { 0 });
                Ok(b)
            }
            None => {
                // ── Deterministic case ───────────────────────────────────────
                // Accumulate the product of paired stabilizer rows into the
                // pre-allocated scratch row (row 2n) using the SIMD-accelerated
                // rowsum, giving O(n²/w) complexity instead of O(n²) scalar work.
                let scratch = self.scratch_row();
                self.clear_scratch();

                for i in 0..self.n {
                    // Include destabilizer i only if it has an X-component at qubit `qubit`.
                    // Then rowsum the paired stabilizer n+i into scratch.
                    if self.x_bit(i, qubit) {
                        self.rowsum(scratch, self.n + i);
                    }
                }

                // Phase 0 = +1 eigenvalue (outcome 0); phase 2 = -1 eigenvalue (outcome 1).
                Ok(self.phases[scratch] == 2)
            }
        }
    }

    /// Measures all qubits in order, returning the bit string.
    pub fn measure_all(&mut self) -> Vec<bool> {
        (0..self.n)
            .map(|qubit| self.measure(qubit).unwrap())
            .collect()
    }

    /// Performs a forced (post-selected) Z-basis measurement on `qubit`.
    ///
    /// Like `measure`, but the outcome is forced to `outcome` rather than random.
    /// Returns `true` if the forced outcome has non-zero probability (measurement is
    /// consistent with the state), `false` if probability is zero (outcome is impossible).
    ///
    /// Used internally by `probability_of`.
    fn force_measure(&mut self, qubit: usize, outcome: bool) -> bool {
        // Find first stabilizer row with x[.][qubit] = 1.
        let maybe_p = (self.n..2 * self.n).find(|&row| self.x_bit(row, qubit));

        match maybe_p {
            Some(p) => {
                // Random measurement: force to `outcome`.
                for i in 0..2 * self.n {
                    if i != p && self.x_bit(i, qubit) {
                        self.rowsum(i, p);
                    }
                }
                let dest = p - self.n;
                let p_base = Self::row_base(p, self.row_len);
                let d_base = Self::row_base(dest, self.row_len);
                // SAFETY: dest < n ≤ p.
                unsafe {
                    let src = self.tableau.as_slice()[p_base..p_base + 2 * self.row_len].as_ptr();
                    let dst =
                        self.tableau.as_mut_slice()[d_base..d_base + 2 * self.row_len].as_mut_ptr();
                    std::ptr::copy_nonoverlapping(src, dst, 2 * self.row_len);
                }
                self.phases[dest] = self.phases[p];
                let p_base = Self::row_base(p, self.row_len);
                for w in 0..2 * self.row_len {
                    self.tableau[p_base + w] = 0;
                }
                self.set_z_bit(p, qubit, true);
                self.set_phase(p, if outcome { 2 } else { 0 });
                true // random case: forced outcome is always possible
            }
            None => {
                // Deterministic case: check whether outcome matches.
                let scratch = self.scratch_row();
                self.clear_scratch();
                for i in 0..self.n {
                    if self.x_bit(i, qubit) {
                        self.rowsum(scratch, self.n + i);
                    }
                }
                let actual = self.phases[scratch] == 2;
                actual == outcome // false → this outcome is impossible
            }
        }
    }

    /// Returns the probability of measuring the given computational basis state.
    ///
    /// Performs O(n²) work per qubit, O(n³) total, on a cloned (non-destructive) copy.
    ///
    /// # Returns
    /// - `Ok(0.0)` if the bitstring is inconsistent with the stabilizer state.
    /// - `Ok(1.0 / 2.0^k)` where `k` is the number of qubits with random outcomes,
    ///   if the bitstring is in the support.
    /// - `Err(QubitMismatch)` if `bits.len() != n`.
    ///
    /// # Example
    /// ```rust
    /// use cqlib_core::qis::StabilizerState;
    ///
    /// let mut s = StabilizerState::new(2);
    /// s.apply_h(0).unwrap();
    /// s.apply_cx(0, 1).unwrap(); // Bell state |Φ⁺⟩
    ///
    /// // Only |00⟩ and |11⟩ are possible, each with probability 0.5
    /// assert!((s.probability_of(&[false, false]).unwrap() - 0.5).abs() < 1e-10);
    /// assert!((s.probability_of(&[true,  true ]).unwrap() - 0.5).abs() < 1e-10);
    /// assert_eq!(s.probability_of(&[false, true]).unwrap(), 0.0);
    /// assert_eq!(s.probability_of(&[true,  false]).unwrap(), 0.0);
    /// ```
    pub fn probability_of(&self, bits: &[bool]) -> Result<f64, QisError> {
        if bits.len() != self.n {
            return Err(QisError::QubitMismatch {
                expected: self.n,
                actual: bits.len(),
            });
        }
        let mut s = self.clone();
        let mut prob = 1.0_f64;
        for (qubit, &desired) in bits.iter().enumerate() {
            // Check if measurement is random (some stabilizer has X on this qubit).
            let is_random = (s.n..2 * s.n).any(|row| s.x_bit(row, qubit));
            if is_random {
                prob *= 0.5;
                s.force_measure(qubit, desired);
            } else {
                // Deterministic: force_measure returns false if outcome is impossible.
                if !s.force_measure(qubit, desired) {
                    return Ok(0.0);
                }
            }
        }
        Ok(prob)
    }

    /// Returns the full probability distribution over all computational basis states.
    ///
    /// Returns a `Vec<f64>` of length `2^n` where entry `i` is the probability of
    /// measuring the state whose binary representation is `i` (qubit 0 = LSB).
    ///
    /// # Limitations
    /// Only feasible for small systems. Returns `Err(InvalidParameterValue)` if `n > 20`
    /// (which would require a 8 MB vector and O(2^20 × n³) computation).
    ///
    /// # Example
    /// ```rust
    /// use cqlib_core::qis::StabilizerState;
    ///
    /// let mut s = StabilizerState::new(2);
    /// s.apply_h(0).unwrap();
    /// s.apply_cx(0, 1).unwrap(); // Bell state
    ///
    /// let probs = s.probabilities().unwrap();
    /// assert_eq!(probs.len(), 4);
    /// assert!((probs[0b00] - 0.5).abs() < 1e-10); // |00⟩
    /// assert_eq!(probs[0b01], 0.0);                // |10⟩ (qubit-0 is LSB)
    /// assert_eq!(probs[0b10], 0.0);                // |01⟩
    /// assert!((probs[0b11] - 0.5).abs() < 1e-10); // |11⟩
    /// ```
    pub fn probabilities(&self) -> Result<Vec<f64>, QisError> {
        const MAX_QUBITS: usize = 20;
        if self.n > MAX_QUBITS {
            return Err(QisError::InvalidParameterValue(format!(
                "probabilities() is only supported for n ≤ {MAX_QUBITS} qubits \
                 (requested n = {}); use sample_shots() for large systems",
                self.n
            )));
        }
        let size = 1usize << self.n;
        let mut probs = vec![0.0_f64; size];
        for (i, prob) in probs.iter_mut().enumerate() {
            let bits: Vec<bool> = (0..self.n).map(|q| (i >> q) & 1 == 1).collect();
            *prob = self.probability_of(&bits)?;
        }
        Ok(probs)
    }

    /// Runs `shots` independent measurements in parallel using Rayon.
    ///
    /// Seeds for each shot are derived sequentially from the primary RNG (ensuring
    /// reproducibility), then each shot runs on an independently-seeded clone in
    /// parallel. This produces the correct marginal distribution while being
    /// deterministic: the same initial state always yields the same joint sample set.
    ///
    /// # Example
    /// ```rust
    /// use cqlib_core::qis::StabilizerState;
    /// let mut s = StabilizerState::new(2);
    /// s.apply_h(0).unwrap();
    /// s.apply_cx(0, 1).unwrap(); // Bell state
    /// let results = s.sample_shots(1000);
    /// assert_eq!(results.len(), 1000);
    /// for shot in &results { assert_eq!(shot[0], shot[1]); }
    /// ```
    pub fn sample_shots(&self, shots: usize) -> Vec<Vec<bool>> {
        use rayon::prelude::*;
        // Derive independent seeds sequentially so the overall sequence is
        // deterministic and reproducible from the same initial RNG state.
        let seeds: Vec<u64> = {
            let mut rng = self.rng.clone();
            (0..shots).map(|_| rng.random()).collect()
        };
        seeds
            .into_par_iter()
            .map(|seed| {
                let mut shot = self.clone();
                shot.rng = SmallRng::seed_from_u64(seed);
                shot.measure_all()
            })
            .collect()
    }

    // ── Circuit integration ──────────────────────────────────────────────────

    /// Constructs a `StabilizerState` by simulating a Clifford circuit.
    ///
    /// Executes all gates in the circuit sequentially. Non-Clifford gates
    /// (T, TDG, RX, RY, RZ, U, …) return `Err(QisError::NonCliffordGate)`.
    ///
    /// # Supported gates
    /// `I, H, X, Y, Z, S, SDG, X2P, X2M, Y2P, Y2M, CX, CY, CZ, SWAP`, Barriers (ignored), Delays (ignored).
    ///
    /// # Example
    /// ```rust
    /// use cqlib_core::circuit::Circuit;
    /// use cqlib_core::qis::StabilizerState;
    ///
    /// let mut c = Circuit::new(2);
    /// c.h(0.into());
    /// c.cx(0.into(), 1.into());
    /// let stab = StabilizerState::from_circuit(&c).unwrap();
    /// ```
    pub fn from_circuit(circuit: &Circuit) -> Result<Self, QisError> {
        let circuit = circuit.decompose()?;
        let mut state = StabilizerState::new(circuit.num_qubits());

        // Map Qubit → physical index.
        let qubits = circuit.qubits();
        let qubit_map: std::collections::HashMap<_, _> = qubits
            .iter()
            .enumerate()
            .map(|(idx, q)| (*q, idx))
            .collect();

        // Pre-evaluate parameters (none should be symbolic for Clifford gates,
        // but we handle fixed parameters for completeness).
        let parameter_values: Vec<Option<f64>> = circuit
            .parameters()
            .iter()
            .map(|p| p.evaluate(&None).ok())
            .collect();

        for op in circuit.operations() {
            let qubit_indices: Vec<usize> =
                op.qubits
                    .iter()
                    .map(|q| {
                        qubit_map.get(q).copied().ok_or_else(|| {
                            QisError::CircuitError(CircuitError::QubitNotFound(q.id()))
                        })
                    })
                    .collect::<Result<_, _>>()?;

            // Resolve parameters (needed to identify parametric non-Clifford gates).
            let params: Vec<f64> = op
                .params
                .iter()
                .map(|p| match p {
                    CircuitParam::Fixed(v) => Ok(*v),
                    CircuitParam::Index(idx) => parameter_values
                        .get(*idx as usize)
                        .copied()
                        .flatten()
                        .ok_or(QisError::CircuitError(CircuitError::SymbolicParameterError)),
                })
                .collect::<Result<_, _>>()?;

            match &op.instruction {
                Instruction::Standard(gate) => {
                    state.apply_clifford_gate(*gate, &qubit_indices, &params)?;
                }
                Instruction::McGate(mc) if mc.num_ctrl_qubits() == 1 => {
                    let ctrl = qubit_indices[0];
                    let tgt = qubit_indices[1];
                    match mc.base_gate() {
                        StandardGate::X => state.apply_cx(ctrl, tgt)?,
                        StandardGate::Y => state.apply_cy(ctrl, tgt)?,
                        StandardGate::Z => state.apply_cz(ctrl, tgt)?,
                        g => {
                            return Err(QisError::UnsupportedOperation(format!(
                                "Controlled-{g} is not a Clifford gate"
                            )));
                        }
                    }
                }
                Instruction::McGate(mc) => {
                    return Err(QisError::UnsupportedOperation(format!(
                        "{}-controlled gates are not supported in stabilizer simulation",
                        mc.num_ctrl_qubits()
                    )));
                }
                Instruction::Directive(_) | Instruction::Delay => {
                    // Barriers and delays are no-ops.
                }
                Instruction::UnitaryGate(_) => {
                    return Err(QisError::UnsupportedOperation(
                        "Arbitrary unitary gates are not Clifford".to_string(),
                    ));
                }
                Instruction::CircuitGate(_) => {
                    return Err(QisError::CircuitError(CircuitError::InvalidOperation(
                        "CircuitGate should have been decomposed".to_string(),
                    )));
                }
                Instruction::ControlFlowGate(_) => {
                    return Err(QisError::UnsupportedOperation(
                        "Control flow gates are not supported in stabilizer simulation".to_string(),
                    ));
                }
            }
        }

        Ok(state)
    }

    /// Dispatches a [`StandardGate`] to the appropriate Clifford tableau method.
    /// Returns `Err` for any non-Clifford gate.
    fn apply_clifford_gate(
        &mut self,
        gate: StandardGate,
        qubits: &[usize],
        _params: &[f64],
    ) -> Result<(), QisError> {
        match gate {
            StandardGate::I => {}
            StandardGate::H => self.apply_h(qubits[0])?,
            StandardGate::X => self.apply_x(qubits[0])?,
            StandardGate::Y => self.apply_y(qubits[0])?,
            StandardGate::Z => self.apply_z(qubits[0])?,
            StandardGate::S => self.apply_s(qubits[0])?,
            StandardGate::SDG => self.apply_sdg(qubits[0])?,
            StandardGate::X2P => self.apply_x2p(qubits[0])?,
            StandardGate::X2M => self.apply_x2m(qubits[0])?,
            StandardGate::Y2P => self.apply_y2p(qubits[0])?,
            StandardGate::Y2M => self.apply_y2m(qubits[0])?,
            StandardGate::CX => self.apply_cx(qubits[0], qubits[1])?,
            StandardGate::CY => self.apply_cy(qubits[0], qubits[1])?,
            StandardGate::CZ => self.apply_cz(qubits[0], qubits[1])?,
            StandardGate::SWAP => self.apply_swap(qubits[0], qubits[1])?,
            g => {
                return Err(QisError::NonCliffordGate(format!(
                    "Gate '{g}' is not a Clifford gate — use Statevector for universal simulation"
                )));
            }
        }
        Ok(())
    }

    /// There are `n` stabilizer generators (rows `n..2n`).
    pub fn get_stabilizers(&self) -> Vec<PauliString> {
        (0..self.n)
            .map(|i| self.row_to_pauli_string(self.n + i))
            .collect()
    }

    /// Returns the destabilizer generators as [`PauliString`]s.
    pub fn get_destabilizers(&self) -> Vec<PauliString> {
        (0..self.n).map(|i| self.row_to_pauli_string(i)).collect()
    }

    /// Converts tableau row `row` to a [`PauliString`].
    fn row_to_pauli_string(&self, row: usize) -> PauliString {
        let mut ps = PauliString::new(self.n);
        ps.phase = if self.phase(row) == 0 {
            Phase::Plus
        } else {
            Phase::Minus
        };
        for q in 0..self.n {
            let x = self.x_bit(row, q);
            let z = self.z_bit(row, q);
            let pauli = match (x, z) {
                (false, false) => Pauli::I,
                (true, false) => Pauli::X,
                (true, true) => Pauli::Y,
                (false, true) => Pauli::Z,
            };
            ps.set_pauli(q, pauli);
        }
        ps
    }

    /// Returns the expectation value of a single-qubit Pauli operator (+1 or -1).
    ///
    /// Only works if the state is an eigenstate of the given operator.
    /// Returns `0` if the state is not an eigenstate.
    pub fn pauli_expectation(&self, pauli: &PauliString) -> Result<i32, QisError> {
        if pauli.num_qubits != self.n {
            return Err(QisError::QubitMismatch {
                expected: self.n,
                actual: pauli.num_qubits,
            });
        }
        for i in 0..self.n {
            let stab = self.row_to_pauli_string(self.n + i);
            if stab == *pauli {
                return Ok(if stab.phase == Phase::Plus { 1 } else { -1 });
            }
        }
        Ok(0)
    }

    /// Exports the stabilizer tableau in Stim-compatible text format.
    pub fn to_stim_format(&self) -> String {
        let mut out = String::new();
        for i in 0..self.n {
            let stab = self.row_to_pauli_string(self.n + i);
            out.push_str(&stab.to_string());
            out.push('\n');
        }
        out
    }
}

impl Clone for StabilizerState {
    /// O(n²) deep copy — clones the flat tableau, phases, and RNG state.
    ///
    /// The RNG is cloned (not re-seeded) so that parallel samples produced from
    /// independent clones diverge deterministically from the same seed, which is
    /// required for reproducible multi-shot sampling with Rayon.
    fn clone(&self) -> Self {
        StabilizerState {
            n: self.n,
            row_len: self.row_len,
            tableau: self.tableau.clone(),
            phases: self.phases.clone(),
            rng: self.rng.clone(),
        }
    }
}

#[cfg(test)]
#[path = "./stabilizer_test.rs"]
mod stabilizer_test;
