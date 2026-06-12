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

//! Typed, side-effect-free runtime classical expressions.
//!
//! This module defines a strongly-typed expression AST for classical
//! computation that lives alongside quantum operations in a dynamic circuit.
//! Expressions read runtime [`ClassicalVar`] and [`ClassicalValue`] handles, but
//! they never measure qubits, write classical storage, or transfer control.
//! Control-flow statements ([`IfOp`], [`WhileOp`], [`ForOp`], [`SwitchOp`])
//! consume these expressions to make branching decisions.
//!
//! [`ClassicalVar`]: super::ClassicalVar
//! [`ClassicalValue`]: super::ClassicalValue
//! [`IfOp`]: super::control_flow::IfOp
//! [`WhileOp`]: super::control_flow::WhileOp
//! [`ForOp`]: super::control_flow::ForOp
//! [`SwitchOp`]: super::control_flow::SwitchOp
//!
//! # Type system
//!
//! Every expression has a static [`ClassicalType`]:
//!
//! | Type | Width | Description |
//! |------|-------|-------------|
//! | `Bool` | 1 | Logical boolean |
//! | `Bit` | 1 | Single measured/assigned bit |
//! | `UInt(w)` | w | Unsigned integer (non-zero width) |
//! | `BitVec(w)` | w | Ordered bit-vector (non-zero width) |
//!
//! `Bit` and `Bool` are intentionally distinct: control-flow conditions
//! require `Bool`, so a measured `Bit` must be explicitly converted via
//! [`ClassicalExpr::to_bool`]. `BitVec` interprets index `0` as the
//! least-significant bit; [`ClassicalExpr::to_uint`] provides a little-endian
//! conversion to `UInt`.
//!
//! [`ClassicalType`]: super::ClassicalType
//!
//! # Building expressions
//!
//! Expressions are constructed from literals, runtime classical handles, and
//! typed builders on [`ClassicalExpr`]. [`ClassicalVar`] and [`ClassicalValue`]
//! handles can be used directly as expression sources; direct operator use is
//! equivalent to converting the handle with `.expr()` first.
//!
//! ## Leaf nodes
//!
//! | Constructor | Produces | Description |
//! |-------------|----------|-------------|
//! | `var(v)` | type of `v` | Reads the current runtime value of a mutable [`ClassicalVar`] |
//! | `value(v)` | type of `v` | Reads an immutable [`ClassicalValue`] produced by measurement |
//! | `bool_literal(b)` | `Bool` | Compile-time boolean constant |
//! | `bit_literal(b)` | `Bit` | Compile-time bit constant (`false` = 0, `true` = 1) |
//! | `uint_literal(w, v)` | `UInt(w)` | Compile-time unsigned integer (width ≤ 128, value fits in width) |
//! | `bit_vec_literal(w, v)` | `BitVec(w)` | Compile-time bit-vector, same constraints as `uint_literal` |
//!
//! ```rust
//! use cqlib_core::circuit::{ClassicalExpr, ClassicalType, ClassicalVar};
//!
//! let flag = ClassicalVar::new(Default::default(), 0, ClassicalType::Bool);
//! let expr = flag.expr();                        // Bool
//! let lit  = ClassicalExpr::bool_literal(true);  // Bool
//! let num  = ClassicalExpr::uint_literal(8, 42).unwrap(); // UInt(8)
//! ```
//!
//! ## Unary operations
//!
//! | Operation | Signature | Produces |
//! |-----------|-----------|----------|
//! | `!expr` | `Bool → Bool`, `Bit → Bit` | same type as input |
//! | `try_not(expr)` | `Bool → Bool`, `Bit → Bit` | same type as input |
//!
//! Negation on `Bool` is logical negation; on `Bit` it is bitwise inversion.
//! Use `try_not` when invalid input should return [`CircuitError`] instead of
//! panicking through the operator trait.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalType, ClassicalVar};
//! # let flag = ClassicalVar::new(Default::default(), 0, ClassicalType::Bool);
//! let cond = !flag; // Bool
//! ```
//!
//! ## Binary operations
//!
//! | Operation | Signature | Produces |
//! |-----------|-----------|----------|
//! | `lhs & rhs` / `try_and(lhs, rhs)` | `Bool×Bool → Bool`, `Bit×Bit → Bit` | same type |
//! | `lhs \| rhs` / `try_or(lhs, rhs)` | same | same type |
//! | `lhs ^ rhs` / `try_xor(lhs, rhs)` | same | same type |
//!
//! Both operands must have the **same** type. `Bool` and `Bit` cannot be mixed.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalType, ClassicalVar, CircuitId};
//! # let cid = CircuitId::new();
//! let a = ClassicalVar::new(cid, 0, ClassicalType::Bool);
//! let b = ClassicalVar::new(cid, 1, ClassicalType::Bool);
//! let both = a & b; // Bool
//! ```
//!
//! ## Comparison operations
//!
//! | Constructor | Signature | Produces |
//! |-------------|-----------|----------|
//! | `eq(lhs, rhs)` | any type × same type | `Bool` |
//! | `ne(lhs, rhs)` | any type × same type | `Bool` |
//! | `lt(lhs, rhs)` | `UInt(w)×UInt(w)` | `Bool` |
//! | `le(lhs, rhs)` | `UInt(w)×UInt(w)` | `Bool` |
//! | `gt(lhs, rhs)` | `UInt(w)×UInt(w)` | `Bool` |
//! | `ge(lhs, rhs)` | `UInt(w)×UInt(w)` | `Bool` |
//!
//! Equality and inequality work on **all** types (including `BitVec`).
//! Ordered comparisons (`lt`, `le`, `gt`, `ge`) are restricted to `UInt`.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalExpr, ClassicalType, ClassicalVar, CircuitId};
//! # let cid = CircuitId::new();
//! let x = ClassicalExpr::var(ClassicalVar::new(cid, 0, ClassicalType::uint(8).unwrap()));
//! let y = ClassicalExpr::var(ClassicalVar::new(cid, 1, ClassicalType::uint(8).unwrap()));
//! let is_less = ClassicalExpr::lt(x, y).unwrap(); // Bool
//! ```
//!
//! ## Type casts
//!
//! | Constructor | Signature | Produces |
//! |-------------|-----------|----------|
//! | `expr.to_bool()` / `bit_to_bool(expr)` | `Bit → Bool` | `Bool` |
//! | `expr.to_uint()` / `bit_vec_to_uint(expr)` | `BitVec(w) → UInt(w)` | `UInt(w)` |
//!
//! Casts are **explicit** — there is no implicit promotion. `to_uint`
//! interprets bit index `0` as the least-significant bit.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalExpr, ClassicalType, ClassicalValue, CircuitId};
//! # let cid = CircuitId::new();
//! let measured = ClassicalExpr::value(ClassicalValue::new(cid, 0, ClassicalType::Bit));
//! let condition = measured.to_bool().unwrap(); // Bool
//! ```
//!
//! ## Conditional selection
//!
//! | Constructor | Signature | Produces |
//! |-------------|-----------|----------|
//! | `select(cond, then, else)` | `Bool × T × T → T` | same type as branches |
//!
//! Both branches must have the same type. The condition must be `Bool`.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalExpr, ClassicalType, ClassicalVar, CircuitId};
//! # let cid = CircuitId::new();
//! let cond = ClassicalExpr::bool_literal(true);
//! let a = ClassicalExpr::uint_literal(8, 10).unwrap();
//! let b = ClassicalExpr::uint_literal(8, 20).unwrap();
//! let chosen = ClassicalExpr::select(cond, a, b).unwrap(); // UInt(8)
//! ```
//!
//! ## Bit extraction and manipulation
//!
//! | Constructor | Signature | Produces |
//! |-------------|-----------|----------|
//! | `extract_bit(val, idx)` | `UInt(w)×u32 → Bit`, `BitVec(w)×u32 → Bit` | `Bit` |
//! | `extract_bits(val, off, w)` | `UInt×u32×u32 → BitVec(w)`, `BitVec×u32×u32 → BitVec(w)` | `BitVec(w)` |
//! | `concat(parts)` | `[Bit or BitVec] → BitVec(total_width)` | `BitVec` |
//! | `pack_bits(bits)` | `[Bit] → BitVec(n)` | `BitVec(n)` |
//!
//! Index `0` is the least-significant bit. `concat` places the first part in
//! the least-significant output bits. `pack_bits` places the first bit at
//! index `0`.
//!
//! ```rust
//! # use cqlib_core::circuit::{ClassicalExpr, ClassicalType, ClassicalVar, CircuitId};
//! # let cid = CircuitId::new();
//! # let bv = ClassicalExpr::var(ClassicalVar::new(cid, 0, ClassicalType::bit_vec(8).unwrap()));
//! let lsb = ClassicalExpr::extract_bit(bv.clone(), 0).unwrap();    // Bit — bit 0
//! let hi  = ClassicalExpr::extract_bits(bv, 4, 4).unwrap();       // BitVec(4) — bits [4..8)
//!
//! let a = ClassicalExpr::bit_literal(true);
//! let b = ClassicalExpr::bit_literal(false);
//! let packed = ClassicalExpr::pack_bits([a, b]).unwrap();          // BitVec(2), a at bit 0
//! ```
//!
//! ## Expression introspection
//!
//! | Method | Returns |
//! |--------|---------|
//! | `ty()` | `ClassicalType` — the static type of this expression |
//! | `kind()` | `&ClassicalExprKind` — the AST node variant |
//! | `vars()` | `BTreeSet<ClassicalVar>` — all mutable variables read |
//! | `values()` | `BTreeSet<ClassicalValue>` — all immutable values read |
//! | `remap_classical_ids(var_map, val_map)` | `ClassicalExpr` — clone with remapped handles |
//! | `simplified()` | `ClassicalExpr` — structurally simplified copy |
//! | `is_bool_true()` / `is_bool_false()` | `bool` — literal predicate |
//! | `is_bit_true()` / `is_bit_false()` | `bool` — literal predicate |
//!
//! # Simplification
//!
//! [`simplify`] performs bottom-up algebraic simplification without evaluating
//! runtime variable or value reads. It is **idempotent**: applying it twice
//! yields the same expression.
//!
//! ## Rules (23 total)
//!
//! **Double negation:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `not(not(x))` | `x` |
//!
//! **Identity elements:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `and(x, true)` / `and(true, x)` | `x` |
//! | `or(x, false)` / `or(false, x)` | `x` |
//! | `xor(x, false)` / `xor(false, x)` | `x` |
//!
//! **Idempotence and self-inverse:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `and(x, x)` | `x` |
//! | `or(x, x)` | `x` |
//! | `xor(x, x)` | `false` |
//!
//! **Complement:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `and(x, not(x))` | `false` |
//! | `or(x, not(x))` | `true` |
//! | `xor(x, not(x))` | `true` |
//!
//! **Comparison reflexivity:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `eq(x, x)` / `le(x, x)` / `ge(x, x)` | `true` |
//! | `ne(x, x)` / `lt(x, x)` / `gt(x, x)` | `false` |
//!
//! **Select and cast folding:**
//!
//! | Pattern | Result |
//! |---------|--------|
//! | `select(true, a, b)` | `a` |
//! | `select(false, a, b)` | `b` |
//! | `bit_to_bool(bit_literal(v))` | `bool_literal(v)` |
//! | `bit_vec_to_uint(bit_vec_literal{w, v})` | `uint_literal(w, v)` |
//!
//! ```rust
//! use cqlib_core::circuit::ClassicalExpr;
//!
//! // not(not(a)) → a
//! let a = ClassicalExpr::bool_literal(true);
//! let expr = ClassicalExpr::try_not(ClassicalExpr::try_not(a.clone()).unwrap()).unwrap();
//! assert_eq!(expr.simplified(), a);
//!
//! // and(b, true) → b
//! let b = ClassicalExpr::bool_literal(false);
//! let expr = ClassicalExpr::try_and(b.clone(), ClassicalExpr::bool_literal(true)).unwrap();
//! assert_eq!(expr.simplified(), b);
//!
//! // eq(x, x) → true
//! let x = ClassicalExpr::uint_literal(8, 42).unwrap();
//! let expr = ClassicalExpr::eq(x.clone(), x).unwrap();
//! assert_eq!(expr.simplified(), ClassicalExpr::bool_literal(true));
//! ```
//!
//! ## Non-goals
//!
//! Simplification deliberately does **not** perform:
//!
//! - **Value-dependent constant folding** — `and(x, false) → false` would
//!   eliminate a runtime variable read and belongs to a higher-level
//!   optimization pass.
//! - **Commutation normalization** — `and(a, b)` is not reordered to
//!   `and(b, a)`.
//! - **Bit-width-aware extract/concat inversion** — `extract_bits(concat(a, b), ...)`
//!   is not reduced.
//!
//! # Integration with control flow
//!
//! The typical flow from measurement to control:
//!
//! ```text
//! measure q[0] → value: Bit
//! value.to_bool() → condition: Bool
//! if condition { ... }
//! ```
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//!
//! fn build() -> Result<Circuit, Box<dyn std::error::Error>> {
//!     let mut circuit = Circuit::new(2);
//!     let q0 = Qubit::new(0);
//!     let q1 = Qubit::new(1);
//!
//!     let measured = circuit.measure(q0)?;
//!     let condition = measured.expr().to_bool()?;
//!     circuit.if_(condition, |body| {
//!         body.x(q1)?;
//!         Ok(())
//!     })?;
//!
//!     Ok(circuit)
//! }
//! ```
//!
//! For loop-carried state, allocate a [`ClassicalVar`] and `store` into it:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, ClassicalExpr, ClassicalType, Qubit};
//!
//! fn build() -> Result<Circuit, Box<dyn std::error::Error>> {
//!     let mut circuit = Circuit::new(1);
//!     let q0 = Qubit::new(0);
//!
//!     let running = circuit.var(ClassicalType::Bool);
//!     circuit.store(running, ClassicalExpr::bool_literal(true))?;
//!
//!     circuit.while_(running.expr(), |body| {
//!         let measured = body.measure(q0)?;
//!         body.store(running, measured.expr().to_bool()?)?;
//!         Ok(())
//!     })?;
//!
//!     Ok(circuit)
//! }
//! ```

pub mod expr;
pub mod simplify;

pub use expr::{
    ClassicalBinaryOp, ClassicalCast, ClassicalCompareOp, ClassicalExpr, ClassicalExprKind,
    ClassicalExprNode, ClassicalUnaryOp,
};
pub use simplify::simplify;
