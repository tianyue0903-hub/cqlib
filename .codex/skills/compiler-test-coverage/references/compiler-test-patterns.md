# Compiler Test Pattern Baseline

## Scope Baseline

This skill is intentionally scoped to compiler Rust/Python test work:

- Rust compiler mapping tests:
  - `crates/cqlib-core/src/compile/mapping/mod.rs`
  - `crates/cqlib-core/src/compile/mapping/sabre.rs`
- Python compiler tests:
  - `tests/python/compiler/test_vf2.py`
  - `tests/python/compiler/test_sabre.py`
- Python compiler API surface:
  - `crates/binding-python/cqlib/compiler/__init__.py`

## Current Observed Patterns

1. Rust compiler tests are `#[test]` unit tests in compile mapping modules.
2. Python compiler tests are class-based (`class Test...`) and method-based (`def test_...`).
3. Python compiler test modules include a module docstring with a `Test coverage` section.
4. Compiler categories are covered in both languages:
- topology
- policy/config behavior
- strict vs fallback paths
- candidate search
- fidelity behavior
- validation failures
- stability/determinism/randomized behavior
- workflow success paths

## Informational Count Snapshot

At skill creation time, compiler mapping test counts are balanced:

- Rust compiler mapping tests: 27
- Python compiler tests: 27

These counts are informational only and are not enforced as hard numeric thresholds.

## Out Of Scope

C binding tests exist in the repository:

- `crates/binding-c/tests/ffi_test.rs`
- `crates/binding-c/tests/test_c_abi.c`

They are intentionally excluded from this compiler coverage skill.
