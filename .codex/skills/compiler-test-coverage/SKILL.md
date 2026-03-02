---
name: compiler-test-coverage
description: Audit compiler Rust/Python test-case coverage parity and Python API-to-test usage for review gating. Use when checking compiler-related changes in `crates/cqlib-core/src/compile`, `crates/binding-python/src/compile.rs`, `crates/binding-python/cqlib/compiler`, and `tests/python/compiler`, especially when reviewing others' work before merge.
---

# Compiler Test Coverage Skill

Run this skill to enforce compiler test-case coverage standards and block merges when required scenarios are missing in Rust or Python.

## Required Commands

```bash
python .codex/skills/compiler-test-coverage/scripts/check_compiler_test_coverage.py
cargo test -p cqlib-core compile::mapping -- --nocapture
pytest -q tests/python/compiler
```

Treat a non-zero exit code from the checker as merge-blocking.

## Coverage Gate Scope

The checker is intentionally limited to compiler Rust/Python surfaces:

- `crates/cqlib-core/src/compile/**/*.rs`
- `crates/binding-python/src/compile.rs`
- `crates/binding-python/cqlib/compiler/__init__.py`
- `tests/python/compiler/test_*.py`

C binding tests are excluded from this skill.

## What The Checker Enforces

1. Rust compile test discovery from `#[test]` functions.
2. Python compiler test structure:
- Module docstring includes `Test coverage`.
- Test organization uses `class Test...`.
- No top-level `def test_...`.
- Every `test_...` method has a docstring.
3. Python API-to-test usage:
- Read exported compiler symbols from `__all__`.
- Require each export to appear in compiler Python tests.
4. Category parity across Rust and Python:
- `topology`
- `policy_config`
- `strict_vs_fallback`
- `candidate_search`
- `fidelity`
- `validation_failures`
- `stability`
- `workflow_success`

## Output Modes

- Text mode (default): human-readable violations and summary.
- JSON mode: `python .../check_compiler_test_coverage.py --format json`

Use JSON mode for CI integration or automation.

## Reference Baseline

Use [`references/compiler-test-patterns.md`](references/compiler-test-patterns.md) as the observed baseline for current compiler test patterns and scope assumptions.
