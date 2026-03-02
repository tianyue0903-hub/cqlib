---
name: code-commends
description: Enforce Cqlib documentation and test-comment standards for compile/compiler modules. Use when editing compile-related Rust/Python APIs or compiler Python tests, and when validating the doc/test quality gates with the repository check scripts.
---

# Compile/Compiler Documentation Standards

Follow this workflow for any change in compile/compiler scope:

1. Apply the legal header in each modified file.
2. Keep module and API docs consistent with current standards.
3. Update tests with documented coverage and class-based structure.
4. Run the same validation commands used by CI before finishing.

## Scope

Apply this skill for files in these areas:

- `crates/cqlib-core/src/compile`
- `crates/binding-python/src/compile.rs`
- `crates/binding-python/cqlib/compiler`
- `tests/python/compiler`

## Required Standards

### Shared legal header

Ensure each scoped file includes these tokens near the top:

- `This code is part of Cqlib.`
- `(C) Copyright China Telecom Quantum Group 2026`
- `Apache License, Version 2.0`
- `modified files need to carry a notice indicating`

### Rust (`.rs`) standards

- Add module-level Rustdoc (`//!`) for non-trivial modules.
- Add declaration docs (`///`) for public module/type/function declarations.
- Keep docs precise about behavior, constraints, and errors.

### Python API (`.pyi`) standards

- Add docstrings for public classes and public functions.
- For top-level public functions, include sections:
  - `Args:`
  - `Returns:`
  - `Raises:`

### Python compiler tests standards

- Keep a module docstring with a `Test coverage` section.
- Organize tests into `class Test...` groups.
- Avoid top-level `test_...` functions.
- Add docstrings for each test method.
- Avoid these patterns:
  - `from __init__ import`
  - `if __name__ == "__main__"`

## Commands

Run these commands locally:

```bash
python .codex/skills/code-commends/scripts/check_compile_comment_standards.py
python .codex/skills/code-commends/scripts/check_compiler_test_standards.py
```

Both commands must pass before merging.

## Quick Completion Checklist

1. Confirm legal header tokens are present.
2. Confirm Rust/Python docs are updated for all changed public APIs.
3. Confirm test file structure and docstrings follow the standard.
4. Run both Check commands and verify success.
