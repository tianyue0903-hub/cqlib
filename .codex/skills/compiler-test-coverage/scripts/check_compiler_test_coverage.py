#!/usr/bin/env python3
"""Gate compiler test-case coverage parity across Rust and Python."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


RUST_SCOPE_DIR = Path("crates/cqlib-core/src/compile")
RUST_SCOPE_FILE = Path("crates/binding-python/src/compile.rs")
PY_API_FILE = Path("crates/binding-python/cqlib/compiler/__init__.py")
PY_TEST_DIR = Path("tests/python/compiler")

REQUIRED_CATEGORIES = [
    "topology",
    "policy_config",
    "strict_vs_fallback",
    "candidate_search",
    "fidelity",
    "validation_failures",
    "stability",
    "workflow_success",
]

CATEGORY_PATTERNS = {
    "topology": ("topology", "connected", "isomorphic", "edge", "qubit"),
    "policy_config": ("policy", "config", "sabreconfig", "vf2_policy"),
    "strict_vs_fallback": (
        "strict",
        "fallback",
        "initial_layout",
        "initial_only",
        "direct_then_sabre",
        "disabled",
    ),
    "candidate_search": ("candidate", "candidates", "topk", "top_k", "max_matches"),
    "fidelity": ("fidelity",),
    "validation_failures": ("reject", "invalid", "unsupported", "overflow", "error", "too_small"),
    "stability": ("deterministic", "random", "seed", "stable"),
    "workflow_success": (
        "workflow",
        "standalone",
        "direct_pipeline",
        "default_config",
        "fast_path",
        "routes_with_sabre",
        "output_uses",
        "module_exports",
    ),
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def collect_rust_tests(repo_root: Path, violations: list[Violation]) -> list[str]:
    rust_files: list[Path] = []
    rust_dir = repo_root / RUST_SCOPE_DIR
    if not rust_dir.is_dir():
        violations.append(Violation(RUST_SCOPE_DIR, 1, "scope directory is missing"))
    else:
        rust_files.extend(sorted(path.relative_to(repo_root) for path in rust_dir.rglob("*.rs")))

    rust_scope_file = repo_root / RUST_SCOPE_FILE
    if not rust_scope_file.is_file():
        violations.append(Violation(RUST_SCOPE_FILE, 1, "scope file is missing"))
    else:
        rust_files.append(RUST_SCOPE_FILE)

    tests: list[str] = []
    test_attr_re = re.compile(r"^\s*#\[\s*test\s*\]")
    fn_re = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for rel_path in sorted(set(rust_files)):
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if not test_attr_re.search(line):
                continue
            for j in range(idx + 1, min(idx + 10, len(lines))):
                fn_match = fn_re.search(lines[j])
                if fn_match:
                    tests.append(normalize_name(fn_match.group(1)))
                    break

    if not tests:
        violations.append(Violation(RUST_SCOPE_DIR, 1, "no Rust #[test] functions found in scope"))
    return tests


def extract_exports(api_path: Path, text: str, violations: list[Violation]) -> list[str]:
    try:
        tree = ast.parse(text, filename=str(api_path))
    except SyntaxError as exc:
        violations.append(Violation(api_path, exc.lineno or 1, f"syntax error: {exc.msg}"))
        return []

    exports: list[str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            violations.append(Violation(api_path, node.lineno, "could not evaluate `__all__`"))
            return []
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            violations.append(Violation(api_path, node.lineno, "`__all__` must be a list/tuple of strings"))
            return []
        exports = list(value)
        break

    if exports is None:
        violations.append(Violation(api_path, 1, "missing `__all__` in compiler API module"))
        return []
    return exports


def collect_python_tests(
    repo_root: Path, violations: list[Violation]
) -> tuple[list[str], set[str], set[str]]:
    scope_dir = repo_root / PY_TEST_DIR
    if not scope_dir.is_dir():
        violations.append(Violation(PY_TEST_DIR, 1, "scope directory is missing"))
        return [], set(), set()

    files = sorted(path.relative_to(repo_root) for path in scope_dir.rglob("test_*.py"))
    if not files:
        violations.append(Violation(PY_TEST_DIR, 1, "no test files matching `test_*.py` were found"))
        return [], set(), set()

    test_names: list[str] = []
    identifiers: set[str] = set()
    class_names: set[str] = set()
    for rel_path in files:
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(rel_path))
        except SyntaxError as exc:
            violations.append(Violation(rel_path, exc.lineno or 1, f"syntax error: {exc.msg}"))
            continue

        module_doc = ast.get_docstring(tree)
        if module_doc is None:
            violations.append(Violation(rel_path, 1, "missing module docstring"))
        elif "Test coverage" not in module_doc:
            violations.append(Violation(rel_path, 1, "module docstring missing `Test coverage` section"))

        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        test_classes = [node for node in classes if node.name.startswith("Test")]
        if not test_classes:
            violations.append(Violation(rel_path, 1, "no `class Test...` group found"))

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                violations.append(Violation(rel_path, node.lineno, "top-level test function is not allowed"))

        for cls in test_classes:
            class_names.add(normalize_name(cls.name))
            for node in cls.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                    continue
                if ast.get_docstring(node) is None:
                    violations.append(
                        Violation(
                            rel_path,
                            node.lineno,
                            f"test method `{cls.name}.{node.name}` is missing docstring",
                        )
                    )
                test_names.append(normalize_name(f"{cls.name}_{node.name}"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)

    if not test_names:
        violations.append(Violation(PY_TEST_DIR, 1, "no Python test methods found in scope"))
    return test_names, class_names, identifiers


def detect_categories(names: list[str]) -> set[str]:
    hits: set[str] = set()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(any(pattern in name for pattern in patterns) for name in names):
            hits.add(category)
    return hits


def build_result(format_name: str) -> int:
    repo_root = find_repo_root(Path(__file__).resolve())
    violations: list[Violation] = []

    rust_tests = collect_rust_tests(repo_root, violations)
    python_tests, python_classes, python_identifiers = collect_python_tests(repo_root, violations)

    api_path = repo_root / PY_API_FILE
    if not api_path.is_file():
        violations.append(Violation(PY_API_FILE, 1, "API module is missing"))
        exports: list[str] = []
    else:
        exports = extract_exports(PY_API_FILE, api_path.read_text(encoding="utf-8"), violations)

    api_missing = sorted(symbol for symbol in exports if symbol not in python_identifiers)
    for symbol in api_missing:
        violations.append(
            Violation(PY_TEST_DIR, 1, f"compiler export `{symbol}` is not referenced by Python compiler tests")
        )

    rust_categories = detect_categories(rust_tests)
    python_categories = detect_categories(python_tests + sorted(python_classes))

    rust_missing = [cat for cat in REQUIRED_CATEGORIES if cat not in rust_categories]
    python_missing = [cat for cat in REQUIRED_CATEGORIES if cat not in python_categories]
    parity_gaps = [
        cat
        for cat in REQUIRED_CATEGORIES
        if (cat in rust_categories and cat not in python_categories)
        or (cat in python_categories and cat not in rust_categories)
    ]

    for category in rust_missing:
        violations.append(Violation(RUST_SCOPE_DIR, 1, f"missing Rust category coverage: {category}"))
    for category in python_missing:
        violations.append(Violation(PY_TEST_DIR, 1, f"missing Python category coverage: {category}"))
    for category in parity_gaps:
        violations.append(Violation(PY_TEST_DIR, 1, f"cross-language parity gap for category: {category}"))

    deduped = sorted({(v.path, v.line, v.message) for v in violations}, key=lambda x: (str(x[0]), x[1], x[2]))
    violation_dicts = [{"path": str(path), "line": line, "message": message} for path, line, message in deduped]

    result = {
        "status": "fail" if violation_dicts else "pass",
        "counts": {
            "rust_tests": len(rust_tests),
            "python_tests": len(python_tests),
        },
        "api_coverage": {"missing": api_missing},
        "category_coverage": {
            "rust_missing": rust_missing,
            "python_missing": python_missing,
            "parity_gaps": parity_gaps,
        },
        "violations": violation_dicts,
    }

    if format_name == "json":
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        if violation_dicts:
            for item in violation_dicts:
                print(f"{item['path']}:{item['line']}: {item['message']}")
            print()
            print("Compiler test coverage check failed.")
        else:
            print("Compiler test coverage check passed.")
        print(f"Rust tests: {result['counts']['rust_tests']}")
        print(f"Python tests: {result['counts']['python_tests']}")
        print(f"API coverage missing: {result['api_coverage']['missing']}")
        print(f"Category coverage: {result['category_coverage']}")

    return 1 if violation_dicts else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    return build_result(args.format)


if __name__ == "__main__":
    sys.exit(main())
