#!/usr/bin/env python3
"""Validate compiler Python test structure and documentation standards."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())

TEST_SCOPE_DIR = Path("tests/python/compiler")

HEADER_TOKENS = [
    "This code is part of Cqlib.",
    "(C) Copyright China Telecom Quantum Group 2026",
    "Apache License, Version 2.0",
    "modified files need to carry a notice indicating",
]

DISALLOWED_PATTERNS = [
    "from __init__ import",
    "if __name__ == \"__main__\"",
]


@dataclass
class Violation:
    path: Path
    line: int
    message: str


class Checker:
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def run(self) -> int:
        for rel_path in self._collect_target_files():
            path = REPO_ROOT / rel_path

            text = path.read_text(encoding="utf-8")
            self._check_header(rel_path, text)
            self._check_disallowed_patterns(rel_path, text)
            self._check_ast(rel_path, text)

        if self.violations:
            for v in sorted(self.violations, key=lambda x: (str(x.path), x.line, x.message)):
                print(f"{v.path}:{v.line}: {v.message}")
            print(f"\nFound {len(self.violations)} compiler test standards violation(s).")
            return 1

        print("Compiler test standards check passed.")
        return 0

    def _collect_target_files(self) -> list[Path]:
        scope_dir = REPO_ROOT / TEST_SCOPE_DIR
        if not scope_dir.is_dir():
            self._error(TEST_SCOPE_DIR, 1, "scope directory is missing")
            return []

        files = sorted(
            path.relative_to(REPO_ROOT)
            for path in scope_dir.rglob("test_*.py")
            if path.is_file()
        )

        if not files:
            self._error(TEST_SCOPE_DIR, 1, "no test files matching `test_*.py` were found")
        return files

    def _error(self, path: Path, line: int, message: str) -> None:
        self.violations.append(Violation(path=path, line=line, message=message))

    def _check_header(self, rel_path: Path, text: str) -> None:
        head = "\n".join(text.splitlines()[:20])
        for token in HEADER_TOKENS:
            if token not in head:
                self._error(rel_path, 1, f"missing legal header token: {token}")

    def _check_disallowed_patterns(self, rel_path: Path, text: str) -> None:
        for pattern in DISALLOWED_PATTERNS:
            line_no = self._line_of(text, pattern)
            if line_no is not None:
                self._error(rel_path, line_no, f"disallowed pattern present: {pattern}")

    def _check_ast(self, rel_path: Path, text: str) -> None:
        try:
            tree = ast.parse(text, filename=str(rel_path))
        except SyntaxError as exc:
            self._error(rel_path, exc.lineno or 1, f"syntax error: {exc.msg}")
            return

        doc = ast.get_docstring(tree)
        if not doc:
            self._error(rel_path, 1, "missing module docstring")
        elif "Test coverage" not in doc:
            self._error(rel_path, 1, "module docstring missing 'Test coverage' section")

        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if not classes:
            self._error(rel_path, 1, "no class definitions found")

        test_classes = [cls for cls in classes if cls.name.startswith("Test")]
        if not test_classes:
            self._error(rel_path, 1, "no `class Test...` group found")

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                self._error(rel_path, node.lineno, "top-level test function is not allowed")

        for cls in test_classes:
            for node in cls.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not node.name.startswith("test_"):
                    continue
                if ast.get_docstring(node) is None:
                    self._error(
                        rel_path,
                        node.lineno,
                        f"test method `{cls.name}.{node.name}` is missing docstring",
                    )

    @staticmethod
    def _line_of(text: str, pattern: str) -> int | None:
        for idx, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                return idx
        return None


def main() -> int:
    return Checker().run()


if __name__ == "__main__":
    sys.exit(main())
