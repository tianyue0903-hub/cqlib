#!/usr/bin/env python3
"""Validate compile-module documentation standards in compile/compiler scope."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())

RUST_SCOPE_DIRS = [
    Path("crates/cqlib-core/src/compile"),
]

RUST_SCOPE_FILES = [
    Path("crates/binding-python/src/compile.rs"),
]

PYTHON_SCOPE_DIRS = [
    Path("crates/binding-python/cqlib/compiler"),
]

LEGAL_HEADER_LINES = [
    "This code is part of Cqlib.",
    "(C) Copyright China Telecom Quantum Group 2026",
    "Apache License, Version 2.0",
    "modified files need to carry a notice indicating",
]

RUST_DECL_PATTERNS = [
    re.compile(r"^\s*pub\s+mod\s+[A-Za-z_][A-Za-z0-9_]*\s*;"),
    re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?(?:struct|enum|type)\s+[A-Za-z_][A-Za-z0-9_]*\b"),
    re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*\b"),
]

PYI_CLASS_PAT = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PYI_DEF_PAT = re.compile(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\b")
SECTION_KEYWORDS = ["Args:", "Returns:", "Raises:"]


@dataclass
class Violation:
    path: Path
    line: int
    message: str


class Checker:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.violations: list[Violation] = []

    def run(self) -> int:
        for rel in self._collect_scoped_files():
            path = self.repo_root / rel
            text = path.read_text(encoding="utf-8")
            self._check_legal_header(rel, text)

            if path.suffix == ".rs":
                self._check_rust_module_doc(rel, text)
                self._check_rust_decl_docs(rel, text)
            elif path.suffix == ".pyi":
                self._check_pyi_docs(rel, text)

        if self.violations:
            for v in sorted(self.violations, key=lambda x: (str(x.path), x.line, x.message)):
                print(f"{v.path}:{v.line}: {v.message}")
            print(f"\nFound {len(self.violations)} documentation standard violation(s).")
            return 1

        print("All compile documentation checks passed.")
        return 0

    def _collect_scoped_files(self) -> list[Path]:
        files: set[Path] = set()

        for rel_dir in RUST_SCOPE_DIRS:
            abs_dir = self.repo_root / rel_dir
            if not abs_dir.is_dir():
                self._error(rel_dir, 1, "scope directory is missing")
                continue
            for path in abs_dir.rglob("*.rs"):
                if path.is_file():
                    files.add(path.relative_to(self.repo_root))

        for rel_file in RUST_SCOPE_FILES:
            abs_file = self.repo_root / rel_file
            if not abs_file.is_file():
                self._error(rel_file, 1, "scope file is missing")
                continue
            files.add(rel_file)

        for rel_dir in PYTHON_SCOPE_DIRS:
            abs_dir = self.repo_root / rel_dir
            if not abs_dir.is_dir():
                self._error(rel_dir, 1, "scope directory is missing")
                continue
            for path in abs_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix not in {".py", ".pyi"}:
                    continue
                files.add(path.relative_to(self.repo_root))

        return sorted(files)

    def _error(self, path: Path, line: int, message: str) -> None:
        self.violations.append(Violation(path, line, message))

    def _check_legal_header(self, path: Path, text: str) -> None:
        head = "\n".join(text.splitlines()[:20])
        for token in LEGAL_HEADER_LINES:
            if token not in head:
                self._error(path, 1, f"missing legal header token: {token}")

    def _check_rust_module_doc(self, path: Path, text: str) -> None:
        lines = text.splitlines()
        limit = min(len(lines), 80)
        if not any(lines[i].lstrip().startswith("//!") for i in range(limit)):
            self._error(path, 1, "missing module-level rustdoc comment (`//!`)")

    def _check_rust_decl_docs(self, path: Path, text: str) -> None:
        lines = text.splitlines()
        prod_end = self._production_end(lines)

        for idx in range(prod_end):
            line = lines[idx]
            if not any(p.search(line) for p in RUST_DECL_PATTERNS):
                continue
            if not self._has_doc_comment_before(lines, idx):
                self._error(path, idx + 1, "missing rustdoc (`///`) for declaration")

    @staticmethod
    def _production_end(lines: list[str]) -> int:
        for i, line in enumerate(lines):
            if re.match(r"^\s*#\[cfg\(test\)\]", line):
                return i
        return len(lines)

    @staticmethod
    def _is_attr_line(stripped: str) -> bool:
        if stripped.startswith("#["):
            return True
        if stripped in {")]", "))]", ")))]", "))))]", "(", ")", "),", ")),", "))),"}:
            return True
        if stripped.endswith("]") and stripped.startswith("#"):
            return True
        if stripped.endswith("]") and stripped.startswith(")"):
            return True
        if stripped.endswith("]") and stripped.startswith("("):
            return True
        return False

    def _has_doc_comment_before(self, lines: list[str], idx: int) -> bool:
        j = idx - 1
        waiting_attr_start = False
        while j >= 0:
            stripped = lines[j].strip()
            if not stripped:
                j -= 1
                continue
            if stripped.startswith("///"):
                return True
            if stripped.startswith("//!"):
                return True
            if self._is_attr_line(stripped):
                waiting_attr_start = True
                j -= 1
                continue
            if waiting_attr_start:
                j -= 1
                continue
            return False
        return False

    def _check_pyi_docs(self, path: Path, text: str) -> None:
        lines = text.splitlines()

        for i, line in enumerate(lines):
            class_m = PYI_CLASS_PAT.match(line)
            if class_m and not self._has_immediate_docstring(lines, i):
                self._error(path, i + 1, f"class `{class_m.group(1)}` is missing docstring")

            def_m = PYI_DEF_PAT.match(line)
            if not def_m:
                continue

            indent, name = def_m.group(1), def_m.group(2)
            if name.startswith("_"):
                continue

            doc = self._extract_docstring_after_def(lines, i, len(indent))
            if doc is None:
                self._error(path, i + 1, f"function `{name}` is missing docstring")
                continue

            if len(indent) == 0:
                for keyword in SECTION_KEYWORDS:
                    if keyword not in doc:
                        self._error(path, i + 1, f"function `{name}` docstring missing section `{keyword}`")

    def _has_immediate_docstring(self, lines: list[str], class_idx: int) -> bool:
        class_indent = len(lines[class_idx]) - len(lines[class_idx].lstrip(" "))
        i = class_idx + 1
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip(" "))
            if not stripped:
                i += 1
                continue
            if indent <= class_indent:
                return False
            return stripped.startswith('"""')
        return False

    def _extract_docstring_after_def(
        self,
        lines: list[str],
        def_idx: int,
        def_indent: int,
    ) -> str | None:
        i = def_idx
        while i < len(lines):
            sig_line = lines[i].rstrip()
            if sig_line.endswith(":"):
                i += 1
                break
            i += 1

        if i >= len(lines):
            return None

        doc_started = False
        doc_lines: list[str] = []

        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            indent = len(raw) - len(raw.lstrip(" "))

            if not stripped:
                i += 1
                continue

            if indent <= def_indent and not doc_started:
                return None

            if not doc_started:
                if stripped.startswith('"""'):
                    doc_started = True
                    doc_lines.append(stripped)
                    if stripped.count('"""') >= 2 and len(stripped) > 6:
                        break
                    i += 1
                    continue
                return None

            doc_lines.append(stripped)
            if '"""' in stripped:
                break
            i += 1

        if not doc_lines:
            return None
        return "\n".join(doc_lines)


def main() -> int:
    checker = Checker(REPO_ROOT)
    return checker.run()


if __name__ == "__main__":
    sys.exit(main())
