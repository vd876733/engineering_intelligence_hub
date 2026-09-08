"""
ast_chunker.py — Extract function and class definitions from Python repositories.

Uses tree-sitter + tree-sitter-python to parse source files and return
structured code chunks suitable for embedding / RAG indexing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# ---------------------------------------------------------------------------
# Tree-sitter setup (modern API, tree-sitter >= 0.21)
# ---------------------------------------------------------------------------

PY_LANGUAGE = Language(tspython.language())

# Node types we want to extract
_TARGET_NODE_TYPES = frozenset({"function_definition", "class_definition"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_parser() -> Parser:
    """Return a ready-to-use tree-sitter parser for Python."""
    parser = Parser(PY_LANGUAGE)
    return parser


def _find_python_files(repo_path: str | Path) -> List[Path]:
    """Recursively find every *.py file under *repo_path*."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    return sorted(root.rglob("*.py"))


def _extract_name(node) -> str:
    """Extract the identifier name from a function/class definition node.

    The grammar places an ``identifier`` child immediately after the
    ``def`` / ``class`` keyword.
    """
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return "<anonymous>"


def _byte_offset_to_line(source_bytes: bytes, offset: int) -> int:
    """Convert a byte offset into a 1-based line number."""
    return source_bytes[:offset].count(b"\n") + 1


def _collect_chunks_from_tree(
    tree,
    source_bytes: bytes,
    file_path: str,
) -> List[Dict[str, object]]:
    """Walk the concrete syntax tree and collect target node chunks.

    Performs a depth-first traversal so that nested definitions (e.g. a
    method inside a class) are captured individually.

    Line numbers are derived from byte offsets rather than
    ``node.start_point`` to stay compatible with tree-sitter ≥ 0.26
    on early Python 3.14 builds where the ``Point`` C accessor can
    segfault.
    """
    chunks: List[Dict[str, object]] = []

    def _visit(node):
        if node.type in _TARGET_NODE_TYPES:
            start_line = _byte_offset_to_line(source_bytes, node.start_byte)
            end_line = _byte_offset_to_line(source_bytes, node.end_byte)

            code_snippet = source_bytes[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )

            chunks.append(
                {
                    "code_snippet": code_snippet,
                    "file_path": file_path,
                    "function_name": _extract_name(node),
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

        # Always recurse so we capture nested definitions
        for child in node.children:
            _visit(child)

    _visit(tree.root_node)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_repository(repo_path: str | Path) -> List[Dict[str, object]]:
    """Parse every Python file in *repo_path* and return code chunks.

    Each chunk is a dictionary with the keys:

    * ``code_snippet``  — the verbatim source text of the definition
    * ``file_path``     — path to the file (relative to *repo_path*)
    * ``function_name`` — name of the function / class
    * ``start_line``    — first line number (1-based)
    * ``end_line``      — last line number (1-based)

    Parameters
    ----------
    repo_path:
        Absolute or relative path to a local git repository (or any
        directory tree containing ``.py`` files).

    Returns
    -------
    list[dict]
        A list of chunk dictionaries, ordered by file then position.
    """
    repo_root = Path(repo_path).resolve()
    parser = _build_parser()
    py_files = _find_python_files(repo_root)

    all_chunks: List[Dict[str, object]] = []

    for py_file in py_files:
        source_bytes = py_file.read_bytes()

        tree = parser.parse(source_bytes)

        # Store a portable, repo-relative path
        try:
            relative_path = str(py_file.relative_to(repo_root))
        except ValueError:
            relative_path = str(py_file)

        file_chunks = _collect_chunks_from_tree(tree, source_bytes, relative_path)
        all_chunks.extend(file_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ast_chunker.py <repo_path>")
        sys.exit(1)

    target = sys.argv[1]
    results = parse_repository(target)

    print(f"\n✓ Extracted {len(results)} chunks from {target}\n")
    # Pretty-print the first 3 chunks as a preview
    for chunk in results[:3]:
        print(json.dumps(chunk, indent=2, default=str))
        print("---")

    if len(results) > 3:
        print(f"... and {len(results) - 3} more.")
