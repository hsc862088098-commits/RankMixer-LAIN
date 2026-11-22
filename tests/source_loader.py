"""Load isolated source classes when optional I/O dependencies are unavailable.

These tests exercise actual method bodies, not a replacement implementation.
Parquet and full-model integration are tested separately with real imports.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_symbols(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text(encoding='utf-8'))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError(f'Missing source symbols in {relative_path}: {names}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative_path, 'exec'), namespace)
    return namespace
