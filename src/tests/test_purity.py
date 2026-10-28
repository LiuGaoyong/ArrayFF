"""Kernels must not import a concrete array library.

``energy_arrays`` is the piece handed to ``autojit``: if it called ``np`` or
``torch`` directly, autoray could not dispatch and compiled paths would
silently skip those operations. This AST check keeps the rule honest.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

BACKEND_NAMES = {"np", "numpy", "torch", "jax", "jnp"}
KERNEL_FUNCTIONS = {"energy_arrays"}


def _kernel_sources() -> list[tuple[pathlib.Path, ast.FunctionDef]]:
    root = pathlib.Path(__file__).parents[1] / "arrayff"
    found: list[tuple[pathlib.Path, ast.FunctionDef]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name not in KERNEL_FUNCTIONS:
                    continue
                found.append((path, node))
    return found


@pytest.mark.parametrize("path,node", _kernel_sources(), ids=lambda x: None)
def test_kernel_does_not_touch_backends(path, node):
    offenders = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in BACKEND_NAMES:
            offenders.append(f"{path.name}:{node.name} uses {child.id!r}")
        is_attr = isinstance(child, ast.Attribute)
        if is_attr and isinstance(child.value, ast.Name):
            if child.value.id in BACKEND_NAMES:
                offenders.append(
                    f"{path.name}:{node.name} uses "
                    f"{child.value.id}.{child.attr}"
                )
    assert not offenders, "\n".join(offenders)


def test_some_kernels_are_found():
    assert _kernel_sources(), "no energy_arrays kernel was discovered"
