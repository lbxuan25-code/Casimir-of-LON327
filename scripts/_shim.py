"""Helpers for backwards-compatible script entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def load_impl(wrapper_file: str, relative_impl: str) -> ModuleType:
    impl_path = Path(wrapper_file).resolve().parent / relative_impl
    module_name = f"_lno327_script_impl_{impl_path.parent.name}_{impl_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script implementation: {impl_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def export_impl(module: ModuleType, namespace: dict[str, object]) -> None:
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        namespace[name] = value
