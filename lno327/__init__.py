"""Development-tree import shim for the real ``src/lno327`` package.

Editable installs import ``lno327`` directly from ``src``. This shim keeps
repository-root smoke commands working the same way, without duplicating package
logic or formulas outside the source tree.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "lno327"
if not _SRC_PACKAGE.is_dir():
    raise ImportError(f"Cannot locate source package at {_SRC_PACKAGE}")

__path__ = [str(_SRC_PACKAGE)]
__file__ = str(_SRC_PACKAGE / "__init__.py")

with open(__file__, "rb") as _source:
    exec(compile(_source.read(), __file__, "exec"), globals(), globals())
