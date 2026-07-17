from __future__ import annotations

import base64
from pathlib import Path


def test_export_engine_source_for_exact_cleanup_edit() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/lno327/casimir/fixed_transverse_point_engine.py"
    )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    print("BEGIN_ENGINE_SOURCE_BASE64")
    print(encoded)
    print("END_ENGINE_SOURCE_BASE64")
    raise AssertionError("temporary exact-source export")
