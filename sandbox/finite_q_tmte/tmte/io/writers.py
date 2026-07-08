"""Output writers for TM/TE sandbox scans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .complex_json import to_jsonable


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

