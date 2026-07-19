"""Strict production CLI wrapper for transverse-point certification."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

from .fixed_transverse_point_certification import main as _certification_main


def validate_q_points_file_argument(argv: Sequence[str]) -> None:
    """Validate structural fields that argparse's string coercion would hide."""

    raw = list(argv)
    if "--q-points-file" not in raw:
        return
    index = raw.index("--q-points-file")
    if index + 1 >= len(raw) or raw[index + 1].startswith("--"):
        raise ValueError("--q-points-file requires a path")
    path = Path(raw[index + 1])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read q-points file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"q-points file is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("q-points file must contain a nonempty JSON list")
    labels: set[str] = set()
    for record_index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(
                f"q-points file record {record_index} must be an object"
            )
        label = record.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"q-points file record {record_index} must contain a nonempty string label"
            )
        if label in labels:
            raise ValueError(f"q-points file contains duplicate label {label!r}")
        labels.add(label)


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    validate_q_points_file_argument(raw)
    _certification_main(raw)


if __name__ == "__main__":
    main()
