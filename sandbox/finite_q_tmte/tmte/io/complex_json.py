"""Robust JSON conversion for complex numpy payloads."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert numpy and complex values to plain JSON-compatible objects."""

    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "shape": list(value.shape),
                "data": [[to_jsonable(item) for item in row] for row in value.reshape(value.shape[0], -1)]
                if value.ndim >= 2
                else [to_jsonable(item) for item in value],
            }
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def complex_matrix_from_json(payload: dict[str, Any]) -> np.ndarray:
    """Decode a matrix encoded by to_jsonable."""

    shape = tuple(int(x) for x in payload["shape"])
    data = payload["data"]
    flat = []
    if len(shape) >= 2:
        for row in data:
            flat.extend(row)
    else:
        flat = list(data)
    values = [complex(float(item["real"]), float(item["imag"])) for item in flat]
    return np.asarray(values, dtype=complex).reshape(shape)

