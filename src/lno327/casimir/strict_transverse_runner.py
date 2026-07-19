"""Canonical subprocess runner with strict q-point file validation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .fixed_chain import (
    FixedCasimirConfig,
    FixedCasimirExecutionError,
    _CertificationRun,
    _thread_environment,
    _transverse_certification_command,
)
from .fixed_outer_q import OuterQNodeManifest

_STRICT_CERTIFIER_MODULE = "lno327.casimir.fixed_transverse_point_cli"


def run_strict_transverse_certifier(
    config: FixedCasimirConfig,
    manifest: OuterQNodeManifest,
    output: Path,
) -> _CertificationRun:
    q_points_file = output.with_name("q_points.json")
    q_points_payload = [
        {
            "label": str(label),
            "q_lab": [float(q[0]), float(q[1])],
        }
        for label, q in zip(manifest.labels, manifest.q_model, strict=True)
    ]
    q_points_file.write_text(
        json.dumps(q_points_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    command = _transverse_certification_command(
        config,
        manifest,
        output,
        q_points_file=q_points_file,
    )
    module_index = command.index("-m") + 1
    command[module_index] = _STRICT_CERTIFIER_MODULE
    completed = subprocess.run(
        command,
        env=_thread_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FixedCasimirExecutionError(
            "production transverse-point certification failed with return code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedCasimirExecutionError(
            f"cannot read production transverse-point certification payload: {exc}"
        ) from exc
    if payload.get("schema") != "transverse-point-sweet-spot-v4":
        raise FixedCasimirExecutionError(
            "production transverse-point certification returned an unexpected schema"
        )
    return _CertificationRun(
        payload=payload,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=tuple(command),
    )


__all__ = ["run_strict_transverse_certifier"]
