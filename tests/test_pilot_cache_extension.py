from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig
from scripts.full_casimir import workflow
from scripts.full_casimir.cache_extension import prepare_extension_cache
from scripts.full_casimir.cache_migration import CACHE_SCHEMA


def _write_source_run(
    run: Path,
    *,
    config: FixedCasimirConfig,
    entries: list[dict],
) -> None:
    cache = run / "cache" / "certified_points.json"
    cache.parent.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "outer_tail_config": {
                    "joint_config": {
                        "radial_config": {"point_config": config.as_dict()}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cache.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "policy_fingerprint": certified_point_policy_fingerprint(
                    config,
                    frequency_extendable=True,
                ),
                "frequency_extendable": True,
                "active_matsubara_indices": list(config.matsubara_indices),
                "point_policy": certified_point_policy_payload(
                    config,
                    frequency_extendable=True,
                ),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _entry(*, qx: float, established: bool) -> dict:
    return {
        "pairing": "dwave",
        "n": 1,
        "qx_hex": float(qx).hex(),
        "qy_hex": float(-qx).hex(),
        "point_result": {
            "pairing": "dwave",
            "n": 1,
            "history": [],
            "sweet_spot": {
                "status": "established" if established else "not_established",
                "working_N": 192 if established else None,
                "audit_N": 256 if established else None,
            },
        },
    }


def test_prefix_extension_retains_established_and_drops_unresolved(
    tmp_path: Path,
) -> None:
    source = FixedCasimirConfig(
        pairings=("dwave",),
        plate_angles_deg=(0.0, 0.0),
        N_candidates=(128, 192, 256),
    )
    target = replace(source, N_candidates=(128, 192, 256, 384, 512))
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    _write_source_run(
        source_run,
        config=source,
        entries=[
            _entry(qx=0.1, established=True),
            _entry(qx=0.2, established=False),
        ],
    )

    report = prepare_extension_cache(
        pairing="dwave",
        source_run_dir=source_run,
        target_run_dir=target_run,
        target_point_config=target,
    )

    assert report.mode == "N_ladder_prefix_extension"
    assert report.source_entry_count == 2
    assert report.retained_entry_count == 1
    assert report.dropped_unresolved_count == 1
    assert report.dropped_identities == (
        ("dwave", 1, float(0.2).hex(), float(-0.2).hex()),
    )
    payload = json.loads(
        (target_run / "cache" / "certified_points.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["qx_hex"] == float(0.1).hex()
    assert payload["policy_fingerprint"] == certified_point_policy_fingerprint(
        target,
        frequency_extendable=True,
    )
    saved_report = json.loads(
        (target_run / "cache" / "extension_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_report["dropped_unresolved_count"] == 1
    assert saved_report["source_cache_sha256"] == report.source_cache_sha256
    assert saved_report["target_cache_sha256"] == report.target_cache_sha256


def test_extension_rejects_nonprefix_N_change(tmp_path: Path) -> None:
    source = FixedCasimirConfig(
        pairings=("dwave",),
        plate_angles_deg=(0.0, 0.0),
        N_candidates=(128, 192, 256),
    )
    target = replace(source, N_candidates=(128, 192, 320, 384))
    source_run = tmp_path / "source"
    _write_source_run(
        source_run,
        config=source,
        entries=[_entry(qx=0.1, established=True)],
    )

    with pytest.raises(ValueError, match="complete source ladder as a prefix"):
        prepare_extension_cache(
            pairing="dwave",
            source_run_dir=source_run,
            target_run_dir=tmp_path / "target",
            target_point_config=target,
        )


def test_identical_point_policy_can_seed_outer_cutoff_extension(
    tmp_path: Path,
) -> None:
    config = FixedCasimirConfig(
        pairings=("dwave",),
        plate_angles_deg=(0.0, 0.0),
        N_candidates=(128, 192, 256),
    )
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    _write_source_run(
        source_run,
        config=config,
        entries=[_entry(qx=0.1, established=True)],
    )

    report = prepare_extension_cache(
        pairing="dwave",
        source_run_dir=source_run,
        target_run_dir=target_run,
        target_point_config=config,
    )

    assert report.mode == "identical_point_policy"
    assert report.retained_entry_count == 1
    assert report.dropped_unresolved_count == 0


def test_prepare_extension_command_dispatches_without_running_energy(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(workflow, "_resources", lambda args: object())
    monkeypatch.setattr(workflow, "validate_pairings", lambda values: tuple(values))
    monkeypatch.setattr(workflow, "_energy_options", lambda args: object())

    def fake_prepare(args, pairings, resources, options, *, target_profile):
        seen["source_profile"] = args.source_profile
        seen["target_profile"] = target_profile
        seen["pairings"] = pairings

    monkeypatch.setattr(workflow, "_prepare_extension", fake_prepare)
    monkeypatch.setattr(
        workflow,
        "run_energy_cases",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache preparation must not start an energy run")
        ),
    )

    assert workflow.main(
        [
            "prepare-pilot-extension",
            "--pairings",
            "dwave",
            "--source-profile",
            "0deg_pilot_v3",
            "--profile",
            "0deg_pilot_v4",
        ]
    ) == 0
    assert seen == {
        "source_profile": "0deg_pilot_v3",
        "target_profile": "0deg_pilot_v4",
        "pairings": ("dwave",),
    }
