from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "casimir" / "finite_q_bdg_casimir_pipeline.py"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("finite_q_bdg_casimir_pipeline_atomic_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_atomic_json_write_uses_unique_tmp_files_under_concurrency(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    path = tmp_path / "shared.json"

    def write_one(index: int) -> None:
        module._atomic_write_json(path, {"index": index, "payload": "ok"})

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write_one, range(16)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["payload"] == "ok"
    assert isinstance(payload["index"], int)
    assert not list(tmp_path.glob("shared.json.tmp*"))


def test_run_config_resume_creation_is_concurrency_safe(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    path = tmp_path / "run_config.json"
    current = {"config_hash": "same-hash", "payload": {"grid": "same"}}
    paths = {"run_config": path}

    def check_or_create(_index: int) -> None:
        module._check_or_write_run_config(paths, current, resume=True, allow_mismatch=False)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(check_or_create, range(16)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config_hash"] == "same-hash"
    assert not list(tmp_path.glob("run_config.json.tmp*"))
