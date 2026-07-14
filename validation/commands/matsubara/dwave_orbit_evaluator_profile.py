"""Profile one exact complete-orbit d-wave primitive callback at a chosen t."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from lno327.constants import KB_EV_PER_K
from validation.lib.commensurate_orbit_workspace import (
    CompleteOrbitAggregateWorkspace,
)
from validation.lib.dwave_orbit_primitive_evaluator import (
    DWaveOrbitPrimitiveEvaluator,
)
from validation.lib.finite_q_validation_models import (
    get_finite_q_validation_model,
)

DEFAULT_OUTPUT = Path(
    "validation/outputs/positive_matsubara/dwave_orbit_evaluator_profile/"
    "dwave_orbit_evaluator_profile.json"
)


def _matsubara_energy_eV(index: int, temperature_K: float) -> float:
    return float(
        2.0
        * np.pi
        * int(index)
        * KB_EV_PER_K
        * float(temperature_K)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nk", type=int, default=1256)
    parser.add_argument("--mx", type=int, default=6)
    parser.add_argument("--my", type=int, default=4)
    parser.add_argument(
        "--matsubara-indices",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--phase", type=float, default=0.5)
    parser.add_argument("--shift-s", type=float, default=0.5)
    parser.add_argument(
        "--subgrid-average",
        choices=("auto", "none"),
        default="auto",
    )
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.nk <= 0:
        parser.error("--nk must be positive")
    if args.mx == 0 and args.my == 0:
        parser.error("at least one of --mx,--my must be nonzero")
    if any(index <= 0 for index in args.matsubara_indices):
        parser.error("all Matsubara indices must be positive")
    if not np.isfinite(args.phase):
        parser.error("--phase must be finite")
    return args


def _summary(payload: dict[str, object]) -> str:
    profile = payload["evaluator_profile"]
    assert isinstance(profile, dict)
    geometry = float(payload["orbit_geometry_seconds"])
    evaluator = float(profile["total_seconds"])
    total = geometry + evaluator
    lines = [
        "d-wave exact complete-orbit primitive evaluator profile",
        "=" * 72,
        f"nk = {payload['nk']}; m = ({payload['mx']}, {payload['my']})",
        f"phase = {payload['phase']}; origins = {payload['orbit_origins']}",
        f"points per t = {payload['points_per_t']}",
        (
            "material workspace implementation = "
            f"{profile['material_workspace_implementation']}"
        ),
        (
            "q workspace implementation = "
            f"{profile['q_workspace_implementation']}"
        ),
        "",
        f"orbit geometry / wrap       {geometry:12.6f} s  "
        f"{geometry / max(total, 1e-30):8.2%}",
        f"material workspace          "
        f"{float(profile['material_workspace_seconds']):12.6f} s  "
        f"{float(profile['material_workspace_seconds']) / max(total, 1e-30):8.2%}",
        f"q workspace                 "
        f"{float(profile['q_workspace_seconds']):12.6f} s  "
        f"{float(profile['q_workspace_seconds']) / max(total, 1e-30):8.2%}",
        f"Kubo factors                "
        f"{float(profile['kubo_factor_seconds']):12.6f} s  "
        f"{float(profile['kubo_factor_seconds']) / max(total, 1e-30):8.2%}",
        f"Kubo contraction            "
        f"{float(profile['kubo_contraction_seconds']):12.6f} s  "
        f"{float(profile['kubo_contraction_seconds']) / max(total, 1e-30):8.2%}",
        f"primitive packing           "
        f"{float(profile['primitive_packing_seconds']):12.6f} s  "
        f"{float(profile['primitive_packing_seconds']) / max(total, 1e-30):8.2%}",
        "-" * 72,
        f"complete callback total     {total:12.6f} s",
        "",
        "diagnostic_only = True",
        "valid_for_casimir_input = False",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    indices = tuple(
        sorted(set(int(value) for value in args.matsubara_indices))
    )
    xi_values = np.asarray(
        [
            _matsubara_energy_eV(index, args.temperature_K)
            for index in indices
        ],
        dtype=float,
    )
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(
        "dwave",
        phase_vertex="bond_endpoint_gauge",
    )
    pairing = model.build_pairing_params(args.delta0_eV)
    evaluator = DWaveOrbitPrimitiveEvaluator(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi_values,
        temperature_K=args.temperature_K,
        eta_eV=args.eta_eV,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
    )
    workspace = CompleteOrbitAggregateWorkspace(
        evaluator=evaluator,
        nk=args.nk,
        mx=args.mx,
        my=args.my,
        shift_s=args.shift_s,
        subgrid_average=args.subgrid_average,
        max_unique_transverse_evaluations=1,
    )
    packed = workspace.evaluate_phase(args.phase)
    profile = evaluator.profile_snapshot()
    payload: dict[str, object] = {
        "schema": "dwave_complete_orbit_evaluator_profile_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nk": int(args.nk),
        "mx": int(args.mx),
        "my": int(args.my),
        "phase": float(args.phase % 1.0),
        "matsubara_indices": indices,
        "xi_eV_values": tuple(float(value) for value in xi_values),
        "primitive_direction": tuple(
            int(value) for value in workspace.primitive_direction
        ),
        "transverse_direction": tuple(
            int(value) for value in workspace.transverse_direction
        ),
        "orbit_origins": workspace.orbit_origins,
        "points_per_t": int(workspace.points_per_t),
        "packed_primitive_width": int(packed.size),
        "orbit_geometry_seconds": float(workspace.geometry_wall_seconds),
        "evaluator_profile": profile.as_dict(),
        "status": {
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = _summary(payload)
    output.with_suffix(".summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print(summary)
    print(f"JSON:    {output}")
    print(f"Summary: {output.with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()
