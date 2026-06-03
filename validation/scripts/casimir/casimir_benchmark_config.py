"""Shared metadata for local-response Casimir benchmark scripts."""

TORQUE_TOLERANCE = 1e-20

BENCHMARK_METADATA = {
    "local_response": True,
    "finite_momentum_resolved": False,
    "n0_policy": "skip",
    "benchmark_only": True,
    "not_final_casimir_conclusion": True,
}

BENCHMARK_NOTE_PARTS = (
    "local-response Casimir benchmark only",
    "n=0 policy: skip",
    "finite momentum response not included",
    "not a final Casimir conclusion",
)
