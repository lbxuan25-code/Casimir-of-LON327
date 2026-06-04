"""Shared metadata for the preliminary local-response Casimir calculation."""

TORQUE_TOLERANCE = 1e-20

BENCHMARK_METADATA = {
    "local_response": True,
    "finite_momentum_resolved": False,
    "n0_policy": "skip",
    "benchmark_only": True,
    "preliminary_local_response_conclusion": True,
    "not_final_casimir_conclusion": True,
}

BENCHMARK_NOTE_PARTS = (
    "preliminary local-response Casimir conclusion",
    "n=0 policy: skip",
    "finite momentum response not included",
    "not a final Casimir conclusion",
)
