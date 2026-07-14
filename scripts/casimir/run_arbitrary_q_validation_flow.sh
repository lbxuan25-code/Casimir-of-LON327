#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

stage="${1:-help}"
head_tag="$(git rev-parse --short=12 HEAD)"
out_root="${ARBITRARY_Q_OUT_ROOT:-validation/outputs/matsubara/arbitrary_q_staged_flow/${head_tag}}"
mkdir -p "${out_root}"

require_clean() {
  local status
  status="$(git status --porcelain --untracked-files=all)"
  if [[ -n "${status}" ]]; then
    printf '%s\n' "Formal stage requires a clean worktree:" >&2
    printf '%s\n' "${status}" >&2
    exit 2
  fi
}

case "${stage}" in
  stage1|performance-smoke)
    python -m validation diagnostic arbitrary-q-performance-smoke \
      --pairings spm \
      --N 128 \
      --q-tasks 4 \
      --workers 4 \
      --matsubara-indices 0 1 2 4 8 \
      --canonical-block-size 4096 \
      --runtime-chunk-sizes 4096 16384 \
      --output "${out_root}/stage1_performance_smoke.json"
    ;;

  stage2|physics-smoke)
    python -m validation diagnostic arbitrary-q-physics-smoke \
      --pairings spm dwave \
      --N-values 128 192 \
      --workers 4 \
      --reference-nk 1256 \
      --matsubara-indices 0 1 8 \
      --canonical-block-size 4096 \
      --runtime-chunk-size 16384 \
      --temperature-K 10 \
      --delta0-eV 0.1 \
      --eta-eV 1e-8 \
      --separation-nm 20 \
      --ward-tolerance 1e-7 \
      --ward-absolute-tolerance 1e-12 \
      --output "${out_root}/stage2_physics_smoke.json"
    ;;

  diagnostics)
    bash "$0" stage1
    bash "$0" stage2
    ;;

  stage3|formal-performance)
    require_clean
    python -m validation matsubara arbitrary-q-performance-preflight \
      --pairings spm dwave \
      --N 128 \
      --q-tasks 8 \
      --workers 8 \
      --matsubara-indices 0 1 2 4 8 \
      --canonical-block-size 4096 \
      --runtime-chunk-sizes 4096 16384 \
      --temperature-K 10 \
      --delta0-eV 0.1 \
      --eta-eV 1e-8 \
      --comparison-atol 2e-12 \
      --comparison-rtol 2e-11 \
      --minimum-speedup 4 \
      --minimum-cpu-wall-ratio 4 \
      --maximum-pool-overhead-fraction 0.05 \
      --output "${out_root}/stage3_formal_performance.json"
    ;;

  stage4|formal-qualification)
    require_clean
    manifest="${ARBITRARY_Q_PERFORMANCE_MANIFEST:-${out_root}/stage3_formal_performance.json}"
    if [[ ! -f "${manifest}" ]]; then
      printf 'Performance manifest not found: %s\n' "${manifest}" >&2
      exit 2
    fi
    python -m validation matsubara arbitrary-q-periodic-bz-qualification \
      --performance-manifest "${manifest}" \
      --pairings spm dwave \
      --N-values 256 384 512 \
      --reference-nk 1256 \
      --reference-order 384 \
      --reference-panel-count 16 \
      --reference-workers 8 \
      --reference-task-size 4 \
      --workers 4 \
      --matsubara-indices 0 1 8 \
      --canonical-block-size 4096 \
      --runtime-chunk-size 16384 \
      --temperature-K 10 \
      --delta0-eV 0.1 \
      --eta-eV 1e-8 \
      --separation-nm 20 \
      --primitive-tolerance 1e-3 \
      --primitive-atol 1e-12 \
      --reflection-tolerance 3e-4 \
      --reflection-atol 1e-12 \
      --logdet-tolerance 3e-4 \
      --logdet-atol 1e-14 \
      --diagonal-observable-tolerance 1e-3 \
      --diagonal-observable-atol 1e-12 \
      --ward-tolerance 1e-7 \
      --ward-absolute-tolerance 1e-12 \
      --output "${out_root}/stage4_formal_qualification.json"
    ;;

  help|-h|--help)
    cat <<'EOF'
Usage: bash scripts/casimir/run_arbitrary_q_validation_flow.sh <stage>

Stages:
  stage1 | performance-smoke      Small timing breakdown and optimization-structure check
  stage2 | physics-smoke          Small-N physical closure check without convergence claims
  diagnostics                     Run stage1 then stage2
  stage3 | formal-performance     Clean-head real-hardware formal performance preflight
  stage4 | formal-qualification   Clean-head N=256/384/512 formal numerical qualification

Environment:
  ARBITRARY_Q_OUT_ROOT             Override output directory
  ARBITRARY_Q_PERFORMANCE_MANIFEST Override stage4 performance manifest path

Stage 5 q-envelope is intentionally absent until separation range, angle range,
q quadrature/cutoff rule, and tail tolerance are frozen.
EOF
    ;;

  *)
    printf 'Unknown stage: %s\n' "${stage}" >&2
    exit 2
    ;;
esac
