from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/lno327/casimir/fixed_chain.py",
    '''def _transverse_certification_command(
    config: FixedCasimirConfig,
    manifest: OuterQNodeManifest,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lno327.casimir.fixed_transverse_point_certification",
    ]
    for label, q in zip(manifest.labels, manifest.q_model, strict=True):
        command.extend(["--q-point", label, repr(float(q[0])), repr(float(q[1]))])
''',
    '''def _command_float(value: float) -> str:
    """Return an exact positional token that argparse treats as a number."""

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("command-line float must be finite")
    if scalar == 0.0:
        return "0.0"
    return np.format_float_positional(scalar, unique=True, trim="-")


def _transverse_certification_command(
    config: FixedCasimirConfig,
    manifest: OuterQNodeManifest,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lno327.casimir.fixed_transverse_point_certification",
    ]
    for label, q in zip(manifest.labels, manifest.q_model, strict=True):
        command.extend(["--q-point", label, _command_float(q[0]), _command_float(q[1])])
''',
)

replace_once(
    "src/lno327/casimir/adaptive_matsubara_tail.py",
    '''    max_total_microscopic_point_entries: int = 1_000_000
    point_cache_path: Path | None = None
''',
    '''    max_total_microscopic_point_entries: int = 1_000_000
    certifier_q_batch_size: int = 384
    point_cache_path: Path | None = None
''',
)
replace_once(
    "src/lno327/casimir/adaptive_matsubara_tail.py",
    '''        entries = int(self.max_total_microscopic_point_entries)
        if entries <= 0:
            raise ValueError("max_total_microscopic_point_entries must be positive")
        object.__setattr__(self, "max_total_microscopic_point_entries", entries)
        if self.point_cache_path is not None:
''',
    '''        entries = int(self.max_total_microscopic_point_entries)
        if entries <= 0:
            raise ValueError("max_total_microscopic_point_entries must be positive")
        object.__setattr__(self, "max_total_microscopic_point_entries", entries)
        batch_size = int(self.certifier_q_batch_size)
        if batch_size <= 0:
            raise ValueError("certifier_q_batch_size must be positive")
        object.__setattr__(self, "certifier_q_batch_size", batch_size)
        if self.point_cache_path is not None:
''',
)
replace_once(
    "src/lno327/casimir/adaptive_matsubara_tail.py",
    '''            "per_term_outer_budget_fraction": self.per_term_outer_budget_fraction,
            "tail_start_n": self.tail_start_n,
''',
    '''            "per_term_outer_budget_policy": "active_cutoff_term_count",
            "per_term_outer_budget_fraction_at_maximum_cutoff": (
                self.per_term_outer_budget_fraction
            ),
            "tail_start_n": self.tail_start_n,
''',
)
replace_once(
    "src/lno327/casimir/adaptive_matsubara_tail.py",
    '''            "max_total_microscopic_point_entries": (
                self.max_total_microscopic_point_entries
            ),
            "point_cache_path": (
''',
    '''            "max_total_microscopic_point_entries": (
                self.max_total_microscopic_point_entries
            ),
            "certifier_q_batch_size": self.certifier_q_batch_size,
            "point_cache_path": (
''',
)
replace_once(
    "src/lno327/casimir/adaptive_matsubara_tail.py",
    '''            active_provider = FrequencyExtendableCertifiedOuterQProvider(
                first_point,
                cache_path=config.point_cache_path,
            )
''',
    '''            active_provider = FrequencyExtendableCertifiedOuterQProvider(
                first_point,
                cache_path=config.point_cache_path,
                certifier_q_batch_size=config.certifier_q_batch_size,
            )
''',
)
