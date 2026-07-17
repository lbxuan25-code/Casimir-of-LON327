"""Source-tree provenance for formal arbitrary-q evidence.

A git commit id alone is insufficient when the working tree contains tracked or
untracked edits.  Formal performance and numerical commands use this module to
fail closed unless the checked-out source tree is clean, and to bind manifests
to the same commit, tree object, tracked-file index and source fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import subprocess
from typing import Sequence


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to resolve git provenance for {' '.join(args)}") from exc


@dataclass(frozen=True)
class SourceTreeProvenance:
    git_head: str
    git_tree_sha: str
    worktree_clean: bool
    source_tree_fingerprint: str
    tracked_index_fingerprint: str
    status_porcelain: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "git_head": self.git_head,
            "git_tree_sha": self.git_tree_sha,
            "worktree_clean": bool(self.worktree_clean),
            "source_tree_fingerprint": self.source_tree_fingerprint,
            "tracked_index_fingerprint": self.tracked_index_fingerprint,
            "status_porcelain": list(self.status_porcelain),
        }

    def require_clean(self) -> None:
        if not self.worktree_clean:
            preview = "; ".join(self.status_porcelain[:8])
            raise RuntimeError(
                "formal arbitrary-q evidence requires a clean worktree; "
                f"git status reported: {preview or '<unknown>'}"
            )


def source_tree_provenance() -> SourceTreeProvenance:
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    status_text = _git("status", "--porcelain", "--untracked-files=all")
    status = tuple(line for line in status_text.splitlines() if line)
    index_text = _git("ls-files", "-s")
    index_fingerprint = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    payload = "\n".join(
        (
            "ArbitraryQSourceTreeProvenance-v1",
            head,
            tree,
            index_fingerprint,
            *status,
        )
    ).encode("utf-8")
    return SourceTreeProvenance(
        git_head=head,
        git_tree_sha=tree,
        worktree_clean=not status,
        source_tree_fingerprint=hashlib.sha256(payload).hexdigest(),
        tracked_index_fingerprint=index_fingerprint,
        status_porcelain=status,
    )


def provenance_compatibility(
    left: SourceTreeProvenance | dict[str, object],
    right: SourceTreeProvenance | dict[str, object],
) -> tuple[str, ...]:
    def mapping(value: SourceTreeProvenance | dict[str, object]) -> dict[str, object]:
        return value.as_dict() if isinstance(value, SourceTreeProvenance) else dict(value)

    a, b = mapping(left), mapping(right)
    violations: list[str] = []
    for key in (
        "git_head",
        "git_tree_sha",
        "source_tree_fingerprint",
        "tracked_index_fingerprint",
    ):
        if a.get(key) != b.get(key):
            violations.append(f"source provenance differs for {key}")
    if a.get("worktree_clean") is not True or b.get("worktree_clean") is not True:
        violations.append("formal source provenance must be clean")
    return tuple(violations)


__all__ = [
    "SourceTreeProvenance",
    "provenance_compatibility",
    "source_tree_provenance",
]
