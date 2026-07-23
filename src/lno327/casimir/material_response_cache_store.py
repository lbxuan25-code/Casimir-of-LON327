"""Atomic filesystem store for certified material-response artifacts.

Artifact structure and NPZ/JSON encoding live in dedicated modules. This module
owns only content-addressed paths, cache modes, locking, atomic installation,
and existing-entry conflict handling.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Literal

from lno327.casimir.material_response_cache_artifact import (
    MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA,
    CachedCertifiedMaterialResponse,
)
from lno327.casimir.material_response_cache_codec import (
    load_cached_certified_material_response,
    responses_compatible,
    write_cache_artifact,
)
from lno327.casimir.material_response_cache_errors import (
    MaterialResponseCacheConflictError,
    MaterialResponseCacheCorruptionError,
    MaterialResponseCacheError,
    MaterialResponseCacheIdentityError,
    MaterialResponseCacheLockError,
    MaterialResponseCacheMiss,
    MaterialResponseCacheReadOnlyError,
    UnsupportedMaterialResponseCacheSchema,
)
from lno327.casimir.material_response_cache_identity import (
    MATERIAL_RESPONSE_CACHE_SCHEMA,
    MaterialResponseCacheIdentity,
)

MaterialResponseCacheMode = Literal["disabled", "populate", "read_only"]

# Private compatibility aliases keep fault-injection tests and downstream local
# diagnostics stable while implementation responsibilities remain separated.
_write_npz = write_cache_artifact
_responses_compatible = responses_compatible


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class MaterialResponseCacheStore:
    """Content-addressed persistent store with explicit operational mode."""

    root: Path
    mode: MaterialResponseCacheMode = "populate"

    def __post_init__(self) -> None:
        root = Path(self.root)
        mode = str(self.mode)
        if mode not in {"disabled", "populate", "read_only"}:
            raise ValueError("cache mode must be disabled, populate, or read_only")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "mode", mode)

    def path_for(self, identity: MaterialResponseCacheIdentity) -> Path:
        if not isinstance(identity, MaterialResponseCacheIdentity):
            raise TypeError("identity must be a MaterialResponseCacheIdentity")
        digest = identity.sha256
        return (
            self.root
            / MATERIAL_RESPONSE_CACHE_SCHEMA
            / digest[:2]
            / digest[2:4]
            / f"{digest}.npz"
        )

    def get(
        self, identity: MaterialResponseCacheIdentity
    ) -> CachedCertifiedMaterialResponse | None:
        """Load an exact artifact, return miss in populate mode, or fail read-only."""

        if self.mode == "disabled":
            return None
        path = self.path_for(identity)
        if not path.is_file():
            if self.mode == "read_only":
                raise MaterialResponseCacheMiss(
                    f"certified material response cache miss: {identity.sha256}"
                )
            return None
        return load_cached_certified_material_response(
            path,
            expected_identity=identity,
        )

    def put(
        self, artifact: CachedCertifiedMaterialResponse
    ) -> CachedCertifiedMaterialResponse:
        """Validate and atomically install one certified immutable artifact."""

        if not isinstance(artifact, CachedCertifiedMaterialResponse):
            raise TypeError("artifact must be a CachedCertifiedMaterialResponse")
        if self.mode != "populate":
            raise MaterialResponseCacheReadOnlyError(
                f"cache mode {self.mode!r} does not permit writes"
            )

        final_path = self.path_for(artifact.identity)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{artifact.identity.sha256}.",
            suffix=".tmp",
            dir=final_path.parent,
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        temp_path.unlink()
        lock_path = final_path.with_suffix(".lock")
        lock_descriptor: int | None = None
        try:
            _write_npz(temp_path, artifact)
            load_cached_certified_material_response(
                temp_path,
                expected_identity=artifact.identity,
            )
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.write(
                    lock_descriptor,
                    f"pid={os.getpid()}\nidentity={artifact.identity.sha256}\n".encode(
                        "utf-8"
                    ),
                )
                os.fsync(lock_descriptor)
            except FileExistsError as exc:
                raise MaterialResponseCacheLockError(
                    f"cache identity lock already exists: {lock_path}"
                ) from exc

            if final_path.exists():
                existing = load_cached_certified_material_response(
                    final_path,
                    expected_identity=artifact.identity,
                )
                if not _responses_compatible(
                    existing.snapshot,
                    artifact.snapshot,
                    identity=artifact.identity,
                ):
                    raise MaterialResponseCacheConflictError(
                        "existing certified response conflicts with new response"
                    )
                return existing

            os.replace(temp_path, final_path)
            _fsync_directory(final_path.parent)
            return load_cached_certified_material_response(
                final_path,
                expected_identity=artifact.identity,
            )
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "MATERIAL_RESPONSE_CACHE_ARTIFACT_SCHEMA",
    "CachedCertifiedMaterialResponse",
    "MaterialResponseCacheConflictError",
    "MaterialResponseCacheCorruptionError",
    "MaterialResponseCacheError",
    "MaterialResponseCacheIdentityError",
    "MaterialResponseCacheLockError",
    "MaterialResponseCacheMiss",
    "MaterialResponseCacheMode",
    "MaterialResponseCacheReadOnlyError",
    "MaterialResponseCacheStore",
    "UnsupportedMaterialResponseCacheSchema",
    "load_cached_certified_material_response",
]
