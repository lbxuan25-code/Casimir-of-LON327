"""Typed fail-closed errors for persistent material-response caching."""
from __future__ import annotations


class MaterialResponseCacheError(RuntimeError):
    """Base class for persistent response-cache failures."""


class MaterialResponseCacheMiss(MaterialResponseCacheError):
    """The requested exact response identity is absent."""


class MaterialResponseCacheReadOnlyError(MaterialResponseCacheError):
    """A write was attempted in disabled or strict read-only mode."""


class MaterialResponseCacheIdentityError(MaterialResponseCacheError):
    """Artifact, filename, manifest, or requested identities disagree."""


class MaterialResponseCacheCorruptionError(MaterialResponseCacheError):
    """A cache artifact is incomplete or fails integrity validation."""


class MaterialResponseCacheConflictError(MaterialResponseCacheError):
    """The same identity produced incompatible certified response values."""


class UnsupportedMaterialResponseCacheSchema(MaterialResponseCacheError):
    """The artifact or cache schema is unknown to this reader."""


class MaterialResponseCacheLockError(MaterialResponseCacheError):
    """An identity writer lock already exists or cannot be acquired."""


__all__ = [
    "MaterialResponseCacheConflictError",
    "MaterialResponseCacheCorruptionError",
    "MaterialResponseCacheError",
    "MaterialResponseCacheIdentityError",
    "MaterialResponseCacheLockError",
    "MaterialResponseCacheMiss",
    "MaterialResponseCacheReadOnlyError",
    "UnsupportedMaterialResponseCacheSchema",
]
