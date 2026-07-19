from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/lno327/casimir/certified_point_provider.py",
    '''    def _consume_payload(
        self,
        payload: Mapping[str, Any],
        *,
        labels: tuple[str, ...],
        q_keys: tuple[tuple[str, str], ...],
        requested_config: FixedCasimirConfig,
    ) -> None:
        if payload.get("schema") != "transverse-point-sweet-spot-v4":
            raise CertifiedPointCacheError(
                "transverse certifier returned an unexpected schema"
            )
        q_by_label = dict(zip(labels, q_keys, strict=True))
        for point in payload.get("point_results", []):
            label = str(point.get("q_label", ""))
            q_key = q_by_label.get(label)
            if q_key is None:
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unknown q label"
                )
            pairing = str(point.get("pairing", ""))
            n = int(point.get("n", -1))
            if (
                pairing not in requested_config.pairings
                or n not in requested_config.matsubara_indices
            ):
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unrequested point"
                )
            key = _entry_key(pairing, n, q_key)
            self._entries[key] = dict(point)
            self._q_by_entry[key] = q_key
''',
    '''    def _consume_payload(
        self,
        payload: Mapping[str, Any],
        *,
        labels: tuple[str, ...],
        q_keys: tuple[tuple[str, str], ...],
        requested_config: FixedCasimirConfig,
    ) -> None:
        if payload.get("schema") != "transverse-point-sweet-spot-v4":
            raise CertifiedPointCacheError(
                "transverse certifier returned an unexpected schema"
            )
        points = payload.get("point_results")
        if not isinstance(points, list):
            raise CertifiedPointCacheError("transverse certifier point_results must be a list")
        q_by_label = dict(zip(labels, q_keys, strict=True))
        expected = {
            (label, pairing, int(n))
            for label in labels
            for pairing in requested_config.pairings
            for n in requested_config.matsubara_indices
        }
        seen: set[tuple[str, str, int]] = set()
        pending: list[tuple[str, tuple[str, str], Mapping[str, Any]]] = []
        for point in points:
            if not isinstance(point, Mapping):
                raise CertifiedPointCacheError("transverse certifier returned a non-object point")
            label = str(point.get("q_label", ""))
            q_key = q_by_label.get(label)
            if q_key is None:
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unknown q label"
                )
            pairing = str(point.get("pairing", ""))
            n = int(point.get("n", -1))
            identity = (label, pairing, n)
            if identity not in expected:
                raise CertifiedPointCacheError(
                    "transverse certifier returned an unrequested point"
                )
            if identity in seen:
                raise CertifiedPointCacheError(
                    "transverse certifier returned a duplicate point"
                )
            returned_q = point.get("q_lab")
            if returned_q is None or _q_key(returned_q) != q_key:
                raise CertifiedPointCacheError(
                    "transverse certifier changed an exact q coordinate"
                )
            seen.add(identity)
            pending.append((_entry_key(pairing, n, q_key), q_key, point))
        if seen != expected:
            missing = sorted(expected - seen)
            raise CertifiedPointCacheError(
                f"transverse certifier omitted {len(missing)} requested points"
            )
        for key, q_key, point in pending:
            self._entries[key] = dict(point)
            self._q_by_entry[key] = q_key
''',
)
