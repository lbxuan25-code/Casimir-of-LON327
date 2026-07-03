"""Model-independent momentum-transfer helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumTransfer:
    qx: float = 0.0
    qy: float = 0.0
    convention: str = "symmetric_q_over_2"

    def _validate_convention(self) -> None:
        if self.convention != "symmetric_q_over_2":
            raise ValueError("convention must be 'symmetric_q_over_2'")

    def left(self, kx: float, ky: float) -> tuple[float, float]:
        self._validate_convention()
        return (kx + 0.5 * self.qx, ky + 0.5 * self.qy)

    def right(self, kx: float, ky: float) -> tuple[float, float]:
        self._validate_convention()
        return (kx - 0.5 * self.qx, ky - 0.5 * self.qy)


def shifted_momenta(
    kx: float,
    ky: float,
    qx: float = 0.0,
    qy: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    transfer = MomentumTransfer(qx=qx, qy=qy)
    return transfer.left(kx, ky), transfer.right(kx, ky)
