"""Publication-oriented plotting helpers for diagnostic scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

PUBLICATION_DPI = 300


def configure_publication_matplotlib() -> None:
    """Apply a restrained Matplotlib style suitable for paper drafts."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
        }
    )


def style_publication_axis(ax, *, legend: bool = True, grid: bool = True) -> None:
    """Apply consistent axis polish without changing plotted data."""

    if grid:
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        ax.grid(True, which="minor", color="0.94", linewidth=0.4)
    ax.tick_params(direction="in", top=True, right=True, width=0.8)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()


def save_publication_figure(
    fig,
    png_path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save a high-resolution PNG for reports and paper drafts."""

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PUBLICATION_DPI, metadata=dict(metadata or {}))
    return png_path
