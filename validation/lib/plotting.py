"""Publication-oriented plotting helpers for retained diagnostic scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

PUBLICATION_DPI = 300
PUBLICATION_COLORS = (
    "#1f77b4",
    "#d95f02",
    "#2ca02c",
    "#9467bd",
    "#7f7f7f",
    "#17becf",
)


def configure_publication_matplotlib() -> None:
    """Apply a restrained Matplotlib style suitable for paper drafts."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.titlepad": 6,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": mpl.cycler(color=PUBLICATION_COLORS),
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.8,
            "legend.handlelength": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_publication_axis(ax, *, legend: bool = True, grid: bool = True) -> None:
    """Apply consistent axis polish without changing plotted data."""

    if grid:
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        ax.grid(True, which="minor", color="0.94", linewidth=0.35)
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True, width=0.8)
    if legend:
        handles, _labels = ax.get_legend_handles_labels()
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
    figure_metadata = {"Creator": "lno327 publication plotting"}
    figure_metadata.update(dict(metadata or {}))
    fig.savefig(png_path, dpi=PUBLICATION_DPI, metadata=figure_metadata)
    return png_path
