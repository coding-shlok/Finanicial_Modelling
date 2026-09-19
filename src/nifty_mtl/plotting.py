"""Matplotlib style for paper figures: fixed categorical palette (assigned by
entity, never cycled), thin marks, recessive grid, single y-axis per panel."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
}
# fixed entity -> colour mapping used across every figure
SERIES = {
    "MTL (ours)": PALETTE["blue"],
    "MTL no-graph": PALETTE["violet"],
    "MTL return-only": PALETTE["magenta"],
    "MTL no-cost": PALETTE["aqua"],
    "MTL +cost-term": PALETTE["aqua"],
    "Momentum": PALETTE["orange"],
    "Inverse-vol": PALETTE["yellow"],
    "Logistic": PALETTE["green"],
    "Random": "#9a9992",
    "Nifty 50": "#0b0b0b",
}
TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e0"; SURFACE = "#fcfcfb"


def setup() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })


def color(name: str) -> str:
    return SERIES.get(name, PALETTE["blue"])


def save(fig, path) -> None:
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)
