"""Draws the report's system diagram -> reports/figures/fig40_system.png (run: make report)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from creditrisk.config import get_path

MAIN, SIDE = 0.48, 0.2   # box widths: the main path (x = 0.5) and the side boxes
# (x, y, label, colour); the main path runs top to bottom
BOXES = {
    "data": (0.5, 0.94, "7 Home Credit tables\n307,511 labelled loans", "#dfe6ee"),
    "features": (0.5, 0.80, "240 leak-free features\n(one row per applicant)", "#dfe6ee"),
    "splits": (0.5, 0.66, "Frozen splits: train 60 / valid 10\nconformal 10 / test 20", "#dfe6ee"),
    "model": (0.5, 0.52, "Tuned LightGBM\n(14-model ladder, 5-fold CV)", "#cfe3cf"),
    "decide": (0.5, 0.38, "Calibration check + money rule\n"
                          "Conformal: approve / refer / decline", "#cfe3cf"),
    "explain": (0.5, 0.24, "TreeSHAP reasons, reason codes,\ncounterfactuals", "#cfe3cf"),
    "serve": (0.5, 0.10, "FastAPI + Streamlit app (Docker)", "#f3e1c7"),
    "fair": (0.12, 0.38, "Fairness audit\n+ mitigations", "#f2d0d0"),
    "psi": (0.88, 0.10, "PSI drift\nmonitor", "#f2d0d0"),
    "test": (0.88, 0.52, "Test split,\nused once", "#f2d0d0"),
}
EDGES = [("data", "features"), ("features", "splits"), ("splits", "model"), ("model", "decide"),
         ("decide", "explain"), ("explain", "serve"), ("decide", "fair"), ("model", "test"),
         ("serve", "psi")]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    ax.set_axis_off()
    for x, y, label, colour in BOXES.values():
        w = MAIN if x == 0.5 else SIDE
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.045), w, 0.09, boxstyle="round,pad=0.01",
                                    fc=colour, ec="#444", lw=0.8))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)
    for a, b in EDGES:
        (xa, ya, *_), (xb, yb, *_) = BOXES[a], BOXES[b]
        if xa == xb:   # vertical: bottom of a to top of b
            start, end = (xa, ya - 0.055), (xb, yb + 0.055)
        else:          # sideways: edge of the middle box to the side box
            sign = 1 if xb > xa else -1
            start, end = (xa + sign * MAIN / 2, ya), (xb - sign * SIDE / 2, yb)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     color="#444", lw=0.8))
    ax.set(xlim=(0, 1), ylim=(0.03, 1.0))
    fig.tight_layout()
    fig.savefig(get_path("figures") / "fig40_system.png", dpi=200)


if __name__ == "__main__":
    main()
