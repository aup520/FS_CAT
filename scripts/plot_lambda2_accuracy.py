from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def choose_serif_font() -> str:
    """Choose a locally available serif font close to the paper style."""

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "STIXGeneral",
        "DejaVu Serif",
    ):
        if candidate in available:
            return candidate
    return "DejaVu Serif"


def configure_style() -> str:
    font_name = choose_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name, "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font_name


def plot_lambda2_accuracy() -> None:
    font_name = configure_style()

    # Values are digitized/approximated from the provided screenshot.
    lambda2 = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.25, 0.40, 0.55, 0.75, 1.00])
    avg_acc = np.array([32.3, 37.1, 35.1, 38.5, 38.7, 38.9, 38.0, 36.5, 37.3, 34.7, 31.4])

    fig, ax = plt.subplots(figsize=(6.51, 4.06), dpi=100)

    ax.plot(
        lambda2,
        avg_acc,
        color="green",
        linestyle=":",
        linewidth=2.0,
        marker="s",
        markersize=6.0,
        markerfacecolor="green",
        markeredgecolor="green",
        markeredgewidth=0.8,
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(30, 42)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(30, 43, 2))

    ax.set_xlabel(r"Hyperparameter $\lambda_2$", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Accuracy (%)", fontsize=13, fontweight="bold")

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, color="#bcbcbc", alpha=0.55)
    ax.tick_params(axis="both", which="major", labelsize=12, width=0.8, length=3, pad=2)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontfamily(font_name)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")

    fig.tight_layout(pad=0.9)

    png_path = OUT_DIR / "lambda2_accuracy.png"
    pdf_path = OUT_DIR / "lambda2_accuracy.pdf"
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"font={font_name}")
    print(f"saved={png_path}")
    print(f"saved={pdf_path}")


if __name__ == "__main__":
    plot_lambda2_accuracy()
