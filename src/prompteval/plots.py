"""Figures: reliability diagrams and accuracy by condition."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9d4edd", "#6c757d"]


def reliability_plots(rel: pd.DataFrame, out: Path) -> None:
    if rel.empty:
        return
    for (m, l), g in rel.groupby(["model_id", "language"]):
        fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=150)
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="Perfect calibration")
        for k, (c, h) in enumerate(g.groupby("condition")):
            h = h[h["n"] > 0]
            ax.plot(h["mean_conf"], h["accuracy"], marker="o", ms=3,
                    color=PALETTE[k % len(PALETTE)], lw=1.5, label=c)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Stated confidence"); ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{m} · {l.upper()}", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"reliability_{m}_{l}.png")
        plt.close(fig)


def accuracy_plot(metrics: pd.DataFrame, out: Path) -> None:
    if metrics.empty:
        return
    groups = list(metrics.groupby(["model_id", "language"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.4), dpi=150,
                             sharey=True, squeeze=False)
    for ax, ((m, l), g) in zip(axes[0], groups):
        x = np.arange(len(g))
        y = g["accuracy"].to_numpy() * 100
        lo = y - g["accuracy_lo"].to_numpy() * 100
        hi = g["accuracy_hi"].to_numpy() * 100 - y
        ax.bar(x, y, color=[PALETTE[i % len(PALETTE)] for i in range(len(g))], width=0.65)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=3, lw=1)
        ax.set_xticks(x, g["condition"], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{m} · {l.upper()}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Accuracy (%)")
    ymin = max(0, np.nanmin(metrics["accuracy_lo"].to_numpy() * 100) - 10)
    axes[0][0].set_ylim(ymin, 100)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_condition.png")
    plt.close(fig)
